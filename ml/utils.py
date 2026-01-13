"""
共享工具函数和常量
包含化学工具函数、数据分割、常量定义等
"""

from __future__ import annotations
# 在所有导入之前设置环境变量，避免PyTorch的shm.dll加载问题
import os
os.environ.setdefault("TORCH_SHM_DISABLE", "1")

import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# torch / pyg (optional) - 在RDKit之前导入，避免DLL冲突
import sys

TORCH_ERROR = None
PYG_ERROR = None

try:
    # 设置环境变量，帮助PyTorch找到DLL
    # 尝试多个可能的torch lib路径
    possible_paths = [
        os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib"),
        os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages", "torch", "lib"),
    ]
    
    torch_lib_path = None
    for path in possible_paths:
        if os.path.exists(path):
            torch_lib_path = path
            break
    
    if torch_lib_path:
        # 将torch的lib目录添加到PATH，帮助找到DLL依赖
        current_path = os.environ.get("PATH", "")
        if torch_lib_path not in current_path:
            os.environ["PATH"] = torch_lib_path + os.pathsep + current_path
            print(f"[utils] Added torch lib to PATH: {torch_lib_path}", file=sys.stderr, flush=True)
    
    # 延迟导入torch，确保PATH已设置
    import torch
    HAS_TORCH = True
    print(f"[utils] PyTorch imported successfully: {torch.__version__}", file=sys.stderr, flush=True)
except Exception as e:
    HAS_TORCH = False
    TORCH_ERROR = str(e)
    print(f"[utils] PyTorch import failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    torch = None  # type: ignore

# RDKit - 在PyTorch之后导入
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    if HAS_TORCH:
        from torch_geometric.data import Data
        HAS_PYG = True
        print(f"[utils] PyTorch Geometric imported successfully", file=sys.stderr, flush=True)
    else:
        HAS_PYG = False
        print(f"[utils] PyTorch Geometric skipped (HAS_TORCH=False)", file=sys.stderr, flush=True)
except Exception as e:
    HAS_PYG = False
    PYG_ERROR = str(e)
    print(f"[utils] PyTorch Geometric import failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    Data = None  # type: ignore

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(ROOT, "saved_models")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 定义所有要预测的分子性质及其数据文件
PROPERTIES = {
    "molecular_weight": {"name": "分子量 (MW)", "unit": "g/mol", "data_file": "molecular_weight.csv"},
    "logp": {"name": "LogP (脂溶性)", "unit": "", "data_file": "logp.csv"},
    "logs": {"name": "LogS (水溶解度)", "unit": "log(mol/L)", "data_file": "logs.csv"},
    "pka": {"name": "pKa", "unit": "", "data_file": "pka.csv"},
    "boiling_point": {"name": "沸点", "unit": "°C", "data_file": "boiling_point.csv"},
    "melting_point": {"name": "熔点", "unit": "°C", "data_file": "melting_point.csv"},
    "refractive_index": {"name": "折射率", "unit": "", "data_file": "refractive_index.csv"},
    "vapor_pressure": {"name": "蒸气压", "unit": "Pa", "data_file": "vapor_pressure.csv"},
    "density": {"name": "密度", "unit": "g/cm³", "data_file": "density.csv"},
    "flash_point": {"name": "闪点", "unit": "°C", "data_file": "flash_point.csv"}
}

RNG = np.random.default_rng(42)
warnings.filterwarnings("ignore")

# 化学工具函数

# 标准化和验证SMILES字符串，返回规范化的SMILES和分子对象
def sanitize_smiles(smiles: str) -> Tuple[str, Optional[Chem.Mol]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        mol = Chem.AddHs(mol)
        return Chem.MolToSmiles(Chem.RemoveHs(mol)), mol
    except Exception:
        return smiles, None

# 将分子对象转换为SDF格式字符串，包含3D坐标信息
def mol_to_sdf(mol: Chem.Mol) -> str:
    try:
        m = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        AllChem.EmbedMolecule(m, params)
        AllChem.UFFOptimizeMolecule(m, maxIters=100)
        return Chem.MolToMolBlock(m)
    except Exception:
        return ""

DESC_LIST = [
    Descriptors.MolWt, Descriptors.HeavyAtomCount, Descriptors.NumHAcceptors,
    Descriptors.NumHDonors, Descriptors.NumRotatableBonds, Descriptors.TPSA,
    Descriptors.RingCount, Crippen.MolLogP
]

# 提取分子的传统描述符特征（分子量、原子数、氢键受体/供体等）
def featurize_descriptors(mol: Chem.Mol) -> np.ndarray:
    feats = [f(mol) for f in DESC_LIST]
    # 芳香性比例
    arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    feats.append(arom / max(1, mol.GetNumAtoms()))
    return np.array(feats, dtype=float)

# 生成ECFP（扩展连接性指纹）特征向量，返回指纹数组和位信息映射
def featurize_ecfp(mol: Chem.Mol, nbits: int = 1024, radius: int = 2) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, int]]]]:
    bitInfo: Dict[int, List[Tuple[int,int]]] = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.RemoveHs(mol), radius, nBits=nbits, bitInfo=bitInfo)
    arr = np.zeros((nbits,), dtype=np.int8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32), bitInfo

# PyTorch Geometric 图构建器

ATOM_FEATS = {
    "atomic_num": list(range(1, 119)),
    "degree": list(range(0, 6)),
    "formal_charge": [-2,-1,0,1,2],
    "chiral_tag": [0,1,2,3],
    "num_hs": list(range(0,5)),
    "hybridization": [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
}

# 将原子转换为特征向量（原子序数、度数、电荷、手性等）
def atom_feature_vector(atom: Chem.Atom) -> List[float]:
    onehots = []
    def oh(val, choices):
        vec = [1.0 if val == c else 0.0 for c in choices]
        return vec
    onehots += oh(atom.GetAtomicNum(), ATOM_FEATS["atomic_num"])
    onehots += oh(atom.GetTotalDegree(), ATOM_FEATS["degree"])
    onehots += oh(atom.GetFormalCharge(), ATOM_FEATS["formal_charge"])
    onehots += oh(int(atom.GetChiralTag()), ATOM_FEATS["chiral_tag"])
    onehots += oh(atom.GetTotalNumHs(), ATOM_FEATS["num_hs"])
    onehots += oh(atom.GetHybridization(), ATOM_FEATS["hybridization"])
    onehots.append(1.0 if atom.GetIsAromatic() else 0.0)
    return onehots

# 将化学键转换为特征向量（单键、双键、三键、芳香键）
def bond_feature_vector(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    return [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0
    ]

# 将分子转换为PyTorch Geometric图数据结构，包含节点特征、边索引和边属性
def build_graph(mol: Chem.Mol) -> Optional[Data]:
    if not (HAS_TORCH and HAS_PYG):
        return None
    mol = Chem.RemoveHs(mol)
    xs = [atom_feature_vector(a) for a in mol.GetAtoms()]
    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_feature_vector(b)
        edge_index += [[i,j],[j,i]]
        edge_attr += [f, f]
    x = torch.tensor(xs, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0,4), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# 骨架分割（Murcko方法）

# 提取分子的Murcko骨架结构
def murcko_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return ""
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff)
    except Exception:
        return ""

# 基于分子骨架进行数据集划分，确保相同骨架的分子在同一集合中
def scaffold_split(df: pd.DataFrame, frac=(0.8,0.1,0.1), seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaff2rows: Dict[str, List[int]] = {}
    for i, smi in enumerate(df["smiles"].tolist()):
        s = murcko_scaffold(smi)
        scaff2rows.setdefault(s, []).append(i)
    n = len(df)
    n_train_target = int(frac[0] * n)
    n_val_target = int(frac[1] * n)
    n_test_target = n - n_train_target - n_val_target
    
    # 将大聚类拆分成小片段，以便更灵活地分配
    max_cluster_size = max(50, int(0.1 * n))  # 最大聚类大小，超过则拆分
    clusters = []
    rng = np.random.default_rng(seed)
    
    for cluster in scaff2rows.values():
        if len(cluster) <= max_cluster_size:
            clusters.append(cluster)
        else:
            # 拆分大聚类
            cluster_copy = cluster.copy()
            rng.shuffle(cluster_copy)
            for i in range(0, len(cluster_copy), max_cluster_size):
                clusters.append(cluster_copy[i:i + max_cluster_size])
    
    # 按聚类大小排序（小聚类优先），然后随机打乱
    clusters.sort(key=len)
    rng.shuffle(clusters)
    
    train_idx, val_idx, test_idx = [], [], []
    
    # 贪心分配：选择使偏差最小的集合
    for c in clusters:
        c_size = len(c)
        train_size = len(train_idx)
        val_size = len(val_idx)
        test_size = len(test_idx)
        
        # 计算加入各集合后的偏差变化（负值表示偏差减小）
        train_diff = abs((train_size + c_size) - n_train_target) - abs(train_size - n_train_target)
        val_diff = abs((val_size + c_size) - n_val_target) - abs(val_size - n_val_target)
        test_diff = abs((test_size + c_size) - n_test_target) - abs(test_size - n_test_target)
        
        # 选择偏差变化最小的集合
        if train_diff <= val_diff and train_diff <= test_diff:
            train_idx += c
        elif val_diff <= test_diff:
            val_idx += c
        else:
            test_idx += c
    
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

# 数据预处理函数

def preprocess_data(df: pd.DataFrame, target_col: str, remove_outliers: bool = True, outlier_method: str = "iqr") -> pd.DataFrame:
    """
    数据预处理：处理缺失值、异常值
    
    参数:
        df: 输入数据框，必须包含 'smiles' 和 target_col 列
        target_col: 目标列名
        remove_outliers: 是否移除异常值
        outlier_method: 异常值检测方法 ('iqr' 或 'zscore')
    
    返回:
        预处理后的数据框
    """
    df = df.copy()
    
    # 1. 处理缺失值
    # 移除SMILES为空的行
    df = df.dropna(subset=["smiles"])
    # 移除目标值为空的行
    df = df.dropna(subset=[target_col])
    
    # 2. 处理异常值（使用IQR方法或Z-score方法）
    if remove_outliers and len(df) > 10:  # 至少需要10个样本才进行异常值检测
        target_values = pd.to_numeric(df[target_col], errors='coerce')
        valid_mask = ~target_values.isna()
        
        if outlier_method == "iqr":
            # IQR方法：移除超出 Q1-1.5*IQR 到 Q3+1.5*IQR 范围的值
            Q1 = target_values[valid_mask].quantile(0.25)
            Q3 = target_values[valid_mask].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (target_values >= lower_bound) & (target_values <= upper_bound)
        else:  # zscore
            # Z-score方法：移除Z-score绝对值大于3的值
            mean_val = target_values[valid_mask].mean()
            std_val = target_values[valid_mask].std()
            if std_val > 0:
                z_scores = np.abs((target_values - mean_val) / std_val)
                outlier_mask = z_scores <= 3
            else:
                outlier_mask = valid_mask
        
        # 保留非异常值
        df = df[outlier_mask | ~valid_mask]
    
    # 3. 确保数据类型正确
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    return df.reset_index(drop=True)

# 确保演示数据集存在，如果不存在则创建默认数据集
def ensure_demo_dataset(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    # 如果文件不存在，尝试从logp.csv读取（作为默认）
    default_path = os.path.join(ROOT, "data", "logp.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    # 如果都不存在，创建默认数据
    rows = [
        ("CCO", -0.18), ("CC(=O)O", -0.31), ("c1ccccc1", 2.13), ("Cc1ccccc1", 2.73),
        ("CCN(CC)CC", 0.62), ("CCOC(=O)C", 0.18), ("CCC", 2.3), ("CCCC", 2.9),
        ("CCOCC", -0.1), ("CC(=O)N", -1.2), ("O=C=O", -0.7), ("C#N", -0.9),
        ("CCS", 0.5), ("CCCl", 1.7), ("CCBr", 1.8), ("c1ccncc1", 0.6),
        ("c1ccccc1O", 1.5), ("CC(=O)OC", 0.1), ("C1CCCCC1", 2.4), ("CC(C)O", -0.05)
    ]
    df = pd.DataFrame(rows, columns=["smiles","target"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df
