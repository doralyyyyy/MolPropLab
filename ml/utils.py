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


def _try_embed_3d(mol: Chem.Mol, seed: int = 42, max_iter: int = 200) -> Optional[Chem.Mol]:
    """使用ETKDG生成3D构象并返回含坐标的分子；失败返回None。"""
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = False
        if AllChem.EmbedMolecule(mol_h, params=params) != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=max_iter)
        except Exception:
            # 优化失败时继续使用初始构象
            pass
        mol_no_h = Chem.RemoveHs(mol_h, updateExplicitCount=True, sanitize=False)
        return mol_no_h
    except Exception:
        return None


def _distance_bins(dist: float, max_distance: float, num_bins: int) -> int:
    """将距离映射到[1, num_bins]的桶索引（1起始，0保留给bond）。"""
    if dist <= 0:
        return 1
    bin_size = max_distance / float(num_bins)
    idx = int(dist / bin_size)
    idx = min(idx, num_bins - 1)
    return idx + 1  # 1..num_bins


# 将分子转换为PyTorch Geometric图数据结构，包含节点特征、边索引、边类型和边属性
def build_graph(
    mol: Chem.Mol,
    use_3d: bool = True,
    max_distance: float = 5.0,
    num_distance_bins: int = 4,
    max_spatial_neighbors: Optional[int] = None,
    seed: int = 42,
) -> Optional[Data]:
    """
    构建PotentialNet风格的图：
    - bond边：edge_type=0
    - spatial边：按距离分箱，edge_type=1..num_distance_bins
    - edge_attr: 距离标量，形状[E,1]
    """
    if not (HAS_TORCH and HAS_PYG):
        return None

    # 生成3D坐标（可选），失败则退化为bond-only
    mol_for_geom = _try_embed_3d(mol, seed=seed) if use_3d else None
    has_3d = mol_for_geom is not None
    mol_no_h = Chem.RemoveHs(mol_for_geom if has_3d else mol, updateExplicitCount=True, sanitize=False)

    xs = [atom_feature_vector(a) for a in mol_no_h.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float)

    bond_edge_index = []
    bond_edge_attr = []
    for b in mol_no_h.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # 对于bond边，若有坐标则使用键长，否则用默认占位1.0
        if has_3d:
            pos_i = mol_no_h.GetConformer().GetAtomPosition(i)
            pos_j = mol_no_h.GetConformer().GetAtomPosition(j)
            dist = float(pos_i.Distance(pos_j))
        else:
            dist = 1.0
        bond_edge_index += [[i, j], [j, i]]
        bond_edge_attr += [[dist], [dist]]

    spatial_edge_index = []
    spatial_edge_attr = []
    spatial_edge_type = []

    if has_3d and max_distance > 0:
        conf = mol_no_h.GetConformer()
        n = mol_no_h.GetNumAtoms()
        # 预构建距离矩阵，避免重复计算
        for i in range(n):
            pos_i = conf.GetAtomPosition(i)
            for j in range(i + 1, n):
                pos_j = conf.GetAtomPosition(j)
                dist = float(pos_i.Distance(pos_j))
                if dist == 0 or dist > max_distance:
                    continue
                # 跳过已存在的共价键
                if mol_no_h.GetBondBetweenAtoms(i, j):
                    continue
                spatial_edge_index += [[i, j], [j, i]]
                spatial_edge_attr += [[dist], [dist]]
                etype = _distance_bins(dist, max_distance, num_distance_bins)
                spatial_edge_type += [etype, etype]

        # 如果需要限制邻居数量，基于距离截断
        if max_spatial_neighbors is not None and max_spatial_neighbors > 0:
            # 简易截断：按目标节点分组，保留最近的k条
            keep_mask = [True] * len(spatial_edge_index)
            from collections import defaultdict

            dst_to_edges = defaultdict(list)
            for idx, (src_dst, dist_attr) in enumerate(zip(spatial_edge_index, spatial_edge_attr)):
                dst_to_edges[src_dst[1]].append((idx, dist_attr[0]))
            for dst, items in dst_to_edges.items():
                if len(items) <= max_spatial_neighbors:
                    continue
                items.sort(key=lambda x: x[1])
                for drop_idx, _ in items[max_spatial_neighbors:]:
                    keep_mask[drop_idx] = False
            spatial_edge_index = [e for e, k in zip(spatial_edge_index, keep_mask) if k]
            spatial_edge_attr = [e for e, k in zip(spatial_edge_attr, keep_mask) if k]
            spatial_edge_type = [e for e, k in zip(spatial_edge_type, keep_mask) if k]

    # 合并边
    edge_index = bond_edge_index + spatial_edge_index
    edge_attr = bond_edge_attr + spatial_edge_attr
    edge_types = [0] * len(bond_edge_index) + spatial_edge_type

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1) if edge_attr else torch.empty((0,1), dtype=torch.float)
    edge_type_t = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.empty((0,), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        edge_type=edge_type_t,
    )
    if has_3d:
        # 保存坐标以便后续可视化或扩展
        conf = mol_no_h.GetConformer()
        pos = []
        for i in range(mol_no_h.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pos.append([p.x, p.y, p.z])
        data.pos = torch.tensor(pos, dtype=torch.float)

    data.n_bond_edges = int(len(bond_edge_index))
    data.n_spatial_edges = int(len(spatial_edge_index))
    data.has_3d = bool(has_3d)
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
    """
    Scaffold split with *no scaffold leakage*:
    - 每个 Murcko scaffold 作为一个整体 cluster 分配到 train/val/test
    - 不拆分大 cluster（否则同一 scaffold 会跨集合，造成数据泄露）
    - 使用 seed 控制 scaffold 组的随机顺序，结果可复现
    """
    if df is None or len(df) == 0:
        return df.iloc[0:0], df.iloc[0:0], df.iloc[0:0]

    scaff2rows: Dict[str, List[int]] = {}
    for i, smi in enumerate(df["smiles"].tolist()):
        s = murcko_scaffold(smi)
        scaff2rows.setdefault(s, []).append(i)

    n = len(df)
    n_train_target = int(frac[0] * n)
    n_val_target = int(frac[1] * n)
    n_test_target = n - n_train_target - n_val_target

    # clusters: list of row-index lists, one per scaffold
    clusters: List[List[int]] = list(scaff2rows.values())

    # 标准做法：按 cluster size 从大到小排序，减少“单个大簇挤占剩余配额”导致的波动
    # 同 size 的 cluster 再用 seed 打乱，保证可复现且不过度依赖输入顺序
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for c in clusters:
        c_size = len(c)
        # 先填 train，再填 val，其余进 test（不拆分 scaffold）
        if len(train_idx) + c_size <= n_train_target:
            train_idx.extend(c)
        elif len(val_idx) + c_size <= n_val_target:
            val_idx.extend(c)
        else:
            test_idx.extend(c)

    # 若因大簇导致 val/test 过小，做一次“整簇迁移”修正：把 test 中的整簇搬到 val，直到达到目标
    if len(val_idx) < n_val_target and len(test_idx) > 0:
        # 以 cluster 为单位重建映射（仅针对 test 内的 scaffold）
        test_scaff2rows: Dict[str, List[int]] = {}
        for i in test_idx:
            s = murcko_scaffold(df.iloc[i]["smiles"])
            test_scaff2rows.setdefault(s, []).append(i)
        move_clusters = list(test_scaff2rows.values())
        rng.shuffle(move_clusters)
        move_clusters.sort(key=len)  # 优先搬小簇，尽量贴近目标
        for mc in move_clusters:
            if len(val_idx) >= n_val_target:
                break
            # 从 test_idx 移除该簇，加入 val_idx
            mc_set = set(mc)
            test_idx = [i for i in test_idx if i not in mc_set]
            val_idx.extend(mc)

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
