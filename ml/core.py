"""
主要功能：
- sanitize_smiles, mol_to_sdf：SMILES标准化和SDF生成
- featurize_descriptors, featurize_ecfp：分子描述符和ECFP指纹特征提取
- build_graph：使用PyTorch Geometric构建分子图
- scaffold_split：基于Murcko骨架的数据集分割
- BaselineModel：基线模型（LightGBM或RandomForest），支持SHAP解释
- GIN GNN：图神经网络模型，使用MC-Dropout估计不确定性，支持梯度显著性分析
- 高级训练和评估辅助函数
"""

from __future__ import annotations
# 在所有导入之前设置环境变量，避免PyTorch的shm.dll加载问题
import os
os.environ.setdefault("TORCH_SHM_DISABLE", "1")

import json, math, pickle, warnings, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import yaml

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
            print(f"[core] Added torch lib to PATH: {torch_lib_path}", file=sys.stderr, flush=True)
    
    # 延迟导入torch，确保PATH已设置
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    print(f"[core] PyTorch imported successfully: {torch.__version__}", file=sys.stderr, flush=True)
except Exception as e:
    HAS_TORCH = False
    TORCH_ERROR = str(e)
    print(f"[core] PyTorch import failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    # 创建虚拟神经网络类，避免在torch缺失时类定义崩溃
    class _DummyModule:
        pass

    class _DummyNN:
        Module = _DummyModule

    nn = _DummyNN()  # type: ignore
    F = None  # type: ignore

# RDKit - 在PyTorch之后导入
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# sklearn / lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    if HAS_TORCH:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINConv, global_add_pool
        HAS_PYG = True
        print(f"[core] PyTorch Geometric imported successfully", file=sys.stderr, flush=True)
    else:
        HAS_PYG = False
        print(f"[core] PyTorch Geometric skipped (HAS_TORCH=False)", file=sys.stderr, flush=True)
except Exception as e:
    HAS_PYG = False
    PYG_ERROR = str(e)
    print(f"[core] PyTorch Geometric import failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)


ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(ROOT, "saved_models")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 定义所有要预测的分子性质及其数据文件
PROPERTIES = {
    "logp": {"name": "LogP (脂水分配系数)", "unit": "", "data_file": "logp.csv"},
    "solubility": {"name": "溶解度 (Solubility)", "unit": "log(mol/L)", "data_file": "solubility.csv"},
    "boiling_point": {"name": "沸点 (Boiling Point)", "unit": "°C", "data_file": "boiling_point.csv"},
    "melting_point": {"name": "熔点 (Melting Point)", "unit": "°C", "data_file": "melting_point.csv"},
    "pka": {"name": "pKa", "unit": "", "data_file": "pka.csv"},
    "toxicity": {"name": "毒性 (Toxicity)", "unit": "LD50 (mol/kg)", "data_file": "toxicity.csv"},
    "bioactivity": {"name": "生物活性 (Bioactivity)", "unit": "-log(IC50)", "data_file": "bioactivity.csv"},
    "admet_clearance": {"name": "ADMET 清除率", "unit": "mL/min/kg", "data_file": "admet_clearance.csv"}
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
def build_graph(mol: Chem.Mol) -> Optional["Data"]:
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
    import torch
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
    clusters = list(scaff2rows.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n = len(df)
    n_train = int(frac[0]*n)
    n_val = int(frac[1]*n)
    train_idx, val_idx, test_idx = [], [], []
    for c in clusters:
        if len(train_idx) + len(c) <= n_train: train_idx += c
        elif len(val_idx) + len(c) <= n_val: val_idx += c
        else: test_idx += c
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

# 基线模型（LightGBM/RandomForest）

class BaselineModel:
    # 初始化基线模型，设置集成模型数量和ECFP位数
    def __init__(self, n_models: int = 3, nbits: int = 1024):
        self.n_models = n_models
        self.nbits = nbits
        self.models = []  # LightGBM或RandomForest模型列表
        self.feature_names = [f"desc_{i}" for i in range(len(DESC_LIST)+1)] + [f"ecfp_{i}" for i in range(nbits)]
        self.is_lgb = HAS_LGB

    # 训练单个模型（LightGBM或RandomForest），使用指定的随机种子
    def _train_one(self, X: np.ndarray, y: np.ndarray, seed: int):
        if HAS_LGB:
            # 根据数据集大小动态调整参数，避免小数据集上的警告和特征被过滤
            n_samples = len(X)
            # 极小数据集（<30个样本）：使用非常宽松的参数，确保特征能被使用
            if n_samples < 30:
                n_estimators = max(5, min(20, n_samples // 2))
                min_data_in_leaf = 1  # 最小值为1，允许单样本叶子
                min_data_in_bin = 1
                min_child_samples = 1  # 允许单样本分裂
                subsample = 1.0  # 使用全部数据
                colsample_bytree = 1.0  # 使用全部特征
                max_depth = 3  # 限制树深度，避免过拟合
                min_gain_to_split = 0.0  # 降低分裂阈值，确保能使用特征
            # 小数据集（30-100个样本）：适度调整参数
            elif n_samples < 100:
                n_estimators = min(50, max(10, n_samples // 2))
                min_data_in_leaf = max(1, min(2, n_samples // 15))
                min_data_in_bin = 1
                min_child_samples = 1
                subsample = min(1.0, max(0.8, 1.0 - 10.0 / n_samples))
                colsample_bytree = min(1.0, max(0.8, 1.0 - 10.0 / n_samples))
                max_depth = 5
                min_gain_to_split = 0.0
            else:
                # 对于中等数据集（100-500个样本），使用适中的参数
                if n_samples < 500:
                    n_estimators = 200
                    min_data_in_leaf = 10
                    min_data_in_bin = 3
                    min_child_samples = 10
                    subsample = 0.85
                    colsample_bytree = 0.85
                    max_depth = 7
                    min_gain_to_split = 0.0
                else:
                    # 大数据集使用默认参数
                    n_estimators = 400
                    min_data_in_leaf = 20
                    min_data_in_bin = 5
                    min_child_samples = 20
                    subsample = 0.9
                    colsample_bytree = 0.9
                    max_depth = -1  # 不限制深度
                    min_gain_to_split = 0.0
            
            # 彻底抑制LightGBM的警告输出（包括stderr）
            import sys
            import contextlib
            from io import StringIO
            
            # 重定向stderr以捕获LightGBM的警告
            stderr_buffer = StringIO()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with contextlib.redirect_stderr(stderr_buffer):
                    m = lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        learning_rate=0.05,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        min_data_in_leaf=min_data_in_leaf,
                        min_data_in_bin=min_data_in_bin,
                        min_child_samples=min_child_samples,
                        max_depth=max_depth,
                        min_gain_to_split=min_gain_to_split,
                        random_state=seed,
                        verbose=-1,  # 禁用标准输出
                        force_col_wise=True,  # 避免列模式警告
                        boosting_type='gbdt'  # 明确指定boosting类型
                    )
                    m.fit(X, y)
            return m
        else:
            # RandomForest 也根据数据集大小调整
            n_samples = len(X)
            n_estimators = min(300, max(10, n_samples * 2)) if n_samples < 100 else 300
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
            rf.fit(X, y)
            return rf

    # 训练集成模型，从数据框中提取特征并训练多个模型
    def fit(self, df: pd.DataFrame, target_col: str):
        X_list, y = [], []
        for smi, tgt in zip(df["smiles"], df[target_col]):
            _, mol = sanitize_smiles(smi)
            if not mol: continue
            desc = featurize_descriptors(mol)
            ecfp,_ = featurize_ecfp(mol, self.nbits)
            feat = np.concatenate([desc, ecfp], axis=0)
            X_list.append(feat); y.append(float(tgt))
        X = np.vstack(X_list); y = np.array(y)
        self.models = [self._train_one(X, y, seed=i+7) for i in range(self.n_models)]
        return self

    # 对SMILES字符串进行预测，返回预测值、不确定性和原子重要性
    def predict(self, smiles: str, return_shap: bool = True) -> Dict[str, Any]:
        _, mol = sanitize_smiles(smiles)
        if not mol:
            return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": ""}
        desc = featurize_descriptors(mol)
        ecfp, bitInfo = featurize_ecfp(mol, self.nbits)
        x = np.concatenate([desc, ecfp], axis=0).reshape(1, -1)
        preds = np.array([m.predict(x)[0] for m in self.models])
        pred = float(preds.mean())
        unc = float(preds.std(ddof=1) if len(preds)>1 else 0.0)
        sdf = mol_to_sdf(mol)
        atom_imps = []
        # 使用SHAP解释树模型
        if return_shap:
            try:
                import shap
                m0 = self.models[0]
                explainer = shap.Explainer(m0)
                sv = explainer(x)  # 形状为 [1, d]
                vals = np.array(sv.values)[0]  # 形状为 (d,)
                # 通过bitInfo将ECFP的SHAP值映射到原子
                atom_scores = np.zeros(mol.GetNumAtoms(), dtype=float)
                # ECFP部分的索引偏移量
                offset = len(DESC_LIST)+1
                onbits = np.where(ecfp > 0)[0]
                for b in onbits:
                    shap_v = vals[offset + b]
                    info = bitInfo.get(int(b), [])
                    for (atom_idx, _radius) in info:
                        if atom_idx < len(atom_scores):
                            atom_scores[atom_idx] += float(shap_v)
                # 归一化到0-1范围
                if np.max(np.abs(atom_scores)) > 0:
                    atom_imps = (np.abs(atom_scores) / np.max(np.abs(atom_scores))).tolist()
                else:
                    atom_imps = atom_scores.tolist()
            except Exception:
                atom_imps = [0.0]*mol.GetNumAtoms()
        return {"prediction": pred, "uncertainty": unc, "atom_importances": atom_imps, "sdf": sdf}

    # 保存模型到文件
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"n_models": self.n_models, "nbits": self.nbits, "models": self.models, "is_lgb": self.is_lgb}, f)

    # 从文件加载模型
    @staticmethod
    def load(path: str) -> "BaselineModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        bm = BaselineModel(n_models=obj["n_models"], nbits=obj["nbits"])
        bm.models = obj["models"]; bm.is_lgb = obj.get("is_lgb", True)
        return bm

# 图神经网络（GIN）

class GINRegressor(nn.Module):
    # 初始化GIN（图同构网络）回归模型
    def __init__(self, in_dim, hidden=64, layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        mlps = []
        last = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(last, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            mlps.append(GINConv(mlp))
            last = hidden
        self.convs = nn.ModuleList(mlps)
        self.lin = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    # 前向传播，处理图数据并返回预测值
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = xs[-1]
        if hasattr(data, "batch"):
            x = global_add_pool(x, data.batch)
        else:
            x = x.sum(dim=0, keepdim=True)
        out = self.lin(x)
        return out.view(-1)

# 从数据框构建图数据集，将SMILES转换为图结构
def build_graph_dataset(df: pd.DataFrame, target_col: str):
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("PyTorch Geometric not installed.")
    graphs, ys = [], []
    for smi, tgt in zip(df["smiles"], df[target_col]):
        _, mol = sanitize_smiles(smi)
        if not mol: continue
        g = build_graph(mol)
        if g is None: continue
        g.y = torch.tensor([float(tgt)], dtype=torch.float)
        graphs.append(g); ys.append(float(tgt))
    return graphs

@dataclass
class GNNPack:
    model: Any
    in_dim: int

# 训练GNN模型，使用配置参数进行训练并返回最佳模型（支持数据标准化）
def train_gnn(df: pd.DataFrame, target_col: str, config: Dict[str, Any]) -> GNNPack:
    if not (HAS_TORCH and HAS_PYG): raise RuntimeError("GNN deps not available")
    train_df, val_df, _ = scaffold_split(df, (0.8,0.2,0.0))
    train_graphs = build_graph_dataset(train_df, target_col)
    val_graphs = build_graph_dataset(val_df, target_col)
    if not train_graphs: 
        raise RuntimeError(f"Empty graph dataset. Original data: {len(df)} rows, train: {len(train_df)} rows, valid graphs: {len(train_graphs)}")
    in_dim = train_graphs[0].x.size(-1)
    
    # 计算训练集的均值和标准差用于标准化
    train_targets = [g.y.item() for g in train_graphs]
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets)) or 1.0
    
    # 标准化目标值
    for g in train_graphs:
        g.y = (g.y - target_mean) / target_std
    for g in val_graphs:
        g.y = (g.y - target_mean) / target_std
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据数据集大小调整模型参数
    n_samples = len(train_graphs)
    if n_samples == 0:
        raise RuntimeError("No valid graphs in training set")
    if n_samples < 20:
        hidden = 16
        layers = 2
        dropout = 0.1
        batch_size = max(1, min(4, n_samples))  # 确保至少为1
        epochs = 10
    elif n_samples < 50:
        hidden = 32
        layers = 2
        dropout = 0.15
        batch_size = max(1, min(8, n_samples))  # 确保至少为1
        epochs = 15
    else:
        hidden = config.get("hidden", 64)
        layers = config.get("layers", 3)
        dropout = config.get("dropout", 0.2)
        batch_size = config.get("batch_size", 32)
        # 对于较大的数据集，增加训练轮数以提高模型性能
        epochs = config.get("epochs", 50) if n_samples > 200 else config.get("epochs", 30)
    
    model = GINRegressor(in_dim, hidden=hidden, layers=layers, dropout=dropout).to(device)
    # 根据数据集大小调整学习率
    if n_samples < 100:
        lr = 5e-4  # 小数据集使用较小的学习率
    elif n_samples < 500:
        lr = 1e-3  # 中等数据集使用标准学习率
    else:
        lr = config.get("lr", 1e-3)  # 大数据集使用配置的学习率
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    dl_tr = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(val_graphs, batch_size=max(1, min(64, len(val_graphs))))
    best = float("inf"); best_state = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for b in dl_tr:
            b = b.to(device)
            opt.zero_grad()
            pred = model(b)
            loss = F.mse_loss(pred, b.y.view(-1))
            loss.backward(); opt.step()
            losses.append(loss.item())
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss = 0.0; n=0
            for b in dl_va:
                b = b.to(device)
                p = model(b)
                val_loss += F.mse_loss(p, b.y.view(-1), reduction="sum").item()
                n += b.y.numel()
            val_rmse = math.sqrt(val_loss / max(1,n))
        if val_rmse < best:
            best = val_rmse
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    
    # 保存标准化参数和模型配置到模型（通过添加属性）
    model.target_mean = target_mean
    model.target_std = target_std
    model.hidden_dim = hidden
    model.num_layers = layers
    model.dropout_rate = dropout
    
    return GNNPack(model=model, in_dim=in_dim)

# 使用GNN模型进行预测，使用MC-Dropout估计不确定性，计算原子重要性（支持反标准化）
def gnn_predict_atom_importance(pack: GNNPack, smiles: str) -> Dict[str, Any]:
    _, mol = sanitize_smiles(smiles)
    if not mol: return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": ""}
    g = build_graph(mol)
    if g is None: return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": ""}
    device = next(pack.model.parameters()).device
    
    # 获取标准化参数（如果存在）
    target_mean = getattr(pack.model, 'target_mean', 0.0)
    target_std = getattr(pack.model, 'target_std', 1.0)
    
    pack.model.eval()
    # 使用MC-Dropout估计不确定性
    T = 20
    preds_normalized = []
    for _ in range(T):
        pack.model.train()  # 启用dropout以进行不确定性估计
        with torch.no_grad():
            p = pack.model(g.to(device)).item()
            preds_normalized.append(p)
    
    # 计算标准化空间的不确定性
    pred_normalized = float(np.mean(preds_normalized))
    unc_normalized = float(np.std(preds_normalized, ddof=1))
    
    # 反标准化预测值
    pred = pred_normalized * target_std + target_mean
    # 不确定性：标准化空间的不确定性乘以标准差（不需要加均值）
    unc = unc_normalized * target_std
    # 基于梯度的节点重要性计算
    pack.model.eval()
    g = g.to(device)
    g.x.requires_grad_(True)
    y = pack.model(g)
    y = y.view(-1)[0]
    y.backward()
    grads = g.x.grad.detach().abs().sum(dim=1).cpu().numpy()  # 每个节点的梯度
    if grads.max() > 0:
        imps = (grads / grads.max()).tolist()
    else:
        imps = grads.tolist()
    sdf = mol_to_sdf(mol)
    return {"prediction": pred, "uncertainty": unc, "atom_importances": imps, "sdf": sdf}

# 高级API接口

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

# 返回基线模型权重文件的路径（支持多性质）
def quick_baseline_weights_path(property_name: str = "logp") -> str:
    return os.path.join(SAVE_DIR, f"baseline_{property_name}_v1.pkl")

# 返回GNN模型权重文件的路径（支持多性质）
def quick_gnn_weights_path(property_name: str = "logp") -> str:
    return os.path.join(SAVE_DIR, f"gnn_{property_name}_v1.pth")

# 训练演示用的基线模型并保存（支持多性质）
def train_baseline_demo(property_name: str = "logp") -> str:
    data_file = os.path.join(ROOT, "data", PROPERTIES.get(property_name, PROPERTIES["logp"])["data_file"])
    df = ensure_demo_dataset(data_file)
    bm = BaselineModel(n_models=3, nbits=1024).fit(df, "target")
    out = quick_baseline_weights_path(property_name)
    bm.save(out)
    return out

# 训练演示用的GNN模型并保存（支持多性质）
def train_gnn_demo(property_name: str = "logp") -> str:
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("GNN deps missing")
    data_file = os.path.join(ROOT, "data", PROPERTIES.get(property_name, PROPERTIES["logp"])["data_file"])
    df = ensure_demo_dataset(data_file)
    with open(os.path.join(ROOT, "configs", "gnn.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    pack = train_gnn(df, "target", cfg)
    out = quick_gnn_weights_path(property_name)
    torch.save(pack.model.state_dict(), out)
    # 保存标准化参数和模型配置
    norm_path = out.replace(".pth", "_norm.json")
    with open(norm_path, "w") as f:
        json.dump({
            "mean": getattr(pack.model, 'target_mean', 0.0), 
            "std": getattr(pack.model, 'target_std', 1.0),
            "hidden": getattr(pack.model, 'hidden_dim', 64),
            "layers": getattr(pack.model, 'num_layers', 3),
            "dropout": getattr(pack.model, 'dropout_rate', 0.2)
        }, f)
    return out

# 训练所有性质的模型
def train_all_properties(model_type: str = "baseline"):
    """训练所有性质的模型"""
    results = {}
    for prop_key in PROPERTIES.keys():
        try:
            if model_type == "baseline":
                path = train_baseline_demo(prop_key)
            else:
                path = train_gnn_demo(prop_key)
            results[prop_key] = path
            print(f"✓ Trained {prop_key} model: {path}")
        except Exception as e:
            print(f"✗ Failed to train {prop_key}: {e}")
            results[prop_key] = None
    return results

# 预测单个性质的接口
def predict_property(smiles: str, property_name: str, model: str = "baseline") -> Dict[str, Any]:
    import sys
    def debug(msg):
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)
    
    smiles = smiles.strip()
    prop_info = PROPERTIES.get(property_name, PROPERTIES["logp"])
    data_file = os.path.join(ROOT, "data", prop_info["data_file"])
    
    version = "v1"
    if model == "baseline":
        path = quick_baseline_weights_path(property_name)
        if os.path.exists(path):
            bm = BaselineModel.load(path)
        else:
            bm = BaselineModel(n_models=3, nbits=1024)
            df = ensure_demo_dataset(data_file)
            bm.fit(df, "target")
            bm.save(path)
        out = bm.predict(smiles, return_shap=True)
        out["model"] = "baseline"
        out["version"] = version
        return out
    else:
        if not (HAS_TORCH and HAS_PYG):
            return predict_property(smiles, property_name, model="baseline")
        path = quick_gnn_weights_path(property_name)
        _, mol = sanitize_smiles(smiles)
        if not mol:
            return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": "", "model":"gnn","version":version}
        g = build_graph(mol)
        if g is None:
            return predict_property(smiles, property_name, model="baseline")
        in_dim = g.x.size(-1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 尝试加载保存的模型配置
        norm_path = path.replace(".pth", "_norm.json")
        if os.path.exists(norm_path):
            with open(norm_path, "r") as f:
                saved_config = json.load(f)
            hidden = saved_config.get("hidden", 64)
            layers = saved_config.get("layers", 3)
            dropout = saved_config.get("dropout", 0.2)
        else:
            # 如果没有保存的配置，使用默认配置
            with open(os.path.join(ROOT, "configs", "gnn.yaml"), "r") as f:
                cfg = yaml.safe_load(f)
            hidden = cfg.get("hidden", 64)
            layers = cfg.get("layers", 3)
            dropout = cfg.get("dropout", 0.2)
        
        model_g = GINRegressor(in_dim, hidden=hidden, layers=layers, dropout=dropout).to(device)
        if os.path.exists(path):
            sd = torch.load(path, map_location=device)
            model_g.load_state_dict(sd, strict=False)
            # 加载标准化参数
            if os.path.exists(norm_path):
                with open(norm_path, "r") as f:
                    norm_data = json.load(f)
                    model_g.target_mean = norm_data.get("mean", 0.0)
                    model_g.target_std = norm_data.get("std", 1.0)
        else:
            df = ensure_demo_dataset(data_file)
            with open(os.path.join(ROOT, "configs", "gnn.yaml"), "r") as f:
                cfg = yaml.safe_load(f)
            pack = train_gnn(df, "target", cfg)
            model_g = pack.model
            torch.save(model_g.state_dict(), path)
            # 保存标准化参数和模型配置
            norm_path = path.replace(".pth", "_norm.json")
            with open(norm_path, "w") as f:
                json.dump({
                    "mean": getattr(model_g, 'target_mean', 0.0), 
                    "std": getattr(model_g, 'target_std', 1.0),
                    "hidden": getattr(model_g, 'hidden_dim', 64),
                    "layers": getattr(model_g, 'num_layers', 3),
                    "dropout": getattr(model_g, 'dropout_rate', 0.2)
                }, f)
        pack = GNNPack(model=model_g, in_dim=in_dim)
        out = gnn_predict_atom_importance(pack, smiles)
        out["model"] = "gnn"
        out["version"] = version
        return out

# 预测所有性质的接口（保留向后兼容）
def predict(smiles: str, model: str = "baseline") -> Dict[str, Any]:
    """预测所有性质，返回包含所有性质预测结果的字典"""
    results = {}
    # 使用第一个性质的原子重要性（所有性质应该相似）
    first_prop = list(PROPERTIES.keys())[0]
    first_result = predict_property(smiles, first_prop, model)
    
    # 预测所有性质
    for prop_key, prop_info in PROPERTIES.items():
        try:
            prop_result = predict_property(smiles, prop_key, model)
            results[prop_key] = {
                "name": prop_info["name"],
                "unit": prop_info["unit"],
                "prediction": prop_result["prediction"],
                "uncertainty": prop_result["uncertainty"]
            }
        except Exception as e:
            results[prop_key] = {
                "name": prop_info["name"],
                "unit": prop_info["unit"],
                "prediction": float("nan"),
                "uncertainty": float("nan"),
                "error": str(e)
            }
    
    # 返回第一个结果的原子重要性和SDF（用于3D可视化）
    return {
        "properties": results,
        "atom_importances": first_result.get("atom_importances", []),
        "sdf": first_result.get("sdf", ""),
        "model": first_result.get("model", model),
        "version": first_result.get("version", "v1")
    }

# 训练和评估包装函数

# 训练基线模型的主函数，处理命令行参数
def train_baseline_main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "logp.csv"))
    p.add_argument("--target", default="target")
    p.add_argument("--out", default=quick_baseline_weights_path("logp"))
    a = p.parse_args(args)
    df = ensure_demo_dataset(a.data) if not os.path.exists(a.data) else pd.read_csv(a.data)
    bm = BaselineModel(n_models=3).fit(df, a.target)
    bm.save(a.out)
    print(f"Saved baseline to {a.out}")

# 评估基线模型的主函数，计算RMSE和R2分数
def eval_baseline_main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "logp.csv"))
    p.add_argument("--target", default="target")
    p.add_argument("--model", default=quick_baseline_weights_path("logp"))
    a = p.parse_args(args)
    df = pd.read_csv(a.data)
    bm = BaselineModel.load(a.model)
    # 简单随机分割
    tr, te = train_test_split(df, test_size=0.3, random_state=42)
    y_true, y_pred = [], []
    for smi, y in zip(te["smiles"], te[a.target]):
        y_true.append(float(y))
        y_pred.append(bm.predict(smi, return_shap=False)["prediction"])
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(json.dumps({"rmse": rmse, "r2": r2}))

# 训练GNN模型的主函数，处理命令行参数
def train_gnn_main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "logp.csv"))
    p.add_argument("--target", default="target")
    p.add_argument("--cfg", default=os.path.join(ROOT, "configs", "gnn.yaml"))
    p.add_argument("--out", default=quick_gnn_weights_path("logp"))
    a = p.parse_args(args)
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("PyTorch Geometric not available")
    df = ensure_demo_dataset(a.data) if not os.path.exists(a.data) else pd.read_csv(a.data)
    with open(a.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    pack = train_gnn(df, a.target, cfg)
    torch.save(pack.model.state_dict(), a.out)
    # 保存标准化参数和模型配置
    norm_path = a.out.replace(".pth", "_norm.json")
    with open(norm_path, "w") as f:
        json.dump({
            "mean": getattr(pack.model, 'target_mean', 0.0), 
            "std": getattr(pack.model, 'target_std', 1.0),
            "hidden": getattr(pack.model, 'hidden_dim', 64),
            "layers": getattr(pack.model, 'num_layers', 3),
            "dropout": getattr(pack.model, 'dropout_rate', 0.2)
        }, f)
    print(f"Saved gnn to {a.out}")

# 评估GNN模型的主函数，计算RMSE和R2分数
def eval_gnn_main(args=None):
    if not (HAS_TORCH and HAS_PYG):
        print(json.dumps({"error":"gnn deps missing"})); return
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "logp.csv"))
    p.add_argument("--target", default="target")
    p.add_argument("--model", default=quick_gnn_weights_path("logp"))
    p.add_argument("--cfg", default=os.path.join(ROOT, "configs", "gnn.yaml"))
    a = p.parse_args(args)
    df = pd.read_csv(a.data)
    
    # 尝试加载保存的模型配置
    norm_path = a.model.replace(".pth", "_norm.json")
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            saved_config = json.load(f)
        hidden = saved_config.get("hidden", 64)
        layers = saved_config.get("layers", 3)
        dropout = saved_config.get("dropout", 0.2)
    else:
        with open(a.cfg, "r") as f:
            cfg = yaml.safe_load(f)
        hidden = cfg.get("hidden", 64)
        layers = cfg.get("layers", 3)
        dropout = cfg.get("dropout", 0.2)
    
    # 简单随机分割
    tr, te = train_test_split(df, test_size=0.3, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建一个虚拟图以获取特征维度
    _, mol = sanitize_smiles(te.iloc[0]["smiles"])
    g = build_graph(mol)
    in_dim = g.x.size(-1)
    model_g = GINRegressor(in_dim, hidden=hidden, layers=layers, dropout=dropout).to(device)
    if os.path.exists(a.model):
        model_g.load_state_dict(torch.load(a.model, map_location=device), strict=False)
        # 加载标准化参数
        if os.path.exists(norm_path):
            with open(norm_path, "r") as f:
                norm_data = json.load(f)
                model_g.target_mean = norm_data.get("mean", 0.0)
                model_g.target_std = norm_data.get("std", 1.0)
    y_true, y_pred = [], []
    for smi, y in zip(te["smiles"], te[a.target]):
        y_true.append(float(y))
        out = gnn_predict_atom_importance(GNNPack(model_g, in_dim), smi)
        y_pred.append(out["prediction"])
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(json.dumps({"rmse": rmse, "r2": r2}))
