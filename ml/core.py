"""
核心模块 - 向后兼容层
将所有功能重新导出，保持向后兼容性
"""

# 从 utils 导入共享工具函数和常量
from utils import (
    # 常量
    ROOT, SAVE_DIR, LOG_DIR, PROPERTIES, RNG,
    HAS_TORCH, HAS_PYG, TORCH_ERROR, PYG_ERROR,
    # 化学工具函数
    sanitize_smiles, mol_to_sdf, featurize_descriptors, featurize_ecfp,
    build_graph, atom_feature_vector, bond_feature_vector,
    murcko_scaffold, scaffold_split, ensure_demo_dataset,
    # 数据预处理
    preprocess_data,
    # 常量
    DESC_LIST, ATOM_FEATS
)

# 从 train_baseline 导入基线模型相关
from train_baseline import (
    BaselineModel,
    train_baseline_main,
    train_baseline_demo,
    quick_baseline_weights_path
)

# 从 train_gnn 导入GNN相关
try:
    from train_gnn import (
        GINRegressor,
        GNNPack,
        train_gnn,
        train_gnn_main,
        train_gnn_demo,
        build_graph_dataset,
        quick_gnn_weights_path
    )
except (RuntimeError, ImportError):
    # 如果GNN依赖不可用，创建占位符
    GINRegressor = None
    GNNPack = None
    train_gnn = None
    train_gnn_main = None
    train_gnn_demo = None
    build_graph_dataset = None
    quick_gnn_weights_path = None

# 从 eval_baseline 导入评估函数
from eval_baseline import eval_baseline_main, calculate_metrics

# 从 eval_gnn 导入评估函数
try:
    from eval_gnn import (
        eval_gnn_main,
        gnn_predict_atom_importance
    )
except (RuntimeError, ImportError):
    eval_gnn_main = None
    gnn_predict_atom_importance = None

# 从 inference 导入预测函数
from inference import predict, predict_property

# 训练所有性质的函数
def train_all_properties(model_type: str = "baseline"):
    """训练所有性质的模型"""
    if model_type == "baseline":
        from train_baseline import train_all_properties as train_baseline_all
        return train_baseline_all("baseline")
    else:
        from train_gnn import train_all_properties as train_gnn_all
        return train_gnn_all("gnn")

# 导出所有内容，保持向后兼容
__all__ = [
    # 常量
    "ROOT", "SAVE_DIR", "LOG_DIR", "PROPERTIES", "RNG",
    "HAS_TORCH", "HAS_PYG", "TORCH_ERROR", "PYG_ERROR",
    "DESC_LIST", "ATOM_FEATS",
    # 化学工具函数
    "sanitize_smiles", "mol_to_sdf", "featurize_descriptors", "featurize_ecfp",
    "build_graph", "atom_feature_vector", "bond_feature_vector",
    "murcko_scaffold", "scaffold_split", "ensure_demo_dataset",
    # 数据预处理
    "preprocess_data",
    # 基线模型
    "BaselineModel", "train_baseline_main", "train_baseline_demo",
    "quick_baseline_weights_path",
    # GNN模型
    "GINRegressor", "GNNPack", "train_gnn", "train_gnn_main", "train_gnn_demo",
    "build_graph_dataset", "quick_gnn_weights_path",
    # 评估函数
    "eval_baseline_main", "eval_gnn_main", "gnn_predict_atom_importance", "calculate_metrics",
    # 预测函数
    "predict", "predict_property",
    # 训练所有性质
    "train_all_properties"
]
