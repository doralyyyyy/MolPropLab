"""
评估GNN模型
计算RMSE和R2分数，包含GNN预测函数
"""

import os
import json
import math
import argparse
import pandas as pd
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import ROOT, HAS_TORCH, HAS_PYG, sanitize_smiles, build_graph, scaffold_split, preprocess_data
from train_gnn import GINRegressor, PotentialNetRegressor, GNNPack, quick_gnn_weights_path
from eval_baseline import calculate_metrics

if not (HAS_TORCH and HAS_PYG):
    print(json.dumps({"error":"gnn deps missing"}))
    exit(0)

import torch
import numpy as np
from typing import Dict, Any, Optional

# 使用GNN模型进行预测，使用MC-Dropout估计不确定性，计算原子重要性
def gnn_predict_atom_importance(pack: GNNPack, smiles: str, graph_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from utils import mol_to_sdf
    _, mol = sanitize_smiles(smiles)
    if not mol: return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": ""}
    cfg = graph_cfg or getattr(pack.model, "graph_cfg", {}) or {}
    try:
        g = build_graph(
            mol,
            use_3d=cfg.get("use_3d", True),
            max_distance=cfg.get("max_distance", 5.0),
            num_distance_bins=cfg.get("num_distance_bins", 4),
            max_spatial_neighbors=cfg.get("max_spatial_neighbors"),
            seed=cfg.get("seed", 42),
        )
    except TypeError:
        g = build_graph(mol)
    if g is None: return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": ""}
    device = next(pack.model.parameters()).device
    
    # 获取标准化参数
    target_mean = getattr(pack.model, 'target_mean', 0.0)
    target_std = getattr(pack.model, 'target_std', 1.0)
    
    pack.model.eval()
    # 使用MC-Dropout估计不确定性
    T = 20
    preds_normalized = []
    for _ in range(T):
        # 手动设置dropout层为训练模式
        for module in pack.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
        with torch.no_grad():
            p = pack.model(g.to(device)).item()
            preds_normalized.append(p)
        # 恢复dropout为eval模式
        for module in pack.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
    
    # 计算标准化空间的不确定性
    pred_normalized = float(np.mean(preds_normalized))
    unc_normalized = float(np.std(preds_normalized, ddof=1))
    
    # 反标准化预测值
    pred = pred_normalized * target_std + target_mean
    # 不确定性：标准化空间的不确定性乘以标准差
    unc = unc_normalized * target_std
    # 基于梯度的节点重要性计算
    pack.model.eval()
    g = g.to(device)
    g.x.requires_grad_(True)
    y = pack.model(g)
    y = y.view(-1)[0]
    y.backward()
    grads = g.x.grad.detach().abs().sum(dim=1).cpu().numpy()
    if grads.max() > 0:
        imps = (grads / grads.max()).tolist()
    else:
        imps = grads.tolist()
    sdf = mol_to_sdf(mol)
    return {"prediction": pred, "uncertainty": unc, "atom_importances": imps, "sdf": sdf}

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
    # 数据预处理（与训练时一致）
    df = preprocess_data(df, a.target, remove_outliers=True, outlier_method="iqr")
    
    # 尝试加载保存的模型配置
    cfg = {}
    norm_path = a.model.replace(".pth", "_norm.json")
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            saved_config = json.load(f)
        cfg = saved_config
        hidden = saved_config.get("hidden", 128)
        layers = saved_config.get("layers", 4)
        dropout = saved_config.get("dropout", 0.2)
        use_edge_attr = saved_config.get("use_edge_attr", False)
        use_skip = saved_config.get("use_skip", True)
        use_bn = saved_config.get("use_bn", True)
        model_name = saved_config.get("model_name", "potentialnet")
        K_bond = saved_config.get("K_bond", 3)
        K_spatial = saved_config.get("K_spatial", 3)
        num_distance_bins = saved_config.get("num_distance_bins", 4)
        max_distance = saved_config.get("max_distance", 5.0)
        use_3d = saved_config.get("use_3d", True)
        pool = saved_config.get("pool", "sum")
    else:
        with open(a.cfg, "r") as f:
            cfg = yaml.safe_load(f)
        hidden = cfg.get("hidden", 128)
        layers = cfg.get("layers", 4)
        dropout = cfg.get("dropout", 0.2)
        use_edge_attr = cfg.get("use_edge_attr", False)
        use_skip = cfg.get("use_skip", True)
        use_bn = cfg.get("use_bn", True)
        model_name = cfg.get("model_name", "potentialnet")
        K_bond = cfg.get("K_bond", 3)
        K_spatial = cfg.get("K_spatial", 3)
        num_distance_bins = cfg.get("num_distance_bins", 4)
        max_distance = cfg.get("max_distance", 5.0)
        use_3d = cfg.get("use_3d", True)
        pool = cfg.get("pool", "sum")
    
    # 使用scaffold_split进行数据集划分（与训练时一致，使用配置中的 seed）
    seed = cfg.get("seed", 42)
    train_df, val_df, test_df = scaffold_split(df, frac=(0.7, 0.15, 0.15), seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_cfg = {
        "use_3d": use_3d,
        "max_distance": max_distance,
        "num_distance_bins": num_distance_bins,
        "max_spatial_neighbors": cfg.get("max_spatial_neighbors"),
        "seed": seed,
    }
    # 构建一个虚拟图以获取特征维度
    _, mol = sanitize_smiles(test_df.iloc[0]["smiles"])
    g = build_graph(
        mol,
        use_3d=graph_cfg["use_3d"],
        max_distance=graph_cfg["max_distance"],
        num_distance_bins=graph_cfg["num_distance_bins"],
        max_spatial_neighbors=graph_cfg.get("max_spatial_neighbors"),
        seed=graph_cfg.get("seed", 42),
    )
    in_dim = g.x.size(-1)
    if model_name == "gin":
        model_g = GINRegressor(
            in_dim, 
            hidden=hidden, 
            layers=layers, 
            dropout=dropout,
            use_edge_attr=use_edge_attr,
            use_skip=use_skip,
            use_bn=use_bn
        ).to(device)
    else:
        model_g = PotentialNetRegressor(
            in_dim,
            hidden=hidden,
            K_bond=K_bond,
            K_spatial=K_spatial,
            num_distance_bins=num_distance_bins,
            edge_attr_dim=1,
            dropout=dropout,
            pool=pool,
        ).to(device)
    # 先加载标准化参数，确保即使模型文件不存在也能设置默认值
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            norm_data = json.load(f)
            model_g.target_mean = norm_data.get("mean", 0.0)
            model_g.target_std = norm_data.get("std", 1.0)
    else:
        # 如果没有标准化参数文件，使用默认值
        model_g.target_mean = 0.0
        model_g.target_std = 1.0
    
    if os.path.exists(a.model):
        # 尝试加载模型权重，如果架构不匹配则给出提示
        try:
            state_dict = torch.load(a.model, map_location=device)
            model_g.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(json.dumps({
                "error": f"Model architecture mismatch. Please retrain the model with the new architecture. Original error: {str(e)}"
            }, indent=2))
            return
    
    # 在测试集上评估
    y_true, y_pred = [], []
    for smi, y in zip(test_df["smiles"], test_df[a.target]):
        y_true.append(float(y))
        out = gnn_predict_atom_importance(GNNPack(model_g, in_dim), smi, graph_cfg=graph_cfg)
        y_pred.append(out["prediction"])
    
    metrics = calculate_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    eval_gnn_main()
