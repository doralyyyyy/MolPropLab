"""
训练GNN模型
包含 GINRegressor 类和训练相关函数
"""

from __future__ import annotations
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

from utils import (
    ROOT, SAVE_DIR, PROPERTIES, HAS_TORCH, HAS_PYG,
    sanitize_smiles, build_graph, scaffold_split, ensure_demo_dataset, preprocess_data
)

if not (HAS_TORCH and HAS_PYG):
    raise RuntimeError("PyTorch Geometric not available")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

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

@dataclass
class GNNPack:
    model: Any
    in_dim: int

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
        # 根据数据集大小调整batch_size和训练轮数
        if n_samples < 200:
            batch_size = config.get("batch_size", 32)
            epochs = config.get("epochs", 30)
        elif n_samples < 1000:
            # 中等大数据集（200-1000）：适度增加batch_size和训练轮数
            batch_size = config.get("batch_size", 32)
            epochs = config.get("epochs", 50)
        else:
            # 大数据集（>=1000）：使用更大的batch_size和更多训练轮数以充分利用数据
            batch_size = min(config.get("batch_size", 64), 64)  # 最大64，避免内存溢出
            epochs = config.get("epochs", 80)  # 大数据集使用更多训练轮数
    
    model = GINRegressor(in_dim, hidden=hidden, layers=layers, dropout=dropout).to(device)
    # 根据数据集大小调整学习率
    if n_samples < 100:
        lr = 5e-4  # 小数据集使用较小的学习率
    elif n_samples < 500:
        lr = 1e-3  # 中等数据集使用标准学习率
    elif n_samples < 1000:
        lr = config.get("lr", 1e-3)  # 中等大数据集使用配置的学习率
    else:
        # 大数据集（>=1000）：可以使用稍小的学习率以获得更稳定的训练
        lr = config.get("lr", 8e-4)  # 大数据集使用稍小的学习率
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

# 返回GNN模型权重文件的路径（支持多性质）
def quick_gnn_weights_path(property_name: str = "logp") -> str:
    return os.path.join(SAVE_DIR, f"gnn_{property_name}_v1.pth")

# 训练演示用的GNN模型并保存（支持多性质）
def train_gnn_demo(property_name: str = "logp") -> str:
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("GNN deps missing")
    data_file = os.path.join(ROOT, "data", PROPERTIES.get(property_name, PROPERTIES["logp"])["data_file"])
    df = ensure_demo_dataset(data_file)
    # 数据预处理：处理缺失值和异常值
    df = preprocess_data(df, "target", remove_outliers=True, outlier_method="iqr")
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
def train_all_properties(model_type: str = "gnn"):
    """训练所有性质的模型"""
    if model_type != "gnn":
        # 如果是baseline，需要从train_baseline导入
        from train_baseline import train_all_properties as train_baseline_all
        return train_baseline_all("baseline")
    
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("GNN dependencies not available")
    
    results = {}
    for prop_key in PROPERTIES.keys():
        try:
            path = train_gnn_demo(prop_key)
            results[prop_key] = path
            print(f"✓ Trained {prop_key} model: {path}")
        except Exception as e:
            print(f"✗ Failed to train {prop_key}: {e}")
            results[prop_key] = None
    return results

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
    # 数据预处理：处理缺失值和异常值
    df = preprocess_data(df, a.target, remove_outliers=True, outlier_method="iqr")
    print(f"数据预处理后剩余样本数: {len(df)}")
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

if __name__ == "__main__":
    print("开始训练所有性质的GNN模型...")
    print("=" * 50)
    results = train_all_properties("gnn")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"成功训练: {success_count}/{len(results)} 个模型")
    
    if any(v is None for v in results.values()):
        print("\n失败的模型:")
        for prop, path in results.items():
            if path is None:
                print(f"  - {prop}")
