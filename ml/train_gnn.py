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
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm

# 图神经网络（GIN）

class GINRegressor(nn.Module):
    # 初始化GIN（图同构网络）回归模型
    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.2, use_edge_attr=False, use_skip=True, use_bn=True):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip
        self.use_edge_attr = use_edge_attr
        
        # 输入投影层
        self.input_proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        
        mlps = []
        bns = []
        for i in range(layers):
            # 每层的MLP：使用更深的网络
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden) if use_bn else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden) if use_bn else nn.Identity(),
                nn.ReLU()
            )
            # 这里先使用标准GINConv，边特征可以通过其他方式融合
            mlps.append(GINConv(mlp, train_eps=True))
            if use_bn:
                # 使用LayerNorm替代BatchNorm，避免batch_size=1的问题
                bns.append(LayerNorm(hidden))
            else:
                bns.append(nn.Identity())
        
        self.convs = nn.ModuleList(mlps)
        self.bns = nn.ModuleList(bns)
        
        # 如果使用跳跃连接，需要融合多层特征
        if use_skip:
            pool_dim = hidden * layers  # 所有层的特征拼接
        else:
            pool_dim = hidden
        
        # 使用多种池化方式的组合
        pool_dim *= 3  # mean + max + sum
        
        # 最终预测层：使用更深的网络
        self.lin = nn.Sequential(
            nn.Linear(pool_dim, hidden * 2),
            nn.LayerNorm(hidden * 2) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    # 前向传播，处理图数据并返回预测值
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 存储所有层的特征（用于跳跃连接）
        xs = [x]
        
        # 通过GIN层
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        
        # 池化：使用所有层（如果使用跳跃连接）或仅最后一层
        if self.use_skip:
            # 拼接所有层的特征
            x_all = torch.cat(xs[1:], dim=-1)  # 跳过输入层
        else:
            x_all = xs[-1]
        
        # 多种池化方式的组合
        if hasattr(data, "batch"):
            x_sum = global_add_pool(x_all, data.batch)
            x_mean = global_mean_pool(x_all, data.batch)
            x_max = global_max_pool(x_all, data.batch)
        else:
            x_sum = x_all.sum(dim=0, keepdim=True)
            x_mean = x_all.mean(dim=0, keepdim=True)
            x_max = x_all.max(dim=0, keepdim=True)[0]
        
        # 拼接多种池化结果
        x_pooled = torch.cat([x_sum, x_mean, x_max], dim=-1)
        
        # 最终预测
        out = self.lin(x_pooled)
        return out.view(-1)


class PotentialNetLayer(nn.Module):
    """
    PotentialNet风格的单步消息传递层：按edge_type区分的MLP + GRU更新。
    """
    def __init__(self, hidden: int, edge_attr_dim: int, num_edge_types: int, dropout: float = 0.0, use_layernorm: bool = False):
        super().__init__()
        self.hidden = hidden
        self.edge_attr_dim = edge_attr_dim
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        mlps = []
        for _ in range(num_edge_types):
            mlps.append(nn.Sequential(
                nn.Linear(hidden + edge_attr_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            ))
        self.mlps = nn.ModuleList(mlps)
        self.gru = nn.GRUCell(hidden, hidden)
        self.ln = nn.LayerNorm(hidden) if use_layernorm else nn.Identity()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        device = h.device
        msg = torch.zeros_like(h, device=device)
        for etype, mlp in enumerate(self.mlps):
            mask = (edge_type == etype)
            if mask.sum() == 0:
                continue
            ei = edge_index[:, mask]
            src, dst = ei[0], ei[1]
            h_src = h[src]
            if edge_attr is not None and edge_attr.numel() > 0:
                ea = edge_attr[mask]
                if ea.dim() == 1:
                    ea = ea.unsqueeze(-1)
                h_in = torch.cat([h_src, ea], dim=-1)
            else:
                h_in = h_src
            m = mlp(h_in)
            m = F.dropout(m, p=self.dropout, training=self.training)
            msg.index_add_(0, dst, m)
        h_new = self.gru(msg, h)
        h_new = self.ln(h_new)
        return h_new


class PotentialNetRegressor(nn.Module):
    """
    基于Feinberg et al. 2018的PotentialNet回归模型（ligand-only）。
    - Stage1: bond-only K_bond步
    - Stage2: bond + spatial K_spatial步
    - Gather: 全分子sum pool
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        K_bond: int = 3,
        K_spatial: int = 3,
        num_distance_bins: int = 4,
        edge_attr_dim: int = 1,
        dropout: float = 0.1,
        use_layernorm: bool = False,
        pool: str = "sum",
    ):
        super().__init__()
        self.hidden = hidden
        self.K_bond = K_bond
        self.K_spatial = K_spatial
        self.num_distance_bins = num_distance_bins
        self.edge_attr_dim = edge_attr_dim
        self.pool = pool

        self.input_proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        bond_types = 1  # edge_type=0
        total_edge_types = bond_types + num_distance_bins

        self.bond_layers = nn.ModuleList([
            PotentialNetLayer(hidden, edge_attr_dim, bond_types, dropout=dropout, use_layernorm=use_layernorm)
            for _ in range(K_bond)
        ])
        self.spatial_layers = nn.ModuleList([
            PotentialNetLayer(hidden, edge_attr_dim, total_edge_types, dropout=dropout, use_layernorm=use_layernorm)
            for _ in range(K_spatial)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        if self.pool == "max":
            return global_max_pool(x, batch)
        return global_add_pool(x, batch)

    def forward(self, data: Data) -> torch.Tensor:
        h = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        edge_type = getattr(data, "edge_type", None)
        batch = getattr(data, "batch", torch.zeros(h.size(0), dtype=torch.long, device=h.device))

        # Stage 1: bond-only（edge_type==0）
        if edge_type is not None and self.K_bond > 0:
            mask = (edge_type == 0)
            if mask.any():
                bond_ei = edge_index[:, mask]
                bond_et = edge_type[mask]
                bond_ea = edge_attr[mask] if edge_attr is not None else None
                for layer in self.bond_layers:
                    h = layer(h, bond_ei, bond_et, bond_ea)

        # Stage 2: bond + spatial（全部边）
        if edge_type is not None and edge_index.numel() > 0 and self.K_spatial > 0:
            for layer in self.spatial_layers:
                h = layer(h, edge_index, edge_type, edge_attr)

        g = self._pool(h, batch)
        out = self.readout(g)
        return out.view(-1)

@dataclass
class GNNPack:
    model: Any
    in_dim: int

def build_graph_dataset(df: pd.DataFrame, target_col: str, graph_cfg: Dict[str, Any]):
    """从DataFrame构建PyG图列表，并收集简单统计量。"""
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("PyTorch Geometric not installed.")
    graphs = []
    stats = {
        "n_graphs": 0,
        "n_nodes": 0,
        "bond_edges": 0,
        "spatial_edges": 0,
        "n_3d_success": 0,
    }
    for smi, tgt in zip(df["smiles"], df[target_col]):
        _, mol = sanitize_smiles(smi)
        if not mol:
            continue
        g = build_graph(
            mol,
            use_3d=graph_cfg.get("use_3d", True),
            max_distance=graph_cfg.get("max_distance", 5.0),
            num_distance_bins=graph_cfg.get("num_distance_bins", 4),
            max_spatial_neighbors=graph_cfg.get("max_spatial_neighbors"),
            seed=graph_cfg.get("seed", 42),
        )
        if g is None:
            continue
        g.y = torch.tensor([float(tgt)], dtype=torch.float)
        graphs.append(g)
        stats["n_graphs"] += 1
        stats["n_nodes"] += g.num_nodes
        stats["bond_edges"] += int(getattr(g, "n_bond_edges", 0))
        stats["spatial_edges"] += int(getattr(g, "n_spatial_edges", 0))
        stats["n_3d_success"] += 1 if getattr(g, "has_3d", False) else 0
    return graphs, stats

# 训练GNN模型，使用配置参数进行训练并返回最佳模型（支持数据标准化）
def train_gnn(df: pd.DataFrame, target_col: str, config: Dict[str, Any]) -> GNNPack:
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("GNN deps not available")

    defaults = {
        "model_name": "potentialnet",
        "hidden": 128,
        "layers": 4,
        "dropout": 0.2,
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "weight_decay": 1e-4,
        "patience": 20,
        "use_edge_attr": False,
        "use_skip": True,
        "use_bn": True,
        "K_bond": 3,
        "K_spatial": 3,
        "num_distance_bins": 4,
        "max_distance": 5.0,
        "use_3d": True,
        "max_spatial_neighbors": 32,
        "pool": "sum",
        "use_layernorm": False,
        "seed": 42,
    }
    cfg = {**defaults, **(config or {})}

    train_df, val_df, _ = scaffold_split(df, (0.7, 0.15, 0.15), seed=cfg["seed"])
    graph_cfg = {
        "use_3d": cfg["use_3d"],
        "max_distance": cfg["max_distance"],
        "num_distance_bins": cfg["num_distance_bins"],
        "max_spatial_neighbors": cfg.get("max_spatial_neighbors"),
        "seed": cfg.get("seed", 42),
    }
    train_graphs, stats_tr = build_graph_dataset(train_df, target_col, graph_cfg)
    val_graphs, stats_va = build_graph_dataset(val_df, target_col, graph_cfg)
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
    hidden = cfg["hidden"]
    dropout = cfg["dropout"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    K_bond = cfg["K_bond"]
    K_spatial = cfg["K_spatial"]
    layers = cfg["layers"]
    if n_samples < 20:
        hidden = min(hidden, 64)
        dropout = min(dropout, 0.1)
        batch_size = max(1, min(4, n_samples))
        epochs = min(epochs, 15)
        K_bond = min(K_bond, 2)
        K_spatial = min(K_spatial, 2)
        lr = min(lr, 5e-4)
    elif n_samples < 50:
        hidden = min(hidden, 96)
        dropout = min(dropout, 0.15)
        batch_size = max(1, min(8, n_samples))
        epochs = min(epochs, 25)
        K_bond = min(K_bond, 3)
        K_spatial = min(K_spatial, 3)
        lr = min(lr, 1e-3)
    else:
        # 中等及大数据集保持配置
        batch_size = cfg.get("batch_size", 32)
        epochs = cfg.get("epochs", 100)

    model_name = cfg.get("model_name", "potentialnet").lower()
    if model_name == "gin":
        model = GINRegressor(
            in_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            use_edge_attr=cfg.get("use_edge_attr", False),
            use_skip=cfg.get("use_skip", True),
            use_bn=cfg.get("use_bn", True)
        ).to(device)
    else:
        model = PotentialNetRegressor(
            in_dim,
            hidden=hidden,
            K_bond=K_bond,
            K_spatial=K_spatial,
            num_distance_bins=cfg["num_distance_bins"],
            edge_attr_dim=1,
            dropout=dropout,
            use_layernorm=cfg.get("use_layernorm", False),
            pool=cfg.get("pool", "sum"),
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10, verbose=False, min_lr=1e-6
    )

    dl_tr = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(val_graphs, batch_size=max(1, min(64, len(val_graphs))))

    best = float("inf")
    best_state = None
    patience_counter = 0
    patience = cfg.get("patience", 20)

    def _print_graph_stats():
        total_graphs = stats_tr["n_graphs"] + stats_va["n_graphs"]
        if total_graphs == 0:
            return
        n_nodes = stats_tr["n_nodes"] + stats_va["n_nodes"]
        n_bonds = stats_tr["bond_edges"] + stats_va["bond_edges"]
        n_spatial = stats_tr["spatial_edges"] + stats_va["spatial_edges"]
        n3d = stats_tr["n_3d_success"] + stats_va["n_3d_success"]
        print(f"[graph] avg nodes: {n_nodes / total_graphs:.2f}, bond edges: {n_bonds / total_graphs:.2f}, spatial edges: {n_spatial / max(1,total_graphs):.2f}, 3d success rate: {n3d / total_graphs:.2f}")

    _print_graph_stats()

    for epoch in range(epochs):
        model.train()
        losses = []
        for b in dl_tr:
            b = b.to(device)
            opt.zero_grad()
            pred = model(b)
            loss = F.mse_loss(pred, b.y.view(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n = 0
            for b in dl_va:
                b = b.to(device)
                p = model(b)
                val_loss += F.mse_loss(p, b.y.view(-1), reduction="sum").item()
                n += b.y.numel()
            val_rmse = math.sqrt(val_loss / max(1, n))

        scheduler.step(val_rmse)

        if val_rmse < best:
            best = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val_rmse: {best:.6f}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # 保存标准化参数和模型配置到模型（通过添加属性）
    model.target_mean = target_mean
    model.target_std = target_std
    model.hidden_dim = hidden
    model.num_layers = layers
    model.dropout_rate = dropout
    model.model_name = model_name
    model.K_bond = K_bond
    model.K_spatial = K_spatial
    model.num_distance_bins = cfg["num_distance_bins"]
    model.max_distance = cfg["max_distance"]
    model.use_3d = cfg["use_3d"]
    model.pool = cfg.get("pool", "sum")
    model.graph_cfg = graph_cfg

    return GNNPack(model=model, in_dim=in_dim)

# 返回GNN模型权重文件的路径（支持多性质）
def quick_gnn_weights_path(property_name: str = "logp") -> str:
    return os.path.join(SAVE_DIR, f"gnn_{property_name}_v1.pth")

def run_smoke_test(cfg_path: str = None):
    """最小Smoke Test：2-4个SMILES构图并前向传播，验证输出shape。"""
    if not (HAS_TORCH and HAS_PYG):
        print("Skipping smoke test: missing torch/pyg")
        return
    cfg = {}
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    hidden = cfg.get("hidden", 64)
    K_bond = cfg.get("K_bond", 2)
    K_spatial = cfg.get("K_spatial", 2)
    num_distance_bins = cfg.get("num_distance_bins", 4)
    dropout = cfg.get("dropout", 0.1)
    pool = cfg.get("pool", "sum")
    graph_cfg = {
        "use_3d": cfg.get("use_3d", True),
        "max_distance": cfg.get("max_distance", 5.0),
        "num_distance_bins": num_distance_bins,
        "max_spatial_neighbors": cfg.get("max_spatial_neighbors", None),
        "seed": cfg.get("seed", 42),
    }
    sample_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    graphs = []
    for smi in sample_smiles:
        _, mol = sanitize_smiles(smi)
        if not mol:
            continue
        g = build_graph(
            mol,
            use_3d=graph_cfg["use_3d"],
            max_distance=graph_cfg["max_distance"],
            num_distance_bins=graph_cfg["num_distance_bins"],
            max_spatial_neighbors=graph_cfg.get("max_spatial_neighbors"),
            seed=graph_cfg.get("seed", 42),
        )
        if g is None:
            continue
        g.y = torch.tensor([0.0], dtype=torch.float)
        graphs.append(g)
    if not graphs:
        print("Smoke test skipped: graph construction failed")
        return
    in_dim = graphs[0].x.size(-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PotentialNetRegressor(
        in_dim,
        hidden=hidden,
        K_bond=K_bond,
        K_spatial=K_spatial,
        num_distance_bins=num_distance_bins,
        edge_attr_dim=1,
        dropout=dropout,
        pool=pool,
    ).to(device)
    dl = DataLoader(graphs, batch_size=2)
    with torch.no_grad():
        for batch in dl:
            out = model(batch.to(device))
            print(f"[smoke] batch size {batch.num_graphs}, out shape {tuple(out.shape)}")
    total_nodes = sum(g.num_nodes for g in graphs)
    total_bonds = sum(getattr(g, "n_bond_edges", 0) for g in graphs)
    total_spatial = sum(getattr(g, "n_spatial_edges", 0) for g in graphs)
    print(f"[smoke] graphs={len(graphs)}, avg nodes={total_nodes/len(graphs):.2f}, bond edges={total_bonds/len(graphs):.2f}, spatial edges={total_spatial/len(graphs):.2f}")

# 训练演示用的GNN模型并保存
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
            "hidden": getattr(pack.model, 'hidden_dim', 128),
            "layers": getattr(pack.model, 'num_layers', 4),
            "dropout": getattr(pack.model, 'dropout_rate', 0.2),
            "use_edge_attr": getattr(pack.model, 'use_edge_attr', False),
            "use_skip": getattr(pack.model, 'use_skip', True),
            "use_bn": getattr(pack.model, 'use_bn', True),
            "model_name": getattr(pack.model, 'model_name', 'potentialnet'),
            "K_bond": getattr(pack.model, 'K_bond', 3),
            "K_spatial": getattr(pack.model, 'K_spatial', 3),
            "num_distance_bins": getattr(pack.model, 'num_distance_bins', 4),
            "max_distance": getattr(pack.model, 'max_distance', 5.0),
            "use_3d": getattr(pack.model, 'use_3d', True),
            "pool": getattr(pack.model, 'pool', 'sum')
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
    p.add_argument("--smoke", action="store_true", help="run smoke test and exit")
    a = p.parse_args(args)
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("PyTorch Geometric not available")
    if a.smoke:
        run_smoke_test(a.cfg)
        return
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
            "hidden": getattr(pack.model, 'hidden_dim', 128),
            "layers": getattr(pack.model, 'num_layers', 4),
            "dropout": getattr(pack.model, 'dropout_rate', 0.2),
            "use_edge_attr": getattr(pack.model, 'use_edge_attr', False),
            "use_skip": getattr(pack.model, 'use_skip', True),
            "use_bn": getattr(pack.model, 'use_bn', True),
            "model_name": getattr(pack.model, 'model_name', 'potentialnet'),
            "K_bond": getattr(pack.model, 'K_bond', 3),
            "K_spatial": getattr(pack.model, 'K_spatial', 3),
            "num_distance_bins": getattr(pack.model, 'num_distance_bins', 4),
            "max_distance": getattr(pack.model, 'max_distance', 5.0),
            "use_3d": getattr(pack.model, 'use_3d', True),
            "pool": getattr(pack.model, 'pool', 'sum')
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
