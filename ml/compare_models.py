"""
模型比较脚本
比较Baseline和GNN模型在不同性质上的表现，并生成比较报告
"""

import os
import json
import pandas as pd
from typing import Dict, Any

from utils import ROOT, PROPERTIES, scaffold_split, preprocess_data
from train_baseline import BaselineModel, quick_baseline_weights_path
from eval_baseline import calculate_metrics

try:
    from train_gnn import GINRegressor, PotentialNetRegressor, GNNPack, quick_gnn_weights_path
    from eval_gnn import gnn_predict_atom_importance
    from utils import HAS_TORCH, HAS_PYG, sanitize_smiles, build_graph
    import torch
    import yaml
    GNN_AVAILABLE = HAS_TORCH and HAS_PYG
except (RuntimeError, ImportError):
    GNN_AVAILABLE = False

def compare_models(property_name: str = "logp") -> Dict[str, Any]:
    """
    比较Baseline和GNN模型在指定性质上的表现
    
    参数:
        property_name: 性质名称
    
    返回:
        包含比较结果的字典
    """
    data_file = os.path.join(ROOT, "data", PROPERTIES.get(property_name, PROPERTIES["logp"])["data_file"])
    
    if not os.path.exists(data_file):
        return {"error": f"Data file not found: {data_file}"}
    
    df = pd.read_csv(data_file)
    # 数据预处理
    df = preprocess_data(df, "target", remove_outliers=True, outlier_method="iqr")
    
    # 使用scaffold_split划分数据集
    train_df, val_df, test_df = scaffold_split(df, frac=(0.7, 0.15, 0.15), seed=42)
    
    results = {
        "property": property_name,
        "property_name": PROPERTIES.get(property_name, {}).get("name", property_name),
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "baseline": {},
        "gnn": {}
    }
    
    # 评估Baseline模型
    baseline_path = quick_baseline_weights_path(property_name)
    if os.path.exists(baseline_path):
        try:
            bm = BaselineModel.load(baseline_path)
            y_true, y_pred = [], []
            for smi, y in zip(test_df["smiles"], test_df["target"]):
                y_true.append(float(y))
                y_pred.append(bm.predict(smi, return_shap=False)["prediction"])
            results["baseline"] = calculate_metrics(y_true, y_pred)
            results["baseline"]["model_path"] = baseline_path
        except Exception as e:
            results["baseline"] = {"error": str(e)}
    else:
        results["baseline"] = {"error": f"Model not found: {baseline_path}"}
    
    # 评估GNN模型
    if GNN_AVAILABLE:
        gnn_path = quick_gnn_weights_path(property_name)
        norm_path = gnn_path.replace(".pth", "_norm.json")
        
        if os.path.exists(gnn_path):
            try:
                # 加载模型配置（统一使用 config 变量，简化逻辑）
                config = {}
                if os.path.exists(norm_path):
                    with open(norm_path, "r") as f:
                        config = json.load(f)
                else:
                    with open(os.path.join(ROOT, "configs", "gnn.yaml"), "r") as f:
                        config = yaml.safe_load(f) or {}
                
                # 提取配置参数（使用与 train_gnn.py 一致的默认值）
                hidden = config.get("hidden", 128)
                layers = config.get("layers", 4)
                dropout = config.get("dropout", 0.2)
                use_edge_attr = config.get("use_edge_attr", False)
                use_skip = config.get("use_skip", True)
                use_bn = config.get("use_bn", True)
                model_name = config.get("model_name", "potentialnet")
                K_bond = config.get("K_bond", 3)
                K_spatial = config.get("K_spatial", 3)
                num_distance_bins = config.get("num_distance_bins", 4)
                max_distance = config.get("max_distance", 5.0)
                use_3d = config.get("use_3d", True)
                pool = config.get("pool", "sum")
                seed = config.get("seed", 42)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                graph_cfg = {
                    "use_3d": use_3d,
                    "max_distance": max_distance,
                    "num_distance_bins": num_distance_bins,
                    "max_spatial_neighbors": config.get("max_spatial_neighbors"),
                    "seed": seed,
                }
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
                
                # 加载标准化参数
                if os.path.exists(norm_path):
                    with open(norm_path, "r") as f:
                        norm_data = json.load(f)
                        model_g.target_mean = norm_data.get("mean", 0.0)
                        model_g.target_std = norm_data.get("std", 1.0)
                
                # 尝试加载模型权重，如果架构不匹配则给出提示
                try:
                    state_dict = torch.load(gnn_path, map_location=device)
                    model_g.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    # 如果加载失败，可能是旧模型架构，提示需要重新训练
                    results["gnn"] = {
                        "error": f"Model architecture mismatch. Please retrain the model with the new architecture. Original error: {str(e)}"
                    }
                else:
                    # 只有在模型加载成功时才继续评估
                    model_g.graph_cfg = graph_cfg
                    pack = GNNPack(model_g, in_dim)
                    
                    y_true, y_pred = [], []
                    for smi, y in zip(test_df["smiles"], test_df["target"]):
                        y_true.append(float(y))
                        out = gnn_predict_atom_importance(pack, smi, graph_cfg=graph_cfg)
                        y_pred.append(out["prediction"])
                    
                    results["gnn"] = calculate_metrics(y_true, y_pred)
                    results["gnn"]["model_path"] = gnn_path
            except Exception as e:
                results["gnn"] = {"error": str(e)}
        else:
            results["gnn"] = {"error": f"Model not found: {gnn_path}"}
    else:
        results["gnn"] = {"error": "GNN dependencies not available"}
    
    # 确定更好的模型（如果两个模型都成功）
    if "error" not in results["baseline"] and "error" not in results["gnn"]:
        baseline_rmse = results["baseline"].get("rmse", float('inf'))
        gnn_rmse = results["gnn"].get("rmse", float('inf'))
        results["better_model"] = "gnn" if gnn_rmse < baseline_rmse else "baseline"
    
    # 固定保存结果到 ml 目录
    output_file = os.path.join(ROOT, f"{property_name}_comparison.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存到: {output_file}")
    
    return results

def compare_all_properties() -> Dict[str, Any]:
    """
    比较所有性质的模型表现
    
    返回:
        包含所有性质比较结果的字典
    """
    all_results = {}
    summary = {
        "baseline_wins": 0,
        "gnn_wins": 0,
        "properties": []
    }
    
    for prop_key in PROPERTIES.keys():
        print(f"比较 {prop_key} 模型...")
        result = compare_models(prop_key)
        all_results[prop_key] = result
        
        # 更新摘要
        if "better_model" in result:
            if result["better_model"] == "baseline":
                summary["baseline_wins"] += 1
            else:
                summary["gnn_wins"] += 1
            summary["properties"].append({
                "property": prop_key,
                "better_model": result["better_model"]
            })
    
    # 固定保存摘要到 ml 目录
    summary_file = os.path.join(ROOT, "comparison_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n比较摘要已保存到: {summary_file}")
    
    return {"summary": summary, "details": all_results}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--property", default=None, help="要比较的性质名称（默认：所有性质）")
    a = p.parse_args()
    
    if a.property:
        result = compare_models(a.property)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = compare_all_properties()
        print("\n=== 模型比较摘要 ===")
        print(f"Baseline 更好的性质数: {result['summary']['baseline_wins']}")
        print(f"GNN 更好的性质数: {result['summary']['gnn_wins']}")
        print("\n详细结果:")
        for prop_info in result['summary']['properties']:
            print(f"  {prop_info['property']}: {prop_info['better_model']} 更好")
