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
    from train_gnn import GINRegressor, GNNPack, quick_gnn_weights_path
    from eval_gnn import gnn_predict_atom_importance
    from utils import HAS_TORCH, HAS_PYG, sanitize_smiles, build_graph
    import torch
    import yaml
    GNN_AVAILABLE = HAS_TORCH and HAS_PYG
except (RuntimeError, ImportError):
    GNN_AVAILABLE = False

def compare_models(property_name: str = "logp", output_file: str = None) -> Dict[str, Any]:
    """
    比较Baseline和GNN模型在指定性质上的表现
    
    参数:
        property_name: 性质名称
        output_file: 输出文件路径（可选，如果提供则保存结果）
    
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
                # 加载模型配置
                if os.path.exists(norm_path):
                    with open(norm_path, "r") as f:
                        saved_config = json.load(f)
                    hidden = saved_config.get("hidden", 64)
                    layers = saved_config.get("layers", 3)
                    dropout = saved_config.get("dropout", 0.2)
                else:
                    with open(os.path.join(ROOT, "configs", "gnn.yaml"), "r") as f:
                        cfg = yaml.safe_load(f)
                    hidden = cfg.get("hidden", 64)
                    layers = cfg.get("layers", 3)
                    dropout = cfg.get("dropout", 0.2)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _, mol = sanitize_smiles(test_df.iloc[0]["smiles"])
                g = build_graph(mol)
                in_dim = g.x.size(-1)
                model_g = GINRegressor(in_dim, hidden=hidden, layers=layers, dropout=dropout).to(device)
                
                # 加载标准化参数
                if os.path.exists(norm_path):
                    with open(norm_path, "r") as f:
                        norm_data = json.load(f)
                        model_g.target_mean = norm_data.get("mean", 0.0)
                        model_g.target_std = norm_data.get("std", 1.0)
                
                model_g.load_state_dict(torch.load(gnn_path, map_location=device), strict=False)
                pack = GNNPack(model_g, in_dim)
                
                y_true, y_pred = [], []
                for smi, y in zip(test_df["smiles"], test_df["target"]):
                    y_true.append(float(y))
                    out = gnn_predict_atom_importance(pack, smi)
                    y_pred.append(out["prediction"])
                
                results["gnn"] = calculate_metrics(y_true, y_pred)
                results["gnn"]["model_path"] = gnn_path
            except Exception as e:
                results["gnn"] = {"error": str(e)}
        else:
            results["gnn"] = {"error": f"Model not found: {gnn_path}"}
    else:
        results["gnn"] = {"error": "GNN dependencies not available"}
    
    # 计算改进百分比（如果两个模型都成功）
    if "error" not in results["baseline"] and "error" not in results["gnn"]:
        baseline_rmse = results["baseline"].get("rmse", float('inf'))
        gnn_rmse = results["gnn"].get("rmse", float('inf'))
        if baseline_rmse > 0:
            improvement = ((baseline_rmse - gnn_rmse) / baseline_rmse) * 100
            results["improvement_percent"] = float(improvement)
            results["better_model"] = "gnn" if improvement > 0 else "baseline"
    
    # 保存结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

def compare_all_properties(output_dir: str = None) -> Dict[str, Any]:
    """
    比较所有性质的模型表现
    
    参数:
        output_dir: 输出目录（可选）
    
    返回:
        包含所有性质比较结果的字典
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    summary = {
        "baseline_wins": 0,
        "gnn_wins": 0,
        "properties": []
    }
    
    for prop_key in PROPERTIES.keys():
        print(f"比较 {prop_key} 模型...")
        output_file = os.path.join(output_dir, f"{prop_key}_comparison.json") if output_dir else None
        result = compare_models(prop_key, output_file)
        all_results[prop_key] = result
        
        # 更新摘要
        if "better_model" in result:
            if result["better_model"] == "baseline":
                summary["baseline_wins"] += 1
            else:
                summary["gnn_wins"] += 1
            summary["properties"].append({
                "property": prop_key,
                "better_model": result["better_model"],
                "improvement": result.get("improvement_percent", 0)
            })
    
    if output_dir:
        summary_file = os.path.join(output_dir, "comparison_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": all_results}, f, indent=2, ensure_ascii=False)
        print(f"\n比较结果已保存到: {summary_file}")
    
    return {"summary": summary, "details": all_results}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--property", default=None, help="要比较的性质名称（默认：所有性质）")
    p.add_argument("--output", default=None, help="输出文件或目录路径")
    a = p.parse_args()
    
    if a.property:
        result = compare_models(a.property, a.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = compare_all_properties(a.output)
        print("\n=== 模型比较摘要 ===")
        print(f"Baseline 更好的性质数: {result['summary']['baseline_wins']}")
        print(f"GNN 更好的性质数: {result['summary']['gnn_wins']}")
        print("\n详细结果:")
        for prop_info in result['summary']['properties']:
            print(f"  {prop_info['property']}: {prop_info['better_model']} 更好 (改进: {prop_info['improvement']:.2f}%)")
