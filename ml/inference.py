"""
命令行接口和可导入的预测函数封装。

用法示例:
 python inference.py --smiles "CCO" --model baseline --json
 python inference.py --csv data/logp.csv --output out.csv
 python inference.py --xlsx data/example.xlsx --output out.csv
"""
import sys, json, argparse, os, csv
from typing import Dict, Any

from utils import ROOT, PROPERTIES, HAS_TORCH, HAS_PYG, sanitize_smiles, build_graph, ensure_demo_dataset, preprocess_data, scaffold_split
from train_baseline import BaselineModel, quick_baseline_weights_path

# GNN相关导入（可能不可用）
try:
    from train_gnn import GINRegressor, PotentialNetRegressor, GNNPack, quick_gnn_weights_path
    from eval_gnn import gnn_predict_atom_importance
    GNN_AVAILABLE = True
except (RuntimeError, ImportError):
    GINRegressor = None
    GNNPack = None
    quick_gnn_weights_path = None
    gnn_predict_atom_importance = None
    GNN_AVAILABLE = False

if HAS_TORCH and HAS_PYG:
    import torch
    import yaml

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
            # 回退训练：使用正确的预处理和数据划分，避免数据泄露
            df = ensure_demo_dataset(data_file)
            df = preprocess_data(df, "target", remove_outliers=True, outlier_method="iqr")
            train_df, _, _ = scaffold_split(df, frac=(0.7, 0.15, 0.15), seed=42)
            bm = BaselineModel(n_models=3, nbits=1024)
            bm.fit(train_df, "target")  # 仅使用训练集
            bm.save(path)
        out = bm.predict(smiles, return_shap=True)
        out["model"] = "baseline"
        out["version"] = version
        return out
    else:
        if not (HAS_TORCH and HAS_PYG) or not GNN_AVAILABLE:
            return predict_property(smiles, property_name, model="baseline")
        path = quick_gnn_weights_path(property_name)
        _, mol = sanitize_smiles(smiles)
        if not mol:
            return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": "", "model":"gnn","version":version}
        # 读取配置
        norm_path = path.replace(".pth", "_norm.json")
        # 读取配置（统一默认值与 train_gnn.py 一致）
        config = {}
        if os.path.exists(norm_path):
            with open(norm_path, "r") as f:
                config = json.load(f)
        else:
            # 如果没有保存的配置，使用 gnn.yaml 配置
            with open(os.path.join(ROOT, "configs", "gnn.yaml"), "r") as f:
                config = yaml.safe_load(f) or {}
        
        # 使用与 train_gnn.py 一致的默认值
        hidden = config.get("hidden", 128)
        layers = config.get("layers", 4)
        dropout = config.get("dropout", 0.2)
        model_name = config.get("model_name", "potentialnet")
        K_bond = config.get("K_bond", 3)
        K_spatial = config.get("K_spatial", 3)
        num_distance_bins = config.get("num_distance_bins", 4)
        max_distance = config.get("max_distance", 5.0)
        use_3d = config.get("use_3d", True)
        pool = config.get("pool", "sum")
        seed = config.get("seed", 42)
        
        graph_cfg = {
            "use_3d": use_3d,
            "max_distance": max_distance,
            "num_distance_bins": num_distance_bins,
            "max_spatial_neighbors": config.get("max_spatial_neighbors"),
            "seed": seed,
        }
        g = build_graph(
            mol,
            use_3d=graph_cfg["use_3d"],
            max_distance=graph_cfg["max_distance"],
            num_distance_bins=graph_cfg["num_distance_bins"],
            max_spatial_neighbors=graph_cfg.get("max_spatial_neighbors"),
            seed=graph_cfg.get("seed", 42),
        )
        if g is None:
            return predict_property(smiles, property_name, model="baseline")
        in_dim = g.x.size(-1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "gin":
            model_g = GINRegressor(in_dim, hidden=hidden, layers=layers, dropout=dropout).to(device)
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
            # 回退训练：添加数据预处理，train_gnn 内部会做 scaffold_split
            df = ensure_demo_dataset(data_file)
            df = preprocess_data(df, "target", remove_outliers=True, outlier_method="iqr")
            with open(os.path.join(ROOT, "configs", "gnn.yaml"), "r") as f:
                train_cfg = yaml.safe_load(f)
            from train_gnn import train_gnn
            try:
                pack = train_gnn(df, "target", train_cfg)
            except RuntimeError as e:
                # 如果训练失败，回退到baseline
                return predict_property(smiles, property_name, model="baseline")
            model_g = pack.model
            torch.save(model_g.state_dict(), path)
            # 保存标准化参数和模型配置（使用与 train_gnn.py 一致的默认值）
            norm_path = path.replace(".pth", "_norm.json")
            with open(norm_path, "w") as f:
                json.dump({
                    "mean": getattr(model_g, 'target_mean', 0.0), 
                    "std": getattr(model_g, 'target_std', 1.0),
                    "hidden": getattr(model_g, 'hidden_dim', 128),
                    "layers": getattr(model_g, 'num_layers', 4),
                    "dropout": getattr(model_g, 'dropout_rate', 0.2),
                    "model_name": getattr(model_g, 'model_name', 'potentialnet'),
                    "K_bond": getattr(model_g, 'K_bond', 3),
                    "K_spatial": getattr(model_g, 'K_spatial', 3),
                    "num_distance_bins": getattr(model_g, 'num_distance_bins', 4),
                    "max_distance": getattr(model_g, 'max_distance', 5.0),
                    "use_3d": getattr(model_g, 'use_3d', True),
                    "pool": getattr(model_g, 'pool', 'sum')
                }, f)
        model_g.graph_cfg = graph_cfg
        pack = GNNPack(model=model_g, in_dim=in_dim)
        out = gnn_predict_atom_importance(pack, smiles, graph_cfg=graph_cfg)
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

# 主函数，处理命令行参数并执行单条预测或批量预测
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles", type=str, default=None)
    p.add_argument("--model", type=str, default="gnn", choices=["gnn","baseline"])
    p.add_argument("--json", action="store_true", help="print JSON result")
    p.add_argument("--csv", type=str, default=None, help="input CSV with 'smiles' column")
    p.add_argument("--xlsx", type=str, default=None, help="input XLSX with 'smiles' column")
    p.add_argument("--output", type=str, default=None, help="CSV output path")
    a = p.parse_args()

    if a.smiles:
        out = predict(a.smiles, model=a.model)
        if a.json:
            print(json.dumps(out))
        else:
            print(out)
        sys.exit(0)

    if a.csv or a.xlsx:
        in_path = None
        data = []
        
        if a.csv:
            in_path = a.csv if os.path.isabs(a.csv) else os.path.join(os.path.dirname(__file__), a.csv)
            if not os.path.exists(in_path):
                print(f"Missing file: {in_path}", file=sys.stderr); sys.exit(1)
            with open(in_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                data = list(reader)
        elif a.xlsx:
            try:
                import pandas as pd
            except ImportError:
                print("XLSX支持需要pandas和openpyxl。请安装: pip install pandas openpyxl", file=sys.stderr)
                sys.exit(1)
            in_path = a.xlsx if os.path.isabs(a.xlsx) else os.path.join(os.path.dirname(__file__), a.xlsx)
            if not os.path.exists(in_path):
                print(f"Missing file: {in_path}", file=sys.stderr); sys.exit(1)
            df = pd.read_excel(in_path)
            data = df.to_dict("records")
        
        rows = []
        total = len(data) or 1
        for i, row in enumerate(data, 1):
            smi = row.get("smiles","")
            res: Dict[str, Any] = predict(smi, model=a.model)
            # 构建行数据，包含所有性质的预测
            row_data = {"smiles": smi}
            if res.get("properties"):
                # 多性质格式
                for prop_key, prop_data in res["properties"].items():
                    row_data[f"{prop_key}_prediction"] = prop_data.get("prediction", "")
                    row_data[f"{prop_key}_uncertainty"] = prop_data.get("uncertainty", "")
            else:
                # 向后兼容：单性质格式
                row_data["prediction"] = res.get("prediction", "")
                row_data["uncertainty"] = res.get("uncertainty", "")
            row_data["atom_importances_json"] = json.dumps(res.get("atom_importances", []))
            rows.append(row_data)
            print(f"PROGRESS {i}/{total}")
            sys.stdout.flush()
        outp = a.output or os.path.join(os.path.dirname(in_path), "results.csv")
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {outp}")
        sys.exit(0)

    p.print_help()

if __name__ == "__main__":
    main()
