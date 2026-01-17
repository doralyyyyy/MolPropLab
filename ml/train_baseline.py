"""
训练基线模型（LightGBM/RandomForest）
包含 BaselineModel 类和训练相关函数
"""

from __future__ import annotations
import os
import pickle
import warnings
import argparse
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from utils import (
    ROOT, SAVE_DIR, PROPERTIES, DESC_LIST,
    sanitize_smiles, featurize_descriptors, featurize_ecfp, mol_to_sdf,
    ensure_demo_dataset, preprocess_data, scaffold_split
)

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import RandomForestRegressor

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
                min_data_in_leaf = 1     # 最小值为1，允许单样本叶子
                min_data_in_bin = 1
                min_child_samples = 1    # 允许单样本分裂
                subsample = 1.0          # 使用全部数据
                colsample_bytree = 1.0   # 使用全部特征
                max_depth = 3            # 限制树深度，避免过拟合
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
                    # 大数据集（>=500）根据数据量进一步优化参数
                    if n_samples < 1000:
                        # 中等大数据集（500-1000）：使用适中的参数
                        n_estimators = 400
                        min_data_in_leaf = 20
                        min_data_in_bin = 5
                        min_child_samples = 20
                        subsample = 0.9
                        colsample_bytree = 0.9
                        max_depth = -1         # 不限制深度
                        min_gain_to_split = 0.0
                    else:
                        # 超大数据集（>=1000）：增加模型复杂度以获得更好性能
                        n_estimators = 600     # 增加树的数量
                        min_data_in_leaf = 30  # 稍微增加叶子节点最小样本数，防止过拟合
                        min_data_in_bin = 5
                        min_child_samples = 30
                        subsample = 0.9
                        colsample_bytree = 0.9
                        max_depth = -1         # 不限制深度
                        min_gain_to_split = 0.0
            
            # 彻底抑制LightGBM的警告输出
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
        
        # 特征归一化
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self.feature_mean) / self.feature_std
        
        self.models = [self._train_one(X_normalized, y, seed=i+7) for i in range(self.n_models)]
        return self

    # 对SMILES字符串进行预测，返回预测值、不确定性和原子重要性
    def predict(self, smiles: str, return_shap: bool = True) -> Dict[str, Any]:
        _, mol = sanitize_smiles(smiles)
        if not mol:
            return {"prediction": float("nan"), "uncertainty": float("nan"), "atom_importances": [], "sdf": ""}
        desc = featurize_descriptors(mol)
        ecfp, bitInfo = featurize_ecfp(mol, self.nbits)
        x = np.concatenate([desc, ecfp], axis=0).reshape(1, -1)
        # 应用特征归一化（使用训练时的均值和标准差）
        if hasattr(self, 'feature_mean') and hasattr(self, 'feature_std'):
            x = (x - self.feature_mean) / self.feature_std
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
            pickle.dump({
                "n_models": self.n_models, 
                "nbits": self.nbits, 
                "models": self.models, 
                "is_lgb": self.is_lgb,
                "feature_mean": getattr(self, 'feature_mean', None),
                "feature_std": getattr(self, 'feature_std', None)
            }, f)

    # 从文件加载模型
    @staticmethod
    def load(path: str) -> "BaselineModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        bm = BaselineModel(n_models=obj["n_models"], nbits=obj["nbits"])
        bm.models = obj["models"]; bm.is_lgb = obj.get("is_lgb", True)
        # 加载归一化参数
        bm.feature_mean = obj.get("feature_mean", None)
        bm.feature_std = obj.get("feature_std", None)
        return bm

# 返回基线模型权重文件的路径
def quick_baseline_weights_path(property_name: str = "logp") -> str:
    return os.path.join(SAVE_DIR, f"baseline_{property_name}_v1.pkl")

# 训练演示用的基线模型并保存
def train_baseline_demo(property_name: str = "logp") -> str:
    data_file = os.path.join(ROOT, "data", PROPERTIES.get(property_name, PROPERTIES["logp"])["data_file"])
    df = ensure_demo_dataset(data_file)
    # 数据预处理：处理缺失值和异常值
    df = preprocess_data(df, "target", remove_outliers=True, outlier_method="iqr")
    # 使用scaffold_split进行数据划分（与评估时一致，避免数据泄露）
    train_df, val_df, _ = scaffold_split(df, frac=(0.7, 0.15, 0.15), seed=42)
    # 只使用训练集进行训练
    bm = BaselineModel(n_models=3, nbits=1024).fit(train_df, "target")
    out = quick_baseline_weights_path(property_name)
    bm.save(out)
    return out

# 训练所有性质的模型
def train_all_properties(model_type: str = "baseline"):
    """训练所有性质的模型"""
    if model_type != "baseline":
        # 如果是GNN，需要从train_gnn导入
        try:
            from train_gnn import train_all_properties as train_gnn_all
            return train_gnn_all("gnn")
        except RuntimeError:
            print("GNN dependencies not available, falling back to baseline")
            model_type = "baseline"
    
    results = {}
    for prop_key in PROPERTIES.keys():
        try:
            path = train_baseline_demo(prop_key)
            results[prop_key] = path
            print(f"✓ Trained {prop_key} model: {path}")
        except Exception as e:
            print(f"✗ Failed to train {prop_key}: {e}")
            results[prop_key] = None
    return results

# 训练基线模型的主函数，处理命令行参数
def train_baseline_main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "logp.csv"))
    p.add_argument("--target", default="target")
    p.add_argument("--out", default=quick_baseline_weights_path("logp"))
    a = p.parse_args(args)
    df = ensure_demo_dataset(a.data) if not os.path.exists(a.data) else pd.read_csv(a.data)
    # 数据预处理：处理缺失值和异常值
    df = preprocess_data(df, a.target, remove_outliers=True, outlier_method="iqr")
    # 使用scaffold_split进行数据划分（与评估时一致，避免数据泄露）
    train_df, val_df, _ = scaffold_split(df, frac=(0.7, 0.15, 0.15), seed=42)
    print(f"数据预处理后剩余样本数: {len(df)}, 训练集: {len(train_df)}")
    bm = BaselineModel(n_models=3).fit(train_df, a.target)
    bm.save(a.out)
    print(f"Saved baseline to {a.out}")

if __name__ == "__main__":
    print("开始训练所有性质的基线模型...")
    print("=" * 50)
    results = train_all_properties("baseline")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"成功训练: {success_count}/{len(results)} 个模型")
    
    if any(v is None for v in results.values()):
        print("\n失败的模型:")
        for prop, path in results.items():
            if path is None:
                print(f"  - {prop}")
