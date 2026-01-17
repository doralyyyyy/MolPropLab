"""
评估基线模型
计算多种评价指标：RMSE、MAE、R2、MAPE等
使用scaffold_split进行数据集划分，确保与训练时一致
"""

import os
import json
import math
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import ROOT, scaffold_split, preprocess_data
from train_baseline import BaselineModel, quick_baseline_weights_path

def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """
    计算多种评价指标
    
    参数:
        y_true: 真实值列表
        y_pred: 预测值列表
    
    返回:
        包含各种评价指标的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 移除NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return {"error": "No valid predictions"}
    
    # RMSE
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # R2
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    # 避免除零，只计算非零真实值
    non_zero_mask = np.abs(y_true) > 1e-8
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('nan')
    
    # 相关系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else float('nan')
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape) if not np.isnan(mape) else None,
        "correlation": float(correlation) if not np.isnan(correlation) else None,
        "n_samples": int(len(y_true))
    }

# 评估基线模型的主函数，计算多种评价指标
def eval_baseline_main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "logp.csv"))
    p.add_argument("--target", default="target")
    p.add_argument("--model", default=quick_baseline_weights_path("logp"))
    a = p.parse_args(args)
    df = pd.read_csv(a.data)
    # 数据预处理（与训练时一致）
    df = preprocess_data(df, a.target, remove_outliers=True, outlier_method="iqr")
    bm = BaselineModel.load(a.model)
    
    # 使用scaffold_split进行数据集划分（与训练时一致）
    # 注意：这里使用全部数据，因为评估时我们需要测试集
    train_df, val_df, test_df = scaffold_split(df, frac=(0.7, 0.15, 0.15), seed=42)
    
    # 在测试集上评估
    y_true, y_pred = [], []
    for smi, y in zip(test_df["smiles"], test_df[a.target]):
        y_true.append(float(y))
        y_pred.append(bm.predict(smi, return_shap=False)["prediction"])
    
    metrics = calculate_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    eval_baseline_main()
