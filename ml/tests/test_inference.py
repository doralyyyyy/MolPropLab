import json
from core import predict

def test_single_smiles_baseline():
    out = predict("CCO", model="baseline")
    # 检查返回的结构包含 properties、atom_importances 等字段
    assert "properties" in out and "atom_importances" in out
    assert isinstance(out["atom_importances"], list)
    # 检查 properties 中包含各性质的预测结果
    assert isinstance(out["properties"], dict)
    for prop_key, prop_data in out["properties"].items():
        assert "prediction" in prop_data and "uncertainty" in prop_data

def test_single_smiles_gnn_fallback_ok():
    # GNN may fallback to baseline if deps missing
    out = predict("CCO", model="gnn")
    assert "properties" in out and "atom_importances" in out
