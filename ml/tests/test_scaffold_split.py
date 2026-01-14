import pandas as pd

from utils import murcko_scaffold, scaffold_split


def test_scaffold_split_no_leakage():
    # 构造一个小数据集：包含多个重复 scaffold 的分子
    # benzene scaffold
    benzene = ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Clc1ccccc1"]
    # cyclohexane scaffold
    cyclo = ["C1CCCCC1", "CC1CCCCC1", "OC1CCCCC1"]
    # ethanol-like (empty scaffold is possible for small molecules)
    small = ["CCO", "CCCO", "CC(C)O", "CCN"]

    smiles = benzene + cyclo + small
    df = pd.DataFrame({"smiles": smiles, "target": list(range(len(smiles)))})

    tr, va, te = scaffold_split(df, frac=(0.6, 0.2, 0.2), seed=42)

    tr_scaff = set(murcko_scaffold(s) for s in tr["smiles"].tolist())
    va_scaff = set(murcko_scaffold(s) for s in va["smiles"].tolist())
    te_scaff = set(murcko_scaffold(s) for s in te["smiles"].tolist())

    # 关键要求：scaffold 不得跨集合（允许 scaffold 为空串 ""，也必须整体落在一个集合里）
    assert tr_scaff.isdisjoint(va_scaff)
    assert tr_scaff.isdisjoint(te_scaff)
    assert va_scaff.isdisjoint(te_scaff)

