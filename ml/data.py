"""
Data curation utilities for populating the CSVs under ``ml/data``.

Features added for this project:
- 与 utils.PROPERTIES 对齐的属性名称
- 自动标准化/去重 SMILES（使用 utils.sanitize_smiles）
- 先用 RDKit 计算可离线的性质（MW、LogP、ESOL LogS），
  其他性质再调用 PubChem，并带有简单缓存与退避
- 兼容现有带表头的 CSV（columns: smiles,target）
- CLI：按属性刷新数据文件，支持继续、跳过已有值与节流
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

from utils import PROPERTIES, ROOT, sanitize_smiles

PUBCHEM_PUGREST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUGVIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
DATA_DIR = Path(ROOT) / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) RDKit: MW / LogP ---
def rdkit_mw_logp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)  # Wildman–Crippen LogP
    return mw, logp


# --- 2) ESOL: LogS (log10 mol/L) ---
# Coefficients from a common RDKit-based ESOL refit implementation
def esol_logS(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    aromatic_atoms = mol.GetSubstructMatches(Chem.MolFromSmarts("a"))
    ap = len(aromatic_atoms) / mol.GetNumAtoms() if mol.GetNumAtoms() else 0.0

    intercept = 0.26121066137801696
    coef = {
        "mw": -0.0066138847738667125,
        "logp": -0.7416739523408995,
        "rotors": 0.003451545565957996,
        "ap": -0.42624840441316975,
    }
    return (
        intercept
        + coef["logp"] * logp
        + coef["mw"] * mw
        + coef["rotors"] * rotors
        + coef["ap"] * ap
    )


# --- 3) PubChem helpers ---
def smiles_to_cid(smiles: str):
    url = f"{PUBCHEM_PUGREST}/compound/smiles/cids/txt"
    r = requests.post(url, data={"smiles": smiles}, timeout=30)
    if r.status_code != 200:
        return None
    txt = r.text.strip()
    m = re.search(r"(\d+)", txt)
    return int(m.group(1)) if m else None


def pugview_heading_json(cid: int, heading: str):
    url = f"{PUBCHEM_PUGVIEW}/data/compound/{cid}/JSON"
    params = {"heading": heading}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()


_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_first_number_from_pugview(j: dict):
    if not j:
        return None

    def walk(x):
        if isinstance(x, dict):
            for _, v in x.items():
                yield from walk(v)
        elif isinstance(x, list):
            for it in x:
                yield from walk(it)
        elif isinstance(x, str):
            yield x

    for s in walk(j):
        m = _num.search(s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                pass
    return None


# --- 4) heading map (aligned to utils.PROPERTIES keys) ---
PUGVIEW_HEADINGS: Dict[str, str] = {
    "boiling_point": "Boiling+Point",
    "melting_point": "Melting+Point",
    "refractive_index": "Refractive+Index",
    "vapor_pressure": "Vapor+Pressure",
    "density": "Density",
    "flash_point": "Flash+Point",
    "pka": "Dissociation+Constants",
}


def _load_cache(target: str) -> Dict[str, float]:
    path = CACHE_DIR / f"{target}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(target: str, cache: Dict[str, float]):
    path = CACHE_DIR / f"{target}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_property(smiles: str, target: str) -> Optional[float]:
    """Resolve a property value for one SMILES with fallbacks."""
    if target == "molecular_weight" or target == "mw":
        mw, _ = rdkit_mw_logp(smiles)
        return mw
    if target == "logp":
        _, logp = rdkit_mw_logp(smiles)
        return logp
    if target == "logs":
        return esol_logS(smiles)

    cid = smiles_to_cid(smiles)
    if cid is None or target not in PUGVIEW_HEADINGS:
        return None
    heading = PUGVIEW_HEADINGS[target]
    j = pugview_heading_json(cid, heading)
    return extract_first_number_from_pugview(j)


def fetch_property_with_cache(smiles: str, target: str, cache: Dict[str, float], sleep_s: float, retries: int = 2) -> Optional[float]:
    """Fetch a property with caching and simple exponential backoff."""
    if smiles in cache:
        return cache[smiles]

    backoff = sleep_s
    for _ in range(retries + 1):
        try:
            val = get_property(smiles, target)
            if val is not None:
                cache[smiles] = float(val)
            return val
        except Exception:
            val = None
        time.sleep(backoff)
        backoff *= 2
    return None


def load_smiles_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with columns [smiles,target], clean and deduplicate."""
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV {path} must have at least two columns: smiles,target")
    df = df.rename(columns={df.columns[0]: "smiles", df.columns[1]: "target"})
    cleaned = []
    for smi in df["smiles"].astype(str):
        canon, mol = sanitize_smiles(smi)
        if mol:
            cleaned.append(canon)
        else:
            cleaned.append(None)
    df["smiles"] = cleaned
    df = df.dropna(subset=["smiles"]).drop_duplicates("smiles")
    return df.reset_index(drop=True)


def refresh_csv(property_name: str, input_csv: Optional[Path] = None, output_csv: Optional[Path] = None,
                sleep_s: float = 0.25, retries: int = 2, recompute_all: bool = True, use_cache: bool = True):
    """
    Fill or refresh a property CSV.

    Args:
        property_name: key in utils.PROPERTIES (e.g., 'boiling_point')
        input_csv: source CSV path (defaults to data/<prop>.csv)
        output_csv: output path (defaults to overwrite input)
        sleep_s: base sleep to respect PubChem rate limit
        retries: retries with exponential backoff for PubChem calls
        recompute_all: if False, keep existing numeric targets
        use_cache: load/save cache to data/cache/<prop>.json
    """
    input_csv = input_csv or (DATA_DIR / PROPERTIES[property_name]["data_file"])
    output_csv = output_csv or input_csv

    df = load_smiles_csv(Path(input_csv))
    cache = _load_cache(property_name) if use_cache else {}

    values = []
    for smi, cur in zip(df["smiles"], df["target"]):
        if not recompute_all and pd.notna(cur):
            try:
                values.append(float(cur))
                continue
            except Exception:
                pass
        v = fetch_property_with_cache(smi, property_name, cache, sleep_s=sleep_s, retries=retries)
        values.append(v)

    df["target"] = values
    if use_cache:
        _save_cache(property_name, cache)
    df.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Refresh property CSVs with RDKit/PubChem.")
    parser.add_argument("-p", "--property", choices=list(PROPERTIES.keys()), required=True, help="Property key, e.g., boiling_point")
    parser.add_argument("--input", type=Path, help="Input CSV (defaults to data/<prop>.csv)")
    parser.add_argument("--output", type=Path, help="Output CSV (defaults to overwrite input)")
    parser.add_argument("--sleep", type=float, default=0.25, help="Base sleep seconds between PubChem calls")
    parser.add_argument("--retries", type=int, default=2, help="Retries for PubChem calls")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing numeric targets; only fill missing/invalid")
    parser.add_argument("--no-cache", action="store_true", help="Do not read/write cache files")
    args = parser.parse_args()

    refresh_csv(
        property_name=args.property,
        input_csv=args.input,
        output_csv=args.output,
        sleep_s=args.sleep,
        retries=args.retries,
        recompute_all=not args.keep_existing,
        use_cache=not args.no_cache,
    )
    print(f"[data] refreshed {args.property} -> {args.output or args.input or PROPERTIES[args.property]['data_file']}")


if __name__ == "__main__":
    main()
