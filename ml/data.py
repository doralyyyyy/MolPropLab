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
from typing import Dict, Optional, List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

from utils import PROPERTIES, ROOT, sanitize_smiles

PUBCHEM_PUGREST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUGVIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
DATA_DIR = Path(ROOT) / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _requests_session() -> requests.Session:
    """
    PubChem 会有间歇性 429/5xx，使用 Session + Retry 提升稳定性。
    """
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "MolPropLab/1.0 (data curation)"})
    return s

_SESSION = _requests_session()

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
    """
    Resolve first CID for a SMILES via PUG REST.
    """
    url = f"{PUBCHEM_PUGREST}/compound/smiles/cids/JSON"
    try:
        r = _SESSION.post(url, data={"smiles": smiles}, timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        cids = (((j or {}).get("IdentifierList") or {}).get("CID") or [])
        return int(cids[0]) if cids else None
    except Exception:
        # 回退为txt解析（兼容极端情况）
        try:
            url_txt = f"{PUBCHEM_PUGREST}/compound/smiles/cids/txt"
            r2 = _SESSION.post(url_txt, data={"smiles": smiles}, timeout=30)
            if r2.status_code != 200:
                return None
            txt = r2.text.strip()
            m = re.search(r"(\d+)", txt)
            return int(m.group(1)) if m else None
        except Exception:
            return None


def pugview_heading_json(cid: int, heading: str):
    url = f"{PUBCHEM_PUGVIEW}/data/compound/{cid}/JSON"
    params = {"heading": heading}
    r = _SESSION.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()


_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_num_with_unit = re.compile(
    r"(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[A-Za-zµμ/%°][A-Za-z0-9µμ/%°\-\s]*?)\b"
)

def _walk_strings(j: object):
    if isinstance(j, dict):
        for v in j.values():
            yield from _walk_strings(v)
    elif isinstance(j, list):
        for it in j:
            yield from _walk_strings(it)
    elif isinstance(j, str):
        yield j

def _unit_normalize(u: str) -> str:
    u = (u or "").strip()
    u = u.replace("μ", "u").replace("µ", "u")
    u = u.replace("°", "deg")
    u = re.sub(r"\s+", " ", u)
    return u.lower()

def _convert_temperature(value: float, unit: str) -> Optional[float]:
    u = _unit_normalize(unit)
    if u in ("degc", "c", "deg c", "degree celsius", "degrees celsius"):
        return float(value)
    if u in ("degf", "f", "deg f", "degree fahrenheit", "degrees fahrenheit"):
        return (float(value) - 32.0) * 5.0 / 9.0
    if u in ("k", "kelvin"):
        return float(value) - 273.15
    return None

def _convert_pressure(value: float, unit: str) -> Optional[float]:
    u = _unit_normalize(unit)
    v = float(value)
    if u in ("pa",):
        return v
    if u in ("kpa",):
        return v * 1_000.0
    if u in ("mpa",):
        return v * 1_000_000.0
    if u in ("bar",):
        return v * 100_000.0
    if u in ("mbar",):
        return v * 100.0
    if u in ("atm",):
        return v * 101_325.0
    if u in ("mmhg", "mm hg", "torr"):
        return v * 133.322368
    if u in ("psi",):
        return v * 6894.757293168
    return None

def _convert_density(value: float, unit: str) -> Optional[float]:
    u = _unit_normalize(unit)
    v = float(value)
    # g/cm3 & g/mL 等价
    if u in ("g/cm3", "g/cm^3", "g cm-3", "g ml-1", "g/ml", "g/mL".lower()):
        return v
    if u in ("kg/m3", "kg/m^3", "kg m-3"):
        return v / 1000.0
    return None

def _convert_refractive_index(value: float, unit: str) -> Optional[float]:
    # 通常无量纲（或标注为 "RI" / "nD"）
    _ = unit
    return float(value)

def _convert_pka(value: float, unit: str) -> Optional[float]:
    # pKa 基本无量纲
    _ = unit
    return float(value)

def _extract_candidates_from_pugview(j: dict) -> List[Tuple[float, str, str]]:
    """
    Extract numeric candidates with light context from PUG-View JSON.
    Return list of (value, unit, raw_string).
    """
    out: List[Tuple[float, str, str]] = []
    for s in _walk_strings(j):
        # 优先匹配 "数值 + 单位" 的模式
        for m in _num_with_unit.finditer(s):
            try:
                v = float(m.group("val"))
            except Exception:
                continue
            unit = m.group("unit").strip()
            out.append((v, unit, s))
        # 退化：只有数字没有单位，也收集（用于 refractive index / pKa 等）
        if not _num_with_unit.search(s):
            m2 = _num.search(s)
            if m2:
                try:
                    v2 = float(m2.group(0))
                except Exception:
                    continue
                out.append((v2, "", s))
    return out

def _choose_value(property_key: str, candidates: List[Tuple[float, str, str]]) -> Optional[float]:
    """
    Choose a stable value from candidates:
    - 先做单位转换到项目内部标准单位
    - 多个候选取中位数（降低噪声），并剔除明显不合理值
    """
    converters = {
        "boiling_point": _convert_temperature,
        "melting_point": _convert_temperature,
        "flash_point": _convert_temperature,
        "vapor_pressure": _convert_pressure,
        "density": _convert_density,
        "refractive_index": _convert_refractive_index,
        "pka": _convert_pka,
    }
    conv = converters.get(property_key)
    if conv is None:
        return None

    vals: List[float] = []
    for v, unit, raw in candidates:
        # 过滤明显是“噪声数字”：比如 PubChem JSON 里出现年份、编号等
        # 这里只做非常轻的限定，避免过拟合到某些格式
        if property_key in ("boiling_point", "melting_point", "flash_point"):
            x = conv(v, unit)
            if x is None:
                continue
            # 常见有机物温度范围粗过滤
            if -200.0 <= x <= 1000.0:
                vals.append(x)
        elif property_key == "vapor_pressure":
            x = conv(v, unit)
            if x is None:
                continue
            if 0.0 <= x <= 1e9:
                vals.append(x)
        elif property_key == "density":
            x = conv(v, unit)
            if x is None:
                continue
            if 0.0 < x < 10.0:
                vals.append(x)
        elif property_key == "refractive_index":
            # 折射率典型范围
            if 1.0 <= v <= 3.0:
                vals.append(float(v))
        elif property_key == "pka":
            # 常见 pKa 取值范围（宽松）
            if -5.0 <= v <= 25.0:
                vals.append(float(v))

    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)

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
    # 更稳健：解析“数值+单位”，转换为项目内部统一单位，并在多候选时取中位数
    cands = _extract_candidates_from_pugview(j or {})
    chosen = _choose_value(target, cands)
    if chosen is not None:
        return float(chosen)
    # 回退：保底逻辑（旧实现），确保尽量不返回空
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
