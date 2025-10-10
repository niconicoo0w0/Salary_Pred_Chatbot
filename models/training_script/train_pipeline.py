# train_pipeline.py
import re
import json
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from utils.featurizers import SeniorityAdder, LocationTierAdder
from typing import Optional, Union

# ----------------- Salary parsing (unchanged logic) -----------------
def parse_salary_estimate(text: Optional[str]) -> Optional[Tuple[int, int, float]]:
    """
    Parse a salary *range* from free-form text.
    Supports formats like:
      - "$120K-$160K", "90k–110k", "$95,000 - $105,000"
      - "10000~20000", "10,000 ~ 20,000", "10000to20000", "10000 to 20000"
      - "ranging from 10k to 20k", "range from 10k to 20k", "between 10,000 and 20,000"
    Notes:
      - Returns (lo, hi, avg). If no range is found, returns None.
      - Recognizes 'k' (thousand) and 'm' (million) suffixes, and comma separators.
      - Ignores currency signs and whitespace variations.
    """
    if not text:
        return None

    s = str(text).lower()
    # normalize dashes/whitespace
    s = s.replace("—", "-").replace("–", "-").replace("〜", "~").replace("～", "~")

    # --- helpers ---
    def _to_number(tok: str) -> Optional[float]:
        """Parse a token like '95,000', '95k', '1.2m' -> absolute number in dollars."""
        tok = tok.strip()
        if not tok:
            return None
        # strip leading currency symbols
        tok = re.sub(r"^[\$\£\€]", "", tok)
        # remove commas/spaces
        tok_clean = tok.replace(",", "").replace(" ", "")
        # detect suffix
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*([km])?$", tok_clean)
        if not m:
            return None
        val = float(m.group(1))
        suf = m.group(2)
        if suf == "k":
            val *= 1_000
        elif suf == "m":
            val *= 1_000_000
        return val

    def _as_tuple(a: float, b: float) -> Tuple[int, int, float]:
        lo, hi = (a, b) if a <= b else (b, a)
        lo_i, hi_i = int(round(lo)), int(round(hi))
        avg = (lo + hi) / 2.0
        return lo_i, hi_i, avg

    # --- patterns: try in order of most explicit to most general ---

    # 1) worded ranges: "ranging from X to Y", "range from X to Y", "from X to Y"
    worded_patterns = [
        r"ranging\s+from\s+([$\d][\d,\.]*\s*[km]?)\s+to\s+([$\d][\d,\.]*\s*[km]?)",
        r"range\s+from\s+([$\d][\d,\.]*\s*[km]?)\s+to\s+([$\d][\d,\.]*\s*[km]?)",
        r"from\s+([$\d][\d,\.]*\s*[km]?)\s+to\s+([$\d][\d,\.]*\s*[km]?)",
        r"between\s+([$\d][\d,\.]*\s*[km]?)\s+and\s+([$\d][\d,\.]*\s*[km]?)",
    ]
    for pat in worded_patterns:
        m = re.search(pat, s)
        if m:
            a, b = _to_number(m.group(1)), _to_number(m.group(2))
            if a is not None and b is not None:
                return _as_tuple(a, b)

    # 2) symbol joiners: "-", "~", "to" without words around them
    #    Examples: "$120k-$160k", "95,000 - 105,000", "10000~20000", "10000 to 20000"
    symbol_patterns = [
        r"([$\d][\d,\.]*\s*[km]?)\s*-\s*([$\d][\d,\.]*\s*[km]?)",
        r"([$\d][\d,\.]*\s*[km]?)\s*~\s*([$\d][\d,\.]*\s*[km]?)",
        r"([$\d][\d,\.]*\s*[km]?)\s+to\s+([$\d][\d,\.]*\s*[km]?)",
    ]
    for pat in symbol_patterns:
        m = re.search(pat, s)
        if m:
            a, b = _to_number(m.group(1)), _to_number(m.group(2))
            if a is not None and b is not None:
                return _as_tuple(a, b)

    # 3) super compact "10000to20000"（无空格）
    m = re.search(r"([$\d][\d,\.]*\s*[km]?)[ ]?to[ ]?([$\d][\d,\.]*\s*[km]?)", s.replace(" ", ""))
    if m:
        a, b = _to_number(m.group(1)), _to_number(m.group(2))
        if a is not None and b is not None:
            return _as_tuple(a, b)

    # 4) fallback: two consecutive money-ish tokens anywhere: e.g., "$95,000 $105,000"
    toks = re.findall(r"[$\d][\d,\.]*\s*[km]?", s)
    if len(toks) >= 2:
        a, b = _to_number(toks[0]), _to_number(toks[1])
        if a is not None and b is not None and a != b:
            return _as_tuple(a, b)

    # Single point like "$75,000" -> 按你现有测试规范返回 None
    return None

# ----------------- Robust Size / Founded feature engineering -----------------
def parse_size_to_min_max(size_str) -> Tuple[float, float]:
    """
    Parse Glassdoor-style 'Size' to numeric (min_size, max_size).
    Handles:
      - '501 to 1000 employees'
      - '10000+ employees'  -> (10000, 10000)  (policy)
      - 'Unknown' / '-' / NaN
    Returns (np.nan, np.nan) when not parseable.
    """
    if pd.isna(size_str):
        return (np.nan, np.nan)
    s = str(size_str).strip().lower()
    if not s or s in {"unknown", "-", "n/a", "na"}:
        return (np.nan, np.nan)

    s = s.replace(",", "")
    s = re.sub(r"\s+employees?$", "", s)

    m = re.match(r"^\s*(\d+)\s*to\s*(\d+)\s*$", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return (float(lo), float(hi))

    m = re.match(r"^\s*(\d+)\s*(\+|plus)\s*$", s)
    if m:
        lo = int(m.group(1))
        return (float(lo), float(lo))  # policy: (threshold, threshold)

    m = re.match(r"^\s*(\d+)\s*$", s)
    if m:
        n = int(m.group(1))
        return (float(n), float(n))

    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) == 2:
        return (float(min(nums)), float(max(nums)))
    if len(nums) == 1:
        return (float(nums[0]), float(nums[0]))
    return (np.nan, np.nan)


def compute_company_age(founded, ref_year: Optional[int] = None) -> float:
    if ref_year is None:
        ref_year = pd.Timestamp.now().year
    try:
        y = int(founded)
        if 1800 <= y <= ref_year:
            return float(ref_year - y)
    except Exception:
        pass
    return np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric engineered features expected by the model:
      - age from Founded
      - min_size, max_size from Size
    Ensures numeric dtype for ['Rating','age','min_size','max_size'].
    """
    def parse_size_to_band(size_str: Optional[str]) -> Union[str, float]:
        if pd.isna(size_str): 
            return np.nan
        s = str(size_str).strip().lower().replace(",", "")
        s = s.replace("employees", "").strip()
        if not s or s in {"unknown", "-", "n/a", "na"}:
            return np.nan

        # numeric extraction
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) == 2:
            n = (nums[0] + nums[1]) / 2
        elif len(nums) == 1:
            n = float(nums[0])
        else:
            return np.nan

        if n < 50: return "Small"
        if n < 200: return "Mid"
        if n < 1000: return "Large"
        if n < 10_000: return "XL"
        return "Enterprise"

    df = df.copy()

    # Salary target triplet if needed
    if "avg_salary" not in df.columns and "Salary Estimate" in df.columns:
        df[["min_salary", "max_salary", "avg_salary"]] = df["Salary Estimate"].apply(
            lambda x: pd.Series(parse_salary_estimate(x))
        )

    # Engineer size bands
    if "Size" in df.columns:
        mins, maxs = zip(*df["Size"].map(parse_size_to_min_max))
        df["min_size"] = np.array(mins, dtype="float")
        df["max_size"] = np.array(maxs, dtype="float")
        df["size_band"] = df["Size"].map(parse_size_to_band)
    else:
        df["min_size"] = np.nan
        df["max_size"] = np.nan
        df["size_band"] = np.nan

    # Engineer age
    if "Founded" in df.columns:
        df["age"] = df["Founded"].map(compute_company_age).astype("float")
    elif "age" in df.columns and "age" not in df.columns:
        # Back-compat if the CSV already had 'age'; normalize name
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = np.nan

    # Coerce numerics
    for c in ["Rating", "age", "min_size", "max_size"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    return df


# ----------------- Training / Pipeline -----------------
NUMERIC = ["Rating", "age"]
CATEGORICAL_BASE = ["Sector", "Type of ownership", "size_band"]
RAW_INPUTS = NUMERIC + CATEGORICAL_BASE + ["Job Title", "Location"]  # pass to featurizers


def make_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                CATEGORICAL_BASE + ["seniority", "loc_tier"]),
    ])
    return preprocess


def train(csv_path: str, out_path: str = "models/pipeline.pkl", schema_path: str = "models/schema.json",
          diagnostics: bool = True) -> None:
    df = pd.read_csv(csv_path)

    # Build engineered features first
    df = build_features(df)
    print(df.columns)
    

    # Require target
    if "avg_salary" not in df.columns:
        raise ValueError("avg_salary not found/could not be created; check 'Salary Estimate' parsing.")

    # Minimal integrity on raw fields used by featurizers
    df = df.dropna(subset=["Job Title", "Location"]).copy()

    X = df[RAW_INPUTS].copy()
    print(X.columns)
    y = df["avg_salary"].astype(float).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = make_preprocessor()
    reg = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    )

    pipe = Pipeline([
        ("seniority", SeniorityAdder(title_col="Job Title")),
        ("loc_tier", LocationTierAdder(loc_col="Location", drop_location=True)),
        ("prep", preprocess),
        ("reg", reg),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\n=== Metrics ===")
    print("Train R²:", pipe.score(X_train, y_train))
    print("Test  R²:", pipe.score(X_test, y_test))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    # Save model
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"\nSaved pipeline to {out_path}")

    # Save schema for inference-time assertion
    schema = {
        "raw_inputs": RAW_INPUTS,
        "numeric": NUMERIC,
        "categorical_base": CATEGORICAL_BASE,
        "added_by_featurizers": ["seniority", "loc_tier"],
    }
    Path(schema_path).parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Saved schema to {schema_path}")

    # Quick acceptance checks (no all-NaN numeric engineered columns in TRAIN split)
    X_train_num = X_train[NUMERIC]
    all_nan_cols = [c for c in NUMERIC if X_train_num[c].isna().all()]
    if all_nan_cols:
        raise AssertionError(f"All-NaN engineered columns before imputer: {all_nan_cols}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to glassdoor CSV")
    ap.add_argument("--out", default="models/pipeline.pkl", help="Output model path")
    ap.add_argument("--schema", default="models/schema.json", help="Output schema json")
    ap.add_argument("--no-diagnostics", action="store_true", help="Disable prints")
    args = ap.parse_args()
    train(args.csv_path, out_path=args.out, schema_path=args.schema, diagnostics=not args.no_diagnostics)

if __name__ == "__main__":
    main()
