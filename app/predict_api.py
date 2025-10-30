# app/predict_api.py
# -*- coding: utf-8 -*-
"""
Small, UI-free prediction API so both Gradio UI and chatbot can call
the same logic without importing each other (avoids circular import).
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from utils import helpers, jd_parsing, us_locations, config

# load constants from config
PIPELINE_PATH = config.PIPELINE_PATH
SCHEMA_PATH = config.SCHEMA_PATH

# fallback schema
_FALLBACK_SCHEMA = {
    "raw_inputs": ["Rating","age","Sector","Type of ownership","size_band","Job Title","Location"],
    "numeric": ["Rating","age"],
    "categorical_base": ["Sector","Type of ownership","size_band"],
    "added_by_featurizers": ["seniority","loc_tier"],
}

# load schema
try:
    with open(SCHEMA_PATH, "r") as f:
        _SCHEMA = json.load(f)
except Exception:
    _SCHEMA = _FALLBACK_SCHEMA

# force exact order
_EXPECTED_RAW = _FALLBACK_SCHEMA["raw_inputs"]
if _SCHEMA.get("raw_inputs") != _EXPECTED_RAW:
    _SCHEMA = _FALLBACK_SCHEMA

# load pipeline once
_pipe = joblib.load(PIPELINE_PATH)


def _derive_features_with_pipeline_steps(row: Dict[str, Any]) -> Dict[str, Any]:
    df0 = pd.DataFrame([row], columns=_SCHEMA["raw_inputs"])
    seniority_step = _pipe.named_steps.get("seniority", None)
    loc_tier_step = _pipe.named_steps.get("loc_tier", None)

    df1 = df0
    if seniority_step is not None:
        df1 = seniority_step.transform(df1)
    if loc_tier_step is not None:
        df1 = loc_tier_step.transform(df1)

    seniority = df1["seniority"].iloc[0] if "seniority" in df1.columns else None
    loc_tier  = df1["loc_tier"].iloc[0]  if "loc_tier"  in df1.columns else None
    return {"seniority": seniority, "loc_tier": loc_tier}


def map_sector_to_training(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = str(s).strip()
    low = t.lower()
    if low in jd_parsing._SECTOR_MAP:
        return jd_parsing._SECTOR_MAP[low]
    if t in jd_parsing._TRAIN_SECTORS:
        return t
    for k, v in jd_parsing._SECTOR_MAP.items():
        if k in low:
            return v
    return t


def make_model_row(
    job_title: str,
    location: str,
    rating: float,
    age: float,
    sector: str,
    type_of_ownership: str,
    size_band: str,
) -> Dict[str, Any]:
    def _nan_if_blank(s):
        s = (s or "").strip()
        return s if s else np.nan

    row_full = {
        "Job Title": (job_title or "").strip(),
        "Location": (location or "").strip(),
        "Rating": float(rating) if rating is not None else float("nan"),
        "age": float(age) if age not in (None, "") else float("nan"),
        "Sector": _nan_if_blank(map_sector_to_training((sector or "").strip())),
        "Type of ownership": _nan_if_blank(type_of_ownership),
        "size_band": _nan_if_blank(size_band),
    }
    cols = _SCHEMA["raw_inputs"]
    return {k: row_full.get(k) for k in cols}


def predict_salary(row: Dict[str, Any]) -> Tuple[float, float, float]:
    X = pd.DataFrame([row], columns=_SCHEMA["raw_inputs"])
    y = float(_pipe.predict(X)[0])
    return y, max(0.0, y * 0.9), y * 1.1


def run_prediction(
    *,
    job_title: str,
    city: str,
    state_abbrev: str,
    rating: float,
    age: float,
    sector: str,
    type_of_ownership: str,
    size_band: str,
    jd_text: str = "",
    company_name: str = "",
) -> Dict[str, Any]:
    location = f"{city}, {state_abbrev}"
    row = make_model_row(
        job_title=job_title,
        location=location,
        rating=rating,
        age=age,
        sector=sector,
        type_of_ownership=type_of_ownership,
        size_band=size_band,
    )
    point, low, high = predict_salary(row)
    derived = _derive_features_with_pipeline_steps(row)

    return {
        "Predicted Base Salary (USD)": f"${point:,.0f}",
        "Suggested Range (USD)": f"${low:,.0f} - ${high:,.0f}",
        "Derived features (from pipeline)": derived,
        "Inputs used by the model": row,
    }
