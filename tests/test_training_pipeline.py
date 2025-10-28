# tests/test_training_pipeline.py
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models.training_script.train_pipeline import (
    parse_salary_estimate,
    parse_size_to_min_max,
    compute_company_age,
    make_preprocessor,
    build_features,
)
from utils.constants import NUMERIC, CATEGORICAL_BASE, RAW_INPUTS

def _toy_training_df():
    # 为了兼容当前 make_preprocessor 的实现（直接引用 'seniority','loc_tier'），
    # 这里显式提供这两列；实际训练时由 featurizers 生成。
    data = {
        "Rating": [4.2, np.nan, 3.9, 4.8],
        "age": [25, 10, np.nan, 50],
        "Sector": ["Information Technology", "Finance", "UnknownSector", "Information Technology"],
        "Type of ownership": ["Company - Public", "Private", "Company - Private", "Company - Public"],
        "size_band": ["Mid", "Large", "XL", "Mid"],
        "Job Title": ["Software Engineer II", "Junior Data Scientist", "VP of Engineering", "Staff ML Engineer"],
        "Location": ["San Jose, CA", "Austin, TX", "Remote", "Seattle, WA"],
        "seniority": ["senior", "junior", "vp", "staff"],
        "loc_tier": ["high", "mid", "very_high", "low"],
    }
    return pd.DataFrame(data)

def test_contract_and_transform_matrix():
    df = _toy_training_df()
    prep = make_preprocessor()
    X = prep.fit_transform(df)

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(df)
    assert not np.isnan(X).any()

    # 校验 ColumnTransformer 契约
    cols_num = next(cols for name, trans, cols in prep.transformers_ if name == "num")
    cols_cat = next(cols for name, trans, cols in prep.transformers_ if name == "cat")
    assert cols_num == ["Rating", "age"]
    assert cols_cat == ["Sector", "Type of ownership", "size_band", "seniority", "loc_tier"]

def test_helper_parsers_basic():
    # parse_salary_estimate 返回 (min, max, avg)
    r = parse_salary_estimate("$120K-$160K")
    assert r[:2] == (120000, 160000)
    r = parse_salary_estimate("$95,000 - $105,000")
    assert r[:2] == (95000, 105000)
    assert parse_salary_estimate("unpaid") is None

    # parse_size_to_min_max
    assert parse_size_to_min_max("51 to 200 employees") == (51, 200)
    min_, max_ = parse_size_to_min_max("10000+ employees")
    assert int(min_) == 10000 and int(max_) == 10000

    # compute_company_age
    assert compute_company_age(2000) >= 20
    assert compute_company_age(None) is np.nan

def test_build_features_derives_size_band_from_size():
    df = pd.DataFrame({
        "Size": [
            "1 to 49 employees",           # <50 -> Small
            "51 to 199 employees",         # <200 -> Mid
            "201 to 999 employees",        # <1000 -> Large
            "1,000 to 9,999 employees",    # <10_000 -> XL
            "10000+ employees",            # >=10_000 -> Enterprise
            None, "unknown",
        ],
        "Founded": [2000]*7,
        "Rating": [4.0]*7,
    })
    out = build_features(df)

    expected = ["Small", "Mid", "Large", "XL", "Enterprise", np.nan, np.nan]
    got = out["size_band"].tolist()
    assert got[:5] == expected[:5]
    assert pd.isna(got[5]) and pd.isna(got[6])

    # 数值列存在且为数值 dtype
    for col in ["Rating", "age", "min_size", "max_size"]:
        assert col in out.columns
        assert pd.api.types.is_numeric_dtype(out[col])
    assert (out["age"] >= 0).all()
