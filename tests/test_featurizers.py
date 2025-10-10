# tests/test_featurizers.py
import sys
from pathlib import Path
import pytest
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.featurizers import LocationTierAdder

def _toy_df():
    return pd.DataFrame({
        "Location": ["High City", "High City", "Mid City", "Low City", "Unknown City"],
        "y":         [200000,    210000,      120000,    80000,      100000],
    })

def test_location_tier_adder_basic_boundaries():
    df = _toy_df()
    X, y = df[["Location"]], df["y"]

    # 不提供 target_col；fit 需要 y
    adder = LocationTierAdder(loc_col="Location", drop_location=False)
    adder = adder.fit(X, y)
    out = adder.transform(X)

    assert hasattr(adder, "tier_map_")
    assert hasattr(adder, "quantiles_")
    q25, q50, q75 = adder.quantiles_
    assert q25 <= q50 <= q75

    for k in ["High City", "Mid City", "Low City"]:
        assert k in adder.tier_map_

    assert "loc_tier" in out.columns
    assert set(out["loc_tier"].unique()) <= {"low", "mid", "high", "very_high"}

def test_location_tier_adder_unknown_defaults_mid_and_drop():
    df = _toy_df()
    X, y = df[["Location"]], df["y"]

    adder = LocationTierAdder(drop_location=True)
    out = adder.fit(X, y).transform(X)

    # Unknown City 回退至 mid（与你实现一致）
    assert out["loc_tier"].iloc[-1] == "mid"
    # drop_location=True 删掉 Location 列
    assert "Location" not in out.columns

def test_location_tier_adder_fit_self():
    df = _toy_df()
    X, y = df[["Location"]], df["y"]
    adder = LocationTierAdder()
    assert adder.fit(X, y) is adder
