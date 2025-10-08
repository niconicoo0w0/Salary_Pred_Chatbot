# train_pipeline.py
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from featurizers import SeniorityAdder, LocationTierAdder
import jd_parser

# ----------------- Helpers from your original code -----------------
def parse_salary_estimate(s):
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    s = str(s)
    if "per hour" in s.lower() or " /hr" in s.lower():
        return (np.nan, np.nan, np.nan)
    s = s.replace("(Glassdoor est.)","").replace("(Employer est.)","")
    s = s.replace("$","").replace(",","").strip()
    m = re.search(r'(\d+)\s*[kK]\s*-\s*(\d+)\s*[kK]', s)
    if not m:
        return (np.nan, np.nan, np.nan)
    lo, hi = int(m.group(1))*1000, int(m.group(2))*1000
    return (lo, hi, (lo+hi)/2)

# ----------------- Train script -----------------
def main(csv_path: str, out_path: str = "models/pipeline.pkl"):
    df = pd.read_csv(csv_path)

    if "avg_salary" not in df.columns:
        df[["min_salary","max_salary","avg_salary"]] = df["Salary Estimate"].apply(
            lambda x: pd.Series(parse_salary_estimate(x))
        )
    df = df.dropna(subset=["avg_salary"]).copy()

    # Ensure numeric columns exist
    numeric = ["Rating", "age", "min_size", "max_size"]
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan

    categorical_base = ["Sector", "Type of ownership", "Size"]
    raw_cols = numeric + categorical_base + ["Job Title", "Location"]
    df = df.dropna(subset=["Job Title", "Location"])  # keep minimal integrity

    X = df[raw_cols].copy()
    y = df["avg_salary"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess after feature adders
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Columns AFTER adders:
    # numeric -> same
    # categorical -> categorical_base + 'seniority' + 'loc_tier'
    preprocess = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical_base + ["seniority", "loc_tier"]),
    ])

    gbr = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42
    )

    full_pipe = Pipeline([
        ("seniority", SeniorityAdder(title_col="Job Title")),
        ("loc_tier", LocationTierAdder(loc_col="Location", drop_location=True)),
        ("prep", preprocess),
        ("reg", gbr),
    ])

    full_pipe.fit(X_train, y_train)
    y_pred = full_pipe.predict(X_test)

    print("Train R²:", full_pipe.score(X_train, y_train))
    print("Test  R²:",  full_pipe.score(X_test, y_test))
    print("MAE:",       mean_absolute_error(y_test, y_pred))
    print("RMSE:",      np.sqrt(mean_squared_error(y_test, y_pred)))

    # Save ONE artifact with everything inside
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(full_pipe, out_path)
    print(f"Saved pipeline to {out_path}")

if __name__ == "__main__":
    # Example: python train_pipeline.py data/salary.csv
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_pipeline.py <csv_path>")
        raise SystemExit(2)
    main(sys.argv[1])
