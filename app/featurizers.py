# script/featurizers.py
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def extract_seniority(title: str) -> str:
    t = str(title).lower()
    if any(k in t for k in ["intern", "internship", "co-op", "co op"]): return "intern"
    if re.search(r"\b(junior|jr\.?)\b", t): return "junior"
    if "entry level" in t or "entry-level" in t: return "entry"
    if re.search(r"\bassociate\b", t): return "entry"
    if re.search(r"\b(ii|iii|iv|v)\b", t): return "senior"
    if re.search(r"\b(i)\b", t): return "mid"
    mnum = re.search(r"\b(\d)\b", t)
    if mnum:
        return "senior" if int(mnum.group(1)) >= 2 else "mid"
    if re.search(r"\b(sr\.?|senior)\b", t): return "senior"
    if re.search(r"\b(staff)\b", t): return "staff"
    if re.search(r"\b(principal)\b", t): return "principal"
    if re.search(r"\b(lead|tech lead)\b", t): return "lead"
    if re.search(r"\b(head)\b", t): return "manager"
    if re.search(r"\b(manager|mgr\.?)\b", t): return "manager"
    if re.search(r"\b(director|dir\.?)\b", t): return "director"
    if re.search(r"\b(vice president|vp|svp|avp)\b", t): return "vp"
    if re.search(r"\b(ceo|cto|cfo|cpo|cio|ciso|chief)\b", t): return "cxo"
    return "mid"

class SeniorityAdder(BaseEstimator, TransformerMixin):
    def __init__(self, title_col: str = "Job Title"):
        self.title_col = title_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["seniority"] = X[self.title_col].apply(extract_seniority)
        return X

class LocationTierAdder(BaseEstimator, TransformerMixin):
    def __init__(self, loc_col: str = "Location", drop_location: bool = True):
        self.loc_col = loc_col
        self.drop_location = drop_location
        self.tier_map_ = {}
        self.quantiles_ = None
    def fit(self, X, y):
        tmp = pd.DataFrame({"loc": X[self.loc_col].astype(str), "y": np.ravel(y)})
        loc_median = tmp.groupby("loc")["y"].median()
        q25, q50, q75 = loc_median.quantile([0.25, 0.50, 0.75])
        def tier(v):
            if v <= q25: return "low"
            elif v <= q50: return "mid"
            elif v <= q75: return "high"
            else: return "very_high"
        self.tier_map_ = loc_median.apply(tier).to_dict()
        self.quantiles_ = (float(q25), float(q50), float(q75))
        return self
    def transform(self, X):
        X = X.copy()
        X["loc_tier"] = X[self.loc_col].astype(str).map(self.tier_map_).fillna("mid")
        if self.drop_location:
            X = X.drop(columns=[self.loc_col])
        return X
