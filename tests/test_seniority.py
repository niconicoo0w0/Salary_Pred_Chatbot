# tests/test_seniority.py
import sys
from pathlib import Path
import pytest
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.featurizers import extract_seniority, SeniorityAdder

@pytest.mark.parametrize("title,expected", [
    ("Software Engineer Intern", "intern"),
    ("Data Science Internship", "intern"),
    ("ML Co-Op", "intern"),
    ("Junior Software Engineer", "junior"),
    ("Entry Level Data Analyst", "entry"),
    ("Associate Backend Engineer", "entry"),
    ("Software Engineer II", "senior"),
    ("SWE III", "senior"),
    ("Software Engineer I", "mid"),
    ("Engineer 2", "senior"),
    ("Senior Machine Learning Engineer", "senior"),
    ("Staff ML Engineer", "staff"),
    ("Principal Scientist", "principal"),  # 实现返回 principal，而非 staff
    ("Lead Data Engineer", "lead"),
    ("Engineering Manager", "manager"),
    ("Director of Data", "director"),
    ("VP of Engineering", "vp"),
    ("Chief Data Officer", "cxo"),
    ("Software Engineer", "mid"),
    ("Developer", "mid"),
    ("", "mid"),
    (None, "mid"),
])
def test_extract_seniority(title, expected):
    assert extract_seniority(title) == expected

def test_seniority_adder_dataframe():
    df = pd.DataFrame({"Job Title": ["Junior Dev", "Staff SWE", "VP of Eng", None]})
    adder = SeniorityAdder(title_col="Job Title")
    out = adder.fit_transform(df)
    assert "seniority" in out.columns
    assert out["seniority"].tolist() == ["junior", "staff", "vp", "mid"]

def test_seniority_adder_fit_self():
    adder = SeniorityAdder()
    assert adder.fit(None) is adder
