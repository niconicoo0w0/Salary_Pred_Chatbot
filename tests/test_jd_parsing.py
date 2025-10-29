# tests/test_jd_parsing.py
import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.jd_parsing import parse_years_experience, parse_location, parse_salary_range, parse_education, parse_jd

@pytest.mark.parametrize("jd,years", [
    ("3+ years experience in Python", 3),
    ("At least 5 years of relevant experience", 5),
    ("1-3 years exp.", 3),       # Expected: take the upper bound of the range
    ("0-1 year", 1),             # Expected: same logic as above
    ("no experience required", None),
    ("Minimum two years experience", None),   # English word numbers are not parsed
    ("Seven (7) years of exp", None),        # The "(7)" part is currently ignored by parsing logic
])
def test_parse_years_experience(jd, years):
    assert parse_years_experience(jd) == years

@pytest.mark.parametrize("jd,loc", [
    ("Location: San Francisco, CA", "San Francisco, CA"),
    ("Based in New York, NY", "New York, NY"),
    ("Remote within US (Austin, TX preferred)", "Austin, TX"),
    ("Hybrid - Seattle, WA", "Seattle, WA"),
    ("Remote only", None),
])
def test_parse_location(jd, loc):
    assert parse_location(jd) == loc

@pytest.mark.parametrize("jd,expected", [
    ("Compensation: $150,000 - $190,000 + bonus", (150000, 190000)),
    ("Salary range: 120k–160k", (120000, 160000)),
    ("$90k-$110k base", (90000, 110000)),
    ("Pay: $75,000", None),  # A single salary number should return None
    ("unpaid internship", None),
])
def test_parse_salary_range(jd, expected):
    assert parse_salary_range(jd) == expected

@pytest.mark.parametrize("text,expected", [
    ("Bachelor's degree required", "Bachelor"),
    ("Master's in Computer Science", "Master"),
    ("PhD preferred", "PhD"),
    ("MBA or equivalent", "MBA"),
    ("Associate degree", "Associate"),
    ("High School diploma", "High School"),
    ("Doctorate in Physics", "Doctorate"),
    ("No degree required", None),
])
def test_parse_education(text, expected):
    assert parse_education(text) == expected

def test_parse_jd():
    """Test parsing a complete job description block"""
    jd = """
    Senior Machine Learning Engineer
    Location: San Francisco, CA
    3+ years experience required
    Bachelor's degree in Computer Science
    Salary: $150,000 - $190,000
    """
    result = parse_jd(jd)
    assert result["years_experience"] == 3
    assert result["education_level"] == "Bachelor"
    assert result["location"] == "San Francisco, CA"
    assert result["salary_range"] == (150000, 190000)
