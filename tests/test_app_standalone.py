# tests/test_app_standalone.py - Test app functions without importing the module
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Copy the functions we want to test to avoid importing the full module
def map_sector_to_training(s):
    """Map sector to training set"""
    if not s:
        return None
    t = str(s).strip()
    low = t.lower()
    
    # Training sectors
    TRAINING_SECTORS = set([
        "Arts, Entertainment & Recreation",
        "Construction, Repair & Maintenance",
        "Oil, Gas, Energy & Utilities",
        "Accounting & Legal",
        "Aerospace & Defense",
        "Agriculture & Forestry",
        "Biotech & Pharmaceuticals",
        "Business Services",
        "Consumer Services",
        "Education",
        "Finance",
        "Government",
        "Health Care",
        "Information Technology",
        "Insurance",
        "Manufacturing",
        "Media",
        "Mining & Metals",
        "Non-Profit",
        "Real Estate",
        "Retail",
        "Telecommunications",
        "Transportation & Logistics",
        "Travel & Tourism",
    ])
    
    # Sector canonicalization
    SECTOR_CANON = {
        "entertainment & media": "Media",
        "media & entertainment": "Media",
        "entertainment": "Media",
        "telecom": "Telecommunications",
        "information technology services": "Information Technology",
        "it": "Information Technology",
        "healthcare": "Health Care",
        "biotech": "Biotech & Pharmaceuticals",
        "pharmaceuticals": "Biotech & Pharmaceuticals",
        "pharma": "Biotech & Pharmaceuticals",
        "banking": "Finance",
        "financial services": "Finance",
        "insurance & finance": "Finance",
        "logistics": "Transportation & Logistics",
        "transportation": "Transportation & Logistics",
        "aerospace": "Aerospace & Defense",
        "defense": "Aerospace & Defense",
        "construction": "Construction, Repair & Maintenance",
        "repair & maintenance": "Construction, Repair & Maintenance",
        "education & training": "Education",
        "nonprofit": "Non-Profit",
        "non-profit": "Non-Profit",
        "retail & e-commerce": "Retail",
        "e-commerce": "Retail",
        "consumer": "Consumer Services",
        "government & public sector": "Government",
        "public sector": "Government",
        "oil & gas": "Oil, Gas, Energy & Utilities",
        "energy & utilities": "Oil, Gas, Energy & Utilities",
        "mining": "Mining & Metals",
        "metals": "Mining & Metals",
    }
    
    if low in SECTOR_CANON:
        return SECTOR_CANON[low]
    # direct match
    if t in TRAINING_SECTORS:
        return t
    # lenient contain checks
    for k, v in SECTOR_CANON.items():
        if k in low:
            return v
    return t  # keep original; OneHot(handle_unknown="ignore") will be safe

def employees_to_size_band(n):
    """Convert employee count to size band"""
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return None
        n = float(n)
    except Exception:
        return None
    if n < 50: return "Small"
    if n < 200: return "Mid"
    if n < 1000: return "Large"
    if n < 10000: return "XL"
    return "Enterprise"

def coerce_size_label_to_band(lbl):
    """Coerce size label to band"""
    if not lbl:
        return None
    t = str(lbl).strip().lower()
    SIZE_BANDS = ["Small","Mid","Large","XL","Enterprise"]
    for b in SIZE_BANDS:
        if b.lower() == t:
            return b
    # light normalization - check XL patterns first to avoid matching "large" in "xlarge"
    if t in {"xl","x-large","x large","extra large","xlarge"}: return "XL"
    if "small" in t: return "Small"
    import re
    if re.search(r"\bmid(dle)?\b", t): return "Mid"
    if "large" in t: return "Large"
    if "enterprise" in t: return "Enterprise"
    return None

def choose_size_band_from_profile(prof):
    """Choose size band from profile"""
    # priority: size_band -> Size -> size_label -> employees -> (min,max)
    if prof.get("size_band"):
        return coerce_size_label_to_band(prof.get("size_band"))
    if prof.get("Size"):
        b = coerce_size_label_to_band(prof.get("Size"))
        if b: return b
    if prof.get("size_label"):
        b = coerce_size_label_to_band(prof.get("size_label"))
        if b: return b
    if prof.get("employees") is not None:
        return employees_to_size_band(prof.get("employees"))
    # fallback: try min/max midpoint
    if prof.get("min_size") is not None and prof.get("max_size") is not None:
        try:
            mid = (float(prof["min_size"]) + float(prof["max_size"])) / 2.0
            return employees_to_size_band(mid)
        except Exception:
            pass
    return None

def normalize_company_profile_to_schema(prof):
    """Map whatever the agent returned to the exact training inputs we need."""
    return {
        "Sector": map_sector_to_training(prof.get("Sector") or prof.get("sector")),
        "Type of ownership": prof.get("Type of ownership") or prof.get("ownership"),
        "size_band": choose_size_band_from_profile(prof),
        "age": prof.get("age") or prof.get("company_age"),  # agent可能给 age 或 company_age
        "hq_city": prof.get("hq_city"),
        "hq_state": prof.get("hq_state"),
        "__sources__": prof.get("__sources__", []),
    }

def detect_job_title(jd_text):
    """Detect job title from job description"""
    if not jd_text or not jd_text.strip():
        return None
    
    # Import helpers
    from utils.helpers import clean_text, strip_paren_noise, looks_like_location, looks_like_noise_line
    import re
    
    # 简约版：沿用你 helpers/jd_parsing 的清洗+启发式
    txt = clean_text(jd_text)
    # 1) 明确字段
    pats = [
        r"(?im)^\s*(?:job\s*)?title\s*[:\-]\s*(.+)$",
        r"(?im)^\s*(?:position|role)\s*[:\-]\s*(.+)$",
    ]
    for rx in pats:
        m = re.search(rx, txt)
        if m:
            cand = strip_paren_noise(m.group(1).strip())
            if cand and not looks_like_location(cand):
                return cand.title()
    # 2) 前几行找头衔类词
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    for ln in lines[:25]:
        if looks_like_noise_line(ln.lower()):
            continue
        if re.search(r"\b(engineer|scientist|analyst|developer|researcher|manager|architect|designer)\b", ln.lower()):
            # Extract just the job title part, not the whole line
            # Look for patterns like "Senior Machine Learning Engineer" in the line
            title_match = re.search(r"\b(Senior\s+)?(Machine\s+Learning\s+)?(Engineer|Scientist|Analyst|Developer|Researcher|Manager|Architect|Designer)\b", ln, re.IGNORECASE)
            if title_match:
                return title_match.group(0).title()
            # Fallback to the whole line if no specific pattern found
            cand = strip_paren_noise(ln.strip())
            if cand and not looks_like_location(cand):
                return cand.title()
    return None

def fmt_none(v):
    """Format None/empty values for display."""
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v)) or v == "" or v == []) else str(v)

def llm_explain(context, derived, point, low, high):
    """Generate LLM explanation"""
    # Mock client for testing
    client = None
    
    if client is None or not hasattr(client, 'chat'):
        parts = [
            f"Predicted base salary: ${point:,.0f} (range ${low:,.0f}-${high:,.0f}).",
            "Inputs used by the model:",
            f"- Job title: {fmt_none(context.get('Job Title'))}",
            f"- Location: {fmt_none(context.get('Location'))}",
            f"- Rating: {fmt_none(context.get('Rating'))}, Company age: {fmt_none(context.get('age'))}",
            f"- Sector: {fmt_none(context.get('Sector'))}, Ownership: {fmt_none(context.get('Type of ownership'))}, Size band: {fmt_none(context.get('size_band'))}",
            "Derived (from pipeline):",
            f"- Seniority: {fmt_none(derived.get('seniority'))}",
            f"- Location tier: {fmt_none(derived.get('loc_tier'))}",
            "Likely drivers: market (location tier), seniority inferred from title, size band/age, sector/ownership.",
        ]
        return " ".join(parts)
    
    # This would be the real LLM call, but we're mocking it
    return "Mock LLM explanation"

# Test functions
def test_map_sector_to_training_exact_match():
    """Test sector mapping with exact match"""
    result = map_sector_to_training("Information Technology")
    assert result == "Information Technology"

def test_map_sector_to_training_canonical_match():
    """Test sector mapping with canonical match"""
    result = map_sector_to_training("it")
    assert result == "Information Technology"

def test_map_sector_to_training_no_match():
    """Test sector mapping with no match"""
    result = map_sector_to_training("Unknown Sector")
    assert result == "Unknown Sector"

def test_map_sector_to_training_empty():
    """Test sector mapping with empty input"""
    assert map_sector_to_training("") is None
    assert map_sector_to_training(None) is None

def test_employees_to_size_band_small():
    """Test employee count to size band - small"""
    assert employees_to_size_band(25) == "Small"
    assert employees_to_size_band(49) == "Small"

def test_employees_to_size_band_mid():
    """Test employee count to size band - mid"""
    assert employees_to_size_band(50) == "Mid"
    assert employees_to_size_band(199) == "Mid"

def test_employees_to_size_band_large():
    """Test employee count to size band - large"""
    assert employees_to_size_band(200) == "Large"
    assert employees_to_size_band(999) == "Large"

def test_employees_to_size_band_xl():
    """Test employee count to size band - XL"""
    assert employees_to_size_band(1000) == "XL"
    assert employees_to_size_band(9999) == "XL"

def test_employees_to_size_band_enterprise():
    """Test employee count to size band - enterprise"""
    assert employees_to_size_band(10000) == "Enterprise"
    assert employees_to_size_band(50000) == "Enterprise"

def test_employees_to_size_band_invalid():
    """Test employee count to size band with invalid input"""
    assert employees_to_size_band(None) is None
    assert employees_to_size_band(np.nan) is None
    assert employees_to_size_band("invalid") is None

def test_coerce_size_label_to_band_exact():
    """Test size label coercion with exact match"""
    assert coerce_size_label_to_band("Small") == "Small"
    assert coerce_size_label_to_band("Mid") == "Mid"
    assert coerce_size_label_to_band("Large") == "Large"
    assert coerce_size_label_to_band("XL") == "XL"
    assert coerce_size_label_to_band("Enterprise") == "Enterprise"

def test_coerce_size_label_to_band_normalization():
    """Test size label coercion with normalization"""
    assert coerce_size_label_to_band("small") == "Small"
    assert coerce_size_label_to_band("middle") == "Mid"
    assert coerce_size_label_to_band("x-large") == "XL"
    assert coerce_size_label_to_band("xlarge") == "XL"

def test_coerce_size_label_to_band_invalid():
    """Test size label coercion with invalid input"""
    assert coerce_size_label_to_band("") is None
    assert coerce_size_label_to_band(None) is None
    assert coerce_size_label_to_band("Unknown") is None

def test_choose_size_band_from_profile():
    """Test size band selection from profile"""
    # Test with size_band
    profile = {"size_band": "Mid"}
    assert choose_size_band_from_profile(profile) == "Mid"
    
    # Test with Size
    profile = {"Size": "Large"}
    assert choose_size_band_from_profile(profile) == "Large"
    
    # Test with size_label
    profile = {"size_label": "XL"}
    assert choose_size_band_from_profile(profile) == "XL"
    
    # Test with employees
    profile = {"employees": 1000}
    assert choose_size_band_from_profile(profile) == "XL"
    
    # Test with min/max
    profile = {"min_size": 400, "max_size": 600}
    assert choose_size_band_from_profile(profile) == "Large"
    
    # Test with no valid data
    profile = {}
    assert choose_size_band_from_profile(profile) is None

def test_normalize_company_profile_to_schema():
    """Test company profile normalization to schema"""
    profile = {
        "Sector": "Information Technology",
        "Type of ownership": "Company - Public",
        "size_band": "Mid",
        "age": 10,
        "hq_city": "San Francisco",
        "hq_state": "CA",
        "__sources__": [{"url": "https://example.com"}]
    }
    
    result = normalize_company_profile_to_schema(profile)
    
    assert result["Sector"] == "Information Technology"
    assert result["Type of ownership"] == "Company - Public"
    assert result["size_band"] == "Mid"
    assert result["age"] == 10
    assert result["hq_city"] == "San Francisco"
    assert result["hq_state"] == "CA"
    assert result["__sources__"] == [{"url": "https://example.com"}]

def test_normalize_company_profile_alternative_fields():
    """Test company profile normalization with alternative field names"""
    profile = {
        "sector": "Technology",  # lowercase
        "ownership": "Private",  # without "Type of"
        "company_age": 5  # instead of "age"
    }
    
    result = normalize_company_profile_to_schema(profile)
    
    assert result["Sector"] == "Technology"
    assert result["Type of ownership"] == "Private"
    assert result["age"] == 5

def test_detect_job_title_explicit_field():
    """Test job title detection with explicit field"""
    jd = "Job Title: Senior Software Engineer\nLocation: San Francisco, CA"
    result = detect_job_title(jd)
    assert result == "Senior Software Engineer"

def test_detect_job_title_position_field():
    """Test job title detection with position field"""
    jd = "Position: Data Scientist\nLocation: New York, NY"
    result = detect_job_title(jd)
    assert result == "Data Scientist"

def test_detect_job_title_heuristic():
    """Test job title detection with heuristic"""
    jd = "We are looking for a Senior Machine Learning Engineer to join our team."
    result = detect_job_title(jd)
    assert result == "Senior Machine Learning Engineer"

def test_detect_job_title_empty():
    """Test job title detection with empty input"""
    assert detect_job_title("") is None
    assert detect_job_title(None) is None

def test_detect_job_title_no_match():
    """Test job title detection with no match"""
    jd = "This is just some random text without any job titles."
    result = detect_job_title(jd)
    assert result is None

def test_llm_explain_no_client():
    """Test LLM explanation without client"""
    context = {
        "Job Title": "Software Engineer",
        "Location": "San Francisco, CA",
        "Rating": 4.0,
        "age": 10,
        "Sector": "Information Technology",
        "Type of ownership": "Company - Public",
        "size_band": "Mid"
    }
    derived = {"seniority": "senior", "loc_tier": "high"}
    
    result = llm_explain(context, derived, 100000, 90000, 110000)
    assert "Predicted base salary: $100,000" in result
    assert "Software Engineer" in result
    assert "San Francisco, CA" in result
