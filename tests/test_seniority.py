# tests/test_seniority.py
import pytest

# Import from your app module. Adjust the path/module name if needed.
from app import detect_seniority, MAP6

def norm6(bucket: str) -> str:
    """Map the detector's rich bucket to the 6 standard buckets."""
    return MAP6.get(bucket, bucket)

@pytest.mark.parametrize(
    "title,jd,expected",
    [
        # --- Intern / Entry ---
        ("Software Engineer Intern", "", "intern"),
        ("Data Analyst (Internship)", "", "intern"),
        ("Software Engineer (Entry-Level)", "", "entry"),
        ("Junior Data Engineer", "", "entry"),
        ("Associate Machine Learning Engineer", "", "entry"),
        ("New Grad Software Engineer", "", "entry"),

        # --- Numeric / Roman levels ---
        ("Software Engineer II", "", "entry"),        # II -> junior -> entry
        ("SDE III", "", "entry"),                     # III -> mid -> entry
        ("Backend Engineer L4", "", "senior"),        # L4 -> senior
        ("ML Engineer L5", "", "staff"),              # L5 -> staff
        ("Senior Software Engineer L6", "", "staff"), # L6 -> principal -> staff

        # --- Senior / Staff / Principal / Lead ---
        ("Senior Data Scientist", "", "senior"),
        ("Staff Software Engineer", "", "staff"),
        ("Principal Engineer, Platforms", "", "staff"),
        ("Lead Backend Engineer", "", "staff"),

        # --- Management / Leadership ---
        ("Engineering Manager", "", "manager"),
        ("Director of Data Science", "", "manager"),
        ("Head of Machine Learning", "", "manager"),  # head of -> director -> manager
        ("VP of Engineering", "", "vp"),
        ("Chief Technology Officer (CTO)", "", "vp"),  # cxo -> vp

        # --- From JD body (years of exp) ---
        ("Software Engineer", "We need 6+ years of experience building systems.", "senior"),
        ("Software Engineer", "At least 8 years experience in distributed systems.", "staff"),
        ("Software Engineer", "0-1 years of experience welcome.", "entry"),

        # --- Default when no cues (falls back to 'mid' -> entry) ---
        ("Software Engineer", "", "entry"),
    ]
)
def test_detect_seniority_map6(title, jd, expected):
    bucket, conf, reasons = detect_seniority(title, jd)
    got6 = norm6(bucket or "")
    assert got6 == expected, f"title='{title}', jd='{jd[:60]}...', raw='{bucket}', conf={conf}, reasons={reasons}"

def test_confidence_nonzero_when_signals_present():
    bucket, conf, _ = detect_seniority("Senior Staff Software Engineer L6", "")
    assert conf > 0.5
    assert norm6(bucket) == "staff"

def test_confidence_low_when_default_mid():
    bucket, conf, _ = detect_seniority("Software Engineer", "")
    # default path returns 'mid' with ~0.45 in current implementation
    assert norm6(bucket) == "entry"
    assert 0.3 <= conf <= 0.5

@pytest.mark.parametrize(
    "title,jd,prefer_lower",
    [
        # If both entry & senior appear, tie-break should bias to lower level
        ("Entry-Level Senior Software Engineer", "", "entry"),
        ("Junior Senior Data Analyst", "", "entry"),
    ]
)
def test_tiebreak_biases_to_lower_level(title, jd, prefer_lower):
    bucket, conf, reasons = detect_seniority(title, jd)
    assert norm6(bucket) == prefer_lower, f"reasons={reasons}"

@pytest.mark.parametrize(
    "title,jd,expected",
    [
        # Managerial tokens should override IC tokens if both appear
        ("Senior Engineering Manager", "", "manager"),
        ("Staff Director of Engineering", "", "manager"),
        ("Lead VP of Product Engineering", "", "vp"),
    ]
)
def test_managerial_override(title, jd, expected):
    bucket, conf, reasons = detect_seniority(title, jd)
    assert norm6(bucket) == expected, f"reasons={reasons}"
