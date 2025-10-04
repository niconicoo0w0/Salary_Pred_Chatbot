# script/jd_parser.py
import re
from typing import Dict, Optional, Tuple

def parse_years_experience(text: str) -> Optional[int]:
    # Matches "3 years", "5+ years", "7 yrs"
    m = re.search(r'(\d+)\s*\+?\s*(?:years?|yrs?)', text, flags=re.I)
    if m:
        return int(m.group(1))
    return None

def parse_education(text: str) -> Optional[str]:
    levels = ["Bachelor", "Master", "PhD", "Doctorate", "MBA", "Associate", "High School"]
    for lvl in levels:
        if re.search(lvl, text, re.I):
            return lvl
    return None

def parse_location(text: str) -> Optional[str]:
    # Very naive: looks for "City, ST" pattern
    m = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*,\s*[A-Z]{2})', text)
    if m:
        return m.group(1)
    return None

def parse_salary_range(text: str) -> Optional[Tuple[int,int]]:
    """
    Extract "$80,000 - $120,000" or "$50K–90K" style ranges.
    Returns (low, high) annual USD.
    """
    text = text.replace(",", "")
    # $80,000 - $120,000
    m = re.search(r'\$?(\d{2,6})(?:\s*-\s*|\s*–\s*)\$?(\d{2,6})', text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        # normalize: if small numbers, assume K
        if lo < 1000: lo *= 1000
        if hi < 1000: hi *= 1000
        return lo, hi
    # $50K-90K
    m = re.search(r'\$?(\d+)\s*[kK]\s*[-–]\s*\$?(\d+)\s*[kK]', text)
    if m:
        return int(m.group(1))*1000, int(m.group(2))*1000
    return None

def parse_jd(jd: str) -> Dict[str, Optional[str]]:
    return {
        "years_experience": parse_years_experience(jd),
        "education_level": parse_education(jd),
        "location": parse_location(jd),
        "salary_range": parse_salary_range(jd),
    }
