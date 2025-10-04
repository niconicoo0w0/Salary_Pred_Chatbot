# helpers.py - Utility functions and helper methods
import re
import unicodedata
import numpy as np
from typing import Union, Optional, Tuple, Any, Dict

def titlecase(s: str) -> str:
    """Convert string to title case."""
    s = (s or "").strip()
    return re.sub(r"\S+", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), s)

def looks_like_location(text: str) -> bool:
    """Check if text looks like a location (city, state format)."""
    from .us_locations import US_STATES
    
    t = (text or "").strip().lower()
    if re.match(r"^[a-z\s]+,\s*[a-z]{2}$", t):  # "san jose, ca"
        return True
    if t.upper() in US_STATES:
        return True
    return False

def clean_text(s: str) -> str:
    """Clean and normalize text content."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\uFFFD", "")
    for junk in ["\u20ac\u20ac\u20ac", "\u00c2", "\u20ac\u20ac\u201c", "\u20ac\u20ac\u201d", "\u20ac\u20ac\u2019", "\u20ac\u20ac\u201c", "\u20ac\u20ac\u009d"]:
        s = s.replace(junk, " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_paren_noise(title: str) -> str:
    """Remove parenthetical noise from job titles."""
    t = re.sub(r"\s*\((?:no cpt|cpt|wfh|onsite|on-site|remote|hybrid|contract|full[- ]?time|part[- ]?time)\)\s*$",
               "", title, flags=re.I)
    t = re.sub(r"\s*\([^)]{0,40}\)\s*", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" --—")
    return t.strip()

def looks_like_noise_line(line_lower: str) -> bool:
    """Check if line looks like noise content."""
    from .jd_parsing import NOISE_LINE_HINTS
    return any(h in line_lower for h in NOISE_LINE_HINTS)

def candidate_title_from_line(line: str) -> bool:
    """Check if line could be a job title."""
    from .jd_parsing import ROLE_KEYWORDS
    
    t = line.strip()
    if not 3 <= len(t) <= 80:
        return False
    words = t.split()
    if not 2 <= len(words) <= 8:
        return False
    tl = t.lower()
    if any(k in tl for k in ROLE_KEYWORDS):
        return True
    if re.search(r"\b(engineer|scientist|analyst|developer|researcher)\b", tl):
        return True
    return False

def fmt_none(v):
    """Format None/empty values for display."""
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v)) or v == "" or v == []) else str(v)
