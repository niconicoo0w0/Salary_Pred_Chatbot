# jd_parsing.py - Job description parsing constants and patterns
import re
from typing import List

# Role keywords for job title detection
ROLE_KEYWORDS = [
    "artificial intelligence engineer", "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "senior data scientist", "data engineer", "software engineer",
    "mlops engineer", "research scientist", "nlp engineer", "computer vision engineer",
    "generative ai engineer", "deep learning engineer"
]

# Regex patterns for extracting job titles
TITLE_REGEXES = [
    r"(?im)^\s*job\s*title\s*[:\-]\s*(.+)$",
    r"(?im)^\s*title\s*[:\-]\s*(.+)$",
    r"(?im)^\s*(?:position|role)\s*[:\-]\s*(.+)$",
]

# Noise line hints to filter out irrelevant content
NOISE_LINE_HINTS = [
    "logo", "share", "save", "easy apply", "promoted", "actively reviewing",
    "on-site", "onsite", "hybrid", "remote", "full-time", "part-time",
    "contract", "matches your job preferences", "message", "about the job",
    "rate-", "location-", "multiple locations", "apply", "premium", "job poster"
]

# City name validation regex
CITY_REGEX = re.compile(r"^[A-Za-z][A-Za-z\s\-\.\']{1,48}$")
