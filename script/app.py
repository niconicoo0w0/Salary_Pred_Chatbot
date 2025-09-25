# app.py — Training features only + derived features; US-only location with dependent City dropdown
# JD textbox to autofill Title/Location; NEW: Company Lookup tab to prefill company info.
import os, re, unicodedata, datetime as dt
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import gradio as gr
import joblib

# Optional LLM
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception:
    client = None
    OPENAI_MODEL = None

# Needed for unpickling custom transformers used in the pipeline
import featurizers  # noqa: F401
import jd_parser    # we will use parse_jd(job_description_text)

PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "models/pipeline.pkl")
pipe = joblib.load(PIPELINE_PATH)

# === Training inputs (exactly as in training) ===
NUMERIC = ["Rating", "company_age", "min_size", "max_size"]
CATEGORICAL_BASE = ["Sector", "Type of ownership", "Size"]
RAW_INPUTS = NUMERIC + CATEGORICAL_BASE + ["Job Title", "Location"]

# -------------------------- States & cities --------------------------
US_STATES: List[str] = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI",
    "MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT",
    "VT","VA","WA","WV","WI","WY","DC"
]

STATE_TO_CITIES: Dict[str, List[str]] = {
    "AL": ["Birmingham", "Montgomery", "Mobile", "Huntsville", "Tuscaloosa"],
    "AK": ["Anchorage", "Fairbanks", "Juneau", "Sitka"],
    "AZ": ["Phoenix", "Tucson", "Mesa", "Chandler", "Scottsdale"],
    "AR": ["Little Rock", "Fayetteville", "Fort Smith", "Springdale"],
    "CA": ["Los Angeles", "San Diego", "San Jose", "San Francisco", "Sacramento", "Oakland", "Fresno", "Irvine", "Palo Alto", "Santa Clara"],
    "CO": ["Denver", "Colorado Springs", "Aurora", "Fort Collins", "Boulder"],
    "CT": ["Bridgeport", "New Haven", "Stamford", "Hartford", "Norwalk"],
    "DE": ["Wilmington", "Dover", "Newark"],
    "FL": ["Miami", "Tampa", "Orlando", "Jacksonville", "Tallahassee", "Fort Lauderdale"],
    "GA": ["Atlanta", "Savannah", "Augusta", "Athens", "Macon"],
    "HI": ["Honolulu", "Hilo", "Kailua", "Kaneohe"],
    "ID": ["Boise", "Idaho Falls", "Meridian", "Twin Falls"],
    "IL": ["Chicago", "Aurora", "Naperville", "Peoria", "Evanston"],
    "IN": ["Indianapolis", "Fort Wayne", "Evansville", "South Bend"],
    "IA": ["Des Moines", "Cedar Rapids", "Davenport", "Iowa City"],
    "KS": ["Wichita", "Overland Park", "Kansas City", "Topeka"],
    "KY": ["Louisville", "Lexington", "Bowling Green", "Covington"],
    "LA": ["New Orleans", "Baton Rouge", "Shreveport", "Lafayette"],
    "ME": ["Portland", "Bangor", "Augusta"],
    "MD": ["Baltimore", "Silver Spring", "Rockville", "Bethesda"],
    "MA": ["Boston", "Cambridge", "Worcester", "Springfield", "Somerville"],
    "MI": ["Detroit", "Grand Rapids", "Ann Arbor", "Lansing"],
    "MN": ["Minneapolis", "Saint Paul", "Rochester", "Duluth"],
    "MS": ["Jackson", "Gulfport", "Hattiesburg"],
    "MO": ["Kansas City", "Saint Louis", "Springfield", "Columbia"],
    "MT": ["Billings", "Missoula", "Bozeman", "Helena"],
    "NE": ["Omaha", "Lincoln", "Bellevue"],
    "NV": ["Las Vegas", "Reno", "Henderson", "Carson City"],
    "NH": ["Manchester", "Nashua", "Concord"],
    "NJ": ["Newark", "Jersey City", "Paterson", "Hoboken"],
    "NM": ["Albuquerque", "Santa Fe", "Las Cruces"],
    "NY": ["New York", "Buffalo", "Rochester", "Albany", "Syracuse", "White Plains"],
    "NC": ["Charlotte", "Raleigh", "Durham", "Greensboro"],
    "ND": ["Fargo", "Bismarck", "Grand Forks"],
    "OH": ["Columbus", "Cleveland", "Cincinnati", "Toledo", "Dayton"],
    "OK": ["Oklahoma City", "Tulsa", "Norman", "Broken Arrow"],
    "OR": ["Portland", "Eugene", "Salem", "Bend"],
    "PA": ["Philadelphia", "Pittsburgh", "Harrisburg", "Allentown"],
    "RI": ["Providence", "Warwick", "Cranston"],
    "SC": ["Charleston", "Columbia", "Greenville"],
    "SD": ["Sioux Falls", "Rapid City"],
    "TN": ["Nashville", "Memphis", "Knoxville", "Chattanooga"],
    "TX": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth", "Plano", "Irving"],
    "UT": ["Salt Lake City", "Provo", "Ogden", "St. George"],
    "VT": ["Burlington", "Montpelier"],
    "VA": ["Arlington", "Alexandria", "Richmond", "Norfolk", "Reston"],
    "WA": ["Seattle", "Spokane", "Tacoma", "Bellevue", "Redmond"],
    "WV": ["Charleston", "Morgantown", "Huntington"],
    "WI": ["Milwaukee", "Madison", "Green Bay"],
    "WY": ["Cheyenne", "Casper", "Laramie"],
    "DC": ["Washington"],
}
CITY_REGEX = re.compile(r"^[A-Za-z][A-Za-z\s\-\.\']{1,48}$")

# -------------------------- Company catalog (local) --------------------------
# You can freely extend/modify this. It’s a local cache used to prefill fields.
# Each entry may include:
#   "employees": int,              -> used to infer size label & min/max range
#   "size_label": str,             -> optional; else inferred from employees
#   "sector": str,
#   "ownership": str,              -> e.g., "Company - Public" / "Company - Private"
#   "founded": int,                -> used to compute company_age
#   "hq_city": str, "hq_state": str  (two-letter code)
COMPANY_CATALOG: Dict[str, Dict[str, Any]] = {
    "google":        {"employees": 180000, "sector": "Information Technology", "ownership": "Company - Public", "founded": 1998, "hq_city": "Mountain View", "hq_state": "CA"},
    "alphabet":      {"employees": 180000, "sector": "Information Technology", "ownership": "Company - Public", "founded": 2015, "hq_city": "Mountain View", "hq_state": "CA"},
    "microsoft":     {"employees": 220000, "sector": "Information Technology", "ownership": "Company - Public", "founded": 1975, "hq_city": "Redmond", "hq_state": "WA"},
    "apple":         {"employees": 160000, "sector": "Information Technology", "ownership": "Company - Public", "founded": 1976, "hq_city": "Cupertino", "hq_state": "CA"},
    "openai":        {"employees": 2000,   "sector": "Information Technology", "ownership": "Company - Private", "founded": 2015, "hq_city": "San Francisco", "hq_state": "CA"},
    "nvidia":        {"employees": 30000,  "sector": "Semiconductors",        "ownership": "Company - Public", "founded": 1993, "hq_city": "Santa Clara", "hq_state": "CA"},
    "meta":          {"employees": 67000,  "sector": "Information Technology", "ownership": "Company - Public", "founded": 2004, "hq_city": "Menlo Park", "hq_state": "CA"},
    "amazon":        {"employees": 1500000,"sector": "Consumer Services",     "ownership": "Company - Public", "founded": 1994, "hq_city": "Seattle", "hq_state": "WA"},
    "salesforce":    {"employees": 72000,  "sector": "Information Technology", "ownership": "Company - Public", "founded": 1999, "hq_city": "San Francisco", "hq_state": "CA"},
    "netflix":       {"employees": 13000,  "sector": "Entertainment & Media", "ownership": "Company - Public", "founded": 1997, "hq_city": "Los Gatos", "hq_state": "CA"},
    "uber":          {"employees": 30000,  "sector": "Transportation",        "ownership": "Company - Public", "founded": 2009, "hq_city": "San Francisco", "hq_state": "CA"},
    "airbnb":        {"employees": 6700,   "sector": "Travel & Tourism",      "ownership": "Company - Public", "founded": 2008, "hq_city": "San Francisco", "hq_state": "CA"},
    "snowflake":     {"employees": 8000,   "sector": "Information Technology", "ownership": "Company - Public", "founded": 2012, "hq_city": "Bozeman", "hq_state": "MT"},
    "databricks":    {"employees": 7000,   "sector": "Information Technology", "ownership": "Company - Private", "founded": 2013, "hq_city": "San Francisco", "hq_state": "CA"},
    "stripe":        {"employees": 7000,   "sector": "Information Technology", "ownership": "Company - Private", "founded": 2010, "hq_city": "San Francisco", "hq_state": "CA"},
    "cadence":       {"employees": 10500,  "sector": "Semiconductors",        "ownership": "Company - Public", "founded": 1988, "hq_city": "San Jose", "hq_state": "CA"},
    "qualcomm":      {"employees": 51000,  "sector": "Semiconductors",        "ownership": "Company - Public", "founded": 1985, "hq_city": "San Diego", "hq_state": "CA"},
}

# -------------------------- Helpers --------------------------
def _titlecase(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\S+", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), s)

def _looks_like_location(text: str) -> bool:
    t = (text or "").strip().lower()
    if re.match(r"^[a-z\s]+,\s*[a-z]{2}$", t):  # "san jose, ca"
        return True
    if t.upper() in US_STATES:
        return True
    return False

def predict_point_range(df_row: pd.DataFrame) -> Tuple[float, float, float]:
    y = float(pipe.predict(df_row)[0])
    return y, max(0.0, y * 0.9), y * 1.1

def _fmt_none(v):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v)) or v == "" or v == []) else str(v)

def _derive_features_with_pipeline_steps(row: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror pipeline’s featurizers to expose 'seniority' and 'loc_tier'."""
    df0 = pd.DataFrame([row], columns=RAW_INPUTS)
    seniority_step = pipe.named_steps.get("seniority", None)
    loc_tier_step = pipe.named_steps.get("loc_tier", None)

    df1 = df0
    if seniority_step is not None:
        df1 = seniority_step.transform(df1)
    if loc_tier_step is not None:
        df1 = loc_tier_step.transform(df1)

    seniority = df1["seniority"].iloc[0] if "seniority" in df1.columns else None
    loc_tier = df1["loc_tier"].iloc[0] if "loc_tier" in df1.columns else None
    return {"seniority": seniority, "loc_tier": loc_tier}

# ---- JD parsing helpers (robust title extraction) ----
ROLE_KEYWORDS = [
    "artificial intelligence engineer", "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "senior data scientist", "data engineer", "software engineer",
    "mlops engineer", "research scientist", "nlp engineer", "computer vision engineer",
    "generative ai engineer", "deep learning engineer"
]
TITLE_REGEXES = [
    r"(?im)^\s*job\s*title\s*[:\-]\s*(.+)$",
    r"(?im)^\s*title\s*[:\-]\s*(.+)$",
    r"(?im)^\s*(?:position|role)\s*[:\-]\s*(.+)$",
]
NOISE_LINE_HINTS = [
    "logo", "share", "save", "easy apply", "promoted", "actively reviewing",
    "on-site", "onsite", "hybrid", "remote", "full-time", "part-time",
    "contract", "matches your job preferences", "message", "about the job",
    "rate-", "location-", "multiple locations", "apply", "premium", "job poster"
]

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\uFFFD", "")
    for junk in ["â€¿", "Â", "â€“", "â€”", "â€™", "â€œ", "â€\x9d"]:
        s = s.replace(junk, " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _strip_paren_noise(title: str) -> str:
    t = re.sub(r"\s*\((?:no cpt|cpt|wfh|onsite|on-site|remote|hybrid|contract|full[- ]?time|part[- ]?time)\)\s*$",
               "", title, flags=re.I)
    t = re.sub(r"\s*\([^)]{0,40}\)\s*", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" -–—")
    return t.strip()

def _looks_like_noise_line(line_lower: str) -> bool:
    return any(h in line_lower for h in NOISE_LINE_HINTS)

def _candidate_title_from_line(line: str) -> bool:
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

def _try_parse_title_from_jd(jd_text: str) -> Optional[str]:
    if not jd_text or not jd_text.strip():
        return None
    txt = _clean_text(jd_text)
    try:
        parsed = jd_parser.parse_jd(txt)
    except Exception:
        parsed = None
    if parsed:
        title = (parsed or {}).get("title") or (parsed or {}).get("job_title")
        if title and not _looks_like_location(title):
            return _titlecase(_strip_paren_noise(title))
    for rx in TITLE_REGEXES:
        m = re.search(rx, txt)
        if m:
            cand = _strip_paren_noise(m.group(1).strip())
            if cand and not _looks_like_location(cand):
                return _titlecase(cand)
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    for ln in lines[:25]:
        if _looks_like_noise_line(ln.lower()):
            continue
        if _candidate_title_from_line(ln) and not _looks_like_location(ln):
            return _titlecase(_strip_paren_noise(ln))
    return None

def _try_parse_location_from_jd(jd_text: str) -> Optional[Tuple[str, str]]:
    if not jd_text or not jd_text.strip():
        return None
    try:
        parsed = jd_parser.parse_jd(jd_text)
    except Exception:
        return None
    loc = (parsed or {}).get("location") or ""
    if not loc or "," not in loc:
        return None
    city_part, state_part = [p.strip() for p in loc.split(",", 1)]
    state_abbrev = state_part.upper()
    city_norm = _titlecase(city_part)
    if state_abbrev in STATE_TO_CITIES and city_norm in STATE_TO_CITIES[state_abbrev]:
        return (city_norm, state_abbrev)
    return None

# -------------------------- Company lookup logic --------------------------
# Size label heuristic (fallback if not provided)
def _infer_size_label_from_employees(n: Optional[int]) -> Optional[str]:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return None
    try:
        n = int(n)
    except Exception:
        return None
    if n < 50:         return "Small"
    if n < 200:        return "Mid"
    if n < 1000:       return "Large"
    if n < 10000:      return "XL"
    return "Enterprise"

def _employees_to_min_max(n: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if n is None: return (None, None)
    try: n = int(n)
    except Exception: return (None, None)
    # crude ±20% band snapped to sensible bounds
    lo = max(1, int(n * 0.8))
    hi = int(n * 1.2)
    return (lo, hi)

def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())

def company_lookup_local(name: str) -> Dict[str, Any]:
    """
    Local provider: looks in COMPANY_CATALOG. Designed so you can swap in real APIs later.
    Returns normalized fields for UI prefill; missing values come back as None.
    """
    key = _normalize_key(name)
    # Fuzzy: try exact, then loose contains
    entry = COMPANY_CATALOG.get(key)
    if not entry:
        for k, v in COMPANY_CATALOG.items():
            if key and key in k:
                entry = v
                break
    if not entry:
        return {
            "Sector": None,
            "Type of ownership": None,
            "Size": None,
            "min_size": None,
            "max_size": None,
            "company_age": None,
            "hq_city": None,
            "hq_state": None,
        }
    employees = entry.get("employees")
    size_label = entry.get("size_label") or _infer_size_label_from_employees(employees)
    min_emp, max_emp = _employees_to_min_max(employees)
    founded = entry.get("founded")
    this_year = dt.datetime.now().year
    company_age = (this_year - int(founded)) if founded else None
    return {
        "Sector": entry.get("sector"),
        "Type of ownership": entry.get("ownership"),
        "Size": size_label,
        "min_size": min_emp,
        "max_size": max_emp,
        "company_age": company_age,
        "hq_city": entry.get("hq_city"),
        "hq_state": entry.get("hq_state"),
    }

# -------------------------- Explanation (LLM or fallback) --------------------------
def llm_explain(context: Dict[str, Any], derived: Dict[str, Any], point: float, low: float, high: float) -> str:
    if client is None or not os.environ.get("OPENAI_API_KEY"):
        parts = [
            f"Predicted base salary: ${point:,.0f} (range ${low:,.0f}–${high:,.0f}).",
            "Inputs used by the model:",
            f"- Job title: {_fmt_none(context.get('Job Title'))}",
            f"- Location: {_fmt_none(context.get('Location'))}",
            f"- Rating: {_fmt_none(context.get('Rating'))}, Company age: {_fmt_none(context.get('company_age'))}",
            f"- Company size (min–max): {_fmt_none(context.get('min_size'))}–{_fmt_none(context.get('max_size'))}",
            f"- Sector: {_fmt_none(context.get('Sector'))}, Ownership: {_fmt_none(context.get('Type of ownership'))}, Size label: {_fmt_none(context.get('Size'))}",
            "Derived (from pipeline):",
            f"- Seniority: {_fmt_none(derived.get('seniority'))}",
            f"- Location tier: {_fmt_none(derived.get('loc_tier'))}",
            "Likely drivers: market (location tier), seniority inferred from title, company size/age, sector/ownership.",
        ]
        return " ".join(parts)

    prompt = f"""
You are a careful compensation assistant.

RULES
- ≤ 110 words. No invented numbers.
- Only reference training inputs and derived features:
  Inputs: Job Title, Location, Rating, company_age, min_size, max_size, Sector, Type of ownership, Size.
  Derived: seniority (from title), loc_tier (from location).
- Mention 2–3 likely drivers tied to these. Neutral tone.

INPUTS
- Job title: {context.get('Job Title','—')}
- Location: {context.get('Location','—')}
- Rating: {context.get('Rating','—')}, Company age: {context.get('company_age','—')}
- Company size (min–max): {context.get('min_size','—')}–{context.get('max_size','—')}
- Sector: {context.get('Sector','—')}, Ownership: {context.get('Type of ownership','—')}, Size label: {context.get('Size','—')}

DERIVED (pipeline)
- Seniority: {_fmt_none(derived.get('seniority'))}
- Location tier: {_fmt_none(derived.get('loc_tier'))}

MODEL
- Predicted base salary: ${point:,.0f}
- Range: ${low:,.0f}–${high:,.0f}

FORMAT
1) One-sentence summary (point + range).
2) 2–3 bullet drivers tied ONLY to inputs/derived above.
3) One compact recap of inputs + derived.
"""
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful, cautious compensation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return (f"Explanation unavailable (LLM error: {e}). Predicted ${point:,.0f} "
                f"(range ${low:,.0f}–${high:,.0f}).")

# -------------------------- Backend serving --------------------------
def serve(
    job_title: str,
    state_abbrev: str,
    city: str,
    rating: float,
    company_age: float,
    min_size: float,
    max_size: float,
    sector: str,
    type_of_ownership: str,
    size_label: str,
    job_description_text: str,
):
    # Title (allow JD autofill)
    job_title = (job_title or "").strip()
    if not job_title:
        parsed_title = _try_parse_title_from_jd(job_description_text)
        if parsed_title:
            job_title = parsed_title
    if not job_title:
        sample = (_clean_text(job_description_text)[:140] + "…") if job_description_text else ""
        return {"error": "Job Title is required. I couldn't parse a title from the JD. "
                         "Please type it (e.g., 'Artificial Intelligence Engineer'). "
                         f"JD preview: {sample}"}
    if _looks_like_location(job_title):
        return {"error": "Job Title looks like a location. Please enter a real job title (e.g., 'Senior Data Scientist')."}
    job_title = _titlecase(job_title)

    # Location: prefer UI selection; else JD parsing
    state_abbrev = (state_abbrev or "").strip().upper()
    city = (city or "").strip()

    def _valid_city_state(c: str, st: str) -> bool:
        return (st in STATE_TO_CITIES) and (c in STATE_TO_CITIES[st]) and bool(CITY_REGEX.match(c))

    if not (state_abbrev and state_abbrev in US_STATES and city and _valid_city_state(city, state_abbrev)):
        jd_loc = _try_parse_location_from_jd(job_description_text)
        if jd_loc:
            city, state_abbrev = jd_loc

    if not (state_abbrev and state_abbrev in US_STATES):
        return {"error": "Please select a valid US state or provide a JD with a parsable US location (e.g., 'San Jose, CA')."}
    if not (city and _valid_city_state(city, state_abbrev)):
        return {"error": f"Please select a valid City for state {state_abbrev} or provide a JD with a parsable '{city}, {state_abbrev}'."}

    location = f"{city}, {state_abbrev}"

    # numeric checks
    def _num(v, name, minv=None, maxv=None):
        try:
            x = float(v)
        except Exception:
            return None, f"{name} must be a number."
        if minv is not None and x < minv:
            return None, f"{name} must be ≥ {minv}."
        if maxv is not None and x > maxv:
            return None, f"{name} must be ≤ {maxv}."
        return x, None

    rating, err = _num(rating, "Rating", 0, 5)
    if err: return {"error": err}
    company_age, err = _num(company_age, "Company Age", 0, 200)
    if err: return {"error": err}
    min_size, err = _num(min_size, "Company Size (min employees)", 1, 1e6)
    if err: return {"error": err}
    max_size, err = _num(max_size, "Company Size (max employees)", 1, 1e7)
    if err: return {"error": err}
    if max_size < min_size:
        return {"error": "Max company size must be ≥ min company size."}

    # categorical normalization
    sector = (sector or "").strip()
    type_of_ownership = (type_of_ownership or "").strip()
    size_label = (size_label or "").strip()

    # Assemble model row
    row = {
        "Job Title": job_title,
        "Location": location,
        "Rating": rating,
        "company_age": company_age,
        "min_size": min_size,
        "max_size": max_size,
        "Sector": sector,
        "Type of ownership": type_of_ownership,
        "Size": size_label,
    }
    X = pd.DataFrame([row], columns=RAW_INPUTS)

    try:
        point, low, high = predict_point_range(X)
        derived = _derive_features_with_pipeline_steps(row)  # seniority + loc_tier
    except Exception as e:
        return {"error": f"Prediction failed. Check inputs/columns. Details: {e}"}

    explanation = llm_explain(row, derived, point, low, high)

    return {
        "Predicted Base Salary (USD)": f"${point:,.0f}",
        "Suggested Range (USD)": f"${low:,.0f} – ${high:,.0f}",
        "Derived features (from pipeline)": derived,
        "Explanation": explanation,
        "Inputs used by the model": row,
    }

# -------------------------- Company Lookup backend --------------------------
def prefill_from_company(company_name: str, current_state: str):
    """
    Returns updates for: sector, ownership, size label, min_size, max_size, company_age,
    and optionally updates City/State if we have an HQ match in the catalog.
    """
    prof = company_lookup_local(company_name or "")
    updates = {
        "sector": gr.update(value=prof.get("Sector") or ""),
        "ownership": gr.update(value=prof.get("Type of ownership") or ""),
        "size_lbl": gr.update(value=prof.get("Size") or ""),
        "min_size": gr.update(value=prof.get("min_size") if prof.get("min_size") is not None else gr.update()),  # keeps old if None
        "max_size": gr.update(value=prof.get("max_size") if prof.get("max_size") is not None else gr.update()),
        "company_age": gr.update(value=prof.get("company_age") if prof.get("company_age") is not None else gr.update()),
    }

    # If we have HQ city/state and they're in our lists, set them (and refresh city choices if state differs)
    hq_city, hq_state = prof.get("hq_city"), prof.get("hq_state")
    if hq_state in STATE_TO_CITIES and hq_city in STATE_TO_CITIES[hq_state]:
        # If state changed, refresh city choices; else just set city value
        if hq_state != (current_state or ""):
            new_cities = STATE_TO_CITIES.get(hq_state, [])
            updates["state"] = gr.update(value=hq_state, choices=US_STATES)
            updates["city"]  = gr.update(value=hq_city, choices=new_cities)
        else:
            # no change to choices; just set city selection
            updates["state"] = gr.update()  # no change
            updates["city"]  = gr.update(value=hq_city)
    else:
        updates["state"] = gr.update()
        updates["city"] = gr.update()

    return updates["sector"], updates["ownership"], updates["size_lbl"], updates["min_size"], updates["max_size"], updates["company_age"], updates["state"], updates["city"]

# -------------------------- UI --------------------------
def update_city_choices(state_code: str):
    cities = STATE_TO_CITIES.get(state_code, [])
    default_val = cities[0] if cities else None
    return gr.update(choices=cities, value=default_val)

with gr.Blocks(title="Salary Prediction Chatbot") as demo:
    gr.Markdown(
        "# 💬 Salary Prediction Chatbot (US-only with JD & Company Lookup)\n"
        "- Only training features are collected.\n"
        "- Seniority and Location Tier are computed by the model and shown after prediction.\n"
        "- Leave Job Title/Location blank and paste a JD: I’ll try to parse them.\n"
        "- Or use the Company Lookup tab to prefill sector/ownership/size/age (and HQ location when available)."
    )

    with gr.Tabs():
        # ------- TAB 1: Predict -------
        with gr.Tab("Predict"):
            with gr.Row():
                job_title = gr.Textbox(label="Job Title (leave blank to parse from JD)", placeholder="e.g., Senior Data Scientist")

            with gr.Row():
                state = gr.Dropdown(choices=US_STATES, value="CA", label="State (US)")
                init_cities = STATE_TO_CITIES.get("CA", [])
                city = gr.Dropdown(choices=init_cities, value=(init_cities[0] if init_cities else None), label="City")

            with gr.Row():
                rating      = gr.Number(label="Rating (0–5)", value=3.5, precision=2)
                company_age = gr.Number(label="Company Age (years)", value=10, precision=1)

            with gr.Row():
                min_size = gr.Number(label="Company Size (min employees)", value=50, precision=0)
                max_size = gr.Number(label="Company Size (max employees)", value=200, precision=0)

            with gr.Row():
                sector   = gr.Textbox(label="Sector", placeholder="e.g., Information Technology")
                type_own = gr.Textbox(label="Type of ownership", placeholder="e.g., Company - Private")
                size_lbl = gr.Textbox(label="Size (label)", placeholder="e.g., Mid")

            gr.Markdown("### Optional: Paste Job Description (for autofill)")
            jd_text = gr.Textbox(label="Job Description", lines=8, placeholder="Paste the JD here; I'll try to extract job title and location like 'City, ST'.")

            state.change(fn=update_city_choices, inputs=state, outputs=city, queue=False)

            go  = gr.Button("Predict & Explain")
            out = gr.JSON(label="Result")

            go.click(
                fn=serve,
                inputs=[job_title, state, city, rating, company_age, min_size, max_size, sector, type_own, size_lbl, jd_text],
                outputs=out
            )

        # ------- TAB 2: Company Lookup -------
        with gr.Tab("Company Lookup"):
            gr.Markdown("Type a company name to prefill sector / ownership / size / company age and (if known) HQ location.")
            with gr.Row():
                company_name = gr.Textbox(label="Company", placeholder="e.g., Cadence, Google, OpenAI")
                lookup_btn   = gr.Button("Search & Prefill")

            # Show current state/city so we can update them on lookup
            with gr.Row():
                cur_state = gr.Dropdown(choices=US_STATES, value="CA", label="Current State (mirror of Predict tab)")
                # keep this simple; user can repopulate Predict tab manually if desired
                cur_city  = gr.Dropdown(choices=STATE_TO_CITIES.get("CA", []), value=(STATE_TO_CITIES.get("CA", ["San Jose"])[0]), label="Current City (mirror)")

            # Outputs: we will update the PREDICT tab fields directly
            lookup_btn.click(
                fn=prefill_from_company,
                inputs=[company_name, cur_state],
                outputs=[sector, type_own, size_lbl, min_size, max_size, company_age, state, city],
            )

            # Keep the mirror dropdowns in sync when state updates from lookup
            state.change(fn=update_city_choices, inputs=state, outputs=city, queue=False)

if __name__ == "__main__":
    demo.launch()
