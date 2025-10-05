# app.py — Training features only + derived features; US-only location with dependent City dropdown
# JD textbox to autofill Title/Location

import os
import sys
import re
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import gradio as gr
import joblib

# ---- Project utils (assumed to exist in utils/) ----
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import PIPELINE_PATH, RAW_INPUTS, OPENAI_MODEL
from utils.us_locations import US_STATES, STATE_TO_CITIES
from utils.helpers import (
    titlecase, looks_like_location, clean_text, strip_paren_noise,
    looks_like_noise_line, fmt_none
)
from utils.jd_parsing import CITY_REGEX

# ---- Optional LLM client ----
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
except Exception:
    client = None

# ---- Pipeline and custom transformers (needed for unpickling) ----
import featurizers   # noqa: F401
import jd_parser     # used for JD parse of location/title if needed

# ---- Company agent (optional enrichment) ----
from agent import CompanyAgent
_agent = CompanyAgent()

pipe = joblib.load(PIPELINE_PATH)


# ====================== Model helpers ======================
def predict_point_range(df_row: pd.DataFrame) -> Tuple[float, float, float]:
    y = float(pipe.predict(df_row)[0])
    return y, max(0.0, y * 0.9), y * 1.1


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


# ====================== Seniority detection (robust) ======================
from collections import Counter

LEVEL_BUCKETS = ["intern", "entry", "junior", "mid", "senior", "staff", "principal", "lead", "manager", "director", "vp", "cxo"]
LEVEL_ORDER = {b: i for i, b in enumerate(LEVEL_BUCKETS)}

# mapping to your 6 final buckets
MAP6 = {
    "intern": "intern",
    "entry": "entry",
    "junior": "entry",
    "mid": "entry",
    "senior": "senior",
    "staff": "staff",
    "principal": "staff",
    "lead": "staff",
    "manager": "manager",
    "director": "manager",
    "vp": "vp",
    "cxo": "vp",
}

TOKEN_TO_BUCKET = {
    r"\bintern(ship)?\b": "intern",
    r"\b(entry[-\s]?level|new\s*grad|new[-\s]?graduate)\b": "entry",
    r"\b(jr|junior|assoc(iate)?)\b": "junior",
    r"\bmid(-| )?level\b": "mid",
    r"\b(sr|senior|sr\.)\b": "senior",
    r"\bstaff\b": "staff",
    r"\b(principal|princi?pal)\b": "principal",
    r"\blead\b": "lead",
    r"\b(architect)\b": "principal",
}

MANAGER_TOKENS = {
    r"\b(manager|mgr)\b": "manager",
    r"\b(director|dir\.)\b": "director",
    r"\b(vp|vice\s*president)\b": "vp",
    r"\b(head\s+of|head,)\b": "director",
    r"\b(cdo|cto|cpo|cio|ciso|chief\s+[a-z]+(?:\s+officer)?)\b": "cxo",
}

LEVEL_NUM_TO_BUCKET = {
    1: "entry", 2: "junior", 3: "mid", 4: "senior", 5: "staff", 6: "principal", 7: "principal"
}
ROMAN_TO_INT = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}

YEARS_TO_BUCKET = [
    (0, 1, "entry"),
    (1, 3, "junior"),
    (3, 5, "mid"),
    (5, 7, "senior"),
    (7, 10, "staff"),
    (10, 99, "principal"),
]

def _extract_years_of_exp(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.lower()
    m = re.search(r"(\d{1,2})\s*\+?\s*(?:years|yrs)\b.*?(?:experience|exp)?", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _bucket_from_years(years: int) -> Optional[str]:
    for lo, hi, b in YEARS_TO_BUCKET:
        if lo <= years < hi:
            return b
    return None

def _extract_numeric_level(title: str) -> Optional[int]:
    if not title:
        return None
    t = title.lower()
    m = re.search(r"\b(?:l|level)[-\s]?(\d)\b", t)  # L5 / Level 5
    if m:
        return int(m.group(1))
    m = re.search(r"\b(iii|ii|iv|v|vi|vii)\b", t)   # II / III / IV...
    if m:
        return ROMAN_TO_INT.get(m.group(1).lower())
    m = re.search(r"\b(?:eng(?:ineer)?|sde|developer|software\s+engineer)\s*(\d)\b", t)  # Engineer 2
    if m:
        return int(m.group(1))
    m = re.search(r"\b(?:sde|engineer)\s*(ii|iii|iv|v)\b", t)  # SDE II
    if m:
        return ROMAN_TO_INT.get(m.group(1).lower())
    return None

def _score_votes(votes: List[str]) -> Tuple[Optional[str], float, List[str]]:
    if not votes:
        return None, 0.0, []
    if any(v in ["intern"] for v in votes):
        return "intern", 1.0, ["forced intern rule", f"votes={votes}"]
    if any(v in ["entry", "new grad", "new graduate", "college grad"] for v in votes):
        return "entry", 1.0, ["forced entry rule", f"votes={votes}"]

    counts = Counter(votes)
    most_common, freq = counts.most_common(1)[0]
    max_freq = max(counts.values())
    tied = [b for b, c in counts.items() if c == max_freq]
    if len(tied) > 1:
        agg = min(tied, key=lambda b: LEVEL_ORDER[b])  # bias to lower seniority when tied
    else:
        agg = most_common

    conf = freq / len(votes)
    idxs = [LEVEL_ORDER[v] for v in votes if v in LEVEL_ORDER]
    if idxs and (max(idxs) - min(idxs) > 2):
        conf *= 0.7  # penalize wide disagreement
    return agg, round(conf, 2), [f"votes={votes}", f"agg={agg}", f"conf={conf:.2f}"]

def detect_seniority(title: str, jd_text: str = "") -> Tuple[Optional[str], float, List[str]]:
    votes: List[str] = []
    reasons: List[str] = []
    t = (title or "").lower()

    # managerial overrides
    for rx, bucket in MANAGER_TOKENS.items():
        if re.search(rx, t):
            votes.append(bucket)
            reasons.append(f"mgr:{bucket}")

    # IC tokens
    for rx, bucket in TOKEN_TO_BUCKET.items():
        if re.search(rx, t):
            votes.append(bucket)
            reasons.append(f"tok:{bucket}")

    # numeric/roman levels
    lvl = _extract_numeric_level(t)
    if lvl is not None and lvl in LEVEL_NUM_TO_BUCKET:
        b = LEVEL_NUM_TO_BUCKET[lvl]
        votes.append(b)
        reasons.append(f"level:{lvl}->{b}")

    # years of experience from JD body
    yrs = _extract_years_of_exp(jd_text or "")
    if yrs is not None:
        b = _bucket_from_years(yrs)
        if b:
            votes.append(b)
        reasons.append(f"yoe:{yrs}->{b}")

    if not votes:
        return "mid", 0.45, ["default:mid"]

    bucket, conf, score_reasons = _score_votes(votes)
    return bucket, conf, reasons + score_reasons


# ====================== Job Title detection ======================
import difflib

SENIORITY_WORDS = [
    "intern", "junior", "jr", "mid", "senior", "sr", "staff", "principal", "lead",
    "manager", "director", "vp", "head", "chief", "architect", "fellow", "ii", "iii", "iv", "l4", "l5", "l6", "l7"
]
EMPLOYMENT_HINTS = [
    "full-time", "full time", "part-time", "contract", "temporary", "internship", "co-op",
    "remote", "hybrid", "on-site", "onsite"
]
DEPT_WORDS = [
    "platform", "infrastructure", "backend", "front end", "frontend", "full stack", "mobile", "ios", "android", "web",
    "data", "ml", "ai", "nlp", "llm", "cv", "vision", "security", "cloud", "devops", "sre", "qa", "test", "release",
    "embedded", "firmware", "robotics", "systems", "site reliability", "database", "dba", "analytics", "bi", "etl",
    "governance", "privacy", "compliance", "salesforce", "sap", "erp", "product", "program", "project"
]

TITLE_TAXONOMY = {
    "software engineer": ["swe", "software developer", "application engineer", "full stack engineer",
                          "backend engineer", "frontend engineer", "web developer", "mobile engineer",
                          "ios engineer", "android engineer", "game developer", "unity developer"],
    "data scientist": ["applied scientist", "machine learning scientist", "ml scientist",
                       "research scientist (ml)", "quantitative researcher", "statistician"],
    "machine learning engineer": ["ml engineer", "ai engineer", "deep learning engineer", "llm engineer",
                                  "gen ai engineer", "nlp engineer", "computer vision engineer", "cv engineer",
                                  "research engineer"],
    "data engineer": ["etl engineer", "data platform engineer", "analytics engineer", "bi engineer",
                      "big data engineer", "spark engineer", "dbt engineer", "data ops engineer"],
    "data analyst": ["business analyst", "bi analyst", "analytics analyst", "product analyst", "insights analyst"],
    "ml ops engineer": ["mlops engineer", "ml platform engineer", "ml infra engineer", "model ops engineer"],
    "site reliability engineer": ["sre", "reliability engineer"],
    "devops engineer": ["cloud engineer", "platform engineer", "infra engineer"],
    "security engineer": ["application security engineer", "appsec", "cloud security engineer", "security analyst"],
    "product manager": ["pm", "technical product manager", "ai product manager", "platform product manager"],
    "program manager": ["technical program manager", "tpm"],
    "project manager": [],
    "solutions architect": ["solution architect", "enterprise architect", "data architect", "cloud architect"],
    "qa engineer": ["test engineer", "software test engineer", "quality engineer", "automation engineer"],
    "database administrator": ["dba", "database engineer"],
    "ux designer": ["product designer", "ui/ux designer", "interaction designer", "ux researcher"],
    "sales engineer": ["solutions engineer", "pre-sales engineer"],
    "technical writer": ["documentation engineer"],
}
TITLE_CANON = set(TITLE_TAXONOMY.keys())
TITLE_ALIASES = {alias: canon for canon, aliases in TITLE_TAXONOMY.items() for alias in aliases}

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\(\)\[\]\{\}]+", " ", s)
    s = re.sub(r"[/|]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def _postclean_title(t: str) -> str:
    t = strip_paren_noise(t)
    t = re.sub(r"\b(remote|hybrid|on[-\s]?site|full[-\s]?time|part[-\s]?time)\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip(" -–—")
    return titlecase(t.strip())

def _line_looks_like_title(line: str) -> bool:
    t = _norm(line)
    parts = t.split()
    if len(parts) < 2 or len(parts) > 12:
        return False
    if any(h in t for h in EMPLOYMENT_HINTS):
        return False
    return any(w in t for w in ["engineer", "developer", "scientist", "analyst", "architect",
                                 "manager", "designer", "writer", "researcher", "administrator", "admin"])

def _guess_from_patterns(txt: str) -> Optional[str]:
    pats = [
        r"(?im)^\s*(?:job\s*)?title\s*[:\-]\s*(.+)$",
        r"(?im)^\s*(?:position|role)\s*[:\-]\s*(.+)$",
        r"(?is)\bwe\s+are\s+looking\s+for\s+(an?\s+)?([A-Za-z0-9 \-\/&\(\)\.']{3,80}?)(?=\s+(?:to|who|that|with|for|in)\b)",
        r"(?is)\bas\s+(an?\s+)?([A-Za-z0-9 \-\/&\(\)\.']{3,80}?)(?=\s+(?:you|the|we)\b)"
    ]
    for rx in pats:
        m = re.search(rx, txt)
        if m:
            cand = m.group(1) if m.lastindex == 1 else (m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(0))
            cand = _postclean_title(cand)
            if _line_looks_like_title(cand):
                return cand
    return None

def _canonize_head(t: str) -> str:
    tnorm = _norm(t)
    pool = list(TITLE_CANON) + list(TITLE_ALIASES.keys())
    match = difflib.get_close_matches(tnorm, pool, n=1, cutoff=0.85)
    if match:
        m = match[0]
        return TITLE_ALIASES.get(m, m)
    for head in TITLE_CANON:
        if head in tnorm:
            return head
    return t

def detect_job_title(jd_text: str, top_n: int = 3) -> Tuple[Optional[str], List[Tuple[str, float, str]]]:
    """
    Returns (best_title, debug_list) where debug_list = [(candidate, score, reason), ...]
    """
    if not jd_text or not jd_text.strip():
        return None, []
    txt = clean_text(jd_text)

    m = _guess_from_patterns(txt)
    if m:
        return _postclean_title(m), [(m, 0.95, "pattern")]

    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    cand_scores: List[Tuple[str, float, str]] = []
    for i, ln in enumerate(lines[:40]):
        if looks_like_noise_line(ln.lower()):
            continue
        if not _line_looks_like_title(ln):
            continue

        t = _postclean_title(ln)
        tnorm = _norm(t)
        score = 0.0
        reason = []

        score += max(0, 1.0 - i / 40.0) * 0.5  # earlier line → higher score
        reason.append(f"pos={i}")

        if any(k in tnorm for k in ["engineer", "developer", "scientist", "analyst", "architect", "manager", "designer", "writer", "researcher", "administrator"]):
            score += 0.5; reason.append("has_head")

        if any(re.search(rf"\b{re.escape(w)}\b", tnorm) for w in SENIORITY_WORDS):
            score += 0.2; reason.append("seniority")

        if any(w in tnorm for w in DEPT_WORDS):
            score += 0.2; reason.append("dept")

        words = t.split()
        if 2 <= len(words) <= 8:
            score += 0.2; reason.append("len_ok")
        else:
            score -= 0.1; reason.append("len_penalty")

        head = _canonize_head(t)
        if head in TITLE_CANON:
            score += 0.2; reason.append(f"canon={head}")

        cand_scores.append((t, score, ",".join(reason)))

    cand_scores.sort(key=lambda x: x[1], reverse=True)
    best = cand_scores[0][0] if cand_scores else None

    # Optional: LLM fallback if configured
    if not best and client and os.environ.get("OPENAI_API_KEY"):
        prompt = ("Extract the exact job title from this job description. "
                  "Return only the title, no extra text:\n\n" + txt[:6000])
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            guess = (r.choices[0].message.content or "").strip()
            if guess:
                best = _postclean_title(guess)
                cand_scores = [(best, 0.7, "llm")]
        except Exception:
            pass

    return best, cand_scores[:top_n]


def _try_parse_location_from_jd(jd_text: str) -> Optional[Tuple[str, str]]:
    if not jd_text or not jd_text.strip():
        return None
    try:
        parsed = jd_parser.parse_jd(jd_text)
    except Exception:
        return None
    loc = (parsed or {}).get("location") or ""
    if not isinstance(loc, str) or "," not in loc:
        return None
    city_part, state_part = [p.strip() for p in loc.split(",", 1)]
    state_abbrev = state_part.upper()
    city_norm = titlecase(city_part)
    if state_abbrev in STATE_TO_CITIES and city_norm in STATE_TO_CITIES[state_abbrev]:
        return (city_norm, state_abbrev)
    return None


# ====================== Explanation (LLM or fallback) ======================
def llm_explain(context: Dict[str, Any], derived: Dict[str, Any], point: float, low: float, high: float) -> str:
    if client is None or not os.environ.get("OPENAI_API_KEY"):
        parts = [
            f"Predicted base salary: ${point:,.0f} (range ${low:,.0f}-${high:,.0f}).",
            "Inputs used by the model:",
            f"- Job title: {fmt_none(context.get('Job Title'))}",
            f"- Location: {fmt_none(context.get('Location'))}",
            f"- Rating: {fmt_none(context.get('Rating'))}, Company age: {fmt_none(context.get('company_age'))}",
            f"- Company size (min-max): {fmt_none(context.get('min_size'))}-{fmt_none(context.get('max_size'))}",
            f"- Sector: {fmt_none(context.get('Sector'))}, Ownership: {fmt_none(context.get('Type of ownership'))}, Size label: {fmt_none(context.get('Size'))}",
            "Derived (from pipeline):",
            f"- Seniority: {fmt_none(derived.get('seniority'))}",
            f"- Location tier: {fmt_none(derived.get('loc_tier'))}",
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
- Mention 2-3 likely drivers tied to these. Neutral tone.

INPUTS
- Job title: {context.get('Job Title','—')}
- Location: {context.get('Location','—')}
- Rating: {context.get('Rating','—')}, Company age: {context.get('company_age','—')}
- Company size (min-max): {context.get('min_size','—')}-{context.get('max_size','—')}
- Sector: {context.get('Sector','—')}, Ownership: {context.get('Type of ownership','—')}, Size label: {context.get('Size','—')}

DERIVED (pipeline)
- Seniority: {fmt_none(derived.get('seniority'))}
- Location tier: {fmt_none(derived.get('loc_tier'))}

MODEL
- Predicted base salary: ${point:,.0f}
- Range: ${low:,.0f}-${high:,.0f}

FORMAT
1) One-sentence summary (point + range).
2) 2-3 bullet drivers tied ONLY to inputs/derived above.
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
                f"(range ${low:,.0f}-${high:,.0f}).")


# ====================== Backend serving ======================
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
    company_name: str = "",
    use_agent_flag: bool = True,
    overwrite_defaults: bool = True
):
    # UI update placeholders
    state_ui = gr.update()
    city_ui = gr.update()
    min_ui = gr.update()
    max_ui = gr.update()
    company_age_ui = gr.update()
    sector_ui = gr.update()
    own_ui = gr.update()
    size_lbl_ui = gr.update()

    # --- Company Agent backfill (optional) ---
    web_prof = {}
    if use_agent_flag and company_name:
        try:
            web_prof = _agent.lookup(company_name) or {}
        except Exception:
            web_prof = {}

        def pick_str(curr, new):
            if new is None or new == "":
                return curr
            if not curr or str(curr).strip() == "":
                return new
            return new if overwrite_defaults else curr

        def pick_num(curr, new):
            if new is None:
                return curr
            try:
                new = float(new)
            except Exception:
                return curr
            if curr in (None, ""):
                return new
            return new if overwrite_defaults else curr

        sector            = pick_str(sector,            web_prof.get("Sector"))
        type_of_ownership = pick_str(type_of_ownership, web_prof.get("Type of ownership"))
        size_label        = pick_str(size_label,        web_prof.get("Size"))
        min_size          = pick_num(min_size,          web_prof.get("min_size"))
        max_size          = pick_num(max_size,          web_prof.get("max_size"))
        company_age       = pick_num(company_age,       web_prof.get("company_age"))

        if sector:            sector_ui      = gr.update(value=sector)
        if type_of_ownership: own_ui         = gr.update(value=type_of_ownership)
        if size_label:        size_lbl_ui    = gr.update(value=size_label)
        if min_size is not None:   min_ui    = gr.update(value=min_size)
        if max_size is not None:   max_ui    = gr.update(value=max_size)
        if company_age is not None: company_age_ui = gr.update(value=company_age)

    # --- Title (parse if missing) ---
    job_title = (job_title or "").strip()
    if not job_title:
        parsed_title, _dbg = detect_job_title(job_description_text)
        if parsed_title:
            job_title = parsed_title

    if not job_title:
        sample = (clean_text(job_description_text)[:140] + "…") if job_description_text else ""
        return (
            {"error": "Job Title is required. I couldn't parse a title from the JD. "
                      "Please type it (e.g., 'Artificial Intelligence Engineer'). "
                      f"JD preview: {sample}"},
            state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
        )

    if looks_like_location(job_title):
        return (
            {"error": "Job Title looks like a location. Please enter a real job title (e.g., 'Senior Data Scientist')."},
            state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
        )
    job_title = titlecase(job_title)

    # Seniority (smart) mapped to 6 buckets
    smart_seniority, sen_conf, _sen_reasons = detect_seniority(job_title, job_description_text or "")
    smart_seniority6 = MAP6.get(smart_seniority or "", None)

    # --- Location logic ---
    state_abbrev = (state_abbrev or "").strip().upper()
    city = (city or "").strip()

    def _valid_city_state(c: str, st: str) -> bool:
        return (st in US_STATES) and (c in STATE_TO_CITIES.get(st, [])) and bool(CITY_REGEX.match(c))

    auto_loc_from_hq = False

    # 1) Try JD parse first
    if not (state_abbrev and state_abbrev in US_STATES and city and _valid_city_state(city, state_abbrev)):
        jd_loc = _try_parse_location_from_jd(job_description_text)
        if jd_loc:
            city, state_abbrev = jd_loc
            choices = list(STATE_TO_CITIES.get(state_abbrev, []))
            if city not in choices:
                choices.append(city)  # ensure selectable even if not prelisted
            state_ui = gr.update(value=state_abbrev)
            city_ui = gr.update(value=city, choices=choices)

    # 2) If still invalid, try company HQ
    if not (state_abbrev and state_abbrev in US_STATES and city and _valid_city_state(city, state_abbrev)):
        hq_city = (web_prof or {}).get("hq_city") or None
        hq_state = (web_prof or {}).get("hq_state") or None
        if hq_city and hq_state:
            hq_state_norm = hq_state.strip().upper()
            hq_city_norm = titlecase(str(hq_city).strip())
            if (hq_state_norm in US_STATES) and CITY_REGEX.match(hq_city_norm):
                state_abbrev = hq_state_norm
                city = hq_city_norm
                auto_loc_from_hq = True
                choices = list(STATE_TO_CITIES.get(state_abbrev, []))
                if city not in choices:
                    choices.append(city)
                state_ui = gr.update(value=state_abbrev)
                city_ui = gr.update(value=city, choices=choices)

    # 3) Final validation
    if state_abbrev not in US_STATES:
        return (
            {"error": "Please select a valid US state or provide a JD with a parsable US location (e.g., 'San Jose, CA')."},
            state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
        )

    if auto_loc_from_hq:
        if not city or not CITY_REGEX.match(city):
            return (
                {"error": f"HQ city '{city}' is invalid. Please provide a valid city name or choose a city for {state_abbrev}."},
                state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
            )
    else:
        if not (city and _valid_city_state(city, state_abbrev)):
            return (
                {"error": f"Please select a valid City for state {state_abbrev} or provide a JD with a parsable '{city}, {state_abbrev}'."},
                state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
            )

    location = f"{city}, {state_abbrev}"

    # --- Numeric checks ---
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
    if err:
        return ({"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui)
    company_age, err = _num(company_age, "Company Age", 0, 200)
    if err:
        return ({"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui)
    min_size, err = _num(min_size, "Company Size (min employees)", 1, 1e6)
    if err:
        return ({"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui)
    max_size, err = _num(max_size, "Company Size (max employees)", 1, 1e7)
    if err:
        return ({"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui)
    if max_size < min_size:
        return (
            {"error": "Max company size must be ≥ min company size."},
            state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
        )

    # --- Assemble model row ---
    sector = (sector or "").strip()
    type_of_ownership = (type_of_ownership or "").strip()
    size_label = (size_label or "").strip()

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

    # --- Predict & Derived ---
    try:
        point, low, high = predict_point_range(X)
        derived = _derive_features_with_pipeline_steps(row)  # seniority + loc_tier
    except Exception as e:
        return (
            {"error": f"Prediction failed. Check inputs/columns. Details: {e}"},
            state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
        )

    # Use smart seniority if confident; map to 6 buckets
    if smart_seniority6 and sen_conf >= 0.65:
        derived["seniority"] = smart_seniority6
        derived["seniority_conf"] = round(sen_conf, 2)

    explanation = llm_explain(row, derived, point, low, high)

    return (
        {
            "Predicted Base Salary (USD)": f"${point:,.0f}",
            "Suggested Range (USD)": f"${low:,.0f} - ${high:,.0f}",
            "Derived features (from pipeline)": derived,
            "Explanation": explanation,
            "Inputs used by the model": row,
        },
        state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    )


# ====================== UI ======================
def update_city_choices(state_code: str):
    cities = STATE_TO_CITIES.get(state_code, [])
    return gr.update(choices=cities)

with gr.Blocks(title="Salary Prediction Chatbot") as demo:
    gr.Markdown(
        "# 📈 Salary Prediction Chatbot (US-only)\n"
        "- Leave Job Title/Location blank and paste a JD: I'll try to parse them.\n"
        "- If Location is missing, I'll auto-fill from the company HQ when possible."
    )

    with gr.Tabs():
        with gr.Tab("Predict"):
            with gr.Row():
                job_title = gr.Textbox(label="Job Title (leave blank to parse from JD)", placeholder="e.g., Senior Data Scientist")

            with gr.Row():
                state = gr.Dropdown(choices=US_STATES, value=None, label="State (US)")
                init_cities = STATE_TO_CITIES.get(None, [])
                city = gr.Dropdown(choices=init_cities, value=None, label="City")

            with gr.Row():
                rating = gr.Number(label="Rating (0-5)", value=3.5, precision=2)
                company_age = gr.Number(label="Company Age (years)", value=None, precision=1)

            with gr.Row():
                min_size = gr.Number(label="Company Size (min employees)", value=None, precision=0)
                max_size = gr.Number(label="Company Size (max employees)", value=None, precision=0)

            with gr.Row():
                sector = gr.Textbox(label="Sector", placeholder="e.g., Information Technology")
                type_own = gr.Textbox(label="Type of ownership", placeholder="e.g., Company - Private")
                size_lbl = gr.Textbox(label="Size (label)", placeholder="e.g., Mid")

            with gr.Row():
                company_input = gr.Textbox(label="Company (optional)", placeholder="e.g., Databricks")
                use_agent = gr.Checkbox(label="Use Web Agent to fill missing company fields", value=True)
                overwrite_defaults = gr.Checkbox(label="Overwrite default values with agent data", value=True)

            gr.Markdown("### Optional: Paste Job Description (for autofill)")
            jd_text = gr.Textbox(label="Job Description", lines=8,
                                 placeholder="Paste the JD here; I'll try to extract job title and location like 'City, ST'.")

            state.change(fn=update_city_choices, inputs=state, outputs=city, queue=False)

            go = gr.Button("Predict & Explain")
            out = gr.JSON(label="Result")

            go.click(
                fn=serve,
                inputs=[job_title, state, city, rating, company_age, min_size, max_size,
                        sector, type_own, size_lbl, jd_text, company_input, use_agent, overwrite_defaults],
                outputs=[out, state, city, min_size, max_size, company_age, sector, type_own, size_lbl]
            )

if __name__ == "__main__":
    demo.launch()
