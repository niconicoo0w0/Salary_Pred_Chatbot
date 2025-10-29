# app.py — Aligns to new schema: size_band only (no min/max); training features only + derived features
# US-only location with dependent City dropdown; optional CompanyAgent backfill.

import os
import sys
import re
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import gradio as gr
import joblib
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# --- Local project imports (must exist in project root)
from utils import featurizers
from utils import helpers
from utils import jd_parsing
from utils import us_locations
from utils import constants

# --- Optional LLM client ---
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
except Exception:
    client = None

# --- Company agent (optional enrichment) ---
try:
    from agent import CompanyAgent
    _agent = CompanyAgent()
except Exception:
    _agent = None

# --- Load pipeline & schema ---
PIPELINE_PATH = os.environ.get("PIPELINE_PATH", getattr(constants, "PIPELINE_PATH", "models/pipeline_new.pkl"))
SCHEMA_PATH   = os.environ.get("SCHEMA_PATH", "models/schema.json")

# Fallback schema (must match training exactly)
_FALLBACK_SCHEMA = {
    "raw_inputs": ["Rating","age","Sector","Type of ownership","size_band","Job Title","Location"],
    "numeric": ["Rating","age"],
    "categorical_base": ["Sector","Type of ownership","size_band"],
    "added_by_featurizers": ["seniority","loc_tier"]
}

try:
    with open(SCHEMA_PATH, "r") as f:
        _SCHEMA = (f.read() or "").strip()
        _SCHEMA = __import__("json").loads(_SCHEMA)
except Exception:
    _SCHEMA = _FALLBACK_SCHEMA

# Validate essential pieces
_EXPECTED_RAW = ["Rating","age","Sector","Type of ownership","size_band","Job Title","Location"]
if _SCHEMA.get("raw_inputs") != _EXPECTED_RAW:
    # Hard override to ensure inference aligns with training
    _SCHEMA = _FALLBACK_SCHEMA

# Load pipeline AFTER importing featurizers
pipe = joblib.load(PIPELINE_PATH)

# ------------- Helpers (thin wrappers around local modules) -------------
titlecase = helpers.titlecase
looks_like_location = helpers.looks_like_location
clean_text = helpers.clean_text
strip_paren_noise = helpers.strip_paren_noise
looks_like_noise_line = helpers.looks_like_noise_line
fmt_none = helpers.fmt_none

CITY_REGEX = jd_parsing.CITY_REGEX
US_STATES = us_locations.US_STATES
STATE_TO_CITIES = us_locations.STATE_TO_CITIES

OPENAI_MODEL = getattr(constants, "OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
SIZE_BANDS   = getattr(constants, "SIZE_BANDS", ["Small","Mid","Large","XL","Enterprise"])


def predict_point_range(df_row: pd.DataFrame) -> Tuple[float, float, float]:
    y = float(pipe.predict(df_row)[0])
    return y, max(0.0, y * 0.9), y * 1.1


def _derive_features_with_pipeline_steps(row: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror pipeline’s featurizers to expose 'seniority' and 'loc_tier' for display."""
    df0 = pd.DataFrame([row], columns=_SCHEMA["raw_inputs"])
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


# ---------------- Sector mapping to training set ----------------
_TRAINING_SECTORS = set([
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

_SECTOR_CANON = {
    # common variants -> training sector
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

def map_sector_to_training(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = str(s).strip()
    low = t.lower()
    if low in _SECTOR_CANON:
        return _SECTOR_CANON[low]
    # direct match
    if t in _TRAINING_SECTORS:
        return t
    # lenient contain checks
    for k, v in _SECTOR_CANON.items():
        if k in low:
            return v
    return t  # keep original; OneHot(handle_unknown="ignore") will be safe


# -------------- Company profile -> schema normalization --------------
def employees_to_size_band(n: Optional[float]) -> Optional[str]:
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

def coerce_size_label_to_band(lbl: Optional[str]) -> Optional[str]:
    if not lbl:
        return None
    t = str(lbl).strip().lower()
    for b in SIZE_BANDS:
        if b.lower() == t:
            return b
    # light normalization
    if "small" in t: return "Small"
    if re.search(r"\bmid(dle)?\b", t): return "Mid"
    if "large" in t: return "Large"
    if t in {"xl","x-large","x large","extra large","xlarge"}: return "XL"
    if "enterprise" in t: return "Enterprise"
    return None

def choose_size_band_from_profile(prof: Dict[str, Any]) -> Optional[str]:
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


def normalize_company_profile_to_schema(prof: Dict[str, Any]) -> Dict[str, Any]:
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


# ---------------- JD title & location ----------------
def detect_job_title(jd_text: str) -> Optional[str]:
    if not jd_text or not jd_text.strip():
        return None
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
                return titlecase(cand)
    # 2) 前几行找头衔类词
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    for ln in lines[:25]:
        if looks_like_noise_line(ln.lower()):
            continue
        if re.search(r"\b(engineer|scientist|analyst|developer|researcher|manager|architect|designer)\b", ln.lower()):
            cand = strip_paren_noise(ln.strip())
            if cand and not looks_like_location(cand):
                return titlecase(cand)
    return None

def try_parse_location_from_jd(jd_text: str) -> Optional[Tuple[str, str]]:
    if not jd_text or not jd_text.strip():
        return None
    try:
        parsed = jd_parsing.parse_jd(jd_text)
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


# ---------------- LLM explanation ----------------
def llm_explain(context: Dict[str, Any], derived: Dict[str, Any], point: float, low: float, high: float) -> str:
    if client is None or not os.environ.get("OPENAI_API_KEY"):
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

    prompt = f"""
You are a careful compensation assistant.

RULES
- ≤ 110 words. No invented numbers.
- Only reference training inputs and derived features:
  Inputs: Job Title, Location, Rating, age, Sector, Type of ownership, size_band.
  Derived: seniority (from title), loc_tier (from location).
- Mention 2-3 likely drivers tied to these. Neutral tone.

INPUTS
- Job title: {context.get('Job Title','—')}
- Location: {context.get('Location','—')}
- Rating: {context.get('Rating','—')}, Company age: {context.get('age','—')}
- Sector: {context.get('Sector','—')}, Ownership: {context.get('Type of ownership','—')}, Size band: {context.get('size_band','—')}

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


# ---------------- Core serve() ----------------
def _to_model_row_from_ui(
    job_title: str,
    location: str,
    rating: float,
    age: float,
    sector: str,
    type_of_ownership: str,
    size_band: str,
) -> Dict[str, Any]:
    def _nan_if_blank(s):
        s = (s or "").strip()
        return s if s else np.nan
    row_full = {
        "Job Title": (job_title or "").strip(),
        "Location": (location or "").strip(),
        "Rating": float(rating) if rating is not None else float("nan"),
        "age": float(age) if age not in (None, "") else float("nan"),
        "Sector": _nan_if_blank(map_sector_to_training((sector or "").strip())),
        "Type of ownership": _nan_if_blank(type_of_ownership),
        "size_band": _nan_if_blank(size_band if size_band in SIZE_BANDS else None),
    }
    cols = _SCHEMA["raw_inputs"]  # ensure exact order
    return {k: row_full.get(k) for k in cols}


def serve(
    job_title: str,
    state_abbrev: str,
    city: str,
    rating: float,
    age: float,
    sector: str,
    type_of_ownership: str,
    size_band: str,
    job_description_text: str,
    company_name: str = "",
    use_agent_flag: bool = True,
    overwrite_defaults: bool = True
):
    # UI update placeholders
    state_ui = gr.update()
    city_ui = gr.update()
    age_ui = gr.update()
    sector_ui = gr.update()
    own_ui = gr.update()
    size_band_ui = gr.update()

    # --- Company Agent backfill (optional) ---
    web_prof = {}
    if use_agent_flag and _agent and company_name:
        try:
            web_raw = _agent.lookup(company_name) or {}
        except Exception:
            web_raw = {}
        web_prof = normalize_company_profile_to_schema(web_raw)

        def pick_str(curr, new):
            new = (new or "").strip()
            if not new:
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
        size_band         = pick_str(size_band,         web_prof.get("size_band"))
        age               = pick_num(age,               web_prof.get("age"))

        if sector:            sector_ui   = gr.update(value=sector)
        if type_of_ownership: own_ui      = gr.update(value=type_of_ownership)
        if size_band:         size_band_ui= gr.update(value=size_band)
        if age is not None:   age_ui      = gr.update(value=age)

    # --- Title (parse if missing) ---
    job_title = (job_title or "").strip()
    if not job_title:
        parsed_title = detect_job_title(job_description_text)
        if parsed_title:
            job_title = parsed_title

    if not job_title:
        sample = (clean_text(job_description_text)[:140] + "…") if job_description_text else ""
        return (
            {"error": "Job Title is required. I couldn't parse a title from the JD. "
                      "Please type it (e.g., 'Artificial Intelligence Engineer'). "
                      f"JD preview: {sample}"},
            state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui
        )

    if looks_like_location(job_title):
        return (
            {"error": "Job Title looks like a location. Please enter a real job title (e.g., 'Senior Data Scientist')."},
            state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui
        )
    job_title = titlecase(job_title)

    # --- Location logic ---
    state_abbrev = (state_abbrev or "").strip().upper()
    city = (city or "").strip()

    def _valid_city_state(c: str, st: str) -> bool:
        return (st in US_STATES) and (c in STATE_TO_CITIES.get(st, [])) and bool(CITY_REGEX.match(c))

    auto_loc_from_hq = False

    # 1) Try JD parse first
    if not (state_abbrev and state_abbrev in US_STATES and city and _valid_city_state(city, state_abbrev)):
        jd_loc = try_parse_location_from_jd(job_description_text)
        if jd_loc:
            city, state_abbrev = jd_loc
            choices = list(STATE_TO_CITIES.get(state_abbrev, []))
            if city not in choices:
                choices.append(city)  # ensure selectable
            state_ui = gr.update(value=state_abbrev)
            city_ui = gr.update(value=city, choices=choices)

    # 2) If still invalid, try company HQ
    if not (state_abbrev and state_abbrev in US_STATES and city and _valid_city_state(city, state_abbrev)) and web_prof:
        hq_city = web_prof.get("hq_city") or None
        hq_state = web_prof.get("hq_state") or None
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
            state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui
        )

    if auto_loc_from_hq:
        if not city or not CITY_REGEX.match(city):
            return (
                {"error": f"HQ city '{city}' is invalid. Please provide a valid city name or choose a city for {state_abbrev}."},
                state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui
            )
    else:
        if not (city and _valid_city_state(city, state_abbrev)):
            return (
                {"error": f"Please select a valid City for state {state_abbrev} or provide a JD with a parsable '{city}, {state_abbrev}'."},
                state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui
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
        return ({"error": err}, state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui)
    age, err = _num(age, "Company Age", 0, 200)
    if err:
        return ({"error": err}, state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui)

    # --- Assemble model row ---
    sector = (sector or "").strip()
    type_of_ownership = (type_of_ownership or "").strip()
    size_band = size_band if size_band in SIZE_BANDS else ""

    row = _to_model_row_from_ui(
        job_title=job_title,
        location=location,
        rating=rating,
        age=age,
        sector=sector,
        type_of_ownership=type_of_ownership,
        size_band=size_band,
    )
    X = pd.DataFrame([row], columns=_SCHEMA["raw_inputs"])

    # --- Predict & Derived ---
    try:
        point, low, high = predict_point_range(X)
        derived = _derive_features_with_pipeline_steps(row)  # seniority + loc_tier
    except Exception as e:
        return (
            {"error": f"Prediction failed. Check inputs/columns. Details: {e}"},
            state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui
        )

    explanation = llm_explain(row, derived, point, low, high)

    return (
        {
            "Predicted Base Salary (USD)": f"${point:,.0f}",
            "Suggested Range (USD)": f"${low:,.0f} - ${high:,.0f}",
            "Derived features (from pipeline)": derived,
            "Explanation": explanation,
            "Inputs used by the model": row,
        },
        state_ui, city_ui, age_ui, sector_ui, own_ui, size_band_ui
    )


# ---------------- UI ----------------
def update_city_choices(state_code: str):
    cities = STATE_TO_CITIES.get(state_code, [])
    return gr.update(choices=cities)

with gr.Blocks(title="Salary Prediction Chatbot") as demo:
    gr.Markdown(
        "# 📈 Salary Prediction Chatbot (US-only)\n"
        "- Paste a JD and leave Job Title/Location blank: I'll try to parse them.\n"
        "- If Location is missing, I'll auto-fill from the company HQ when possible.\n"
        "- **Model schema**: Rating, age, Sector, Type of ownership, size_band, Job Title, Location."
    )

    with gr.Tabs():
        with gr.Tab("Predict"):
            with gr.Row():
                job_title = gr.Textbox(label="Job Title (leave blank to parse from JD)", placeholder="e.g., Senior Data Scientist")

            with gr.Row():
                state = gr.Dropdown(choices=US_STATES, value=None, label="State (US)")
                city = gr.Dropdown(choices=[], value=None, label="City")

            with gr.Row():
                rating = gr.Number(label="Rating (0-5)", value=3.5, precision=2)
                age = gr.Number(label="Company Age (years)", value=None, precision=1)

            with gr.Row():
                sector = gr.Textbox(label="Sector", placeholder="e.g., Information Technology / Media / Finance ...")
                type_own = gr.Textbox(label="Type of ownership", placeholder="e.g., Company - Private / Company - Public")
                size_band = gr.Dropdown(label="Size band", choices=SIZE_BANDS, value=None)

            with gr.Row():
                company_input = gr.Textbox(label="Company (optional)", placeholder="e.g., Databricks")
                use_agent = gr.Checkbox(label="Use Web Agent to fill missing company fields", value=True)
                overwrite_defaults = gr.Checkbox(label="Overwrite default values with agent data", value=True)

            gr.Markdown("### Optional: Paste Job Description (for autofill)")
            jd_text = gr.Textbox(
                label="Job Description",
                lines=8,
                placeholder="Paste the JD here; I'll try to extract job title and location like 'City, ST'."
            )

            state.change(fn=update_city_choices, inputs=state, outputs=city, queue=False)

            go = gr.Button("Predict & Explain")
            out = gr.JSON(label="Result")

            go.click(
                fn=serve,
                inputs=[job_title, state, city, rating, age, sector, type_own, size_band, jd_text, company_input, use_agent, overwrite_defaults],
                outputs=[out, state, city, age, sector, type_own, size_band]
            )

if __name__ == "__main__":
    demo.launch()
