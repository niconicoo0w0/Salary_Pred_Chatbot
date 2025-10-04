# app.py — Training features only + derived features; US-only location with dependent City dropdown
# JD textbox to autofill Title/Location
import os, re, unicodedata, datetime as dt
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import gradio as gr
import joblib
from typing import Union

# Import utilities
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import PIPELINE_PATH, RAW_INPUTS, OPENAI_MODEL
from utils.us_locations import US_STATES, STATE_TO_CITIES
from utils.helpers import titlecase, looks_like_location, clean_text, strip_paren_noise, looks_like_noise_line, candidate_title_from_line, fmt_none
from utils.jd_parsing import CITY_REGEX, TITLE_REGEXES

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
except Exception:
    client = None

# Needed for unpickling custom transformers used in the pipeline
import featurizers  # noqa: F401
import jd_parser    # we will use parse_jd(job_description_text)

# app.py top section
from agent import CompanyAgent
_agent = CompanyAgent()

pipe = joblib.load(PIPELINE_PATH)


def predict_point_range(df_row: pd.DataFrame) -> Tuple[float, float, float]:
    y = float(pipe.predict(df_row)[0])
    return y, max(0.0, y * 0.9), y * 1.1

# _fmt_none moved to utils/helpers.py

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

# Helper functions moved to utils/helpers.py

def _try_parse_title_from_jd(jd_text: str) -> Optional[str]:
    if not jd_text or not jd_text.strip():
        return None
    txt = clean_text(jd_text)
    try:
        parsed = jd_parser.parse_jd(txt)
    except Exception:
        parsed = None
    if parsed:
        title = (parsed or {}).get("title") or (parsed or {}).get("job_title")
        if title and not looks_like_location(title):
            return titlecase(strip_paren_noise(title))
    for rx in TITLE_REGEXES:
        m = re.search(rx, txt)
        if m:
            cand = strip_paren_noise(m.group(1).strip())
            if cand and not looks_like_location(cand):
                return titlecase(cand)
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    for ln in lines[:25]:
        if looks_like_noise_line(ln.lower()):
            continue
        if candidate_title_from_line(ln) and not looks_like_location(ln):
            return titlecase(strip_paren_noise(ln))
    return None

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

# -------------------------- Explanation (LLM or fallback) --------------------------
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
    company_name: str = "",
    use_agent_flag: bool = True,
    overwrite_defaults: bool = True
):
    state_ui = gr.update()
    city_ui = gr.update()
    min_ui = gr.update()
    max_ui = gr.update()
    company_age_ui = gr.update()
    sector_ui = gr.update()
    own_ui = gr.update()
    size_lbl_ui = gr.update()
    # —— Agent backfill: try if company_name is provided (not just sector)
    if use_agent_flag and company_name:
        try:
            web_prof = _agent.lookup(company_name)
        except Exception:
            web_prof = {}

        if web_prof:
            # Text fields: fill if empty; if overwrite is checked, directly overwrite
            def pick_str(curr, new):
                if new is None or new == "": 
                    return curr
                if not curr or curr.strip() == "":
                    return new
                return new if overwrite_defaults else curr

            sector = pick_str(sector, web_prof.get("Sector"))
            type_of_ownership = pick_str(type_of_ownership, web_prof.get("Type of ownership"))
            size_label = pick_str(size_label, web_prof.get("Size"))

            # Numeric fields: fill if empty (None/""/nan); if overwrite is checked, overwrite
            def pick_num(curr, new):
                if new is None:
                    return curr
                try:
                    # Allow string numbers
                    new = float(new)
                except Exception:
                    return curr
                if curr in (None, ""):
                    return new
                # Gradio Number might give int/float; only replace when overwrite switch is on
                return new if overwrite_defaults else curr

            min_size   = pick_num(min_size,   web_prof.get("min_size"))
            max_size   = pick_num(max_size,   web_prof.get("max_size"))
            company_age= pick_num(company_age,web_prof.get("company_age"))
            
            sector_ui = gr.update(value=sector) if sector else gr.update()
            own_ui = gr.update(value=type_of_ownership) if type_of_ownership else gr.update()
            size_lbl_ui = gr.update(value=size_label) if size_label else gr.update()
            min_ui = gr.update(value=min_size) if min_size is not None else gr.update()
            max_ui = gr.update(value=max_size) if max_size is not None else gr.update()
            company_age_ui = gr.update(value=company_age) if company_age is not None else gr.update()

    job_title = (job_title or "").strip()
    if not job_title:
        parsed_title = _try_parse_title_from_jd(job_description_text)
        if parsed_title:
            job_title = parsed_title
    if not job_title:
        sample = (clean_text(job_description_text)[:140] + "…") if job_description_text else ""
        return {"error": "Job Title is required. I couldn't parse a title from the JD. "
                         "Please type it (e.g., 'Artificial Intelligence Engineer'). "
                         f"JD preview: {sample}"}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    if looks_like_location(job_title):
        return {"error": "Job Title looks like a location. Please enter a real job title (e.g., 'Senior Data Scientist')."}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    job_title = titlecase(job_title)

    # Location: prefer UI selection; else JD parsing
    state_abbrev = (state_abbrev or "").strip().upper()
    city = (city or "").strip()

    def _valid_city_state(c: str, st: str) -> bool:
        return (st in US_STATES) and (c in STATE_TO_CITIES.get(st, [])) and bool(CITY_REGEX.match(c))

    auto_loc_from_hq = False  # <<< NEW: track whether we auto-filled from HQ

    # First try to parse from JD (still requires being in controlled list)
    if not (state_abbrev and state_abbrev in US_STATES and city and _valid_city_state(city, state_abbrev)):
        jd_loc = _try_parse_location_from_jd(job_description_text)
        if jd_loc:
            city, state_abbrev = jd_loc
            choices = list(STATE_TO_CITIES.get(state_abbrev, []))
            if city not in choices:
                choices.append(city)
            state_ui = gr.update(value=state_abbrev)
            city_ui  = gr.update(value=city, choices=choices)

    # If still no valid Location, try to auto-fill with company HQ
    if not (state_abbrev and state_abbrev in US_STATES and city and _valid_city_state(city, state_abbrev)):
        hq_city = hq_state = None

        # ensure web_prof exists
        try:
            web_prof  # noqa: F821
        except NameError:
            web_prof = {}

        if use_agent_flag and company_name:
            hq_city  = (web_prof or {}).get("hq_city") or None
            hq_state = (web_prof or {}).get("hq_state") or None

        if hq_city and hq_state:
            hq_state_norm = hq_state.strip().upper()
            hq_city_norm  = titlecase(str(hq_city).strip())
            # Relaxed acceptance for HQ: only require state in US and city matches regex
            if (hq_state_norm in US_STATES) and CITY_REGEX.match(hq_city_norm):
                state_abbrev = hq_state_norm
                city = hq_city_norm
                auto_loc_from_hq = True
                choices = list(STATE_TO_CITIES.get(state_abbrev, []))
                if city not in choices:
                    choices.append(city)
                state_ui = gr.update(value=state_abbrev)
                city_ui  = gr.update(value=city, choices=choices)

    # Final validation:
    if state_abbrev not in US_STATES:
        return {"error": "Please select a valid US state or provide a JD with a parsable US location (e.g., 'San Jose, CA')."}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui

    if auto_loc_from_hq:
        # Relaxed check when the value comes from HQ
        if not city or not CITY_REGEX.match(city):
            return {"error": f"HQ city '{city}' is invalid. Please provide a valid city name or choose a city for {state_abbrev}."}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    else:
        # Original strict check for user/JD-provided city
        if not (city and _valid_city_state(city, state_abbrev)):
            return {"error": f"Please select a valid City for state {state_abbrev} or provide a JD with a parsable '{city}, {state_abbrev}'."}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui

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
    if err: return {"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    company_age, err = _num(company_age, "Company Age", 0, 200)
    if err: return {"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    min_size, err = _num(min_size, "Company Size (min employees)", 1, 1e6)
    if err: return {"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    max_size, err = _num(max_size, "Company Size (max employees)", 1, 1e7)
    if err: return {"error": err}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui
    if max_size < min_size:
        return {"error": "Max company size must be ≥ min company size."}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui

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
        return {"error": f"Prediction failed. Check inputs/columns. Details: {e}"}, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui

    explanation = llm_explain(row, derived, point, low, high)

    return {
        "Predicted Base Salary (USD)": f"${point:,.0f}",
        "Suggested Range (USD)": f"${low:,.0f} - ${high:,.0f}",
        "Derived features (from pipeline)": derived,
        "Explanation": explanation,
        "Inputs used by the model": row,
    }, state_ui, city_ui, min_ui, max_ui, company_age_ui, sector_ui, own_ui, size_lbl_ui

# -------------------------- UI --------------------------
def update_city_choices(state_code: str):
    cities = STATE_TO_CITIES.get(state_code, [])
    return gr.update(choices=cities)

with gr.Blocks(title="Salary Prediction Chatbot") as demo:
    gr.Markdown(
        "# 📈 Salary Prediction Chatbot (US-only)\n"
        "- Feel free to leave Job Title/Location blank and paste a JD: I'll try to parse them.\n"
        "- If Location is missing, I'll auto-fill from the company HQ when possible."
    )

    with gr.Tabs():
        # ------- TAB 1: Predict -------
        with gr.Tab("Predict"):
            with gr.Row():
                job_title = gr.Textbox(label="Job Title (leave blank to parse from JD)", placeholder="e.g., Senior Data Scientist")

            with gr.Row():
                state = gr.Dropdown(choices=US_STATES, value=None, label="State (US)")
                init_cities = STATE_TO_CITIES.get(None, [])
                city = gr.Dropdown(choices=init_cities, value=None, label="City")

            with gr.Row():
                rating      = gr.Number(label="Rating (0-5)", value=3.5, precision=2)
                company_age = gr.Number(label="Company Age (years)", value=None, precision=1)

            with gr.Row():
                min_size    = gr.Number(label="Company Size (min employees)", value=None, precision=0)
                max_size    = gr.Number(label="Company Size (max employees)", value=None, precision=0)

            with gr.Row():
                sector   = gr.Textbox(label="Sector", placeholder="e.g., Information Technology")
                type_own = gr.Textbox(label="Type of ownership", placeholder="e.g., Company - Private")
                size_lbl = gr.Textbox(label="Size (label)", placeholder="e.g., Mid")
            
            with gr.Row():
                company_input = gr.Textbox(label="Company (optional)", placeholder="e.g., Databricks")
                use_agent = gr.Checkbox(label="Use Web Agent to fill missing company fields", value=True)
                overwrite_defaults = gr.Checkbox(label="Overwrite default values with agent data", value=True)

            gr.Markdown("### Optional: Paste Job Description (for autofill)")
            jd_text = gr.Textbox(label="Job Description", lines=8, placeholder="Paste the JD here; I'll try to extract job title and location like 'City, ST'.")

            state.change(fn=update_city_choices, inputs=state, outputs=city, queue=False)

            go  = gr.Button("Predict & Explain")
            out = gr.JSON(label="Result")

            go.click(
                fn=serve,
                inputs=[job_title, state, city, rating, company_age, min_size, max_size, sector, type_own, size_lbl, jd_text, company_input, use_agent, overwrite_defaults],
                outputs=[out, state, city, min_size, max_size, company_age, sector, type_own, size_lbl]
            )
            
if __name__ == "__main__":
    demo.launch()
