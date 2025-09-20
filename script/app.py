# app.py
import os, datetime as dt
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import gradio as gr
import joblib

# Optional LLM (gracefully degrades if no key)
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception:
    client = None
    OPENAI_MODEL = None

import jd_parser
import featurizers

PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "models/pipeline.pkl")
pipe = joblib.load(PIPELINE_PATH)

NUMERIC = ["Rating", "company_age", "min_size", "max_size"]
CATEGORICAL_BASE = ["Sector", "Type of ownership", "Size"]
RAW_INPUTS = NUMERIC + CATEGORICAL_BASE + ["Job Title", "Location"]

def predict_point_range(df_row: pd.DataFrame) -> Tuple[float, float, float]:
    y = float(pipe.predict(df_row)[0])
    return y, max(0.0, y*0.9), y*1.1  # placeholder interval

def llm_explain(
    context: Dict[str, Any],
    jd_info: Dict[str, Any],
    point: float,
    low: float,
    high: float
) -> str:
    """Explain the prediction and explicitly list BOTH user inputs and JD-extracted info."""
    # Pretty strings for JD info
    def _fmt(v): return "—" if v in (None, "", []) else str(v)
    salary_str = "—"
    if jd_info.get("salary_range"):
        lo, hi = jd_info["salary_range"]
        salary_str = f"${lo:,.0f}–${hi:,.0f}"

    # (Short) JD excerpt to keep prompts compact
    jd_excerpt = (context.get("job_description_text") or "").strip()
    if len(jd_excerpt) > 600:
        jd_excerpt = jd_excerpt[:600] + " ..."

    if client is None or not os.environ.get("OPENAI_API_KEY"):
        # Fallback: deterministic, no-LLM explanation that still shows all fields
        parts = [
            f"Predicted base salary: ${point:,.0f} (range ${low:,.0f}–${high:,.0f}).",
            "User-provided inputs:",
            f"- Job title: {context.get('Job Title', '—')}",
            f"- Location: {context.get('Location', '—')}",
            f"- Rating: {context.get('Rating', '—')}, Company age: {context.get('company_age', '—')}",
            f"- Company size (min–max): {context.get('min_size', '—')}–{context.get('max_size', '—')}",
            f"- Sector: {context.get('Sector', '—')}, Ownership: {context.get('Type of ownership', '—')}, Size label: {context.get('Size', '—')}",
            f"- Application time: {context.get('application_ts', '—')}, Relocate: {context.get('relocate', '—')}",
            f"- Role requirements: {context.get('role_requirements', '—')}",
            "Extracted from job description:",
            f"- Years of experience: {_fmt(jd_info.get('years_experience'))}",
            f"- Education level: {_fmt(jd_info.get('education_level'))}",
            f"- JD location: {_fmt(jd_info.get('location'))}",
            f"- JD salary range: {salary_str}",
        ]
        return " ".join(parts)

    prompt = f"""
You are a compensation assistant. Explain a salary prediction in ≤140 words.

MUST INCLUDE (verbatim-style listing):
1) User-provided inputs:
   - Job title: {context.get('Job Title','—')}
   - Location: {context.get('Location','—')}
   - Rating: {context.get('Rating','—')}, Company age: {context.get('company_age','—')}
   - Company size (min–max): {context.get('min_size','—')}–{context.get('max_size','—')}
   - Sector: {context.get('Sector','—')}, Ownership: {context.get('Type of ownership','—')}, Size label: {context.get('Size','—')}
   - Application time: {context.get('application_ts','—')}, Willing to relocate: {context.get('relocate','—')}
   - Role requirements: {context.get('role_requirements','—')}

2) Information extracted from Job Description (JD):
   - Years of experience: {_fmt(jd_info.get('years_experience'))}
   - Education level: {_fmt(jd_info.get('education_level'))}
   - JD location: {_fmt(jd_info.get('location'))}
   - JD salary range: {salary_str}

JD excerpt (for context, do not quote if too long):
{jd_excerpt}

Model output:
- Predicted base salary: ${point:,.0f}
- Suggested range: ${low:,.0f}–${high:,.0f}

Guidelines:
- Be concise and neutral. Mention 2–4 likely drivers (market/location, seniority, company scale/age, requirements).
- Do not invent numbers outside the provided range.
- If a JD salary range exists, ensure your narrative acknowledges alignment to that range.
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


def serve(
    job_title: str,
    location: str,
    rating: float,
    company_age: float,
    min_size: float,
    max_size: float,
    sector: str,
    type_of_ownership: str,
    size_label: str,
    application_date: str,
    application_time: str,
    willing_to_relocate: bool,
    role_requirements: str,
    job_description_text: str,   # <<< NEW
):
    # --- Parse JD (optional) ---
    jd_parsed = jd_parser.parse_jd(job_description_text or "")

    # Optional: auto-fill blank fields from JD
    if not location.strip() and jd_parsed.get("location"):
        location = jd_parsed["location"]
        
    if (not (job_title.strip() and location.strip())) and not (job_description_text and job_description_text.strip()):
        return {"error": "Please provide Job Title and Location, or paste a Job Description."}


    row = {
        "Job Title": job_title.strip(),
        "Location": location.strip(),
        "Rating": rating,
        "company_age": company_age,
        "min_size": min_size,
        "max_size": max_size,
        "Sector": sector.strip(),
        "Type of ownership": type_of_ownership.strip(),
        "Size": size_label.strip(),
    }
    X = pd.DataFrame([row], columns=RAW_INPUTS)

    try:
        point, low, high = predict_point_range(X)

        # Clamp to JD salary range if present
        if jd_parsed.get("salary_range"):
            lo, hi = jd_parsed["salary_range"]
            point = min(max(point, lo), hi)
            low   = max(lo, low)
            high  = min(hi, high)

    except Exception as e:
        return {"error": f"Prediction failed. Check inputs/columns. Details: {e}"}

    # Explanation context (not fed to model)
    try:
        app_ts = dt.datetime.strptime(f"{application_date} {application_time}", "%Y-%m-%d %H:%M")
    except Exception:
        app_ts = dt.datetime.now()

    ctx = {
        **row,
        "application_ts": app_ts.isoformat(timespec="minutes"),
        "relocate": "Yes" if willing_to_relocate else "No",
        "role_requirements": (role_requirements or "").strip(),
        "job_description_text": (job_description_text or "").strip(),
    }

    explanation = llm_explain(ctx, jd_parsed, point, low, high)
    
    jd_block = {
        "years_experience": jd_parsed.get("years_experience"),
        "education_level": jd_parsed.get("education_level"),
        "jd_location": jd_parsed.get("location"),
        "jd_salary_range": jd_parsed.get("salary_range"),
    }

    return {
        "Predicted Base Salary (USD)": f"${point:,.0f}",
        "Suggested Range (USD)": f"${low:,.0f} – ${high:,.0f}",
        "Explanation": explanation,
        "Extracted From JD": jd_block
    }

with gr.Blocks(title="Salary Prediction Chatbot") as demo:
    gr.Markdown("# 💬 Salary Prediction Chatbot (Model-aligned)")
    with gr.Row():
        job_title = gr.Textbox(label="Job Title")
        location  = gr.Textbox(label="Location")
    with gr.Row():
        rating      = gr.Number(label="Rating", value=3.5, precision=2)
        company_age = gr.Number(label="Company Age (years)", value=10, precision=1)
    with gr.Row():
        min_size = gr.Number(label="Company Size (min employees)", value=50)
        max_size = gr.Number(label="Company Size (max employees)", value=200)
    with gr.Row():
        sector   = gr.Textbox(label="Sector")
        type_own = gr.Textbox(label="Type of ownership")
        size_lbl = gr.Textbox(label="Size (label)")

    gr.Markdown("### Optional context")
    with gr.Row():
        application_date = gr.Textbox(label="Application Date (YYYY-MM-DD)", value=dt.date.today().isoformat())
        application_time = gr.Textbox(label="Application Time (HH:MM 24h)", value="09:00")
    with gr.Row():
        willing_to_relocate = gr.Checkbox(label="Willing to Relocate?", value=False)
    role_requirements   = gr.Textbox(label="Role Requirements", lines=3)

    # <<< NEW: Job Description box
    job_description_text = gr.Textbox(label="Job Description (paste here to auto-parse exp/edu/location/salary range)", lines=6)

    go  = gr.Button("Predict & Explain")
    out = gr.JSON(label="Result")

    go.click(
        fn=serve,
        inputs=[job_title, location, rating, company_age, min_size, max_size,
                sector, type_own, size_lbl, application_date, application_time,
                willing_to_relocate, role_requirements, job_description_text],   # <<< include JD
        outputs=out
    )

if __name__ == "__main__":
    demo.launch()