# app/chatbot.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import re

from utils import helpers, jd_parsing, us_locations
from predict_api import run_prediction
from providers import fetch_company_profile_fast, compute_age  # ← use your existing provider

# ---------------------------
# GLOBAL, SERVER-SIDE MEMORY
# ---------------------------
GLOBAL_CONTEXT: Dict[str, Any] = {
    "company": None,
    "job_title": None,
    "location": None,   # "San Jose, CA"
    "city": None,
    "state": None,
    "sector": None,
    "type_of_ownership": None,
    "size_band": None,
    "age": None,
    # confirmation flow
    "awaiting_company_confirm": False,
    "pending_company_profile": None,   # raw provider dict
}

def _extract_company(text: str) -> Optional[str]:
    pats = [
        r"\bfrom\s+([A-Za-z0-9& .\-\(\)]+)",
        r"\bat\s+([A-Za-z0-9& .\-\(\)]+)",
        r"\bwith\s+([A-Za-z0-9& .\-\(\)]+)",
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"\b(in|at|based in)\b", cand, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            cand = cand.split(",", 1)[0].strip()
            if 2 <= len(cand) <= 80:
                return cand
    common = re.search(r"\b(Netflix|Google|Meta|Apple|Amazon|OpenAI|Databricks|Nvidia|Microsoft)\b", text, re.I)
    if common:
        return common.group(1).title()
    return None


def _extract_location(text: str) -> Tuple[Optional[str], Optional[str]]:
    if hasattr(jd_parsing, "parse_jd"):
        try:
            parsed = jd_parsing.parse_jd(text) or {}
            loc = parsed.get("location")
            if loc and "," in loc:
                city, st = [p.strip() for p in loc.split(",", 1)]
                st = st.upper()
                if st in us_locations.US_STATES:
                    return (helpers.titlecase(city), st)
        except Exception:
            pass

    m = re.search(r"\b([A-Z][a-zA-Z .]+),\s*([A-Z]{2})\b", text)
    if m:
        city, st = m.group(1), m.group(2).upper()
        if st in us_locations.US_STATES:
            return (helpers.titlecase(city), st)

    if re.search(r"\bremote\b", text, re.I):
        return (None, None)
    return (None, None)


def _extract_title(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        if any(w in first.lower() for w in ("engineer", "scientist", "developer",
                                            "analyst", "manager", "architect", "designer")):
            return helpers.titlecase(helpers.strip_paren_noise(first))

    low = text.lower()
    for kw in (
        "senior machine learning engineer",
        "machine learning engineer",
        "data scientist",
        "software engineer",
        "senior software engineer",
        "research scientist",
        "ml engineer",
    ):
        if kw in low:
            return helpers.titlecase(kw)
    return None


def _merge_into_global(**kwargs: Any) -> None:
    for k, v in kwargs.items():
        if v is not None and v != "":
            GLOBAL_CONTEXT[k] = v
    # maintain location string
    if GLOBAL_CONTEXT.get("city") and GLOBAL_CONTEXT.get("state"):
        GLOBAL_CONTEXT["location"] = f"{GLOBAL_CONTEXT['city']}, {GLOBAL_CONTEXT['state']}"


# ---------- NEW: provider enrich ----------
def _enrich_company(company: str) -> Dict[str, Any]:
    """Call providers and return normalized info (but do NOT force it into context yet)."""
    try:
        prof, sources = fetch_company_profile_fast(company)
    except Exception as e:
        return {"_error": f"provider_failed: {e}", "name": company}

    # normalize to our schema-ish
    sector = prof.get("sector")
    ownership = prof.get("ownership")
    size_label = prof.get("size_label") or prof.get("Size")
    # try to compute age
    age_val = prof.get("age") or compute_age(prof.get("founded"))
    return {
        "name": company,
        "sector": sector,
        "ownership": ownership,
        "size_band": size_label,
        "age": age_val,
        "__sources__": [s.model_dump() for s in (sources or [])],
        "_diagnostics": prof.get("_diagnostics", {}),
    }


def reset_context() -> Dict[str, Any]:
    """Completely clear the chatbot’s memory."""
    for k in GLOBAL_CONTEXT.keys():
        GLOBAL_CONTEXT[k] = None
    # re-init fixed structure
    GLOBAL_CONTEXT.update({
        "awaiting_company_confirm": False,
        "pending_company_profile": None,
    })
    return {
        "answer": "Context cleared ✅ — Do you want to **start over**? (yes / no)",
        "need_more_info": True,
        "context": GLOBAL_CONTEXT,
    }


def handle_chat(user_text: str) -> Dict[str, Any]:
    text = (user_text or "").strip()

    # 0) handle YES / NO to company confirmation
    if GLOBAL_CONTEXT.get("awaiting_company_confirm"):
        if text.lower().strip() in {"yes", "y", "correct", "yeah"}:
            pending = GLOBAL_CONTEXT.get("pending_company_profile") or {}
            _merge_into_global(
                sector=pending.get("sector"),
                type_of_ownership=pending.get("ownership"),
                size_band=pending.get("size_band"),
                age=pending.get("age"),
            )
            GLOBAL_CONTEXT["awaiting_company_confirm"] = False
            GLOBAL_CONTEXT["pending_company_profile"] = None
            return {
                "answer": "Great — I’ll use that Netflix profile 👌. Now tell me the **job title**.",
                "need_more_info": True,
                "context": GLOBAL_CONTEXT,
            }
        if text.lower().strip() in {"no", "n", "not correct", "wrong"}:
            GLOBAL_CONTEXT["awaiting_company_confirm"] = False
            GLOBAL_CONTEXT["pending_company_profile"] = None
            return {
                "answer": "No problem — tell me the correct sector / ownership / company age, or just paste the JD.",
                "need_more_info": True,
                "context": GLOBAL_CONTEXT,
            }
        # if user typed something else, just re-ask
        return {
            "answer": "Is that company info correct? (yes / no)",
            "need_more_info": True,
            "context": GLOBAL_CONTEXT,
        }

    # 1) normal flow
    if not text:
        return {
            "answer": "Tell me about the job (title + US city, ST + company).",
            "need_more_info": True,
            "context": GLOBAL_CONTEXT,
        }

    company_now = _extract_company(text)
    title_now = _extract_title(text)
    city_now, state_now = _extract_location(text)

    _merge_into_global(
        company=company_now,
        job_title=title_now,
        city=city_now,
        state=state_now,
    )

    # 👇 this is your example: company + location, no title yet
    if GLOBAL_CONTEXT.get("company") and GLOBAL_CONTEXT.get("state") and not GLOBAL_CONTEXT.get("job_title"):
        # look up company right away
        enriched = _enrich_company(GLOBAL_CONTEXT["company"])
        GLOBAL_CONTEXT["awaiting_company_confirm"] = True
        GLOBAL_CONTEXT["pending_company_profile"] = enriched

        # build a friendly confirmation
        desc_bits = []
        if enriched.get("sector"):
            desc_bits.append(f"sector: **{enriched['sector']}**")
        if enriched.get("ownership"):
            desc_bits.append(f"ownership: **{enriched['ownership']}**")
        if enriched.get("size_band"):
            desc_bits.append(f"size: **{enriched['size_band']}**")
        if enriched.get("age") is not None:
            desc_bits.append(f"age: **{enriched['age']} yrs**")

        found_desc = ", ".join(desc_bits) if desc_bits else "(no public details found)"

        return {
            "answer": (
                f"I looked up **{GLOBAL_CONTEXT['company']}** online 👀 and found: {found_desc}.\n"
                "Is this correct? (yes / no)\n"
                "After that, tell me the **job title** (e.g. `Senior Machine Learning Engineer`)."
            ),
            "need_more_info": True,
            "context": GLOBAL_CONTEXT,
        }

    # title but no location
    if GLOBAL_CONTEXT.get("job_title") and not GLOBAL_CONTEXT.get("state"):
        return {
            "answer": "Cool — what's the US location (City, ST)?",
            "need_more_info": True,
            "context": GLOBAL_CONTEXT,
        }

    # have title + location → predict
    if GLOBAL_CONTEXT.get("job_title") and GLOBAL_CONTEXT.get("state"):
        city = GLOBAL_CONTEXT.get("city") or "San Jose"
        result = run_prediction(
            job_title=GLOBAL_CONTEXT["job_title"],
            city=city,
            state_abbrev=GLOBAL_CONTEXT["state"],
            rating=3.5,
            age=GLOBAL_CONTEXT.get("age") or 0,
            sector=GLOBAL_CONTEXT.get("sector") or "",
            type_of_ownership=GLOBAL_CONTEXT.get("type_of_ownership") or "",
            size_band=GLOBAL_CONTEXT.get("size_band") or "",
            jd_text=text,
            company_name=GLOBAL_CONTEXT.get("company") or "",
        )
        return {
            "answer": (
                f"Estimated base: {result['Predicted Base Salary (USD)']} "
                f"(range {result['Suggested Range (USD)']})."
            ),
            "normalized_inputs": result.get("Inputs used by the model"),
            "derived": result.get("Derived features (from pipeline)"),
            "need_more_info": False,
            "context": GLOBAL_CONTEXT,
        }

    # fallback
    return {
        "answer": "I need job title + US location. Optional: company / JD.",
        "need_more_info": True,
        "context": GLOBAL_CONTEXT,
    }
