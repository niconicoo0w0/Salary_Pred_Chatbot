# app/chatbot.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import re

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import helpers, us_locations
from providers import compute_age

# Global chat context (simple in-memory state)
_CTX: Dict[str, Any] = {
    "company": None,
    "company_profile": None,
    "city": None,
    "state": None,
    "title": None,
    "sector": None,
    "ownership": None,
    "size_band": None,
    "age": None,
    "rating": 3.5,
    "awaiting": None,            # "confirm_company" / "confirm_location" / "confirm_run"
    "location_candidates": None,
    "last_asked_missing": [],
}

# Reverse index city -> states so we can resolve things like "San Jose"
_CITY_TO_STATES: Dict[str, List[str]] = {}
for st, cities in us_locations.STATE_TO_CITIES.items():
    for c in cities:
        _CITY_TO_STATES.setdefault(c.lower(), []).append(st)


# ========== 1. Company candidates for provider lookup ==========

def _company_candidates(raw: str) -> List[str]:
    raw = raw.strip()
    # Strip obvious trailing location parts
    base = re.sub(r"\b(at|in)\b.+$", "", raw, flags=re.I).strip(", ").strip()
    cands: List[str] = []
    if base:
        cands.append(base)
    if base != raw:
        cands.append(raw)

    # Common suffixes
    if base:
        cands.append(f"{base} Inc.")
        cands.append(f"{base} Corporation")

    # For short single tokens, try "... Systems"
    if base and " " not in base and len(base) <= 8:
        cands.append(f"{base} Systems")

    # Special-case: many people just say "Cadence"
    if base.lower() == "cadence":
        cands.insert(0, "Cadence Design Systems")

    # Deduplicate while preserving order
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _score_provider_hit(
    prof: Dict[str, Any],
    user_text: str,
    want_city: Optional[str],
    want_state: Optional[str],
) -> float:
    """
    Score each provider result to decide which one is the best match.

    Heuristics:
    - Has sector -> +3
    - Has ownership / size / employees -> +1
    - If the user sounds like an engineer but sector is media/publishing/music -> -3
    - If HQ state/city matches the user's state/city -> +2 / +1
    """
    score = 0.0
    low = user_text.lower()
    sector = (prof.get("sector") or "").lower()
    hq_city = (prof.get("hq_city") or "").lower()
    hq_state = (prof.get("hq_state") or "").upper()

    if prof.get("sector"):
        score += 3.0
    if prof.get("ownership") or prof.get("size_label") or prof.get("employees"):
        score += 1.0

    # Engineering-like description from the user
    is_engy_user = any(k in low for k in ["engineer", "developer", "scientist", "ml", "software", "data "])
    if is_engy_user and sector in ("media", "publishing", "music"):
        score -= 3.0

    # Location consistency
    want_city_low = (want_city or "").lower()
    if want_state and hq_state and want_state == hq_state:
        score += 2.0
    if want_city_low and hq_city and want_city_low == hq_city:
        score += 1.0

    # If we collected nothing, give a small baseline score
    if score == 0.0:
        score = 0.5
    return score


def _fetch_company_profile_multi(
    raw_company: str,
    user_text: str,
    want_city: Optional[str],
    want_state: Optional[str],
) -> Dict[str, Any]:
    from providers import fetch_company_profile_fast

    best_prof: Dict[str, Any] = {}
    best_score = -1.0

    for name in _company_candidates(raw_company):
        prof, sources = fetch_company_profile_fast(name)
        if not prof:
            continue
        s = _score_provider_hit(prof, user_text, want_city, want_state)
        if s > best_score:
            best_score = s
            best_prof = prof
            # Keep some diagnostics for the UI / debugging
            best_prof["_matched_name"] = name
            best_prof["_sources"] = [s.model_dump() for s in sources]

    return best_prof


# ========== 2. Lightweight extraction helpers ==========

def _extract_company(text: str) -> Optional[str]:
    text = text.strip()
    STOP = r"(?=$|,|\.|\!|\?| at\b| in\b| located\b| office\b| branch\b)"
    pats = [
        r"\bjob\s+from\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
        r"\bfrom\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
        r"\boffer\s+from\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
        r"\bat\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
        r"\bwith\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
        r"\bcompany\s+(?:is|=)\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
        r"\bwork(?:ing)?\s+at\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
        r"\bwork(?:ing)?\s+for\s+([A-Z][A-Za-z0-9& .\-]+?)" + STOP,
    ]
    for pat in pats:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip()
    return None


def _extract_location(text: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    # Pattern like "San Jose, CA"
    m = re.search(r"\b([A-Z][a-zA-Z .]+),\s*([A-Z]{2})\b", text)
    if m:
        city = helpers.titlecase(m.group(1).strip())
        st = m.group(2).upper()
        if st in us_locations.US_STATES:
            return city, st, None

    # Pattern like "in San Jose" / "at San Jose"
    m2 = re.search(r"\b(?:at|in)\s+([A-Z][a-zA-Z .]+)\b", text)
    if m2:
        city = helpers.titlecase(m2.group(1).strip())
        cands = _CITY_TO_STATES.get(city.lower(), [])
        if len(cands) == 1:
            return city, cands[0], None
        elif len(cands) > 1:
            return city, None, cands

    return None, None, None


def _extract_title(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        if any(k in first.lower() for k in (
            "engineer", "scientist", "developer", "analyst", "manager", "architect", "designer"
        )):
            return helpers.titlecase(helpers.strip_paren_noise(first))
    low = text.lower()
    for t in [
        "senior machine learning engineer",
        "machine learning engineer",
        "ml engineer",
        "senior data scientist",
        "data scientist",
        "senior software engineer",
        "software engineer",
    ]:
        if t in low:
            return helpers.titlecase(t)
    return None


# ========== 3. Conversation entry points ==========

def reset_context() -> Dict[str, Any]:
    _CTX.clear()
    _CTX.update({
        "company": None,
        "company_profile": None,
        "city": None,
        "state": None,
        "title": None,
        "sector": None,
        "ownership": None,
        "size_band": None,
        "age": None,
        "rating": 3.5,
        "awaiting": None,
        "location_candidates": None,
        "last_asked_missing": [],
    })
    return {
        "answer": (
            "✅ All set — I’ve cleared the previous information.\n"
            "Tell me about your role like: `Senior ML Engineer at Databricks, Denver, CO`,\n"
            "and I’ll estimate a salary range for you."
        ),
        "need_more_info": True,
        "context": _CTX,
    }


def handle_chat(user_text: str) -> Dict[str, Any]:
    text = (user_text or "").strip()
    low = text.lower()

    # Manual reset
    if low in {"reset", "start over", "clear"}:
        return reset_context()

    # 0) User is confirming whether to run with defaults (optional fields missing)
    if _CTX.get("awaiting") == "confirm_run":
        if low in {"yes", "y", "yeah", "ok", "okay", "sure", "run"}:
            _CTX["awaiting"] = None
            return _run_from_ctx()
        elif low in {"no", "n", "nope"}:
            _CTX["awaiting"] = None
            return {
                "answer": (
                    "No worries — we don’t have to run the model yet.\n"
                    "You can share any of the optional details to refine the estimate, for example:\n"
                    "- `The company was founded in 1997`\n"
                    "- `It’s a public company in the Media sector`\n"
                    "Once you’re ready, I’ll re-run the prediction with the updated info."
                ),
                "need_more_info": True,
                "context": _CTX,
            }
        else:
            return {
                "answer": (
                    "Got it — to move forward, please reply with:\n"
                    "- `yes` to run the model now with defaults, or\n"
                    "- `no` if you’d like to add more details first."
                ),
                "need_more_info": True,
                "context": _CTX,
            }

    # 1) User is confirming the company we looked up
    if _CTX.get("awaiting") == "confirm_company":
        if low in {"yes", "y", "yeah", "correct"}:
            prof = _CTX.get("company_profile") or {}

            # If we only have "founded" year, derive the age once here
            if prof.get("age") is None and prof.get("founded") is not None:
                age_val = compute_age(prof["founded"])
                if age_val is not None:
                    prof["age"] = age_val

            # Push provider fields into the context
            if prof.get("sector"):
                _CTX["sector"] = prof["sector"]
            if prof.get("ownership"):
                _CTX["ownership"] = prof["ownership"]
            if prof.get("size_label"):
                _CTX["size_band"] = prof["size_label"]
            if prof.get("age") is not None:
                _CTX["age"] = prof["age"]

            _CTX["awaiting"] = None
            return _try_run_or_ask()

        elif low in {"no", "n", "nope"}:
            _CTX["awaiting"] = None
            return {
                "answer": (
                    "Got it — that wasn’t the right company.\n"
                    "Please send me the exact company name (for example: `Cadence Design Systems`, "
                    "`Netflix`, or `OpenAI`), and I’ll look it up again."
                ),
                "need_more_info": True,
                "context": _CTX,
            }
        else:
            return {
                "answer": (
                    "I just need a quick confirmation: if that company profile looks correct, reply `yes`.\n"
                    "If it’s not the right company, reply `no` and I’ll ask you for the correct name."
                ),
                "need_more_info": True,
                "context": _CTX,
            }

    # 2) User is disambiguating location (e.g., multiple states for the same city)
    if _CTX.get("awaiting") == "confirm_location":
        cands: List[str] = _CTX.get("location_candidates") or []
        chosen = None
        if low.isdigit():
            idx = int(low) - 1
            if 0 <= idx < len(cands):
                chosen = cands[idx]
        else:
            for s in cands:
                if low == s.lower() or low == s:
                    chosen = s
                    break
        if chosen:
            _CTX["state"] = chosen
            _CTX["awaiting"] = None
            _CTX["location_candidates"] = None
            return _try_run_or_ask()
        else:
            return {
                "answer": (
                    "I found several possible states for that city.\n"
                    "Please pick one by replying with the number or the state code, for example: "
                    + ", ".join(f"{i+1}. {s}" for i, s in enumerate(cands))
                ),
                "need_more_info": True,
                "context": _CTX,
            }

    # 3) Normal turn: try to extract new info from the user message
    new_company = _extract_company(text)
    new_city, new_state, loc_cands = _extract_location(text)
    new_title = _extract_title(text)

    # Company mention: fetch profile and ask for confirmation
    if new_company:
        _CTX["company"] = new_company

        prof = _fetch_company_profile_multi(
            raw_company=new_company,
            user_text=text,
            want_city=_CTX.get("city"),
            want_state=_CTX.get("state"),
        )
        _CTX["company_profile"] = prof or {}
        _CTX["awaiting"] = "confirm_company"

        # Capture location if present in the same sentence
        if new_city:
            _CTX["city"] = new_city
        if new_state:
            _CTX["state"] = new_state

        return {
            "answer": (
                f"I looked up **{prof.get('_matched_name') or new_company}**, and here’s what I found:\n"
                f"- HQ: {prof.get('hq_city') or '—'}, {prof.get('hq_state') or '—'}\n"
                f"- Sector: {prof.get('sector') or '—'}\n"
                f"- Ownership: {prof.get('ownership') or '—'}\n"
                f"- Size band: {prof.get('size_label') or '—'}\n"
                f"- Founded: {prof.get('founded') or '—'}\n\n"
                "I’ll use this to infer company size, industry, and age, which all affect compensation.\n"
                "Does this look like the correct company? (`yes` / `no`)"
            ),
            "need_more_info": True,
            "context": _CTX,
        }

    # No new company — maybe we only got location
    if new_city:
        _CTX["city"] = new_city
    if new_state:
        _CTX["state"] = new_state
    if loc_cands:
        _CTX["city"] = new_city
        _CTX["location_candidates"] = loc_cands
        _CTX["awaiting"] = "confirm_location"
        return {
            "answer": (
                f"The city **{new_city}** exists in multiple states: "
                + ", ".join(f"{i+1}. {s}" for i, s in enumerate(loc_cands))
                + ".\nPlease reply with the number (1/2/…) or the state code (e.g. `CA`)."
            ),
            "need_more_info": True,
            "context": _CTX,
        }

    # Only saw a title? Still useful to remember.
    if new_title:
        _CTX["title"] = new_title

    # At this point, see if we have enough to run or need to ask for more
    return _try_run_or_ask()


# ========== 4. Decide whether to run or ask for more info ==========

def _try_run_or_ask() -> Dict[str, Any]:
    need: List[str] = []

    if not _CTX.get("title"):
        need.append("job title")
    if not _CTX.get("city") or not _CTX.get("state"):
        need.append("US location (City, ST)")

    # Hard requirements missing
    if need:
        _CTX["last_asked_missing"] = need
        pretty = ", ".join(need)
        return {
            "answer": (
                "I’m not ready to estimate salary just yet — I’m still missing some key details:\n"
                f"- {pretty}\n\n"
                "You can tell me things like:\n"
                "- `My title is Senior Software Engineer`\n"
                "- `I’m based in San Jose, CA`\n"
                "Once I have those, I can run the model for you."
            ),
            "need_more_info": True,
            "context": _CTX,
        }

    # Optional fields (nice-to-have)
    soft: List[str] = []
    if not _CTX.get("sector"):
        soft.append("sector")
    if not _CTX.get("ownership"):
        soft.append("type of ownership")
    if not _CTX.get("size_band"):
        soft.append("size band")
    if _CTX.get("age") is None:
        soft.append("company age")

    # If we’re missing optional stuff, offer a choice
    if soft:
        _CTX["awaiting"] = "confirm_run"
        _CTX["last_asked_missing"] = soft
        missing_soft = ", ".join(soft)
        return {
            "answer": (
                "I already have enough information to give you a reasonable salary estimate.\n"
                f"There are still a few optional details missing that could refine it: {missing_soft}.\n\n"
                "If you’d like, I can run the model **now** using defaults — just reply `yes`.\n"
                "If you prefer to add more context first, reply `no` and we’ll fill in the gaps together."
            ),
            "need_more_info": True,
            "context": _CTX,
        }

    # We have everything we need (including optional fields) — just run
    return _run_from_ctx()


def _run_from_ctx() -> Dict[str, Any]:
    # Lazy import to avoid circular imports
    from predict_api import run_prediction

    res = run_prediction(
        job_title=_CTX["title"],
        city=_CTX["city"],
        state_abbrev=_CTX["state"],
        rating=_CTX.get("rating", 3.5),
        age=_CTX.get("age") or 0,
        sector=_CTX.get("sector") or "",
        type_of_ownership=_CTX.get("ownership") or "",
        size_band=_CTX.get("size_band") or "",
        jd_text="",
        company_name=_CTX.get("company") or "",
    )

    _CTX["awaiting"] = None

    # Unpack what the model actually used
    inputs = res.get("Inputs used by the model") or {}
    derived = res.get("Derived features (from pipeline)") or {}

    title = _CTX.get("title") or inputs.get("Job Title") or "Unknown"
    city = _CTX.get("city") or ""
    state = _CTX.get("state") or ""
    loc_str = f"{city}, {state}" if city and state else (inputs.get("Location") or "Unknown")

    company = _CTX.get("company") or ""
    sector = _CTX.get("sector") or inputs.get("Sector") or ""
    ownership = _CTX.get("ownership") or inputs.get("Type of ownership") or ""
    size_band = _CTX.get("size_band") or inputs.get("size_band") or ""
    age_years = _CTX.get("age") or inputs.get("age")

    rating = _CTX.get("rating", 3.5)
    seniority = derived.get("seniority")
    loc_tier = derived.get("loc_tier")

    tier_explain = {
        "very_high": "very competitive, high cost-of-living markets (e.g., SF Bay Area / NYC-like)",
        "high": "strong markets with above-average compensation",
        "mid": "typical mid-tier markets",
        "low": "lower-cost markets where salaries tend to be lower",
    }.get(loc_tier, None)

    pred_str = res["Predicted Base Salary (USD)"]
    range_str = res["Suggested Range (USD)"]

    lines: List[str] = []

    # Main result
    lines.append("### 💰 Estimated salary")
    lines.append(f"- **Predicted base**: {pred_str}  *(range {range_str})*")
    lines.append("")

    # Inputs the model relied on
    lines.append("### 📊 What I used to estimate this")
    lines.append(f"- **Job title**: {title}")
    lines.append(f"- **Location**: {loc_str}")

    if company:
        extras = [x for x in [sector or None, ownership or None, size_band or None] if x]
        extra_str = " · ".join(extras) if extras else ""
        if extra_str:
            lines.append(f"- **Company**: {company} ({extra_str})")
        else:
            lines.append(f"- **Company**: {company}")
    else:
        extras = [x for x in [sector or None, ownership or None, size_band or None] if x]
        if extras:
            lines.append(f"- **Company profile**: " + " · ".join(extras))

    if age_years not in (None, "", 0):
        try:
            age_int = int(age_years)
            lines.append(f"- **Company age used**: ~{age_int} years")
        except Exception:
            lines.append(f"- **Company age used**: {age_years}")

    lines.append(f"- **Glassdoor rating used**: {rating}")
    lines.append("")

    # Derived features
    lines.append("### 🧠 Extra signals the model inferred")
    if seniority:
        lines.append(f"- **Estimated seniority**: `{seniority}` (based on your title)")
    if loc_tier:
        if tier_explain:
            lines.append(f"- **Location tier**: `{loc_tier}` – {tier_explain}")
        else:
            lines.append(f"- **Location tier**: `{loc_tier}`")

    if not (seniority or loc_tier):
        lines.append("- (No additional derived features were available for this case.)")

    lines.append("")
    lines.append(
        "### ⚠️ How to use this\n"
        "- This is a **statistical estimate**, not an official offer from any company.\n"
        "- Actual compensation can vary with team, performance, equity, bonus, and timing.\n"
        "- Treat this as a **ballpark range** to sanity-check offers or support negotiations."
    )

    answer = "\n".join(lines)

    return {
        "answer": answer,
        "normalized_inputs": inputs,
        "derived": derived,
        "need_more_info": False,
        "context": _CTX,
    }
