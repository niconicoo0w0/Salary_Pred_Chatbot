# app/agent.py
import re
from typing import Dict, Any, Optional, Tuple

from providers import fetch_company_profile_fast, compute_age
from semantic_company import (
    CompanySemanticMatcher,
    light_clean_company,
)


def _very_light_cleanup(text: str) -> str:
    """
    Extra shield for chatty inputs:
    - strip trailing punctuation: "Cadence!" -> "Cadence"
    - remove leading prepositions: "at Netflix" -> "Netflix"
    """
    if not text:
        return ""
    s = text.strip()
    # remove leading conversational bits
    s = re.sub(r"^(at|with|from)\s+", "", s, flags=re.I)
    # remove trailing punctuation
    s = re.sub(r"[!?,.]+$", "", s)
    return s.strip()


class CompanyAgent:
    def __init__(self, semantic_matcher: Optional[CompanySemanticMatcher] = None):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.semantic_matcher = semantic_matcher or CompanySemanticMatcher()

    def _mkkey(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

    def _semantic_normalize(self, company_name: str) -> Tuple[Optional[str], Optional[float]]:
        """Try to make the name more 'company-like' using semantics."""
        if not self.semantic_matcher:
            return None, None
        cleaned = light_clean_company(company_name)
        score = self.semantic_matcher.score_company_like(cleaned)
        if score >= 0.6:
            return cleaned, score
        return None, score

    def lookup(self, company_name: str) -> Dict[str, Any]:
        # 0) guard
        if not company_name:
            return {"_error": "empty company_name"}

        # 1) normalize super-loud chat input
        original_input = company_name
        preclean = _very_light_cleanup(company_name)          # "Cadence!" -> "Cadence"
        key_preclean = self._mkkey(preclean)

        # 2) cache hit on pre-cleaned
        if key_preclean and key_preclean in self.cache:
            return self.cache[key_preclean]

        # 3) semantic pass
        semantic_name, semantic_score = self._semantic_normalize(preclean)
        name_to_query = semantic_name or preclean

        # 4) provider fetch
        prof, sources = fetch_company_profile_fast(name_to_query)

        # 5) build stable profile for chatbot
        hq_city  = prof.get("hq_city") if prof else None
        hq_state = prof.get("hq_state") if prof else None
        sector   = prof.get("sector") if prof else None
        ownership = prof.get("ownership") if prof else None

        profile = {
            # main fields chatbot cares about
            "company_name":        name_to_query,
            "hq_city":             hq_city,
            "hq_state":            hq_state,
            "sector":              sector,
            "ownership":           ownership,
            "size_band":           prof.get("size_label") if prof else None,
            "min_size":            prof.get("min_size") if prof else None,
            "max_size":            prof.get("max_size") if prof else None,
            "age": (
                compute_age(prof.get("founded"))
                if prof and prof.get("founded") else None
            ),
            # debug / trace
            "_original_input":     original_input,
            "_preclean":           preclean,
            "_queried_name":       name_to_query,
            "_semantic_used":      bool(semantic_name),
            "_semantic_score":     semantic_score,
            "_sources":            [s.model_dump() for s in (sources or [])],
        }

        # 6) write to cache under BOTH keys (raw + semantic)
        if key_preclean:
            self.cache[key_preclean] = profile
        if semantic_name:
            key_sem = self._mkkey(semantic_name)
            if key_sem:
                self.cache[key_sem] = profile

        return profile
