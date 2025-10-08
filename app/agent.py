# -*- coding: utf-8 -*-
import re
from typing import Dict, Any

from providers import fetch_company_profile_fast, compute_age

class CompanyAgent:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _mkkey(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

    def lookup(self, company_name: str) -> Dict[str, Any]:
        key = self._mkkey(company_name)
        if not key:
            return {"_error": "empty company_name"}

        if key in self.cache:
            return self.cache[key]

        prof, sources = fetch_company_profile_fast(company_name)

        profile = {
            "Sector":             prof.get("sector"),
            "Type of ownership":  prof.get("ownership"),
            "Size":               prof.get("size_label"),
            "min_size":           prof.get("min_size"),
            "max_size":           prof.get("max_size"),
            "age":        compute_age(prof.get("founded")) if prof.get("founded") else None,
            "hq_city":            prof.get("hq_city"),
            "hq_state":           prof.get("hq_state"),
            "__sources__":        [s.model_dump() for s in sources],
            "_diagnostics":       prof.get("_diagnostics", {}),
            "_canonical_website": prof.get("_website"),
        }
        
        self.cache[key] = profile
        return profile
