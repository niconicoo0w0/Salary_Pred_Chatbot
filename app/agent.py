# app/agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, Any, Tuple, Union
from datetime import datetime, timedelta

try:
    from .providers import fetch_company_profile_fast, compute_age, Source  # type: ignore
except Exception:  # pragma: no cover
    from providers import fetch_company_profile_fast, compute_age, Source  # type: ignore

CacheEntry = Union[Dict[str, Any], Tuple[Dict[str, Any], datetime]]


class CompanyAgent:
    def __init__(self, cache_ttl_hours: int = 24):
        # These 2 fields are directly inspected in tests
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl: timedelta = timedelta(hours=cache_ttl_hours)

    # ---- Tests call these directly ----
    def clear_cache(self) -> None:
        self.cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired cache entries (only applies to (data, ts) tuples). Returns count removed."""
        now = datetime.now()
        to_del = []
        for k, v in self.cache.items():
            if isinstance(v, tuple) and len(v) == 2:
                _, ts = v
                if now - ts > self.cache_ttl:
                    to_del.append(k)
        for k in to_del:
            del self.cache[k]
        return len(to_del)

    # ---- Core logic (keep original behavior and returned fields unchanged) ----
    def _mkkey(self, name: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

    def _not_expired(self, ts: datetime) -> bool:
        return (datetime.now() - ts) <= self.cache_ttl

    def lookup(self, company_name: str | None) -> Dict[str, Any]:
        key = self._mkkey(company_name)
        if not key:
            return {"_error": "empty company_name"}

        # Cache hit: support both (data, ts) tuple and pure dict from legacy behavior
        if key in self.cache:
            entry = self.cache[key]
            if isinstance(entry, tuple) and len(entry) == 2:
                data, ts = entry
                if self._not_expired(ts):
                    return data
                else:
                    # Expired -> remove and refresh below
                    del self.cache[key]
            elif isinstance(entry, dict):
                # Legacy structure: no expiry check (keep original behavior)
                return entry

        # Miss or expired -> fetch from provider
        try:
            prof, sources = fetch_company_profile_fast(company_name)  # type: ignore[arg-type]
        except Exception as e:
            # Tests expect a dict error instead of raising
            return {"_error": f"fetch_failed: {e}"}

        # Age: tests expect support for company_age; else compute from founded year
        if prof.get("company_age") is not None:
            try:
                age_val = int(prof.get("company_age"))
            except Exception:
                age_val = None
        else:
            age_val = compute_age(prof.get("founded")) if prof.get("founded") else None

        # **Keep correct keys/structure** (App autofill expects these exact names)
        profile = {
            "Sector":             prof.get("sector"),
            "Type of ownership":  prof.get("ownership"),
            "Size":               prof.get("size_label"),
            "min_size":           prof.get("min_size"),
            "max_size":           prof.get("max_size"),
            "age":                age_val,
            "hq_city":            prof.get("hq_city"),
            "hq_state":           prof.get("hq_state"),
            "__sources__":        [s.model_dump() for s in (sources or [])],
            "_diagnostics":       prof.get("_diagnostics", {}),
            "_canonical_website": prof.get("_website"),
        }

        # Write into cache as (profile, timestamp) for expiry checks
        self.cache[key] = (profile, datetime.now())
        return profile
