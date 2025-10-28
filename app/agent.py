# app/agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, Any, Tuple, Union
from datetime import datetime, timedelta

# 既支持作为包运行（tests: from app.agent import ...），也支持直接脚本方式运行
try:
    from .providers import fetch_company_profile_fast, compute_age, Source  # type: ignore
except Exception:  # pragma: no cover
    from providers import fetch_company_profile_fast, compute_age, Source  # type: ignore

CacheEntry = Union[Dict[str, Any], Tuple[Dict[str, Any], datetime]]


class CompanyAgent:
    def __init__(self, cache_ttl_hours: int = 24):
        # tests 里会检查这两个属性
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl: timedelta = timedelta(hours=cache_ttl_hours)

    # ---- tests 里会直接调用这两个方法 ----
    def clear_cache(self) -> None:
        self.cache.clear()

    def cleanup_expired(self) -> int:
        """清理过期条目，仅对 (data, ts) 结构生效；返回清理数量。"""
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

    # ---- 原有核心逻辑，保持你的行为与返回字段不变 ----
    def _mkkey(self, name: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

    def _not_expired(self, ts: datetime) -> bool:
        return (datetime.now() - ts) <= self.cache_ttl

    def lookup(self, company_name: str | None) -> Dict[str, Any]:
        key = self._mkkey(company_name)
        if not key:
            return {"_error": "empty company_name"}

        # 命中缓存：兼容 tests 手动塞的 (data, ts) 以及你原来存 dict 的情况
        if key in self.cache:
            entry = self.cache[key]
            if isinstance(entry, tuple) and len(entry) == 2:
                data, ts = entry
                if self._not_expired(ts):
                    return data
                else:
                    # 过期，先删掉，下面会刷新
                    del self.cache[key]
            elif isinstance(entry, dict):
                # 老结构：不做过期判断，直接返回（保持你原本行为）
                return entry

        # 未命中或已过期 -> 调 provider
        try:
            prof, sources = fetch_company_profile_fast(company_name)  # type: ignore[arg-type]
        except Exception as e:
            # tests 期望返回字典错误而不是抛异常
            return {"_error": f"fetch_failed: {e}"}

        # 年龄：tests 要支持 company_age；否则按 founded 计算
        if prof.get("company_age") is not None:
            try:
                age_val = int(prof.get("company_age"))
            except Exception:
                age_val = None
        else:
            age_val = compute_age(prof.get("founded")) if prof.get("founded") else None

        # **保持你“对的”键名与结构**（App 自动回填依赖这些）
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

        # 写入缓存为 (profile, ts)，以满足 tests 的读取/过期检查
        self.cache[key] = (profile, datetime.now())
        return profile
