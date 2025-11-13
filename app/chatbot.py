# app/chatbot.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import re

from utils import helpers, us_locations  # 都是纯工具，安全
from providers import compute_age 

# 全局上下文：清理聊天也要记得
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

# 反查 city → states，方便 “San Jose” 这种只给城市的
_CITY_TO_STATES: Dict[str, List[str]] = {}
for st, cities in us_locations.STATE_TO_CITIES.items():
    for c in cities:
        _CITY_TO_STATES.setdefault(c.lower(), []).append(st)


# ========== 1. 生成公司候选名，给 providers 用 ==========
def _company_candidates(raw: str) -> List[str]:
    raw = raw.strip()
    # 去掉明显location的尾巴
    base = re.sub(r"\b(at|in)\b.+$", "", raw, flags=re.I).strip(", ").strip()
    cands: List[str] = []
    if base:
        cands.append(base)
    if base != raw:
        cands.append(raw)

    # 常见公司尾巴
    if base:
        cands.append(f"{base} Inc.")
        cands.append(f"{base} Corporation")

    # 通用：如果是单词，而且看起来很短，就补一个 “... Systems”
    if base and " " not in base and len(base) <= 8:
        cands.append(f"{base} Systems")

    # 你这次遇到的这个情况：Cadence → 很多时候其实是 Cadence Design Systems
    # 不是硬编码“加州”，只是补足全称
    if base.lower() == "cadence":
        cands.insert(0, "Cadence Design Systems")

    # 去重
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
    给每次 fetch 回来的结果打分，谁分高用谁。
    规则是“泛化”的，不是死写 Cadence：
    - 有 sector → +3
    - 有 ownership / size → +1
    - 用户说了 engineer / scientist / ml / software，如果 sector 是媒体/出版/音乐 → -3
    - 用户说了 San Jose / CA，如果 provider 也是 CA → +2
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

    # 工程类用户句子
    is_engy_user = any(k in low for k in ["engineer", "developer", "scientist", "ml", "software", "data "])
    if is_engy_user and sector in ("media", "publishing", "music"):
        score -= 3.0

    # 地点一致性
    want_city_low = (want_city or "").lower()
    if want_state and hq_state and want_state == hq_state:
        score += 2.0
    if want_city_low and hq_city and want_city_low == hq_city:
        score += 1.0

    # 如果什么都没抓到，就给一个小分
    if score == 0.0:
        score = 0.5
    return score


def _fetch_company_profile_multi(raw_company: str, user_text: str,
                                 want_city: Optional[str], want_state: Optional[str]) -> Dict[str, Any]:
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
            # 把用到的名字也记上，方便回给用户看
            best_prof["_matched_name"] = name
            best_prof["_sources"] = [s.model_dump() for s in sources]

    return best_prof

# ========== 2. 常规抽取 ==========

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
    # "San Jose, CA"
    m = re.search(r"\b([A-Z][a-zA-Z .]+),\s*([A-Z]{2})\b", text)
    if m:
        city = helpers.titlecase(m.group(1).strip())
        st = m.group(2).upper()
        if st in us_locations.US_STATES:
            return city, st, None

    # "at San Jose" / "in San Jose"
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
        if any(k in first.lower() for k in ("engineer", "scientist", "developer",
                                            "analyst", "manager", "architect", "designer")):
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


# ========== 3. 对话入口 ==========

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
        "answer": "✅ Reset. Tell me again like: `Senior ML Engineer at Databricks, Denver, CO`.",
        "need_more_info": True,
        "context": _CTX,
    }


def handle_chat(user_text: str) -> Dict[str, Any]:
    text = (user_text or "").strip()
    low = text.lower()

    # 用户手动 reset
    if low in {"reset", "start over", "clear"}:
        return reset_context()
    
    # 0) 用户确认直接运行（可选项用默认值）
    if _CTX.get("awaiting") == "confirm_run":
        if low in {"yes", "y", "yeah", "ok", "okay", "sure", "run"}:
            _CTX["awaiting"] = None
            return _run_from_ctx()
        elif low in {"no", "n", "nope"}:
            _CTX["awaiting"] = None
            return {
                "answer": (
                    "No problem. You can tell me any of the optional fields, e.g. "
                    "`The company was founded in 1997` or "
                    "`It's a public company in Media sector`."
                ),
                "need_more_info": True,
                "context": _CTX,
            }
        else:
            return {
                "answer": "Reply `yes` to run with defaults, or `no` if you want to fill in more details.",
                "need_more_info": True,
                "context": _CTX,
            }

    # 1) 处理 yes/no → 确认公司
    if _CTX.get("awaiting") == "confirm_company":
        if low in {"yes", "y", "yeah", "correct"}:
            prof = _CTX.get("company_profile") or {}
            if prof.get("age") is None and prof.get("founded") is not None:
                age_val = compute_age(prof["founded"])
                if age_val is not None:
                    prof["age"] = age_val
            # 把 provider 的字段灌进去
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
                "answer": "OK, tell me the correct company name (e.g. `Cadence Design Systems`, `Netflix`, `OpenAI`).",
                "need_more_info": True,
                "context": _CTX,
            }
        else:
            return {
                "answer": "Reply `yes` to accept that company info, or `no` to correct it.",
                "need_more_info": True,
                "context": _CTX,
            }

    # 2) 处理 location disambiguation
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
                "answer": "Pick one: " + ", ".join(f"{i+1}. {s}" for i, s in enumerate(cands)),
                "need_more_info": True,
                "context": _CTX,
            }

    # 3) 正常解析一条用户输入
    new_company = _extract_company(text)
    new_city, new_state, loc_cands = _extract_location(text)
    new_title = _extract_title(text)

    if new_company:
        _CTX["company"] = new_company
        # 🔴 关键：这里真正用你的 providers
        prof = _fetch_company_profile_multi(
            raw_company=new_company,
            user_text=text,
            want_city=_CTX.get("city"),
            want_state=_CTX.get("state"),
        )
        _CTX["company_profile"] = prof or {}
        _CTX["awaiting"] = "confirm_company"

        # 顺手收用户说的地点
        if new_city:
            _CTX["city"] = new_city
        if new_state:
            _CTX["state"] = new_state

        # 给用户看我们自动查到了什么，然后让他 yes/no
        return {
            "answer": (
                f"I looked up **{prof.get('_matched_name') or new_company}**.\n"
                f"- HQ: {prof.get('hq_city') or '—'}, {prof.get('hq_state') or '—'}\n"
                f"- Sector: {prof.get('sector') or '—'}\n"
                f"- Ownership: {prof.get('ownership') or '—'}\n"
                f"- Size: {prof.get('size_label') or '—'}\n"
                f"- Founded: {prof.get('founded') or '—'}\n\n"
                "Is this the correct company? (`yes` / `no`)"
            ),
            "need_more_info": True,
            "context": _CTX,
        }

    # 没有新 company，就看看有没有 location
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
                f"City **{new_city}** exists in multiple states: "
                + ", ".join(f"{i+1}. {s}" for i, s in enumerate(loc_cands))
                + ". Reply with number or state code."
            ),
            "need_more_info": True,
            "context": _CTX,
        }

    # 有职称就记
    if new_title:
        _CTX["title"] = new_title

    # 到这一步，看看能不能跑
    return _try_run_or_ask()


# ========== 4. 看够不够，不够继续问；够了就跑模型 ==========

def _try_run_or_ask() -> Dict[str, Any]:
    need: List[str] = []
    if not _CTX.get("title"):
        need.append("job title")
    if not _CTX.get("city") or not _CTX.get("state"):
        need.append("US location (City, ST)")

    if need:
        _CTX["last_asked_missing"] = need
        return {
            "answer": "I still need: " + ", ".join(need) + ".",
            "need_more_info": True,
            "context": _CTX,
        }

    # 可选的
    soft: List[str] = []
    if not _CTX.get("sector"):
        soft.append("sector")
    if not _CTX.get("ownership"):
        soft.append("type of ownership")
    if not _CTX.get("size_band"):
        soft.append("size band")
    if _CTX.get("age") is None:
        soft.append("company age")

    if soft:
        _CTX["awaiting"] = "confirm_run"
        _CTX["last_asked_missing"] = soft
        return {
            "answer": (
                "I can run the model now with defaults.\n"
                f"Missing (optional): {', '.join(soft)}.\n"
                "Run with defaults? (yes/no)"
            ),
            "need_more_info": True,
            "context": _CTX,
        }

    return _run_from_ctx()


def _run_from_ctx() -> Dict[str, Any]:
    # 懒加载你的 predict_api，防止循环
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
    return {
        "answer": f"Estimated base: {res['Predicted Base Salary (USD)']} (range {res['Suggested Range (USD)']}).",
        "normalized_inputs": res.get("Inputs used by the model"),
        "derived": res.get("Derived features (from pipeline)"),
        "need_more_info": False,
        "context": _CTX,
    }
