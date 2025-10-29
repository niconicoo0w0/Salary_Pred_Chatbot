# -*- coding: utf-8 -*-
# providers.py — Wikipedia infobox + company web + DDG
import re
import time
import random
import datetime as dt
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from urllib.parse import urlparse

try:
    from duckduckgo_search import DDGS
    HAVE_DDG = True
except Exception:
    HAVE_DDG = False

from utils.jd_parsing import _SECTOR_MAP, _TRAIN_SECTORS
from utils.config import config
from utils.logger import logger, timing_decorator, retry_decorator
from utils.cache import cached

# ---------- HTTP basics ----------
_ses = requests.Session()
_ses.headers.update({"User-Agent": random.choice(config.USER_AGENTS), "Accept-Language": "en-US,en;q=0.8"})
_ses.max_redirects = 5

def _ua() -> Dict[str, str]:
    return {"User-Agent": random.choice(config.USER_AGENTS), "Accept-Language": "en-US,en;q=0.8"}

@retry_decorator(max_retries=config.MAX_RETRIES, delay=config.REQUEST_DELAY_MIN, backoff=1.5)
def _get(url: str) -> Optional[str]:
    try:
        logger.debug(f"Fetching URL: {url}")
        r = _ses.get(url, headers=_ua(), timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and r.text:
            r.encoding = r.apparent_encoding or r.encoding
            logger.debug(f"Successfully fetched {len(r.text)} characters from {url}")
            return r.text
        else:
            logger.warning(f"Failed to fetch {url}: status {r.status_code}")
            return None
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        raise

def _html_to_text(html: str) -> str:
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            soup = BeautifulSoup(html, parser)
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
            if text:
                return text
        except Exception:
            continue
    return html if isinstance(html, str) else ""


# ---------- Pydantic model ----------
class Source(BaseModel):
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None

class CompanyProfile(BaseModel):
    name: str
    sector: Optional[str] = None
    ownership: Optional[str] = None           # "Company - Public" / "Company - Private"
    employees: Optional[int] = None
    founded: Optional[int] = None
    hq_city: Optional[str] = None
    hq_state: Optional[str] = None
    sources: List[Source] = Field(default_factory=list)


# ---------- Wikipedia helpers ----------
_COMPANY_TITLE_TOKENS = ("inc", "llc", "ltd", "(company)", "corporation", "holdings", "platforms")

_EXTRA_WIKI_OVERRIDES = {
    "stripe":   "https://en.wikipedia.org/wiki/Stripe,_Inc.",
    "meta":     "https://en.wikipedia.org/wiki/Meta_Platforms",
    "apple":    "https://en.wikipedia.org/wiki/Apple_Inc.",
    "amazon":   "https://en.wikipedia.org/wiki/Amazon_(company)",
    "google":   "https://en.wikipedia.org/wiki/Google",
    "openai":   "https://en.wikipedia.org/wiki/OpenAI",
    "datadog":  "https://en.wikipedia.org/wiki/Datadog",
}

def _wiki_get_fullurl(title: str) -> Optional[str]:
    try:
        info_api = ("https://en.wikipedia.org/w/api.php"
                    f"?action=query&prop=info&inprop=url&format=json&titles={urllib.parse.quote(title)}")
        info = _ses.get(info_api, headers=_ua(), timeout=10).json()
        pages = (info.get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), {})
        return page.get("fullurl")
    except Exception:
        return None

def _wiki_is_disambiguation(title: str) -> bool:
    try:
        api = ("https://en.wikipedia.org/w/api.php"
               f"?action=query&prop=pageprops&titles={urllib.parse.quote(title)}&format=json")
        js = _ses.get(api, headers=_ua(), timeout=10).json()
        pages = (js.get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), {})
        props = page.get("pageprops") or {}
        return "disambiguation" in props
    except Exception:
        return False

def _looks_like_company(title: str, html: Optional[str]) -> bool:
    low = (title or "").lower()
    if any(tok in low for tok in _COMPANY_TITLE_TOKENS):
        return True
    if not html:
        return False
    # has infobox and contains several typical fields
    if "table" in html and "infobox" in html:
        score = 0
        for kw in ("Headquarters", "Industry", "Traded as", "Type"):
            if kw.lower() in html.lower():
                score += 1
        if score >= 2:
            return True
    return False

def wiki_fetch(company: str) -> Tuple[Optional[str], Optional[str]]:
    key = (company or "").strip().lower()

    # 0) override some high frequency same name companies
    if key in _EXTRA_WIKI_OVERRIDES:
        url = _EXTRA_WIKI_OVERRIDES[key]
        html = _get(url)
        if html and "Wikipedia" in html:
            return url, html

    # 1) use search API, strictly filter/sort
    try:
        q = urllib.parse.quote(company)
        api = ("https://en.wikipedia.org/w/api.php"
               f"?action=query&list=search&srsearch={q}&format=json&utf8=1&srlimit=10&srnamespace=0")
        js = _ses.get(api, headers=_ua(), timeout=10).json()
        hits = (js.get("query") or {}).get("search") or []
        titles = [h.get("title", "") for h in hits if h.get("title")]
        
        # skip disambiguation; prioritize company-related clues; prioritize titles containing the original query substring; shorter is better
        filtered = [t for t in titles if not _wiki_is_disambiguation(t)]
        def rank_key(t: str) -> tuple:
            tl = t.lower()
            return (
                0 if key in tl else 1,
                0 if any(tok in tl for tok in _COMPANY_TITLE_TOKENS) else 1,
                len(tl),
            )
        filtered.sort(key=rank_key)

        for title in filtered:
            url = _wiki_get_fullurl(title)
            if not url:
                continue
            html = _get(url)
            if html and "Wikipedia" in html and _looks_like_company(title, html):
                return url, html
    except Exception:
        pass

    # 2) manual fallback candidates (add common company suffixes)
    c = (company or "").replace(" ", "_")
    candidates = [
        f"https://en.wikipedia.org/wiki/{c}",
        f"https://en.wikipedia.org/wiki/{c}_Inc.",
        f"https://en.wikipedia.org/wiki/{c},_Inc.",
        f"https://en.wikipedia.org/wiki/{c}_LLC",
        f"https://en.wikipedia.org/wiki/{c}_(company)",
        f"https://en.wikipedia.org/wiki/{c}_Corporation",
        f"https://en.m.wikipedia.org/wiki/{c}",
    ]
    if key in _EXTRA_WIKI_OVERRIDES:
        candidates.insert(0, _EXTRA_WIKI_OVERRIDES[key])

    for url in candidates:
        html = _get(url)
        if html and "Wikipedia" in html and _looks_like_company(url.rsplit("/", 1)[-1], html):
            return url, html

    return None, None


# ---------- parse infobox ----------
def _parse_wikipedia_infobox(html: str) -> dict:
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            soup = BeautifulSoup(html, parser)
            box = soup.select_one("table.infobox")
            if not box:
                continue
            data = {}
            for tr in box.select("tr"):
                th = tr.find("th")
                td = tr.find("td")
                if not th or not td:
                    continue
                key = th.get_text(" ", strip=True).lower()
                val = td.get_text(" ", strip=True)
                if key:
                    data[key] = val
            if data:
                return data
        except Exception:
            continue
    return {}


# ---------- normalize & regex ----------
_STATES = {
    # all 50 states + DC
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO",
    "connecticut":"CT","delaware":"DE","district of columbia":"DC","florida":"FL","georgia":"GA",
    "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY",
    "louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN",
    "mississippi":"MS","missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH",
    "new jersey":"NJ","new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND",
    "ohio":"OH","oklahoma":"OK","oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC",
    "south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA",
    "washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY",
}

def _normalize_hq(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    support:
      - '1 Apple Park Way, Cupertino, California, U.S.'
      - '1515 3rd St, San Francisco, CA, US'
      - 'Denver, Colorado, U.S.'
      - 'Googleplex, Mountain View, California, U.S.'
    return (City, US state abbr); non-US return (City, None)
    """
    if not s:
        return (None, None)

    COUNTRY_TOKENS = {
        "us","u.s","u.s.","usa","u.s.a","u.s.a.","united states","united states of america"
    }
    s = re.sub(r"(?i)\bformerly\b.*", "", s)
    s = re.sub(r"\([^)]*\)", "", s)              # remove (2024) etc.
    s = re.sub(r"(?i)\bin\b\s+", "", s)
    s = s.replace("\xa0", " ").strip()

    # only take the first location fragment (when encountering • ; | newline etc.)
    first = re.split(r"[•;|\n]", s, maxsplit=1)[0].strip()
    if not first:
        return (None, None)

    # split by comma, and clean whitespace
    parts = [p.strip() for p in first.split(",") if p.strip()]
    if not parts:
        return (None, None)

    # remove the ending country
    while parts and parts[-1].lower().replace(".", "") in COUNTRY_TOKENS:
        parts.pop()

    if not parts:
        return (None, None)

    def _is_state_token(tok: str) -> Optional[str]:
        t = tok.replace(".", "").strip()
        if len(t) == 2 and t.isalpha():
            return t.upper()
        low = t.lower()
        return _STATES.get(low)

    # find US state (two letters or full name) from right to left, take the left part of the city; skip street/park names
    street_tokens = ("way","street","st","road","rd","ave","avenue","blvd","boulevard","pkwy","parkway","dr","drive")
    for i in range(len(parts)-1, -1, -1):
        st = _is_state_token(parts[i])
        if st:
            city_idx = i - 1
            while city_idx >= 0:
                tok = parts[city_idx].lower()
                if re.search(r"\d", tok) or any(w in tok for w in street_tokens) or "plex" in tok:
                    city_idx -= 1
                else:
                    break
            if city_idx >= 0:
                return (parts[city_idx], st)
            else:
                return (None, st)

    # no US state, return the last "like city" part
    return (parts[-1] if parts else None, None)

def _align_sector_to_training(sector: Optional[str]) -> str:
    """
    Ensure the sector string is one of the TRAIN labels.
    If not, try mild normalization (& -> 'and'); if still not, return '' (ignored by OHE).
    """
    if not sector or not isinstance(sector, str):
        return ""
    s = sector.strip()
    if s in _TRAIN_SECTORS:
        return s
    s2 = s.replace("&", "and")
    if s2 in _TRAIN_SECTORS:
        return s2
    return ""  # safe fallback

def _normalize_sector(industry_text: str) -> str:
    """
    1) Scan industry text for keyword hits (case-insensitive, first match wins).
    2) Map to TRAIN sector and hard-validate via _align_sector_to_training().
    3) If nothing hits, return '' (ignored downstream).
    """
    if not industry_text:
        return ""
    t = industry_text.lower()
    for kw, train_label in _SECTOR_MAP.items():
        if kw in t:
            return _align_sector_to_training(train_label)
    return ""


_EMP_RES = [
    re.compile(r"\bNumber of employees\s*[:\-]?\s*([\d,\.]+(?:\s*(?:million|thousand|k|m))?)", re.I),
    re.compile(r"\b([\d,\.]+\s*(?:million|thousand|k|m))\s+employees\b", re.I),
    re.compile(r"\b(\d[\d,\.]{2,})\s+(?:employees|staff)\b", re.I),
]

def _k_m_to_int(s: str) -> Optional[int]:
    try:
        t = s.lower().replace(",", "").strip()
        if "million" in t or t.endswith("m"):
            return int(float(re.sub(r"[^\d\.]", "", t)) * 1_000_000)
        if "thousand" in t or t.endswith("k"):
            return int(float(re.sub(r"[^\d\.]", "", t)) * 1_000)
        return int(re.sub(r"[^\d]", "", t))
    except Exception:
        return None

def _clean_paren_year(s: str) -> str:
    # convert "12,345 (2024)" -> "12,345"
    return re.sub(r"\([^)]*\)", "", s)

_WEBSITE_RE = re.compile(r"([a-z0-9][a-z0-9\-\._]*\.[a-z]{2,})(?:/[^\s]*)?", re.I)
def _normalize_website(s: str) -> Optional[str]:
    if not s:
        return None
    t = re.sub(r"\s*\.\s*", ".", s)  # remove spaces between domain points
    t = re.sub(r"\s+", "", t)        # remove remaining spaces
    m = _WEBSITE_RE.search(t)
    if m:
        return m.group(1).lower().lstrip("www.")
    return None

def _extract_domains_from_html_cell(html_cell: str) -> List[str]:
    try:
        soup = BeautifulSoup(html_cell, "html.parser")
        domains = []
        seen = set()
        for a in soup.select("a[href]"):
            href = a["href"]
            if href.startswith("//"):
                href = "https:" + href
            if href.startswith("http"):
                host = urlparse(href).netloc.lower()
                if host.startswith("www."):
                    host = host[4:]
                if host and host not in seen:
                    domains.append(host); seen.add(host)
        return domains
    except Exception:
        return []

def _pick_best_website(domains: List[str]) -> Optional[str]:
    """score multiple domains, prefer main site (.com), penalize IR subdomains"""
    if not domains:
        return None
    def score(d: str) -> tuple:
        penalty = 0
        if d.startswith(("ir.", "investor.", "investors.")):
            penalty += 2
        tld_bonus = 0 if d.endswith(".com") else 1
        return (penalty, tld_bonus, len(d))
    domains_sorted = sorted(domains, key=score)
    return domains_sorted[0]


# ---------- extract from infobox (priority path) ----------
def _extract_from_infobox(html: str) -> dict:
    info = _parse_wikipedia_infobox(html)
    if not info:
        return {}

    out: Dict[str, Any] = {}

    emp = info.get("number of employees") or info.get("employees")
    if emp:
        raw = _clean_paren_year(emp)
        raw = re.sub(r"\[\d+\]", "", raw)  # remove footnote [1]
        cands = re.findall(r"(\d[\d,\.]*\s*(?:million|thousand|k|m)?)", raw, flags=re.I)
        nums = []
        for c in cands:
            n = _k_m_to_int(c)
            if n and n > 0:
                nums.append(n)
        if nums:
            n = max(nums)
            if n >= 500:
                out["employees"] = n

    # founded/formation/established (keep the earliest year)
    founded = info.get("founded") or info.get("formation") or info.get("established")
    if founded:
        years = [int(y) for y in re.findall(r"(\d{4})", founded) if 1800 <= int(y) <= dt.datetime.now().year]
        if years:
            out["founded"] = min(years)

    # HQ
    for k in ("headquarters", "headquarters location", "head office", "headquarters address"):
        if info.get(k):
            city, st = _normalize_hq(info[k])
            if city: out["hq_city"] = city
            if st:   out["hq_state"] = st
            if out.get("hq_city") or out.get("hq_state"):
                break

    # ownership: type/traded as
    type_field = info.get("type") or ""
    traded_as  = info.get("traded as") or ""
    if re.search(r"\bpublic\b", type_field, re.I) or traded_as:
        out["ownership"] = "Company - Public"
    elif re.search(r"\bprivate\b", type_field, re.I):
        out["ownership"] = "Company - Private"

    # sector from industry
    industry = info.get("industry") or ""
    sector = _normalize_sector(industry)
    if sector:
        out["sector"] = sector

    # website (prefer the best domain from <a href>; fallback to text cleaning)
    website_cell = info.get("website") or info.get("web site")
    if website_cell:
        try:
            soup = BeautifulSoup(html, "html.parser")
            box = soup.select_one("table.infobox")
            if box:
                for tr in box.select("tr"):
                    th = tr.find("th")
                    td = tr.find("td")
                    if not th or not td:
                        continue
                    key = th.get_text(" ", strip=True).lower()
                    if key in ("website","web site"):
                        domains = _extract_domains_from_html_cell(str(td))
                        best = _pick_best_website(domains)
                        if best:
                            out["_website"] = best
                            break
        except Exception:
            pass
        if "_website" not in out:
            norm = _normalize_website(website_cell)
            if norm:
                out["_website"] = norm

    return out


# ---------- website/IR guess ----------
def site_guess_fetch(company: str) -> List[Tuple[str, str]]:
    """based on common subdomains; return (url, html) list (in order of hit)"""
    base = company.lower().replace(" ", "")
    domains = [
        f"https://www.{base}.com",
        f"https://{base}.com",
        f"https://ir.{base}.com",
        f"https://investor.{base}.com",
        f"https://investors.{base}.com",
        f"https://www.{base}.com/investors",
        f"https://www.{base}.com/about",
        f"https://about.{base}.com",
    ]
    out: List[Tuple[str, str]] = []
    for u in domains:
        html = _get(u)
        if html:
            out.append((u, html))
    return out


# ---------- DDG ----------
def search_web(query: str, max_results: int = 6) -> List[Source]:
    if not HAVE_DDG:
        return []
    out: List[Source] = []
    backoff = 0.8
    for _ in range(3):
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):
                    out.append(Source(url=r.get("href") or r.get("url") or "", title=r.get("title"), snippet=r.get("body")))
            break
        except Exception:
            time.sleep(backoff + random.random() * 0.5)
            backoff *= 1.6
    # 去重
    seen, uniq = set(), []
    for s in out:
        if s.url and s.url not in seen:
            uniq.append(s); seen.add(s.url)
    return uniq


# ---------- text fallback extraction (only when infobox is not available) ----------
_FOUNDED_RE_BODY = re.compile(r"\b(Founded|Formation|Established)[^0-9]{0,12}(\d{4})\b", re.I)
_HQ_RE_BODY = re.compile(r"\b([A-Z][A-Za-z .'\-]+?),\s*([A-Za-z]{2,})(?:\b|$)")

def heuristic_extract(text: str) -> Dict[str, Any]:
    got: Dict[str, Any] = {}

    # employees
    for rx in _EMP_RES:
        m = rx.search(text)
        if m:
            v = _k_m_to_int(m.group(1))
            if v and v > 0:
                got["employees"] = v
                break

    # founded
    m = _FOUNDED_RE_BODY.search(text)
    if m:
        try:
            y = int(m.group(2))
            if 1800 <= y <= dt.datetime.now().year:
                got["founded"] = y
        except Exception:
            pass
        
    if "headquarters" in text.lower():
        m = _HQ_RE_BODY.search(text)
        if m:
            city, st = _normalize_hq(f"{m.group(1)}, {m.group(2)}")
            if city: got["hq_city"] = city
            if st:   got["hq_state"] = st

    if re.search(r"\b(NASDAQ|NYSE|Ticker|public company|listed)\b", text, re.I):
        got["ownership"] = "Company - Public"
    elif re.search(r"\bprivate company|privately held\b", text, re.I):
        got["ownership"] = "Company - Private"

    return got


# ---------- size/age ----------
def employees_to_band(n: Optional[int]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if n is None:
        return (None, None, None)
    lo, hi = max(1, int(n * 0.8)), int(n * 1.2)
    label = "Small" if n < 50 else "Mid" if n < 200 else "Large" if n < 1000 else "XL" if n < 10_000 else "Enterprise"
    return lo, hi, label

def compute_age(founded: Optional[int]) -> Optional[int]:
    if not founded:
        return None
    try:
        y = int(founded)
        now = dt.datetime.now().year
        if 1800 <= y <= now:
            return now - y
    except Exception:
        return None
    return None


# ---------- aggregate entry ----------
@timing_decorator
@cached(ttl_hours=24)
def fetch_company_profile_fast(company: str) -> Tuple[Dict[str, Any], List[Source]]:
    sources: List[Source] = []

    # 1) Wikipedia (prioritize infobox)
    wurl, whtml = wiki_fetch(company)
    if whtml:
        sources.append(Source(url=wurl or "", title="Wikipedia"))
        got = _extract_from_infobox(whtml)

        # if missing key fields, supplement with text extraction
        need_body = (not got) or any(k not in got for k in ("employees", "sector", "ownership"))
        if need_body:
            wtxt = _html_to_text(whtml)
            body = heuristic_extract(wtxt)
            for k in ("employees", "founded", "hq_city", "hq_state", "ownership"):
                if k not in got and k in body:
                    got[k] = body[k]
            # sector again normalize (simple from text extraction)
            if "sector" not in got:
                for key, val in _SECTOR_MAP.items():
                    if key in wtxt.lower():
                        got["sector"] = val
                        break

        if got.get("employees"):
            lo, hi, lbl = employees_to_band(got["employees"])
            got.update({"min_size": lo, "max_size": hi, "size_label": lbl})
        if got:
            got["_diagnostics"] = {"path": "wiki", "url": wurl}
            return got, sources

    # 2) website/IR guess (first hit wiki page)
    for url, html in site_guess_fetch(company):
        sources.append(Source(url=url, title="site"))
        txt = _html_to_text(html)
        got = heuristic_extract(txt)
        if got:
            if got.get("employees"):
                lo, hi, lbl = employees_to_band(got["employees"])
                got.update({"min_size": lo, "max_size": hi, "size_label": lbl})
            got["_diagnostics"] = {"path": "site_guess", "url": url}
            return got, sources

    # 3) DDG fallback (rate limiting backoff)
    hits = search_web(f"{company} number of employees headquarters founded investor relations wikipedia", max_results=6)
    if not hits and not HAVE_DDG:
        return {"_diagnostics": {"path": "none", "reason": "DDG disabled"}}, sources

    for h in hits:
        html = _get(h.url)
        if not html:
            continue
        sources.append(h)
        txt = _html_to_text(html)
        got = heuristic_extract(txt)
        if got:
            if got.get("employees"):
                lo, hi, lbl = employees_to_band(got["employees"])
                got.update({"min_size": lo, "max_size": hi, "size_label": lbl})
            got["_diagnostics"] = {"path": "ddg", "url": h.url}
            return got, sources

    return {"_diagnostics": {"path": "none", "reason": "all paths empty"}}, sources
