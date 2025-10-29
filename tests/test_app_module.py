# tests/test_app_module.py
import builtins
import os
import sys
from types import SimpleNamespace, ModuleType
import types
import pytest

@pytest.fixture(autouse=True)
def stub_modules(monkeypatch):
    import sys, re
    from types import ModuleType, SimpleNamespace
    
    added_or_replaced = {}
    def _setmod(name, module):
        # Remember the previous module (might be None)
        prev = sys.modules.get(name, None)
        added_or_replaced[name] = prev
        sys.modules[name] = module
        return module
    
    utils_pkg = ModuleType("utils")
    utils_pkg.__path__ = []      # Mark as a package
    _setmod("utils", utils_pkg)

    # ---- fake gradio ----
    gr = ModuleType("gradio")
    class _Update(dict): pass
    def update(**kw): return _Update(**kw)
    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    class _Comp:
        def __init__(self, *a, **k): pass
        def change(self, *a, **k): return None
        def click(self, *a, **k): return None
    gr.update = update
    gr.Blocks = _Ctx; gr.Tabs = _Ctx; gr.Tab = _Ctx; gr.Row = _Ctx
    gr.Markdown = _Comp; gr.Textbox = _Comp; gr.Dropdown = _Comp
    gr.Number = _Comp; gr.Checkbox = _Comp; gr.Button = _Comp; gr.JSON = _Comp
    _setmod("gradio", gr)

    # ---- fake utils.helpers ----
    helpers = ModuleType("utils.helpers")
    def titlecase(s): return s.title() if isinstance(s, str) else s
    def clean_text(s): return (s or "")
    def strip_paren_noise(s): return s.split("(")[0].strip()
    def looks_like_location(s): return isinstance(s, str) and ("," in s) and (len(s.split(","))==2)
    def looks_like_noise_line(s): return False
    def fmt_none(v): return "—" if v in (None, "", []) else str(v)
    helpers.titlecase = titlecase
    helpers.clean_text = clean_text
    helpers.strip_paren_noise = strip_paren_noise
    helpers.looks_like_location = looks_like_location
    helpers.looks_like_noise_line = looks_like_noise_line
    helpers.fmt_none = fmt_none
    _setmod("utils.helpers", helpers)
    setattr(utils_pkg, "helpers", helpers)   # ✅ Attach to the fake utils package

    # ---- fake utils.us_locations ----
    usloc = ModuleType("utils.us_locations")
    usloc.US_STATES = ["CA","NY","TX"]
    usloc.STATE_TO_CITIES = {
        "CA": ["San Jose","San Francisco","Los Angeles"],
        "NY": ["New York","Buffalo"],
        "TX": ["Austin","Dallas"],
    }
    _setmod("utils.us_locations", usloc)
    setattr(utils_pkg, "us_locations", usloc)   # ✅

    # ---- fake utils.jd_parsing ----
    jd = ModuleType("utils.jd_parsing")
    jd.CITY_REGEX = re.compile(r"^[A-Za-z][A-Za-z\s\-']+$")
    def parse_jd(txt):
        # If the marker appears, return the location we want
        if "HAS_LOC" in (txt or ""):
            return {"location": "San Jose, CA"}
        return {"location": ""}
    jd.parse_jd = parse_jd
    jd.NOISE_LINE_HINTS = [   # ✅ If real helpers accidentally loads, it won't crash
        "equal opportunity","benefits","background check","work authorization",
        "eeo","vaccination","employer branding","about us","culture"
    ]
    _setmod("utils.jd_parsing", jd)
    setattr(utils_pkg, "jd_parsing", jd)       # ✅

    # ---- fake utils.featurizers / config ----
    feat = ModuleType("utils.featurizers")
    _setmod("utils.featurizers", feat)
    setattr(utils_pkg, "featurizers", feat)    # ✅

    consts = ModuleType("utils.config")
    consts.PIPELINE_PATH = "models/pipeline_new.pkl"
    consts.OPENAI_MODEL = "gpt-4o-mini"
    consts.SIZE_BANDS = ["Small","Mid","Large","XL","Enterprise"]
    _setmod("utils.config", consts)
    setattr(utils_pkg, "config", consts)    # ✅

    # ---- fake agent.CompanyAgent ----
    agent = ModuleType("agent")
    class CompanyAgent:
        def lookup(self, name):
            return {
                "sector": "Information Technology",
                "ownership": "Company - Private",
                "size_band": "Mid",
                "company_age": 12,
                "hq_city": "San Jose",
                "hq_state": "CA",
                "__sources__": [{"url":"https://x.test"}],
            }
    agent.CompanyAgent = CompanyAgent
    _setmod("agent", agent)

    # ---- fake joblib.load -> DummyPipeline ----
    class DummyStep:
        def __init__(self, name): self.name = name
        def transform(self, df):
            df = df.copy()
            if self.name == "seniority":
                jt = str(df["Job Title"].iloc[0]).lower()
                df["seniority"] = "senior" if "senior" in jt else "mid"
            elif self.name == "loc_tier":
                loc = str(df["Location"].iloc[0])
                df["loc_tier"] = "high" if "San Francisco" in loc else "mid"
            return df
    class DummyPipeline:
        def __init__(self):
            self.named_steps = {"seniority": DummyStep("seniority"),
                                "loc_tier": DummyStep("loc_tier")}
        def predict(self, X): return [120000.0]
    import joblib as _joblib_real
    monkeypatch.setattr(_joblib_real, "load", lambda p: DummyPipeline(), raising=True)

    # By default do not use the online LLM
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    yield
    
    for attr in ("helpers","us_locations","jd_parsing","featurizers","config"):
        if hasattr(utils_pkg, attr):
            delattr(utils_pkg, attr)

    # Restore sys.modules one by one
    for name, prev in reversed(list(added_or_replaced.items())):
        if prev is None:
            # Modules we added: simply remove them
            sys.modules.pop(name, None)
        else:
            # Modules we overwrote: restore the original object
            sys.modules[name] = prev

# It is now safe to import app
@pytest.fixture(scope="module")
def app_module():
    import importlib
    app = importlib.import_module("app.app")
    return app


# ---------- 2) Override utility functions (map/coerce/choose etc.) ----------
def test_sector_mapping_direct_and_canon(app_module):
    assert app_module.map_sector_to_training("Information Technology") == "Information Technology"
    assert app_module.map_sector_to_training("it") == "Information Technology"
    assert app_module.map_sector_to_training("Unknown Sector") == "Unknown Sector"
    assert app_module.map_sector_to_training(None) is None

def test_size_band_coercion_and_employees(app_module):
    c = app_module.coerce_size_label_to_band
    e = app_module.employees_to_size_band
    assert c("XL") == "XL"
    assert c("") is None
    assert e(49) == "Small"
    assert e(199) == "Mid"
    assert e(999) == "Large"
    assert e(9999) == "XL"
    assert e(12000) == "Enterprise"
    assert e("oops") is None

def test_choose_size_band_priority(app_module):
    choose = app_module.choose_size_band_from_profile
    assert choose({"size_band": "Mid"}) == "Mid"
    assert choose({"Size": "Large"}) == "Large"
    assert choose({"size_label": "XL"}) == "XL"
    assert choose({"employees": 80}) == "Mid"
    # Fallback on min/max
    assert choose({"min_size": 400, "max_size": 600}) == "Large"
    assert choose({}) is None


# ---------- 3) _to_model_row_from_ui ----------
def test_to_model_row_from_ui_happy(app_module):
    row = app_module._to_model_row_from_ui(
        job_title="Senior Data Scientist",
        location="San Francisco, CA",
        rating=4.2,
        age=10,
        sector="it",
        type_of_ownership="Company - Public",
        size_band="XL",
    )
    # Order is determined by the schema
    assert row["Job Title"] == "Senior Data Scientist"
    assert row["Location"] == "San Francisco, CA"
    assert row["Sector"] == "Information Technology"
    assert row["size_band"] == "XL"


# ---------- 4) serve() multi-branch ----------
def test_serve_parse_title_and_jd_location(app_module, monkeypatch):
    # 1) Force llm_explain to take the offline path
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "client", None)

    # 2) Key: stub location parsing & resources to avoid nondeterminism from import/source differences
    import re
    monkeypatch.setattr(app_module, "try_parse_location_from_jd", lambda txt: ("San Jose", "CA"))
    monkeypatch.setattr(app_module, "CITY_REGEX", re.compile(r"^[A-Za-z][A-Za-z\s\-']+$"), raising=False)
    monkeypatch.setattr(app_module, "US_STATES", ["CA", "NY", "TX"], raising=False)
    monkeypatch.setattr(app_module, "STATE_TO_CITIES",
                        {"CA": ["San Jose", "San Francisco", "Los Angeles"],
                         "NY": ["New York", "Buffalo"],
                         "TX": ["Austin", "Dallas"]},
                        raising=False)

    # Leave title/state/city blank; rely on our stubbed JD parsing
    result, *_updates = app_module.serve(
        job_title="",
        state_abbrev="",
        city="",
        rating=4.0,
        age=5,
        sector="Information Technology",
        type_of_ownership="Company - Private",
        size_band="Mid",
        job_description_text="HAS_LOC We are hiring a Senior Machine Learning Engineer.",
        company_name="Snowflake",
        use_agent_flag=True,
        overwrite_defaults=True
    )

    assert "error" not in result
    derived = result["Derived features (from pipeline)"]
    assert derived["seniority"] in ("senior", "mid")
    assert derived["loc_tier"] in ("very_high", "high", "mid")

def test_serve_invalid_rating(app_module):
    result, *_ = app_module.serve(
        job_title="Software Engineer",
        state_abbrev="CA",
        city="San Jose",
        rating="bad",  # Invalid
        age=3,
        sector="IT",
        type_of_ownership="Company - Private",
        size_band="Mid",
        job_description_text="",
        company_name="",
        use_agent_flag=False,
        overwrite_defaults=True
    )
    assert "error" in result and "must be a number" in result["error"]

def test_serve_hq_location_fallback(app_module, monkeypatch):
    # 1) Stabilize location resources & agent & disable online LLM
    import re
    monkeypatch.setattr(app_module, "CITY_REGEX", re.compile(r"^[A-Za-z][A-Za-z\s\-']+$"), raising=False)
    monkeypatch.setattr(app_module, "US_STATES", ["CA", "NY", "TX"], raising=False)
    monkeypatch.setattr(app_module, "STATE_TO_CITIES",
                        {"CA": ["San Jose", "San Francisco", "Los Angeles"],
                         "NY": ["New York", "Buffalo"],
                         "TX": ["Austin", "Dallas"]},
                        raising=False)

    class _FakeAgent:
        def lookup(self, name):
            return {
                "sector": "Information Technology",
                "ownership": "Company - Private",
                "size_band": "Mid",
                "company_age": 12,
                "hq_city": "San Jose",
                "hq_state": "CA",
                "__sources__": [{"url":"https://x.test"}],
            }
    monkeypatch.setattr(app_module, "_agent", _FakeAgent(), raising=False)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "client", None, raising=False)

    # 2) Enter the HQ autofill path: no state/city provided; JD doesn’t include location either
    result, *_ = app_module.serve(
        job_title="Senior Backend Engineer",
        state_abbrev="", city="",
        rating=4.5, age=8,
        sector="", type_of_ownership="", size_band="",
        job_description_text="no HAS_LOC here",    # Do not trigger JD parsing
        company_name="Netflix",                   # Any name; agent is stubbed
        use_agent_flag=True, overwrite_defaults=True
    )

    assert "error" not in result
    assert result["Inputs used by the model"]["Location"] == "San Jose, CA"

def test_serve_overwrite_defaults_false(app_module):
    # Agent would return IT/Mid/12, but we set overwrite_defaults=False and pass existing values
    result, *_ = app_module.serve(
        job_title="Senior Engineer",
        state_abbrev="CA", city="San Jose",
        rating=4.0, age=7,
        sector="Media", type_of_ownership="Company - Public", size_band="XL",
        job_description_text="HAS_LOC Senior Engineer",
        company_name="ACME", use_agent_flag=True, overwrite_defaults=False
    )
    row = result["Inputs used by the model"]
    # Should not be overwritten by the agent
    assert row["Sector"] == "Media"
    assert row["Type of ownership"] == "Company - Public"
    assert row["size_band"] == "XL"
    assert row["age"] == 7.0  # Should also not be overwritten by 12

def test_serve_job_title_looks_like_location(app_module):
    result, *_ = app_module.serve(
        job_title="San Jose, CA",  # Looks like a location
        state_abbrev="CA", city="San Jose",
        rating=3.0, age=2,
        sector="", type_of_ownership="", size_band="",
        job_description_text="",
        company_name="", use_agent_flag=False, overwrite_defaults=True
    )
    assert "error" in result and "looks like a location" in result["error"]


# ---------- 5) llm_explain path where client exists but raises an error ----------
def test_llm_explain_with_client_error_path(app_module, monkeypatch):
    # Make app_module.client exist and raise; set OPENAI_API_KEY to hit this branch
    class BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("boom")
    monkeypatch.setattr(app_module, "client", BoomClient())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    text = app_module.llm_explain(
        context={"Job Title":"SE","Location":"San Jose, CA","Rating":4,"age":3,
                 "Sector":"IT","Type of ownership":"Public","size_band":"Mid"},
        derived={"seniority":"mid","loc_tier":"mid"},
        point=120000, low=100000, high=140000
    )
    assert "Explanation unavailable" in text


# ---------- 6) update_city_choices ----------
def test_update_city_choices(app_module):
    upd = app_module.update_city_choices("CA")
    # Our fake gr.update returns a dict-like object
    assert isinstance(upd, dict)
    assert "choices" in upd and "San Jose" in upd["choices"]
