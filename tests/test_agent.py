# tests/test_agent.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import pytest
from agent import CompanyAgent

@pytest.mark.parametrize("name", [
    "Datadog",
    "DoorDash",
    "Palantir Technologies",
    "Snowflake",
    "Stripe",
    "Uber",
    "Airbnb",
    "Netflix",
    "Amazon",
    "Apple",
    "Google",
    "Microsoft",
    "Meta"
])

def test_company_agent_minimum_fields(name):
    ag = CompanyAgent()
    prof = ag.lookup(name) or {}
    sources = prof.get("__sources__", [])

    # 打印结果
    print(f"\n=== {name} ===")
    print("Profile:")
    for k, v in prof.items():
        if k != "__sources__":
            print(f"  {k}: {v}")
    print("Sources:")
    for s in sources[:5]:
        print(f"  - {s.get('title') or s.get('url')}: {s.get('url')}")

    # 断言
    assert isinstance(prof, dict)
    assert isinstance(sources, list)
    assert any([
        bool(prof.get("Type of ownership")),
        bool(prof.get("Sector")),
        bool(prof.get("company_age")),
        (prof.get("min_size") is not None and prof.get("max_size") is not None),
    ]), f"agent failed to extract core fields for {name}; profile={prof}, sources={sources[:3]}"