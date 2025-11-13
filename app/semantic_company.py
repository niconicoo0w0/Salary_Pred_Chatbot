# app/semantic_company.py
import re
from typing import Optional

from sentence_transformers import SentenceTransformer, util
import torch
from utils import us_locations  # US_STATES, STATE_TO_CITIES

US_LOCATION_WORDS = set(us_locations.US_STATES)
for cities in us_locations.STATE_TO_CITIES.values():
    for city in cities:
        US_LOCATION_WORDS.add(city.lower())


def light_clean_company(text: str) -> str:
    if not text:
        return ""
    s = text.strip()

    # drop things in parentheses: "Cadence (San Jose office)" -> "Cadence"
    s = re.sub(r"\((.*?)\)", "", s)

    # drop known location tokens
    for w in US_LOCATION_WORDS:
        s = re.sub(rf"\b{re.escape(w)}\b", "", s, flags=re.IGNORECASE)

    # drop "office"/"hq" tails
    s = re.sub(r"\b(office|hq|headquarters)\b", "", s, flags=re.I)

    # unify dashes
    s = re.sub(r"[-–—]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # drop trailing punctuation
    s = re.sub(r"[!?,.]+$", "", s)
    return s


class CompanySemanticMatcher:
    """Does this look like a company name at all?"""

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model = SentenceTransformer(model_name)
        templates = [
            "a US company name",
            "a US technology company",
            "a software company",
            "an organization",
        ]
        embs = self.model.encode(templates, normalize_embeddings=True)
        self.template_mean = torch.tensor(embs).mean(dim=0)

    def score_company_like(self, text: str) -> float:
        if not text:
            return 0.0
        emb = self.model.encode([text], normalize_embeddings=True)
        score = float(util.cos_sim(emb, self.template_mean))
        score = (score + 1.0) / 2.0  # [-1,1] -> [0,1]
        return max(0.0, min(1.0, score))

    def normalize(self, raw_text: str, min_score: float = 0.55) -> Optional[str]:
        cleaned = light_clean_company(raw_text)
        score = self.score_company_like(cleaned)
        if score >= min_score:
            return cleaned
        return None
