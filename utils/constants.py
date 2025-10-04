# constants.py - Application constants and configuration
import os

# Model and pipeline configuration
PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "models/pipeline.pkl")

# Training inputs (exactly as in training)
NUMERIC = ["Rating", "company_age", "min_size", "max_size"]
CATEGORICAL_BASE = ["Sector", "Type of ownership", "Size"]
RAW_INPUTS = NUMERIC + CATEGORICAL_BASE + ["Job Title", "Location"]

# OpenAI configuration
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
