# constants.py - Application constants and configuration
import os

# Model and pipeline configuration
# NOTE: point this to your new model file (pipeline with size_band)
PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "models/pipeline_new.pkl")
SCHEMA_PATH   = os.environ.get("SCHEMA_PATH", "models/schema.json")

# Training inputs (exactly as in training; keep in one place)
NUMERIC = ["Rating", "age"]
CATEGORICAL_BASE = ["Sector", "Type of ownership", "size_band"]
RAW_INPUTS = NUMERIC + CATEGORICAL_BASE + ["Job Title", "Location"]

# For UI defaults / choices
SIZE_BANDS = ["Small", "Mid", "Large", "XL", "Enterprise"]

# OpenAI configuration
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
