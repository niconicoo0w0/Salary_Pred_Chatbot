# -*- coding: utf-8 -*-
# config.py — Centralized application configuration manager

import os
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration settings"""

    # Model configuration
    PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", "models/pipeline_new.pkl")
    SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", "models/schema.json")
    
    # Training inputs (exactly as in training)
    NUMERIC: list[str] = None
    CATEGORICAL_BASE: list[str] = None
    RAW_INPUTS: list[str] = None

    # UI defaults / choices
    SIZE_BANDS: list[str] = None

    # OpenAI configuration
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Web request configuration
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "12"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_DELAY_MIN: float = float(os.getenv("REQUEST_DELAY_MIN", "0.6"))
    REQUEST_DELAY_MAX: float = float(os.getenv("REQUEST_DELAY_MAX", "1.0"))

    # Cache settings
    CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # OpenAI configuration
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Performance monitoring
    MAX_PREDICTION_TIME: float = float(os.getenv("MAX_PREDICTION_TIME", "1.0"))
    ENABLE_PERFORMANCE_MONITORING: bool = (
        os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    )

    # Pool of User-Agent strings for web requests
    USER_AGENTS: List[str] = None

    def __post_init__(self):
        """Post-initialization: populate defaults and apply overrides"""
        # ----- Training/Schema constants -----
        if self.NUMERIC is None:
            self.NUMERIC = ["Rating", "age"]
        if self.CATEGORICAL_BASE is None:
            self.CATEGORICAL_BASE = ["Sector", "Type of ownership", "size_band"]
        if self.RAW_INPUTS is None:
            self.RAW_INPUTS = self.NUMERIC + self.CATEGORICAL_BASE + ["Job Title", "Location"]

        # ----- UI choices -----
        if self.SIZE_BANDS is None:
            self.SIZE_BANDS = ["Small", "Mid", "Large", "XL", "Enterprise"]
            
        # Default User-Agent set
        if self.USER_AGENTS is None:
            self.USER_AGENTS = [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            ]

        # Allow adding custom User-Agents via environment variable
        custom_ua = os.getenv("CUSTOM_USER_AGENTS")
        if custom_ua:
            self.USER_AGENTS.extend(custom_ua.split(","))


# Global configuration instance (imported throughout the application)
config = Config()
