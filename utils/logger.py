# utils/logger.py — Logging utilities
from __future__ import annotations
import logging
import sys
import time
import functools
from typing import Optional, Callable, Any

# Make tests able to patch this name: utils.logger.config
from .config import config  # NOTE: tests patch with patch('utils.logger.config')

def _safe_level(level_str: Optional[str]) -> int:
    """
    Safely convert a string log level to the corresponding logging numeric level.
    Falls back to INFO on invalid input (matches tests’ expectation).
    """
    if not level_str:
        return logging.INFO
    try:
        lvl = getattr(logging, str(level_str).upper())
        if isinstance(lvl, int):
            return lvl
    except Exception:
        pass
    return logging.INFO


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Idempotently configure a logger: avoid adding duplicate handlers and
    update existing handlers’ format and level if they already exist.
    """
    logger = logging.getLogger(name)

    # Resolve level (prefer function argument, then config.LOG_LEVEL, then INFO)
    log_level = _safe_level(level or getattr(config, "LOG_LEVEL", "INFO"))
    fmt = getattr(config, "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.setLevel(log_level)
    logger.propagate = False  # tests assert no propagation to parent logger

    formatter = logging.Formatter(fmt)

    if logger.handlers:
        # If handlers already exist: do not add new ones, but **update** existing
        # handlers’ level and formatter (tests check for this behavior).
        for h in logger.handlers:
            h.setLevel(log_level)
            # Only update formatter for stream handlers to keep parity with our setup
            if isinstance(h, logging.StreamHandler):
                h.setFormatter(formatter)
        return logger

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Create a default logger (used by tests.TestLoggerIntegration::test_logger_import)
# The module name here is "utils.logger"
logger = setup_logger(__name__)

app_logger = logger
__all__ = [
    "setup_logger",
    "timing_decorator",
    "retry_decorator",
    "logger",
    "app_logger",
    "config",
]


# -------------------------
# Performance timing decorator (tests reference the name timing_decorator)
# -------------------------
def timing_decorator(func: Callable) -> Callable:
    """
    Measure execution time. Tests may patch config / logger / time.time to verify.
    Logs an info-level line with the elapsed time. If MAX_PREDICTION_TIME is set
    and the function name is predict_point_range or serve, warn when exceeded.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if not getattr(config, "ENABLE_PERFORMANCE_MONITORING", True):
            return func(*args, **kwargs)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            # Info-level message; tests assert text like "took 0.500 seconds"
            logger.info(f"{func.__name__} took {duration:.3f} seconds")

            # If duration exceeds the configured max for prediction paths, warn
            max_pred = getattr(config, "MAX_PREDICTION_TIME", None)
            if max_pred is not None and func.__name__ in ("predict_point_range", "serve") and duration > max_pred:
                logger.warning(
                    f"{func.__name__} exceeded maximum time limit: "
                    f"{duration:.3f}s > {max_pred}s"
                )
    return wrapper


# -------------------------
# Retry decorator (tests reference the name retry_decorator)
# -------------------------
def retry_decorator(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry wrapper:
      - When max_retries == 0, fail immediately after the first attempt and log error (tested).
      - When max_retries < 0, the for-loop will not run; last_exception stays None
        and raising it yields TypeError (exactly what tests expect).
      - Backoff schedule: delay * (backoff ** attempt), with attempt starting at 0 (tested).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # For negative retries: range(max_retries + 1) is empty; last_exception
            # remains None; raising it produces a TypeError (as tests expect).
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # Let tests capture all kinds of exceptions
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )

            # Matches tests’ expectation for negative retry scenario: raise None -> TypeError
            raise last_exception

        return wrapper
    return decorator
