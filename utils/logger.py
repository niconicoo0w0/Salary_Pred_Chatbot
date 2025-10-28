# utils/logger.py - 日志记录系统
from __future__ import annotations
import logging
import sys
import time
import functools
from typing import Optional, Callable, Any

# 让 tests 能够 patch 到这个名字：utils.logger.config
from .config import config  # 注意：tests 会用 patch('utils.logger.config') 覆盖它

def _safe_level(level_str: Optional[str]) -> int:
    """
    将字符串日志级别安全地转换为 logging 的数值级别。
    无效输入时回退到 INFO（符合 tests 期望的行为）。
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
    """设置日志记录器（幂等），避免重复添加处理器；同时在已有处理器时更新其格式与级别。"""
    logger = logging.getLogger(name)

    # 解析级别（优先函数参数，其次 config.LOG_LEVEL，最后 INFO）
    log_level = _safe_level(level or getattr(config, "LOG_LEVEL", "INFO"))
    fmt = getattr(config, "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.setLevel(log_level)
    logger.propagate = False  # tests 里会检查不向父 logger 传播

    formatter = logging.Formatter(fmt)

    if logger.handlers:
        # 已有处理器：不再新增，但**更新**已有处理器的级别与格式（tests 会验这个行为）
        for h in logger.handlers:
            h.setLevel(log_level)
            # 仅当是流处理器时更新其 formatter（保持与我们创建的一致）
            if isinstance(h, logging.StreamHandler):
                h.setFormatter(formatter)
        return logger

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# 创建默认日志记录器（tests.TestLoggerIntegration::test_logger_import 会用到）
# 要求名字为 "utils.logger"
logger = setup_logger(__name__)  # __name__ 在该模块中即 "utils.logger"

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
# 性能监控装饰器（tests 对名字 timing_decorator 有引用）
# -------------------------
def timing_decorator(func: Callable) -> Callable:
    """性能计时装饰器：可被 tests 通过 patch(config / logger / time.time) 验证。"""
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
            # info 级别日志，tests 断言文本里包含 "took 0.500 seconds" 这样的格式
            logger.info(f"{func.__name__} took {duration:.3f} seconds")

            # 如果超过最大预测时间，且函数名是 predict_point_range 或 serve，则告警
            max_pred = getattr(config, "MAX_PREDICTION_TIME", None)
            if max_pred is not None and func.__name__ in ("predict_point_range", "serve") and duration > max_pred:
                logger.warning(
                    f"{func.__name__} exceeded maximum time limit: "
                    f"{duration:.3f}s > {max_pred}s"
                )
    return wrapper


# -------------------------
# 重试装饰器（tests 对名字 retry_decorator 有引用）
# -------------------------
def retry_decorator(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器：
      - 当 max_retries 为 0 时，失败立即记录 error，不重试（tests 有检查）
      - 当 max_retries 为负数时，循环体不会执行，最后 raise None 触发 TypeError（tests 有检查）
      - 退避时间：delay * (backoff ** attempt)，其中 attempt 从 0 开始（tests 有检查）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 负数重试：range(max_retries + 1) 为空，last_exception 保持 None，
            # 函数末尾 raise None -> TypeError（正是 tests 的预期）
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # 让 tests 能捕到各类异常
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

            # 与 tests 对负重试场景的预期一致：这里会 raise None -> TypeError
            raise last_exception

        return wrapper
    return decorator
