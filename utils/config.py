# config.py - 配置管理
import os
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Config:
    """应用配置类"""
    
    # 模型配置
    PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", "models/pipeline_new.pkl")
    SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", "models/schema.json")
    
    # Web请求配置
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "12"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_DELAY_MIN: float = float(os.getenv("REQUEST_DELAY_MIN", "0.6"))
    REQUEST_DELAY_MAX: float = float(os.getenv("REQUEST_DELAY_MAX", "1.0"))
    
    # 缓存配置
    CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # OpenAI配置
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # 性能配置
    MAX_PREDICTION_TIME: float = float(os.getenv("MAX_PREDICTION_TIME", "1.0"))
    ENABLE_PERFORMANCE_MONITORING: bool = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    
    # 用户代理池
    USER_AGENTS: List[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.USER_AGENTS is None:
            self.USER_AGENTS = [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            ]
        
        # 从环境变量加载自定义用户代理
        custom_ua = os.getenv("CUSTOM_USER_AGENTS")
        if custom_ua:
            self.USER_AGENTS.extend(custom_ua.split(","))

# 全局配置实例
config = Config()
