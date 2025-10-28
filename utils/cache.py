# cache.py - 缓存管理
import time
import json
import hashlib
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .config import config
from .logger import logger

class TTLCache:
    """带过期时间的缓存"""
    
    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # 检查是否过期
        if isinstance(timestamp, float):
            # 如果是时间戳，转换为datetime
            timestamp = datetime.fromtimestamp(timestamp)
        
        if datetime.now() - timestamp > self.ttl:
            del self.cache[key]
            logger.debug(f"Cache expired for key: {key}")
            return None
        
        logger.debug(f"Cache hit for key: {key}")
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        self.cache[key] = (value, datetime.now())
        logger.debug(f"Cache set for key: {key}")
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """清理过期缓存，返回清理的数量"""
        now = datetime.now()
        expired_keys = []
        
        for key, (_, timestamp) in self.cache.items():
            if isinstance(timestamp, float):
                timestamp = datetime.fromtimestamp(timestamp)
            
            if now - timestamp > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self.cache)

# 全局缓存实例
cache = TTLCache(ttl_hours=config.CACHE_TTL_HOURS) if config.ENABLE_CACHE else None

def get_cache_key(*args, **kwargs) -> str:
    """生成缓存键"""
    # 将参数序列化为字符串
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    
    # 生成哈希值
    return hashlib.md5(key_str.encode()).hexdigest()

def cached(ttl_hours: Optional[int] = None):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not config.ENABLE_CACHE or cache is None:
                return func(*args, **kwargs)
            
            # 生成缓存键
            key = get_cache_key(func.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator
