# health.py - 健康检查和性能监控
import time
import psutil
import os
from typing import Dict, Any, Optional
from datetime import datetime
from utils.config import config
from utils.logger import logger

class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.prediction_times = []
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_usage = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            }
            
            # CPU使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_usage = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
            
            # 进程信息
            process = psutil.Process(os.getpid())
            process_info = {
                "pid": process.pid,
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "create_time": process.create_time()
            }
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "memory": memory_usage,
                "cpu_percent": cpu_percent,
                "disk": disk_usage,
                "process": process_info,
                "requests": {
                    "total": self.request_count,
                    "errors": self.error_count,
                    "success_rate": 100.0 if self.request_count == 0 else (self.request_count - self.error_count) / self.request_count * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_model_health(self, pipe: Optional[Any] = None, agent: Optional[Any] = None) -> Dict[str, Any]:
        """获取模型健康状态"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        # 检查主模型
        if pipe is not None:
            try:
                # 简单的健康检查：尝试预测一个样本
                import pandas as pd
                import numpy as np
                
                sample_data = pd.DataFrame([{
                    "Rating": 4.0,
                    "age": 10,
                    "Sector": "Information Technology",
                    "Type of ownership": "Company - Public",
                    "size_band": "Mid",
                    "Job Title": "Software Engineer",
                    "Location": "San Francisco, CA"
                }])
                
                start_time = time.time()
                prediction = pipe.predict(sample_data)
                prediction_time = time.time() - start_time
                
                health["models"]["pipeline"] = {
                    "status": "healthy",
                    "prediction_time_ms": prediction_time * 1000,
                    "prediction_value": float(prediction[0]) if len(prediction) > 0 else None
                }
                
            except Exception as e:
                health["models"]["pipeline"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            health["models"]["pipeline"] = {
                "status": "not_loaded",
                "error": "Pipeline not available"
            }
        
        # 检查公司代理
        if agent is not None:
            try:
                # 简单的健康检查：尝试查找一个已知公司
                test_result = agent.lookup("Google")
                health["models"]["agent"] = {
                    "status": "healthy",
                    "cache_size": len(agent.cache),
                    "test_result_keys": list(test_result.keys()) if isinstance(test_result, dict) else []
                }
            except Exception as e:
                health["models"]["agent"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            health["models"]["agent"] = {
                "status": "not_loaded",
                "error": "Agent not available"
            }
        
        return health
    
    def record_request(self, success: bool = True, prediction_time: Optional[float] = None):
        """记录请求统计"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        if prediction_time is not None:
            self.prediction_times.append(prediction_time)
            # 只保留最近100个预测时间
            if len(self.prediction_times) > 100:
                self.prediction_times = self.prediction_times[-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.prediction_times:
            return {"prediction_times": "no_data"}
        
        prediction_times = self.prediction_times
        return {
            "prediction_times": {
                "count": len(prediction_times),
                "mean": sum(prediction_times) / len(prediction_times),
                "min": min(prediction_times),
                "max": max(prediction_times),
                "p95": sorted(prediction_times)[int(len(prediction_times) * 0.95)] if len(prediction_times) > 20 else max(prediction_times)
            }
        }

# 全局健康检查器实例
health_checker = HealthChecker()

def health_check(pipe: Optional[Any] = None, agent: Optional[Any] = None) -> Dict[str, Any]:
    """健康检查端点"""
    system_health = health_checker.get_system_health()
    model_health = health_checker.get_model_health(pipe, agent)
    performance_metrics = health_checker.get_performance_metrics()
    
    return {
        **system_health,
        "model_health": model_health,
        "performance": performance_metrics
    }
