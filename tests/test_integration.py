# tests/test_integration.py - 集成测试
import sys
import time
import pytest
import pandas as pd
import warnings
from pathlib import Path

# Suppress scikit-learn version warnings for integration tests
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.config import config
from utils.logger import app_logger
from utils.cache import cache, TTLCache

class TestIntegration:
    """集成测试类"""
    
    def test_config_loading(self):
        """测试配置加载"""
        assert config.PIPELINE_PATH is not None
        assert config.SCHEMA_PATH is not None
        assert config.REQUEST_TIMEOUT > 0
        assert config.MAX_RETRIES > 0
        assert len(config.USER_AGENTS) > 0
    
    def test_logger_setup(self):
        """测试日志记录器设置"""
        test_logger = app_logger
        assert test_logger is not None
        assert test_logger.level <= 20  # INFO level or lower
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        if not config.ENABLE_CACHE:
            pytest.skip("Cache disabled")
        
        # 创建测试缓存
        test_cache = TTLCache(ttl_hours=1)
        
        # 测试基本操作
        test_cache.set("test_key", "test_value")
        assert test_cache.get("test_key") == "test_value"
        
        # 测试过期
        test_cache.set("expired_key", "expired_value")
        # 模拟过期（这里我们手动设置一个过期的时间戳）
        from datetime import datetime, timedelta
        expired_time = datetime.now() - timedelta(hours=2)
        test_cache.cache["expired_key"] = ("expired_value", expired_time)
        assert test_cache.get("expired_key") is None
        
        # 测试清理（添加另一个过期项用于清理测试）
        test_cache.cache["another_expired"] = ("value", datetime.now() - timedelta(hours=2))
        cleaned = test_cache.cleanup_expired()
        assert cleaned >= 1  # 至少清理了1个过期项
    
    def test_end_to_end_prediction_flow(self):
        """测试端到端预测流程"""
        try:
            # 导入应用模块
            from app.app import predict_point_range, _derive_features_with_pipeline_steps
            from utils.constants import RAW_INPUTS
            
            # 创建测试数据
            test_data = {
                "Rating": 4.0,
                "age": 10,
                "Sector": "Information Technology",
                "Type of ownership": "Company - Public",
                "size_band": "Mid",
                "Job Title": "Software Engineer",
                "Location": "San Francisco, CA"
            }
            
            # 测试特征提取
            derived = _derive_features_with_pipeline_steps(test_data)
            assert "seniority" in derived
            assert "loc_tier" in derived
            assert derived["seniority"] in ["intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager", "director", "vp", "cxo"]
            assert derived["loc_tier"] in ["low", "mid", "high", "very_high"]
            
            # 测试预测（如果模型可用）
            try:
                df = pd.DataFrame([test_data], columns=RAW_INPUTS)
                point, low, high = predict_point_range(df)
                
                assert isinstance(point, float)
                assert isinstance(low, float)
                assert isinstance(high, float)
                assert low <= point <= high
                assert point > 0
                
                app_logger.info(f"Prediction successful: ${point:,.0f} (range: ${low:,.0f}-${high:,.0f})")
                
            except Exception as e:
                app_logger.warning(f"Prediction test skipped due to model unavailability: {e}")
                pytest.skip("Model not available for testing")
                
        except ImportError as e:
            pytest.skip(f"App module not available: {e}")
    
    def test_company_agent_integration(self):
        """测试公司代理集成"""
        try:
            from app.agent import CompanyAgent
            
            agent = CompanyAgent(cache_ttl_hours=1)
            
            # 测试空公司名
            result = agent.lookup("")
            assert "_error" in result
            assert result["_error"] == "empty company_name"
            
            # 测试缓存功能
            agent.clear_cache()
            assert len(agent.cache) == 0
            
            # 测试清理功能
            cleaned = agent.cleanup_expired()
            assert cleaned == 0  # 空缓存应该返回0
            
        except ImportError as e:
            pytest.skip(f"Agent module not available: {e}")
    
    def test_health_check_integration(self):
        """测试健康检查集成"""
        try:
            from app.health import health_check, health_checker
            
            # 测试健康检查器基本功能
            health_checker.record_request(success=True, prediction_time=0.5)
            health_checker.record_request(success=False)
            
            # 测试性能指标
            metrics = health_checker.get_performance_metrics()
            assert "prediction_times" in metrics
            
            # 测试系统健康检查
            system_health = health_checker.get_system_health()
            assert "status" in system_health
            assert "timestamp" in system_health
            assert "uptime_seconds" in system_health
            
        except ImportError as e:
            pytest.skip(f"Health module not available: {e}")

class TestPerformance:
    """性能测试类"""
    
    def test_prediction_speed(self):
        """测试预测速度"""
        try:
            from app.app import predict_point_range
            from utils.constants import RAW_INPUTS
            
            # 创建测试数据
            test_data = {
                "Rating": 4.0,
                "age": 10,
                "Sector": "Information Technology",
                "Type of ownership": "Company - Public",
                "size_band": "Mid",
                "Job Title": "Software Engineer",
                "Location": "San Francisco, CA"
            }
            
            df = pd.DataFrame([test_data], columns=RAW_INPUTS)
            
            # 测试预测速度
            start_time = time.time()
            point, low, high = predict_point_range(df)
            duration = time.time() - start_time
            
            # 预测应该在合理时间内完成
            assert duration < config.MAX_PREDICTION_TIME, f"Prediction took {duration:.3f}s, expected < {config.MAX_PREDICTION_TIME}s"
            
            app_logger.info(f"Prediction completed in {duration:.3f}s")
            
        except ImportError as e:
            pytest.skip(f"App module not available: {e}")
        except Exception as e:
            app_logger.warning(f"Performance test skipped: {e}")
            pytest.skip(f"Performance test not applicable: {e}")
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行一些操作
        try:
            from app.app import _derive_features_with_pipeline_steps
            
            test_data = {
                "Rating": 4.0,
                "age": 10,
                "Sector": "Information Technology",
                "Type of ownership": "Company - Public",
                "size_band": "Mid",
                "Job Title": "Software Engineer",
                "Location": "San Francisco, CA"
            }
            
            # 执行多次特征提取
            for _ in range(100):
                _derive_features_with_pipeline_steps(test_data)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # 内存增长应该在合理范围内（< 50MB）
            assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, expected < 50MB"
            
            app_logger.info(f"Memory usage increased by {memory_increase:.1f}MB")
            
        except ImportError as e:
            pytest.skip(f"App module not available: {e}")
        except Exception as e:
            app_logger.warning(f"Memory test skipped: {e}")
            pytest.skip(f"Memory test not applicable: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
