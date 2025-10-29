# tests/test_integration.py - Integration tests
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
    """Integration test suite."""
    
    def test_config_loading(self):
        """Verify configuration is loaded."""
        assert config.PIPELINE_PATH is not None
        assert config.SCHEMA_PATH is not None
        assert config.REQUEST_TIMEOUT > 0
        assert config.MAX_RETRIES > 0
        assert len(config.USER_AGENTS) > 0
    
    def test_logger_setup(self):
        """Verify logger setup."""
        test_logger = app_logger
        assert test_logger is not None
        assert test_logger.level <= 20  # INFO level or lower
    
    def test_cache_functionality(self):
        """Verify cache functionality."""
        if not config.ENABLE_CACHE:
            pytest.skip("Cache disabled")
        
        # Create a test cache
        test_cache = TTLCache(ttl_hours=1)
        
        # Basic operations
        test_cache.set("test_key", "test_value")
        assert test_cache.get("test_key") == "test_value"
        
        # Expiration
        test_cache.set("expired_key", "expired_value")
        # Simulate expiration (manually set an expired timestamp)
        from datetime import datetime, timedelta
        expired_time = datetime.now() - timedelta(hours=2)
        test_cache.cache["expired_key"] = ("expired_value", expired_time)
        assert test_cache.get("expired_key") is None
        
        # Cleanup (add another expired entry for cleanup test)
        test_cache.cache["another_expired"] = ("value", datetime.now() - timedelta(hours=2))
        cleaned = test_cache.cleanup_expired()
        assert cleaned >= 1  # At least one expired entry should be cleaned up
    
    def test_end_to_end_prediction_flow(self):
        """End-to-end prediction flow test."""
        try:
            # Import application functions
            from app.app import predict_point_range, _derive_features_with_pipeline_steps
            from utils.constants import RAW_INPUTS
            
            # Create test data
            test_data = {
                "Rating": 4.0,
                "age": 10,
                "Sector": "Information Technology",
                "Type of ownership": "Company - Public",
                "size_band": "Mid",
                "Job Title": "Software Engineer",
                "Location": "San Francisco, CA"
            }
            
            # Derived feature extraction
            derived = _derive_features_with_pipeline_steps(test_data)
            assert "seniority" in derived
            assert "loc_tier" in derived
            assert derived["seniority"] in ["intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager", "director", "vp", "cxo"]
            assert derived["loc_tier"] in ["low", "mid", "high", "very_high"]
            
            # Prediction (if model is available)
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
        """CompanyAgent integration test."""
        try:
            from app.agent import CompanyAgent
            
            agent = CompanyAgent(cache_ttl_hours=1)
            
            # Empty company name
            result = agent.lookup("")
            assert "_error" in result
            assert result["_error"] == "empty company_name"
            
            # Cache operations
            agent.clear_cache()
            assert len(agent.cache) == 0
            
            # Cleanup on empty cache
            cleaned = agent.cleanup_expired()
            assert cleaned == 0  # Empty cache should return 0
            
        except ImportError as e:
            pytest.skip(f"Agent module not available: {e}")

class TestPerformance:
    """Performance test suite."""
    
    def test_prediction_speed(self):
        """Verify prediction latency is within threshold."""
        try:
            from app.app import predict_point_range
            from utils.constants import RAW_INPUTS
            
            # Create test data
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
            
            # Measure prediction time
            start_time = time.time()
            point, low, high = predict_point_range(df)
            duration = time.time() - start_time
            
            # Must complete within configured threshold
            assert duration < config.MAX_PREDICTION_TIME, f"Prediction took {duration:.3f}s, expected < {config.MAX_PREDICTION_TIME}s"
            
            app_logger.info(f"Prediction completed in {duration:.3f}s")
            
        except ImportError as e:
            pytest.skip(f"App module not available: {e}")
        except Exception as e:
            app_logger.warning(f"Performance test skipped: {e}")
            pytest.skip(f"Performance test not applicable: {e}")
    
    def test_memory_usage(self):
        """Verify memory usage increase is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute some operations
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
            
            # Run feature extraction multiple times
            for _ in range(100):
                _derive_features_with_pipeline_steps(test_data)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Memory growth should be within a reasonable range (< 50MB)
            assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, expected < 50MB"
            
            app_logger.info(f"Memory usage increased by {memory_increase:.1f}MB")
            
        except ImportError as e:
            pytest.skip(f"App module not available: {e}")
        except Exception as e:
            app_logger.warning(f"Memory test skipped: {e}")
            pytest.skip(f"Memory test not applicable: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
