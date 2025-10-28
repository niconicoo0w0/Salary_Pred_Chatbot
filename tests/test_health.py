# tests/test_health.py - Comprehensive tests for health.py
import sys
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.health import HealthChecker, health_check, health_checker

class TestHealthChecker:
    """Test HealthChecker class"""
    
    def test_init(self):
        """Test HealthChecker initialization"""
        checker = HealthChecker()
        assert checker.start_time > 0
        assert checker.request_count == 0
        assert checker.error_count == 0
        assert checker.prediction_times == []
    
    @patch('app.health.psutil.virtual_memory')
    @patch('app.health.psutil.cpu_percent')
    @patch('app.health.psutil.disk_usage')
    @patch('app.health.psutil.Process')
    def test_get_system_health_success(self, mock_process_class, mock_disk, mock_cpu, mock_memory):
        """Test successful system health check"""
        # Mock memory info
        mock_memory_obj = Mock()
        mock_memory_obj.total = 8000000000
        mock_memory_obj.available = 4000000000
        mock_memory_obj.percent = 50.0
        mock_memory_obj.used = 4000000000
        mock_memory.return_value = mock_memory_obj
        
        # Mock CPU info
        mock_cpu.return_value = 25.5
        
        # Mock disk info
        mock_disk_obj = Mock()
        mock_disk_obj.total = 1000000000000
        mock_disk_obj.used = 500000000000
        mock_disk_obj.free = 500000000000
        mock_disk.return_value = mock_disk_obj
        
        # Mock process info
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.memory_info.return_value.rss = 100000000
        mock_process.cpu_percent.return_value = 5.0
        mock_process.create_time.return_value = time.time() - 3600
        mock_process_class.return_value = mock_process
        
        checker = HealthChecker()
        result = checker.get_system_health()
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "uptime_seconds" in result
        assert result["memory"]["total"] == 8000000000
        assert result["memory"]["available"] == 4000000000
        assert result["memory"]["percent"] == 50.0
        assert result["memory"]["used"] == 4000000000
        assert result["cpu_percent"] == 25.5
        assert result["disk"]["total"] == 1000000000000
        assert result["disk"]["used"] == 500000000000
        assert result["disk"]["free"] == 500000000000
        assert result["process"]["pid"] == 12345
        assert result["process"]["memory_mb"] == 100000000 / 1024 / 1024
        assert result["process"]["cpu_percent"] == 5.0
        # Check that create_time is reasonable (within the last hour)
        assert result["process"]["create_time"] < time.time()
        assert result["process"]["create_time"] > time.time() - 7200  # Within last 2 hours
        assert result["requests"]["total"] == 0
        assert result["requests"]["errors"] == 0
        # Success rate should be 100% when there are no requests (no failures)
        assert result["requests"]["success_rate"] == 100.0 or result["requests"]["success_rate"] == 0.0
    
    @patch('app.health.psutil.virtual_memory')
    def test_get_system_health_exception(self, mock_memory):
        """Test system health check with exception"""
        mock_memory.side_effect = Exception("System error")
        
        checker = HealthChecker()
        result = checker.get_system_health()
        
        assert result["status"] == "unhealthy"
        assert "timestamp" in result
        assert "error" in result
        assert result["error"] == "System error"
    
    def test_get_model_health_with_pipeline(self):
        """Test model health check with pipeline"""
        mock_pipe = Mock()
        mock_pipe.predict.return_value = [100000.0]
        
        checker = HealthChecker()
        result = checker.get_model_health(pipe=mock_pipe)
        
        assert "timestamp" in result
        assert "models" in result
        assert "pipeline" in result["models"]
        assert result["models"]["pipeline"]["status"] == "healthy"
        assert "prediction_time_ms" in result["models"]["pipeline"]
        assert result["models"]["pipeline"]["prediction_value"] == 100000.0
    
    def test_get_model_health_pipeline_exception(self):
        """Test model health check with pipeline exception"""
        mock_pipe = Mock()
        mock_pipe.predict.side_effect = Exception("Prediction error")
        
        checker = HealthChecker()
        result = checker.get_model_health(pipe=mock_pipe)
        
        assert result["models"]["pipeline"]["status"] == "unhealthy"
        assert "error" in result["models"]["pipeline"]
        assert result["models"]["pipeline"]["error"] == "Prediction error"
    
    def test_get_model_health_no_pipeline(self):
        """Test model health check without pipeline"""
        checker = HealthChecker()
        result = checker.get_model_health(pipe=None)
        
        assert result["models"]["pipeline"]["status"] == "not_loaded"
        assert "error" in result["models"]["pipeline"]
        assert result["models"]["pipeline"]["error"] == "Pipeline not available"
    
    def test_get_model_health_with_agent(self):
        """Test model health check with agent"""
        mock_agent = Mock()
        mock_agent.lookup.return_value = {"sector": "Technology"}
        mock_agent.cache = {"test": "data"}
        
        checker = HealthChecker()
        result = checker.get_model_health(agent=mock_agent)
        
        assert result["models"]["agent"]["status"] == "healthy"
        assert result["models"]["agent"]["cache_size"] == 1
        assert "test_result_keys" in result["models"]["agent"]
    
    def test_get_model_health_agent_exception(self):
        """Test model health check with agent exception"""
        mock_agent = Mock()
        mock_agent.lookup.side_effect = Exception("Agent error")
        
        checker = HealthChecker()
        result = checker.get_model_health(agent=mock_agent)
        
        assert result["models"]["agent"]["status"] == "unhealthy"
        assert "error" in result["models"]["agent"]
        assert result["models"]["agent"]["error"] == "Agent error"
    
    def test_get_model_health_no_agent(self):
        """Test model health check without agent"""
        checker = HealthChecker()
        result = checker.get_model_health(agent=None)
        
        assert result["models"]["agent"]["status"] == "not_loaded"
        assert "error" in result["models"]["agent"]
        assert result["models"]["agent"]["error"] == "Agent not available"
    
    def test_record_request_success(self):
        """Test recording successful request"""
        checker = HealthChecker()
        checker.record_request(success=True, prediction_time=0.5)
        
        assert checker.request_count == 1
        assert checker.error_count == 0
        assert checker.prediction_times == [0.5]
    
    def test_record_request_failure(self):
        """Test recording failed request"""
        checker = HealthChecker()
        checker.record_request(success=False)
        
        assert checker.request_count == 1
        assert checker.error_count == 1
        assert checker.prediction_times == []
    
    def test_record_request_multiple(self):
        """Test recording multiple requests"""
        checker = HealthChecker()
        checker.record_request(success=True, prediction_time=0.3)
        checker.record_request(success=True, prediction_time=0.7)
        checker.record_request(success=False)
        checker.record_request(success=True, prediction_time=0.4)
        
        assert checker.request_count == 4
        assert checker.error_count == 1
        assert checker.prediction_times == [0.3, 0.7, 0.4]
    
    def test_record_request_no_prediction_time(self):
        """Test recording request without prediction time"""
        checker = HealthChecker()
        checker.record_request(success=True)
        
        assert checker.request_count == 1
        assert checker.error_count == 0
        assert checker.prediction_times == []
    
    def test_get_performance_metrics_no_data(self):
        """Test performance metrics with no data"""
        checker = HealthChecker()
        result = checker.get_performance_metrics()
        
        assert result == {"prediction_times": "no_data"}
    
    def test_get_performance_metrics_with_data(self):
        """Test performance metrics with data"""
        checker = HealthChecker()
        checker.prediction_times = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result = checker.get_performance_metrics()
        
        assert "prediction_times" in result
        assert result["prediction_times"]["count"] == 5
        assert result["prediction_times"]["mean"] == 0.3
        assert result["prediction_times"]["min"] == 0.1
        assert result["prediction_times"]["max"] == 0.5
        assert result["prediction_times"]["p95"] == 0.5  # With 5 items, p95 is max
    
    def test_get_performance_metrics_p95_calculation(self):
        """Test performance metrics p95 calculation with many items"""
        checker = HealthChecker()
        # Create 100 prediction times
        checker.prediction_times = [i * 0.01 for i in range(100)]
        
        result = checker.get_performance_metrics()
        
        assert result["prediction_times"]["count"] == 100
        assert result["prediction_times"]["mean"] == 0.495
        assert result["prediction_times"]["min"] == 0.0
        assert result["prediction_times"]["max"] == 0.99
        # p95 should be the 95th percentile (index 94 out of 100 items)
        assert abs(result["prediction_times"]["p95"] - 0.95) < 0.01
    
    def test_prediction_times_limit(self):
        """Test that prediction times are limited to 100 items"""
        checker = HealthChecker()
        
        # Add more than 100 prediction times
        for i in range(150):
            checker.record_request(success=True, prediction_time=i * 0.01)
        
        assert len(checker.prediction_times) == 100
        # Should keep the most recent 100 items
        assert checker.prediction_times[0] == 0.5  # 50th item
        assert checker.prediction_times[-1] == 1.49  # 149th item

class TestHealthCheckFunction:
    """Test health_check function"""
    
    @patch('app.health.health_checker.get_system_health')
    @patch('app.health.health_checker.get_model_health')
    @patch('app.health.health_checker.get_performance_metrics')
    def test_health_check_success(self, mock_performance, mock_model, mock_system):
        """Test successful health check"""
        mock_system.return_value = {"status": "healthy", "timestamp": "2024-01-01T00:00:00"}
        mock_model.return_value = {"models": {"pipeline": {"status": "healthy"}}}
        mock_performance.return_value = {"prediction_times": {"count": 10}}
        
        result = health_check()
        
        assert result["status"] == "healthy"
        assert result["timestamp"] == "2024-01-01T00:00:00"
        assert "model_health" in result
        assert "performance" in result
        assert result["model_health"]["models"]["pipeline"]["status"] == "healthy"
        assert result["performance"]["prediction_times"]["count"] == 10
    
    @patch('app.health.health_checker.get_system_health')
    @patch('app.health.health_checker.get_model_health')
    @patch('app.health.health_checker.get_performance_metrics')
    def test_health_check_with_pipeline_and_agent(self, mock_performance, mock_model, mock_system):
        """Test health check with pipeline and agent"""
        mock_system.return_value = {"status": "healthy"}
        mock_model.return_value = {"models": {"pipeline": {"status": "healthy"}, "agent": {"status": "healthy"}}}
        mock_performance.return_value = {"prediction_times": "no_data"}
        
        mock_pipe = Mock()
        mock_agent = Mock()
        
        result = health_check(pipe=mock_pipe, agent=mock_agent)
        
        mock_system.assert_called_once()
        mock_model.assert_called_once_with(mock_pipe, mock_agent)
        mock_performance.assert_called_once()
    
    def test_health_checker_global_instance(self):
        """Test that health_checker is a global instance"""
        from app.health import health_checker
        
        assert isinstance(health_checker, HealthChecker)
        assert health_checker.start_time > 0

class TestHealthCheckerEdgeCases:
    """Test HealthChecker edge cases"""
    
    def test_initial_state(self):
        """Test initial state of HealthChecker"""
        checker = HealthChecker()
        
        assert checker.request_count == 0
        assert checker.error_count == 0
        assert checker.prediction_times == []
        assert checker.start_time > 0
    
    def test_success_rate_calculation_zero_requests(self):
        """Test success rate calculation with zero requests"""
        checker = HealthChecker()
        
        # Should not divide by zero
        system_health = checker.get_system_health()
        assert system_health["requests"]["success_rate"] == 100.0
    
    def test_success_rate_calculation_all_errors(self):
        """Test success rate calculation with all errors"""
        checker = HealthChecker()
        checker.record_request(success=False)
        checker.record_request(success=False)
        
        system_health = checker.get_system_health()
        assert system_health["requests"]["success_rate"] == 0.0
    
    def test_success_rate_calculation_mixed(self):
        """Test success rate calculation with mixed results"""
        checker = HealthChecker()
        checker.record_request(success=True)
        checker.record_request(success=False)
        checker.record_request(success=True)
        
        system_health = checker.get_system_health()
        assert system_health["requests"]["success_rate"] == 66.66666666666666
    
    def test_uptime_calculation(self):
        """Test uptime calculation"""
        checker = HealthChecker()
        time.sleep(0.01)  # Small delay to ensure uptime > 0
        
        system_health = checker.get_system_health()
        assert system_health["uptime_seconds"] > 0
    
    def test_timestamp_format(self):
        """Test timestamp format"""
        checker = HealthChecker()
        system_health = checker.get_system_health()
        
        # Should be ISO format
        timestamp = system_health["timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise exception
    
    def test_memory_calculation_accuracy(self):
        """Test memory calculation accuracy"""
        checker = HealthChecker()
        
        with patch('app.health.psutil.virtual_memory') as mock_memory:
            mock_memory_obj = Mock()
            mock_memory_obj.total = 1000000000  # 1GB
            mock_memory_obj.available = 500000000  # 500MB
            mock_memory_obj.percent = 50.0
            mock_memory_obj.used = 500000000  # 500MB
            mock_memory.return_value = mock_memory_obj
            
            with patch('app.health.psutil.cpu_percent', return_value=0):
                with patch('app.health.psutil.disk_usage') as mock_disk:
                    mock_disk_obj = Mock()
                    mock_disk_obj.total = 1000000000
                    mock_disk_obj.used = 500000000
                    mock_disk_obj.free = 500000000
                    mock_disk.return_value = mock_disk_obj
                    
                    with patch('app.health.psutil.Process') as mock_process_class:
                        mock_process = Mock()
                        mock_process.pid = 12345
                        mock_process.memory_info.return_value.rss = 100000000
                        mock_process.cpu_percent.return_value = 0
                        mock_process.create_time.return_value = time.time()
                        mock_process_class.return_value = mock_process
                        
                        system_health = checker.get_system_health()
                        
                        assert system_health["memory"]["total"] == 1000000000
                        assert system_health["memory"]["available"] == 500000000
                        assert system_health["memory"]["percent"] == 50.0
                        assert system_health["memory"]["used"] == 500000000
