# tests/test_logger.py - Comprehensive tests for logger.py
import sys
import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger, timing_decorator, retry_decorator, app_logger

class TestSetupLogger:
    """Test setup_logger function"""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INFO"
            mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            logger = setup_logger("test_logger")
            
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_logger"
            assert logger.level == logging.INFO
    
    def test_setup_logger_custom_level(self):
        """Test logger setup with custom level"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "DEBUG"
            mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            # Clear any existing handlers first
            logger = logging.getLogger("test_logger_custom")
            logger.handlers.clear()
            
            logger = setup_logger("test_logger_custom", level="WARNING")
            
            assert logger.level == logging.WARNING
    
    def test_setup_logger_existing_handlers(self):
        """Test logger setup with existing handlers"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INFO"
            mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            # Create logger with existing handler
            logger1 = setup_logger("test_logger")
            logger2 = setup_logger("test_logger")  # Should return existing logger
            
            assert logger1 is logger2
            assert len(logger1.handlers) == 1  # Should not add duplicate handlers
    
    def test_setup_logger_invalid_level(self):
        """Test logger setup with invalid level"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INVALID"
            mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            logger = setup_logger("test_logger")
            
            # Should default to INFO level
            assert logger.level == logging.INFO
    
    def test_setup_logger_handler_properties(self):
        """Test logger handler properties"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INFO"
            mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            logger = setup_logger("test_logger")
            
            assert len(logger.handlers) == 1
            handler = logger.handlers[0]
            assert isinstance(handler, logging.StreamHandler)
            assert handler.level == logging.INFO
            assert isinstance(handler.formatter, logging.Formatter)
            assert not logger.propagate  # Should not propagate to parent
    
    def test_setup_logger_multiple_loggers(self):
        """Test setup of multiple loggers"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INFO"
            mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            logger1 = setup_logger("logger1")
            logger2 = setup_logger("logger2")
            
            assert logger1.name == "logger1"
            assert logger2.name == "logger2"
            assert logger1 is not logger2

class TestTimingDecorator:
    """Test timing_decorator function"""
    
    def test_timing_decorator_disabled(self):
        """Test timing decorator when performance monitoring is disabled"""
        with patch('utils.logger.config') as mock_config:
            mock_config.ENABLE_PERFORMANCE_MONITORING = False
            
            @timing_decorator
            def test_function():
                return "test_result"
            
            result = test_function()
            assert result == "test_result"
    
    def test_timing_decorator_enabled(self):
        """Test timing decorator when performance monitoring is enabled"""
        with patch('utils.logger.config') as mock_config, \
             patch('utils.logger.logger') as mock_logger, \
             patch('time.time') as mock_time:
            
            mock_config.ENABLE_PERFORMANCE_MONITORING = True
            mock_time.side_effect = [0.0, 0.5]  # Start and end times
            
            @timing_decorator
            def test_function():
                return "test_result"
            
            result = test_function()
            
            assert result == "test_result"
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "test_function took 0.500 seconds" in call_args
    
    def test_timing_decorator_with_exception(self):
        """Test timing decorator with exception"""
        with patch('utils.logger.config') as mock_config, \
             patch('utils.logger.logger') as mock_logger, \
             patch('time.time') as mock_time:
            
            mock_config.ENABLE_PERFORMANCE_MONITORING = True
            mock_time.side_effect = [0.0, 0.3]  # Start and end times
            
            @timing_decorator
            def test_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                test_function()
            
            # Should still log timing even with exception
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "test_function took 0.300 seconds" in call_args
    
    def test_timing_decorator_prediction_time_warning(self):
        """Test timing decorator with prediction time warning"""
        with patch('utils.logger.config') as mock_config, \
             patch('utils.logger.logger') as mock_logger, \
             patch('time.time') as mock_time:
            
            mock_config.ENABLE_PERFORMANCE_MONITORING = True
            mock_config.MAX_PREDICTION_TIME = 0.1
            mock_time.side_effect = [0.0, 0.5]  # Exceeds max time
            
            @timing_decorator
            def predict_point_range():
                return "prediction_result"
            
            result = predict_point_range()
            
            assert result == "prediction_result"
            # Should log both info and warning
            assert mock_logger.info.call_count == 1
            assert mock_logger.warning.call_count == 1
            
            warning_call = mock_logger.warning.call_args[0][0]
            assert "predict_point_range exceeded maximum time limit" in warning_call
            assert "0.500s > 0.1s" in warning_call
    
    def test_timing_decorator_serve_time_warning(self):
        """Test timing decorator with serve time warning"""
        with patch('utils.logger.config') as mock_config, \
             patch('utils.logger.logger') as mock_logger, \
             patch('time.time') as mock_time:
            
            mock_config.ENABLE_PERFORMANCE_MONITORING = True
            mock_config.MAX_PREDICTION_TIME = 0.1
            mock_time.side_effect = [0.0, 0.5]  # Exceeds max time
            
            @timing_decorator
            def serve():
                return "serve_result"
            
            result = serve()
            
            assert result == "serve_result"
            # Should log both info and warning
            assert mock_logger.info.call_count == 1
            assert mock_logger.warning.call_count == 1
            
            warning_call = mock_logger.warning.call_args[0][0]
            assert "serve exceeded maximum time limit" in warning_call
            assert "0.500s > 0.1s" in warning_call
    
    def test_timing_decorator_other_function_no_warning(self):
        """Test timing decorator with other function (no warning)"""
        with patch('utils.logger.config') as mock_config, \
             patch('utils.logger.logger') as mock_logger, \
             patch('time.time') as mock_time:
            
            mock_config.ENABLE_PERFORMANCE_MONITORING = True
            mock_config.MAX_PREDICTION_TIME = 0.1
            mock_time.side_effect = [0.0, 0.5]  # Exceeds max time
            
            @timing_decorator
            def other_function():
                return "other_result"
            
            result = other_function()
            
            assert result == "other_result"
            # Should only log info, no warning
            assert mock_logger.info.call_count == 1
            assert mock_logger.warning.call_count == 0

class TestRetryDecorator:
    """Test retry_decorator function"""
    
    def test_retry_decorator_success_first_attempt(self):
        """Test retry decorator with success on first attempt"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
            def test_function():
                return "success"
            
            result = test_function()
            
            assert result == "success"
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()
            mock_sleep.assert_not_called()
    
    def test_retry_decorator_success_after_retries(self):
        """Test retry decorator with success after retries"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            call_count = 0
            @retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
            def test_function():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ValueError("Temporary error")
                return "success"
            
            result = test_function()
            
            assert result == "success"
            assert call_count == 3
            assert mock_logger.warning.call_count == 2  # Two retry attempts
            mock_logger.error.assert_not_called()
            assert mock_sleep.call_count == 2  # Two sleep calls
    
    def test_retry_decorator_all_attempts_fail(self):
        """Test retry decorator with all attempts failing"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=2, delay=1.0, backoff=2.0)
            def test_function():
                raise ValueError("Persistent error")
            
            with pytest.raises(ValueError, match="Persistent error"):
                test_function()
            
            assert mock_logger.warning.call_count == 2  # Two retry attempts
            assert mock_logger.error.call_count == 1  # Final error
            assert mock_sleep.call_count == 2  # Two sleep calls
            
            # Check warning messages
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert "test_function attempt 1 failed" in warning_calls[0]
            assert "test_function attempt 2 failed" in warning_calls[1]
            
            # Check error message
            error_call = mock_logger.error.call_args[0][0]
            assert "test_function failed after 3 attempts" in error_call
            assert "Persistent error" in error_call
    
    def test_retry_decorator_backoff_calculation(self):
        """Test retry decorator backoff calculation"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
            def test_function():
                raise ValueError("Error")
            
            with pytest.raises(ValueError):
                test_function()
            
            # Check sleep calls with backoff
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls[0] == 1.0  # delay * (backoff ** 0)
            assert sleep_calls[1] == 2.0  # delay * (backoff ** 1)
            assert sleep_calls[2] == 4.0  # delay * (backoff ** 2)
    
    def test_retry_decorator_custom_parameters(self):
        """Test retry decorator with custom parameters"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=1, delay=0.5, backoff=1.5)
            def test_function():
                raise ValueError("Error")
            
            with pytest.raises(ValueError):
                test_function()
            
            assert mock_logger.warning.call_count == 1  # One retry attempt
            assert mock_logger.error.call_count == 1  # Final error
            assert mock_sleep.call_count == 1  # One sleep call
            assert mock_sleep.call_args[0][0] == 0.5  # Custom delay
    
    def test_retry_decorator_different_exceptions(self):
        """Test retry decorator with different exception types"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=2, delay=0.1, backoff=1.0)
            def test_function():
                raise RuntimeError("Runtime error")
            
            with pytest.raises(RuntimeError, match="Runtime error"):
                test_function()
            
            assert mock_logger.warning.call_count == 2
            assert mock_logger.error.call_count == 1
    
    def test_retry_decorator_function_arguments(self):
        """Test retry decorator preserves function arguments"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=1, delay=0.1, backoff=1.0)
            def test_function(arg1, arg2, kwarg1=None, kwarg2=None):
                return f"{arg1}_{arg2}_{kwarg1}_{kwarg2}"
            
            result = test_function("a", "b", kwarg1="c", kwarg2="d")
            assert result == "a_b_c_d"
    
    def test_retry_decorator_zero_retries(self):
        """Test retry decorator with zero retries"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=0, delay=1.0, backoff=2.0)
            def test_function():
                raise ValueError("Error")
            
            with pytest.raises(ValueError):
                test_function()
            
            mock_logger.warning.assert_not_called()
            assert mock_logger.error.call_count == 1
            mock_sleep.assert_not_called()
    
    def test_retry_decorator_negative_retries(self):
        """Test retry decorator with negative retries"""
        with patch('utils.logger.logger') as mock_logger, \
             patch('time.sleep') as mock_sleep:
            
            @retry_decorator(max_retries=-1, delay=1.0, backoff=2.0)
            def test_function():
                raise ValueError("Error")
            
            # With negative retries, the function should raise TypeError due to last_exception being None
            with pytest.raises(TypeError):
                test_function()
            
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()  # No error logged because loop doesn't execute
            mock_sleep.assert_not_called()

class TestLoggerIntegration:
    """Test logger integration and edge cases"""
    
    def test_logger_import(self):
        """Test that logger is properly imported"""
        from utils.logger import logger
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "utils.logger"
    
    def test_logger_level_validation(self):
        """Test logger level validation"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INVALID_LEVEL"
            mock_config.LOG_FORMAT = "%(message)s"
            
            logger = setup_logger("test_logger")
            
            # Should default to INFO when level is invalid
            assert logger.level == logging.INFO
    
    def test_logger_formatter_configuration(self):
        """Test logger formatter configuration"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "DEBUG"
            mock_config.LOG_FORMAT = "CUSTOM: %(message)s"
            
            # Clear any existing handlers first
            logger = logging.getLogger("test_logger_formatter")
            logger.handlers.clear()
            
            logger = setup_logger("test_logger_formatter")
            
            handler = logger.handlers[0]
            formatter = handler.formatter
            assert isinstance(formatter, logging.Formatter)
            assert formatter._fmt == "CUSTOM: %(message)s"
    
    def test_logger_propagate_setting(self):
        """Test logger propagate setting"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INFO"
            mock_config.LOG_FORMAT = "%(message)s"
            
            logger = setup_logger("test_logger")
            
            # Should not propagate to parent loggers
            assert not logger.propagate
    
    def test_logger_handler_duplication_prevention(self):
        """Test that handlers are not duplicated"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "INFO"
            mock_config.LOG_FORMAT = "%(message)s"
            
            # Setup logger multiple times
            logger1 = setup_logger("test_logger")
            logger2 = setup_logger("test_logger")
            logger3 = setup_logger("test_logger")
            
            # Should be the same logger instance
            assert logger1 is logger2 is logger3
            
            # Should only have one handler
            assert len(logger1.handlers) == 1
