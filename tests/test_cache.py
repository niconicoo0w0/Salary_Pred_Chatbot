# tests/test_cache.py - Comprehensive tests for cache.py
import sys
import pytest
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.cache import TTLCache, get_cache_key, cached, cache

class TestTTLCache:
    """Test TTLCache class"""
    
    def test_init(self):
        """Test TTLCache initialization"""
        cache = TTLCache(ttl_hours=12)
        assert cache.cache == {}
        assert cache.ttl == timedelta(hours=12)
    
    def test_init_default_ttl(self):
        """Test TTLCache initialization with default TTL"""
        cache = TTLCache()
        assert cache.cache == {}
        assert cache.ttl == timedelta(hours=24)
    
    def test_set_and_get(self):
        """Test basic set and get operations"""
        cache = TTLCache(ttl_hours=1)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.set("key2", {"nested": "value"})
        assert cache.get("key2") == {"nested": "value"}
    
    def test_get_nonexistent_key(self):
        """Test getting nonexistent key"""
        cache = TTLCache(ttl_hours=1)
        
        assert cache.get("nonexistent") is None
    
    def test_get_expired_key(self):
        """Test getting expired key"""
        cache = TTLCache(ttl_hours=1)
        
        # Set a key with expired timestamp
        expired_time = datetime.now() - timedelta(hours=2)
        cache.cache["expired_key"] = ("expired_value", expired_time)
        
        assert cache.get("expired_key") is None
        assert "expired_key" not in cache.cache  # Should be removed
    
    def test_get_key_with_float_timestamp(self):
        """Test getting key with float timestamp"""
        cache = TTLCache(ttl_hours=1)
        
        # Set a key with float timestamp (legacy format)
        float_timestamp = (datetime.now() - timedelta(minutes=30)).timestamp()
        cache.cache["float_key"] = ("float_value", float_timestamp)
        
        assert cache.get("float_key") == "float_value"
    
    def test_get_key_with_expired_float_timestamp(self):
        """Test getting key with expired float timestamp"""
        cache = TTLCache(ttl_hours=1)
        
        # Set a key with expired float timestamp
        expired_float_timestamp = (datetime.now() - timedelta(hours=2)).timestamp()
        cache.cache["expired_float_key"] = ("expired_float_value", expired_float_timestamp)
        
        assert cache.get("expired_float_key") is None
        assert "expired_float_key" not in cache.cache  # Should be removed
    
    def test_clear(self):
        """Test cache clearing"""
        cache = TTLCache(ttl_hours=1)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache.cache) == 2
        
        cache.clear()
        assert len(cache.cache) == 0
    
    def test_cleanup_expired_no_expired(self):
        """Test cleanup with no expired entries"""
        cache = TTLCache(ttl_hours=1)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 0
        assert len(cache.cache) == 2
    
    def test_cleanup_expired_with_expired(self):
        """Test cleanup with expired entries"""
        cache = TTLCache(ttl_hours=1)
        
        # Add current entries
        cache.set("current1", "value1")
        cache.set("current2", "value2")
        
        # Add expired entries
        expired_time = datetime.now() - timedelta(hours=2)
        cache.cache["expired1"] = ("expired_value1", expired_time)
        cache.cache["expired2"] = ("expired_value2", expired_time)
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 2
        assert len(cache.cache) == 2
        assert "current1" in cache.cache
        assert "current2" in cache.cache
        assert "expired1" not in cache.cache
        assert "expired2" not in cache.cache
    
    def test_cleanup_expired_mixed_timestamps(self):
        """Test cleanup with mixed timestamp formats"""
        cache = TTLCache(ttl_hours=1)
        
        # Add current entry
        cache.set("current", "value")
        
        # Add expired entries with different timestamp formats
        expired_datetime = datetime.now() - timedelta(hours=2)
        expired_float = (datetime.now() - timedelta(hours=2)).timestamp()
        
        cache.cache["expired_datetime"] = ("expired_value1", expired_datetime)
        cache.cache["expired_float"] = ("expired_value2", expired_float)
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 2
        assert len(cache.cache) == 1
        assert "current" in cache.cache
    
    def test_cleanup_expired_all_expired(self):
        """Test cleanup with all entries expired"""
        cache = TTLCache(ttl_hours=1)
        
        expired_time = datetime.now() - timedelta(hours=2)
        cache.cache["expired1"] = ("expired_value1", expired_time)
        cache.cache["expired2"] = ("expired_value2", expired_time)
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 2
        assert len(cache.cache) == 0
    
    def test_size(self):
        """Test cache size method"""
        cache = TTLCache(ttl_hours=1)
        
        assert cache.size() == 0
        
        cache.set("key1", "value1")
        assert cache.size() == 1
        
        cache.set("key2", "value2")
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0
    
    def test_cache_with_different_value_types(self):
        """Test cache with different value types"""
        cache = TTLCache(ttl_hours=1)
        
        # String
        cache.set("string", "hello")
        assert cache.get("string") == "hello"
        
        # Number
        cache.set("number", 42)
        assert cache.get("number") == 42
        
        # List
        cache.set("list", [1, 2, 3])
        assert cache.get("list") == [1, 2, 3]
        
        # Dict
        cache.set("dict", {"key": "value"})
        assert cache.get("dict") == {"key": "value"}
        
        # None
        cache.set("none", None)
        assert cache.get("none") is None
    
    def test_cache_key_overwrite(self):
        """Test cache key overwrite"""
        cache = TTLCache(ttl_hours=1)
        
        cache.set("key", "value1")
        assert cache.get("key") == "value1"
        
        cache.set("key", "value2")
        assert cache.get("key") == "value2"
        assert cache.size() == 1

class TestGetCacheKey:
    """Test get_cache_key function"""
    
    def test_get_cache_key_basic(self):
        """Test basic cache key generation"""
        key = get_cache_key("func_name", "arg1", "arg2", kwarg1="value1", kwarg2="value2")
        
        # Should be a valid MD5 hash
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)
    
    def test_get_cache_key_different_orders(self):
        """Test cache key generation with different argument orders"""
        key1 = get_cache_key("func", "arg1", "arg2", kwarg1="val1", kwarg2="val2")
        key2 = get_cache_key("func", "arg2", "arg1", kwarg2="val2", kwarg1="val1")
        
        # Should be different due to different positional argument order
        assert key1 != key2
    
    def test_get_cache_key_same_arguments(self):
        """Test cache key generation with same arguments"""
        key1 = get_cache_key("func", "arg1", "arg2", kwarg1="val1", kwarg2="val2")
        key2 = get_cache_key("func", "arg1", "arg2", kwarg1="val1", kwarg2="val2")
        
        # Should be identical
        assert key1 == key2
    
    def test_get_cache_key_different_functions(self):
        """Test cache key generation for different functions"""
        key1 = get_cache_key("func1", "arg1", "arg2")
        key2 = get_cache_key("func2", "arg1", "arg2")
        
        # Should be different
        assert key1 != key2
    
    def test_get_cache_key_no_arguments(self):
        """Test cache key generation with no arguments"""
        key = get_cache_key("func")
        
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)
    
    def test_get_cache_key_complex_objects(self):
        """Test cache key generation with complex objects"""
        key1 = get_cache_key("func", {"nested": "dict"}, [1, 2, 3])
        key2 = get_cache_key("func", {"nested": "dict"}, [1, 2, 3])
        
        # Should be identical
        assert key1 == key2
    
    def test_get_cache_key_serialization(self):
        """Test cache key generation with JSON serialization"""
        # Test that the function properly serializes arguments
        key_data = {
            'args': ("func_name", "arg1", "arg2"),
            'kwargs': [("kwarg1", "value1"), ("kwarg2", "value2")]
        }
        expected_json = json.dumps(key_data, sort_keys=True, default=str)
        expected_hash = hashlib.md5(expected_json.encode()).hexdigest()
        
        key = get_cache_key("func_name", "arg1", "arg2", kwarg1="value1", kwarg2="value2")
        assert key == expected_hash

class TestCachedDecorator:
    """Test cached decorator"""
    
    def test_cached_decorator_disabled(self):
        """Test cached decorator when cache is disabled"""
        with patch('utils.cache.config') as mock_config, \
             patch('utils.cache.cache', None):
            
            mock_config.ENABLE_CACHE = False
            
            @cached(ttl_hours=1)
            def test_function(arg1, arg2):
                return f"{arg1}_{arg2}"
            
            result = test_function("a", "b")
            assert result == "a_b"
    
    def test_cached_decorator_enabled(self):
        """Test cached decorator when cache is enabled"""
        with patch('utils.cache.config') as mock_config, \
             patch('utils.cache.cache') as mock_cache:
            
            mock_config.ENABLE_CACHE = True
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = None
            
            @cached(ttl_hours=1)
            def test_function(arg1, arg2):
                return f"{arg1}_{arg2}"
            
            result = test_function("a", "b")
            
            assert result == "a_b"
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_called_once()
    
    def test_cached_decorator_cache_hit(self):
        """Test cached decorator with cache hit"""
        with patch('utils.cache.config') as mock_config, \
             patch('utils.cache.cache') as mock_cache:
            
            mock_config.ENABLE_CACHE = True
            mock_cache.get.return_value = "cached_result"
            
            @cached(ttl_hours=1)
            def test_function(arg1, arg2):
                return f"{arg1}_{arg2}"
            
            result = test_function("a", "b")
            
            assert result == "cached_result"
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_not_called()
    
    def test_cached_decorator_different_arguments(self):
        """Test cached decorator with different arguments"""
        with patch('utils.cache.config') as mock_config, \
             patch('utils.cache.cache') as mock_cache:
            
            mock_config.ENABLE_CACHE = True
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = None
            
            @cached(ttl_hours=1)
            def test_function(arg1, arg2):
                return f"{arg1}_{arg2}"
            
            # First call
            result1 = test_function("a", "b")
            assert result1 == "a_b"
            
            # Second call with different arguments
            result2 = test_function("c", "d")
            assert result2 == "c_d"
            
            # Should have been called twice for get and set
            assert mock_cache.get.call_count == 2
            assert mock_cache.set.call_count == 2
    
    def test_cached_decorator_keyword_arguments(self):
        """Test cached decorator with keyword arguments"""
        with patch('utils.cache.config') as mock_config, \
             patch('utils.cache.cache') as mock_cache:
            
            mock_config.ENABLE_CACHE = True
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = None
            
            @cached(ttl_hours=1)
            def test_function(arg1, kwarg1=None, kwarg2=None):
                return f"{arg1}_{kwarg1}_{kwarg2}"
            
            result = test_function("a", kwarg1="b", kwarg2="c")
            
            assert result == "a_b_c"
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_called_once()
    
    def test_cached_decorator_exception_handling(self):
        """Test cached decorator with exception in function"""
        with patch('utils.cache.config') as mock_config, \
             patch('utils.cache.cache') as mock_cache:
            
            mock_config.ENABLE_CACHE = True
            mock_cache.get.return_value = None  # Cache miss
            
            @cached(ttl_hours=1)
            def test_function(arg1):
                raise ValueError("Test error")
            
            with pytest.raises(ValueError, match="Test error"):
                test_function("a")
            
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_not_called()  # Should not cache exceptions
    
    def test_cached_decorator_no_ttl_hours(self):
        """Test cached decorator without ttl_hours parameter"""
        with patch('utils.cache.config') as mock_config, \
             patch('utils.cache.cache') as mock_cache:
            
            mock_config.ENABLE_CACHE = True
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = None
            
            @cached()  # No ttl_hours specified
            def test_function(arg1):
                return f"result_{arg1}"
            
            result = test_function("a")
            
            assert result == "result_a"
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_called_once()

class TestGlobalCache:
    """Test global cache instance"""
    
    def test_global_cache_creation_enabled(self):
        """Test global cache creation when enabled"""
        # Test that cache is created and is a TTLCache instance
        from utils.cache import cache, TTLCache
        assert cache is not None
        assert isinstance(cache, TTLCache)
    
    def test_global_cache_creation_disabled(self):
        """Test global cache creation when disabled"""
        # Test that cache exists (it's created at module level)
        from utils.cache import cache
        assert cache is not None

class TestCacheEdgeCases:
    """Test cache edge cases"""
    
    def test_cache_with_very_large_values(self):
        """Test cache with very large values"""
        cache = TTLCache(ttl_hours=1)
        
        large_value = "x" * 10000
        cache.set("large_key", large_value)
        
        assert cache.get("large_key") == large_value
    
    def test_cache_with_unicode_values(self):
        """Test cache with unicode values"""
        cache = TTLCache(ttl_hours=1)
        
        unicode_value = "Hello 世界 🌍"
        cache.set("unicode_key", unicode_value)
        
        assert cache.get("unicode_key") == unicode_value
    
    def test_cache_with_special_characters_in_key(self):
        """Test cache with special characters in key"""
        cache = TTLCache(ttl_hours=1)
        
        special_key = "key with spaces & symbols!@#$%"
        cache.set(special_key, "value")
        
        assert cache.get(special_key) == "value"
    
    def test_cache_cleanup_with_mixed_timestamps(self):
        """Test cache cleanup with mixed timestamp formats"""
        cache = TTLCache(ttl_hours=1)
        
        # Add current entry
        cache.set("current", "value")
        
        # Add expired entries with different timestamp formats
        expired_datetime = datetime.now() - timedelta(hours=2)
        expired_float = (datetime.now() - timedelta(hours=2)).timestamp()
        current_float = (datetime.now() - timedelta(minutes=30)).timestamp()
        
        cache.cache["expired_datetime"] = ("expired_value1", expired_datetime)
        cache.cache["expired_float"] = ("expired_value2", expired_float)
        cache.cache["current_float"] = ("current_value", current_float)
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 2  # Only the expired ones
        assert len(cache.cache) == 2  # current + current_float
        assert "current" in cache.cache
        assert "current_float" in cache.cache
        assert "expired_datetime" not in cache.cache
        assert "expired_float" not in cache.cache
    
    def test_cache_key_generation_with_none_values(self):
        """Test cache key generation with None values"""
        key1 = get_cache_key("func", None, None, kwarg=None)
        key2 = get_cache_key("func", None, None, kwarg=None)
        
        assert key1 == key2
        assert len(key1) == 32
    
    def test_cache_key_generation_with_boolean_values(self):
        """Test cache key generation with boolean values"""
        key1 = get_cache_key("func", True, False, kwarg=True)
        key2 = get_cache_key("func", True, False, kwarg=True)
        
        assert key1 == key2
        assert len(key1) == 32
    
    def test_cache_key_generation_with_numeric_values(self):
        """Test cache key generation with numeric values"""
        key1 = get_cache_key("func", 42, 3.14, kwarg=100)
        key2 = get_cache_key("func", 42, 3.14, kwarg=100)
        
        assert key1 == key2
        assert len(key1) == 32
