# tests/test_agent.py - Comprehensive tests for agent.py
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.agent import CompanyAgent
from app.providers import Source

class TestCompanyAgent:
    """Test CompanyAgent class"""
    
    def test_init(self):
        """Test CompanyAgent initialization"""
        agent = CompanyAgent(cache_ttl_hours=12)
        assert agent.cache == {}
        assert agent.cache_ttl == timedelta(hours=12)
    
    def test_mkkey(self):
        """Test key generation"""
        agent = CompanyAgent()
        assert agent._mkkey("Test Company") == "testcompany"
        assert agent._mkkey("Test-Company Inc.") == "testcompanyinc"
        assert agent._mkkey("") == ""
        assert agent._mkkey(None) == ""
    
    def test_lookup_empty_company_name(self):
        """Test lookup with empty company name"""
        agent = CompanyAgent()
        result = agent.lookup("")
        assert result["_error"] == "empty company_name"
        
        result = agent.lookup(None)
        assert result["_error"] == "empty company_name"
    
    def test_lookup_cache_hit(self):
        """Test lookup with cache hit"""
        agent = CompanyAgent()
        test_data = {"name": "Test Company", "sector": "Technology"}
        agent.cache["testcompany"] = (test_data, datetime.now())
        
        result = agent.lookup("Test Company")
        assert result == test_data
    
    def test_lookup_cache_expired(self):
        """Test lookup with expired cache"""
        agent = CompanyAgent(cache_ttl_hours=1)
        test_data = {"name": "Test Company", "sector": "Technology"}
        # Set cache with expired timestamp
        expired_time = datetime.now() - timedelta(hours=2)
        agent.cache["testcompany"] = (test_data, expired_time)
        
        with patch('app.agent.fetch_company_profile_fast') as mock_fetch:
            mock_fetch.return_value = ({"sector": "New Technology"}, [])
            result = agent.lookup("Test Company")
            
        # Should fetch new data and update cache
        mock_fetch.assert_called_once_with("Test Company")
        assert "Sector" in result
    
    @patch('app.agent.fetch_company_profile_fast')
    def test_lookup_success(self, mock_fetch):
        """Test successful lookup"""
        
        mock_fetch.return_value = (
            {
                "sector": "Information Technology",
                "ownership": "Company - Public",
                "employees": 1000,
                "founded": 2020,
                "hq_city": "San Francisco",
                "hq_state": "CA"
            },
            [Source(url="https://example.com", title="Test", snippet="Test snippet")]
        )
        
        agent = CompanyAgent()
        result = agent.lookup("Test Company")
        
        assert result["Sector"] == "Information Technology"
        assert result["Type of ownership"] == "Company - Public"
        assert result["age"] is not None  # computed from founded year
        assert result["hq_city"] == "San Francisco"
        assert result["hq_state"] == "CA"
        assert len(result["__sources__"]) == 1
        assert result["_diagnostics"] == {}
        assert result["_canonical_website"] is None
    
    @patch('app.agent.fetch_company_profile_fast')
    def test_lookup_with_website(self, mock_fetch):
        """Test lookup with website information"""
        mock_fetch.return_value = (
            {
                "sector": "Technology",
                "_website": "example.com"
            },
            []
        )
        
        agent = CompanyAgent()
        result = agent.lookup("Test Company")
        
        assert result["_canonical_website"] == "example.com"
    
    @patch('app.agent.fetch_company_profile_fast')
    def test_lookup_with_diagnostics(self, mock_fetch):
        """Test lookup with diagnostics information"""
        mock_fetch.return_value = (
            {
                "sector": "Technology",
                "_diagnostics": {"path": "wiki", "url": "https://wiki.com"}
            },
            []
        )
        
        agent = CompanyAgent()
        result = agent.lookup("Test Company")
        
        assert result["_diagnostics"]["path"] == "wiki"
        assert result["_diagnostics"]["url"] == "https://wiki.com"
    
    @patch('app.agent.fetch_company_profile_fast')
    def test_lookup_fetch_exception(self, mock_fetch):
        """Test lookup with fetch exception"""
        mock_fetch.side_effect = Exception("Network error")
        
        agent = CompanyAgent()
        result = agent.lookup("Test Company")
        
        assert result["_error"] == "fetch_failed: Network error"
    
    @patch('app.agent.fetch_company_profile_fast')
    def test_lookup_with_age_computation(self, mock_fetch):
        """Test lookup with age computation"""
        mock_fetch.return_value = (
            {
                "sector": "Technology",
                "founded": 2020
            },
            []
        )
        
        agent = CompanyAgent()
        result = agent.lookup("Test Company")
        
        # Age should be computed from founded year
        assert result["age"] == datetime.now().year - 2020
    
    @patch('app.agent.fetch_company_profile_fast')
    def test_lookup_with_company_age_field(self, mock_fetch):
        """Test lookup with company_age field instead of founded"""
        mock_fetch.return_value = (
            {
                "sector": "Technology",
                "company_age": 5
            },
            []
        )
        
        agent = CompanyAgent()
        result = agent.lookup("Test Company")
        
        assert result["age"] == 5
    
    def test_clear_cache(self):
        """Test cache clearing"""
        agent = CompanyAgent()
        agent.cache["test"] = ({"data": "test"}, datetime.now())
        
        assert len(agent.cache) == 1
        agent.clear_cache()
        assert len(agent.cache) == 0
    
    def test_cleanup_expired_no_expired(self):
        """Test cleanup with no expired entries"""
        agent = CompanyAgent()
        agent.cache["test"] = ({"data": "test"}, datetime.now())
        
        cleaned = agent.cleanup_expired()
        assert cleaned == 0
        assert len(agent.cache) == 1
    
    def test_cleanup_expired_with_expired(self):
        """Test cleanup with expired entries"""
        agent = CompanyAgent(cache_ttl_hours=1)
        # Add current entry
        agent.cache["current"] = ({"data": "current"}, datetime.now())
        # Add expired entry
        expired_time = datetime.now() - timedelta(hours=2)
        agent.cache["expired"] = ({"data": "expired"}, expired_time)
        
        cleaned = agent.cleanup_expired()
        assert cleaned == 1
        assert len(agent.cache) == 1
        assert "current" in agent.cache
        assert "expired" not in agent.cache
    
    def test_cleanup_expired_all_expired(self):
        """Test cleanup with all entries expired"""
        agent = CompanyAgent(cache_ttl_hours=1)
        expired_time = datetime.now() - timedelta(hours=2)
        agent.cache["expired1"] = ({"data": "expired1"}, expired_time)
        agent.cache["expired2"] = ({"data": "expired2"}, expired_time)
        
        cleaned = agent.cleanup_expired()
        assert cleaned == 2
        assert len(agent.cache) == 0
    
    def test_lookup_caching_behavior(self):
        """Test that successful lookups are cached"""
        with patch('app.agent.fetch_company_profile_fast') as mock_fetch:
            mock_fetch.return_value = ({"sector": "Technology"}, [])
            
            agent = CompanyAgent()
            
            # First lookup should call fetch
            result1 = agent.lookup("Test Company")
            assert mock_fetch.call_count == 1
            
            # Second lookup should use cache
            result2 = agent.lookup("Test Company")
            assert mock_fetch.call_count == 1  # No additional calls
            assert result1 == result2
    
    def test_lookup_different_companies(self):
        """Test lookup with different companies"""
        with patch('app.agent.fetch_company_profile_fast') as mock_fetch:
            def mock_fetch_side_effect(company):
                if company == "Company A":
                    return ({"sector": "Tech A"}, [])
                elif company == "Company B":
                    return ({"sector": "Tech B"}, [])
                return ({}, [])
            
            mock_fetch.side_effect = mock_fetch_side_effect
            
            agent = CompanyAgent()
            
            result_a = agent.lookup("Company A")
            result_b = agent.lookup("Company B")
            
            assert result_a["Sector"] == "Tech A"
            assert result_b["Sector"] == "Tech B"
            assert len(agent.cache) == 2
    
    def test_lookup_case_insensitive_caching(self):
        """Test that caching is case insensitive"""
        with patch('app.agent.fetch_company_profile_fast') as mock_fetch:
            mock_fetch.return_value = ({"sector": "Technology"}, [])
            
            agent = CompanyAgent()
            
            # First lookup
            result1 = agent.lookup("Test Company")
            assert mock_fetch.call_count == 1
            
            # Second lookup with different case
            result2 = agent.lookup("test company")
            assert mock_fetch.call_count == 1  # Should use cache
            assert result1 == result2
    
    def test_lookup_with_special_characters(self):
        """Test lookup with special characters in company name"""
        with patch('app.agent.fetch_company_profile_fast') as mock_fetch:
            mock_fetch.return_value = ({"sector": "Technology"}, [])
            
            agent = CompanyAgent()
            
            # Test with various special characters
            result = agent.lookup("Test-Company Inc. & Co.")
            assert mock_fetch.call_count == 1
            assert "Sector" in result
    
    def test_lookup_empty_string_after_normalization(self):
        """Test lookup with string that becomes empty after normalization"""
        agent = CompanyAgent()
        
        # String with only special characters
        result = agent.lookup("---")
        assert result["_error"] == "empty company_name"
        
        # String with only spaces
        result = agent.lookup("   ")
        assert result["_error"] == "empty company_name"
