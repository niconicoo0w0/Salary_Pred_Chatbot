# tests/test_jd_parsing.py
import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.jd_parsing import (
    ROLE_KEYWORDS, TITLE_REGEXES, NOISE_LINE_HINTS, CITY_REGEX,
    _TRAIN_SECTORS, _SECTOR_MAP
)

class TestJDParsingConstants:
    """Test the constants and patterns in jd_parsing module"""
    
    def test_role_keywords(self):
        """Test that ROLE_KEYWORDS contains expected roles"""
        expected_roles = [
            "artificial intelligence engineer", "ai engineer", "ml engineer",
            "data scientist", "software engineer", "data engineer"
        ]
        
        for role in expected_roles:
            assert role in ROLE_KEYWORDS, f"Missing role: {role}"
            
    def test_title_regexes(self):
        """Test that TITLE_REGEXES can match job titles"""
        import re
        
        test_cases = [
            "Job Title: Senior Software Engineer",
            "Title: Data Scientist",
            "Position: ML Engineer"
        ]
        
        for test_case in test_cases:
            matched = False
            for regex in TITLE_REGEXES:
                if re.search(regex, test_case):
                    matched = True
                    break
            assert matched, f"No regex matched: {test_case}"
            
    def test_noise_line_hints(self):
        """Test that NOISE_LINE_HINTS contains expected noise patterns"""
        expected_noise = [
            "logo", "share", "save", "easy apply", "promoted",
            "on-site", "remote", "full-time", "apply"
        ]
        
        for noise in expected_noise:
            assert noise in NOISE_LINE_HINTS, f"Missing noise hint: {noise}"
            
    def test_city_regex(self):
        """Test that CITY_REGEX matches valid city names"""
        valid_cities = [
            "San Francisco",
            "New York",
            "Los Angeles",
            "Austin",
            "Seattle",
            "Boston",
            "Chicago",
            "Denver",
            "Miami",
            "Portland"
        ]
        
        for city in valid_cities:
            assert CITY_REGEX.match(city), f"Valid city not matched: {city}"
            
        # Test invalid cities
        invalid_cities = [
            "123 City",  # Starts with number
            "A" * 50,    # Too long
            "",          # Empty
            "City-123",  # Contains number
        ]
        
        for city in invalid_cities:
            assert not CITY_REGEX.match(city), f"Invalid city matched: {city}"
            
    def test_train_sectors(self):
        """Test that _TRAIN_SECTORS contains expected sectors"""
        expected_sectors = [
            "Information Technology",
            "Finance", 
            "Health Care",
            "Education",
            "Government"
        ]
        
        for sector in expected_sectors:
            assert sector in _TRAIN_SECTORS, f"Missing sector: {sector}"
            
    def test_sector_map(self):
        """Test that _SECTOR_MAP contains expected mappings"""
        # Check that all train sectors are in the map
        for sector in _TRAIN_SECTORS:
            assert sector in _SECTOR_MAP, f"Missing sector in map: {sector}"
            
        # Check that mapped values are valid
        for sector, mapped in _SECTOR_MAP.items():
            assert mapped in _TRAIN_SECTORS, f"Invalid mapped sector: {mapped} for {sector}"
            
    def test_sector_mapping_consistency(self):
        """Test that sector mappings are consistent"""
        # Test some specific mappings
        test_mappings = [
            ("Technology", "Information Technology"),
            ("Tech", "Information Technology"),
            ("IT", "Information Technology"),
            ("Financial Services", "Finance"),
            ("Healthcare", "Health Care"),
            ("Education", "Education"),
        ]
        
        for input_sector, expected_output in test_mappings:
            if input_sector in _SECTOR_MAP:
                assert _SECTOR_MAP[input_sector] == expected_output, \
                    f"Wrong mapping for {input_sector}: expected {expected_output}, got {_SECTOR_MAP[input_sector]}"
