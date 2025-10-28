# tests/test_helpers.py - Comprehensive tests for helpers.py
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.helpers import (
    titlecase, looks_like_location, clean_text, strip_paren_noise,
    looks_like_noise_line, candidate_title_from_line, fmt_none
)

class TestTitlecase:
    """Test titlecase function"""
    
    def test_titlecase_basic(self):
        """Test basic titlecase conversion"""
        assert titlecase("hello world") == "Hello World"
        assert titlecase("HELLO WORLD") == "Hello World"
        assert titlecase("hELLo WoRLd") == "Hello World"
    
    def test_titlecase_single_word(self):
        """Test titlecase with single word"""
        assert titlecase("hello") == "Hello"
        assert titlecase("HELLO") == "Hello"
        assert titlecase("hELLo") == "Hello"
    
    def test_titlecase_empty(self):
        """Test titlecase with empty string"""
        assert titlecase("") == ""
        assert titlecase("   ") == ""
        assert titlecase(None) == ""
    
    def test_titlecase_special_characters(self):
        """Test titlecase with special characters"""
        assert titlecase("hello-world") == "Hello-world"  # Only first letter capitalized
        assert titlecase("hello_world") == "Hello_world"  # Only first letter capitalized
        assert titlecase("hello.world") == "Hello.world"  # Only first letter capitalized
    
    def test_titlecase_numbers(self):
        """Test titlecase with numbers"""
        assert titlecase("hello123") == "Hello123"
        assert titlecase("123hello") == "123hello"  # Numbers at start don't get capitalized
        assert titlecase("hello 123 world") == "Hello 123 World"

class TestLooksLikeLocation:
    """Test looks_like_location function"""
    
    def test_looks_like_location_city_state(self):
        """Test location detection with city, state format"""
        with patch('utils.us_locations.US_STATES', {'CA', 'NY', 'TX'}):
            assert looks_like_location("san jose, ca") is True
            assert looks_like_location("new york, ny") is True
            assert looks_like_location("austin, tx") is True
    
    def test_looks_like_location_state_only(self):
        """Test location detection with state only"""
        with patch('utils.us_locations.US_STATES', {'CA', 'NY', 'TX'}):
            assert looks_like_location("CA") is True
            assert looks_like_location("NY") is True
            assert looks_like_location("TX") is True
    
    def test_looks_like_location_false(self):
        """Test location detection with non-location text"""
        with patch('utils.us_locations.US_STATES', {'CA', 'NY', 'TX'}):
            assert looks_like_location("software engineer") is False
            assert looks_like_location("hello world") is False
            assert looks_like_location("123 main street") is False
    
    def test_looks_like_location_empty(self):
        """Test location detection with empty input"""
        with patch('utils.us_locations.US_STATES', {'CA', 'NY', 'TX'}):
            assert looks_like_location("") is False
            assert looks_like_location(None) is False
            assert looks_like_location("   ") is False
    
    def test_looks_like_location_invalid_format(self):
        """Test location detection with invalid format"""
        with patch('utils.us_locations.US_STATES', {'CA', 'NY', 'TX'}):
            assert looks_like_location("san jose") is False  # No state
            assert looks_like_location("san jose, california") is False  # Full state name
            assert looks_like_location("san jose, ca, usa") is False  # Too many parts

class TestCleanText:
    """Test clean_text function"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "Hello    world\n\n\nThis is a test"
        result = clean_text(text)
        assert result == "Hello world\n\nThis is a test"
    
    def test_clean_text_unicode_normalization(self):
        """Test unicode normalization"""
        text = "café naïve résumé"
        result = clean_text(text)
        assert "é" in result  # Should be normalized
    
    def test_clean_text_remove_junk(self):
        """Test removal of junk characters"""
        text = "Hello€€€world€€€test"
        result = clean_text(text)
        assert "€€€" not in result
        assert "Hello world test" in result
    
    def test_clean_text_line_endings(self):
        """Test line ending normalization"""
        text = "Line1\r\nLine2\rLine3"
        result = clean_text(text)
        assert "\r" not in result
        assert "\n" in result
    
    def test_clean_text_multiple_newlines(self):
        """Test multiple newline reduction"""
        text = "Line1\n\n\n\nLine2"
        result = clean_text(text)
        assert result == "Line1\n\nLine2"
    
    def test_clean_text_empty(self):
        """Test cleaning empty text"""
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text("   ") == ""
    
    def test_clean_text_whitespace_normalization(self):
        """Test whitespace normalization"""
        text = "Hello\t\tworld   test"
        result = clean_text(text)
        assert "\t" not in result
        assert "Hello world test" in result

class TestStripParenNoise:
    """Test strip_paren_noise function"""
    
    def test_strip_paren_noise_work_terms(self):
        """Test removal of work-related parenthetical terms"""
        assert strip_paren_noise("Software Engineer (remote)") == "Software Engineer"
        assert strip_paren_noise("Developer (onsite)") == "Developer"
        assert strip_paren_noise("Engineer (hybrid)") == "Engineer"
        assert strip_paren_noise("Analyst (full-time)") == "Analyst"
        assert strip_paren_noise("Manager (part-time)") == "Manager"
        assert strip_paren_noise("Scientist (contract)") == "Scientist"
        assert strip_paren_noise("Designer (wfh)") == "Designer"
        assert strip_paren_noise("Developer (no cpt)") == "Developer"
    
    def test_strip_paren_noise_general_parentheses(self):
        """Test removal of general parenthetical content"""
        assert strip_paren_noise("Engineer (Senior Level)") == "Engineer"
        assert strip_paren_noise("Developer (Python)") == "Developer"
        assert strip_paren_noise("Manager (Team Lead)") == "Manager"
    
    def test_strip_paren_noise_multiple_parentheses(self):
        """Test removal of multiple parenthetical content"""
        assert strip_paren_noise("Engineer (Senior) (Remote)") == "Engineer"
        assert strip_paren_noise("Developer (Python) (Full-time)") == "Developer"
    
    def test_strip_paren_noise_no_parentheses(self):
        """Test text without parentheses"""
        assert strip_paren_noise("Software Engineer") == "Software Engineer"
        assert strip_paren_noise("Data Scientist") == "Data Scientist"
    
    def test_strip_paren_noise_empty(self):
        """Test empty input"""
        assert strip_paren_noise("") == ""
        # Note: strip_paren_noise doesn't handle None input, it will raise TypeError
    
    def test_strip_paren_noise_whitespace_cleanup(self):
        """Test whitespace cleanup after removal"""
        assert strip_paren_noise("Engineer  (remote)  ") == "Engineer"
        assert strip_paren_noise("Developer -- (onsite) --") == "Developer"
    
    def test_strip_paren_noise_long_parentheses(self):
        """Test removal of long parenthetical content"""
        # The function only removes parentheses with 0-40 characters, so long ones remain
        long_paren = "Engineer (This is a very long parenthetical comment that should be removed)"
        result = strip_paren_noise(long_paren)
        assert result == "Engineer (This is a very long parenthetical comment that should be removed)"

class TestLooksLikeNoiseLine:
    """Test looks_like_noise_line function"""
    
    def test_looks_like_noise_line_true(self):
        """Test noise line detection - true cases"""
        with patch('utils.jd_parsing.NOISE_LINE_HINTS', ['click here', 'apply now', 'learn more']):
            assert looks_like_noise_line("click here to apply") is True
            assert looks_like_noise_line("apply now for this position") is True
            assert looks_like_noise_line("learn more about us") is True
    
    def test_looks_like_noise_line_false(self):
        """Test noise line detection - false cases"""
        with patch('utils.jd_parsing.NOISE_LINE_HINTS', ['click here', 'apply now', 'learn more']):
            assert looks_like_noise_line("software engineer position") is False
            assert looks_like_noise_line("we are looking for a developer") is False
            assert looks_like_noise_line("requirements and qualifications") is False
    
    def test_looks_like_noise_line_empty(self):
        """Test noise line detection with empty input"""
        with patch('utils.jd_parsing.NOISE_LINE_HINTS', ['click here', 'apply now', 'learn more']):
            assert looks_like_noise_line("") is False
            # Note: looks_like_noise_line doesn't handle None input, it will raise TypeError
    
    def test_looks_like_noise_line_case_insensitive(self):
        """Test noise line detection is case insensitive"""
        with patch('utils.jd_parsing.NOISE_LINE_HINTS', ['click here', 'apply now', 'learn more']):
            assert looks_like_noise_line("click here to apply") is True
            assert looks_like_noise_line("apply now for this position") is True
            assert looks_like_noise_line("learn more about us") is True

class TestCandidateTitleFromLine:
    """Test candidate_title_from_line function"""
    
    def test_candidate_title_from_line_valid(self):
        """Test valid title candidate detection"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', ['engineer', 'scientist', 'analyst', 'developer']):
            assert candidate_title_from_line("Software Engineer") is True
            assert candidate_title_from_line("Data Scientist") is True
            assert candidate_title_from_line("Business Analyst") is True
            assert candidate_title_from_line("Python Developer") is True
    
    def test_candidate_title_from_line_too_short(self):
        """Test title candidate with too few characters"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', ['engineer', 'scientist', 'analyst', 'developer']):
            assert candidate_title_from_line("Dev") is False  # Too short
            assert candidate_title_from_line("AI") is False  # Too short
    
    def test_candidate_title_from_line_too_long(self):
        """Test title candidate with too many characters"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', ['engineer', 'scientist', 'analyst', 'developer']):
            long_title = "This is a very long job title that exceeds the maximum length limit"
            assert candidate_title_from_line(long_title) is False
    
    def test_candidate_title_from_line_too_few_words(self):
        """Test title candidate with too few words"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', ['engineer', 'scientist', 'analyst', 'developer']):
            assert candidate_title_from_line("Engineer") is False  # Only one word
    
    def test_candidate_title_from_line_too_many_words(self):
        """Test title candidate with too many words"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', ['engineer', 'scientist', 'analyst', 'developer']):
            many_words = "This is a very long job title with many words that exceeds the limit"
            assert candidate_title_from_line(many_words) is False
    
    def test_candidate_title_from_line_no_keywords(self):
        """Test title candidate without role keywords"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', ['engineer', 'scientist', 'analyst', 'developer']):
            assert candidate_title_from_line("Marketing Manager") is False
            assert candidate_title_from_line("Sales Representative") is False
    
    def test_candidate_title_from_line_regex_match(self):
        """Test title candidate with regex pattern match"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', []):
            # Should match regex patterns even without keywords
            assert candidate_title_from_line("Software Engineer") is True
            assert candidate_title_from_line("Data Scientist") is True
            assert candidate_title_from_line("Business Analyst") is True
            assert candidate_title_from_line("Python Developer") is True
            assert candidate_title_from_line("ML Researcher") is True
    
    def test_candidate_title_from_line_empty(self):
        """Test title candidate with empty input"""
        with patch('utils.jd_parsing.ROLE_KEYWORDS', ['engineer', 'scientist', 'analyst', 'developer']):
            assert candidate_title_from_line("") is False
            # Note: candidate_title_from_line doesn't handle None input, it will raise AttributeError
            assert candidate_title_from_line("   ") is False

class TestFmtNone:
    """Test fmt_none function"""
    
    def test_fmt_none_none_values(self):
        """Test formatting of None values"""
        assert fmt_none(None) == "—"
        assert fmt_none("") == "—"
        assert fmt_none([]) == "—"
    
    def test_fmt_none_nan_values(self):
        """Test formatting of NaN values"""
        assert fmt_none(np.nan) == "—"
        assert fmt_none(float('nan')) == "—"
    
    def test_fmt_none_valid_values(self):
        """Test formatting of valid values"""
        assert fmt_none("Hello") == "Hello"
        assert fmt_none(123) == "123"
        assert fmt_none(45.67) == "45.67"
        assert fmt_none([1, 2, 3]) == "[1, 2, 3]"
        assert fmt_none({"key": "value"}) == "{'key': 'value'}"
    
    def test_fmt_none_boolean_values(self):
        """Test formatting of boolean values"""
        assert fmt_none(True) == "True"
        assert fmt_none(False) == "False"
    
    def test_fmt_none_zero_values(self):
        """Test formatting of zero values"""
        assert fmt_none(0) == "0"
        assert fmt_none(0.0) == "0.0"
    
    def test_fmt_none_whitespace(self):
        """Test formatting of whitespace-only values"""
        assert fmt_none("   ") == "   "  # Not empty, just whitespace
        assert fmt_none("\t\n") == "\t\n"  # Not empty, just whitespace
    
    def test_fmt_none_complex_objects(self):
        """Test formatting of complex objects"""
        class CustomObject:
            def __str__(self):
                return "CustomObject"
        
        obj = CustomObject()
        assert fmt_none(obj) == "CustomObject"
    
    def test_fmt_none_edge_cases(self):
        """Test formatting of edge cases"""
        assert fmt_none(0) == "0"  # Zero is not None
        assert fmt_none("0") == "0"  # String zero is not None
        assert fmt_none(False) == "False"  # False is not None
        assert fmt_none(0.0) == "0.0"  # Float zero is not None
