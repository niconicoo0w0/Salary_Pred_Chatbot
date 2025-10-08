# tests/test_training_pipeline.py
import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import training functions
from models.training_script.train_pipeline import (
    parse_salary_estimate, parse_size_to_min_max, compute_company_age,
    build_features, make_preprocessor
)

class TestSalaryParsing:
    """Test salary parsing functions"""
    
    def test_parse_salary_estimate_valid(self):
        """Test parsing valid salary estimates"""
        test_cases = [
            ("$50k - $70k", (50000, 70000, 60000)),
            ("$100K - $150K", (100000, 150000, 125000)),
            ("$80k - $120k (Glassdoor est.)", (80000, 120000, 100000)),
            ("$60K - $90K (Employer est.)", (60000, 90000, 75000)),
        ]
        
        for salary_str, expected in test_cases:
            result = parse_salary_estimate(salary_str)
            assert result == expected, f"Failed for '{salary_str}': expected {expected}, got {result}"
            
    def test_parse_salary_estimate_invalid(self):
        """Test parsing invalid salary estimates"""
        invalid_cases = [
            "$50 per hour",
            "$50 /hr",
            "Not specified",
            "",
            None,
            "$50k - $70k per hour",
        ]
        
        for salary_str in invalid_cases:
            result = parse_salary_estimate(salary_str)
            assert result == (np.nan, np.nan, np.nan), f"Should return NaN for '{salary_str}', got {result}"

class TestSizeParsing:
    """Test company size parsing functions"""
    
    def test_parse_size_to_min_max_valid(self):
        """Test parsing valid company sizes"""
        test_cases = [
            ("501 to 1000 employees", (501, 1000)),
            ("10000+ employees", (10000, 10000)),
            ("50-200 employees", (50, 200)),
            ("1000 employees", (1000, 1000)),
        ]
        
        for size_str, expected in test_cases:
            result = parse_size_to_min_max(size_str)
            assert result == expected, f"Failed for '{size_str}': expected {expected}, got {result}"
            
    def test_parse_size_to_min_max_invalid(self):
        """Test parsing invalid company sizes"""
        invalid_cases = [
            "Unknown",
            "-",
            "N/A",
            "",
            None,
            "Not specified",
        ]
        
        for size_str in invalid_cases:
            result = parse_size_to_min_max(size_str)
            assert result == (np.nan, np.nan), f"Should return NaN for '{size_str}', got {result}"

class TestCompanyAge:
    """Test company age computation"""
    
    def test_compute_company_age_valid(self):
        """Test computing company age from founded year"""
        current_year = 2024
        test_cases = [
            (2020, 4),
            (2010, 14),
            (2000, 24),
            (1990, 34),
        ]
        
        for founded_year, expected_age in test_cases:
            result = compute_company_age(founded_year, current_year)
            assert result == expected_age, f"Failed for founded {founded_year}: expected {expected_age}, got {result}"
            
    def test_compute_company_age_invalid(self):
        """Test computing company age with invalid inputs"""
        invalid_cases = [
            None,
            "",
            "invalid",
            1800,  # Too old
            2030,  # Future year
        ]
        
        for founded in invalid_cases:
            result = compute_company_age(founded)
            assert np.isnan(result), f"Should return NaN for {founded}, got {result}"

class TestBuildFeatures:
    """Test feature building functions"""
    
    def test_build_features_basic(self):
        """Test basic feature building"""
        data = {
            "Job Title": ["Software Engineer", "Data Scientist"],
            "Location": ["San Francisco, CA", "New York, NY"],
            "Rating": [4.5, 4.2],
            "Sector": ["Information Technology", "Finance"],
            "Type of ownership": ["Public", "Private"],
            "Size": ["1000-5000 employees", "500-1000 employees"],
            "Founded": [2010, 2015],
            "Salary Estimate": ["$100k - $150k", "$80k - $120k"]
        }
        df = pd.DataFrame(data)
        
        result = build_features(df)
        
        # Check that required columns exist
        assert "avg_salary" in result.columns
        assert "min_size" in result.columns
        assert "max_size" in result.columns
        assert "age" in result.columns
        assert "size_band" in result.columns
        
        # Check that salary was parsed
        assert not result["avg_salary"].isna().all()
        
        # Check that size was parsed
        assert not result["min_size"].isna().all()
        assert not result["max_size"].isna().all()
        
        # Check that age was computed
        assert not result["age"].isna().all()
        
    def test_build_features_missing_columns(self):
        """Test feature building with missing columns"""
        data = {
            "Job Title": ["Software Engineer"],
            "Location": ["San Francisco, CA"],
            "Rating": [4.5]
        }
        df = pd.DataFrame(data)
        
        result = build_features(df)
        
        # Should still work with missing columns
        assert "age" in result.columns
        assert "min_size" in result.columns
        assert "max_size" in result.columns
        assert "size_band" in result.columns

class TestPreprocessor:
    """Test the preprocessor creation"""
    
    def test_make_preprocessor(self):
        """Test that preprocessor is created correctly"""
        preprocessor = make_preprocessor()
        
        # Check that it's a ColumnTransformer
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)
        
        # Check that it has transformers
        assert len(preprocessor.transformers) > 0
        assert preprocessor.transformers[0][0] == "num"
        assert preprocessor.transformers[1][0] == "cat"
        
    def test_preprocessor_fit_transform(self):
        """Test that preprocessor can fit and transform data"""
        # Create test data
        data = {
            "Rating": [4.5, 4.2, 3.8],
            "age": [10, 15, 5],
            "Sector": ["Technology", "Finance", "Healthcare"],
            "Type of ownership": ["Public", "Private", "Public"],
            "size_band": ["Large", "Mid", "Small"],
            "seniority": ["senior", "mid", "junior"],
            "loc_tier": ["high", "mid", "low"]
        }
        df = pd.DataFrame(data)
        
        preprocessor = make_preprocessor()
        
        # Fit and transform
        result = preprocessor.fit_transform(df)
        
        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(df)  # Same number of rows
        
        # Check that there are no NaN values after preprocessing
        assert not np.isnan(result).any()
