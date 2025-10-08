# tests/test_app_integration.py
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

class TestAppIntegration:
    """Test integration between different app components"""
    
    def test_featurizers_integration(self):
        """Test that featurizers work together"""
        from utils.featurizers import SeniorityAdder, LocationTierAdder
        
        # Create test data
        data = {
            "Job Title": [
                "Senior Software Engineer",
                "Junior Data Scientist", 
                "Engineering Manager",
                "Staff ML Engineer"
            ],
            "Location": [
                "San Francisco, CA",
                "Austin, TX", 
                "New York, NY",
                "Seattle, WA"
            ],
            "Rating": [4.5, 4.2, 4.0, 4.8],
            "Sector": ["Technology", "Technology", "Technology", "Technology"],
            "Type of ownership": ["Public", "Public", "Public", "Public"],
            "size_band": ["Large", "Mid", "Large", "Large"]
        }
        df = pd.DataFrame(data)
        
        # Create salary data for LocationTierAdder
        salaries = [150000, 100000, 140000, 160000]
        
        # Test SeniorityAdder
        seniority_adder = SeniorityAdder(title_col="Job Title")
        result1 = seniority_adder.fit_transform(df)
        assert "seniority" in result1.columns
        assert result1["seniority"].tolist() == ["senior", "junior", "manager", "staff"]
        
        # Test LocationTierAdder
        location_adder = LocationTierAdder(loc_col="Location", drop_location=True)
        result2 = location_adder.fit_transform(result1, salaries)
        assert "loc_tier" in result2.columns
        assert "Location" not in result2.columns
        
    def test_utils_imports(self):
        """Test that all utils modules can be imported"""
        try:
            from utils import featurizers
            from utils import helpers
            from utils import jd_parsing
            from utils import us_locations
            from utils import constants
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import utils modules: {e}")
            
    def test_app_imports(self):
        """Test that app modules can be imported"""
        try:
            from app import agent
            from app import providers
            from app import app
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import app modules: {e}")
            
    def test_training_script_imports(self):
        """Test that training script can be imported"""
        try:
            from models.training_script import train_pipeline
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import training script: {e}")
            
    def test_featurizers_with_sklearn_pipeline(self):
        """Test that featurizers work in sklearn pipeline"""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import GradientBoostingRegressor
        from utils.featurizers import SeniorityAdder, LocationTierAdder
        
        # Create test data
        data = {
            "Job Title": ["Software Engineer", "Data Scientist"] * 10,
            "Location": ["San Francisco, CA", "Austin, TX"] * 10,
            "Rating": [4.5, 4.2] * 10,
            "Sector": ["Technology", "Finance"] * 10,
            "Type of ownership": ["Public", "Private"] * 10,
            "size_band": ["Large", "Mid"] * 10
        }
        df = pd.DataFrame(data)
        
        # Create target variable
        y = np.random.normal(100000, 20000, len(df))
        
        # Create pipeline
        pipeline = Pipeline([
            ("seniority", SeniorityAdder(title_col="Job Title")),
            ("location", LocationTierAdder(loc_col="Location", drop_location=True)),
            ("regressor", GradientBoostingRegressor(n_estimators=10, random_state=42))
        ])
        
        # Fit pipeline
        pipeline.fit(df, y)
        
        # Make predictions
        predictions = pipeline.predict(df)
        
        # Check that predictions are reasonable
        assert len(predictions) == len(df)
        assert not np.isnan(predictions).any()
        assert predictions.min() > 0  # Salaries should be positive
        
    def test_data_consistency(self):
        """Test that data flows consistently through the pipeline"""
        from utils.featurizers import extract_seniority
        from utils.jd_parsing import ROLE_KEYWORDS
        
        # Test that seniority extraction works with role keywords
        for role in ROLE_KEYWORDS[:5]:  # Test first 5 roles
            seniority = extract_seniority(role)
            assert seniority in ["intern", "junior", "entry", "mid", "senior", "staff", 
                               "principal", "lead", "manager", "director", "vp", "cxo"]
            
    def test_constants_consistency(self):
        """Test that constants are consistent across modules"""
        from utils.jd_parsing import _TRAIN_SECTORS, _SECTOR_MAP
        
        # Check that all mapped sectors are in train sectors
        for sector, mapped_sector in _SECTOR_MAP.items():
            assert mapped_sector in _TRAIN_SECTORS, \
                f"Mapped sector {mapped_sector} not in train sectors for {sector}"
