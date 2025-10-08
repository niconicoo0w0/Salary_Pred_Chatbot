# tests/test_featurizers.py
import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.featurizers import LocationTierAdder

class TestLocationTierAdder:
    """Test the LocationTierAdder transformer"""
    
    def test_location_tier_adder_fit_transform(self):
        """Test that LocationTierAdder correctly adds location tier column"""
        # Create test data with different salary levels
        data = {
            "Location": ["San Francisco, CA", "New York, NY", "Austin, TX", "Detroit, MI"] * 10,
            "Job Title": ["Software Engineer"] * 40
        }
        df = pd.DataFrame(data)
        
        # Create salary data that varies by location
        salaries = []
        for i, loc in enumerate(df["Location"]):
            if "San Francisco" in loc:
                salaries.append(150000)  # High salary
            elif "New York" in loc:
                salaries.append(130000)  # High salary
            elif "Austin" in loc:
                salaries.append(100000)  # Medium salary
            else:  # Detroit
                salaries.append(80000)   # Low salary
                
        y = np.array(salaries)
        
        # Test the transformer
        adder = LocationTierAdder(loc_col="Location", drop_location=True)
        result = adder.fit_transform(df, y)
        
        # Check that location tier column was added
        assert "loc_tier" in result.columns
        assert "Location" not in result.columns  # Should be dropped
        
        # Check that tiers are assigned correctly
        sf_tiers = result[df["Location"] == "San Francisco, CA"]["loc_tier"].unique()
        ny_tiers = result[df["Location"] == "New York, NY"]["loc_tier"].unique()
        austin_tiers = result[df["Location"] == "Austin, TX"]["loc_tier"].unique()
        detroit_tiers = result[df["Location"] == "Detroit, MI"]["loc_tier"].unique()
        
        # Each location should have consistent tier assignment
        assert len(sf_tiers) == 1
        assert len(ny_tiers) == 1
        assert len(austin_tiers) == 1
        assert len(detroit_tiers) == 1
        
        # Check that quantiles were computed
        assert adder.quantiles_ is not None
        assert len(adder.quantiles_) == 3  # q25, q50, q75
        
    def test_location_tier_adder_keep_location(self):
        """Test LocationTierAdder with drop_location=False"""
        data = {
            "Location": ["San Francisco, CA", "Austin, TX"],
            "Job Title": ["Software Engineer", "Data Scientist"]
        }
        df = pd.DataFrame(data)
        y = np.array([150000, 100000])
        
        adder = LocationTierAdder(loc_col="Location", drop_location=False)
        result = adder.fit_transform(df, y)
        
        # Location column should be kept
        assert "Location" in result.columns
        assert "loc_tier" in result.columns
        
    def test_location_tier_adder_unknown_location(self):
        """Test LocationTierAdder with unknown location in transform"""
        # Fit on known locations
        train_data = {
            "Location": ["San Francisco, CA", "Austin, TX"],
            "Job Title": ["Software Engineer", "Data Scientist"]
        }
        train_df = pd.DataFrame(train_data)
        train_y = np.array([150000, 100000])
        
        adder = LocationTierAdder(loc_col="Location", drop_location=True)
        adder.fit(train_df, train_y)
        
        # Transform with unknown location
        test_data = {
            "Location": ["Unknown City, XX"],
            "Job Title": ["Software Engineer"]
        }
        test_df = pd.DataFrame(test_data)
        
        result = adder.transform(test_df)
        
        # Unknown location should get default "mid" tier
        assert result["loc_tier"].iloc[0] == "mid"
        
    def test_location_tier_adder_tier_mapping(self):
        """Test that tier mapping is created correctly"""
        data = {
            "Location": ["High City", "Mid City", "Low City"] * 5,
            "Job Title": ["Engineer"] * 15
        }
        df = pd.DataFrame(data)
        
        # Create clear salary separation
        salaries = []
        for loc in df["Location"]:
            if "High" in loc:
                salaries.append(200000)
            elif "Mid" in loc:
                salaries.append(100000)
            else:  # Low
                salaries.append(50000)
        y = np.array(salaries)
        
        adder = LocationTierAdder(loc_col="Location", drop_location=True)
        adder.fit(df, y)
        
        # Check tier mapping
        assert "High City" in adder.tier_map_
        assert "Mid City" in adder.tier_map_
        assert "Low City" in adder.tier_map_
        
        # Verify tier assignments make sense
        high_tier = adder.tier_map_["High City"]
        mid_tier = adder.tier_map_["Mid City"]
        low_tier = adder.tier_map_["Low City"]
        
        # High salary city should get higher tier
        assert high_tier in ["high", "very_high"]
        assert low_tier in ["low", "mid"]
