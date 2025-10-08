# tests/test_seniority.py
import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.featurizers import extract_seniority, SeniorityAdder

class TestExtractSeniority:
    """Test the extract_seniority function"""
    
    @pytest.mark.parametrize("title,expected", [
        # Intern cases
        ("Software Engineer Intern", "intern"),
        ("Data Analyst (Internship)", "intern"),
        ("ML Engineer Co-op", "intern"),
        ("Software Engineer Co Op", "intern"),
        
        # Junior/Entry cases
        ("Junior Software Engineer", "junior"),
        ("Jr. Data Scientist", "junior"),
        ("Entry Level Software Engineer", "entry"),
        ("Entry-Level ML Engineer", "entry"),
        ("Associate Data Engineer", "entry"),
        
        # Numeric levels
        ("Software Engineer I", "mid"),
        ("Software Engineer II", "senior"),
        ("Software Engineer III", "senior"),
        ("Software Engineer IV", "senior"),
        ("Software Engineer V", "senior"),
        ("Backend Engineer L1", "mid"),
        ("Backend Engineer L2", "senior"),
        ("Backend Engineer L3", "senior"),
        
        # Senior cases
        ("Senior Software Engineer", "senior"),
        ("Sr. Data Scientist", "senior"),
        
        # Staff cases
        ("Staff Software Engineer", "staff"),
        
        # Principal cases
        ("Principal Engineer", "principal"),
        
        # Lead cases
        ("Lead Software Engineer", "lead"),
        ("Tech Lead", "lead"),
        
        # Management cases
        ("Engineering Manager", "manager"),
        ("Mgr. Data Science", "manager"),
        ("Head of Engineering", "manager"),
        ("Director of Engineering", "director"),
        ("Dir. of ML", "director"),
        
        # VP cases
        ("VP of Engineering", "vp"),
        ("SVP of Data", "vp"),
        ("AVP of AI", "vp"),
        
        # C-level cases
        ("CEO", "cxo"),
        ("CTO", "cxo"),
        ("CFO", "cxo"),
        ("Chief Data Officer", "cxo"),
        
        # Default case
        ("Software Engineer", "mid"),
        ("Data Scientist", "mid"),
    ])
    def test_extract_seniority(self, title, expected):
        """Test seniority extraction from job titles"""
        result = extract_seniority(title)
        assert result == expected, f"Expected {expected} for '{title}', got {result}"

class TestSeniorityAdder:
    """Test the SeniorityAdder transformer"""
    
    def test_seniority_adder_fit_transform(self):
        """Test that SeniorityAdder correctly adds seniority column"""
        import pandas as pd
        
        # Create test data
        data = {
            "Job Title": [
                "Software Engineer Intern",
                "Senior Software Engineer", 
                "Engineering Manager",
                "Data Scientist"
            ],
            "Location": ["San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX"]
        }
        df = pd.DataFrame(data)
        
        # Test the transformer
        adder = SeniorityAdder(title_col="Job Title")
        result = adder.fit_transform(df)
        
        # Check that seniority column was added
        assert "seniority" in result.columns
        assert result["seniority"].tolist() == ["intern", "senior", "manager", "mid"]
        
    def test_seniority_adder_custom_column(self):
        """Test SeniorityAdder with custom title column name"""
        import pandas as pd
        
        data = {
            "Position": ["Junior Developer", "Staff Engineer"],
            "Location": ["SF", "NYC"]
        }
        df = pd.DataFrame(data)
        
        adder = SeniorityAdder(title_col="Position")
        result = adder.fit_transform(df)
        
        assert "seniority" in result.columns
        assert result["seniority"].tolist() == ["junior", "staff"]
        
    def test_seniority_adder_fit_returns_self(self):
        """Test that fit method returns self"""
        adder = SeniorityAdder()
        result = adder.fit(None)
        assert result is adder