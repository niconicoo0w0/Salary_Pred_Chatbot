#!/usr/bin/env python3
"""
Simple test runner for the Salary Prediction Chatbot project.
Run this script to execute all tests.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run all tests in the tests directory"""
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    print("Running tests for Salary Prediction Chatbot...")
    print(f"Project root: {project_root}")
    print(f"Tests directory: {tests_dir}")
    print("-" * 50)
    
    # Run pytest on the tests directory
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(tests_dir),
            "-v",  # verbose output
            "--tb=short",  # shorter traceback format
            "--no-header",  # no pytest header
        ], cwd=project_root, capture_output=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ All tests passed!")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
