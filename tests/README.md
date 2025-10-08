# Tests for Salary Prediction Chatbot

This directory contains comprehensive tests for the Salary Prediction Chatbot project.

## Test Structure

### Core Module Tests

- **`test_seniority.py`** - Tests for seniority extraction functionality
  - Tests `extract_seniority()` function with various job titles
  - Tests `SeniorityAdder` transformer class
  - Covers intern, junior, senior, staff, principal, lead, manager, director, VP, and C-level positions

- **`test_featurizers.py`** - Tests for feature engineering transformers
  - Tests `LocationTierAdder` transformer
  - Tests location-based salary tier assignment
  - Tests handling of unknown locations
  - Tests caching and tier mapping functionality

- **`test_jd_parsing.py`** - Tests for job description parsing constants
  - Tests role keywords, title regexes, and noise patterns
  - Tests city name validation
  - Tests sector mapping consistency
  - Tests training sector definitions

- **`test_training_pipeline.py`** - Tests for training pipeline components
  - Tests salary parsing functions
  - Tests company size parsing
  - Tests company age computation
  - Tests feature building functions
  - Tests preprocessor creation and functionality

### Integration Tests

- **`test_app_integration.py`** - Tests for component integration
  - Tests featurizers working together
  - Tests module imports
  - Tests sklearn pipeline integration
  - Tests data consistency across components

## Running Tests

### Option 1: Using the test runner script
```bash
python run_tests.py
```

### Option 2: Using pytest directly
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_seniority.py -v

# Run specific test class
pytest tests/test_seniority.py::TestExtractSeniority -v

# Run specific test method
pytest tests/test_seniority.py::TestExtractSeniority::test_extract_seniority -v
```

### Option 3: Using pytest with coverage
```bash
pip install pytest-cov
pytest tests/ --cov=utils --cov=app --cov-report=html
```

## Test Design Principles

1. **Isolation**: Each test is independent and can run in any order
2. **Mocking**: External dependencies (like API calls) are mocked to avoid network calls
3. **Comprehensive Coverage**: Tests cover both happy path and edge cases
4. **Clear Assertions**: Each test has clear, specific assertions
5. **Descriptive Names**: Test names clearly describe what is being tested

## Test Data

Tests use synthetic data to avoid dependencies on external files:
- Generated salary data for location tier tests
- Sample job titles for seniority tests
- Test DataFrames for transformer tests

## Adding New Tests

When adding new functionality:

1. Create corresponding test methods in the appropriate test file
2. Follow the naming convention: `test_<functionality>`
3. Use descriptive test names that explain the expected behavior
4. Include both positive and negative test cases
5. Mock external dependencies
6. Add integration tests if the feature interacts with multiple components

## Test Requirements

- Python 3.8+
- pytest
- pandas
- numpy
- scikit-learn

Install test dependencies:
```bash
pip install pytest pandas numpy scikit-learn
```
