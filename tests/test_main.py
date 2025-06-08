"""
Test suite for main module
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from main import main

def test_main_runs():
    """Test that main function runs without errors"""
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"main() raised {e} unexpectedly!")

def test_project_structure():
    """Test that project structure exists"""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # Check if key directories exist
    assert os.path.exists(os.path.join(project_root, 'src'))
    assert os.path.exists(os.path.join(project_root, 'tests'))
    assert os.path.exists(os.path.join(project_root, 'README.md'))

if __name__ == "__main__":
    test_main_runs()
    test_project_structure()
    print("✅ All tests passed!")
