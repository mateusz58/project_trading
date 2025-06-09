#!/bin/bash
# Test runner script

echo "Running Application Tests..."

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run unit tests
echo "=== Unit Tests ==="
python -m pytest tests/ -v --tb=short

# Run with coverage if available
if command -v coverage >/dev/null 2>&1; then
    echo "=== Coverage Report ==="
    coverage run -m pytest tests/
    coverage report -m
    coverage html
    echo "HTML coverage report generated in htmlcov/"
fi

echo "Tests completed!"
