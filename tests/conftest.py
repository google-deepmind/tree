"""Shared pytest fixtures and configuration for the tree test suite."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config() -> dict:
    """Provide a mock configuration for testing.
    
    Returns:
        dict: Mock configuration dictionary
    """
    return {
        "debug": False,
        "max_depth": 10,
        "enable_cache": True,
        "timeout": 30,
    }


@pytest.fixture
def sample_nested_data() -> dict:
    """Provide sample nested data structure for testing tree operations.
    
    Returns:
        dict: Sample nested dictionary
    """
    return {
        "level1": {
            "level2a": {
                "level3": ["item1", "item2", "item3"],
                "data": {"key": "value"},
            },
            "level2b": [1, 2, 3, 4, 5],
        },
        "metadata": {
            "version": "1.0",
            "created": "2024-01-01",
        },
        "items": ["a", "b", "c"],
    }


@pytest.fixture
def sample_tree_structure():
    """Provide a sample tree structure for testing.
    
    Returns:
        tuple: Sample tree with various nested structures
    """
    return (
        {"a": 1, "b": 2},
        [3, 4, 5],
        {"nested": {"deep": [6, 7, 8]}},
        "string",
        42,
    )


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_file_system(temp_dir: Path) -> Iterator[Path]:
    """Create a mock file system structure for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Yields:
        Path: Root of the mock file system
    """
    # Create directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "module1.py").write_text("# Module 1")
    (temp_dir / "src" / "module2.py").write_text("# Module 2")
    
    (temp_dir / "data").mkdir()
    (temp_dir / "data" / "input.txt").write_text("test data")
    
    (temp_dir / "config.json").write_text('{"setting": "value"}')
    
    yield temp_dir


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return log messages.
    
    Args:
        caplog: pytest's built-in log capture fixture
        
    Returns:
        list: List of captured log records
    """
    with caplog.at_level("DEBUG"):
        yield caplog.records


@pytest.fixture(scope="session")
def shared_resource():
    """Provide a shared resource that's expensive to create.
    
    This fixture is created once per test session.
    
    Returns:
        object: Shared resource instance
    """
    # Placeholder for expensive resource initialization
    resource = {"initialized": True, "data": list(range(1000))}
    yield resource
    # Cleanup if needed
    resource.clear()


# Pytest hooks for test collection and reporting

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: Mark test as slow-running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        # Add slow marker for specific test patterns
        if "benchmark" in item.name or "stress" in item.name:
            item.add_marker(pytest.mark.slow)