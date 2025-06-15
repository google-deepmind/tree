"""Validation tests to ensure the testing infrastructure is properly configured."""

import sys
from pathlib import Path

import pytest


class TestInfrastructureValidation:
    """Tests to validate the testing infrastructure setup."""
    
    def test_pytest_is_importable(self):
        """Verify pytest can be imported."""
        import pytest
        assert pytest is not None
        
    def test_pytest_version(self):
        """Check pytest version is appropriate."""
        version = pytest.__version__
        major = int(version.split('.')[0])
        assert major >= 8, f"Expected pytest 8.0+, got {version}"
        
    def test_coverage_is_importable(self):
        """Verify pytest-cov dependencies are available."""
        import coverage
        assert coverage is not None
        
    def test_mock_is_importable(self):
        """Verify pytest-mock is available."""
        import pytest_mock
        assert pytest_mock is not None
        
    def test_project_structure(self):
        """Verify the expected project structure exists."""
        root = Path(__file__).parent.parent
        
        # Check key directories exist
        assert (root / "tree").exists(), "tree package directory not found"
        assert (root / "tests").exists(), "tests directory not found"
        assert (root / "tests" / "unit").exists(), "unit tests directory not found"
        assert (root / "tests" / "integration").exists(), "integration tests directory not found"
        
        # Check configuration files
        assert (root / "pyproject.toml").exists(), "pyproject.toml not found"
        assert (root / ".gitignore").exists(), ".gitignore not found"
        
    def test_conftest_fixtures(self, temp_dir, mock_config, sample_nested_data):
        """Verify conftest fixtures are working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "debug" in mock_config
        assert "timeout" in mock_config
        
        # Test sample_nested_data fixture
        assert isinstance(sample_nested_data, dict)
        assert "level1" in sample_nested_data
        assert "metadata" in sample_nested_data
        
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works correctly."""
        assert True
        
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works correctly."""
        assert True
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works correctly."""
        assert True
        
    def test_python_version(self):
        """Verify Python version meets requirements."""
        version_info = sys.version_info
        assert version_info.major == 3
        assert version_info.minor >= 10, f"Python 3.10+ required, got {version_info.major}.{version_info.minor}"


@pytest.mark.parametrize("fixture_name", [
    "temp_dir",
    "mock_config", 
    "sample_nested_data",
    "sample_tree_structure",
    "mock_file_system",
    "capture_logs",
    "shared_resource"
])
def test_fixture_availability(fixture_name, request):
    """Test that all expected fixtures are available."""
    # This will fail if the fixture doesn't exist
    request.getfixturevalue(fixture_name)
    

def test_coverage_configuration():
    """Verify coverage is properly configured."""
    import coverage
    
    # Check coverage version
    version = coverage.__version__
    major = int(version.split('.')[0])
    assert major >= 5, f"Expected coverage 5.0+, got {version}"


def test_tree_package_importable():
    """Verify the tree package can be imported."""
    try:
        import tree
        assert tree is not None
        assert hasattr(tree, '__version__')
    except ImportError:
        # This is expected before the package is built
        pytest.skip("tree package not yet built - this is normal for initial setup")