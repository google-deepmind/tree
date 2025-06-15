# Claude Code Project Guide

## Project Overview
This is the dm-tree project, a library for working with nested data structures developed by DeepMind.

## Testing Infrastructure

### Package Manager
This project uses **Poetry** for dependency management. The configuration is in `pyproject.toml`.

### Running Tests
To run tests, use one of these commands:
```bash
poetry run test
poetry run tests
```

Both commands are equivalent and will run the entire test suite with coverage reporting.

### Test Organization
- `tests/` - Main test directory
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/conftest.py` - Shared pytest fixtures

### Test Markers
The following pytest markers are available:
- `@pytest.mark.unit` - Mark test as a unit test
- `@pytest.mark.integration` - Mark test as an integration test  
- `@pytest.mark.slow` - Mark test as slow-running

### Coverage
Coverage reporting is configured with:
- HTML report: `htmlcov/`
- XML report: `coverage.xml`
- Coverage threshold: Currently set to 0% (should be increased to 80% once actual tests are added)

### Available Fixtures
The following fixtures are available in `conftest.py`:
- `temp_dir` - Temporary directory for test files
- `mock_config` - Mock configuration dictionary
- `sample_nested_data` - Sample nested data structure
- `sample_tree_structure` - Sample tree structure
- `mock_file_system` - Mock file system structure
- `capture_logs` - Log capture utility
- `shared_resource` - Session-scoped shared resource

## Development Commands

### Install Dependencies
```bash
poetry install --with dev
```

### Run Tests
```bash
poetry run test
# or
poetry run tests
```

### Run Tests with Options
All standard pytest options are available:
```bash
poetry run pytest -v  # Verbose output
poetry run pytest -k "test_name"  # Run specific test
poetry run pytest -m unit  # Run only unit tests
poetry run pytest -m integration  # Run only integration tests
```

## Build System
The project uses CMake for building C++ extensions. The build configuration is handled through `setup.py` and the CMake files in the `tree/` directory.

## Code Quality
Before committing changes, ensure tests pass:
```bash
poetry run test
```