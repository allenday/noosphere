# Unit Tests

This directory contains unit tests for the Telegram Export Processor.

Unit tests focus on testing individual components in isolation, with mocked dependencies.

## Running Unit Tests

```bash
# Run all unit tests
pytest tests/a_unit

# Run a specific test file
pytest tests/a_unit/test_config.py

# Run a specific test
pytest tests/a_unit/test_config.py::test_function_name
```

Unit tests should be fast, deterministic, and independent of external services.