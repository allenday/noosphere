# Integration Tests

This directory contains integration tests for the Telegram Export Processor. 

Integration tests verify that multiple components work correctly together and with external dependencies.

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/b_integration

# Run a specific integration test
pytest tests/b_integration/test_specific_integration.py
```

Integration tests may require additional setup like:
- External API access
- Database connections
- Full pipeline configurations