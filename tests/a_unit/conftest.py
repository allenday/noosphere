"""Configuration for unit tests."""
import pytest
import io
import sys
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Mark all tests in this directory as unit tests
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.unit)

@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def captured_logs():
    """Capture logs to a string buffer."""
    from loguru import logger
    
    string_io = io.StringIO()
    handler_id = logger.add(string_io, level="DEBUG")
    
    yield string_io
    
    logger.remove(handler_id)

@contextmanager
def reset_loguru():
    """Reset loguru configuration after test."""
    from loguru import logger
    import logging
    
    # Store original stderr
    orig_stderr = sys.stderr
    
    # Store original handlers
    orig_handlers = logging.root.handlers.copy()
    
    try:
        yield
    finally:
        # Reset loguru
        logger.remove()
        
        # Add default handler back
        logger.add(sys.stderr)
        
        # Restore stderr
        sys.stderr = orig_stderr
        
        # Restore original logging handlers
        logging.root.handlers = orig_handlers

@pytest.fixture
def reset_logging():
    """Reset logging configuration after test."""
    with reset_loguru():
        yield