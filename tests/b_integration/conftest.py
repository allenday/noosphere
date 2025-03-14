"""Configuration for integration tests."""
import pytest
import io
import sys
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Mark all tests in this directory as integration tests
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.integration)
        
# Skip coverage reporting for integration tests when run individually
def pytest_configure(config):
    config.option.no_cov = True

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

@pytest.fixture
def minimal_config():
    """Create a minimal configuration file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
        config_yaml = """
        logging:
          level: "DEBUG"
          distributed: false
          log_file: null
          json_logs: false
          enqueue: false
        
        llm_services:
          - name: test_llm
            type: "ollama"
            url: "http://localhost:11434"
            models:
              - name: test_model
                model: "test"
                default_prompt: "test"
        
        embedding_services:
          - name: test_embedding
            type: "ollama"
            url: "http://localhost:11434"
            models:
              - name: test_model
                model: "test"
                vector_size: 128
        
        text_vectorizer:
          embedding_service: "test_embedding"
          model: "test_model"
          vector_size: 128
          output_type: "vector"
        
        text_summarizer:
          llm_service: "test_llm"
          model: "test_model"
          prompt: "test"
          output_type: "text"
        
        image_summarizer:
          llm_service: "test_llm"
          model: "test_model"
          prompt: "test"
          output_type: "text"
        
        text_vector_storage:
          url: "http://localhost:6333"
          collection: "test"
          vector_size: 128
        
        image_vector_storage:
          url: "http://localhost:6333"
          collection: "test"
          vector_size: 128
        
        window_by_session:
          timeout_seconds: 3600
          min_size: 1
          max_size: 10
        
        window_by_size:
          size: 5
          overlap: 2
        """
        tmp.write(config_yaml.encode('utf-8'))
        config_path = tmp.name
    
    yield config_path
    
    # Clean up
    os.unlink(config_path)