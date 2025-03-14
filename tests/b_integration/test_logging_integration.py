"""Integration tests for the logging module."""
import io
import os
import tempfile
import pytest
import time
import json
from pathlib import Path
from contextlib import contextmanager

import apache_beam as beam
from apache_beam.testing.util import assert_that, equal_to

from telexp.logging import (
    setup_logging, 
    get_logger, 
    LoggingMixin,
    log_performance
)
from telexp.config import LoggingConfig


@contextmanager
def capture_logs():
    """Capture logs to a string buffer."""
    string_io = io.StringIO()
    from loguru import logger
    handler_id = logger.add(string_io, level="DEBUG")
    try:
        yield string_io
    finally:
        logger.remove(handler_id)


def test_logging_to_file():
    """Test that logs are correctly written to a file."""
    # Create a temporary directory for log files
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        
        # Setup logging to file
        setup_logging(
            level="DEBUG",
            log_file=log_file,
            json_logs=False,
            enqueue=False,  # Avoid SimpleQueue for Beam compatibility
        )
        
        # Get a logger and log some messages
        logger = get_logger("test_integration")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        # Verify log file exists and contains expected messages
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content


def test_json_logging():
    """Test that JSON logs are correctly formatted."""
    # Create a temporary directory for log files
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_json.log"
        
        # Setup logging with JSON format
        setup_logging(
            level="INFO",
            log_file=log_file,
            json_logs=True,
            enqueue=False,  # Avoid SimpleQueue for Beam compatibility
        )
        
        # Get a logger and log with context
        logger = get_logger("json_test")
        
        # Log a message with structured context
        logger.info("Structured message", 
                   test_field="value",
                   number=42)
        
        # Verify log file exists
        assert log_file.exists()
        
        # Use a retry loop because file writing might be async
        start_time = time.time()
        timeout = 3  # seconds
        log_entry = None
        
        while time.time() - start_time < timeout:
            # Read the log file
            log_content = log_file.read_text()
            if log_content:
                try:
                    # Try to parse the JSON
                    log_entry = json.loads(log_content)
                    break
                except json.JSONDecodeError:
                    # Wait a bit and try again
                    time.sleep(0.1)
        
        # Verify we got log content
        assert log_entry is not None, f"Failed to read valid JSON from log file after {timeout}s"
        
        # Verify JSON structure contains expected fields
        assert log_entry["message"] == "Structured message"
        assert "extra" in log_entry
        assert "test_field" in log_entry["extra"], f"Missing test_field in {log_entry}"
        assert log_entry["extra"]["test_field"] == "value"
        assert "number" in log_entry["extra"]
        assert log_entry["extra"]["number"] == 42


class SimpleBeamDoFn(beam.DoFn, LoggingMixin):
    """A simple DoFn for testing logging in Apache Beam."""
    
    def __init__(self):
        """Initialize the DoFn."""
        super().__init__()
        self._setup_called = False
        self._setup_logging()
    
    def __getstate__(self):
        """Control which attributes are pickled."""
        state = super().__getstate__()
        # Reset setup flag for unpickling
        state['_setup_called'] = False
        return state
    
    def setup(self):
        """Set up the DoFn."""
        if self._setup_called:
            return
            
        if not hasattr(self, 'logger'):
            self._setup_logging()
        self._setup_called = True
        self.logger.debug("DoFn setup complete")
    
    def teardown(self):
        """Clean up resources."""
        self._setup_called = False
        self.logger.debug("DoFn teardown complete")
    
    @log_performance(threshold_ms=1)
    def process(self, element):
        """Process a single element."""
        # Ensure setup is called
        if not self._setup_called:
            self.setup()
            
        self.logger.info(f"Processing element: {element}")
        time.sleep(0.01)  # Sleep for 10ms to trigger performance logging
        return [element * 2]


def test_logging_in_beam_pipeline():
    """Test that logging works correctly in a Beam pipeline."""
    # Capture logs to verify they're generated
    with capture_logs() as log_buffer:
        # Set up a simple pipeline
        with beam.Pipeline() as pipeline:
            # Create some data
            data = [1, 2, 3, 4, 5]
            
            # Process data
            result = (
                pipeline
                | "Create" >> beam.Create(data)
                | "Process" >> beam.ParDo(SimpleBeamDoFn())
            )
            
            # Verify results
            assert_that(result, equal_to([2, 4, 6, 8, 10]))
        
        # Verify logs contain expected messages
        log_content = log_buffer.getvalue()
        assert "Processing element: " in log_content
        assert "Slow operation detected" in log_content  # Performance logging
        
        # Verify the pipeline completed successfully
        assert "Pipeline execution complete" in log_content


def test_logging_config_integration():
    """Test that LoggingConfig works properly with the setup_logging function."""
    # Create a logging config
    config = LoggingConfig(
        level="DEBUG",
        distributed=True,
        json_logs=True,
        module_levels={
            "apache_beam": "WARNING",
            "httpx": "ERROR"
        }
    )
    
    # Create a temporary directory for log files
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "config_test.log"
        config.log_file = str(log_file)
        
        # Setup logging from config
        setup_logging(
            level=config.level,
            distributed=config.distributed,
            log_file=config.log_file,
            json_logs=config.json_logs,
            intercept_std_logging=config.intercept_std_logging,
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            enqueue=False,  # Override to avoid SimpleQueue issues
            log_startup=config.log_startup
        )
        
        # Get a logger and log some messages
        logger = get_logger("config_test")
        logger.debug("Debug from config")
        logger.info("Info from config", context={"source": "test"})
        
        # Verify log file exists
        assert log_file.exists()
        
        # Read the log file and verify it's JSON format
        log_content = log_file.read_text()
        first_log = json.loads(log_content.strip().split('\n')[0])
        
        # Verify JSON structure
        assert isinstance(first_log, dict)
        assert "message" in first_log
        assert "level" in first_log
        assert "time" in first_log