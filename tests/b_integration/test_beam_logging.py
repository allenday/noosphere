"""Tests for Apache Beam integration with logging system."""
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
import pytest
import pickle
import io
import time

from noosphere.telegram.batch.logging import LoggingMixin, log_performance, get_logger, setup_logging


class TestLoggingDoFn(beam.DoFn, LoggingMixin):
    """A simple DoFn that uses LoggingMixin for testing."""
    
    def __init__(self):
        super().__init__()
        self._setup_called = False  # Add _setup_called flag
        self._setup_logging()
        self.processed_count = 0
    
    def __getstate__(self):
        """Control which attributes are pickled."""
        state = super().__getstate__()
        # Reset setup flag for unpickling
        state['_setup_called'] = False
        return state
    
    def setup(self):
        """Setup for the DoFn - will be called by Beam."""
        # Skip if already set up
        if self._setup_called:
            return
            
        # Ensure logger is properly initialized after deserialization
        if not hasattr(self, 'logger'):
            self._setup_logging()
        self.logger.debug("DoFn setup called")
        self._setup_called = True
    
    def teardown(self):
        """Teardown for the DoFn - will be called by Beam."""
        self._setup_called = False
        self.logger.debug("DoFn teardown called")
    
    @log_performance(threshold_ms=1)
    def process(self, element):
        """Process method with performance tracking."""
        # Ensure setup is called
        if not self._setup_called:
            self.setup()
            
        self.logger.info(f"Processing element: {element}")
        self.processed_count += 1
        
        # Add a small delay to trigger performance logging
        time.sleep(0.01)
        
        # Output transformed element
        yield element * 2


def test_logging_dofn_serialization():
    """Test that LoggingMixin works correctly with beam DoFn serialization."""
    # Initialize logging for test
    setup_logging(level="DEBUG")
    
    # Create DoFn instance
    dofn = TestLoggingDoFn()
    assert hasattr(dofn, 'logger')  # Logger is created in init
    assert dofn._setup_called == False  # Should be False initially
    
    # Call setup and check state
    dofn.setup()
    assert dofn._setup_called == True
    
    # Serialize and deserialize
    serialized = pickle.dumps(dofn)
    deserialized = pickle.loads(serialized)
    
    # Verify state reset and logger recreation behavior
    assert hasattr(deserialized, 'logger')  # Logger should be preserved 
    assert deserialized._setup_called == False  # Setup state should be reset
    
    # Setup should mark as set up
    deserialized.setup()
    assert deserialized._setup_called == True
    assert deserialized.logger_name.startswith('TestLoggingDoFn_')


def test_logging_dofn_in_pipeline():
    """Test LoggingMixin DoFn in an actual beam pipeline."""
    # Configure logging for test
    setup_logging(level="DEBUG", enqueue=False)
    
    # Capture logs
    log_capture = io.StringIO()
    from loguru import logger
    handler_id = logger.add(log_capture, level="DEBUG")
    
    try:
        # Create test pipeline
        with TestPipeline() as pipeline:
            input_data = [1, 2, 3, 4, 5]
            expected_output = [2, 4, 6, 8, 10]
            
            # Process data
            output = (
                pipeline
                | "Create" >> beam.Create(input_data)
                | "Process" >> beam.ParDo(TestLoggingDoFn())
            )
            
            # Assert output
            assert_that(output, equal_to(expected_output))
        
        # Check logs for expected output
        logs = log_capture.getvalue()
        
        # Should have logs from setup and process
        assert "DoFn setup called" in logs
        assert "Processing element:" in logs
        assert "Slow operation detected" in logs  # From performance logging
        
    finally:
        # Clean up logger
        logger.remove(handler_id)


def test_beam_context_preservation():
    """Test that beam pipeline context is preserved in logs."""
    # Configure logging for distributed mode
    setup_logging(level="DEBUG", distributed=True, enqueue=False)
    
    # Capture logs
    log_capture = io.StringIO()
    from loguru import logger
    handler_id = logger.add(log_capture, level="DEBUG")
    
    try:
        # Create test pipeline with special options for context tracking
        options = beam.options.pipeline_options.PipelineOptions([
            '--direct_num_workers=2'  # Use 2 workers in direct runner
        ])
        
        with TestPipeline(options=options) as pipeline:
            input_data = list(range(10))  # More data to ensure work distribution
            
            # Process data
            _ = (
                pipeline
                | "Create" >> beam.Create(input_data)
                | "Process" >> beam.ParDo(TestLoggingDoFn())
                | "LogResults" >> beam.Map(lambda x: logger.info(f"Result: {x}"))
            )
        
        # Check logs
        logs = log_capture.getvalue()
        
        # Verify process info appears in logs (P:<process>)
        assert "Processing element:" in logs
        
    finally:
        # Clean up logger
        logger.remove(handler_id)