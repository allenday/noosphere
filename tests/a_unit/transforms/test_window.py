import pytest
from unittest.mock import Mock, patch
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
import yaml
import uuid
from datetime import datetime, timedelta
from apache_beam.transforms.window import GlobalWindow
from apache_beam.utils.windowed_value import WindowedValue

from noosphere.telegram.batch.transforms.window import WindowBySession, WindowBySize
from noosphere.telegram.batch.config import WindowBySessionConfig, WindowBySizeConfig
from noosphere.telegram.batch.schema import Window, RawMessage

@pytest.fixture
def test_session_config():
    """Test configuration for session-based windowing."""
    return WindowBySessionConfig(
        timeout_seconds=3600,  # 1 hour
        min_size=1,
        max_size=100
    )

@pytest.fixture
def test_size_config():
    """Test configuration for size-based windowing."""
    return WindowBySizeConfig(
        size=5,
        overlap=2
    )

@pytest.fixture
def test_messages():
    """Create test messages with timestamps 1 minute apart."""
    base_time = datetime(2024, 1, 1, 12, 0)
    messages = []
    
    for i in range(10):
        msg_time = base_time + timedelta(minutes=i)
        messages.append({
            'id': i + 1,
            'type': 'text',
            'conversation_id': 'test-conv-1',
            'conversation_type': 'test',
            'date': msg_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'date_unixtime': int(msg_time.timestamp()),
            'from_id': f'user{i % 2 + 1}',
            'text': f'Message {i + 1}',
            'source': {
                'file_path': '/path/to/data/messages.jsonl',
                'file_name': 'messages.jsonl',
                'file_extension': '.jsonl',
                'file_mimetype': 'application/jsonl',
                'file_offset': i * 100,
                'file_line_number': i + 1
            }
        })
    return messages

def test_session_window_timeout(test_session_config, test_messages):
    """Test session windows are created when timeout is reached."""
    # Modify timestamps to create session breaks
    test_messages[5]['date_unixtime'] += test_session_config.timeout_seconds + 1
    
    with TestPipeline() as p:
        output = (
            p
            | beam.Create(test_messages)
            | beam.ParDo(WindowBySession(test_session_config))
        )
        
        def check_output(windows):
            # Convert WindowedValue to Window model
            windows = [Window.model_validate(w) for w in windows]
            assert len(windows) == 2  # Should create 2 windows due to timeout
            
            # First window should have 5 messages
            assert len(windows[0].window_metadata.event_ids) == 5
            # Second window should have 5 messages
            assert len(windows[1].window_metadata.event_ids) == 5
            
            # Verify time ranges
            assert windows[0].window_metadata.unixtime_range[1] < windows[1].window_metadata.unixtime_range[0]
            return True
            
        assert_that(output, check_output)

def test_session_window_max_size(test_session_config, test_messages):
    """Test session windows are created when max size is reached."""
    # Set max size smaller than total messages
    test_session_config.max_size = 3
    
    with TestPipeline() as p:
        output = (
            p
            | beam.Create(test_messages)
            | beam.ParDo(WindowBySession(test_session_config))
        )
        
        def check_output(windows):
            # Convert WindowedValue to Window model
            windows = [Window.model_validate(w) for w in windows]
            assert len(windows) >= 3  # Should create at least 3 windows
            
            # Each window should have max 3 messages
            for window in windows:
                assert len(window.window_metadata.event_ids) <= test_session_config.max_size
            return True
            
        assert_that(output, check_output)

def test_size_window_basic(test_size_config, test_messages):
    """Test basic sliding window functionality."""
    with TestPipeline() as p:
        output = (
            p
            | beam.Create(test_messages)
            | beam.ParDo(WindowBySize(test_size_config))
        )
        
        def check_output(windows):
            # Convert WindowedValue to Window model
            windows = [Window.model_validate(w) for w in windows]
            # Calculate expected number of windows
            # For 10 messages, window size 5, overlap 2:
            # Window 1: 0-4
            # Window 2: 3-7
            # Window 3: 6-10
            expected_windows = ((len(test_messages) - test_size_config.overlap) // 
                              (test_size_config.size - test_size_config.overlap))
            assert len(windows) == expected_windows
            
            # Each window should have exactly size messages
            for window in windows:
                assert len(window.window_metadata.event_ids) == test_size_config.size
                
            # Check overlap
            for i in range(len(windows) - 1):
                current_ids = set(windows[i].window_metadata.event_ids)
                next_ids = set(windows[i + 1].window_metadata.event_ids)
                overlap_count = len(current_ids & next_ids)
                assert overlap_count == test_size_config.overlap
                
            return True
            
        assert_that(output, check_output)

def test_size_window_small_input(test_size_config):
    """Test handling of input smaller than window size."""
    small_messages = [
        {
            'id': i + 1,
            'type': 'text',
            'conversation_id': 'test-conv-1',
            'date': '2024-01-01T12:00:00',
            'date_unixtime': 1704110400,
            'from_id': f'user1',
            'text': f'Message {i + 1}',
            'source': {
                'file_path': '/path/to/data/messages.jsonl',
                'file_name': 'messages.jsonl',
                'file_extension': '.jsonl',
                'file_mimetype': 'application/jsonl',
                'file_offset': i * 100,
                'file_line_number': i + 1
            }
        }
        for i in range(3)  # Only 3 messages
    ]
    
    with TestPipeline() as p:
        output = (
            p
            | beam.Create(small_messages)
            | beam.ParDo(WindowBySize(test_size_config))
        )
        
        def check_output(windows):
            windows = [w.value for w in windows]  # Extract value from WindowedValue
            assert len(windows) == 0  # Should not create any windows
            return True
            
        assert_that(output, check_output)

def test_window_metadata_validation(test_session_config, test_messages):
    """Test window metadata is correctly created and validated."""
    transform = WindowBySession(test_session_config)
    
    # Process messages
    windows = list(transform.process(test_messages[0]))
    assert len(windows) == 0  # First message shouldn't create a window yet
    
    # Process enough messages to create a window
    for msg in test_messages[1:test_session_config.min_size]:
        windows.extend(transform.process(msg))
    
    # Get windows from finish_bundle
    windows.extend(transform.finish_bundle())
    
def test_window_by_session_edge_cases(test_session_config):
    """Test edge cases for session-based windowing."""
    transform = WindowBySession(test_session_config)
    
    # Test with None message
    empty_result = list(transform.process(None))
    assert len(empty_result) == 0
    
    # Test with invalid message (missing date)
    invalid_msg = {}
    invalid_result = list(transform.process(invalid_msg))
    assert len(invalid_result) == 0
    
    # Test with message having all required fields but invalid types
    invalid_msg2 = {
        'id': 'not-an-int',
        'type': 123,  # Should be string
        'date': 12345,  # Should be string
        'date_unixtime': 'not-an-int',
        'conversation_id': None,
        'from_id': None,
        'text': None
    }
    invalid_result2 = list(transform.process(invalid_msg2))
    assert len(invalid_result2) == 0
    
    # Test finish_bundle when no messages have been processed
    empty_finish = list(transform.finish_bundle())
    assert len(empty_finish) == 0
    
def test_window_by_size_edge_cases(test_size_config):
    """Test edge cases for size-based windowing."""
    transform = WindowBySize(test_size_config)
    
    # Test with None message
    none_result = list(transform.process(None))
    assert len(none_result) == 0
    
    # Test with invalid message (missing required fields)
    invalid_msg = {}
    invalid_result = list(transform.process(invalid_msg))
    assert len(invalid_result) == 0
    
    # Test with too few messages (less than window size)
    small_batch = {
        'id': 1,
        'type': 'text',
        'date': '2024-01-01T12:00:00',
        'date_unixtime': 1704110400,
        'conversation_id': 'test-conv',
        'conversation_type': 'personal_chat',
        'from_id': 'user1',
        'text': 'Message 1'
    }
    small_result = list(transform.process(small_batch))
    assert len(small_result) == 0


def test_window_by_size_finish_bundle(test_size_config, test_messages):
    """Test finish_bundle behavior for size-based windowing."""
    transform = WindowBySize(test_size_config)
    
    # Process each message individually to accumulate in buffer
    # (processing all at once may create windows immediately)
    for i, msg in enumerate(test_messages[:test_size_config.size + 1]):
        # Create a copy to avoid modifying the original
        msg_copy = dict(msg)
        # Make sure each message is for the same conversation but has a unique ID
        msg_copy['id'] = i + 1
        msg_copy['conversation_id'] = 'test-finish-bundle'
        # Process the message
        list(transform.process(msg_copy))
    
    # The buffers should have the most recent message
    assert 'test-finish-bundle' in transform._buffers
    
    # Check that finish_bundle properly clears out buffers
    finish_results = list(transform.finish_bundle())
    
    # Verify buffers are cleared after finish_bundle
    assert len(transform._buffers) == 0


def test_window_to_langchain_converter():
    """Test the WindowToLangchainConverter transform."""
    from noosphere.telegram.batch.transforms.window import WindowToLangchainConverter
    from noosphere.telegram.batch.schema import WindowMetadata, Window
    
    # Create a test window with sample data
    window_id = uuid.uuid4()
    metadata = WindowMetadata(
        id=window_id,
        conversation_id="test-convo",
        conversation_type="group",
        date_range=["2024-01-01T12:00:00", "2024-01-01T12:00:00"],
        unixtime_range=[1704110400, 1704110400],
        from_ids={"user1": 1},
        from_names={"user1": "Test User"},
        event_ids=["1", "2", "3"]
    )
    
    window = Window(
        id=window_id,
        conversation_id="test-convo",
        conversation_type="group",
        summary_text="This is a summary of the conversation",
        vector=[0.1, 0.2, 0.3],  # Sample vector
        window_metadata=metadata
    )
    
    # Create and use the converter
    converter = WindowToLangchainConverter()
    converter.setup()
    
    # Process the window
    results = list(converter.process(window))
    
    # Verify conversion
    assert len(results) == 1
    lc_window = results[0]
    
    # Check basic properties
    assert lc_window.id == str(window_id)
    assert lc_window.content == "This is a summary of the conversation"
    assert lc_window.vector == [0.1, 0.2, 0.3]
    
    # Check metadata
    assert lc_window.metadata["conversation_id"] == "test-convo"
    assert lc_window.metadata["conversation_type"] == "group"
    assert lc_window.metadata["pubkey"] == "Test User"  # Should use the first from_name
    assert "file_id" in lc_window.metadata
    assert lc_window.metadata["file_id"] == "group/test-convo"  # Not enough messages to create a window

def test_window_source_tracking(test_session_config, test_messages):
    """Test that source information is preserved in windows."""
    transform = WindowBySession(test_session_config)
    
    # Process messages
    windows = []
    for msg in test_messages[:3]:  # Process first 3 messages
        windows.extend(transform.process(msg))
    windows.extend(transform.finish_bundle())
    
    assert len(windows) == 1
    window = Window.model_validate(windows[0].value)
    
    # Verify source information is preserved in metadata
    event_ids = window.window_metadata.event_ids
    assert len(event_ids) == 3
    
    # Check that original messages are included in the window
    for i, msg_id in enumerate(event_ids):
        original_msg = test_messages[i]
        assert str(original_msg['id']) == msg_id
        assert original_msg['source']['file_line_number'] == i + 1
        assert original_msg['source']['file_offset'] == i * 100

def test_window_conversation_grouping(test_session_config):
    """Test that windows are created per conversation."""
    # Create messages from different conversations
    messages = [
        {
            'id': i + 1,
            'type': 'text',
            'conversation_id': f'conv-{i % 2 + 1}',  # Two different conversations
            'date': '2024-01-01T12:00:00',
            'date_unixtime': 1704110400 + i,
            'from_id': 'user1',
            'text': f'Message {i + 1}',
            'source': {
                'file_path': '/path/to/data/messages.jsonl',
                'file_name': 'messages.jsonl',
                'file_extension': '.jsonl',
                'file_mimetype': 'application/jsonl',
                'file_offset': i * 100,
                'file_line_number': i + 1
            }
        }
        for i in range(6)
    ]
    
    with TestPipeline() as p:
        output = (
            p
            | beam.Create(messages)
            | beam.ParDo(WindowBySession(test_session_config))
        )
        
        def check_output(windows):
            # Convert WindowedValue to Window model
            windows = [Window.model_validate(w) for w in windows]
            # Group windows by conversation
            by_conv = {}
            for window in windows:
                conv_id = window.conversation_id
                by_conv[conv_id] = by_conv.get(conv_id, []) + [window]
            
            # Should have windows for both conversations
            assert len(by_conv) == 2
            assert all(len(w) > 0 for w in by_conv.values())
            
            # Each window should only contain messages from its conversation
            for conv_windows in by_conv.values():
                for window in conv_windows:
                    msg_ids = set(window.window_metadata.event_ids)
                    conv_msgs = [str(m['id']) for m in messages 
                               if m['conversation_id'] == window.conversation_id]
                    assert msg_ids.issubset(set(conv_msgs))
            
            return True
            
        assert_that(output, check_output) 