"""Unit tests for the pipeline process module."""
import pytest
import os
import json
from pathlib import Path
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

from telexp.pipelines.process import TelegramExportProcessor
from telexp.schema import RawMessage, Window, WindowMetadata
from telexp.config import Config


@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = Path(os.path.dirname(__file__)) / '..' / '..' / 'conf.yaml'
    if config_path.exists():
        return Config.from_yaml(str(config_path))
        
    # If no config exists, create a minimal test config
    minimal_config = {
        "logging": {
            "level": "INFO",
            "distributed": False,
            "log_file": None,
            "enqueue": False
        },
        "text_vectorizer": {
            "embedding_service": "test_service",
            "model": "test-model",
            "timeout": 30,
            "retries": 3,
            "input_field": "concatenated_text",
            "output_field": "vector",
            "output_type": "vector",
            "vector_size": 768
        },
        "text_vector_storage": {
            "url": "http://localhost:6333",
            "collection": "test_collection",
            "vector_size": 768,
            "batch_size": 10
        },
        "window_by_size": {
            "size": 10,
            "overlap": 5
        }
    }
    return Config.model_validate(minimal_config)


def test_create_message_window():
    """Test the _create_message_window method properly sets concatenated_text."""
    # Create a processor instance without config (we don't need services for this test)
    processor = TelegramExportProcessor()
    
    # Create a test message with text content
    message = RawMessage(
        id="1",
        type="message",
        date="2024-01-01T12:00:00",
        date_unixtime=1704110400,
        conversation_id="test-convo",
        conversation_type="private",
        from_id="user123",
        from_name="Test User",
        text="This is a test message with content"
    )
    
    # Create a window from the message
    window = processor._create_message_window(message)
    
    # Verify the window has the concatenated_text set
    assert window.concatenated_text is not None
    assert window.concatenated_text == "This is a test message with content"
    assert window.has_text_content() is True


def test_create_message_window_with_formatted_text():
    """Test the _create_message_window method with formatted text entities."""
    processor = TelegramExportProcessor()
    
    # Create a test message with formatted text (like the test data)
    message = RawMessage(
        id="1",
        type="message",
        date="2024-01-01T12:00:00",
        date_unixtime=1704110400,
        conversation_id="test-convo",
        conversation_type="private",
        from_id="user123",
        from_name="Test User",
        text=[
            {"type": "bold", "text": "Hello"},
            {"type": "plain", "text": " everyone! This is a "},
            {"type": "italic", "text": "formatted"},
            {"type": "plain", "text": " message."}
        ]
    )
    
    # Create a window from the message
    window = processor._create_message_window(message)
    
    # Verify the window has the concatenated_text set correctly
    assert window.concatenated_text is not None
    assert window.concatenated_text == "Hello  everyone! This is a  formatted  message."
    assert window.has_text_content() is True


def test_create_message_window_with_empty_text():
    """Test the _create_message_window method with empty text."""
    processor = TelegramExportProcessor()
    
    # Create a test message with empty text
    message = RawMessage(
        id="1",
        type="message",
        date="2024-01-01T12:00:00",
        date_unixtime=1704110400,
        conversation_id="test-convo",
        conversation_type="private",
        from_id="user123",
        from_name="Test User",
        text=""
    )
    
    # Create a window from the message
    window = processor._create_message_window(message)
    
    # Verify the window has the concatenated_text set to None
    assert window.concatenated_text is None
    assert window.has_text_content() is False


def test_create_window_with_text_content():
    """Test the _create_window method properly concatenates text from multiple messages."""
    processor = TelegramExportProcessor()
    
    # Create multiple message windows with text content
    window_id1 = uuid.uuid4()
    window_id2 = uuid.uuid4()
    
    metadata1 = WindowMetadata(
        id=window_id1,
        conversation_id="test-convo",
        conversation_type="group",
        date_range=["2024-01-01T12:00:00", "2024-01-01T12:00:00"],
        unixtime_range=[1704110400, 1704110400],
        from_ids={"user1": 1},
        event_ids=["1"]
    )
    
    metadata2 = WindowMetadata(
        id=window_id2,
        conversation_id="test-convo",
        conversation_type="group",
        date_range=["2024-01-01T12:01:00", "2024-01-01T12:01:00"],
        unixtime_range=[1704110460, 1704110460],
        from_ids={"user2": 1},
        event_ids=["2"]
    )
    
    window1 = Window(
        id=window_id1,
        conversation_id="test-convo",
        conversation_type="group",
        concatenated_text="First message",
        window_metadata=metadata1
    )
    
    window2 = Window(
        id=window_id2,
        conversation_id="test-convo",
        conversation_type="group",
        concatenated_text="Second message",
        window_metadata=metadata2
    )
    
    # Create a combined window
    combined_window = processor._create_window("test-convo", [window1, window2])
    
    # Verify the window has the concatenated_text set correctly
    assert combined_window.concatenated_text is not None
    assert "First message" in combined_window.concatenated_text
    assert "Second message" in combined_window.concatenated_text
    assert combined_window.has_text_content() is True
    
    # Verify metadata properties
    assert combined_window.window_metadata is not None
    assert combined_window.window_metadata.event_ids == ["1", "2"]
    assert combined_window.window_metadata.from_ids == {"user1": 1, "user2": 1}
    assert combined_window.window_metadata.date_range[0] == "2024-01-01T12:00:00"
    assert combined_window.window_metadata.date_range[1] == "2024-01-01T12:01:00"
    assert combined_window.window_metadata.unixtime_range[0] == 1704110400
    assert combined_window.window_metadata.unixtime_range[1] == 1704110460


def test_create_window_with_empty_content():
    """Test the _create_window method properly handles windows with no text content."""
    processor = TelegramExportProcessor()
    
    # Create message windows with no text content
    window_id1 = uuid.uuid4()
    
    metadata1 = WindowMetadata(
        id=window_id1,
        conversation_id="test-convo",
        conversation_type="group",
        date_range=["2024-01-01T12:00:00", "2024-01-01T12:00:00"],
        unixtime_range=[1704110400, 1704110400],
        from_ids={"user1": 1},
        event_ids=["1"]
    )
    
    window1 = Window(
        id=window_id1,
        conversation_id="test-convo",
        conversation_type="group",
        concatenated_text="",  # Empty string (not None)
        window_metadata=metadata1
    )
    
    # Create a window from empty content
    combined_window = processor._create_window("test-convo", [window1])
    
    # Verify the window metadata carries over
    assert combined_window.window_metadata is not None
    assert combined_window.window_metadata.event_ids == ["1"]
    assert combined_window.has_text_content() is False


def test_process_message_with_photo():
    """Test message processing with a photo."""
    processor = TelegramExportProcessor()
    
    # Create a test message with a photo
    message = RawMessage(
        id="1",
        type="message",
        date="2024-01-01T12:00:00",
        date_unixtime=1704110400,
        conversation_id="test-convo",
        conversation_type="private",
        from_id="user123",
        from_name="Test User",
        text="Image caption",
        photo={"file": "/path/to/image.jpg"},
        source={
            "file_path": "/path/to/test.jsonl",
            "file_name": "test.jsonl", 
            "file_extension": ".jsonl",
            "file_mimetype": "application/json",
            "file_offset": 0,
            "file_line_number": 1
        }
    )
    
    # Process the message
    window = processor._create_message_window(message)
    
    # Verify text is preserved
    assert window.concatenated_text is not None
    assert "Image caption" in window.concatenated_text
    assert window.has_text_content() is True
    
    # Photo reference might be preserved depending on implementation
    # Focus on metadata instead
    assert window.window_metadata is not None
    assert window.window_metadata.event_ids[0] == "1"


def test_process_service_message():
    """Test processing of service messages."""
    processor = TelegramExportProcessor()
    
    # Create a service message (chat event like someone joining)
    message = RawMessage(
        id="1",
        type="service",
        date="2024-01-01T12:00:00",
        date_unixtime=1704110400,
        conversation_id="test-convo",
        conversation_type="group",
        from_id="user123",
        from_name="Test User",
        action="user_joined",
        actor="Test User",
        text=None,
        source={
            "file_path": "/path/to/test.jsonl",
            "file_name": "test.jsonl", 
            "file_extension": ".jsonl",
            "file_mimetype": "application/json",
            "file_offset": 0,
            "file_line_number": 1
        }
    )
    
    # Create a window from the service message
    window = processor._create_message_window(message)
    
    # Verify the service message is properly captured in metadata
    assert window.window_metadata is not None
    assert len(window.window_metadata.event_ids) == 1
    assert window.window_metadata.event_ids[0] == "1"
    
    # Service message specifics may depend on implementation
    # Just check that we get something back with the right ID and conversation
    assert window.conversation_id == "test-convo"
    assert window.conversation_type == "group"
    
    
def test_processor_initialization(test_config):
    """Test the processor initialization process."""
    # Create a processor with a specific config
    processor = TelegramExportProcessor(test_config)
    
    # Verify config is loaded
    assert processor.config is not None
    
    # Verify service managers are initialized
    assert processor.llm_manager is not None
    assert processor.embedding_manager is not None
    
    # Verify logger is set up
    assert processor.logger is not None
    
    # Verify stored IDs list is initialized
    assert processor.stored_ids == []


def test_load_messages_from_jsonl(tmp_path):
    """Test loading messages from JSONL files."""
    processor = TelegramExportProcessor()
    
    # Create a test directory structure
    conv_type_dir = tmp_path / "private_chat"
    conv_type_dir.mkdir()
    conv_id_dir = conv_type_dir / "12345678"
    conv_id_dir.mkdir()
    
    # Create a test messages.jsonl file
    messages_file = conv_id_dir / "messages.jsonl"
    test_messages = [
        {
            "id": "1",
            "type": "message",
            "date": "2024-01-01T12:00:00",
            "date_unixtime": "1704110400",
            "from": "User One",
            "text": "Hello world"
        },
        {
            "id": "2",
            "type": "message",
            "date": "2024-01-01T12:05:00",
            "date_unixtime": "1704110700",
            "from": "User Two",
            "text": "Hi there",
            "photo": "attachments/photos/photo_1.jpg"
        }
    ]
    
    with open(messages_file, 'w') as f:
        for msg in test_messages:
            f.write(json.dumps(msg) + '\n')
    
    # Test loading messages
    loaded_messages = processor.load_messages_from_jsonl(tmp_path)
    
    # Verify messages were loaded
    assert len(loaded_messages) == 2
    
    # Check first message
    assert loaded_messages[0]["id"] == "1"
    assert loaded_messages[0]["from_name"] == "User One"
    assert loaded_messages[0]["conversation_id"] == "12345678"
    assert loaded_messages[0]["conversation_type"] == "private_chat"
    
    # Check second message with photo
    assert loaded_messages[1]["id"] == "2"
    assert loaded_messages[1]["from_name"] == "User Two"
    assert isinstance(loaded_messages[1]["photo"], dict)
    assert os.path.isabs(loaded_messages[1]["photo"]["file"])
    
    # Test with non-directory in input_dir
    non_dir_file = tmp_path / "not_a_dir.txt"
    non_dir_file.touch()
    
    # Should not raise an error, just skip non-directories
    loaded_messages = processor.load_messages_from_jsonl(tmp_path)
    assert len(loaded_messages) == 2
    

def test_process_message_chain():
    """Test processing a chain of related messages."""
    processor = TelegramExportProcessor()
    
    # Create a series of messages from the same conversation
    messages = []
    for i in range(5):
        messages.append(RawMessage(
            id=str(i+1),
            type="message",
            date=f"2024-01-01T12:{i:02d}:00",
            date_unixtime=1704110400 + i*60,  # 1 minute increments
            conversation_id="test-chain",
            conversation_type="group",
            from_id=f"user{i%2+1}",
            from_name=f"User {i%2+1}",
            text=f"Message {i+1} in chain",
            source={
                "file_path": "/path/to/test.jsonl",
                "file_name": "test.jsonl", 
                "file_extension": ".jsonl",
                "file_mimetype": "application/json",
                "file_offset": i*100,
                "file_line_number": i+1
            }
        ))
    
    # Create individual windows
    windows = [processor._create_message_window(msg) for msg in messages]
    
    # Create a combined window from all messages
    combined = processor._create_window("test-chain", windows)
    
    # Verify combined window has all messages
    assert combined.window_metadata is not None
    assert len(combined.window_metadata.event_ids) == 5
    assert all(str(i+1) in combined.window_metadata.event_ids for i in range(5))
    
    # Verify combined text contains content from all messages
    assert combined.concatenated_text is not None
    for i in range(5):
        assert f"Message {i+1}" in combined.concatenated_text
    
    # Verify from_ids tracks both users
    assert len(combined.window_metadata.from_ids) == 2
    assert "user1" in combined.window_metadata.from_ids
    assert "user2" in combined.window_metadata.from_ids