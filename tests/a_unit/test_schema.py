import json
import pytest
from pathlib import Path
from noosphere.telegram.batch.schema import (
    MessageType,
    PhotoSize,
    Photo,
    Video,
    TextEntity,
    RawMessage,
    Status,
    ComponentStatus,
    WindowMetadata,
    Window,
    Source
)
import uuid
from pydantic import ValidationError

# Load test data from JSONL files
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
MESSAGES_FILE = next(TEST_DATA_DIR.glob("*/*/messages.jsonl"))
WINDOW_FILE = TEST_DATA_DIR / "window" / "window.jsonl"

def load_test_messages():
    """Load test messages from JSONL file."""
    messages = []
    with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
        # Extract conversation ID from the path (directory name)
        conversation_id = MESSAGES_FILE.parent.name
        for line in f:
            msg = json.loads(line)
            # Add required conversation_id field
            msg['conversation_id'] = conversation_id
            messages.append(msg)
    return messages

def load_test_windows():
    """Load test windows from JSONL file."""
    windows = []
    with open(WINDOW_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            windows.append(json.loads(line))
    return windows

TEST_MESSAGES = load_test_messages()
TEST_WINDOWS = load_test_windows()

def test_window_metadata_basic():
    """Test WindowMetadata with more detailed tests."""
    # Create a simple WindowMetadata instance
    window_id = uuid.uuid4()
    metadata = WindowMetadata(
        id=window_id,
        conversation_id="test-conversation",
        conversation_type="group",
        date_range=["2024-01-01", "2024-01-02"],
        unixtime_range=[1704067200, 1704153600],
        from_ids={"user1": 3, "user2": 2, "user3": 1},
        event_ids=["1", "2", "3", "4", "5", "6"]
    )
    
    # Test the properties
    assert metadata.id == window_id
    assert metadata.conversation_id == "test-conversation"
    assert len(metadata.from_ids) == 3  # Three unique users
    assert len(metadata.event_ids) == 6  # Six events
    assert metadata.date_range[0] == "2024-01-01"
    assert metadata.date_range[1] == "2024-01-02"
    
    # Test sorting of from_ids to find primary
    sorted_ids = sorted(metadata.from_ids.items(), key=lambda x: x[1], reverse=True)
    assert sorted_ids[0][0] == "user1"  # user1 has highest count
    assert sorted_ids[0][1] == 3
    
    # Test to_dict
    data_dict = metadata.model_dump()
    assert data_dict["id"] == window_id
    assert len(data_dict["event_ids"]) == 6
    assert data_dict["from_ids"]["user1"] == 3
    
    # Test JSON serialization
    json_str = metadata.model_dump_json()
    assert "test-conversation" in json_str
    assert "group" in json_str
    
    # Test with from_names
    metadata_with_names = WindowMetadata(
        id=window_id,
        conversation_id="test-conversation",
        conversation_type="group",
        date_range=["2024-01-01", "2024-01-02"],
        unixtime_range=[1704067200, 1704153600],
        from_ids={"user1": 3, "user2": 2},
        from_names={"user1": "Alice", "user2": "Bob"},
        event_ids=["1", "2", "3", "4", "5"]
    )
    
    assert metadata_with_names.from_names is not None
    assert metadata_with_names.from_names["user1"] == "Alice"


def test_raw_message_real_examples():
    """Test RawMessage with real examples from data."""
    for msg_data in TEST_MESSAGES:
        message = RawMessage(**msg_data)
        
        # Basic validation
        assert isinstance(message.id, str)
        assert message.type in ['message', 'service']
        assert isinstance(message.date_unixtime, int)
        
        # Test message type detection
        msg_type = message.get_message_type()
        assert isinstance(msg_type, MessageType)
        
        if message.photo:
            assert isinstance(message.photo, Photo)
            if message.photo.thumbnail:
                assert isinstance(message.photo.thumbnail, PhotoSize)
        
        if message.text:
            text_content = message.get_text_content()
            assert isinstance(text_content, str)
            assert len(text_content) > 0

def test_specific_message_types():
    """Test specific message types from real data."""
    # Group messages by type for specific testing
    text_msgs = [m for m in TEST_MESSAGES if 'text' in m and isinstance(m['text'], str)]
    photo_msgs = [m for m in TEST_MESSAGES if 'photo' in m]
    service_msgs = [m for m in TEST_MESSAGES if m['type'] == 'service']
    
    # Test text message
    if text_msgs:
        msg = RawMessage(**text_msgs[0])
        assert msg.get_message_type() == MessageType.TEXT
        assert isinstance(msg.get_text_content(), str)
        assert len(msg.get_text_content()) > 0
    
    # Test photo message
    if photo_msgs:
        msg = RawMessage(**photo_msgs[0])
        assert msg.get_message_type() == MessageType.PHOTO
    
    # Test service message
    if service_msgs:
        msg = RawMessage(**service_msgs[0])
        assert msg.get_message_type() == MessageType.SERVICE
        assert msg.action is not None

def test_text_entities():
    """Test messages with text entities from real data."""
    entity_msgs = [m for m in TEST_MESSAGES 
                  if 'text' in m and isinstance(m['text'], list)]
    
    if entity_msgs:
        msg = RawMessage(**entity_msgs[0])
        assert isinstance(msg.text, list)
        
        # Test text content concatenation
        text_content = msg.get_text_content()
        assert isinstance(text_content, str)
        assert len(text_content) > 0
        
        # Verify entities
        for item in msg.text:
            if isinstance(item, TextEntity):
                assert item.type
                assert item.text

def test_date_formats():
    """Test date formats from real data."""
    for msg_data in TEST_MESSAGES:
        msg = RawMessage(**msg_data)
        
        # Verify date format
        assert 'T' in msg.date  # Should be ISO format
        
        # Verify unixtime is reasonable
        assert 946684800 <= msg.date_unixtime <= 4102444800  # Between 2000 and 2100

def test_photo_attributes():
    """Test photo attributes from real data."""
    photo_msgs = [m for m in TEST_MESSAGES if 'photo' in m]
    
    if photo_msgs:
        for msg_data in photo_msgs:
            msg = RawMessage(**msg_data)
            assert isinstance(msg.photo, Photo)
            assert msg.photo.file
            
            if msg.photo.thumbnail:
                thumb = msg.photo.thumbnail
                assert isinstance(thumb, PhotoSize)
                assert thumb.width > 0
                assert thumb.height > 0
                assert '.' in thumb.file

def test_service_messages():
    """Test service message attributes from real data."""
    service_msgs = [m for m in TEST_MESSAGES if m['type'] == 'service']
    
    if service_msgs:
        for msg_data in service_msgs:
            msg = RawMessage(**msg_data)
            assert msg.type == 'service'
            assert msg.action is not None
            if msg.actor:
                assert isinstance(msg.actor, str)

def test_message_type_enum():
    assert MessageType.TEXT == "text"
    assert MessageType.SERVICE == "service"
    assert MessageType.PHOTO == "photo"

def test_raw_message_basic_text():
    """Test basic text message from real data."""
    text_msgs = [m for m in TEST_MESSAGES if 'text' in m and isinstance(m['text'], str)]
    if not text_msgs:
        pytest.skip("No text messages found in test data")
    
    message = RawMessage(**text_msgs[0])
    assert message.type == "message"
    assert isinstance(message.text, str)
    assert message.get_message_type() == MessageType.TEXT
    assert message.get_text_content() == message.text

def test_raw_message_text_entities():
    """Test text message with entities from real data."""
    entity_msgs = [m for m in TEST_MESSAGES if 'text' in m and isinstance(m['text'], list)]
    if not entity_msgs:
        pytest.skip("No text messages with entities found in test data")
    
    message = RawMessage(**entity_msgs[0])
    assert isinstance(message.text, list)
    text_content = message.get_text_content()
    assert isinstance(text_content, str)
    assert len(text_content) > 0

def test_raw_message_photo():
    """Test photo message from real data."""
    photo_msgs = [m for m in TEST_MESSAGES if 'photo' in m]
    if not photo_msgs:
        pytest.skip("No photo messages found in test data")
    
    message = RawMessage(**photo_msgs[0])
    assert message.type == "message"
    assert isinstance(message.photo, Photo)
    assert message.get_message_type() == MessageType.PHOTO

def test_raw_message_date_validation():
    """Test date validation in messages."""
    text_msgs = [m for m in TEST_MESSAGES if 'text' in m and isinstance(m['text'], str)]
    if not text_msgs:
        pytest.skip("No text messages found in test data")
    
    # Valid case
    message = RawMessage(**text_msgs[0])
    assert 'T' in message.date  # Should be ISO format
    assert 946684800 <= message.date_unixtime <= 4102444800  # Between 2000 and 2100

    # Invalid date format
    invalid_message = text_msgs[0].copy()
    invalid_message["date"] = "2024-01-01"  # Missing time component
    with pytest.raises(ValueError, match="Date must be in format"):
        RawMessage(**invalid_message)

    # Invalid unixtime
    invalid_message = text_msgs[0].copy()
    invalid_message["date_unixtime"] = 253402300800  # Year 9999
    with pytest.raises(ValueError, match="Unix timestamp must be between"):
        RawMessage(**invalid_message)

def test_window_metadata_validation():
    """Test window metadata validation."""
    window_id = uuid.uuid4()
    # Valid case
    metadata = WindowMetadata(
        id=window_id,
        conversation_id="conv1",
        date_range=["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        unixtime_range=[1704067200, 1704153600],
        from_ids={"user1": 1},
        event_ids=["msg1"]
    )
    assert metadata.id == window_id

    # Invalid UUID
    with pytest.raises(ValueError):
        WindowMetadata(
            id="not-a-uuid",
            conversation_id="conv1",
            date_range=["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
            unixtime_range=[1704067200, 1704153600],
            from_ids={"user1": 1},
            event_ids=["msg1"]
        )

    # Invalid date range length
    with pytest.raises(ValueError, match="Date range must contain exactly two dates"):
        WindowMetadata(
            id=uuid.uuid4(),
            conversation_id="conv1",
            date_range=["2024-01-01T00:00:00"],
            unixtime_range=[1704067200, 1704153600],
            from_ids={"user1": 1},
            event_ids=["msg1"]
        )

    # Invalid date order
    with pytest.raises(ValueError, match="End date must be after start date"):
        WindowMetadata(
            id=uuid.uuid4(),
            conversation_id="conv1",
            date_range=["2024-01-02T00:00:00", "2024-01-01T00:00:00"],  # Reversed order
            unixtime_range=[1704067200, 1704153600],
            from_ids={"user1": 1},
            event_ids=["msg1"]
        )

def test_status():
    status = Status(
        code="success",
        message="Operation completed",
        details={"model": "gpt-3.5"}
    )
    assert status.code == "success"
    assert status.message == "Operation completed"
    assert status.details.model == "gpt-3.5"
    assert isinstance(status.timestamp, float)

def test_component_status():
    status = Status(code="success")
    component = ComponentStatus(latest=status)
    assert component.latest == status
    assert len(component.history) == 0

def test_window_metadata():
    window_id = "12345678-1234-1234-1234-123456789012"  # Valid UUID string
    metadata = WindowMetadata(
        id=window_id,
        conversation_id="conv1",
        conversation_type="group",
        date_range=["2024-01-01", "2024-01-02"],
        unixtime_range=[1704067200, 1704153600],
        from_ids={"user1": 5, "user2": 3},
        event_ids=["msg1", "msg2", "msg3"]
    )
    assert str(metadata.id) == window_id
    assert metadata.conversation_id == "conv1"
    assert len(metadata.date_range) == 2
    assert len(metadata.from_ids) == 2
    assert len(metadata.event_ids) == 3

def test_window():
    """Test window validation."""
    window_id = "12345678-1234-1234-1234-123456789012"  # Valid UUID string
    metadata = WindowMetadata(
        id=window_id,
        conversation_id="conv1",
        date_range=["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        unixtime_range=[1704067200, 1704153600],
        from_ids={"user1": 1},
        event_ids=["msg1"]
    )
    window = Window(
        id=window_id,
        conversation_id="conv1",
        vector=[0.1, 0.2, 0.3],
        window_metadata=metadata
    )
    assert str(window.id) == window_id
    assert len(window.vector) == 3
    assert window.window_metadata == metadata

    # Test mismatched IDs
    different_id = "12345678-1234-1234-1234-123456789013"  # Different UUID
    with pytest.raises(ValueError, match="window_metadata.id must match window.id"):
        Window(
            id=different_id,
            conversation_id="conv1",
            vector=[0.1, 0.2, 0.3],
            window_metadata=metadata
        )

    # Test invalid UUID
    with pytest.raises(ValueError):
        Window(
            id="not-a-uuid",
            conversation_id="conv1",
            vector=[0.1, 0.2, 0.3],
            window_metadata=metadata
        )

def test_photo_size_validation():
    # Valid case
    photo_size = PhotoSize(type="thumbnail", width=100, height=100, file="photo.jpg")
    assert photo_size.width == 100

    # Invalid dimensions
    with pytest.raises(ValueError, match="Dimensions must be positive"):
        PhotoSize(type="thumbnail", width=-100, height=100, file="photo.jpg")
    
    with pytest.raises(ValueError, match="Dimensions must be positive"):
        PhotoSize(type="thumbnail", width=100, height=0, file="photo.jpg")

    # Invalid file path
    with pytest.raises(ValueError, match="File path cannot be empty"):
        PhotoSize(type="thumbnail", width=100, height=100, file="")
    
    with pytest.raises(ValueError, match="File path must have an extension"):
        PhotoSize(type="thumbnail", width=100, height=100, file="photo")

def test_window_utility_methods():
    """Test Window has_text_content and has_vector methods."""
    window_id = uuid.uuid4()
    metadata = WindowMetadata(
        id=window_id,
        conversation_id="test-conversation",
        conversation_type="group",
        date_range=["2024-01-01", "2024-01-02"],
        unixtime_range=[1704067200, 1704153600],
        from_ids={"user1": 3},
        event_ids=["1", "2", "3"]
    )
    
    # Test empty window
    empty_window = Window(
        id=window_id,
        conversation_id="test-conversation",
        window_metadata=metadata
    )
    assert empty_window.has_text_content() is False
    assert empty_window.has_valid_vector() is False
    
    # Test window with empty text
    empty_text_window = Window(
        id=window_id,
        conversation_id="test-conversation",
        window_metadata=metadata,
        concatenated_text=""
    )
    assert empty_text_window.has_text_content() is False
    
    # Test window with text
    text_window = Window(
        id=window_id,
        conversation_id="test-conversation",
        window_metadata=metadata,
        concatenated_text="This is text"
    )
    assert text_window.has_text_content() is True
    
    # Test window with vector
    vector_window = Window(
        id=window_id,
        conversation_id="test-conversation",
        window_metadata=metadata,
        vector=[0.1, 0.2, 0.3]
    )
    assert vector_window.has_valid_vector() is True
    
    # Test that even whitespace-only text returns True for has_text_content
    # since bool('   ') is True in Python
    whitespace_window = Window(
        id=window_id,
        conversation_id="test-conversation",
        window_metadata=metadata,
        concatenated_text="   \n  \t  "
    )
    assert whitespace_window.has_text_content() is True


def test_window_real_examples():
    """Test Window with real examples from data."""
    for window_data in TEST_WINDOWS:
        # Convert string IDs to UUIDs
        window_id = "12345678-1234-1234-1234-123456789012"  # Valid UUID string
        window_data['id'] = window_id
        window_data['window_metadata']['id'] = window_id
        
        window = Window(**window_data)
        
        # Basic validation
        assert isinstance(window.id, uuid.UUID)
        assert isinstance(window.conversation_id, str)
        assert isinstance(window.vector, list)
        assert all(isinstance(x, float) for x in window.vector)
        
        # Test metadata
        assert isinstance(window.window_metadata, WindowMetadata)
        assert window.window_metadata.id == window.id
        assert window.window_metadata.conversation_id == window.conversation_id
        
        # Optional fields
        if window.conversation_type:
            assert isinstance(window.conversation_type, str)
        if window.summary_text:
            assert isinstance(window.summary_text, str)

def test_window_metadata_real_examples():
    """Test WindowMetadata with real examples from data."""
    for window_data in TEST_WINDOWS:
        metadata = window_data.get('window_metadata')
        if not metadata:
            continue
            
        # Convert string ID to UUID
        metadata['id'] = "12345678-1234-1234-1234-123456789012"  # Valid UUID string
        window_metadata = WindowMetadata(**metadata)
        
        # Basic validation
        assert isinstance(window_metadata.id, uuid.UUID)
        assert isinstance(window_metadata.conversation_id, str)
        assert isinstance(window_metadata.date_range, list)
        assert len(window_metadata.date_range) == 2
        assert isinstance(window_metadata.unixtime_range, list)
        assert len(window_metadata.unixtime_range) == 2
        assert isinstance(window_metadata.from_ids, dict)
        assert isinstance(window_metadata.event_ids, list)
        
        # Date validation
        start_date, end_date = window_metadata.date_range
        assert start_date <= end_date  # Dates should be in order
        
        # Unixtime validation
        start_time, end_time = window_metadata.unixtime_range
        assert start_time <= end_time  # Times should be in order
        
        # Conversation type is optional
        if window_metadata.conversation_type:
            assert isinstance(window_metadata.conversation_type, str)

def test_window_text_vectorization():
    """Test window text vectorization methods."""
    for window_data in TEST_WINDOWS:
        window = Window(**window_data)
        
        # Test text content check
        has_text = window.has_text_content()
        text = window.get_text_for_vectorization()
        
        if text:
            assert isinstance(text, str)
            assert len(text) > 0
        
        # With updated logic, has_text_content now checks concatenated_text
        if window.concatenated_text:
            assert has_text
        else:
            assert not has_text

def test_window_vector_dimensions():
    """Test vector dimensions are consistent across windows."""
    if len(TEST_WINDOWS) < 2:
        pytest.skip("Need at least 2 windows to test vector dimensions")
    
    # Get vector dimension from first window
    first_window = Window(**TEST_WINDOWS[0])
    vector_dim = len(first_window.vector)
    
    # Check all other windows have same dimension
    for window_data in TEST_WINDOWS[1:]:
        window = Window(**window_data)
        assert len(window.vector) == vector_dim 

def test_has_valid_vector():
    """Test the Window.has_valid_vector method with various vector scenarios."""
    # Create a base window
    window_id = uuid.uuid4()
    metadata = WindowMetadata(
        id=window_id,
        conversation_id="test-conv",
        conversation_type="test",
        date_range=["2024-01-01T12:00:00", "2024-01-01T12:01:00"],
        unixtime_range=[1704110400, 1704110460],
        from_ids={"user1": 1},
        event_ids=["1"]
    )
    
    # Test case 1: Valid vector with normal values
    window = Window(
        id=window_id,
        conversation_id="test-conv",
        conversation_type="test",
        window_metadata=metadata,
        vector=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert window.has_valid_vector() is True
    
    # Test case 2: None vector
    window.vector = None
    assert window.has_valid_vector() is False
    
    # Test case 3: Empty vector list
    window.vector = []
    assert window.has_valid_vector() is False
    
    # Test case 4: Vector with all zeros
    window.vector = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert window.has_valid_vector() is False
    
    # Test case 5: Vector with wrong type
    window.vector = "not a vector"
    assert window.has_valid_vector() is False
    
    # Additional case: Vector with very small values (should be treated as zeros)
    window.vector = [1e-11, 1e-12, 1e-15]
    assert window.has_valid_vector() is False
    
    # Additional case: Vector with at least one non-zero value
    window.vector = [0.0, 0.0, 0.1, 0.0, 0.0]
    assert window.has_valid_vector() is True

def test_source_validation():
    """Test validation of Source model."""
    # Test valid source
    valid_source = Source(
        file_path="/path/to/data/messages.jsonl",
        file_name="messages.jsonl",
        file_extension=".jsonl",
        file_mimetype="application/jsonl",
        file_offset=100,
        file_line_number=1
    )
    assert valid_source.file_path == "/path/to/data/messages.jsonl"
    assert valid_source.file_line_number == 1
    
    # Test optional fields
    minimal_source = Source(
        file_path="/path/to/data/messages.jsonl",
        file_name="messages.jsonl",
        file_extension=".jsonl",
        file_mimetype="application/jsonl"
    )
    assert minimal_source.file_offset is None
    assert minimal_source.file_line_number is None
    
    # Test invalid file path
    with pytest.raises(ValidationError) as exc_info:
        Source(
            file_path="",  # Empty path
            file_name="messages.jsonl",
            file_extension=".jsonl",
            file_mimetype="application/jsonl"
        )
    assert "file_path cannot be empty" in str(exc_info.value)
    
    # Test invalid file extension
    with pytest.raises(ValidationError) as exc_info:
        Source(
            file_path="/path/to/data/messages.jsonl",
            file_name="messages.jsonl",
            file_extension="",  # Empty extension
            file_mimetype="application/jsonl"
        )
    assert "file_extension cannot be empty" in str(exc_info.value)
    
    # Test invalid file mimetype
    with pytest.raises(ValidationError) as exc_info:
        Source(
            file_path="/path/to/data/messages.jsonl",
            file_name="messages.jsonl",
            file_extension=".jsonl",
            file_mimetype=""  # Empty mimetype
        )
    assert "file_mimetype cannot be empty" in str(exc_info.value)
    
    # Test negative line number
    with pytest.raises(ValidationError) as exc_info:
        Source(
            file_path="/path/to/data/messages.jsonl",
            file_name="messages.jsonl",
            file_extension=".jsonl",
            file_mimetype="application/jsonl",
            file_line_number=-1
        )
    assert "file_line_number must be positive" in str(exc_info.value)
    
    # Test negative offset
    with pytest.raises(ValidationError) as exc_info:
        Source(
            file_path="/path/to/data/messages.jsonl",
            file_name="messages.jsonl",
            file_extension=".jsonl",
            file_mimetype="application/jsonl",
            file_offset=-1
        )
    assert "file_offset must be non-negative" in str(exc_info.value)

def test_raw_message_with_source():
    """Test RawMessage with Source field."""
    # Test valid message with source
    valid_message = RawMessage(
        id=1,
        type="text",
        date="2024-01-01T12:00:00",
        date_unixtime=1704110400,
        conversation_id="test-conv-1",
        text="Test message",
        source=Source(
            file_path="/path/to/data/messages.jsonl",
            file_name="messages.jsonl",
            file_extension=".jsonl",
            file_mimetype="application/jsonl",
            file_offset=100,
            file_line_number=1
        )
    )
    assert valid_message.source.file_path == "/path/to/data/messages.jsonl"
    assert valid_message.source.file_line_number == 1
    
    # Test message without source (should be None)
    minimal_message = RawMessage(
        id=1,
        type="text",
        date="2024-01-01T12:00:00",
        date_unixtime=1704110400,
        conversation_id="test-conv-1",
        text="Test message"
    )
    assert minimal_message.source is None
    
    # Test invalid source in message
    with pytest.raises(ValidationError) as exc_info:
        RawMessage(
            id=1,
            type="text",
            date="2024-01-01T12:00:00",
            date_unixtime=1704110400,
            conversation_id="test-conv-1",
            text="Test message",
            source={  # Missing required fields
                "file_path": "/path/to/data/messages.jsonl"
            }
        )
    assert "Field required" in str(exc_info.value) 