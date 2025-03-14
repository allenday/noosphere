"""Unit tests for message aggregation utilities."""
import pytest
import uuid
from unittest.mock import patch

from telexp.schema import Window, WindowMetadata
from telexp.transforms.message_aggregation import MessageAggregator


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    window_id1 = uuid.uuid4()
    window_id2 = uuid.uuid4()
    window_id3 = uuid.uuid4()
    
    # Create window metadata
    metadata1 = WindowMetadata(
        id=window_id1,
        conversation_id="test-convo",
        conversation_type="private",
        date_range=["2024-01-01T12:00:00", "2024-01-01T12:00:00"],
        unixtime_range=[1704110400, 1704110400],
        from_ids={"user1": 1},
        event_ids=["1"]
    )
    
    metadata2 = WindowMetadata(
        id=window_id2,
        conversation_id="test-convo",
        conversation_type="private",
        date_range=["2024-01-01T12:01:00", "2024-01-01T12:01:00"],
        unixtime_range=[1704110460, 1704110460],
        from_ids={"user2": 1},
        event_ids=["2"]
    )
    
    metadata3 = WindowMetadata(
        id=window_id3,
        conversation_id="test-convo",
        conversation_type="private",
        date_range=["2024-01-01T12:02:00", "2024-01-01T12:02:00"],
        unixtime_range=[1704110520, 1704110520],
        from_ids={"user1": 1},
        event_ids=["3"]
    )
    
    # Create windows
    window1 = Window(
        id=window_id1,
        conversation_id="test-convo",
        conversation_type="private",
        concatenated_text="Hello world",
        window_metadata=metadata1
    )
    
    window2 = Window(
        id=window_id2,
        conversation_id="test-convo",
        conversation_type="private",
        concatenated_text="This is a\nmulti-line message",
        window_metadata=metadata2
    )
    
    window3 = Window(
        id=window_id3,
        conversation_id="test-convo",
        conversation_type="private",
        image_path="/path/to/image.jpg",
        summary_text="An image of a cat",
        window_metadata=metadata3
    )
    
    return [window1, window2, window3]


def test_format_as_conversation(sample_messages):
    """Test formatting messages as a conversation."""
    with patch('logging.getLogger') as mock_logger:
        mock_logger.return_value.info = lambda x: None  # No-op logger
        
        # Format messages as conversation
        conversation = MessageAggregator.format_as_conversation(sample_messages)
        
        # Check for expected format patterns
        assert "[user1]: Hello world" in conversation
        assert "[user2]: This is a" in conversation
        assert "multi-line message" in conversation  # Without username prefix
        assert "[user1]: ![An image of a cat](image.jpg)" in conversation
        
        # Check order
        lines = conversation.split("\n")
        assert len(lines) == 4  # 4 lines including the multi-line message


def test_compute_message_stats(sample_messages):
    """Test computing message statistics."""
    stats = MessageAggregator.compute_message_stats(sample_messages)
    
    assert stats["message_count"] == 3
    assert stats["unique_users"] == 2
    assert set(stats["users"]) == {"user1", "user2"}
    assert stats["message_count_by_user"]["user1"] == 2
    assert stats["message_count_by_user"]["user2"] == 1
    assert stats["total_text_length"] > 0
    assert stats["avg_message_length"] > 0


def test_format_as_conversation_empty():
    """Test formatting an empty message list."""
    conversation = MessageAggregator.format_as_conversation([])
    assert conversation == ""


def test_compute_message_stats_empty():
    """Test computing stats for an empty message list."""
    stats = MessageAggregator.compute_message_stats([])
    assert stats["message_count"] == 0
    assert stats["unique_users"] == 0
    assert stats["users"] == []
    assert stats["message_count_by_user"] == {}
    assert stats["total_text_length"] == 0
    assert stats["avg_message_length"] == 0