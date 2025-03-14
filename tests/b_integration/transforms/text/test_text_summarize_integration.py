"""Integration tests for OllamaSummarize with live services."""
import pytest
import uuid
import yaml

from telexp.transforms.text.summarize import OllamaSummarize
from telexp.schema import Window, WindowMetadata
from telexp.config import Config
from telexp.services.llm import LLMServiceManager

# Mark all tests as integration tests
pytestmark = pytest.mark.integration

def load_config():
    """Load configuration from conf.yaml."""
    with open('conf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config)

@pytest.fixture
def test_config():
    """Load the actual config to use real services."""
    config = load_config().text_summarizer
    # The real config expects concatenated_messages_text but we're using concatenated_text
    # in our test fixture, so update the input_field
    config.input_field = "concatenated_text"
    # Model name should match the name field in the config (not the model field)
    config.model = "openorca"
    return config

@pytest.fixture
def test_service_manager():
    """Create real service manager with actual connections."""
    config = load_config()
    manager = LLMServiceManager(config.llm_services)
    manager.setup()
    return manager

@pytest.fixture
def test_conversation_window():
    """Create test window with conversation-formatted text."""
    window_id = uuid.UUID('12345678-1234-1234-1234-123456789013')
    
    # Format: [Speaker]: Message content
    # - No trailing newline after message content
    # - Single newline between messages
    # - For multi-line messages, speaker name appears only once
    formatted_text = (
        "[Alice]: Hello everyone,\nHow are you all doing today?\nHope this message finds you well!\n" +
        "[Bob]: I'm doing great, thanks for asking!\nI've been working on that project we discussed.\n" +
        "[Alice]: That's fantastic to hear!\nBy the way, here's that picture from yesterday's hike:\n" +
        "[Alice]: ![A beautiful mountain view with a lake in the foreground](/path/to/mountain_lake.jpg)\n" +
        "[Charlie]: Wow, that looks amazing!\nWhere was this taken?\n" +
        "[Alice]: It's from Mount Rainier National Park.\nThe weather was perfect.\n" +
        "[Bob]: We should plan a group hike there sometime."
    )
    
    return Window(
        id=window_id,
        conversation_id='test-conv-2',
        conversation_type='group',
        concatenated_text=formatted_text,
        summary_text=None,
        window_metadata=WindowMetadata(
            id=window_id,
            conversation_id='test-conv-2',
            conversation_type='group',
            date_range=['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
            unixtime_range=[1704067200, 1704070800],
            from_ids={'Alice': 3, 'Bob': 2, 'Charlie': 1},
            event_ids=['1', '2', '3', '4', '5', '6', '7']
        )
    )

@pytest.mark.integration
def test_text_summarization_integration(test_config, test_service_manager, test_conversation_window):
    """Test actual text summarization with live service."""
    transform = OllamaSummarize(test_config)
    transform._service_manager = test_service_manager
    transform.setup()
    
    try:
        # Process the conversation through the transform
        result = list(transform.process(test_conversation_window))
        
        # Validate results
        assert len(result) == 1
        processed_window = result[0]
        
        # Validate that summary_text is populated
        assert processed_window.summary_text is not None
        assert len(processed_window.summary_text) > 10  # Should have reasonable length
        
        # Print the generated summary for inspection
        print(f"\nGenerated text summary: {processed_window.summary_text[:300]}...")
        
        # Verify the summary contains key information from the conversation
        summary = processed_window.summary_text.lower()
        assert any(name.lower() in summary for name in ['alice', 'bob', 'charlie']), "Summary should mention at least one participant"
        assert 'mountain' in summary or 'hike' in summary or 'rainier' in summary, "Summary should mention key topics"
        
    finally:
        # Clean up resources
        transform.teardown()

if __name__ == "__main__":
    pytest.main(["-v", __file__])