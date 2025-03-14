"""Tests for text summarization transformer."""
import pytest
from unittest.mock import Mock, patch
import json
import httpx
import uuid
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
import yaml
from pydantic import ValidationError
import logging

from noosphere.telegram.batch.transforms.text.summarize import OllamaSummarize
from noosphere.telegram.batch.schema import Window, WindowMetadata
from noosphere.telegram.batch.config import Config, TextSummarizerConfig
from noosphere.telegram.batch.services.llm import LLMServiceManager, LLMService, LLMModel

def load_config():
    with open('conf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config).text_summarizer

@pytest.fixture
def real_config():
    return load_config()

@pytest.fixture
def test_config():
    return TextSummarizerConfig(
        llm_service="test_service",
        model="test-model",
        timeout=30,
        retries=3,
        retry_delay=5,
        input_field="concatenated_text",
        output_field="summary_text",
        output_type="text",
        prompt="Summarize: {input_text}"
    )

@pytest.fixture
def test_service_manager():
    """Create test service manager."""
    service = LLMService(
        name="test_service",
        type="ollama",
        url="http://test-ollama:11434",
        models=[
            LLMModel(
                name="test-model",
                model="test-model",
                prompt="Summarize: {input_text}"
            )
        ]
    )
    manager = LLMServiceManager([service])
    manager.setup()  # Initialize the manager
    return manager

@pytest.fixture
def test_window():
    """Create test window."""
    window_id = uuid.UUID('12345678-1234-1234-1234-123456789012')
    return Window(
        id=window_id,
        conversation_id='test-conv-1',
        conversation_type='test',
        concatenated_text='This is a test text to summarize',
        summary_text=None,
        window_metadata=WindowMetadata(
            id=window_id,
            conversation_id='test-conv-1',
            conversation_type='test',
            date_range=['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
            unixtime_range=[1704067200, 1704070800],
            from_ids={'user1': 1},
            event_ids=['1', '2']
        )
    )

@pytest.fixture
def test_conversation_window():
    """Create test window with conversation-formatted text following consistent pattern."""
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

def test_summarize_success(test_config, test_service_manager, test_window):
    """Test successful summarization."""
    transform = OllamaSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()

    mock_summary = "This is a test summary"
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = mock_summary

        result = list(transform.process(test_window))
        
        assert len(result) == 1
        assert result[0].summary_text == mock_summary
        mock_generate.assert_called_once_with(
            service_name=test_config.llm_service,
            model_name=test_config.model,
            input_text=test_window.concatenated_text,
            timeout=float(test_config.timeout)
        )

def test_summarize_no_text(test_config, test_service_manager, test_window):
    """Test summarization when no text is available."""
    window_no_text = Window(**{**test_window.model_dump(), 'concatenated_text': None})
    
    transform = OllamaSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        result = list(transform.process(window_no_text))
        
        assert len(result) == 1
        assert result[0] == window_no_text  # Should remain unchanged
        mock_generate.assert_not_called()

def test_summarize_api_error(test_config, test_service_manager, test_window):
    """Test handling of API errors."""
    transform = OllamaSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.side_effect = httpx.RequestError("API Error")
        
        result = list(transform.process(test_window))
        
        assert len(result) == 1
        assert result[0] == test_window  # Should remain unchanged
        assert mock_generate.call_count == test_config.retries

def test_summarize_batch(test_config, test_service_manager, test_window):
    """Test summarization of multiple windows."""
    transform = OllamaSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()
    
    windows = [
        test_window,
        Window(**{
            **test_window.model_dump(),
            'id': uuid.UUID('12345678-1234-1234-1234-123456789013'),
            'window_metadata': WindowMetadata(**{
                **test_window.window_metadata.model_dump(),
                'id': uuid.UUID('12345678-1234-1234-1234-123456789013')
            }),
            'concatenated_text': 'Another test'
        })
    ]

    mock_summary = "Test summary"
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = mock_summary
        
        result = []
        for window in windows:
            result.extend(transform.process(window))
        
        assert len(result) == 2
        assert all(r.summary_text == mock_summary for r in result)
        assert mock_generate.call_count == 2

def test_summarize_pickling(test_config, test_service_manager, test_window):
    """Test that transform can be pickled and unpickled."""
    import pickle
    
    # Create transform and verify initial state
    transform = OllamaSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    assert not transform._setup_called
    
    # Setup and create some state
    transform.setup()
    assert transform._setup_called
    
    # Pickle and unpickle
    pickled = pickle.dumps(transform)
    unpickled = pickle.loads(pickled)
    
    # Verify state was reset
    assert not unpickled._setup_called
    
    # Set up the unpickled transform with the test service manager
    unpickled._service_manager = test_service_manager
    unpickled.setup()
    
    # Test that transform works after unpickling
    mock_summary = "Test summary"
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = mock_summary
        
        result = list(unpickled.process(test_window))
        
        assert len(result) == 1
        assert result[0].summary_text == mock_summary
        mock_generate.assert_called_once()

def test_summarize_setup_teardown(test_config, test_service_manager):
    """Test setup and teardown behavior."""
    transform = OllamaSummarize(test_config)
    
    # Test initial state
    assert not transform._setup_called
    # Logger is now set up in __init__
    assert hasattr(transform, 'logger')
    
    # Test setup
    transform.setup()
    assert transform._setup_called
    assert hasattr(transform, 'logger')
    
    # Test idempotent setup
    transform.setup()
    assert transform._setup_called
    
    # Test teardown
    transform.teardown()
    assert not transform._setup_called

def test_summarize_invalid_output_type(test_config):
    """Test validation of output_type in config."""
    with pytest.raises(ValidationError) as exc_info:
        TextSummarizerConfig(
            **{
                **test_config.model_dump(),
                'output_type': 'vector'
            }
        )
    assert "output_type must be 'text' for text summarizer" in str(exc_info.value)

def test_conversation_format_handling(test_config, test_service_manager, test_conversation_window):
    """Test that the summarizer properly handles conversation-formatted text with:
    - Speaker name prefixes [Name]:
    - Intra-message newlines
    - Markdown-formatted images
    """
    transform = OllamaSummarize(test_config)
    transform._service_manager = test_service_manager
    transform.setup()
    
    # Use a mock summary that preserves some of the conversation format elements
    mock_summary = (
        "Summary of conversation between Alice, Bob, and Charlie:\n\n"
        "Alice greeted everyone and Bob mentioned working on a project.\n"
        "Alice shared a picture from a hike: ![Mountain view](/path/to/mountain_lake.jpg)\n"
        "Charlie asked where it was taken, and Alice explained it was Mount Rainier.\n"
        "Bob suggested planning a group hike there."
    )
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = mock_summary
        
        result = list(transform.process(test_conversation_window))
        
        # Verify the summary content
        assert len(result) == 1
        assert result[0].summary_text == mock_summary
        
        # Check that the input formatting was preserved when passing to the LLM
        mock_generate.assert_called_once()
        
        # Verify the input text has the expected format elements
        call_args = mock_generate.call_args[1]
        input_text = call_args['input_text']
        
        # Verify speaker name format
        assert "[Alice]:" in input_text
        assert "[Bob]:" in input_text
        assert "[Charlie]:" in input_text
        
        # Verify markdown image format is preserved
        assert "![A beautiful mountain view" in input_text
        assert "](/path/to/mountain_lake.jpg)" in input_text
        
        # Verify multiline messages
        assert "Hello everyone," in input_text
        assert "How are you all doing today?" in input_text
        
        # Count number of speaker turns
        speaker_turns = input_text.count("[Alice]:") + input_text.count("[Bob]:") + input_text.count("[Charlie]:")
        assert speaker_turns == 7  # Should match the number of message events 