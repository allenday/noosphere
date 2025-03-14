"""Tests for text vectorization transformer."""
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

from telexp.transforms.text.vectorize import OllamaVectorize
from telexp.schema import Window, WindowMetadata
from telexp.config import Config, TextVectorizerConfig
from telexp.services.embedding import EmbeddingServiceManager, EmbeddingService, EmbeddingModel

def load_config():
    with open('conf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config).text_vectorizer

@pytest.fixture
def real_config():
    return load_config()

@pytest.fixture
def test_config():
    return TextVectorizerConfig(
        embedding_service="test_service",
        model="test-model",
        timeout=30,
        retries=3,
        retry_delay=5,
        input_field="summary_text",  # Updated to use summary_text which is already in our test_window
        output_field="vector",
        output_type="vector",
        vector_size=768
    )

@pytest.fixture
def test_service_manager():
    """Create test service manager."""
    services = [
        EmbeddingService(
            name='test_service',
            type='ollama',
            url='http://test-ollama:11434',
            models=[
                EmbeddingModel(
                    name='test-model',
                    model='test-model',
                    vector_size=768
                )
            ]
        )
    ]
    manager = EmbeddingServiceManager(services)
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
        summary_text='This is a test summary',
        vector=[0.0] * 768,
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

def test_vectorize_success(test_config, test_service_manager, test_window):
    """Test successful vectorization using Beam TestPipeline."""
    transform = OllamaVectorize(test_config, test_service_manager)
    transform.setup()

    mock_embedding = [0.1] * 768
    
    with patch.object(test_service_manager, 'embed') as mock_embed:
        mock_embed.return_value = mock_embedding

        result = list(transform.process(test_window))
        
        assert len(result) == 1
        assert result[0].vector == mock_embedding
        mock_embed.assert_called_once_with(
            service_name=test_config.embedding_service,
            model_name=test_config.model,
            text=test_window.summary_text,
            timeout=float(test_config.timeout)
        )

def test_vectorize_no_text(test_config, test_service_manager, test_window):
    """Test vectorization when no text is available."""
    window_no_text = Window(**{**test_window.model_dump(), 'summary_text': None})
    
    transform = OllamaVectorize(test_config, test_service_manager)
    transform.setup()
    
    with patch.object(test_service_manager, 'embed') as mock_embed:
        result = list(transform.process(window_no_text))
        
        assert len(result) == 1
        assert result[0] == window_no_text  # Should remain unchanged
        mock_embed.assert_not_called()

def test_vectorize_api_error(test_config, test_service_manager, test_window):
    """Test handling of API errors."""
    transform = OllamaVectorize(test_config, test_service_manager)
    transform.setup()
    
    with patch.object(test_service_manager, 'embed') as mock_embed:
        mock_embed.side_effect = httpx.RequestError("API Error")
        
        result = list(transform.process(test_window))
        
        assert len(result) == 1
        assert result[0] == test_window  # Should remain unchanged
        assert mock_embed.call_count == test_config.retries

def test_vectorize_batch(test_config, test_service_manager, test_window):
    """Test vectorization of multiple windows."""
    transform = OllamaVectorize(test_config, test_service_manager)
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
            'summary_text': 'Another test'
        })
    ]

    mock_embedding = [0.1] * 768
    
    with patch.object(test_service_manager, 'embed') as mock_embed:
        mock_embed.return_value = mock_embedding
        
        result = []
        for window in windows:
            result.extend(transform.process(window))
        
        assert len(result) == 2
        assert all(r.vector == mock_embedding for r in result)
        assert mock_embed.call_count == 2

def test_vectorize_wrong_vector_size(test_config, test_service_manager, test_window):
    """Test validation of received vector size."""
    transform = OllamaVectorize(test_config, test_service_manager)
    transform.setup()
    
    wrong_size_embedding = [0.1] * 512  # Wrong size
    
    with patch.object(test_service_manager, 'embed') as mock_embed:
        mock_embed.return_value = wrong_size_embedding
        
        result = list(transform.process(test_window))
        
        assert len(result) == 1
        assert result[0] == test_window  # Should remain unchanged
        mock_embed.assert_called_once()

def test_vectorize_pickling(test_config, test_service_manager, test_window):
    """Test that transform can be pickled and unpickled."""
    import pickle
    
    # Create transform and verify initial state
    transform = OllamaVectorize(test_config, test_service_manager)
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
    
    # Explicitly call setup on the unpickled instance
    unpickled.setup()
    assert unpickled._setup_called
    
    # Test that transform works after unpickling
    mock_embedding = [0.1] * 768
    
    with patch.object(test_service_manager, 'embed') as mock_embed:
        mock_embed.return_value = mock_embedding
        
        result = list(unpickled.process(test_window))
        
        assert len(result) == 1
        assert result[0].vector == mock_embedding
        mock_embed.assert_called_once()
        assert unpickled._setup_called

def test_vectorize_setup_teardown(test_config, test_service_manager):
    """Test setup and teardown behavior."""
    transform = OllamaVectorize(test_config, test_service_manager)
    
    # Test initial state
    assert not transform._setup_called
    # We no longer set up logger in __init__ so it might not be present initially
    if hasattr(transform, 'logger'):
        # If logger is present, make sure it has the expected attributes
        assert transform.logger_name.startswith('OllamaVectorize_')
    
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

def test_vectorize_invalid_output_type(test_config):
    """Test validation of output_type in config."""
    with pytest.raises(ValidationError) as exc_info:
        TextVectorizerConfig(
            **{
                **test_config.model_dump(),
                'output_type': 'text'
            }
        )
    assert "output_type must be 'vector' for text vectorizer" in str(exc_info.value)

def test_vectorize_invalid_vector_size():
    """Test validation of vector_size in config."""
    # Test negative size
    with pytest.raises(ValueError) as exc_info:
        TextVectorizerConfig(
            embedding_service="test_service",
            model="test-model",
            input_field="text",
            output_field="vector",
            output_type="vector",
            vector_size=-1
        )
    assert "Invalid vector_size" in str(exc_info.value) 