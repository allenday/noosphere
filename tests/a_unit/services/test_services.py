import pytest
from unittest.mock import Mock, patch
import pickle
import httpx
from typing import List

from telexp.services.llm import LLMService, LLMModel, LLMServiceManager
from telexp.services.embedding import EmbeddingService, EmbeddingModel, EmbeddingServiceManager

@pytest.fixture
def test_llm_service():
    """Test LLM service configuration."""
    return LLMService(
        name="test_llm",
        type="ollama",
        url="http://test:11434",
        models=[
            LLMModel(
                name="test-model",
                model="test-model",
                prompt="Test prompt: {input_text}"
            )
        ]
    )

@pytest.fixture
def test_embedding_service():
    """Test embedding service configuration."""
    return EmbeddingService(
        name="test_embedding",
        type="ollama",
        url="http://test:11434",
        models=[
            EmbeddingModel(
                name="test-model",
                model="test-model",
                vector_size=768
            )
        ]
    )

def test_llm_service_manager_init(test_llm_service):
    """Test LLMServiceManager initialization."""
    manager = LLMServiceManager([test_llm_service])
    assert "test_llm" in manager.services
    assert manager.clients == {}

def test_llm_service_manager_get_service(test_llm_service):
    """Test getting service by name."""
    manager = LLMServiceManager([test_llm_service])
    service = manager.get_service("test_llm")
    assert service == test_llm_service
    assert manager.get_service("nonexistent") is None

def test_llm_service_manager_get_model(test_llm_service):
    """Test getting model by service and model name."""
    manager = LLMServiceManager([test_llm_service])
    model = manager.get_model("test_llm", "test-model")
    assert model == test_llm_service.models[0]
    assert manager.get_model("test_llm", "nonexistent") is None
    assert manager.get_model("nonexistent", "test-model") is None

def test_llm_service_manager_get_client(test_llm_service):
    """Test getting HTTP client for service."""
    manager = LLMServiceManager([test_llm_service])
    client = manager.get_client("test_llm")
    assert isinstance(client, httpx.Client)
    assert client.base_url == test_llm_service.url
    assert manager.get_client("nonexistent") is None

def test_llm_service_manager_close_all(test_llm_service):
    """Test closing all HTTP clients."""
    manager = LLMServiceManager([test_llm_service])
    client = manager.get_client("test_llm")
    assert "test_llm" in manager.clients
    manager.close_all()
    assert not manager.clients

def test_llm_service_manager_generate(test_llm_service):
    """Test text generation."""
    manager = LLMServiceManager([test_llm_service])
    
    mock_response = {
        "model": "test-model",
        "response": "Generated text",
        "done": True
    }
    
    with patch('httpx.Client.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        result = manager.generate(
            service_name="test_llm",
            model_name="test-model",
            input_text="Test input"
        )
        
        assert result == "Generated text"
        mock_post.assert_called_once()
        assert mock_post.call_args[1]['json']['prompt'] == "Test prompt: Test input"

def test_llm_service_manager_generate_error(test_llm_service):
    """Test error handling in text generation."""
    manager = LLMServiceManager([test_llm_service])
    
    with patch('httpx.Client.post') as mock_post:
        mock_post.side_effect = httpx.RequestError("API Error")
        
        result = manager.generate(
            service_name="test_llm",
            model_name="test-model",
            input_text="Test input"
        )
        
        assert result is None

def test_embedding_service_manager_init(test_embedding_service):
    """Test EmbeddingServiceManager initialization."""
    manager = EmbeddingServiceManager([test_embedding_service])
    assert "test_embedding" in manager.services
    assert manager.clients == {}

def test_embedding_service_manager_get_service(test_embedding_service):
    """Test getting service by name."""
    manager = EmbeddingServiceManager([test_embedding_service])
    service = manager.get_service("test_embedding")
    assert service == test_embedding_service
    assert manager.get_service("nonexistent") is None

def test_embedding_service_manager_get_model(test_embedding_service):
    """Test getting model by service and model name."""
    manager = EmbeddingServiceManager([test_embedding_service])
    model = manager.get_model("test_embedding", "test-model")
    assert model == test_embedding_service.models[0]
    assert manager.get_model("test_embedding", "nonexistent") is None
    assert manager.get_model("nonexistent", "test-model") is None

def test_embedding_service_manager_get_client(test_embedding_service):
    """Test getting HTTP client for service."""
    manager = EmbeddingServiceManager([test_embedding_service])
    client = manager.get_client("test_embedding")
    assert isinstance(client, httpx.Client)
    assert client.base_url == test_embedding_service.url
    assert manager.get_client("nonexistent") is None

def test_embedding_service_manager_close_all(test_embedding_service):
    """Test closing all HTTP clients."""
    manager = EmbeddingServiceManager([test_embedding_service])
    client = manager.get_client("test_embedding")
    assert "test_embedding" in manager.clients
    manager.close_all()
    assert not manager.clients

def test_embedding_service_manager_embed(test_embedding_service):
    """Test embedding generation."""
    manager = EmbeddingServiceManager([test_embedding_service])
    
    mock_embedding = [0.1] * 768  # Match vector_size
    mock_response = {
        "model": "test-model",
        "embedding": mock_embedding
    }
    
    with patch('httpx.Client.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        result = manager.embed(
            service_name="test_embedding",
            model_name="test-model",
            text="Test input"
        )
        
        assert result == mock_embedding
        mock_post.assert_called_once()
        assert mock_post.call_args[1]['json']['prompt'] == "Test input"

def test_embedding_service_manager_embed_wrong_size(test_embedding_service):
    """Test handling of wrong embedding size."""
    manager = EmbeddingServiceManager([test_embedding_service])
    
    mock_response = {
        "model": "test-model",
        "embedding": [0.1] * 512  # Wrong size
    }
    
    with patch('httpx.Client.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        result = manager.embed(
            service_name="test_embedding",
            model_name="test-model",
            text="Test input"
        )
        
        assert result is None

def test_embedding_service_manager_embed_error(test_embedding_service):
    """Test error handling in embedding generation."""
    manager = EmbeddingServiceManager([test_embedding_service])
    
    with patch('httpx.Client.post') as mock_post:
        mock_post.side_effect = httpx.RequestError("API Error")
        
        result = manager.embed(
            service_name="test_embedding",
            model_name="test-model",
            text="Test input"
        )
        
        assert result is None

def test_llm_service_pickling(test_llm_service):
    """Test pickling/unpickling of LLM service."""
    # Test service pickling
    pickled = pickle.dumps(test_llm_service)
    unpickled = pickle.loads(pickled)
    assert unpickled == test_llm_service
    
    # Test manager pickling
    manager = LLMServiceManager([test_llm_service])
    manager.setup()  # Set up before pickling
    client = manager.get_client("test_llm")  # Create a client
    
    pickled = pickle.dumps(manager)
    unpickled = pickle.loads(pickled)
    
    # Verify state
    assert unpickled.services == manager.services
    assert not unpickled.clients  # Clients should not be pickled
    assert not unpickled._setup_called  # Should need setup again
    assert not hasattr(unpickled, 'logger')  # Logger should be recreated
    
    # Verify functionality still works
    assert isinstance(unpickled.get_client("test_llm"), httpx.Client)
    assert unpickled._setup_called  # Should be set up now

def test_embedding_service_pickling(test_embedding_service):
    """Test pickling/unpickling of embedding service."""
    # Test service pickling
    pickled = pickle.dumps(test_embedding_service)
    unpickled = pickle.loads(pickled)
    assert unpickled == test_embedding_service
    
    # Test manager pickling
    manager = EmbeddingServiceManager([test_embedding_service])
    manager.setup()  # Set up before pickling
    client = manager.get_client("test_embedding")  # Create a client
    
    pickled = pickle.dumps(manager)
    unpickled = pickle.loads(pickled)
    
    # Verify state
    assert unpickled.services == manager.services
    assert not unpickled.clients  # Clients should not be pickled
    assert not unpickled._setup_called  # Should need setup again
    assert not hasattr(unpickled, 'logger')  # Logger should be recreated
    
    # Verify functionality still works
    assert isinstance(unpickled.get_client("test_embedding"), httpx.Client)
    assert unpickled._setup_called  # Should be set up now 