import pytest
from pathlib import Path
import yaml
from telexp.config import Config, load_config

def test_config_from_yaml(tmp_path):
    """Test loading config from a YAML file."""
    # Create a test config file
    config_data = {
        "llm_services": [{
            "name": "test_llm",
            "type": "ollama",
            "url": "http://test:11434",
            "models": [{
                "name": "test-model",
                "model": "test-model",
                "context_window": 4096,
                "max_tokens": 1024,
                "prompt": "Default prompt: {input_text}"
            }]
        }],
        "embedding_services": [{
            "name": "test_embedding",
            "type": "ollama",
            "url": "http://test:11434",
            "models": [{
                "name": "test-model",
                "model": "test-model",
                "vector_size": 768
            }]
        }],
        "text_vectorizer": {
            "url": "http://test:11434",
            "model": "test-model",
            "timeout": 30,
            "retries": 3,
            "input_field": "text",
            "output_field": "vector",
            "output_type": "vector",
            "vector_size": 768,
            "embedding_service": "test_embedding"
        },
        "text_summarizer": {
            "url": "http://test:11434",
            "model": "test-model",
            "timeout": 30,
            "retries": 3,
            "input_field": "text",
            "output_field": "summary",
            "output_type": "text",
            "prompt": "Summarize: {input_text}",
            "llm_service": "test_llm"
        },
        "image_summarizer": {
            "url": "http://test:11434",
            "model": "test-model",
            "timeout": 30,
            "retries": 3,
            "input_field": "file",
            "output_field": "summary",
            "output_type": "text",
            "prompt": "Describe this image",
            "llm_service": "test_llm"
        },
        "text_vector_storage": {
            "url": "http://test-qdrant:6333",
            "collection": "test_vectors",
            "test_collection": "test_vectors_test",
            "vector_size": 768,
            "retries": 3,
            "timeout": 30,
            "batch_size": 100
        },
        "image_vector_storage": {
            "url": "http://test-qdrant:6333",
            "collection": "image_vectors",
            "test_collection": "image_vectors_test",
            "vector_size": 768,
            "retries": 3,
            "timeout": 30,
            "batch_size": 100
        },
        "window_by_session": {
            "timeout_seconds": 3600,
            "min_size": 1,
            "max_size": 100
        },
        "window_by_size": {
            "size": 5,
            "overlap": 2
        }
    }
    
    config_file = tmp_path / "test_conf.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    # Test loading config
    config = Config.from_yaml(config_file)
    assert config.text_vectorizer.url == "http://test:11434"
    assert config.text_summarizer.prompt == "Summarize: {input_text}"
    assert config.image_summarizer.input_field == "file"
    assert config.window_by_session.timeout_seconds == 3600
    assert config.window_by_size.size == 5

def test_config_from_yaml_not_found():
    """Test error when config file not found."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("nonexistent.yaml")

def test_load_config_default(tmp_path):
    """Test loading config from default location."""
    # Create a test config file
    config_data = {
        "llm_services": [{
            "name": "test_llm",
            "type": "ollama",
            "url": "http://test:11434",
            "models": [{
                "name": "test-model",
                "model": "test-model",
                "context_window": 4096,
                "max_tokens": 1024,
                "prompt": "Default prompt: {input_text}"
            }]
        }],
        "embedding_services": [{
            "name": "test_embedding",
            "type": "ollama",
            "url": "http://test:11434",
            "models": [{
                "name": "test-model",
                "model": "test-model",
                "vector_size": 768
            }]
        }],
        "text_vectorizer": {
            "url": "http://test:11434",
            "model": "test-model",
            "timeout": 30,
            "retries": 3,
            "input_field": "text",
            "output_field": "vector",
            "output_type": "vector",
            "vector_size": 768,
            "embedding_service": "test_embedding"
        },
        "text_summarizer": {
            "url": "http://test:11434",
            "model": "test-model",
            "timeout": 30,
            "retries": 3,
            "input_field": "text",
            "output_field": "summary",
            "output_type": "text",
            "prompt": "Summarize: {input_text}",
            "llm_service": "test_llm"
        },
        "image_summarizer": {
            "url": "http://test:11434",
            "model": "test-model",
            "timeout": 30,
            "retries": 3,
            "input_field": "file",
            "output_field": "summary",
            "output_type": "text",
            "prompt": "Describe this image",
            "llm_service": "test_llm"
        },
        "text_vector_storage": {
            "url": "http://test-qdrant:6333",
            "collection": "test_vectors",
            "test_collection": "test_vectors_test",
            "vector_size": 768,
            "retries": 3,
            "timeout": 30,
            "batch_size": 100
        },
        "image_vector_storage": {
            "url": "http://test-qdrant:6333",
            "collection": "image_vectors",
            "test_collection": "image_vectors_test",
            "vector_size": 768,
            "retries": 3,
            "timeout": 30,
            "batch_size": 100
        },
        "window_by_session": {
            "timeout_seconds": 3600,
            "min_size": 1,
            "max_size": 100
        },
        "window_by_size": {
            "size": 5,
            "overlap": 2
        }
    }
    
    # Create config in current directory
    with open(tmp_path / "conf.yaml", "w") as f:
        yaml.dump(config_data, f)
    
    # Change to temp directory and load config
    original_cwd = Path.cwd()
    try:
        Path.cwd = lambda: tmp_path  # Mock cwd() to return our temp path
        config = load_config()
        assert config.text_vectorizer.url == "http://test:11434"
    finally:
        Path.cwd = lambda: original_cwd  # Restore original cwd

def test_load_config_not_found(tmp_path):
    """Test error when no config file found."""
    # Create a temp directory with no conf.yaml
    original_cwd = Path.cwd
    try:
        Path.cwd = lambda: tmp_path  # Mock cwd() to return our empty temp path
        with pytest.raises(FileNotFoundError):
            load_config()
    finally:
        Path.cwd = original_cwd  # Restore original cwd 