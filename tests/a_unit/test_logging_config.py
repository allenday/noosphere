"""Unit tests for the logging configuration system."""
import pytest
import yaml
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from telexp.config import LoggingConfig, Config
import telexp.logging as logging_module


def test_logging_config_defaults():
    """Test that LoggingConfig provides expected defaults."""
    config = LoggingConfig()
    
    # Check defaults
    assert config.level == "INFO"
    assert config.distributed is False
    assert config.log_file is None
    assert config.json_logs is False
    assert config.intercept_std_logging is True
    assert isinstance(config.module_levels, dict)
    assert config.rotation == "10 MB"
    assert config.retention == "1 week"
    assert config.compression == "zip"
    assert config.enqueue is False  # Now default to False for Beam compatibility
    assert config.log_startup is True


def test_logging_config_custom():
    """Test that LoggingConfig can be customized."""
    config = LoggingConfig(
        level="DEBUG",
        distributed=True,
        log_file="logs/test.log",
        json_logs=True,
        intercept_std_logging=False,
        module_levels={
            "test_module": "ERROR"
        },
        rotation="1 day",
        retention="5 days",
        compression="gz",
        enqueue=False,
        log_startup=False
    )
    
    # Check custom values
    assert config.level == "DEBUG"
    assert config.distributed is True
    assert config.log_file == "logs/test.log"
    assert config.json_logs is True
    assert config.intercept_std_logging is False
    assert config.module_levels == {"test_module": "ERROR"}
    assert config.rotation == "1 day"
    assert config.retention == "5 days"
    assert config.compression == "gz"
    assert config.enqueue is False
    assert config.log_startup is False


def test_logging_config_validation():
    """Test that LoggingConfig validates inputs."""
    # Test invalid log level
    with pytest.raises(ValueError):
        LoggingConfig(level="INVALID")
    
    # These should work
    LoggingConfig(level="debug")  # lowercase
    LoggingConfig(level="INFO")  # uppercase
    LoggingConfig(level="Warning")  # mixed case


def test_config_yaml_roundtrip():
    """Test that LoggingConfig can be serialized to YAML and back."""
    # Create a config
    config = LoggingConfig(
        level="DEBUG",
        distributed=True,
        log_file="logs/test.log",
        json_logs=True,
        module_levels={"httpx": "ERROR"}
    )
    
    # Convert to dict
    config_dict = config.model_dump()
    
    # Serialize to YAML
    yaml_str = yaml.dump(config_dict)
    
    # Parse YAML back to dict
    parsed_dict = yaml.safe_load(yaml_str)
    
    # Create new config from parsed dict
    parsed_config = LoggingConfig(**parsed_dict)
    
    # Check values match
    assert parsed_config.level == config.level
    assert parsed_config.distributed == config.distributed
    assert parsed_config.log_file == config.log_file
    assert parsed_config.json_logs == config.json_logs
    assert parsed_config.module_levels == config.module_levels


def test_config_integration():
    """Test that LoggingConfig integrates with main Config class."""
    # Create a minimal config YAML
    config_yaml = """
    logging:
      level: "DEBUG"
      log_file: "logs/app.log"
      module_levels:
        apache_beam: "WARNING"
    
    llm_services:
      - name: test
        type: ollama
        url: http://test
        models:
          - name: test
            model: test
            default_prompt: "test"
    
    embedding_services:
      - name: test
        type: ollama
        url: http://test
        models:
          - name: test
            model: test
            vector_size: 128
    
    text_vectorizer:
      embedding_service: test
      model: test
      vector_size: 128
      output_type: vector
    
    text_summarizer:
      llm_service: test
      model: test
      prompt: test
      output_type: text
    
    image_summarizer:
      llm_service: test
      model: test
      prompt: test
      output_type: text
    
    text_vector_storage:
      url: http://test
      collection: test
      vector_size: 128
    
    image_vector_storage:
      url: http://test
      collection: test
      vector_size: 128
    
    window_by_session:
      timeout_seconds: 3600
      min_size: 1
      max_size: 10
    
    window_by_size:
      size: 5
      overlap: 2
    """
    
    # Write config to temporary file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp.write(config_yaml.encode('utf-8'))
        tmp_path = tmp.name
    
    try:
        # Load config from file
        config = Config.from_yaml(tmp_path)
        
        # Check logging config was loaded correctly
        assert config.logging.level == "DEBUG"
        assert config.logging.log_file == "logs/app.log"
        assert config.logging.module_levels == {"apache_beam": "WARNING"}
        
        # Check defaults were applied
        assert config.logging.distributed is False
        assert config.logging.json_logs is False
    finally:
        # Clean up
        os.unlink(tmp_path)


def test_setup_logging_from_config():
    """Test that setup_logging works with config object."""
    config = LoggingConfig(
        level="DEBUG",
        distributed=True,
        log_file=None,  # No file output for test
        json_logs=True,
        module_levels={"test": "ERROR"}
    )
    
    # Mock the logger configuration
    with patch.object(logging_module.logger, 'configure') as mock_configure:
        # Call setup_logging with config values
        logging_module.setup_logging(
            level=config.level,
            distributed=config.distributed,
            log_file=config.log_file,
            json_logs=config.json_logs,
            intercept_std_logging=config.intercept_std_logging
        )
        
        # Verify configure was called with expected parameters
        mock_configure.assert_called_once()
        args, kwargs = mock_configure.call_args
        handlers = kwargs.get('handlers', [])
        
        # Verify handlers reflect config settings
        assert len(handlers) > 0
        assert handlers[0].get('level') == "DEBUG"
        assert handlers[0].get('serialize') is True  # json_logs=True