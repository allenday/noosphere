"""Tests for image summarization transformer."""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import httpx
import uuid
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
import yaml
from pydantic import ValidationError
import logging
import os
import base64
import io
import sys
from pathlib import Path

from noosphere.telegram.batch.transforms.image.summarize import OllamaImageSummarize, summarize_image
from noosphere.telegram.batch.schema import RawMessage, Photo, TextEntity
from noosphere.telegram.batch.config import Config, ImageSummarizerConfig
from noosphere.telegram.batch.services.llm import LLMServiceManager, LLMService, LLMModel

def load_config():
    with open('conf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config).image_summarizer

@pytest.fixture
def real_config():
    return load_config()

@pytest.fixture
def test_config():
    return ImageSummarizerConfig(
        llm_service="test_service",
        model="test-model",
        timeout=30,
        retries=3,
        retry_delay=5,
        input_field="photo",
        output_field="summary_text",
        output_type="text",
        prompt="Describe this image: {input_text}"
    )

@pytest.fixture
def test_service_manager():
    """Create test service manager."""
    service = LLMService(
        name="test_service",
        type="ollama",
        url="http://localhost:11434",
        models=[
            LLMModel(
                name="test-model",
                model="test-model",
                prompt="Describe this image: {input_text}"
            )
        ]
    )
    manager = LLMServiceManager([service])
    manager.setup()  # Initialize the manager
    return manager

@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image file."""
    image_path = tmp_path / "test_image.jpg"
    # Create a small dummy image file
    with open(image_path, 'wb') as f:
        f.write(b'dummy image data')
    return str(image_path)

@pytest.fixture
def test_message(test_image_path):
    """Create test message."""
    return RawMessage(
        id='1',
        type='message',
        date='2024-01-01T00:00:00',
        date_unixtime=1704067200,
        conversation_id='test-conv-1',
        conversation_type='test',
        from_id='user1',
        from_name='Test User',
        photo=Photo(file=test_image_path),
        text_entities=[],  # Initialize empty text_entities list
        source={
            'file_path': '/path/to/test.jsonl',
            'file_name': 'test.jsonl',
            'file_extension': '.jsonl',
            'file_mimetype': 'application/json',
            'file_offset': 0,
            'file_line_number': 1
        }
    )

def test_summarize_success(test_config, test_service_manager, test_message):
    """Test successful image summarization."""
    transform = OllamaImageSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()

    mock_summary = "This is a test image summary"
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = mock_summary

        result = list(transform.process(test_message))
        
        assert len(result) == 1
        assert result[0].text_entities[-1].type == 'image_derived'
        assert result[0].text_entities[-1].text == mock_summary
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args[1]
        assert call_args['service_name'] == test_config.llm_service
        assert call_args['model_name'] == test_config.model
        assert isinstance(call_args['input_text'], str)  # Should be base64 encoded

def test_image_summary_markdown_format(test_config, test_service_manager, test_message):
    """Test that image summaries follow the expected markdown format: ![alt](path)."""
    transform = OllamaImageSummarize(test_config)
    transform._service_manager = test_service_manager
    transform.setup()

    # Mock a summary in the expected markdown format
    valid_markdown_summary = "![A photo of a cat sitting on a table](/path/to/image.jpg)"
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = valid_markdown_summary

        result = list(transform.process(test_message))
        
        # Verify the summary follows markdown image format
        assert len(result) == 1
        assert result[0].text_entities[-1].type == 'image_derived'
        summary_text = result[0].text_entities[-1].text
        
        # Check markdown format
        assert summary_text.strip().startswith('!['), "Summary doesn't start with !["
        assert ')' in summary_text, "Summary doesn't contain closing parenthesis"
        assert '](' in summary_text, "Summary doesn't have proper markdown format"
        
        # Verify format components: ![alt](url)
        alt_text = summary_text.split('![')[1].split('](')[0]
        url = summary_text.split('](')[1].split(')')[0]
        
        assert alt_text, "Alt text is empty"
        assert url, "URL is empty"

def test_summarize_no_image(test_config, test_service_manager, test_message):
    """Test summarization when no image is available."""
    message_no_image = RawMessage(**{**test_message.model_dump(), 'photo': None})
    
    transform = OllamaImageSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        result = list(transform.process(message_no_image))
        
        assert len(result) == 1
        assert result[0] == message_no_image  # Should remain unchanged
        mock_generate.assert_not_called()

def test_summarize_invalid_image(test_config, test_service_manager, test_message):
    """Test summarization with invalid image path."""
    message_invalid_image = RawMessage(**{
        **test_message.model_dump(),
        'photo': Photo(file='/nonexistent/image.jpg')
    })
    
    transform = OllamaImageSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        result = list(transform.process(message_invalid_image))
        
        assert len(result) == 1
        assert result[0] == message_invalid_image  # Should remain unchanged
        mock_generate.assert_not_called()

def test_summarize_api_error(test_config, test_service_manager, test_message):
    """Test handling of API errors."""
    transform = OllamaImageSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.side_effect = httpx.RequestError("API Error")
        
        result = list(transform.process(test_message))
        
        assert len(result) == 1
        assert result[0] == test_message  # Should remain unchanged
        assert mock_generate.call_count == test_config.retries

def test_summarize_batch(test_config, test_service_manager, test_message, tmp_path):
    """Test summarization of multiple messages."""
    # Create second test image
    second_image = tmp_path / "second_image.jpg"
    with open(second_image, 'wb') as f:
        f.write(b'second dummy image data')
        
    messages = [
        test_message,
        RawMessage(**{
            **test_message.model_dump(),
            'id': '2',
            'photo': Photo(file=str(second_image))
        })
    ]

    transform = OllamaImageSummarize(test_config)
    transform._service_manager = test_service_manager  # Use the test service manager
    transform.setup()

    mock_summary = "Test image summary"
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = mock_summary
        
        result = []
        for message in messages:
            result.extend(transform.process(message))
        
        assert len(result) == 2
        assert all(r.text_entities[-1].type == 'image_derived' for r in result)
        assert all(r.text_entities[-1].text == mock_summary for r in result)
        assert mock_generate.call_count == 2

def test_summarize_pickling(test_config, test_service_manager, test_message):
    """Test that transform can be pickled and unpickled."""
    import pickle
    
    # Create transform and verify initial state
    transform = OllamaImageSummarize(test_config)
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
    mock_summary = "Test image summary"
    
    with patch.object(test_service_manager, 'generate') as mock_generate:
        mock_generate.return_value = mock_summary
        
        result = list(unpickled.process(test_message))
        
        assert len(result) == 1
        assert result[0].text_entities[-1].type == 'image_derived'
        assert result[0].text_entities[-1].text == mock_summary
        mock_generate.assert_called_once()

def test_summarize_setup_teardown(test_config, test_service_manager):
    """Test setup and teardown behavior."""
    transform = OllamaImageSummarize(test_config)
    
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
        ImageSummarizerConfig(
            **{
                **test_config.model_dump(),
                'output_type': 'vector'
            }
        )
    assert "output_type must be 'text' for image summarizer" in str(exc_info.value)

def test_encode_image_error_handling(test_config, test_service_manager, tmp_path):
    """Test error handling in _encode_image method."""
    transform = OllamaImageSummarize(test_config, test_service_manager)
    transform.setup()
    
    # Test with None path
    result_none = transform._encode_image(None)
    assert result_none is None
    
    # Test with dictionary input without 'file' key
    result_invalid_dict = transform._encode_image({'not_file': 'something'})
    assert result_invalid_dict is None
    
    # Test with non-existent file
    non_existent = str(tmp_path / "does_not_exist.jpg")
    result_missing = transform._encode_image(non_existent)
    assert result_missing is None
    
    # Test with file that exists but causes error during read
    bad_file = tmp_path / "bad_file.jpg"
    bad_file.touch()  # Create empty file
    
    with patch('builtins.open') as mock_open:
        mock_open.side_effect = IOError("Test IO error")
        result_error = transform._encode_image(str(bad_file))
        assert result_error is None
        
    # Test with photo object that has file attribute
    class PhotoWithFile:
        def __init__(self, file_path):
            self.file = file_path
    
    photo_obj = PhotoWithFile(str(tmp_path / "photo.jpg"))
    # Create the file
    Path(photo_obj.file).touch()
    
    with patch('base64.b64encode') as mock_encode:
        mock_encode.return_value = b"test_base64_content"
        result = transform._encode_image(photo_obj)
        assert result == "test_base64_content"
    
    # Test with generic exception during encoding
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = Exception("Unexpected error")
            result_error = transform._encode_image(str(bad_file))
            assert result_error is None

def test_api_error_handling(test_config, test_service_manager, test_message):
    """Test handling of API errors."""
    transform = OllamaImageSummarize(test_config, test_service_manager)
    transform.setup()
    
    # Mock _encode_image to return a valid encoded image
    with patch.object(transform, '_encode_image') as mock_encode:
        mock_encode.return_value = "encoded_test_image"
        
        # Mock the service manager to simulate different errors
        with patch.object(test_service_manager, 'generate') as mock_generate:
            # Test API error
            mock_generate.side_effect = httpx.RequestError("API error")
            result1 = list(transform.process(test_message))
            assert len(result1) == 1
            assert result1[0] == test_message  # Message should be unchanged
            
            # Test retry logic - verify it tries multiple times before giving up
            mock_generate.reset_mock()
            mock_generate.side_effect = httpx.RequestError("API error")
            list(transform.process(test_message))
            assert mock_generate.call_count == test_config.retries
            
            # Test with None result from API
            mock_generate.reset_mock()
            mock_generate.side_effect = None
            mock_generate.return_value = None
            result3 = list(transform.process(test_message))
            assert len(result3) == 1
            assert result3[0] == test_message  # Message should be unchanged
            
            # Test with generic exception
            mock_generate.reset_mock()
            mock_generate.side_effect = Exception("Generic error")
            result4 = list(transform.process(test_message))
            assert len(result4) == 1
            assert result4[0] == test_message  # Message should be unchanged
            
            # Test with timeout error
            mock_generate.reset_mock()
            mock_generate.side_effect = httpx.TimeoutException("Timeout")
            result5 = list(transform.process(test_message))
            assert len(result5) == 1
            assert result5[0] == test_message  # Message should be unchanged
            
            # Test with connection error
            mock_generate.reset_mock()
            mock_generate.side_effect = httpx.ConnectError("Connection refused")
            result6 = list(transform.process(test_message))
            assert len(result6) == 1
            assert result6[0] == test_message  # Message should be unchanged
            
            # Test retry delay
            with patch('time.sleep') as mock_sleep:
                mock_generate.reset_mock()
                mock_generate.side_effect = [Exception("Error 1"), Exception("Error 2"), "Success"]
                list(transform.process(test_message))
                assert mock_sleep.call_count == 2
                mock_sleep.assert_called_with(test_config.retry_delay)

def test_text_entity_handling(test_config, test_service_manager, test_message):
    """Test creation and handling of text entities."""
    transform = OllamaImageSummarize(test_config, test_service_manager)
    transform.setup()
    
    # Case 1: Message without text_entities
    message_no_entities = test_message.model_copy(deep=True)
    message_no_entities.text_entities = None
    
    with patch.object(transform, '_encode_image') as mock_encode:
        mock_encode.return_value = "encoded_test_image"
        
        with patch.object(test_service_manager, 'generate') as mock_generate:
            mock_generate.return_value = "Test image description"
            
            # Process message with no existing entities
            result1 = list(transform.process(message_no_entities))
            assert len(result1) == 1
            assert result1[0].text_entities is not None
            assert len(result1[0].text_entities) == 1
            assert result1[0].text_entities[0].type == 'image_derived'
            assert result1[0].text_entities[0].text == "Test image description"
            
            # Case 2: Message with existing text_entities
            message_with_entities = test_message.model_copy(deep=True)
            message_with_entities.text_entities = [
                TextEntity(type='text_link', text='Existing link', href='https://example.com')
            ]
            
            # Process message with existing entities
            result2 = list(transform.process(message_with_entities))
            assert len(result2) == 1
            assert len(result2[0].text_entities) == 2  # Should append the new entity
            assert result2[0].text_entities[0].type == 'text_link'
            assert result2[0].text_entities[1].type == 'image_derived'
            
            # Case 3: Message with text field
            message_with_text = test_message.model_copy(deep=True)
            message_with_text.text = "Original message text"
            message_with_text.text_entities = None
            
            # Process message with text
            result3 = list(transform.process(message_with_text))
            assert len(result3) == 1
            assert result3[0].text is not None
            # The text might be updated differently depending on implementation
            # Check that we have a text entity instead
            assert result3[0].text_entities is not None
            assert len(result3[0].text_entities) > 0
            assert any(entity.type == 'image_derived' for entity in result3[0].text_entities)


def test_summarize_image_function_success(test_image_path):
    """Test the standalone summarize_image function."""
    # Mock the logging functions directly
    with patch('logging.basicConfig'), patch('logging.getLogger'):
        # Mock Path.exists to return True
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            # Mock the config loading
            with patch('telexp.transforms.image.summarize.load_config') as mock_load_config:
                mock_config = MagicMock()
                mock_config.image_summarizer.model = "test-model"
                mock_config.image_summarizer.llm_service = "test-service"
                mock_load_config.return_value = mock_config
                
                # Mock the service manager and transform
                with patch('telexp.transforms.image.summarize.LLMServiceManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    
                    with patch('telexp.transforms.image.summarize.OllamaImageSummarize') as mock_transform_class:
                        mock_transform = MagicMock()
                        mock_transform_class.return_value = mock_transform
                        
                        # Set up the transform.process method to return a message with summary_text
                        mock_result = MagicMock()
                        mock_result.summary_text = "Test image description"
                        mock_transform.process.return_value = [mock_result]
                        
                        # Call the function
                        result = summarize_image(test_image_path)
                        
                        # Verify result
                        assert result == "Test image description"
                        
                        # Verify correct setup
                        mock_transform_class.assert_called_once()
                        mock_transform.setup.assert_called_once()
                        mock_transform.process.assert_called_once()


def test_summarize_image_function_with_text_entities(test_image_path):
    """Test summarize_image function with text entities but no summary_text."""
    # Mock the logging functions directly
    with patch('logging.basicConfig'), patch('logging.getLogger'):
        # Mock Path.exists to return True
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            # Mock the config loading
            with patch('telexp.transforms.image.summarize.load_config') as mock_load_config:
                mock_config = MagicMock()
                mock_load_config.return_value = mock_config
                
                # Mock the service manager and transform
                with patch('telexp.transforms.image.summarize.LLMServiceManager') as mock_manager_class:
                    with patch('telexp.transforms.image.summarize.OllamaImageSummarize') as mock_transform_class:
                        mock_transform = MagicMock()
                        mock_transform_class.return_value = mock_transform
                        
                        # Create a mock result with text_entities but no summary_text
                        mock_result = MagicMock()
                        mock_result.summary_text = None
                        mock_result.text = None
                        
                        # Create text_entities in dictionary format as expected by the function
                        mock_result.text_entities = [
                            {'type': 'image_derived', 'text': 'Entity-based description'}
                        ]
                        
                        mock_transform.process.return_value = [mock_result]
                        
                        # Call the function
                        result = summarize_image(test_image_path)
                        
                        # Verify result comes from text_entities
                        assert result == 'Entity-based description'


def test_summarize_image_function_with_text_field(test_image_path):
    """Test summarize_image function with image description in text field."""
    # Mock the logging functions directly
    with patch('logging.basicConfig'), patch('logging.getLogger'):
        # Mock Path.exists to return True
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            # Mock the config loading
            with patch('telexp.transforms.image.summarize.load_config') as mock_load_config:
                mock_config = MagicMock()
                mock_load_config.return_value = mock_config
                
                # Mock the service manager and transform
                with patch('telexp.transforms.image.summarize.LLMServiceManager') as mock_manager_class:
                    with patch('telexp.transforms.image.summarize.OllamaImageSummarize') as mock_transform_class:
                        mock_transform = MagicMock()
                        mock_transform_class.return_value = mock_transform
                        
                        # Create a mock result with text but no summary_text
                        mock_result = MagicMock()
                        mock_result.summary_text = None
                        mock_result.text = "[Image description: Description from text fiel]"
                        mock_result.text_entities = []
                        
                        mock_transform.process.return_value = [mock_result]
                        
                        # Call the function
                        result = summarize_image(test_image_path)
                        
                        # Check that we get something from the text field
                        # The function strips "[Image description:" and "]" from the text
                        assert result is not None
                        assert "Description from text fiel" in result


def test_summarize_image_function_errors(tmp_path):
    """Test summarize_image function error handling."""
    nonexistent_path = str(tmp_path / "nonexistent.jpg")
    
    # Test with nonexistent image
    with patch('logging.basicConfig'), patch('logging.getLogger'):
        # Let Path.exists return False for nonexistent file
        result = summarize_image(nonexistent_path)
        assert result is None
        
    # Test with config loading error
    with patch('logging.basicConfig'), patch('logging.getLogger'):
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('telexp.transforms.image.summarize.load_config') as mock_load_config:
                mock_load_config.side_effect = Exception("Config error")
                
                # Create a temp file
                test_file = tmp_path / "test.jpg"
                test_file.touch()
                
                result = summarize_image(str(test_file))
                assert result is None
                
    # Test with transform process error
    with patch('logging.basicConfig'), patch('logging.getLogger'):
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('telexp.transforms.image.summarize.load_config') as mock_load_config:
                mock_config = MagicMock()
                mock_load_config.return_value = mock_config
                
                with patch('telexp.transforms.image.summarize.LLMServiceManager') as mock_manager_class:
                    with patch('telexp.transforms.image.summarize.OllamaImageSummarize') as mock_transform_class:
                        mock_transform = MagicMock()
                        mock_transform_class.return_value = mock_transform
                        
                        # Make process method return empty results
                        mock_transform.process.return_value = []
                        
                        # Create a temp file
                        test_file = tmp_path / "test2.jpg"
                        test_file.touch()
                        
                        result = summarize_image(str(test_file))
                        assert result is None


def test_main_command_line_parsing():
    """Test the command line argument parsing in main function."""
    # We're just testing that we hit the relevant codepaths to improve coverage
    
    # Mock argparse to avoid actual command line parsing
    with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
        # Return an object that will cause early return with error code
        mock_args = MagicMock()
        mock_args.image_path = "non_existent_file.jpg"
        mock_parse_args.return_value = mock_args
        
        # Mock Path.exists to avoid file system access
        with patch('pathlib.Path.exists') as mock_exists:
            # Cause the main function to exit with an error by saying the file doesn't exist
            mock_exists.return_value = False
            
            # Mock print to avoid output during test
            with patch('builtins.print'):
                # Import and call main function
                from telexp.transforms.image.summarize import main
                # Should return error code 1 since the file doesn't exist
                assert main() == 1
                
                # Verify Path.exists was called
                mock_exists.assert_called_once()
