"""Image summarization transformer."""
import apache_beam as beam
from typing import Dict, Any, Iterator, Optional
import httpx
from apache_beam import typehints
import time
import base64
import argparse
import os
import json
from pathlib import Path

from noosphere.telegram.batch.config import ImageSummarizerConfig, load_config, Config
from noosphere.telegram.batch.schema import RawMessage
from noosphere.telegram.batch.services.llm import LLMServiceManager, LLMService, LLMModel
from noosphere.telegram.batch.logging import LoggingMixin, get_logger, setup_logging, log_performance

class OllamaImageSummarize(beam.DoFn, LoggingMixin):
    """A transform that summarizes images using Ollama."""

    def __init__(self, config: ImageSummarizerConfig, service_manager: Optional[LLMServiceManager] = None):
        """Initialize the transform.

        Args:
            config: Configuration for the image summarizer.
            service_manager: Optional service manager to use.
        """
        super().__init__()
        self.config = config
        self._service_manager = service_manager
        self._setup_called = False
        # Initialize logging for this instance
        self._setup_logging()
        
    def __getstate__(self):
        """Control which attributes are pickled."""
        state = super().__getstate__()
        # Reset setup flag for unpickling
        state['_setup_called'] = False
        return state

    def setup(self):
        """Set up the transform."""
        if self._setup_called:
            return

        # Logger is already set up via LoggingMixin
        
        if not self._service_manager:
            service = LLMService(
                name=self.config.llm_service,
                type="ollama",
                url="http://localhost:11434",
                models=[
                    LLMModel(
                        name=self.config.model,
                        model=self.config.model,
                        prompt=self.config.prompt
                    )
                ]
            )
            self._service_manager = LLMServiceManager([service])
            self._service_manager.setup()
        
        self._setup_called = True

    def teardown(self):
        """Clean up resources."""
        if self._service_manager:
            self._service_manager.teardown()
        self._setup_called = False

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string or None if failed
        """
        try:
            # Handle different formats of image_path
            if isinstance(image_path, dict):
                image_path = image_path.get('file')
            elif hasattr(image_path, 'file'):
                image_path = image_path.file

            if not image_path:
                self.logger.error("No image path provided")
                return None
                
            # Log the image path at debug level
            self.logger.debug(f"Attempting to encode image at path: {image_path}")

            # Ensure the path exists
            if not os.path.exists(image_path):
                # Log missing images as debug instead of error since this is normal
                self.logger.debug(f"Image path does not exist: {image_path}")
                return None
                
            # Read and encode the image
            with open(image_path, 'rb') as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                self.logger.debug(f"Successfully encoded image, length: {len(encoded)}")
                return encoded
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}", exception=e)
            return None

    @typehints.with_input_types(RawMessage)
    @typehints.with_output_types(RawMessage)
    @log_performance(threshold_ms=500, level="WARNING")
    def process(self, element: Dict[str, Any]) -> Iterator[RawMessage]:
        """Process a single message record."""
        self.setup()
        
        # If element is already a RawMessage, use it; otherwise, convert from dict
        if isinstance(element, RawMessage):
            message = element
            self.logger.debug(f"Processing RawMessage: {message.id}")
        else:
            # It's a dictionary
            self.logger.debug(f"Processing dict message: {element.get('id', 'unknown')}")
            message = RawMessage(**element)
        
        # Create context for structured logging
        msg_ctx = {
            "message_id": message.id,
            "type": message.type,
            "from": message.from_name or "unknown"
        }
            
        # Get image path from photo field - at DEBUG level only
        if not message.photo:
            # Log missing photos as debug since this is normal
            self.logger.debug("No image in message", context=msg_ctx)
            yield message
            return
            
        self.logger.debug("Found image in message", context=msg_ctx)

        # Make sure we have the actual path
        photo_path = None
        if isinstance(message.photo, dict) and 'file' in message.photo:
            photo_path = message.photo['file']
        elif hasattr(message.photo, 'file'):
            photo_path = message.photo.file
        else:
            photo_path = message.photo
            
        self.logger.debug(f"Using image path: {photo_path}", context=msg_ctx)
        
        # Verify the image exists
        if not os.path.exists(photo_path):
            # Use debug level for missing images since this is normal
            self.logger.debug(f"Image file does not exist", context={"path": photo_path, **msg_ctx})
            yield message
            return

        # Encode image to base64
        image_data = self._encode_image(photo_path)
        if not image_data:
            self.logger.warning(f"Failed to encode image", context=msg_ctx)
            yield message
            return
            
        self.logger.debug(f"Successfully encoded image (length: {len(image_data)})", context=msg_ctx)

        for attempt in range(self.config.retries):
            try:
                self.logger.info(
                    f"Sending image to LLM",
                    context={
                        "service": self.config.llm_service, 
                        "model": self.config.model,
                        "attempt": attempt + 1,
                        **msg_ctx
                    }
                )
                
                summary = self._service_manager.generate(
                    service_name=self.config.llm_service,
                    model_name=self.config.model,
                    input_text=image_data,  # Pass base64 encoded image
                    timeout=float(self.config.timeout),
                    is_image=True  # Indicate this is an image
                )
                
                if summary:
                    # Use a truncated version of summary for logging
                    summary_preview = summary[:30] + ("..." if len(summary) > 30 else "")
                    self.logger.info(
                        f"Received image summary: \"{summary_preview}\"", 
                        context={"summary_length": len(summary), **msg_ctx}
                    )
                    
                    # Create new message with summary as text entity
                    message_dict = message.model_dump()
                    
                    # Add summary to text_entities
                    text_entities = message_dict.get('text_entities', [])
                    if text_entities is None:
                        text_entities = []
                    text_entities.append({
                        'type': 'image_derived',
                        'text': summary
                    })
                    message_dict['text_entities'] = text_entities
                    
                    # Add summary to summary_text field
                    message_dict['summary_text'] = summary
                    
                    # Also add to text field for compatibility
                    if 'text' not in message_dict or not message_dict['text']:
                        message_dict['text'] = f"[Image description: {summary}]"
                    
                    # Create the updated message
                    updated_message = RawMessage(**message_dict)
                    self.logger.debug(f"Added image summary to message", context=msg_ctx)
                        
                    yield updated_message
                    return
                else:
                    self.logger.warning(
                        f"No summary returned from LLM", 
                        context={"attempt": f"{attempt + 1}/{self.config.retries}", **msg_ctx}
                    )
            except Exception as e:
                # Add more detailed error context
                error_ctx = {
                    **msg_ctx,
                    "attempt": attempt + 1,
                    "max_attempts": self.config.retries,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "service": self.config.llm_service,
                    "model": self.config.model,
                    "input_field": self.config.input_field,
                    "timeout": self.config.timeout,
                    "image_path": photo_path
                }
                
                self.logger.error(
                    f"Failed to generate image summary for message {msg_ctx['message_id']}: {type(e).__name__}: {str(e)}", 
                    context=error_ctx,
                    exception=e
                )
                
            # Retry with delay if we have attempts left
            if attempt < self.config.retries - 1:
                self.logger.debug(f"Retrying in {self.config.retry_delay} seconds...", context=msg_ctx)
                time.sleep(self.config.retry_delay)

        self.logger.warning(
            f"Exhausted all {self.config.retries} attempts to generate image summary", 
            context={
                **msg_ctx,
                "service": self.config.llm_service,
                "model": self.config.model,
                "image_path": photo_path if photo_path else "unknown"
            }
        )
        yield message


def summarize_image(image_path, config=None, verbose=False):
    """Summarize a single image using the OllamaImageSummarize transform.
    
    Args:
        image_path: Path to the image file
        config: Optional ImageSummarizerConfig or path to config file
        verbose: Whether to print verbose output
        
    Returns:
        Summary text or None if failed
    """
    # Setup logging
    import logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("image_summarizer")
    
    # Verify image exists
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image file does not exist: {image_path}")
        return None
        
    logger.info(f"Processing image: {image_path} (exists: {image_path.exists()})")
    
    # Load or use provided config
    if config is None:
        try:
            # First check for conf.yaml in current directory
            current_config_path = Path.cwd() / 'conf.yaml'
            if current_config_path.exists():
                logger.info(f"Using config from {current_config_path}")
                full_config = load_config(current_config_path)
            else:
                # Try to find config in parent directories
                logger.info("Looking for conf.yaml...")
                full_config = load_config()
            logger.info(f"Config loaded successfully")
            config = full_config.image_summarizer
            logger.info(f"Image summarizer config: Service={config.llm_service}, Model={config.model}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return None
    elif isinstance(config, (str, Path)):
        try:
            logger.info(f"Using config from {config}")
            full_config = load_config(config)
            config = full_config.image_summarizer
        except Exception as e:
            logger.error(f"Failed to load config from {config}: {e}")
            return None
            
    # Create fake RawMessage with photo
    message = RawMessage(
        id="test",
        type="message",
        date="2024-01-01T00:00:00",
        date_unixtime=1704067200,
        conversation_id="test",
        conversation_type="test",
        from_id="test",
        from_name="Test User",
        photo=str(image_path),  # Use string path for compatibility
        source={
            "file_path": "test",
            "file_name": "test",
            "file_extension": str(image_path).split('.')[-1],
            "file_mimetype": "image/jpeg"
        }
    )
    
    # Create the transform with service manager from full config
    if 'full_config' in locals() and hasattr(full_config, 'llm_services'):
        services = [s.name for s in full_config.llm_services]
        logger.info(f"Using LLM services from config: {services}")
        service_manager = LLMServiceManager(full_config.llm_services)
        transform = OllamaImageSummarize(config, service_manager)
    else:
        # Fallback to default behavior
        logger.info(f"Using default service manager")
        transform = OllamaImageSummarize(config)
    
    logger.info("Setting up image summarizer transform")
    transform.setup()
    
    # Process the message
    logger.info("Processing image message")
    result = None
    for output in transform.process(message):
        # First check summary_text field (preferred)
        if hasattr(output, 'summary_text') and output.summary_text:
            logger.info(f"Found summary_text field: {output.summary_text[:100]}...")
            result = output.summary_text
            break
            
        # Then check text field
        if hasattr(output, 'text') and output.text:
            # Handle the case where we're looking for the image description in the text field
            if output.text.startswith("[Image description:"):
                logger.info(f"Found image description in text field: {output.text}")
                result = output.text.strip("[Image description:]").strip(" ]")
            else:
                logger.info(f"Found text field: {output.text}")
                result = output.text
            break
            
        # Check text_entities as a last resort
        if hasattr(output, 'text_entities') and output.text_entities:
            for entity in output.text_entities:
                if isinstance(entity, dict) and entity.get('type') in ('image_description', 'image_derived'):
                    logger.info(f"Found description in text_entities: {entity.get('text', '')[:100]}...")
                    result = entity.get('text', '')
                    break
            
    # Log the result
    if result:
        logger.info(f"Generated summary: {result[:100]}...")
    else:
        logger.error("Failed to generate summary")
        
    return result


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Summarize an image using Ollama")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", "-o", help="Path to write output JSON")
    args = parser.parse_args()
    
    # Make sure the image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
        
    # Summarize the image
    summary = summarize_image(image_path, args.config, args.verbose)
    
    # Print the result
    if summary:
        print(f"\nSummary: {summary}")
        
        # Write output if requested
        if args.output:
            result = {
                "image_path": str(image_path),
                "summary": summary
            }
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote output to {args.output}")
        return 0
    else:
        print("Failed to generate summary")
        return 1


if __name__ == "__main__":
    exit(main())
