"""Text summarization transformer."""
import apache_beam as beam
from typing import Dict, Any, Iterator, Optional
import httpx
from apache_beam import typehints
import time

from noosphere.telegram.batch.config import TextSummarizerConfig
from noosphere.telegram.batch.schema import Window
from noosphere.telegram.batch.services.llm import LLMServiceManager, LLMService, LLMModel
from noosphere.telegram.batch.logging import LoggingMixin, get_logger, log_performance

class OllamaSummarize(beam.DoFn, LoggingMixin):
    """A transform that summarizes text using Ollama."""

    def __init__(self, config: TextSummarizerConfig, service_manager: Optional[LLMServiceManager] = None):
        """Initialize the transform.

        Args:
            config: Configuration for the text summarizer.
            service_manager: Optional service manager to use.
        """
        super().__init__()
        self.config = config
        self._service_manager = service_manager
        self._setup_called = False
        # Initialize logging
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

    @typehints.with_input_types(Window)
    @typehints.with_output_types(Window)
    @log_performance(threshold_ms=500, level="WARNING")
    def process(self, element: Window) -> Iterator[Window]:
        """Process a single window record.
        
        Args:
            element: Window object to process
            
        Returns:
            Iterator[Window]: The window with summary field added
        """
        self.setup()
        # Create context for structured logging
        window_ctx = {
            "window_id": str(element.id),
            "conversation_id": element.conversation_id
        }
        
        # Element is already a Window
        input_field = self.config.input_field
        text = getattr(element, input_field, None)
        
        if not text:
            self.logger.warning(f"No text found for summarization", context={
                "field": input_field, 
                **window_ctx
            })
            yield element
            return
        
        # Only log text length at debug level to avoid verbose output
        self.logger.debug(f"Processing window text", context={
            "length": len(text),
            **window_ctx
        })
        
        # Check if window has conversation data with images
        has_images = False
        if "![" in text and "](" in text:
            has_images = True
            self.logger.debug(f"Window contains image references", context=window_ctx)

        for attempt in range(self.config.retries):
            try:
                # Generate the summary
                self.logger.info(f"Generating text summary", context={
                    "service": self.config.llm_service,
                    "model": self.config.model,
                    "attempt": attempt + 1,
                    **window_ctx
                })
                
                summary = self._service_manager.generate(
                    service_name=self.config.llm_service,
                    model_name=self.config.model,
                    input_text=text,
                    timeout=float(self.config.timeout)
                )
                
                if summary:
                    # Use a truncated version of summary for logging
                    summary_preview = summary[:30] + ("..." if len(summary) > 30 else "")
                    self.logger.info(f"Received window summary: \"{summary_preview}\"", context={
                        "summary_length": len(summary),
                        **window_ctx
                    })
                    
                    # Add summary to the element
                    setattr(element, self.config.output_field, summary)
                    self.logger.debug(f"Window updated with summary", context=window_ctx)
                    
                    yield element
                    return
                else:
                    self.logger.warning(f"No summary returned from LLM", context={
                        "attempt": f"{attempt + 1}/{self.config.retries}",
                        **window_ctx
                    })
            except Exception as e:
                # Add more detailed error context
                error_ctx = {
                    **window_ctx,
                    "attempt": attempt + 1,
                    "max_attempts": self.config.retries,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "service": self.config.llm_service,
                    "model": self.config.model,
                    "input_field": self.config.input_field,
                    "timeout": self.config.timeout,
                    "text_length": len(text) if text else 0
                }
                
                # Include text preview for better context
                if text and len(text) > 0:
                    error_ctx["text_preview"] = (text[:50] + "...") if len(text) > 50 else text
                
                self.logger.error(
                    f"Failed to generate summary for window {window_ctx['window_id']}: {type(e).__name__}: {str(e)}", 
                    context=error_ctx, 
                    exception=e
                )
                
            # Retry with delay if we have attempts left
            if attempt < self.config.retries - 1:
                self.logger.debug(f"Retrying in {self.config.retry_delay} seconds...", context=window_ctx)
                time.sleep(self.config.retry_delay)
        
        self.logger.warning(
            f"Exhausted all {self.config.retries} attempts to generate summary for window", 
            context={
                **window_ctx,
                "service": self.config.llm_service,
                "model": self.config.model,
                "text_length": len(text) if text else 0
            }
        )
        yield element