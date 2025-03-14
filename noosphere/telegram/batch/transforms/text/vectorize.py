"""Text vectorization transformer."""
import apache_beam as beam
from typing import Dict, Any, Iterator
import httpx
import json
from apache_beam import typehints

from noosphere.telegram.batch.config import TextVectorizerConfig
from noosphere.telegram.batch.schema import Window
from noosphere.telegram.batch.services.embedding import EmbeddingServiceManager, EmbeddingService, EmbeddingModel
from noosphere.telegram.batch.logging import LoggingMixin, log_performance

class OllamaVectorize(beam.DoFn, LoggingMixin):
    """Beam DoFn that adds embeddings to windows using Ollama."""

    def __init__(self, config: TextVectorizerConfig, service_manager: EmbeddingServiceManager):
        """Initialize with config from text_vectorizer section."""
        super().__init__()
        self.config = config
        self._service_manager = service_manager
        self._setup_called = False
        # Don't set up logger in __init__, wait for setup() to be called
        
    def __getstate__(self):
        """Control which attributes are pickled."""
        state = super().__getstate__()
        # Reset setup flag for unpickling
        state['_setup_called'] = False
        return state

    def setup(self):
        """Initialize resources."""
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
        self._setup_called = True

    def teardown(self):
        """Clean up resources."""
        self._setup_called = False

    @typehints.with_input_types(Window)
    @typehints.with_output_types(Window)
    @log_performance(threshold_ms=300, level="WARNING")
    def process(self, element: Window) -> Iterator[Window]:
        """Process a single window record.
        
        Args:
            element: Window to process
            
        Returns:
            Iterator[Window]: The window with vector field updated with embeddings
            
        Raises:
            ValueError: If the received vector size doesn't match the expected size
        """
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()

        # Create context dictionary for structured logging
        ctx = {
            "window_id": str(element.id),
            "conversation_id": element.conversation_id,
            "input_field": self.config.input_field,
            "model": self.config.model
        }
        
        self.logger.info("Processing window", context=ctx)
        
        # Get text to vectorize - directly check the field specified in config
        field_name = self.config.input_field
        text = getattr(element, field_name, None)
        
        # Log field availability only at debug level
        debug_ctx = ctx.copy()
        for attr_name in ['concatenated_text', 'summary_text', 'vector']:
            value = getattr(element, attr_name, None)
            if value:
                preview = value[:50] if isinstance(value, str) else str(value)[:50]  
                total_len = len(value) if isinstance(value, str) else len(str(value))
                debug_ctx[f"has_{attr_name}"] = True
                debug_ctx[f"{attr_name}_length"] = total_len
            else:
                debug_ctx[f"has_{attr_name}"] = False
        
        self.logger.debug("Window field inspection", context=debug_ctx)
        
        if not text:
            self.logger.warning("No text found for vectorization", context=ctx)
            yield element
            return

        # Only log text preview at debug level to reduce verbosity
        if len(text) > 100:
            debug_ctx["text_preview"] = text[:100] + "..."
        else:
            debug_ctx["text_preview"] = text
            
        self.logger.debug("Vectorizing text", context=debug_ctx)

        # Get embeddings from service
        last_error = None
        for attempt in range(self.config.retries):
            try:
                attempt_ctx = {**ctx, "attempt": attempt + 1, "max_attempts": self.config.retries}
                self.logger.debug("Embedding attempt", context=attempt_ctx)
                
                # Get embeddings using service manager
                embedding = self._service_manager.embed(
                    service_name=self.config.embedding_service,
                    model_name=self.config.model,
                    text=text,
                    timeout=float(self.config.timeout)
                )
                
                if embedding is None:
                    raise ValueError("Failed to get embeddings from service")
                
                # Validate vector size
                if len(embedding) != self.config.vector_size:
                    error_ctx = {
                        **ctx, 
                        "received_size": len(embedding),
                        "expected_size": self.config.vector_size
                    }
                    self.logger.error("Vector size mismatch", context=error_ctx)
                    yield element  # Return unchanged window on wrong size
                    return
                    
                # Create a new window with updated vector
                updated_element = Window(
                    **{
                        **element.model_dump(),
                        'vector': embedding
                    }
                )
                
                # Log vector statistics
                vector_len = len(updated_element.vector)
                vector_stats = {
                    **ctx,
                    "vector_length": vector_len,
                    "vector_min": min(updated_element.vector) if updated_element.vector else 0.0,
                    "vector_max": max(updated_element.vector) if updated_element.vector else 0.0
                }
                
                # For debugging purposes, add a vector preview regardless of level
                # This simplifies our approach and avoids level checking issues
                vector_preview = str(updated_element.vector[:3])
                vector_stats["vector_preview"] = vector_preview
                
                self.logger.info("Generated vector", context=vector_stats)
                
                # Set element to the updated version with the vector
                element = updated_element
                break
                
            except Exception as e:
                # Add more details to error context
                error_ctx = {
                    **ctx, 
                    "attempt": attempt + 1, 
                    "max_attempts": self.config.retries,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "text_length": len(text) if text else 0,
                    "service": self.config.embedding_service,
                    "model": self.config.model,
                    "input_field": self.config.input_field,
                    "timeout": self.config.timeout
                }
                
                # Include a text preview in error logs for better context
                if text and len(text) > 0:
                    error_ctx["text_preview"] = (text[:50] + "...") if len(text) > 50 else text
                
                # Show specific error type and message in the log text
                self.logger.error(f"Failed to get embeddings for window {ctx['window_id']}: {type(e).__name__}: {str(e)}", 
                                 context=error_ctx, exception=e)
                
                last_error = e
                if attempt == self.config.retries - 1:  # Last attempt
                    self.logger.warning(f"Exhausted all {self.config.retries} attempts to generate embeddings",
                                      context=ctx)
                    yield element  # Return unchanged window on error
                    return
        
        yield element  # Return window with updated vector