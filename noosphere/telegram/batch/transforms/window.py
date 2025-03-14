import apache_beam as beam
from typing import Dict, Any, Iterator, List
import logging
from apache_beam import typehints
import uuid
from datetime import datetime
from apache_beam.transforms.window import GlobalWindow
from apache_beam.utils.windowed_value import WindowedValue
import time

from noosphere.telegram.batch.schema import Window, WindowMetadata, RawMessage, LangchainWindow
from noosphere.telegram.batch.config import WindowBySessionConfig, WindowBySizeConfig
from noosphere.telegram.batch.transforms.message_aggregation import MessageAggregator

from noosphere.telegram.batch.logging import LoggingMixin

class BaseWindowTransform(beam.DoFn, LoggingMixin):
    """Base class for windowing transforms."""

    def __init__(self):
        """Initialize base windowing transform."""
        super().__init__()
        self._setup_logging()

    def setup(self):
        """Initialize the transform."""
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()

class WindowBySession(BaseWindowTransform):
    """Transform that groups messages into windows based on session timeout."""

    def __init__(self, config: WindowBySessionConfig):
        """Initialize with session windowing config."""
        super().__init__()
        self.config = config
        # Dictionary of conversation_id -> (buffer, last_time)
        self._windows: Dict[str, tuple[List[RawMessage], int]] = {}

    @typehints.with_input_types(RawMessage)
    @typehints.with_output_types(Window)
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a single message.
        
        Args:
            element: Raw message to process
            
        Returns:
            Iterator[Window]: Windows created from messages
        """
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()

        # Handle None or empty input
        if element is None:
            self.logger.debug("Received None element, skipping")
            return

        # Validate input message
        try:
            message = RawMessage.model_validate(element) if isinstance(element, dict) else element
        except Exception as e:
            self.logger.error("Invalid message format", exception=e)
            return
            
        # Check for valid message object
        if message is None:
            self.logger.debug("Received None message after validation, skipping")
            return

        try:
            current_time = message.date_unixtime
            conv_id = message.conversation_id
        except AttributeError as e:
            self.logger.error(f"Message missing required attributes: {e}")
            return
        
        # Create context for structured logging
        ctx = {
            "conversation_id": conv_id,
            "message_id": message.id,
            "timestamp": current_time,
            "window_type": "session"
        }
        
        # Get or create buffer for this conversation
        if conv_id not in self._windows:
            self._windows[conv_id] = ([], 0)
            self.logger.debug("Created new session buffer", context=ctx)
            
        buffer, last_time = self._windows[conv_id]
        
        # Log more detailed info at debug level
        if last_time > 0:
            time_diff = current_time - last_time
            self.logger.debug("Time between messages", context={
                **ctx,
                "time_diff": time_diff,
                "timeout": self.config.timeout_seconds,
                "buffer_size": len(buffer),
                "max_size": self.config.max_size
            })
        
        # Check if we need to emit current window
        if (current_time - last_time > self.config.timeout_seconds or 
            len(buffer) >= self.config.max_size):
            if len(buffer) >= self.config.min_size:
                window = MessageAggregator.create_window_from_messages(buffer, conv_id)
                
                window_ctx = {
                    **ctx,
                    "window_id": str(window.id),
                    "messages": len(buffer),
                    "reason": "timeout" if current_time - last_time > self.config.timeout_seconds else "max_size"
                }
                self.logger.info("Emitting session window", context=window_ctx)
                
                yield WindowedValue(
                    window.model_dump(),  # Convert to dict for beam
                    timestamp=current_time,  # Use current message time
                    windows=[GlobalWindow()]
                )
            else:
                self.logger.debug("Buffer too small for window", context={
                    **ctx,
                    "buffer_size": len(buffer),
                    "min_size": self.config.min_size
                })
            buffer = []

        # Add message to buffer
        buffer.append(message)
        self._windows[conv_id] = (buffer, current_time)
        
        # Only log at debug level to reduce verbosity
        self.logger.debug("Added message to buffer", context={
            **ctx,
            "buffer_size": len(buffer),
            "from": message.from_name or "unknown"
        })

    def finish_bundle(self):
        """Emit any remaining messages in the current window."""
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        windows_emitted = 0
        total_buffers = len(self._windows)
        
        for conv_id, (buffer, last_time) in self._windows.items():
            ctx = {
                "conversation_id": conv_id,
                "buffer_size": len(buffer),
                "min_size": self.config.min_size,
                "window_type": "session",
                "stage": "finish_bundle"
            }
            
            if len(buffer) >= self.config.min_size:
                window = MessageAggregator.create_window_from_messages(buffer, conv_id)
                
                window_ctx = {
                    **ctx,
                    "window_id": str(window.id),
                    "messages": len(buffer)
                }
                self.logger.info("Emitting final session window", context=window_ctx)
                
                yield WindowedValue(
                    window.model_dump(),  # Convert to dict for beam
                    timestamp=last_time,  # Use last message time
                    windows=[GlobalWindow()]
                )
                windows_emitted += 1
            else:
                self.logger.debug("Skipping small buffer at finish", context=ctx)
                
        self.logger.info("Session windowing complete", context={
            "total_buffers": total_buffers,
            "windows_emitted": windows_emitted
        })
        
        self._windows = {}

class WindowBySize(BaseWindowTransform):
    """Transform that creates sliding windows of fixed size."""

    def __init__(self, config: WindowBySizeConfig):
        """Initialize with size-based windowing config."""
        super().__init__()
        self.config = config
        # Dictionary of conversation_id -> buffer
        self._buffers: Dict[str, List[RawMessage]] = {}

    @typehints.with_input_types(RawMessage)
    @typehints.with_output_types(Window)
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a single message.
        
        Args:
            element: Raw message to process
            
        Returns:
            Iterator[Window]: Windows created from messages
        """
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()

        # Handle None or empty input
        if element is None:
            self.logger.debug("Received None element, skipping")
            return
            
        # Don't try to process a list
        if isinstance(element, list):
            self.logger.debug("Received list element, expected single message")
            return

        # Validate input message
        try:
            message = RawMessage.model_validate(element) if isinstance(element, dict) else element
        except Exception as e:
            self.logger.error("Invalid message format", exception=e)
            return
            
        # Check for valid message object
        if message is None:
            self.logger.debug("Received None message after validation, skipping")
            return

        try:
            conv_id = message.conversation_id
        except AttributeError as e:
            self.logger.error(f"Message missing required attributes: {e}")
            return
        
        # Create context for structured logging
        ctx = {
            "conversation_id": conv_id,
            "message_id": message.id,
            "window_type": "sliding",
            "window_size": self.config.size,
            "overlap": self.config.overlap
        }
        
        # Get or create buffer for this conversation
        if conv_id not in self._buffers:
            self._buffers[conv_id] = []
            self.logger.debug("Created new sliding window buffer", context=ctx)
            
        buffer = self._buffers[conv_id]
        
        # Add message to buffer
        buffer.append(message)
        
        # Log buffer status at debug level
        self.logger.debug("Added message to buffer", context={
            **ctx,
            "buffer_size": len(buffer),
            "from": message.from_name or "unknown"
        })
        
        # Create windows if we have enough messages
        windows_created = 0
        while len(buffer) >= self.config.size:
            # Create window from current slice
            window = MessageAggregator.create_window_from_messages(
                buffer[:self.config.size],
                conv_id
            )
            
            window_ctx = {
                **ctx,
                "window_id": str(window.id),
                "messages": self.config.size,
                "buffer_size": len(buffer)
            }
            self.logger.info("Emitting sliding window", context=window_ctx)
            
            yield WindowedValue(
                window.model_dump(),  # Convert to dict for beam
                timestamp=message.date_unixtime,  # Use current message time
                windows=[GlobalWindow()]
            )
            windows_created += 1
            
            # Move buffer forward by (size - overlap)
            stride = self.config.size - self.config.overlap
            previous_size = len(buffer)
            buffer = buffer[stride:]
            
            self.logger.debug("Sliding window forward", context={
                **ctx,
                "stride": stride,
                "previous_size": previous_size,
                "new_size": len(buffer)
            })
        
        # Update buffer
        self._buffers[conv_id] = buffer
        
        # Log summary at info level if windows were created
        if windows_created > 0:
            self.logger.info("Created sliding windows", context={
                **ctx,
                "windows_created": windows_created,
                "remaining_buffer": len(buffer)
            })

    def finish_bundle(self):
        """Emit any remaining messages as a final window if enough remain."""
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        windows_emitted = 0
        total_buffers = len(self._buffers)
        
        for conv_id, buffer in self._buffers.items():
            ctx = {
                "conversation_id": conv_id,
                "buffer_size": len(buffer),
                "window_size": self.config.size,
                "window_type": "sliding",
                "stage": "finish_bundle"
            }
            
            if len(buffer) >= self.config.size:
                window = MessageAggregator.create_window_from_messages(
                    buffer[:self.config.size],
                    conv_id
                )
                
                window_ctx = {
                    **ctx,
                    "window_id": str(window.id),
                    "messages": self.config.size
                }
                self.logger.info("Emitting final sliding window", context=window_ctx)
                
                yield WindowedValue(
                    window.model_dump(),  # Convert to dict for beam
                    timestamp=buffer[-1].date_unixtime,  # Use last message time
                    windows=[GlobalWindow()]
                )
                windows_emitted += 1
            else:
                self.logger.debug("Skipping small buffer at finish", context=ctx)
                
        self.logger.info("Sliding windowing complete", context={
            "total_buffers": total_buffers,
            "windows_emitted": windows_emitted
        })
            
        self._buffers = {}
        
        
class WindowToLangchainConverter(beam.DoFn, LoggingMixin):
    """Converts Window objects to LangchainWindow format for storage and retrieval."""
    
    def __init__(self):
        """Initialize the converter."""
        super().__init__()
        self._setup_logging()
        
    def setup(self):
        """Set up the component."""
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
    
    @typehints.with_input_types(Window)
    @typehints.with_output_types(LangchainWindow)
    def process(self, window: Window) -> Iterator[LangchainWindow]:
        """Convert a Window object to LangchainWindow format.
        
        Args:
            window: Window object to convert
            
        Returns:
            Iterator with converted LangchainWindow objects
        """
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        # Extract data from window
        window_id = str(window.id)
        summary_text = window.summary_text or ""
        conversation_id = window.conversation_id
        conversation_type = window.conversation_type or ""
        vector = window.vector  # Keep the vector for storage
        
        # Create logging context
        ctx = {
            "window_id": window_id,
            "conversation_id": conversation_id,
            "conversation_type": conversation_type,
            "has_vector": vector is not None
        }
        
        # Get metadata information
        metadata = window.window_metadata
        
        # Extract display name if available or fallback to conversation_id
        from_names = getattr(metadata, 'from_names', {}) or {}
        display_name = next(iter(from_names.values()), conversation_id) if from_names else conversation_id
        
        # Create file_id from conversation_type and conversation_id
        file_id = f"{conversation_type}/{conversation_id}"
        
        # Add event count to context for logging
        event_count = len(metadata.event_ids) if hasattr(metadata, 'event_ids') else 0
        ctx["event_count"] = event_count
        
        # Build formatted window object
        langchain_window = LangchainWindow(
            id=window_id,
            content=summary_text,
            vector=vector,  # Include vector for storage
            metadata={
                "source": "blob",
                "blobType": "text/plain",
                "loc": {
                    "lines": {
                        "from": 1,
                        "to": event_count
                    }
                },
                "file_id": file_id,
                "pubkey": display_name,
                # Keep other useful metadata
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "date_range": getattr(metadata, 'date_range', []),
                "from_ids": getattr(metadata, 'from_ids', {}),
            }
        )
        
        self.logger.info("Converted window to LangchainWindow format", context=ctx)
        yield langchain_window 