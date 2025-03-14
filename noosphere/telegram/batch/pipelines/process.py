"""Main processing pipeline for Telegram export files.

This module implements the full processing pipeline:
1. Load messages from JSONL files
2. Process images into text descriptions
3. Create windows from messages
4. Generate summaries
5. Generate vectors
6. Store vectors in Qdrant
"""
import apache_beam as beam
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import uuid

from noosphere.telegram.batch.config import Config, load_config
from noosphere.telegram.batch.schema import Window, WindowMetadata, RawMessage, LangchainWindow
from noosphere.telegram.batch.services.llm import LLMServiceManager
from noosphere.telegram.batch.services.embedding import EmbeddingServiceManager
from noosphere.telegram.batch.transforms.image.summarize import OllamaImageSummarize
from noosphere.telegram.batch.transforms.text.summarize import OllamaSummarize
from noosphere.telegram.batch.transforms.text.vectorize import OllamaVectorize
from noosphere.telegram.batch.transforms.storage.qdrant import QdrantVectorStore
from noosphere.telegram.batch.transforms.window import WindowToLangchainConverter
from noosphere.telegram.batch.transforms.message_aggregation import MessageAggregator
from noosphere.telegram.batch.logging import setup_logging, get_logger


class TelegramExportProcessor:
    """Process Telegram export files and store results in a vector database."""

    def __init__(self, config: Optional[Union[Config, str, Path]] = None):
        """Initialize the processor with a configuration.
        
        Args:
            config: Either a Config object, path to a config file, or None to use default config
        """
        # Load configuration first
        if config is None:
            self.config = load_config()
        elif isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config
        
        # Set up logging based on config
        log_config = self.config.logging
        setup_logging(
            level=log_config.level,
            distributed=log_config.distributed,
            log_file=log_config.log_file,
            json_logs=log_config.json_logs,
            intercept_std_logging=log_config.intercept_std_logging,
            rotation=log_config.rotation,
            retention=log_config.retention,
            compression=log_config.compression,
            enqueue=log_config.enqueue,
            log_startup=log_config.log_startup
        )
        
        # Apply module-specific log levels from config
        if log_config.module_levels:
            import logging
            for module, level in log_config.module_levels.items():
                logging.getLogger(module).setLevel(level)
        
        self.logger = get_logger(__name__)
        
        # Initialize service managers
        self.llm_manager = LLMServiceManager(self.config.llm_services)
        self.embedding_manager = EmbeddingServiceManager(self.config.embedding_services)
        
        # Initialize storage for processed window IDs
        self.stored_ids = []

    def load_messages_from_jsonl(self, input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load messages from JSONL files in the input directory.
        
        Args:
            input_dir: Directory containing conversation directories
            
        Returns:
            List of message dictionaries
        
        The directory structure is expected to be:
        <input_dir>/<conversation_type>/<conversation_id>/messages.jsonl
        <input_dir>/<conversation_type>/<conversation_id>/attachments/<attachment_type>/filename
        """
        input_dir = Path(input_dir)
        self.logger.info(f"Loading messages from {input_dir}")
        messages = []
        
        # Count statistics for logging
        conv_count = 0
        file_count = 0
        
        # Walk through conversation type directories
        for conv_type_path in input_dir.glob("*"):
            if not conv_type_path.is_dir():
                continue
                
            conv_type = conv_type_path.name
            
            # Walk through conversation ID directories
            for conv_id_path in conv_type_path.glob("*"):
                if not conv_id_path.is_dir():
                    continue
                    
                conv_id = conv_id_path.name
                conv_count += 1
                messages_file = conv_id_path / "messages.jsonl"
                
                if not messages_file.exists():
                    continue
                
                file_count += 1
                self.logger.info(f"Reading messages", context={"file": str(messages_file)})
                msg_count = 0
                with open(messages_file, "r") as f:
                    for line in f:
                        message = json.loads(line)
                        msg_count += 1
                        
                        # Add conversation metadata from directory structure
                        message["conversation_id"] = conv_id
                        message["conversation_type"] = conv_type
                        
                        # Handle "from" field mapping to "from_name" in RawMessage schema
                        if "from" in message and "from_name" not in message:
                            message["from_name"] = message["from"]
                            self.logger.debug(f"Mapped 'from' to 'from_name'", 
                                             context={"value": message["from_name"]})
                        
                        # Handle photo paths - convert relative to absolute
                        if isinstance(message.get("photo"), str):
                            # Convert string path to photo object format
                            rel_path = message["photo"]
                            abs_path = str(conv_id_path / rel_path)
                            message["photo"] = {"file": abs_path}
                        elif isinstance(message.get("photo"), dict) and "file" in message["photo"]:
                            # Update existing photo path
                            rel_path = message["photo"]["file"]
                            abs_path = str(conv_id_path / rel_path)
                            message["photo"]["file"] = abs_path
                            
                        # Add source field with file information
                        message["source"] = {
                            "file_path": str(messages_file),
                            "file_name": messages_file.name,
                            "file_extension": ".jsonl",
                            "file_mimetype": "application/json",
                            "file_offset": 0,  # We don't track exact offset
                            "file_line_number": 1  # We don't track exact line, but use 1 to pass validation
                        }
                        
                        messages.append(message)
                
                self.logger.info(f"Read {msg_count} messages", 
                                context={"conversation": conv_id, "type": conv_type})
        
        self.logger.info(f"Loaded messages", 
                       context={"count": len(messages), 
                                "conversations": conv_count, 
                                "files": file_count})
        return messages

    def process(self, input_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> List[str]:
        """Process Telegram export files and store results.
        
        Args:
            input_dir: Directory containing conversation directories
            output_dir: Optional directory to write output files. If provided,
                       will write fully processed Window objects to output_dir/windows.jsonl
                       
        Returns:
            List[str]: IDs of the stored windows. Returns an empty list if no windows were processed
                      or if no windows had valid vectors.
        """
        input_dir = Path(input_dir)
        output_path = None
        stored_ids = []  # Initialize empty list to store window IDs
        
        if output_dir:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / "windows" 
        
        # Load messages
        messages = self.load_messages_from_jsonl(input_dir)
        
        # Show pipeline configuration
        self.logger.info("Pipeline configuration", context={
            "image_model": self.config.image_summarizer.model,
            "text_model": self.config.text_summarizer.model,
            "vector_model": self.config.text_vectorizer.model,
            "qdrant_collection": self.config.text_vector_storage.collection,
            "qdrant_batch_size": self.config.text_vector_storage.batch_size
        })
        
        # Convert to RawMessage objects
        raw_messages = []
        for message in messages:
            # Create RawMessage and convert to dict
            raw_message = RawMessage(**message)
            raw_messages.append(raw_message.model_dump())
        
        # Setup pipeline config:
        # Use default collection for text vectors
        text_vector_config = self.config.text_vector_storage
        
        # Run the pipeline
        with beam.Pipeline() as pipeline:
            # Create PCollection from raw messages
            p_messages = (
                pipeline
                | "Create Messages" >> beam.Create(raw_messages)
            )
            
            # Process images into text descriptions
            with_image_descriptions = (
                p_messages
                | "Process Images" >> beam.ParDo(OllamaImageSummarize(
                    self.config.image_summarizer, self.llm_manager))
            )
            
            # Convert messages to windows
            message_windows = (
                with_image_descriptions
                | "Create Message Windows" >> beam.Map(self._create_message_window)
            )
            
            # Group messages by conversation
            grouped_messages = (
                message_windows
                | "Group by Conversation" >> beam.GroupBy(lambda x: x.conversation_id)
            )
            
            # Create windows using the window_by_size configuration
            from noosphere.telegram.batch.transforms.window import WindowBySize
            size_windows = (
                grouped_messages
                | "Create Sliding Windows" >> beam.FlatMapTuple(
                    lambda conv_id, msgs: self._create_sliding_windows(
                        conv_id, 
                        list(msgs), 
                        self.config.window_by_size
                    )
                )
            )
            
            # Create windows using the window_by_session configuration
            from noosphere.telegram.batch.transforms.window import WindowBySession
            session_windows = (
                grouped_messages
                | "Create Session Windows" >> beam.FlatMapTuple(
                    lambda conv_id, msgs: self._create_session_windows(
                        conv_id, 
                        list(msgs), 
                        self.config.window_by_session
                    )
                )
            )
            
            # Combine windows from both algorithms
            windows = (
                (size_windows, session_windows)
                | "Combine Windows" >> beam.Flatten()
            )
            
            # Filter windows with no text content before further processing
            windows_with_content = (
                windows
                | "Filter Empty Windows" >> beam.Filter(
                    lambda window: window.has_text_content() if hasattr(window, 'has_text_content') 
                    else Window.model_validate(window).has_text_content()
                )
            )
            
            # Generate summaries only for windows with text
            with_summaries = (
                windows_with_content
                | "Generate Summaries" >> beam.ParDo(OllamaSummarize(
                    self.config.text_summarizer, self.llm_manager))
                | "Convert to Window" >> beam.Map(
                    lambda x: Window(**x) if isinstance(x, dict) else x)
            )
            
            # Generate vectors
            with_vectors = (
                with_summaries
                | "Generate Vectors" >> beam.ParDo(OllamaVectorize(
                    self.config.text_vectorizer, self.embedding_manager))
            )
            
            # Filter windows with invalid vectors before storage
            with_valid_vectors = (
                with_vectors
                | "Filter Invalid Vectors" >> beam.Filter(
                    lambda window: window.has_valid_vector() if hasattr(window, 'has_valid_vector')
                    else Window.model_validate(window).has_valid_vector()
                )
            )
            
            # Convert windows to LangchainWindow format
            langchain_windows = (
                with_valid_vectors
                | "Convert to LangchainWindow" >> beam.ParDo(WindowToLangchainConverter())
            )
            
            # Store only windows with valid vectors in Qdrant
            # Clear stored IDs list before running
            self.stored_ids = []
            
            # Create a function that will collect IDs within the pipeline
            # Using instance variable to collect results
            def collect_id(window_id):
                self.stored_ids.append(window_id)
                return window_id
            
            # Use the function in the pipeline
            _ = (
                langchain_windows
                | "Store Vectors" >> beam.ParDo(QdrantVectorStore(text_vector_config))
                | "Collect IDs" >> beam.Map(collect_id)
            )
            
            # Write windows to output file if path provided
            if output_path:
                # Custom JSON encoder to handle UUIDs
                class WindowJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, uuid.UUID):
                            return str(obj)
                        # Handle vectors specially - convert to list of floats
                        if isinstance(obj, list) and obj and isinstance(obj[0], float):
                            return obj
                        return json.JSONEncoder.default(self, obj)
                
                # Write complete windows (use with_valid_vectors to ensure we only store valid data)
                _ = (
                    with_valid_vectors
                    | "Serialize Windows" >> beam.Map(
                        lambda window: json.dumps(
                            window.model_dump(),
                            cls=WindowJSONEncoder
                        )
                    )
                    | "Write Windows" >> beam.io.WriteToText(
                        file_path_prefix=str(output_path),
                        file_name_suffix=".jsonl",
                        shard_name_template=""  # No sharding
                    )
                )
        
        # Log completion summary
        self.logger.info("Processing completed", context={
            "windows_stored": len(self.stored_ids),
            "output_path": str(output_dir) if output_dir else None
        })
                
        # Return the list of stored window IDs collected during pipeline execution
        return self.stored_ids

    def _create_message_window(self, message: RawMessage) -> Window:
        """Create a window for a single message.
        
        Args:
            message: RawMessage object
            
        Returns:
            Window object
        """
        # Create context for structured logging
        msg_ctx = {
            "message_id": message.id,
            "conversation_id": message.conversation_id,
            "from": message.from_name or "unknown"
        }
        
        # Only log presence of summary_text at debug level
        if hasattr(message, 'summary_text') and message.summary_text:
            self.logger.debug(f"Message has summary_text", context=msg_ctx)
        
        # Generate deterministic UUID based on conversation_id and timestamp
        window_id = Window.generate_deterministic_uuid(
            conversation_id=message.conversation_id,
            min_timestamp=int(message.date_unixtime),
            max_timestamp=int(message.date_unixtime)
        )
        
        # Create from_names mapping if from_name is available
        from_names = {}
        if message.from_id and message.from_name:
            from_names[message.from_id] = message.from_name
        
        window_metadata = WindowMetadata(
            id=window_id,
            conversation_id=message.conversation_id,
            conversation_type=message.conversation_type,
            date_range=[message.date, message.date],
            unixtime_range=[int(message.date_unixtime), int(message.date_unixtime)],
            from_ids={message.from_id if message.from_id else "unknown": 1},
            from_names=from_names,  # Add the from_names mapping
            event_ids=[str(message.id)]
        )
        
        # Extract text content from message
        message_text = message.get_text_content()
        
        # Check for summary_text (this should be present for image messages)
        summary_text = getattr(message, 'summary_text', None)
        
        # Create the Window object
        window = Window(
            id=window_id,
            conversation_id=message.conversation_id,
            conversation_type=message.conversation_type,
            concatenated_text=message_text if message_text else None,
            image_path=message.photo.file if message.photo else None,
            summary_text=summary_text,  # Include the summary_text field for images
            window_metadata=window_metadata
        )
        
        # Only log detailed information at debug level
        has_image = hasattr(window, 'image_path') and window.image_path
        has_summary = hasattr(window, 'summary_text') and window.summary_text
        
        self.logger.debug(f"Created message window", context={
            "window_id": str(window.id),
            "has_image": has_image,
            "has_summary": has_summary,
            **msg_ctx
        })
            
        return window

    def _create_session_windows(self, conv_id: str, messages: List[Window], config) -> List[Window]:
        """Create session-based windows from messages.
        
        Args:
            conv_id: Conversation ID
            messages: List of Window objects (message-level windows)
            config: WindowBySessionConfig with timeout, min_size, and max_size parameters
            
        Returns:
            List of Window objects, each representing a session window
        """
        # Convert messages iterator to list if needed
        messages = list(messages)
        ctx = {
            "conversation_id": conv_id,
            "message_count": len(messages),
            "window_type": "session"
        }
        
        self.logger.debug(f"Creating session windows", context={
            **ctx,
            "timeout": config.timeout_seconds,
            "min_size": config.min_size,
            "max_size": config.max_size
        })
        
        # If there are fewer messages than the minimum size, just create one window
        if len(messages) <= config.min_size:
            self.logger.debug(f"Creating single window due to small message count", context=ctx)
            return [self._create_window(conv_id, messages)]
        
        # Sort messages by timestamp
        messages = sorted(messages, key=lambda m: m.window_metadata.unixtime_range[0])
        
        # Create session windows
        windows = []
        current_session = []
        last_timestamp = None
        
        for message in messages:
            # Get message timestamp (use the earliest time in the range)
            timestamp = message.window_metadata.unixtime_range[0]
            
            # Start a new session if:
            # 1. This is the first message
            # 2. The time gap is larger than the timeout
            # 3. The current session has reached max_size
            if (last_timestamp is None or 
                timestamp - last_timestamp > config.timeout_seconds or
                len(current_session) >= config.max_size):
                
                # If we have a non-empty session, save it if it meets min_size
                if current_session and len(current_session) >= config.min_size:
                    window = self._create_window(conv_id, current_session)
                    windows.append(window)
                    self.logger.debug(f"Created session window", context={
                        "messages": len(current_session),
                        "window_id": str(window.id),
                        **ctx
                    })
                
                # Start a new session
                current_session = [message]
            else:
                # Add to current session
                current_session.append(message)
            
            # Update last timestamp
            last_timestamp = timestamp
        
        # Handle the last session if it's not empty and meets min_size
        if current_session and len(current_session) >= config.min_size:
            window = self._create_window(conv_id, current_session)
            windows.append(window)
            self.logger.debug(f"Created final session window", context={
                "messages": len(current_session),
                "window_id": str(window.id),
                **ctx
            })
        
        self.logger.info(f"Created session windows", context={
            "count": len(windows), 
            **ctx
        })
        
        return windows
    
    def _create_sliding_windows(self, conv_id: str, messages: List[Window], config) -> List[Window]:
        """Create sliding windows from messages based on size configuration.
        
        Args:
            conv_id: Conversation ID
            messages: List of Window objects (message-level windows)
            config: WindowBySizeConfig with size and overlap parameters
            
        Returns:
            List of Window objects, each representing a sliding window
        """
        # Convert messages iterator to list if needed
        messages = list(messages)
        ctx = {
            "conversation_id": conv_id,
            "message_count": len(messages),
            "window_type": "sliding"
        }
        
        self.logger.debug(f"Creating sliding windows", context={
            **ctx,
            "size": config.size,
            "overlap": config.overlap
        })
        
        # If there are fewer messages than the window size, just create one window
        if len(messages) <= config.size:
            self.logger.debug(f"Creating single window due to small message count", context=ctx)
            return [self._create_window(conv_id, messages)]
        
        # Sort messages by timestamp
        messages = sorted(messages, key=lambda m: m.window_metadata.unixtime_range[0])
        
        # Create sliding windows
        windows = []
        window_size = config.size
        overlap = config.overlap
        step = window_size - overlap
        
        # Ensure step is at least 1 to avoid infinite loop
        if step < 1:
            step = 1
            self.logger.warning(f"Adjusted step size to 1", context={
                "original_step": window_size - overlap,
                **ctx
            })
        
        # Create windows
        for i in range(0, len(messages), step):
            # Get window messages
            end_idx = min(i + window_size, len(messages))
            window_messages = messages[i:end_idx]
            
            # Skip windows smaller than the minimum size (except the last one)
            if len(window_messages) < step and i + step < len(messages):
                continue
                
            # Create window
            window = self._create_window(conv_id, window_messages)
            windows.append(window)
            self.logger.debug(f"Created sliding window", context={
                "range": f"{i+1}-{end_idx}",
                "messages": len(window_messages),
                "window_id": str(window.id),
                **ctx
            })
            
            # If we've reached the end, stop
            if end_idx >= len(messages):
                break
                
        self.logger.info(f"Created sliding windows", context={
            "count": len(windows),
            **ctx
        })
        return windows

    def _create_window(self, conv_id: str, messages: List[Window]) -> Window:
        """Create a window from a list of messages.
        
        Args:
            conv_id: Conversation ID
            messages: List of Window objects
            
        Returns:
            Window object containing aggregated message data
        """
        messages = list(messages)  # Convert iterator to list
        
        # Create context for structured logging
        ctx = {
            "conversation_id": conv_id,
            "message_count": len(messages)
        }
        
        self.logger.debug(f"Creating window", context=ctx)
        
        # Format messages as a conversation transcript using MessageAggregator
        concatenated_text = MessageAggregator.format_as_conversation(messages)
        
        # Compute message statistics
        message_stats = MessageAggregator.compute_message_stats(messages)
        self.logger.debug(f"Window message stats", context={**message_stats, **ctx})
        
        # Get time range from messages
        times = []
        dates = []
        for m in messages:
            times.extend(m.window_metadata.unixtime_range)
            dates.extend(m.window_metadata.date_range)
        
        times = sorted(times)
        dates = sorted(dates)
        
        # Count messages per user by merging from_ids dictionaries
        from_ids = {}
        from_names = {}  # Collect display names from all messages
        
        for m in messages:
            # Merge from_ids
            for from_id, count in m.window_metadata.from_ids.items():
                from_ids[from_id] = from_ids.get(from_id, 0) + count
            
            # Collect from_names if available
            if hasattr(m.window_metadata, 'from_names') and m.window_metadata.from_names:
                for user_id, display_name in m.window_metadata.from_names.items():
                    if user_id not in from_names:
                        from_names[user_id] = display_name
        
        # Generate deterministic UUID for window based on conversation_id and timestamp range
        window_id = Window.generate_deterministic_uuid(
            conversation_id=conv_id,
            min_timestamp=times[0],
            max_timestamp=times[-1]
        )
        
        # Get event_ids from messages
        event_ids = []
        for m in messages:
            event_ids.extend(m.window_metadata.event_ids)
        
        # Create window metadata
        window_metadata = WindowMetadata(
            id=window_id,
            conversation_id=conv_id,
            conversation_type=messages[0].conversation_type,
            date_range=[dates[0], dates[-1]],
            unixtime_range=[times[0], times[-1]],
            from_ids=from_ids,
            from_names=from_names,  # Include the collected display names
            event_ids=event_ids
        )
        
        # Create window with the same UUID used in metadata
        window = Window(
            id=window_id,
            conversation_id=conv_id,
            conversation_type=messages[0].conversation_type,
            concatenated_text=concatenated_text,
            window_metadata=window_metadata
        )
        
        self.logger.debug(f"Created window object", context={
            "window_id": str(window_id),
            "date_range": f"{dates[0]} to {dates[-1]}",
            "senders": len(from_ids),
            "event_count": len(event_ids),
            **ctx
        })
        
        return window


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Telegram export files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--input-dir", required=True, help="Directory containing conversation directories")
    parser.add_argument("--output-dir", help="Optional directory to write output files")
    parser.add_argument("--log-level", help="Override logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", help="Override path to write logs to a file")
    parser.add_argument("--json-logs", action="store_true", help="Override: Output logs in JSON format")
    parser.add_argument("--distributed", action="store_true", help="Override: Use distributed logging format")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Setup logging based on config, with CLI args taking precedence
    log_config = config.logging
    from noosphere.telegram.batch.logging import setup_logging, get_logger
    setup_logging(
        level=args.log_level.upper() if args.log_level else log_config.level,
        distributed=args.distributed if args.distributed else log_config.distributed,
        log_file=args.log_file if args.log_file else log_config.log_file,
        json_logs=args.json_logs if args.json_logs else log_config.json_logs,
        intercept_std_logging=log_config.intercept_std_logging
    )
    
    # Apply module-specific log levels from config
    if log_config.module_levels:
        import logging
        for module, level in log_config.module_levels.items():
            logging.getLogger(module).setLevel(level)
    
    logger = get_logger("telexp.main")
    logger.info("Starting Telegram export processing", context={
        "input_dir": args.input_dir,
        "output_dir": args.output_dir or "None",
        "config_file": args.config or "default"
    })
    
    # Run the processor (pass the already loaded config)
    processor = TelegramExportProcessor(config)
    stored_ids = processor.process(args.input_dir, args.output_dir)
    
    # Final summary
    logger.info("Processing complete", context={
        "windows_processed": len(stored_ids)
    })
    
    return 0


if __name__ == "__main__":
    exit(main())