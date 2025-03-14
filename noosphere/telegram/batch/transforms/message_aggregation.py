"""Message aggregation utilities for telexp."""
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from noosphere.telegram.batch.schema import Window, WindowMetadata, RawMessage


class MessageAggregator:
    """Utility class for aggregating and formatting messages."""

    @staticmethod
    def format_as_conversation(messages: List[Window]) -> str:
        """Format a list of messages as a conversation transcript with usernames.
        
        Args:
            messages: List of Window objects representing individual messages
            
        Returns:
            str: A formatted conversation transcript with usernames and proper formatting
        """
        logger = logging.getLogger(__name__)
        
        # Format messages as a conversation transcript with usernames
        formatted_lines = []
        for i, m in enumerate(messages):
            event_ids = getattr(m.window_metadata, 'event_ids', ['unknown'])
            
            # Get the username from from_ids dictionary
            from_ids = getattr(m.window_metadata, 'from_ids', {})
            from_names = getattr(m.window_metadata, 'from_names', {})
            
            if from_ids:
                user_id = next(iter(from_ids.keys()), "unknown")
                # Use display name if available, otherwise use user_id
                username = from_names.get(user_id, user_id) if from_names else user_id
            else:
                username = f"user{i+1}"
                
            # Check for event ID and log message being processed
            event_id_str = event_ids[0] if event_ids else f"window_{i}"
            logger.info(f"Processing message {i+1} with event_id: {event_id_str}")
            
            # Log all fields for debugging
            for field in ["image_path", "summary_text", "concatenated_text"]:
                if hasattr(m, field) and getattr(m, field):
                    logger.info(f"  Field {field}: {getattr(m, field)}")
            
            # Process image content
            has_image = False
            image_description = None
            
            # Check for image path first
            if hasattr(m, "image_path") and m.image_path:
                has_image = True
                logger.info(f"Found image path: {m.image_path}")
                
                # First check for summary_text directly from the message
                if hasattr(m, "summary_text") and m.summary_text:
                    image_description = m.summary_text.strip()
                    logger.info(f"Found image description in summary_text: {image_description[:50]}...")
                
            # Process text content
            message_text = None
            if hasattr(m, "concatenated_text") and m.concatenated_text:
                message_text = m.concatenated_text.strip()
                
                # If text looks like "[Image description: X]", try to extract description
                if has_image and message_text.startswith("[Image description:") and not image_description:
                    # Extract description from text
                    try:
                        start = message_text.index("[Image description:") + len("[Image description:")
                        end = message_text.index("]", start)
                        image_description = message_text[start:end].strip()
                        logger.info(f"Extracted image description from text: {image_description[:50]}...")
                        # Remove image description from message text to avoid duplication
                        message_text = None
                    except ValueError:
                        logger.warning("Failed to extract image description from text")
                
                # If we already have a description and text duplicates it, skip text
                elif message_text and image_description and message_text.startswith("[Image description:"):
                    logger.info(f"Skipping text that duplicates image description: {message_text[:50]}...")
                    message_text = None
            
            # Format the message with username
            if has_image:
                # Always include images, with or without description
                if image_description:
                    # Use image description if available
                    logger.info(f"Using image description for message {i+1}: {image_description[:50]}...")
                    img_desc = image_description
                else:
                    # Fallback to generic "Image" description
                    logger.warning(f"No image description found for message {i+1}, using generic 'Image'")
                    img_desc = "Image"
                    
                # Format as markdown image with description
                # Extract just the filename from the full path
                image_filename = os.path.basename(m.image_path) if m.image_path else "image.jpg"
                formatted_lines.append(f"[{username}]: ![{img_desc}]({image_filename})")
                logger.info(f"Added image message from {username} with description: {image_filename}")
            elif message_text:
                # Split message by lines for multi-line formatting
                lines = message_text.split('\n')
                if len(lines) == 1:
                    formatted_lines.append(f"[{username}]: {message_text}")
                else:
                    formatted_lines.append(f"[{username}]: {lines[0]}")
                    # Add additional lines without username prefix
                    formatted_lines.extend(lines[1:])
                logger.info(f"Added text message from {username}: {message_text[:50]}...")
            else:
                logger.info(f"Message {i+1} (event_id: {event_ids[0]}) had no content to format")
            
        # Join all formatted lines with newlines to create a conversation transcript
        concatenated_text = "\n".join(formatted_lines)
        logger.info(f"Created formatted conversation with {len(formatted_lines)} messages:")
        if len(formatted_lines) > 0:
            sample_lines = min(3, len(formatted_lines))
            for i in range(sample_lines):
                logger.info(f"  Line {i+1}: {formatted_lines[i]}")
            logger.info(f"(total length: {len(concatenated_text)})")
            
        return concatenated_text
    
    @staticmethod
    def compute_message_stats(messages: List[Window]) -> Dict[str, Any]:
        """Compute statistics for a group of messages.
        
        Args:
            messages: List of Window objects representing individual messages
            
        Returns:
            Dict[str, Any]: Statistics about the message group
        """
        # Get unique users
        unique_users = set()
        message_count_by_user = {}
        total_text_length = 0
        
        for m in messages:
            from_ids = getattr(m.window_metadata, 'from_ids', {})
            for user_id in from_ids.keys():
                unique_users.add(user_id)
                message_count_by_user[user_id] = message_count_by_user.get(user_id, 0) + 1
                
            # Add text length
            if hasattr(m, "concatenated_text") and m.concatenated_text:
                total_text_length += len(m.concatenated_text)
        
        return {
            "message_count": len(messages),
            "unique_users": len(unique_users),
            "users": list(unique_users),
            "message_count_by_user": message_count_by_user,
            "total_text_length": total_text_length,
            "avg_message_length": total_text_length / len(messages) if messages else 0
        }
    
    @staticmethod
    def extract_message_metadata(messages: List[RawMessage]) -> Dict[str, Any]:
        """Extract metadata from a list of messages.
        
        Args:
            messages: List of raw messages
            
        Returns:
            Dict[str, Any]: Extracted metadata including times, dates, from_ids, etc.
        """
        # Extract times and dates
        times = sorted([m.date_unixtime for m in messages])
        dates = sorted([m.date for m in messages])
        
        # Count messages per user and collect display names
        from_ids = {}
        from_names = {}  # Map from_id to display name
        
        for m in messages:
            from_id = m.from_id or 'unknown'
            from_ids[from_id] = from_ids.get(from_id, 0) + 1
            
            # Store display name if available directly on message
            if hasattr(m, 'from_name') and m.from_name and from_id not in from_names:
                from_names[from_id] = m.from_name
        
        # Extract message IDs
        event_ids = [str(m.id) for m in messages]
        
        return {
            "times": times,
            "dates": dates,
            "from_ids": from_ids,
            "from_names": from_names,
            "event_ids": event_ids
        }
    
    @staticmethod
    def create_window_from_messages(messages: List[RawMessage], conversation_id: str) -> Window:
        """Create a window from a list of messages.
        
        Args:
            messages: List of validated messages to include in window
            conversation_id: ID of the conversation
            
        Returns:
            Window: Created window with metadata
        """
        # Extract metadata
        metadata = MessageAggregator.extract_message_metadata(messages)
        
        times = metadata["times"]
        dates = metadata["dates"]
        from_ids = metadata["from_ids"]
        from_names = metadata["from_names"]
        event_ids = metadata["event_ids"]
        
        # Generate deterministic UUID for window based on conversation_id and timestamp range
        window_id = Window.generate_deterministic_uuid(
            conversation_id=conversation_id,
            min_timestamp=times[0],
            max_timestamp=times[-1]
        )
        
        # Create window metadata
        window_metadata = WindowMetadata(
            id=window_id,
            conversation_id=conversation_id,
            conversation_type=messages[0].conversation_type,
            date_range=[dates[0], dates[-1]],
            unixtime_range=[times[0], times[-1]],
            from_ids=from_ids,
            from_names=from_names,  # Add from_names to metadata
            event_ids=event_ids
        )
        
        # Create window
        return Window(
            id=window_id,
            conversation_id=conversation_id,
            conversation_type=messages[0].conversation_type,
            window_metadata=window_metadata,
            vector=[0.0] * 768  # Will be filled by vectorizer
        )