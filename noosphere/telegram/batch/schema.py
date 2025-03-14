from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from datetime import datetime
import uuid

class Source(BaseModel):
    """Source information for a message."""
    file_path: str
    file_name: str
    file_extension: str
    file_mimetype: str
    file_offset: Optional[int] = None
    file_line_number: Optional[int] = None

    @field_validator('file_path')
    def validate_file_path(cls, v):
        """Ensure file path is not empty."""
        if not v or not v.strip():
            raise ValueError("file_path cannot be empty")
        return v

    @field_validator('file_extension')
    def validate_file_extension(cls, v):
        """Ensure file extension is not empty."""
        if not v or not v.strip():
            raise ValueError("file_extension cannot be empty")
        return v

    @field_validator('file_mimetype')
    def validate_file_mimetype(cls, v):
        """Ensure file mimetype is not empty."""
        if not v or not v.strip():
            raise ValueError("file_mimetype cannot be empty")
        return v

    @field_validator('file_line_number')
    def validate_line_number(cls, v):
        """Ensure line number is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("file_line_number must be positive")
        return v

    @field_validator('file_offset')
    def validate_offset(cls, v):
        """Ensure offset is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("file_offset must be non-negative")
        return v

class MessageType(str, Enum):
    """Types of messages in the pipeline."""
    TEXT = "text"
    SERVICE = "service"
    PHOTO = "photo"

class PhotoSize(BaseModel):
    """Photo size information."""
    type: str
    width: int
    height: int
    file: str

    @field_validator('width', 'height')
    def validate_dimensions(cls, v):
        """Ensure dimensions are positive."""
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v

    @field_validator('file')
    def validate_file_path(cls, v):
        """Ensure file path is not empty and has an extension."""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        if '.' not in v:
            raise ValueError("File path must have an extension")
        return v

class Photo(BaseModel):
    """Photo information."""
    file: str
    thumbnail: Optional[PhotoSize] = None
    sizes: Optional[List[PhotoSize]] = None
    width: Optional[int] = None
    height: Optional[int] = None

class Video(BaseModel):
    """Video information."""
    file: str
    thumbnail: Optional[PhotoSize] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[int] = None

class TextEntity(BaseModel):
    """Text entity information."""
    type: str
    text: str

class RawMessage(BaseModel):
    """Raw Telegram message from JSONL export."""
    id: Union[str, int]
    type: str
    date: str
    date_unixtime: int
    conversation_id: str
    conversation_type: Optional[str] = None
    from_id: Optional[str] = None
    from_name: Optional[str] = None
    text: Optional[Union[str, List[Union[str, TextEntity]]]] = None
    photo: Optional[Union[str, Photo]] = None
    reply_to_message_id: Optional[int] = None
    file: Optional[str] = None
    thumbnail: Optional[str] = None
    media_type: Optional[str] = None
    mime_type: Optional[str] = None
    duration_seconds: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    actor_id: Optional[str] = None
    actor: Optional[str] = None
    action: Optional[str] = None
    title: Optional[str] = None
    text_entities: Optional[List[TextEntity]] = None
    summary_text: Optional[str] = None  # Added field for image descriptions
    source: Source = None

    @field_validator('id')
    def convert_id_to_string(cls, v):
        """Convert ID to string if it's an integer."""
        return str(v) if isinstance(v, int) else v

    @field_validator('text', mode='before')
    def validate_text(cls, v):
        """Handle text that can be either string or list of entities."""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return [
                item if isinstance(item, str) else TextEntity(**item)
                for item in v
            ]
        return v

    @field_validator('photo', mode='before')
    def validate_photo(cls, v):
        """Handle photo that can be either string path or object."""
        if isinstance(v, str):
            return Photo(file=v)
        if isinstance(v, dict):
            return Photo(**v)
        return v

    @field_validator('date')
    def validate_date_format(cls, v):
        """Validate date string format."""
        try:
            datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
            return v
        except ValueError:
            raise ValueError("Date must be in format YYYY-MM-DDThh:mm:ss")

    @field_validator('date_unixtime')
    def validate_unixtime(cls, v):
        """Ensure unixtime is reasonable (after 2000 and before 2100)."""
        min_time = 946684800  # 2000-01-01
        max_time = 4102444800  # 2100-01-01
        if not min_time <= v <= max_time:
            raise ValueError("Unix timestamp must be between years 2000 and 2100")
        return v

    def get_message_type(self) -> MessageType:
        """Determine the message type."""
        if self.type == 'service':
            return MessageType.SERVICE
        if self.photo or self.media_type:
            return MessageType.PHOTO if not self.text else MessageType.MIXED
        return MessageType.TEXT

    def get_text_content(self) -> str:
        """Get text content from message."""
        if isinstance(self.text, str):
            return self.text
        if isinstance(self.text, list):
            return ' '.join(
                item.text if isinstance(item, TextEntity) else item
                for item in self.text
            )
        return ''

class StatusDetails(BaseModel):
    """Details for a status update."""
    model: Optional[str] = None
    vector: Optional[List[float]] = None
    summary: Optional[str] = None

class Status(BaseModel):
    """Status information for a component."""
    code: str
    message: Optional[str] = None
    details: Optional[StatusDetails] = None
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

class ComponentStatus(BaseModel):
    """Status tracking for a component."""
    latest: Optional[Status] = None
    history: List[Status] = Field(default_factory=list)

class WindowMetadata(BaseModel):
    """Window metadata information."""
    id: uuid.UUID = Field(description="UUID for the window")
    conversation_id: str
    conversation_type: Optional[str] = None
    date_range: List[str]
    unixtime_range: List[int]
    from_ids: Dict[str, int]
    from_names: Optional[Dict[str, str]] = None  # Map from_id to display name
    event_ids: List[str]

    @field_validator('date_range')
    def validate_date_range(cls, v):
        """Ensure date range has exactly two dates in chronological order."""
        if len(v) != 2:
            raise ValueError("Date range must contain exactly two dates")
        try:
            start = datetime.fromisoformat(v[0])
            end = datetime.fromisoformat(v[1])
            if end < start:
                raise ValueError("End date must be after start date")
        except ValueError as e:
            if "fromisoformat" in str(e):
                raise ValueError("Dates must be in ISO format (YYYY-MM-DDThh:mm:ss)")
            raise
        return v

    @field_validator('unixtime_range')
    def validate_unixtime_range(cls, v):
        """Ensure unixtime range has exactly two timestamps in chronological order."""
        if len(v) != 2:
            raise ValueError("Unixtime range must contain exactly two timestamps")
        if v[1] < v[0]:
            raise ValueError("End timestamp must be after start timestamp")
        return v

class Window(BaseModel):
    """Window data structure."""
    id: uuid.UUID = Field(description="UUID for the window")
    conversation_id: str
    conversation_type: Optional[str] = None
    concatenated_text: Optional[str] = None
    summary_text: Optional[str] = None
    vector: Optional[List[float]] = None
    image_path: Optional[str] = None
    window_metadata: WindowMetadata
    
    def has_valid_vector(self) -> bool:
        """Check if window has a valid vector for storage.
        
        Returns:
            bool: True if the vector is valid (non-empty list of floats)
        """
        if not self.vector:
            return False
        
        if not isinstance(self.vector, list):
            return False
            
        if len(self.vector) == 0:
            return False
            
        # Check that vector contains at least some non-zero values
        if not any(abs(v) > 1e-10 for v in self.vector):
            return False
            
        return True
    
    @staticmethod
    def generate_deterministic_uuid(conversation_id: str, min_timestamp: int, max_timestamp: int) -> uuid.UUID:
        """
        Generate a deterministic UUID based on conversation_id and timestamp range.
        
        Args:
            conversation_id: The conversation ID
            min_timestamp: The minimum/earliest timestamp in the window
            max_timestamp: The maximum/latest timestamp in the window
            
        Returns:
            A deterministic UUID derived from the inputs
        """
        import hashlib
        
        # Create deterministic string from inputs
        input_str = f"{conversation_id}:{min_timestamp}:{max_timestamp}"
        
        # Generate MD5 hash
        md5_hash = hashlib.md5(input_str.encode()).hexdigest()
        
        # Format as UUID (8-4-4-4-12 format)
        uuid_str = f"{md5_hash[:8]}-{md5_hash[8:12]}-{md5_hash[12:16]}-{md5_hash[16:20]}-{md5_hash[20:32]}"
        
        # Return as UUID object
        return uuid.UUID(uuid_str)

    @field_validator('window_metadata')
    def validate_metadata_id(cls, v: WindowMetadata, info: ValidationInfo) -> WindowMetadata:
        """Ensure window_metadata.id matches window.id."""
        if info.data.get('id') is not None and v.id != info.data['id']:
            raise ValueError("window_metadata.id must match window.id")
        return v

    def get_text_for_vectorization(self) -> Optional[str]:
        """Get the best available text for vectorization."""
        # Always use summary text for vectorization
        # If there's no summary, fail explicitly - we don't want to vectorize raw messages
        if self.summary_text:
            return self.summary_text
        return None

    def has_text_content(self) -> bool:
        """Check if window has any text content."""
        return bool(self.concatenated_text)

    def items(self):
        """Return items for dict-like interface."""
        return self.model_dump().items()

    def __getitem__(self, key):
        """Support dict-like access."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Support dict-like assignment."""
        setattr(self, key, value)

    def get(self, key, default=None):
        """Support dict-like get."""
        return getattr(self, key, default) 


class LangchainWindow(BaseModel):
    """Window format compatible with Langchain-style retrieval systems."""
    id: str = Field(description="UUID for the window as string")
    content: str = Field(description="Content text for the window (summary)")
    metadata: Dict[str, Any] = Field(description="Window metadata for retrieval")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "allow"