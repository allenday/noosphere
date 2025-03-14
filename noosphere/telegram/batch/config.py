"""Configuration module for noosphere.telegram.batch."""
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator

from noosphere.telegram.batch.services.llm import LLMService
from noosphere.telegram.batch.services.embedding import EmbeddingService
from noosphere.telegram.batch.transforms.storage.qdrant import VectorStorageConfig

class BaseTransformerConfig(BaseModel):
    """Base configuration for all transformers."""
    url: str = Field(default="http://localhost:11434", description="Ollama API URL")
    model: str = Field(..., description="Model name to use")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retries: int = Field(default=3, description="Number of retries for failed requests")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")
    # Use strictly defined field names to avoid mismatches
    input_field: str = Field(default="concatenated_text", description="Field name to read input from")
    output_field: str = Field(default="summary_text", description="Field name to write output to")
    output_type: str = Field(..., description="Type of output (text or vector)")

class TextVectorizerConfig(BaseTransformerConfig):
    """Configuration for text vectorization."""
    vector_size: int = Field(..., description="Expected size of output vectors")
    embedding_service: str = Field(..., description="Name of embedding service to use")

    @field_validator('vector_size')
    def validate_vector_size(cls, v):
        if v <= 0:
            raise ValueError("Invalid vector_size: must be positive")
        return v

    @field_validator('output_type')
    def validate_output_type(cls, v):
        if v != 'vector':
            raise ValueError("output_type must be 'vector' for text vectorizer")
        return v

class TextSummarizerConfig(BaseTransformerConfig):
    """Configuration for text summarization."""
    prompt: str = Field(..., description="Prompt template for summarization")
    llm_service: str = Field(..., description="Name of LLM service to use")

    @field_validator('output_type')
    def validate_output_type(cls, v):
        if v != 'text':
            raise ValueError("output_type must be 'text' for text summarizer")
        return v

class ImageSummarizerConfig(BaseTransformerConfig):
    """Configuration for image summarization."""
    prompt: str = Field(..., description="Prompt template for image description")
    llm_service: str = Field(..., description="Name of LLM service to use")

    @field_validator('output_type')
    def validate_output_type(cls, v):
        if v != 'text':
            raise ValueError("output_type must be 'text' for image summarizer")
        return v

class WindowBySessionConfig(BaseModel):
    """Configuration for session-based windowing."""
    timeout_seconds: int = Field(description="Session timeout in seconds")
    min_size: int = Field(description="Minimum number of messages per window")
    max_size: int = Field(description="Maximum number of messages per window")

    @field_validator('timeout_seconds', 'min_size', 'max_size')
    def validate_positive(cls, v, field):
        if v <= 0:
            raise ValueError(f"{field.name} must be positive")
        return v

    @field_validator('max_size')
    def validate_max_size(cls, v, info):
        min_size = info.data.get('min_size')
        if min_size and v < min_size:
            raise ValueError("max_size must be greater than or equal to min_size")
        return v

class WindowBySizeConfig(BaseModel):
    """Configuration for size-based sliding windows."""
    size: int = Field(description="Number of messages per window")
    overlap: int = Field(description="Number of messages to overlap between windows")

    @field_validator('size', 'overlap')
    def validate_positive(cls, v, field):
        if v <= 0:
            raise ValueError(f"{field.name} must be positive")
        return v

    @field_validator('overlap')
    def validate_overlap(cls, v, info):
        size = info.data.get('size')
        if size and v >= size:
            raise ValueError("overlap must be less than size")
        return v
        
class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Default logging level")
    distributed: bool = Field(default=False, description="Use distributed logging format")
    log_file: Optional[str] = Field(default=None, description="Optional path to log file")
    json_logs: bool = Field(default=False, description="Output logs in JSON format")
    intercept_std_logging: bool = Field(default=True, description="Intercept standard logging")
    module_levels: Dict[str, str] = Field(
        default_factory=dict,
        description="Specific log levels for modules"
    )
    rotation: str = Field(default="10 MB", description="When to rotate logs (size, time)")
    retention: str = Field(default="1 week", description="How long to keep logs")
    compression: str = Field(default="zip", description="Compression format for rotated logs")
    enqueue: bool = Field(default=False, description="Use thread-safe queue for logging (set to False for Apache Beam compatibility)")
    log_startup: bool = Field(default=True, description="Log startup information")
    
    @field_validator('level')
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

class Config(BaseModel):
    """Root configuration object."""
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm_services: List[LLMService]
    embedding_services: List[EmbeddingService]
    text_vectorizer: TextVectorizerConfig
    text_summarizer: TextSummarizerConfig
    image_summarizer: ImageSummarizerConfig
    text_vector_storage: VectorStorageConfig
    image_vector_storage: VectorStorageConfig
    window_by_session: WindowBySessionConfig
    window_by_size: WindowBySizeConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Config':
        """Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path) as f:
            data = yaml.safe_load(f)
            
        return cls(**data)

def load_config(path: Optional[str | Path] = None) -> Config:
    """Load configuration from default or specified path.
    
    Args:
        path: Optional path to config file. If not provided, looks for conf.yaml
             in current directory or parent directories.
            
    Returns:
        Config: Validated configuration object
        
    Raises:
        FileNotFoundError: If no config file is found
    """
    if path is None:
        # Look for conf.yaml in current directory or parent directories
        current = Path.cwd()
        while current != current.parent:
            config_path = current / 'conf.yaml'
            if config_path.exists():
                return Config.from_yaml(config_path)
            current = current.parent
        raise FileNotFoundError("Could not find conf.yaml in current or parent directories")
    
    return Config.from_yaml(path)