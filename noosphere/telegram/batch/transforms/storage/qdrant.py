import apache_beam as beam
from typing import Dict, Any, Iterator, List, Optional
from apache_beam import typehints
import time
from pydantic import BaseModel, Field, field_validator
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from noosphere.telegram.batch.schema import Window, LangchainWindow
from noosphere.telegram.batch.logging import LoggingMixin

class Point(BaseModel):
    """Point data for Qdrant storage."""
    id: uuid.UUID = Field(description="UUID for the point")
    vector: List[float] = Field(description="Vector data")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Additional point data")

class VectorStorageConfig(BaseModel):
    """Configuration for vector storage."""
    url: str
    collection: str
    vector_size: int
    retries: int = 3
    backoff_factor: float = 0.5
    timeout: int = 30
    batch_size: int = 100  # Added configurable batch size

    @property
    def test_collection(self) -> str:
        """Return a test collection name based on the main collection name.
        
        This is used for integration tests to avoid interfering with production data.
        """
        timestamp = int(time.time())
        return f"{self.collection}_test_{timestamp}"
        
    @field_validator('batch_size')
    def validate_batch_size(cls, v):
        """Ensure batch size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

class QdrantVectorStore(beam.DoFn, LoggingMixin):
    """Beam DoFn that stores vectors in Qdrant."""

    def __init__(self, config: VectorStorageConfig, client: Optional[QdrantClient] = None):
        """Initialize with config from vector_storage section.
        
        Args:
            config: VectorStorageConfig instance with Qdrant settings
            client: Optional pre-configured QdrantClient instance. If provided, this client
                   will be used instead of creating a new one. Useful for testing.
        """
        super().__init__()
        self.config = config
        self._client = client
        self._batch: List[Point] = []
        self._setup_called = False
        self._setup_logging()

    def _validate_uuid(self, id_value: uuid.UUID) -> uuid.UUID:
        """Validate that an ID is a valid UUID."""
        if not isinstance(id_value, uuid.UUID):
            raise ValueError(f"ID must be a UUID, got {type(id_value)}")
        return id_value

    def __getstate__(self):
        """Control which attributes are pickled."""
        state = super().__getstate__()
        # Don't pickle client or batch - they will be recreated
        state['_client'] = None
        state['_batch'] = []  # Clear batch during pickling
        state['_setup_called'] = False  # Reset setup state
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        super().__setstate__(state)
        self._batch = []
        self._setup_called = False

    def setup(self):
        """Initialize the client and ensure collection exists."""
        # Skip if already set up
        if self._setup_called:
            return
            
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        ctx = {
            "url": self.config.url,
            "collection": self.config.collection,
            "vector_size": self.config.vector_size
        }

        # Create client if not already provided
        if self._client is None:
            self.logger.info("Connecting to Qdrant", context=ctx)
            self._client = QdrantClient(
                url=self.config.url,
                timeout=self.config.timeout
            )
            
        self._setup_called = True

        # Test connection and create collection if needed
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.config.collection not in collection_names:
                self.logger.info("Creating collection", context=ctx)
                self._client.create_collection(
                    collection_name=self.config.collection,
                    vectors_config=models.VectorParams(
                        size=self.config.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info("Created collection", context=ctx)
        except Exception as e:
            self.logger.error("Failed to connect to Qdrant", context=ctx, exception=e)
            raise

    def start_bundle(self):
        """Initialize state for this bundle."""
        # Ensure setup has been called
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        # Initialize client if needed
        if self._client is None:
            self.setup()
            
        self._batch = []
        self.logger.debug("Starting new batch")

    def finish_bundle(self):
        """Flush any remaining points in the batch when the bundle is finished."""
        if not self._batch:
            return
            
        try:
            # Ensure setup has been called
            if not hasattr(self, 'logger'):
                self._setup_logging()
                
            if self._client is None:
                self.setup()
                
            batch_size = len(self._batch)
            self.logger.info("Flushing final batch", context={"batch_size": batch_size})
            self._flush_batch()
        except Exception as e:
            self.logger.error("Failed to flush batch in finish_bundle", exception=e)
            # Don't raise to allow other batches to finish

    def teardown(self):
        """Clean up the client and flush any remaining batched points."""
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        # Try to flush remaining points    
        if self._batch:
            try:
                if self._client is None:
                    self.setup()
                    
                self.logger.info("Flushing remaining points in teardown", 
                                context={"batch_size": len(self._batch)})
                self._flush_batch()
            except Exception as e:
                self.logger.error("Failed to flush batch in teardown", exception=e)
                
        # Always try to close client if it exists
        if self._client:
            try:
                self.logger.info("Closing Qdrant client connection")
                self._client.close()
            except Exception as e:
                self.logger.error("Error closing client", exception=e)
            finally:
                self._client = None
                
        # Reset setup state
        self._setup_called = False

    def _flush_batch(self):
        """Upload batched points to Qdrant."""
        if not self._batch:
            return
            
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        # Ensure we have a client before proceeding
        if not self._client:
            self.setup()
            
        if not self._client:
            self.logger.error("No Qdrant client available")
            return

        # Create context for structured logging
        ctx = {
            "batch_size": len(self._batch),
            "collection": self.config.collection,
            "retries": self.config.retries
        }

        # Format points for Qdrant API
        points = [
            models.PointStruct(
                id=str(point.id),
                vector=point.vector,
                payload=point.payload
            )
            for point in self._batch
        ]

        # Upload points with retry
        last_error = None
        batch_to_retry = self._batch.copy()  # Copy batch for retries
        self._batch = []  # Clear batch immediately to avoid duplicates

        for attempt in range(self.config.retries):
            attempt_ctx = {**ctx, "attempt": attempt + 1}
            try:
                self.logger.info("Uploading batch of points", context=attempt_ctx)
                self._client.upsert(
                    collection_name=self.config.collection,
                    points=points,
                    wait=True
                )
                self.logger.info("Successfully uploaded points", context=attempt_ctx)
                return
            except UnexpectedResponse as e:
                last_error = e
                self.logger.error(f"Qdrant API error: {type(e).__name__}: {str(e)}", context=attempt_ctx, exception=e)
                if attempt < self.config.retries - 1:  # Not the last attempt
                    backoff_time = self.config.backoff_factor * (2 ** attempt)
                    self.logger.debug(f"Retrying after backoff", 
                                     context={**attempt_ctx, "backoff_seconds": backoff_time})
                    time.sleep(backoff_time)
            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error uploading batch: {type(e).__name__}: {str(e)}", context=attempt_ctx, exception=e)
                if attempt < self.config.retries - 1:  # Not the last attempt
                    backoff_time = self.config.backoff_factor * (2 ** attempt)
                    self.logger.debug(f"Retrying after backoff", 
                                     context={**attempt_ctx, "backoff_seconds": backoff_time})
                    time.sleep(backoff_time)

        # If we get here, all retries failed
        if last_error:
            self.logger.error("All retry attempts failed", context=ctx)
            raise last_error

    @typehints.with_input_types(LangchainWindow)
    @typehints.with_output_types(str)
    def process(self, element: LangchainWindow) -> Iterator[str]:
        """Process a LangchainWindow record for storage.
        
        Args:
            element: LangchainWindow object to store
            
        Returns:
            Iterator[str]: ID of stored window as string
        """
        # Ensure logger is set up
        if not hasattr(self, 'logger'):
            self._setup_logging()
            
        # Initialize client if needed
        if self._client is None:
            self.setup()
        
        # Extract data from LangchainWindow
        window_id = element.id
        vector = element.vector
        
        # Create a copy of the element without the vector to avoid storing it twice
        payload = element.model_dump()
        if 'vector' in payload:
            del payload['vector']
        
        # Convert window_id to UUID if it's a string
        if isinstance(window_id, str):
            try:
                window_id = uuid.UUID(window_id)
            except ValueError:
                # Always convert to string for consistent return
                window_id_str = str(window_id)
                self.logger.error("Invalid UUID string", context={"window_id": window_id_str})
                yield window_id_str
                return
        
        # Always yield window_id regardless of success/failure
        window_id_str = str(window_id)
        
        # Create context for structured logging
        ctx = {
            "window_id": window_id_str,
            "collection": self.config.collection
        }
        
        self.logger.info("Processing window for storage", context=ctx)
        
        # Validate vector
        if not vector or not isinstance(vector, list) or len(vector) == 0:
            self.logger.error("No valid vector found", context=ctx)
            yield window_id_str
            return
            
        # Check for all-zero vectors
        if not any(abs(v) > 1e-10 for v in vector):
            self.logger.error("Vector contains all zeros", context=ctx)
            yield window_id_str
            return
            
        if len(vector) != self.config.vector_size:
            error_ctx = {
                **ctx,
                "actual_size": len(vector),
                "expected_size": self.config.vector_size
            }
            self.logger.error("Vector size mismatch", context=error_ctx)
            yield window_id_str
            return

        # Create point with UUID
        try:
            point = Point(
                id=self._validate_uuid(window_id),
                vector=vector,
                payload=payload  # Store serialized LangchainWindow as payload
            )

            # Add to batch
            self._batch.append(point)
            
            # Log batch size at debug level
            self.logger.debug("Added point to batch", context={
                **ctx,
                "batch_size": len(self._batch),
                "batch_limit": self.config.batch_size
            })
            
            # Flush if batch is full
            if len(self._batch) >= self.config.batch_size:
                self.logger.info("Batch is full, flushing", context={
                    "batch_size": len(self._batch),
                    "collection": self.config.collection
                })
                self._flush_batch()
                
        except Exception as e:
            self.logger.error(f"Error processing window for storage: {type(e).__name__}: {str(e)}", context=ctx, exception=e)
            # Continue processing even if point creation or flush fails
        
        # Always return ID as string
        yield window_id_str
