import pytest
from unittest.mock import Mock, patch, call
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
import yaml
import uuid
import time
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import CollectionsResponse, CollectionDescription, VectorParams, Distance

from noosphere.telegram.batch.transforms.storage.qdrant import QdrantVectorStore, VectorStorageConfig, Point
from noosphere.telegram.batch.schema import LangchainWindow

def load_config():
    with open('conf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['text_vector_storage']

@pytest.fixture
def real_config():
    """Load real configuration from conf.yaml."""
    return VectorStorageConfig(**load_config())

@pytest.fixture
def test_config():
    """Test configuration for vector storage."""
    return VectorStorageConfig(
        url="http://100.82.220.106:6333",
        collection="test_text",
        test_collection="test_text",
        vector_size=768,
        retries=3,
        backoff_factor=0.5,
        timeout=30
    )

@pytest.fixture
def test_window():
    """Test window data in LangchainWindow format."""
    return LangchainWindow(
        id='12345678-1234-1234-1234-123456789012',
        content='Test summary',
        vector=[0.1] * 768,  # Vector of correct size
        metadata={
            "source": "blob",
            "blobType": "text/plain",
            "loc": {
                "lines": {
                    "from": 0,
                    "to": 999
                }
            },
            "file_id": "test/test-conv-1",
            "pubkey": "test-conv-1",
            "conversation_id": "test-conv-1",
            "conversation_type": "test"
        }
    )

def test_store_vector_success(test_config, test_window):
    """Test successful vector storage."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        # Setup mock responses
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance

        with TestPipeline() as p:
            input_data = [test_window]
            output = (
                p
                | beam.Create(input_data)
                | beam.ParDo(QdrantVectorStore(test_config))
            )

            def check_output(windows):
                assert len(windows) == 1
                return True

            assert_that(output, check_output)
            p.run().wait_until_finish()  # Wait for pipeline to finish

            # Verify points were stored
            assert mock_instance.upsert.call_count >= 1
            # Verify the correct collection was used
            assert all(
                call.kwargs['collection_name'] == test_config.collection
                for call in mock_instance.upsert.call_args_list
            )

def test_store_vector_batch(test_config, test_window):
    """Test batch vector storage."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        # Setup mock responses
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance

        # Create two test windows
        window1 = test_window
        window2 = LangchainWindow(
            id=str(uuid.uuid4()),
            content=test_window.content,
            vector=test_window.vector,
            metadata=test_window.metadata.copy()
        )

        with TestPipeline() as p:
            input_data = [window1, window2]
            output = (
                p
                | beam.Create(input_data)
                | beam.ParDo(QdrantVectorStore(test_config))
            )

            def check_output(windows):
                assert len(windows) == 2
                return True

            assert_that(output, check_output)
            p.run().wait_until_finish()  # Wait for pipeline to finish

            # Verify points were stored
            assert mock_instance.upsert.call_count >= 1
            # Verify points were batched correctly
            for call in mock_instance.upsert.call_args_list:
                assert len(call.kwargs['points']) <= 100  # Check batch size limit

def test_store_vector_invalid_size(test_config, test_window):
    """Test handling of invalid vector size."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance

        # Create a new window with wrong vector size
        invalid_window = LangchainWindow(
            id=test_window.id,
            content=test_window.content,
            vector=[0.1] * (test_config.vector_size + 1),  # Wrong size
            metadata=test_window.metadata.copy()
        )

        with TestPipeline() as p:
            input_data = [invalid_window]
            output = (
                p
                | beam.Create(input_data)
                | beam.ParDo(QdrantVectorStore(test_config))
            )

            def check_output(windows):
                assert len(windows) == 1
                return True

            assert_that(output, check_output)
            p.run().wait_until_finish()  # Wait for pipeline to finish

            # Verify no points were stored
            assert mock_instance.upsert.call_count == 0

def test_store_vector_create_collection(test_config, test_window):
    """Test collection creation."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[]  # Empty collections list
        )
        mock_client.return_value = mock_instance

        with TestPipeline() as p:
            input_data = [test_window]
            output = (
                p
                | beam.Create(input_data)
                | beam.ParDo(QdrantVectorStore(test_config))
            )

            def check_output(windows):
                assert len(windows) == 1
                return True

            assert_that(output, check_output)
            p.run().wait_until_finish()  # Wait for pipeline to finish

            # Verify collection was created
            mock_instance.create_collection.assert_called_once_with(
                collection_name=test_config.collection,
                vectors_config=VectorParams(
                    size=test_config.vector_size,
                    distance=Distance.COSINE
                )
            )

def test_store_vector_upload_retry(test_config, test_window):
    """Test retry on upload failure."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        # First call fails, second succeeds
        mock_instance.upsert.side_effect = [
            UnexpectedResponse(
                status_code=500,
                reason_phrase="Upload failed",
                content=b"Error",
                headers={"content-type": "text/plain"}
            ),
            None,  # Success on second try
            None   # Success for finish_bundle
        ]
        mock_client.return_value = mock_instance

        transform = QdrantVectorStore(test_config)
        transform.setup()
        
        # Process should succeed after retry
        result = list(transform.process(test_window))
        transform.finish_bundle()
        
        # Verify retry behavior
        assert mock_instance.upsert.call_count >= 2
        # Now checking for window ID as string
        assert result == [test_window.id]

def test_store_vector_upload_failure(test_config, test_window):
    """Test handling of upload failure after all retries."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        # All calls fail with same error
        error = UnexpectedResponse(
            status_code=500,
            reason_phrase="Upload failed",
            content=b"Error",
            headers={"content-type": "text/plain"}
        )
        mock_instance.upsert.side_effect = error
        mock_client.return_value = mock_instance

        transform = QdrantVectorStore(test_config)
        transform.setup()
        
        # Process should return window ID even after all retries fail
        result = list(transform.process(test_window))
        
        # Force batch upload - should fail after retries
        try:
            transform.finish_bundle()
        except UnexpectedResponse:
            pass  # Expected error
        
        # Verify retry behavior
        assert mock_instance.upsert.call_count >= test_config.retries
        # Now checking for window ID as string
        assert result == [test_window.id]

def test_store_vector_cleanup(test_config, test_window):
    """Test cleanup of batched points."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance

        with TestPipeline() as p:
            input_data = [test_window]
            output = (
                p
                | beam.Create(input_data)
                | beam.ParDo(QdrantVectorStore(test_config))
            )

            def check_output(windows):
                assert len(windows) == 1
                return True

            assert_that(output, check_output)
            p.run().wait_until_finish()  # Wait for pipeline to finish

            # Verify points were stored during cleanup
            assert mock_instance.upsert.call_count >= 1

def test_store_vector_real_qdrant(real_config, test_window):
    """Integration test using real Qdrant server from conf.yaml."""
    with TestPipeline() as p:
        input_data = [test_window]
        
        output = (
            p 
            | beam.Create(input_data)
            | beam.ParDo(QdrantVectorStore(real_config))
        )
        
        def check_output(ids):
            assert len(ids) == 1
            window_id = ids[0]
            assert window_id == test_window.id
            return True
            
        assert_that(output, check_output)

def test_point_validation():
    """Test Point model validation."""
    # Valid point
    point = Point(
        id=uuid.uuid4(),
        vector=[0.1, 0.2, 0.3],
        payload={"data": "test"}
    )
    assert isinstance(point.id, uuid.UUID)
    assert len(point.vector) == 3
    
    # Invalid UUID
    with pytest.raises(ValueError):
        Point(
            id="not-a-uuid",
            vector=[0.1, 0.2, 0.3]
        )
    
    # Invalid vector type
    with pytest.raises(ValueError):
        Point(
            id=uuid.uuid4(),
            vector="not-a-list"
        )
    
    # Invalid vector elements
    with pytest.raises(ValueError):
        Point(
            id=uuid.uuid4(),
            vector=["not", "numbers"]
        )

def test_store_vector_pickling(test_config, test_window):
    """Test that transform can be pickled and unpickled."""
    import pickle
    from telexp.transforms.storage.qdrant import __name__ as qdrant_module_name
    
    # Create transform and verify initial state
    transform = QdrantVectorStore(test_config)
    assert transform._client is None
    assert hasattr(transform, 'logger')  # Logger is now initialized immediately
    assert transform._setup_called is False
    
    # Verify config is preserved
    assert transform.config == test_config
    # Logger name now uses class_name and id format instead of module name
    assert transform.logger_name.startswith('QdrantVectorStore_')
    
    # Pickle and unpickle
    pickled = pickle.dumps(transform)
    unpickled = pickle.loads(pickled)
    
    # Verify config is preserved
    assert unpickled.config == transform.config
    # Logger name may be different after unpickling, but should follow the same pattern
    assert unpickled.logger_name.startswith('QdrantVectorStore_')
    
    # Verify non-picklable attributes were reset
    assert unpickled._client is None
    assert hasattr(unpickled, 'logger')  # Logger is recreated during unpickling
    assert unpickled._setup_called is False
    
    # Test that transform recreates resources and works after unpickling
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        # Setup client mock
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance
        
        # Process input through the unpickled transform
        result = list(unpickled.process(test_window))
        
        # Verify the result
        assert len(result) == 1
        assert result[0] == test_window.id
        
        # We don't need to verify the logger was used since we're using loguru
        
        # Verify client was recreated
        assert unpickled._client is not None
        assert unpickled._setup_called is True

def test_store_vector_teardown(test_config):
    """Test proper cleanup in teardown."""
    transform = QdrantVectorStore(test_config)
    
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        # Setup mocks
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance
        
        # Setup transform
        transform.setup()
        assert transform._client is not None
        
        # Add a point to the batch
        batch_window = LangchainWindow(
            id='12345678-1234-1234-1234-123456789012',
            content='Test summary',
            vector=[0.1] * test_config.vector_size,
            metadata={"file_id": "test/id", "pubkey": "test"}
        )
        list(transform.process(batch_window))  # Force batch creation
        
        # Teardown
        transform.teardown()
        
        # Verify client was closed and cleared
        mock_instance.close.assert_called_once()
        assert transform._client is None

def test_store_vector_setup_idempotent(test_config):
    """Test that calling setup multiple times is safe."""
    transform = QdrantVectorStore(test_config)
    
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client, \
         patch('logging.getLogger') as mock_logger:
        # Setup mocks
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance
        mock_logger.return_value = Mock()
        
        # Call setup twice
        transform.setup()
        first_client = transform._client
        
        transform.setup()
        second_client = transform._client
        
        # Verify client wasn't recreated
        assert first_client is second_client
        assert mock_client.call_count == 1 

def test_store_vector_with_provided_client(test_config):
    """Test using QdrantVectorStore with a provided client."""
    # Create a mock client
    mock_client = Mock()
    mock_client.get_collections.return_value = CollectionsResponse(
        collections=[CollectionDescription(name=test_config.collection)]
    )
    
    # Create transform with provided client
    transform = QdrantVectorStore(test_config, client=mock_client)
    
    # Verify the client was set correctly
    assert transform._client is mock_client
    
    # Run setup
    with patch('logging.getLogger'):
        transform.setup()
    
    # Verify the client wasn't recreated
    assert transform._client is mock_client
    
    # Verify client method was called during setup
    mock_client.get_collections.assert_called_once()
    
    # Create test window
    test_langchain_window = LangchainWindow(
        id='12345678-1234-1234-1234-123456789012',
        content='Test summary',
        vector=[0.1] * test_config.vector_size,
        metadata={"file_id": "test/id", "pubkey": "test"}
    )
    
    # Process window
    results = list(transform.process(test_langchain_window))
    
    # Verify results
    assert len(results) == 1
    assert results[0] == test_langchain_window.id
    
    # Force flush batch
    transform.finish_bundle()
    
    # Verify upsert was called
    assert mock_client.upsert.call_count == 1

def test_invalid_input_handling(test_config):
    """Test handling of invalid inputs to process()."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        mock_client.return_value = mock_instance
        
        transform = QdrantVectorStore(test_config)
        transform.setup()
        
        # Test case 1: Invalid window ID
        bad_id_window = LangchainWindow(
            id='not-a-uuid',
            content='Test summary',
            vector=[0.1] * test_config.vector_size,
            metadata={"file_id": "test/id", "pubkey": "test"}
        )
        result = list(transform.process(bad_id_window))
        assert len(result) == 1
        assert result[0] == 'not-a-uuid'
        
        # Test case 2: Missing vector
        no_vector_window = LangchainWindow(
            id='12345678-1234-1234-1234-123456789012',
            content='Test summary',
            vector=None,
            metadata={"file_id": "test/id", "pubkey": "test"}
        )
        result = list(transform.process(no_vector_window))
        assert len(result) == 1
        assert result[0] == '12345678-1234-1234-1234-123456789012'
        
        # Test case 3: Wrong vector size
        wrong_size_window = LangchainWindow(
            id='12345678-1234-1234-1234-123456789012',
            content='Test summary',
            vector=[0.1] * (test_config.vector_size - 1),
            metadata={"file_id": "test/id", "pubkey": "test"}
        )
        result = list(transform.process(wrong_size_window))
        assert len(result) == 1
        assert result[0] == '12345678-1234-1234-1234-123456789012'
        
        # Test case 4: Missing client in _flush_batch
        transform._client = None
        valid_window = LangchainWindow(
            id='12345678-1234-1234-1234-123456789012',
            content='Test summary',
            vector=[0.1] * test_config.vector_size,
            metadata={"file_id": "test/id", "pubkey": "test"}
        )
        result = list(transform.process(valid_window))
        assert len(result) == 1
        transform.finish_bundle()  # Should handle missing client and not raise errors
        
        # Restore client for teardown
        transform._client = mock_instance

def test_finish_bundle_error_handling(test_config):
    """Test error handling in finish_bundle."""
    with patch('telexp.transforms.storage.qdrant.QdrantClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collections.return_value = CollectionsResponse(
            collections=[CollectionDescription(name=test_config.collection)]
        )
        
        # Set up upsert to fail
        mock_instance.upsert.side_effect = Exception("Simulated error")
        mock_client.return_value = mock_instance
        
        transform = QdrantVectorStore(test_config)
        transform.setup()
        
        # Add a point to batch
        batch_window = LangchainWindow(
            id='12345678-1234-1234-1234-123456789012',
            content='Test summary',
            vector=[0.1] * test_config.vector_size,
            metadata={"file_id": "test/id", "pubkey": "test"}
        )
        list(transform.process(batch_window))
        
        # This should not raise even though upsert fails
        transform.finish_bundle()
        
        # Verify error was logged
        assert mock_instance.upsert.called