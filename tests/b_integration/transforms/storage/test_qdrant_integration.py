"""Integration tests for QdrantVectorStore with live services."""
import pytest
import uuid
import yaml
import numpy as np
import time
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from telexp.transforms.storage.qdrant import QdrantVectorStore, VectorStorageConfig, Point
from telexp.schema import Window, WindowMetadata
from telexp.config import Config

# Mark all tests as integration tests
pytestmark = pytest.mark.integration

def load_config():
    """Load configuration from conf.yaml."""
    with open('conf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config)

@pytest.fixture
def test_config():
    """Load the actual config but use test collection."""
    config = load_config()
    
    # Use the test collection to avoid affecting production data
    # Generate a unique collection name for this test run to avoid conflicts
    test_collection = f"test_text_{int(time.time())}"
    
    return VectorStorageConfig(
        url=config.text_vector_storage.url,
        collection=test_collection,  # Use the same collection for both
        test_collection=test_collection,  # Use the same collection for both
        vector_size=config.text_vector_storage.vector_size,
        retries=config.text_vector_storage.retries,
        backoff_factor=config.text_vector_storage.backoff_factor,
        timeout=config.text_vector_storage.timeout
    )

@pytest.fixture
def shared_client(test_config):
    """Create a Qdrant client to be used by all tests."""
    # Create a client that will be shared for this test
    client = QdrantClient(url=test_config.url, timeout=test_config.timeout)
    
    # Create or recreate the test collection
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        if any(c.name == test_config.collection for c in collections):
            # Delete existing test collection
            client.delete_collection(collection_name=test_config.collection)
    except Exception as e:
        print(f"Error checking collections: {e}")
    
    # Create the test collection
    client.create_collection(
        collection_name=test_config.collection,
        vectors_config=models.VectorParams(
            size=test_config.vector_size,
            distance=models.Distance.COSINE
        )
    )
    
    print(f"Created collection {test_config.collection}")
    
    # Yield the client for use in tests
    yield client
    
    # Cleanup after all tests
    try:
        print(f"Cleaning up collection {test_config.collection}")
        client.delete_collection(collection_name=test_config.collection)
        print(f"Successfully deleted collection {test_config.collection}")
        client.close()
    except Exception as e:
        print(f"Error cleaning up test collection: {e}")

@pytest.fixture
def test_windows():
    """Create test windows with vectors."""
    # Create 3 windows with random vectors
    windows = []
    for i in range(3):
        window_id = uuid.uuid4()
        vector = np.random.rand(768).tolist()  # Random vector
        
        window = Window(
            id=window_id,
            conversation_id=f'test-conv-{i}',
            conversation_type='test',
            summary_text=f'Test summary {i}',
            vector=vector,
            concatenated_text=f'Test text {i}',
            window_metadata=WindowMetadata(
                id=window_id,
                conversation_id=f'test-conv-{i}',
                conversation_type='test',
                date_range=[f'2024-01-0{i+1}T00:00:00', f'2024-01-0{i+1}T01:00:00'],
                unixtime_range=[1704067200 + i*86400, 1704070800 + i*86400],
                from_ids={f'user{i}': 1},
                event_ids=[f'{i}']
            )
        )
        windows.append(window)
    
    return windows

@pytest.mark.integration
def test_qdrant_store_integration(test_config, shared_client, test_windows):
    """Test actual storage in Qdrant."""
    # Create transform with the shared client
    transform = QdrantVectorStore(test_config, client=shared_client)
    
    # No need to call setup explicitly, it will be called by process
    
    # Store test windows
    result_ids = []
    for window in test_windows:
        results = list(transform.process(window))
        # QdrantVectorStore returns window IDs after storing
        for result in results:
            result_ids.append(result)
    
    # Force flush any remaining batched points
    transform.finish_bundle()
    
    # Verify all windows were processed
    assert len(result_ids) == len(test_windows)
    
    # Wait for vectors to be searchable in Qdrant
    time.sleep(1)
    
    # Verify storage directly using the same client
    for window in test_windows:
        # Search for the window by ID
        print(f"Searching for window {window.id} in collection {test_config.collection}")
        search_result = shared_client.retrieve(
            collection_name=test_config.collection,
            ids=[str(window.id)],
            with_vectors=True  # Explicitly request vectors
        )
        
        # Verify point was stored and has correct data
        assert len(search_result) == 1
        stored_point = search_result[0]
        
        assert str(window.id) == stored_point.id
        assert len(stored_point.vector) == test_config.vector_size
        
        # Verify payload
        assert stored_point.payload.get('conversation_id') == window.conversation_id
        assert stored_point.payload.get('conversation_type') == window.conversation_type
        assert stored_point.payload.get('summary_text') == window.summary_text
        
        # For metadata fields
        metadata = stored_point.payload.get('window_metadata', {})
        assert metadata.get('id') == str(window.id)
        assert 'date_range' in metadata
        assert 'unixtime_range' in metadata
        assert 'from_ids' in metadata
        assert 'event_ids' in metadata
    
    # Test vector similarity search works
    test_vector = test_windows[0].vector
    search_results = shared_client.search(
        collection_name=test_config.collection,
        query_vector=test_vector,
        limit=3
    )
    
    # Verify search returned results
    assert len(search_results) > 0
    
    # First result should be the exact match (itself)
    assert search_results[0].id == str(test_windows[0].id)
    assert search_results[0].score > 0.99  # Should be nearly 1.0 (exact match)

@pytest.mark.integration
def test_qdrant_store_batch_writes(test_config, shared_client, test_windows):
    """Test batch writing to Qdrant."""
    # Create transform with shared client
    transform = QdrantVectorStore(test_config, client=shared_client)
    
    # Test with Beam TestPipeline
    with TestPipeline() as p:
        # Create PCollection from test_windows
        windows_pcoll = p | beam.Create(test_windows)
        
        # Store windows in Qdrant
        ids_pcoll = windows_pcoll | beam.ParDo(transform)
        
        # Verify all IDs are returned
        assert_that(
            ids_pcoll,
            equal_to([str(window.id) for window in test_windows])
        )
    
    # Wait for vectors to be searchable in Qdrant
    time.sleep(1)
    
    # Verify all points were stored
    count_result = shared_client.count(
        collection_name=test_config.collection,
    )
    assert count_result.count == len(test_windows)

@pytest.mark.integration
def test_qdrant_store_update_points(test_config, shared_client):
    """Test updating existing points in Qdrant."""
    # Create transform with shared client
    transform = QdrantVectorStore(test_config, client=shared_client)
    
    # Create test window
    window_id = uuid.uuid4()
    window = Window(
        id=window_id,
        conversation_id='test-conv-update',
        conversation_type='test',
        summary_text='Original summary',
        vector=np.random.rand(768).tolist(),
        window_metadata=WindowMetadata(
            id=window_id,
            conversation_id='test-conv-update',
            conversation_type='test',
            date_range=['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
            unixtime_range=[1704067200, 1704070800],
            from_ids={'user1': 1},
            event_ids=['1']
        )
    )
    
    # First, store original window
    list(transform.process(window))
    transform.finish_bundle()  # Force flush
    
    # Wait for point to be searchable
    time.sleep(1)
    
    # Verify original point was stored
    print(f"Searching for window {window_id} in collection {test_config.collection}")
    search_result = shared_client.retrieve(
        collection_name=test_config.collection,
        ids=[str(window_id)],
        with_vectors=True
    )
    
    assert len(search_result) == 1, f"Expected 1 result, got {len(search_result)}"
    original = search_result[0]
    
    assert original.payload.get('summary_text') == 'Original summary'
    
    # Now update the window with new data
    updated_window = Window(
        id=window_id,  # Same ID
        conversation_id='test-conv-update',
        conversation_type='test',
        summary_text='Updated summary',  # Changed summary
        vector=np.random.rand(768).tolist(),  # New vector
        window_metadata=WindowMetadata(
            id=window_id,
            conversation_id='test-conv-update',
            conversation_type='test',
            date_range=['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
            unixtime_range=[1704067200, 1704070800],
            from_ids={'user1': 1, 'user2': 1},  # Added new user
            event_ids=['1', '2']  # Added new event
        )
    )
    
    # Store the updated window
    list(transform.process(updated_window))
    transform.finish_bundle()  # Force flush
    
    # Wait for point to be updated
    time.sleep(1)
    
    # Verify point was updated
    print(f"Searching for updated window {window_id} in collection {test_config.collection}")
    search_result = shared_client.retrieve(
        collection_name=test_config.collection,
        ids=[str(window_id)],
        with_vectors=True
    )
    
    assert len(search_result) == 1
    updated = search_result[0]
    
    assert updated.payload.get('summary_text') == 'Updated summary'
    assert len(updated.payload.get('window_metadata', {}).get('from_ids', {})) == 2
    assert len(updated.payload.get('window_metadata', {}).get('event_ids', [])) == 2

if __name__ == "__main__":
    pytest.main(["-v", __file__])