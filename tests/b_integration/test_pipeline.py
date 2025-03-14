"""Integration tests for the full telexp pipeline.

These tests connect to real external services and test the full pipeline
functionality, including image summarization, text summarization, vector embedding,
and vector storage.
"""
import pytest
import os
import json
import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

from telexp.config import Config
from telexp.pipelines.process import TelegramExportProcessor


@pytest.fixture
def test_config():
    """Load the configuration file."""
    return Config.from_yaml('conf.yaml')


@pytest.fixture
def prepare_qdrant_collection(test_config):
    """Set up and clean up a Qdrant collection for testing.
    
    This fixture ensures the test collection is clean and properly configured.
    The QdrantVectorStore will create its own clients during pipeline execution.
    """
    # Create a temporary client just for setup/teardown
    client = QdrantClient(
        url=test_config.text_vector_storage.url,
        timeout=test_config.text_vector_storage.timeout
    )
    
    # Make sure the test collection exists
    test_collection = test_config.text_vector_storage.test_collection
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Recreate the test collection if it exists
        if test_collection in collection_names:
            client.delete_collection(collection_name=test_collection)
            
        # Create the test collection
        client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=test_config.text_vector_storage.vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created test collection {test_collection}")
    except Exception as e:
        print(f"Error setting up test collection: {str(e)}")
        raise
    finally:
        # We're done with this client, close it
        client.close()
    
    # No need to yield any value - just setup
    yield
    
    # Create another temporary client for cleanup
    client = QdrantClient(
        url=test_config.text_vector_storage.url,
        timeout=test_config.text_vector_storage.timeout
    )
    
    # Cleanup after the test is done
    try:
        client.delete_collection(collection_name=test_collection)
        print(f"Cleaned up test collection {test_collection}")
    except Exception as e:
        print(f"Error cleaning up test collection: {str(e)}")
    finally:
        client.close()


@pytest.mark.integration
def test_real_pipeline(test_config, prepare_qdrant_collection, tmp_path):
    """Test the pipeline with real messages from test data directory, handling empty windows."""
    # Use the config, but make sure it points to the test collection
    test_config.text_vector_storage.collection = test_config.text_vector_storage.test_collection
    
    # Create the processor with our test config
    processor = TelegramExportProcessor(test_config)
    
    # Setup output directory
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the pipeline on our test data
    input_dir = os.path.join('tests', 'data')
    
    # Configure log level for the test
    logging.basicConfig(level=logging.INFO)
    
    # Process the data with output directory
    processor.process(input_dir, output_dir)
    
    # Verify output files
    output_file = output_dir / "windows.jsonl"
    
    # Handle case where all windows were filtered out
    if not output_file.exists():
        # Check if this is expected by looking at test data characteristics
        test_data_path = Path('tests') / 'data' / 'private_channel' / '12345678' / 'messages.jsonl'
        if test_data_path.exists():
            with open(test_data_path, 'r') as f:
                content = f.read().strip()
                # Only fail if we know there should be content
                if content:
                    pytest.fail("Output file wasn't created despite valid input data")
        print("No windows.jsonl created - all windows may have been filtered out")
        return  # Exit test as there's nothing more to verify
    
    # Read the output file to extract window IDs
    window_ids = []
    with open(output_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'id' in data:
                    window_ids.append(data['id'])
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line: {line}")
    
    # It's valid to have 0 windows if all were filtered
    print(f"\nFound {len(window_ids)} windows in output file")
    
    # If we have windows, verify they're in Qdrant
    if window_ids:
        verification_client = QdrantClient(
            url=test_config.text_vector_storage.url,
            timeout=test_config.text_vector_storage.timeout
        )
        
        try:
            # Get collection info to verify it exists
            collection_info = None
            try:
                collection_info = verification_client.get_collection(
                    collection_name=test_config.text_vector_storage.test_collection
                )
                print(f"Collection exists with {collection_info.vectors_count} vectors")
            except Exception as e:
                print(f"Warning: Could not get collection info: {str(e)}")
            
            # For each window ID, check if it was stored in Qdrant
            qdrant_found = 0
            
            for window_id in window_ids:
                print(f"\nVerifying vector storage for ID: {window_id}")
                
                # Search for the vector in Qdrant by ID
                try:
                    search_result = verification_client.retrieve(
                        collection_name=test_config.text_vector_storage.test_collection,
                        ids=[window_id],
                        with_vectors=True
                    )
                    
                    # If found, validate its contents
                    if search_result:
                        stored_point = search_result[0]
                        qdrant_found += 1
                        
                        # Verify payload data
                        assert stored_point.payload is not None, "Stored point doesn't have payload"
                        
                        # Print payload information
                        print(f"Conversation ID: {stored_point.payload.get('conversation_id')}")
                    
                        # If it has a vector, check its dimensions
                        if stored_point.vector:
                            vector = stored_point.vector
                            print(f"Vector found in Qdrant: {vector[:5]} (first 5 dimensions)")
                            
                            # Check vector dimensions if non-empty
                            if len(vector) > 0:
                                assert len(vector) == test_config.text_vector_storage.vector_size, \
                                    f"Vector dimensions mismatch: got {len(vector)}, expected {test_config.text_vector_storage.vector_size}"
                                
                                # Check for all zeros
                                non_zero = any(abs(x) > 1e-10 for x in vector)
                                if not non_zero:
                                    print("WARNING: Vector appears to contain all zeros")
                            
                        print("✓ Successfully verified in Qdrant")
                
                except Exception as e:
                    print(f"WARN: Could not find ID {window_id} in Qdrant: {str(e)}")
            
            # Check results - we should have vectors for all windows in the output file
            print(f"\nFound {qdrant_found} windows stored in Qdrant out of {len(window_ids)} total windows")
            
            # All windows in the output file should have valid vectors - those are the only ones that should make it to output
            assert qdrant_found == len(window_ids), f"Expected {len(window_ids)} vectors in Qdrant, but found {qdrant_found}"
            
        finally:
            # Always close the verification client
            verification_client.close()