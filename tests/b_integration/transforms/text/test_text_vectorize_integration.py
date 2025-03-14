"""Integration tests for OllamaVectorize with live services."""
import pytest
import uuid
import yaml
import numpy as np

from telexp.transforms.text.vectorize import OllamaVectorize
from telexp.schema import Window, WindowMetadata
from telexp.config import Config
from telexp.services.embedding import EmbeddingServiceManager

# Mark all tests as integration tests
pytestmark = pytest.mark.integration

def load_config():
    """Load configuration from conf.yaml."""
    with open('conf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config)

@pytest.fixture
def test_config():
    """Load the actual config to use real services."""
    config = load_config().text_vectorizer
    # Make sure we're using the right field name 
    config.input_field = "summary_text"
    # The model name should match the name in embedding_services, not the model field
    config.model = "nomic-embed-text"
    return config

@pytest.fixture
def test_service_manager():
    """Create real service manager with actual connections."""
    config = load_config()
    manager = EmbeddingServiceManager(config.embedding_services)
    manager.setup()
    return manager

@pytest.fixture
def test_window():
    """Create test window with sample text."""
    window_id = uuid.UUID('12345678-1234-1234-1234-123456789012')
    return Window(
        id=window_id,
        conversation_id='test-conv-1',
        conversation_type='test',
        summary_text='This is a test summary of a conversation about machine learning and AI technologies. ' + 
                    'The participants discussed the recent advancements in large language models and their ' +
                    'applications in various domains such as healthcare, finance, and education.',
        vector=None,  # Will be populated by the vectorizer
        window_metadata=WindowMetadata(
            id=window_id,
            conversation_id='test-conv-1',
            conversation_type='test',
            date_range=['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
            unixtime_range=[1704067200, 1704070800],
            from_ids={'user1': 1, 'user2': 1},
            event_ids=['1', '2']
        )
    )

@pytest.mark.integration
def test_text_vectorization_integration(test_config, test_service_manager, test_window):
    """Test actual text vectorization with live service."""
    transform = OllamaVectorize(test_config, test_service_manager)
    transform.setup()
    
    try:
        # Process the window through the transform
        result = list(transform.process(test_window))
        
        # Validate results
        assert len(result) == 1
        processed_window = result[0]
        
        # Validate that vector is populated
        assert processed_window.vector is not None
        assert len(processed_window.vector) == test_config.vector_size
        
        # Check vector quality - should not be all zeros or all ones
        vector = np.array(processed_window.vector)
        assert not np.allclose(vector, 0)  # Not all zeros
        assert not np.allclose(vector, 1)  # Not all ones
        assert np.std(vector) > 0.01  # Has some variation
        
        # Print vector statistics for inspection
        print(f"\nVector dimensions: {len(processed_window.vector)}")
        print(f"Vector mean: {np.mean(vector)}")
        print(f"Vector std: {np.std(vector)}")
        print(f"Vector min: {np.min(vector)}")
        print(f"Vector max: {np.max(vector)}")
        print(f"First 5 dimensions: {vector[:5]}")
        
    finally:
        # Clean up resources
        transform.teardown()

@pytest.mark.integration
def test_similar_texts_similar_vectors(test_config, test_service_manager):
    """Test that similar texts produce similar vectors."""
    transform = OllamaVectorize(test_config, test_service_manager)
    transform.setup()
    
    try:
        # Create two similar windows
        window_id1 = uuid.UUID('12345678-1234-1234-1234-123456789012')
        window_id2 = uuid.UUID('12345678-1234-1234-1234-123456789013')
        
        window1 = Window(
            id=window_id1,
            conversation_id='test-conv-1',
            conversation_type='test',
            summary_text='A discussion about Python programming language and its applications in data science.',
            vector=None,
            window_metadata=WindowMetadata(
                id=window_id1,
                conversation_id='test-conv-1',
                conversation_type='test',
                date_range=['2024-01-01T00:00:00', '2024-01-01T01:00:00'],
                unixtime_range=[1704067200, 1704070800],
                from_ids={'user1': 1},
                event_ids=['1']
            )
        )
        
        window2 = Window(
            id=window_id2,
            conversation_id='test-conv-2',
            conversation_type='test',
            summary_text='People talking about Python programming and how it is used for data analysis projects.',
            vector=None,
            window_metadata=WindowMetadata(
                id=window_id2,
                conversation_id='test-conv-2',
                conversation_type='test',
                date_range=['2024-01-02T00:00:00', '2024-01-02T01:00:00'],
                unixtime_range=[1704153600, 1704157200],
                from_ids={'user2': 1},
                event_ids=['2']
            )
        )
        
        # Create a dissimilar window
        window_id3 = uuid.UUID('12345678-1234-1234-1234-123456789014')
        window3 = Window(
            id=window_id3,
            conversation_id='test-conv-3',
            conversation_type='test',
            summary_text='A recipe for making chocolate chip cookies with butter, flour, and sugar.',
            vector=None,
            window_metadata=WindowMetadata(
                id=window_id3,
                conversation_id='test-conv-3',
                conversation_type='test',
                date_range=['2024-01-03T00:00:00', '2024-01-03T01:00:00'],
                unixtime_range=[1704240000, 1704243600],
                from_ids={'user3': 1},
                event_ids=['3']
            )
        )
        
        # Vectorize all windows
        result1 = list(transform.process(window1))[0]
        result2 = list(transform.process(window2))[0]
        result3 = list(transform.process(window3))[0]
        
        # Calculate cosine similarity
        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            return dot_product / (norm1 * norm2)
        
        # Calculate similarities
        sim_1_2 = cosine_similarity(result1.vector, result2.vector)
        sim_1_3 = cosine_similarity(result1.vector, result3.vector)
        sim_2_3 = cosine_similarity(result2.vector, result3.vector)
        
        print(f"\nSimilarity between similar texts (Python discussions): {sim_1_2:.4f}")
        print(f"Similarity between dissimilar texts (Python vs Cookies): {sim_1_3:.4f}")
        print(f"Similarity between dissimilar texts (Python vs Cookies): {sim_2_3:.4f}")
        
        # Similar texts should have higher similarity than dissimilar texts
        assert sim_1_2 > sim_1_3
        assert sim_1_2 > sim_2_3
        
    finally:
        # Clean up resources
        transform.teardown()

if __name__ == "__main__":
    pytest.main(["-v", __file__])