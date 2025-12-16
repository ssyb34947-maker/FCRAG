import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from embeddings.bairen_embedder import BairenEmbedder
from unittest.mock import patch, MagicMock

# Create a temporary config for testing
test_config = {
    'embedding': {
        'api_key': 'test_api_key',
        'model_name': 'test_model',
        'batch_size': 2,
        'timeout': 30,
        'max_retries': 3
    }
}

# Write test config to a temporary file
config_path = Path(__file__).parent.parent / 'config.yaml'
original_config_exists = config_path.exists()

if not original_config_exists:
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)

def test_l2_normalize():
    """Test L2 normalization function"""
    embedder = BairenEmbedder()
    
    # Test normalization
    vector = [3.0, 4.0]  # Length 5 vector
    normalized = embedder._l2_normalize(vector)
    
    # Check that the normalized vector has unit length
    length = sum(x*x for x in normalized) ** 0.5
    assert abs(length - 1.0) < 1e-6
    
    # Check that the normalized vector is in the same direction
    assert abs(normalized[0] - 0.6) < 1e-6  # 3/5
    assert abs(normalized[1] - 0.8) < 1e-6  # 4/5

@patch('embeddings.bairen_embedder.requests.post')
def test_embed(mock_post):
    """Test embedding function"""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'data': [
            {'embedding': [0.1, 0.2, 0.3]},
            {'embedding': [0.4, 0.5, 0.6]}
        ]
    }
    mock_post.return_value = mock_response
    
    embedder = BairenEmbedder()
    
    texts = ["测试文本1", "测试文本2"]
    embeddings = embedder.embed(texts)
    
    # Check that we got the expected number of embeddings
    assert len(embeddings) == 2
    
    # Check that embeddings are normalized
    for emb in embeddings:
        length = sum(x*x for x in emb) ** 0.5
        assert abs(length - 1.0) < 1e-6

@patch('embeddings.bairen_embedder.requests.post')
def test_embed_with_retry(mock_post):
    """Test embedding function with retry logic"""
    # Mock the API to fail the first time and succeed the second time
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'data': [
            {'embedding': [0.1, 0.2, 0.3]}
        ]
    }
    
    mock_post.side_effect = [Exception("API Error"), mock_response]
    
    embedder = BairenEmbedder()
    
    texts = ["测试文本"]
    embeddings = embedder.embed(texts)
    
    # Check that we got the expected embedding
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 3

# Clean up temporary config file if it didn't exist before
if not original_config_exists and config_path.exists():
    config_path.unlink()

if __name__ == "__main__":
    test_l2_normalize()
    test_embed()
    test_embed_with_retry()
    print("All embedding tests passed!")