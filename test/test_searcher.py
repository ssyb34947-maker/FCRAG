import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from retrieval.searcher import Searcher
from unittest.mock import patch, MagicMock

# Create a temporary config for testing
test_config = {
    'retrieval': {
        'top_k': 5,
        'domain_filter': True
    },
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

@patch('retrieval.searcher.BairenEmbedder')
@patch('retrieval.searcher.MilvusClient')
def test_search(mock_milvus_client, mock_embedder):
    """Test search functionality"""
    # Mock the embedder
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_embedder.return_value = mock_embedder_instance
    
    # Mock the Milvus client
    mock_milvus_instance = MagicMock()
    mock_milvus_instance.search.return_value = [
        {
            'id': 1,
            'distance': 0.85,
            'domain': 'test',
            'content': 'This is a test result',
            'metadata': {}
        }
    ]
    mock_milvus_client.return_value = mock_milvus_instance
    
    searcher = Searcher()
    
    results = searcher.search("测试查询", "test_domain")
    
    # Check that embedder was called
    mock_embedder_instance.embed.assert_called_once_with(["测试查询"])
    
    # Check that Milvus client was called
    mock_milvus_instance.search.assert_called_once()
    
    # Check results
    assert len(results) == 1
    assert results[0]['content'] == 'This is a test result'

# Clean up temporary config file if it didn't exist before
if not original_config_exists and config_path.exists():
    config_path.unlink()

if __name__ == "__main__":
    test_search()
    print("All searcher tests passed!")