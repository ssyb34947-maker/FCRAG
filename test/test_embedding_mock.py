import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.bairen_embedder import BairenEmbedder
import yaml

def test_embedding_with_mock():
    """Test embedding functionality with mocked API calls"""
    print("Testing Bairen Embedder with mocked API...")
    print("=" * 50)
    
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
    
    try:
        # Mock the requests.post function
        with patch('embeddings.bairen_embedder.requests.post') as mock_post:
            # Configure the mock to return a predefined response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'data': [
                    {'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]},
                    {'embedding': [0.2, 0.3, 0.4, 0.5, 0.6]}
                ]
            }
            mock_post.return_value = mock_response
            
            # Initialize embedder
            embedder = BairenEmbedder()
            print(f"Embedder initialized successfully")
            print(f"Model: {embedder.model_name}")
            print(f"Batch size: {embedder.batch_size}")
            print()
            
            # Test texts
            test_texts = [
                "这是第一个测试句子。",
                "这是第二个测试句子。"
            ]
            
            print("Input texts:")
            for i, text in enumerate(test_texts):
                print(f"{i+1}. {text}")
            print()
            
            # Generate embeddings
            print("Generating embeddings (mocked)...")
            embeddings = embedder.embed(test_texts)
            
            print(f"Successfully generated {len(embeddings)} embeddings")
            print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
            print()
            
            # Show embeddings
            for i, emb in enumerate(embeddings):
                print(f"Embedding {i+1}: {[round(val, 6) for val in emb]}")
            
            # Verify L2 normalization
            for i, emb in enumerate(embeddings):
                norm = np.linalg.norm(emb)
                print(f"L2 norm of embedding {i+1}: {norm:.6f}")
            
            # Verify that requests.post was called
            print(f"\nMock verification:")
            print(f"API calls made: {mock_post.call_count}")
            if mock_post.call_count > 0:
                print("API call successful!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary config file if it didn't exist before
        if not original_config_exists and config_path.exists():
            config_path.unlink()

def test_embedding_batching():
    """Test batching functionality with mock"""
    print("\n\nTesting batch processing with mocked API...")
    print("=" * 50)
    
    # Create a temporary config for testing
    test_config = {
        'embedding': {
            'api_key': 'test_api_key',
            'model_name': 'test_model',
            'batch_size': 3,
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
    
    try:
        # Mock the requests.post function
        with patch('embeddings.bairen_embedder.requests.post') as mock_post:
            # Configure the mock to return a predefined response
            def mock_response_func(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.status_code = 200
                # Return different embeddings based on the number of inputs
                json_data = kwargs.get('json', {})
                input_texts = json_data.get('input', [])
                mock_response.json.return_value = {
                    'data': [
                        {'embedding': [0.1*i, 0.2*i, 0.3*i, 0.4*i, 0.5*i] for i in range(1, len(input_texts)+1)}
                    ]
                }
                return mock_response
            
            mock_post.side_effect = mock_response_func
            
            # Initialize embedder
            embedder = BairenEmbedder()
            print(f"Embedder initialized successfully")
            print(f"Batch size: {embedder.batch_size}")
            print()
            
            # Create more texts than batch size
            test_texts = [
                f"测试句子 {i+1} 用于验证批处理功能。" 
                for i in range(7)  # 7 texts, batch size is 3, so should make 3 API calls
            ]
            
            print(f"Processing {len(test_texts)} texts with batch size {embedder.batch_size}")
            print("Input texts:")
            for i, text in enumerate(test_texts):
                print(f"{i+1}. {text}")
            print()
            
            # Generate embeddings
            print("Generating embeddings (mocked)...")
            embeddings = embedder.embed(test_texts)
            
            print(f"Successfully generated {len(embeddings)} embeddings")
            print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
            print()
            
            # Verify batching - should have made 3 API calls (7 texts with batch size 3)
            expected_calls = 3  # ceil(7/3) = 3
            print(f"Expected API calls: {expected_calls}")
            print(f"Actual API calls: {mock_post.call_count}")
            
            if mock_post.call_count == expected_calls:
                print("Batching works correctly!")
            else:
                print("Batching issue detected!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary config file if it didn't exist before
        if not original_config_exists and config_path.exists():
            config_path.unlink()

if __name__ == "__main__":
    test_embedding_with_mock()
    test_embedding_batching()
    print("\n\nMock embedding tests completed!")