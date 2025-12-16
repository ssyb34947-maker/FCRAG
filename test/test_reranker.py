import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from retrieval.reranker import Reranker

# Create a temporary config for testing
test_config = {
    'reranker': {
        'mode': 'embedding_bm25_mixed',
        'weights': {
            'ann': 0.7,
            'bm25': 0.3
        }
    }
}

# Write test config to a temporary file
config_path = Path(__file__).parent.parent / 'config.yaml'
original_config_exists = config_path.exists()

if not original_config_exists:
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)

def test_bm25_rerank():
    """Test BM25 reranking"""
    config = test_config.copy()
    config['reranker']['mode'] = 'cross_encoder'  # Actually uses BM25 simulation
    reranker = Reranker()
    
    query = "什么是人工智能"
    candidates = [
        {
            'content': '人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。',
            'distance': 0.85
        },
        {
            'content': '机器学习是人工智能的一个子集，它使计算机能够从数据中学习并做出决策或预测。',
            'distance': 0.75
        }
    ]
    
    reranked = reranker.rerank(query, candidates)
    
    assert len(reranked) == len(candidates)
    # Check that final scores are added
    assert all('final_score' in candidate for candidate in reranked)

def test_mixed_rerank():
    """Test mixed reranking"""
    reranker = Reranker()
    
    query = "什么是人工智能"
    candidates = [
        {
            'content': '人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。',
            'distance': 0.85
        },
        {
            'content': '机器学习是人工智能的一个子集，它使计算机能够从数据中学习并做出决策或预测。',
            'distance': 0.75
        }
    ]
    
    reranked = reranker.rerank(query, candidates)
    
    assert len(reranked) == len(candidates)
    # Check that final scores are added
    assert all('final_score' in candidate for candidate in reranked)

# Clean up temporary config file if it didn't exist before
if not original_config_exists and config_path.exists():
    config_path.unlink()

if __name__ == "__main__":
    test_bm25_rerank()
    test_mixed_rerank()
    print("All reranker tests passed!")