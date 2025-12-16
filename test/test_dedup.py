import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from utils.dedup import Deduplicator

# Create a temporary config for testing
test_config = {
    'dedup': {
        'strategy': 'md5',
        'md5_threshold': 1.0,
        'simhash_threshold': 3,
        'embedding_threshold': 0.95
    }
}

# Write test config to a temporary file
config_path = Path(__file__).parent.parent / 'config.yaml'
original_config_exists = config_path.exists()

if not original_config_exists:
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)

def test_md5_dedup():
    """Test MD5 deduplication"""
    config = test_config['dedup'].copy()
    config['strategy'] = 'md5'
    deduplicator = Deduplicator(config)
    
    doc1 = {'text': 'This is a test document'}
    doc2 = {'text': 'This is a test document'}  # Same content
    doc3 = {'text': 'This is another test document'}  # Different content
    
    # First document should not be duplicate
    assert not deduplicator.is_duplicate(doc1)
    
    # Second document with same content should be duplicate
    assert deduplicator.is_duplicate(doc2)
    
    # Third document with different content should not be duplicate
    assert not deduplicator.is_duplicate(doc3)

def test_embedding_dedup():
    """Test embedding deduplication"""
    config = test_config['dedup'].copy()
    config['strategy'] = 'embedding'
    deduplicator = Deduplicator(config)
    
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.99, 0.0, 0.0]  # Very similar to emb1
    emb3 = [0.0, 1.0, 0.0]   # Different from emb1
    
    # First embedding should not be duplicate
    assert not deduplicator.is_duplicate({'text': 'doc1'}, emb1)
    
    # Second embedding very similar to first should be duplicate
    assert deduplicator.is_duplicate({'text': 'doc2'}, emb2)
    
    # Third embedding different from first should not be duplicate
    assert not deduplicator.is_duplicate({'text': 'doc3'}, emb3)

# Clean up temporary config file if it didn't exist before
if not original_config_exists and config_path.exists():
    config_path.unlink()

if __name__ == "__main__":
    test_md5_dedup()
    test_embedding_dedup()
    print("All deduplication tests passed!")