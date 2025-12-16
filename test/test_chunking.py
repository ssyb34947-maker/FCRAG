import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from chunking.splitter import TextSplitter

# Create a temporary config for testing
test_config = {
    'chunking': {
        'strategy': 'sliding_token',
        'chunk_size': 128,
        'chunk_overlap': 32
    },
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

def test_sliding_token_split():
    """Test sliding token splitting strategy"""
    splitter = TextSplitter(test_config['chunking'])
    
    # Create a test document
    document = {
        'text': ' '.join(['word'] * 200),  # 200 words
        'domain': 'test',
        'metadata': {}
    }
    
    chunks = splitter.split(document)
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all(len(chunk['text']) > 0 for chunk in chunks)

def test_sentence_split():
    """Test sentence splitting strategy"""
    config = test_config['chunking'].copy()
    config['strategy'] = 'sentence'
    splitter = TextSplitter(config)
    
    # Create a test document with sentences
    document = {
        'text': 'This is the first sentence. This is the second sentence! Is this the third sentence?'
                '这是第四句。这是第五句！这是第六句？',
        'domain': 'test',
        'metadata': {}
    }
    
    chunks = splitter.split(document)
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)

def test_paragraph_split():
    """Test paragraph splitting strategy"""
    config = test_config['chunking'].copy()
    config['strategy'] = 'paragraph'
    splitter = TextSplitter(config)
    
    # Create a test document with paragraphs
    document = {
        'text': 'This is the first paragraph.\n\nThis is the second paragraph.\n\n这是第三个段落。\n\n这是第四个段落。',
        'domain': 'test',
        'metadata': {}
    }
    
    chunks = splitter.split(document)
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)

def test_hybrid_split():
    """Test hybrid splitting strategy"""
    config = test_config['chunking'].copy()
    config['strategy'] = 'hybrid'
    splitter = TextSplitter(config)
    
    # Create a test document
    document = {
        'text': 'First paragraph with multiple sentences. Second sentence in first paragraph!\n\n'
                'Second paragraph also with multiple sentences. Another sentence in second paragraph?\n\n'
                '第三段也有多个句子。第三段的另一个句子！',
        'domain': 'test',
        'metadata': {}
    }
    
    chunks = splitter.split(document)
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)

# Clean up temporary config file if it didn't exist before
if not original_config_exists and config_path.exists():
    config_path.unlink()

if __name__ == "__main__":
    test_sliding_token_split()
    test_sentence_split()
    test_paragraph_split()
    test_hybrid_split()
    print("All chunking tests passed!")