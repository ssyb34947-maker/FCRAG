import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from utils.tokenizer import Tokenizer

def test_tokenizer():
    """Test tokenizer functionality"""
    # Test basic tokenization
    text = "这是一个测试句子"
    tokens = Tokenizer.tokenize(text)
    
    assert len(tokens) > 0
    assert isinstance(tokens, list)
    
    # Test token counting
    count = Tokenizer.count_tokens(text)
    assert count == len(tokens)
    assert isinstance(count, int)
    assert count > 0

if __name__ == "__main__":
    test_tokenizer()
    print("All tokenizer tests passed!")