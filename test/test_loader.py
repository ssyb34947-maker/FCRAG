import sys
import os
import tempfile
import pytest
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loader_manager import DocumentLoaderManager
from chunking.splitter import TextSplitter
import yaml

# Create a temporary config for testing
test_config = {
    'chunking': {
        'strategy': 'sliding_token',
        'chunk_size': 128,
        'chunk_overlap': 32
    }
}

# Write test config to a temporary file
config_path = Path(__file__).parent.parent / 'config.yaml'
if not config_path.exists():
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)

class TestDocumentLoaderManager:
    """Test the DocumentLoaderManager class"""
    
    def setup_method(self):
        """Set up test resources"""
        self.loader_manager = DocumentLoaderManager()
        
        # Create temporary test files
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create a test text file with UTF-8 encoding
        self.test_text_file = self.test_dir / "test.txt"
        with open(self.test_text_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document.\nIt contains multiple lines of text.\nUsed to test document loader functionality.")
            
    def teardown_method(self):
        """Clean up test resources"""
        # Clean up temporary files
        if self.test_text_file.exists():
            self.test_text_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_load_document(self):
        """Test loading a single document"""
        print(f"\nLoading document from: {self.test_text_file}")
        documents = self.loader_manager.load_document(self.test_text_file, "test_domain")
        
        print(f"Loaded {len(documents)} documents")
        for i, doc in enumerate(documents):
            print(f"Document {i+1}:")
            print(f"  ID: {doc['id']}")
            print(f"  Domain: {doc['domain']}")
            print(f"  Source: {doc['source']}")
            print(f"  Text preview: {doc['text'][:50]}...")
            print(f"  Metadata: {doc['metadata']}")
        
        assert len(documents) == 1
        doc = documents[0]
        assert doc['domain'] == "test_domain"
        assert doc['source'] == str(self.test_text_file.absolute())
        assert 'This is a test document' in doc['text']
        assert 'metadata' in doc
        
    def test_load_directory(self):
        """Test loading documents from a directory"""
        print(f"\nLoading documents from directory: {self.test_dir}")
        documents = self.loader_manager.load_directory(self.test_dir, "test_domain")
        
        print(f"Loaded {len(documents)} documents from directory")
        for i, doc in enumerate(documents):
            print(f"Document {i+1}:")
            print(f"  ID: {doc['id']}")
            print(f"  Domain: {doc['domain']}")
            print(f"  Source: {doc['source']}")
            print(f"  Text preview: {doc['text'][:50]}...")
            print(f"  Metadata: {doc['metadata']}")
        
        assert len(documents) == 1
        doc = documents[0]
        assert doc['domain'] == "test_domain"
        assert doc['source'] == str(self.test_text_file.absolute())
        assert 'This is a test document' in doc['text']

if __name__ == "__main__":
    pytest.main([__file__])