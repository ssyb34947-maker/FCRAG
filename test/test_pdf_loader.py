import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loader_manager import DocumentLoaderManager
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

def test_pdf_loading():
    """Test loading a PDF document"""
    loader_manager = DocumentLoaderManager()
    
    # Path to the test PDF file
    test_pdf_path = Path(__file__).parent / "data/test.pdf"
    
    # Check if the test PDF file exists
    if not test_pdf_path.exists():
        print(f"Test PDF file not found at {test_pdf_path}")
        print("Please place a test PDF file named 'test.pdf' in the test directory")
        return
    
    print(f"Loading PDF document from: {test_pdf_path}")
    
    try:
        # Load the PDF document
        documents = loader_manager.load_document(test_pdf_path, "test_pdf_domain")
        
        print(f"Successfully loaded {len(documents)} documents from PDF")
        
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"  ID: {doc['id']}")
            print(f"  Domain: {doc['domain']}")
            print(f"  Source: {doc['source']}")
            print(f"  Text preview: {doc['text'][:200]}...")
            print(f"  Metadata keys: {list(doc['metadata'].keys())}")
            
        print("\nPDF loading test completed successfully!")
        
    except Exception as e:
        print(f"Error loading PDF document: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_loading()