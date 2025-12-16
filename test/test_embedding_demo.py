import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.bairen_embedder import BairenEmbedder
import yaml

# Load configuration
config_path = Path(__file__).parent.parent / 'config.yaml'

def test_embedding():
    """Test embedding functionality"""
    print("Testing Bairen Embedder (Multimodal)...")
    print("=" * 50)
    
    # Check if config file exists
    if not config_path.exists():
        print("Config file not found. Please create config.yaml with your API key.")
        return
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check if API key is configured
    api_key = config.get('embedding', {}).get('api_key')
    print(f"API key: {api_key}")
    model_name = config.get('embedding', {}).get('model_name')
    if not api_key or api_key == 'your_bailian_api_key_here':
        print("API key not configured in config.yaml")
        print("Please set your actual API key in config.yaml")
        return
    
    print(f"Using model: {model_name}")
    
    # Initialize embedder
    try:
        embedder = BairenEmbedder()
        print(f"Embedder initialized successfully")
        print(f"Model: {embedder.model_name}")
        print(f"Batch size: {embedder.batch_size}")
        print()
    except Exception as e:
        print(f"Failed to initialize embedder: {e}")
        return
    
    # Test texts
    test_texts = [
        "人工智能是计算机科学的一个重要分支。",
        "机器学习是实现人工智能的一种方法。",
        "深度学习是机器学习的一个子领域。",
        "自然语言处理是人工智能的关键技术之一。",
        "计算机视觉让机器能够理解和分析图像。"
    ]
    
    print("Input texts:")
    for i, text in enumerate(test_texts):
        print(f"{i+1}. {text}")
    print()
    
    # Generate embeddings
    try:
        print("Generating embeddings...")
        embeddings = embedder.embed(test_texts)
        
        print(f"Successfully generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        print()
        
        # Show first few values of first embedding
        if embeddings:
            print("First embedding (first 10 values):")
            print([round(val, 6) for val in embeddings[0][:10]])
            print()
            
        # Test embedding similarity
        if len(embeddings) >= 2:
            # Calculate cosine similarity between first two embeddings
            import numpy as np
            
            # Normalize vectors
            vec1 = np.array(embeddings[0])
            vec2 = np.array(embeddings[1])
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            print("Similarity test:")
            print(f"Cosine similarity between texts 1 and 2: {similarity:.6f}")
            print(f"Text 1: {test_texts[0]}")
            print(f"Text 2: {test_texts[1]}")
            
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("This might be due to:")
        print("1. Incorrect API key in config.yaml")
        print("2. Network connectivity issues")
        print("3. API service availability")
        print("4. Incorrect model name or API endpoint")
        import traceback
        traceback.print_exc()

def test_single_text():
    """Test embedding a single text"""
    print("\n\nTesting single text embedding...")
    print("=" * 50)
    
    # Check if config file exists
    if not config_path.exists():
        print("Config file not found. Please create config.yaml with your API key.")
        return
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check if API key is configured
    api_key = config.get('embedding', {}).get('api_key')
    if not api_key or api_key == 'your_bailian_api_key_here':
        print("API key not configured in config.yaml")
        print("Please set your actual API key in config.yaml")
        return
    
    try:
        embedder = BairenEmbedder()
        
        single_text = "这是一个测试句子，用于验证嵌入功能是否正常工作。"
        print(f"Input text: {single_text}")
        
        embeddings = embedder.embed([single_text])
        
        print(f"Embedding generated successfully")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print("First 10 values:", [round(val, 6) for val in embeddings[0][:10]])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n\nTesting batch processing...")
    print("=" * 50)
    
    # Check if config file exists
    if not config_path.exists():
        print("Config file not found. Please create config.yaml with your API key.")
        return
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check if API key is configured
    api_key = config.get('embedding', {}).get('api_key')
    if not api_key or api_key == 'your_bailian_api_key_here':
        print("API key not configured in config.yaml")
        print("Please set your actual API key in config.yaml")
        return
    
    try:
        embedder = BairenEmbedder()
        
        # Create a larger batch of texts
        batch_texts = [
            f"这是批处理测试句子 {i+1}，用来验证批处理功能是否正常工作。" 
            for i in range(5)
        ]
        
        print(f"Processing batch of {len(batch_texts)} texts")
        for i, text in enumerate(batch_texts):
            print(f"{i+1}. {text}")
        
        embeddings = embedder.embed(batch_texts)
        
        print(f"\nSuccessfully processed batch")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Verify all embeddings are normalized (L2 norm should be 1.0)
        import numpy as np
        norms = [np.linalg.norm(emb) for emb in embeddings]
        print(f"L2 norms (should be ~1.0): {[round(norm, 6) for norm in norms[:3]]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding()
    print("\n"+'==='*50+"\n")
    test_single_text()
    print("\n'"+'==='*50+"\n")
    test_batch_processing()
    print("\n"+'==='*50+"\n")
    print("\n\nEmbedding tests completed!")