import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from chunking.splitter import TextSplitter

# Create a sample document for testing
sample_document = {
    'id': 'test_doc_001',
    'domain': 'economics',
    'source': 'test_source',
    'text': '''经济学是社会科学的一个分支，研究人类行为和资源分配。它是关于人们如何做出选择的学科，尤其是在资源稀缺的情况下。

在现代经济体系中，存在多种不同的经济模型。市场经济是一种由供需关系决定资源配置的系统。在这样的系统中，价格机制发挥着重要作用，它可以帮助调节商品和服务的分配。

计划经济是另一种经济体系，在这种体系中，政府或中央机构负责制定生产和分配决策。这种体系在20世纪的一些国家中得到了广泛应用。

混合经济结合了市场经济和计划经济的特点，试图利用两者的优点。大多数现代发达国家都采用了某种形式的混合经济。

微观经济学关注个体经济单位的行为，如家庭、企业等。它研究消费者如何做出购买决策，企业如何决定生产什么以及如何生产。

宏观经济学则着眼于整个经济体，研究国民生产总值、通货膨胀、失业率等宏观经济指标。它关注经济增长、经济周期等问题。

国际贸易是经济学的重要分支，研究不同国家之间的商品和服务交换。比较优势理论是这一领域的基础理论之一。

货币政策和财政政策是政府调控经济的两大工具。货币政策主要通过调节货币供应量来影响经济活动，而财政政策则通过政府支出和税收来发挥作用。''',
    'metadata': {
        'author': 'test_author',
        'created_date': '2025-01-01'
    }
}

def test_chunking_strategies():
    """Test different chunking strategies"""
    
    # Test configurations for different strategies
    configs = [
        {
            'name': 'Sliding Token Chunking',
            'chunking': {
                'strategy': 'sliding_token',
                'chunk_size': 128,
                'chunk_overlap': 32
            }
        },
        {
            'name': 'Sentence Chunking',
            'chunking': {
                'strategy': 'sentence',
                'chunk_size': 128,
                'chunk_overlap': 32
            }
        },
        {
            'name': 'Paragraph Chunking',
            'chunking': {
                'strategy': 'paragraph',
                'chunk_size': 128,
                'chunk_overlap': 32
            }
        },
        {
            'name': 'Hybrid Chunking',
            'chunking': {
                'strategy': 'hybrid',
                'chunk_size': 128,
                'chunk_overlap': 32
            }
        }
    ]
    
    print("Testing different chunking strategies...\n")
    print("=" * 80)
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 50)
        
        # Create splitter with config
        splitter = TextSplitter(config['chunking'])
        
        # Split document
        chunks = splitter.split(sample_document)
        
        print(f"Total chunks generated: {len(chunks)}")
        
        # Display chunks
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"Text: {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")
            print(f"Metadata: {chunk['metadata']}")
            
        print("-" * 50)

def test_different_sizes():
    """Test chunking with different sizes"""
    print("\n\nTesting different chunk sizes with sliding token strategy...")
    print("=" * 80)
    
    sizes = [64, 128, 256]
    
    for size in sizes:
        print(f"\nChunk size: {size}")
        print("-" * 30)
        
        config = {
            'strategy': 'sliding_token',
            'chunk_size': size,
            'chunk_overlap': size // 4  # 25% overlap
        }
        
        splitter = TextSplitter(config)
        chunks = splitter.split(sample_document)
        
        print(f"Chunks generated: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1} (length: {len(chunk['text'])} chars):")
            print(f"Text: {chunk['text'][:150]}{'...' if len(chunk['text']) > 150 else ''}")

if __name__ == "__main__":
    test_chunking_strategies()
    test_different_sizes()
    print("\n\nChunking tests completed!")