import argparse
import yaml
import os
from typing import List, Dict, Any
from loaders.loader_manager import DocumentLoaderManager
from chunking.splitter import TextSplitter
from embeddings.bairen_embedder import BairenEmbedder
from storage.milvus_client import MilvusClient
from retrieval.searcher import Searcher
from retrieval.reranker import Reranker
from utils.dedup import Deduplicator
from utils.llm_processor import LLMProcessor
from utils.logger import get_logger

logger = get_logger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class RAGEngine:
    """RAG Engine encapsulating the full pipeline."""
    
    def __init__(self):
        """Initialize RAG engine components."""
        self.loader_manager = DocumentLoaderManager()
        self.text_splitter = TextSplitter(config['chunking'])
        self.embedder = BairenEmbedder()
        self.milvus_client = MilvusClient()
        self.searcher = Searcher()
        self.reranker = Reranker()
        self.deduplicator = Deduplicator(config['dedup'])
        self.llm_processor = LLMProcessor()
        
        logger.info("RAG Engine initialized")
    
    def ingest(self, path: str, domain: str = ""):
        """
        Ingest documents from path into the vector store.
        
        Args:
            path (str): Path to file or directory
            domain (str): Domain for the documents
        """
        logger.info(f"Starting ingestion from {path} with domain {domain}")
        
        # Load documents
        if os.path.isfile(path):
            documents = self.loader_manager.load_document(path, domain)
        elif os.path.isdir(path):
            documents = self.loader_manager.load_directory(path, domain)
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")
            
        # Process each document
        ingested_count = 0
        for document in documents:
            # Split document into chunks
            chunks = self.text_splitter.split(document)
            
            # Use LLM to process chunks
            processed_chunks = self.llm_processor.process_chunks(chunks)
            
            # Process each chunk
            chunk_embeddings = []
            valid_chunks = []
            
            for chunk in processed_chunks:
                # Check for duplicates
                if not self.deduplicator.is_duplicate(chunk):
                    valid_chunks.append(chunk)
                    
            if not valid_chunks:
                continue
                
            # Generate embeddings for valid chunks
            chunk_texts = [chunk['text'] for chunk in valid_chunks]
            embeddings = self.embedder.embed(chunk_texts)
            
            # Store in Milvus
            inserted_count = self.milvus_client.insert(valid_chunks, embeddings)
            ingested_count += inserted_count
            
        logger.info(f"Ingestion completed. {ingested_count} chunks ingested.")
    
    def query(self, q: str, domain: str = None) -> List[Dict[str, Any]]:
        """
        Query the vector store and return ranked results.
        
        Args:
            q (str): Query text
            domain (str): Domain filter
            
        Returns:
            List[Dict[str, Any]]: Ranked results
        """
        logger.info(f"Processing query: {q}")
        
        # Search for candidates
        candidates = self.searcher.search(q, domain)
        
        # Rerank candidates
        reranked = self.reranker.rerank(q, candidates)
        
        logger.info(f"Query completed. Returned {len(reranked)} results.")
        return reranked
    
    def close(self):
        """Clean up resources."""
        self.milvus_client.close()
        logger.info("RAG Engine closed")

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('--path', required=True, help='Path to file or directory')
    ingest_parser.add_argument('--domain', required=True, help='Domain for the documents')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query documents')
    query_parser.add_argument('--q', required=True, help='Query text')
    query_parser.add_argument('--domain', help='Domain filter')
    
    args = parser.parse_args()
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    try:
        if args.command == 'ingest':
            rag.ingest(args.path, args.domain)
        elif args.command == 'query':
            results = rag.query(args.q, args.domain)
            for i, result in enumerate(results[:5]):  # Show top 5 results
                print(f"\n--- Result {i+1} ---")
                print(f"Score: {result.get('final_score', 'N/A')}")
                print(f"Domain: {result.get('domain', 'N/A')}")
                print(f"Content: {result.get('content', 'N/A')[:200]}...")
        else:
            parser.print_help()
    finally:
        rag.close()

if __name__ == "__main__":
    main()