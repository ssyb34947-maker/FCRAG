from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import yaml
import os
from storage.milvus_client import MilvusClient
from embeddings.bairen_embedder import BairenEmbedder
from chunking.splitter import TextSplitter
from utils.dedup import Deduplicator
from loaders.loader_manager import DocumentLoaderManager
from retrieval.searcher import Searcher
from retrieval.reranker import Reranker
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="RAG Backend API", description="Backend API for RAG system", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
logger.info("Initializing RAG components...")
loader_manager = DocumentLoaderManager()
text_splitter = TextSplitter(config['chunking'])
embedder = BairenEmbedder()
milvus_client = MilvusClient()
deduplicator = Deduplicator(config['dedup'])
searcher = Searcher()
reranker = Reranker()
logger.info("RAG components initialized successfully")

class UploadResponse(BaseModel):
    status: str
    chunks: int
    indexed: int
    message: str

class SearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    source: Optional[str] = None
    timestamp_start: Optional[int] = None
    timestamp_end: Optional[int] = None

class SearchResult(BaseModel):
    content: str
    domain: str
    metadata: dict
    score: float
    source: Optional[str] = None
    timestamp: Optional[int] = None

class FilterOptions(BaseModel):
    domain: Optional[str] = None
    source: Optional[str] = None
    timestamp_start: Optional[int] = None
    timestamp_end: Optional[int] = None

class DocumentCard(BaseModel):
    id: int
    content: str
    domain: str
    source: str
    timestamp: int
    metadata: dict

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy"}

# RAG specific endpoints
@app.post("/rag/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), domain: str = "default"):
    """Upload and index a file"""
    logger.info(f"Upload request received for file: {file.filename}, domain: {domain}")
    try:
        # Save uploaded file to temporary location with correct extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # Load document
        documents = loader_manager.load_document(tmp_file_path, domain)
        logger.info(f"Loaded {len(documents)} documents from {file.filename}")
        
        # Process each document
        total_chunks = 0
        ingested_count = 0
        for document in documents:
            # Split document into chunks
            chunks = text_splitter.split(document)
            total_chunks += len(chunks)
            
            # Process each chunk
            valid_chunks = []
            for chunk in chunks:
                # Check for duplicates
                if not deduplicator.is_duplicate(chunk):
                    valid_chunks.append(chunk)
                    
            if not valid_chunks:
                continue
                
            # Generate embeddings for valid chunks
            chunk_texts = [chunk['text'] for chunk in valid_chunks]
            embeddings = embedder.embed(chunk_texts)
            
            # Store in Milvus
            inserted_count = milvus_client.insert(valid_chunks, embeddings)
            ingested_count += inserted_count
            
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        logger.info(f"Upload completed. Total chunks: {total_chunks}, Indexed: {ingested_count}")
        return UploadResponse(
            status="success",
            chunks=total_chunks,
            indexed=ingested_count,
            message="indexed successfully"
        )
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        # Clean up temporary file in case of error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query", response_model=List[SearchResult])
async def query_documents(request: SearchRequest):
    """Query documents using the full RAG pipeline"""
    logger.info(f"Query request received: {request.query}, domain: {request.domain}")
    try:
        # Generate query embedding
        query_embeddings = embedder.embed([request.query])
        query_embedding = query_embeddings[0]
        
        # Prepare timestamp filter
        timestamp_filter = {}
        if request.timestamp_start is not None:
            timestamp_filter['start'] = request.timestamp_start
        if request.timestamp_end is not None:
            timestamp_filter['end'] = request.timestamp_end
        
        # Search in Milvus with all filters
        search_results = milvus_client.search(
            query_embedding=query_embedding,
            domain=request.domain if request.domain else None,
            source=request.source if request.source else None,
            timestamp_filter=timestamp_filter or None,
            limit=10
        )
        
        # Use the searcher to find candidates from search results
        candidates = searcher.format_results_for_reranking(search_results)
        logger.info(f"Found {len(candidates)} candidates")
        
        # Rerank candidates
        reranked_results = reranker.rerank(request.query, candidates)
        logger.info(f"Reranked results: {len(reranked_results)} items")
        
        # Format results
        formatted_results = []
        for result in reranked_results:
            formatted_results.append(SearchResult(
                content=result.get("content", "")[:200] + "...",
                domain=result.get("domain", ""),
                metadata=result.get("metadata", {}),
                score=result.get("final_score", result.get("score", 0)),
                source=result.get("source", ""),
                timestamp=result.get("timestamp", 0)
            ))
        
        logger.info("Query completed successfully")
        return formatted_results
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Direct search endpoint (simplified version for frontend compatibility)"""
    logger.info(f"Search request received: {request.query}, domain: {request.domain}")
    try:
        # Generate query embedding
        query_embeddings = embedder.embed([request.query])
        query_embedding = query_embeddings[0]
        
        # Prepare timestamp filter
        timestamp_filter = {}
        if request.timestamp_start is not None:
            timestamp_filter['start'] = request.timestamp_start
        if request.timestamp_end is not None:
            timestamp_filter['end'] = request.timestamp_end
        
        # Search in Milvus
        results = milvus_client.search(
            query_embedding=query_embedding,
            domain=request.domain if request.domain else None,
            source=request.source if request.source else None,
            timestamp_filter=timestamp_filter or None,
            limit=5
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(SearchResult(
                content=result.get("content", ""),
                domain=result.get("domain", ""),
                metadata=result.get("metadata", {}),
                score=result.get("distance", 0),
                source=result.get("source", ""),
                timestamp=result.get("timestamp", 0)
            ))
        
        logger.info(f"Search completed successfully with {len(formatted_results)} results")
        return formatted_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/cards", response_model=List[DocumentCard])
async def get_document_cards(filter_options: FilterOptions):
    """Get all document cards with optional filters"""
    logger.info(f"Get document cards request received with filters: {filter_options}")
    try:
        # Prepare timestamp filter
        timestamp_filter = {}
        if filter_options.timestamp_start is not None:
            timestamp_filter['start'] = filter_options.timestamp_start
        if filter_options.timestamp_end is not None:
            timestamp_filter['end'] = filter_options.timestamp_end
        
        # Get all documents from Milvus
        results = milvus_client.get_all_documents(
            domain=filter_options.domain if filter_options.domain else None,
            source=filter_options.source if filter_options.source else None,
            timestamp_filter=timestamp_filter or None,
            limit=1000  # Adjust limit as needed
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(DocumentCard(
                id=result.get("id", 0),
                content=result.get("content", ""),
                domain=result.get("domain", ""),
                source=result.get("source", ""),
                timestamp=result.get("timestamp", 0),
                metadata=result.get("metadata", {})
            ))
        
        logger.info(f"Get document cards completed successfully with {len(formatted_results)} results")
        return formatted_results
    except Exception as e:
        logger.error(f"Get document cards failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting RAG backend server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)