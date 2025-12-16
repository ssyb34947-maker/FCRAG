import json
from typing import List, Dict, Any, Optional
import yaml
import os
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class MilvusClient:
    """Milvus client for vector storage and retrieval."""
    
    def __init__(self):
        """Initialize Milvus client and connect to the server."""
        milvus_config = config['milvus']
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=milvus_config['host'],
            port=milvus_config['port'],
            user=milvus_config['user'] if milvus_config['user'] else None,
            password=milvus_config['password'] if milvus_config['password'] else None
        )
        
        self.collection_name = milvus_config['collection_name']
        self.dim = milvus_config['dim']
        self.index_params = milvus_config['index_params']
        
        # Initialize collection
        self.collection = self._get_or_create_collection()
        logger.info(f"Connected to Milvus collection: {self.collection_name}")
    
    def _get_or_create_collection(self) -> Collection:
        """
        Get existing collection or create a new one.
        
        Returns:
            Collection: Milvus collection instance
        """
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        else:
            # Create schema
            schema = self._create_schema()
            
            # Create collection
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            # Create index
            self._create_index(collection)
            
            logger.info(f"Created new collection: {self.collection_name}")
            
        # Load collection
        collection.load()
        return collection
    
    def _create_schema(self) -> CollectionSchema:
        """
        Create collection schema.
        
        Returns:
            CollectionSchema: Milvus collection schema
        """
        # Define fields
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="domain",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=1024
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON
            )
        ]
        
        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description="RAG System Collection"
        )
        
        return schema
    
    def _create_index(self, collection: Collection):
        """
        Create index for the collection.
        
        Args:
            collection (Collection): Milvus collection instance
        """
        index_params = self.index_params
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        logger.info("Created index for collection")
    
    def insert(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Insert documents with embeddings into the collection.
        
        Args:
            documents (List[Dict[str, Any]]): Documents to insert
            embeddings (List[List[float]]): Corresponding embeddings
            
        Returns:
            int: Number of inserted entities
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
            
        # Prepare data
        domains = [doc.get('domain', '') for doc in documents]
        contents = [doc.get('text', '') for doc in documents]
        sources = [doc.get('source', '') for doc in documents]
        timestamps = [doc.get('timestamp', 0) for doc in documents]
        metadata_list = [json.dumps(doc.get('metadata', {})) for doc in documents]
        
        # Insert data
        entities = [
            domains,
            contents,
            sources,
            timestamps,
            embeddings,
            metadata_list
        ]
        
        result = self.collection.insert(entities)
        self.collection.flush()
        
        logger.info(f"Inserted {len(result.primary_keys)} entities into collection")
        return len(result.primary_keys)
    
    def search(self, query_embedding: List[float], domain: Optional[str] = None, 
               limit: int = 10, source: Optional[str] = None, 
               timestamp_filter: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding (List[float]): Query embedding
            domain (Optional[str]): Domain filter
            limit (int): Maximum number of results
            source (Optional[str]): Source filter
            timestamp_filter (Optional[dict]): Timestamp filter with keys 'start' and/or 'end'
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        # Prepare search parameters
        search_params = {
            "metric_type": self.index_params["metric_type"],
            "params": self.index_params["params"]
        }
        
        # Prepare filter expression
        filters = []
        if domain:
            filters.append(f"domain == '{domain}'")
        if source:
            filters.append(f"source == '{source}'")
        if timestamp_filter:
            if 'start' in timestamp_filter:
                filters.append(f"timestamp >= {timestamp_filter['start']}")
            if 'end' in timestamp_filter:
                filters.append(f"timestamp <= {timestamp_filter['end']}")
        
        expr = " and ".join(filters) if filters else None
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["domain", "content", "source", "timestamp", "metadata"]
        )
        
        # Process results
        processed_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "domain": hit.entity.get("domain"),
                    "content": hit.entity.get("content"),
                    "source": hit.entity.get("source"),
                    "timestamp": hit.entity.get("timestamp"),
                    "metadata": json.loads(hit.entity.get("metadata") or "{}")
                }
                processed_results.append(result)
                
        logger.info(f"Search returned {len(processed_results)} results")
        return processed_results
    
    def get_all_documents(self, domain: Optional[str] = None, source: Optional[str] = None, 
                          timestamp_filter: Optional[dict] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all documents with optional filters.
        
        Args:
            domain (Optional[str]): Domain filter
            source (Optional[str]): Source filter
            timestamp_filter (Optional[dict]): Timestamp filter with keys 'start' and/or 'end'
            limit (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: All documents matching the filters
        """
        # Prepare filter expression
        filters = []
        if domain:
            filters.append(f"domain == '{domain}'")
        if source:
            filters.append(f"source == '{source}'")
        if timestamp_filter:
            if 'start' in timestamp_filter:
                filters.append(f"timestamp >= {timestamp_filter['start']}")
            if 'end' in timestamp_filter:
                filters.append(f"timestamp <= {timestamp_filter['end']}")
        
        expr = " and ".join(filters) if filters else None
        
        # Query all documents
        results = self.collection.query(
            expr=expr,
            output_fields=["domain", "content", "source", "timestamp", "metadata"],
            limit=limit
        )
        
        # Process results
        processed_results = []
        for result in results:
            processed_result = {
                "id": result.get("id"),
                "domain": result.get("domain"),
                "content": result.get("content"),
                "source": result.get("source"),
                "timestamp": result.get("timestamp"),
                "metadata": json.loads(result.get("metadata") or "{}")
            }
            processed_results.append(processed_result)
                
        logger.info(f"Get all documents returned {len(processed_results)} results")
        return processed_results
    
    def close(self):
        """Close the connection to Milvus."""
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")