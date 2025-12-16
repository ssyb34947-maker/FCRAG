#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database initialization script.
This script initializes the Milvus database by creating the required collection
and indexes based on the existing schema.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.milvus_client import MilvusClient
from utils.logger import get_logger

logger = get_logger(__name__)

def init_database():
    """
    Initialize the Milvus database.
    This function creates the collection and indexes if they don't exist.
    """
    try:
        # Initialize Milvus client which will automatically create the collection
        # and indexes based on the schema defined in milvus_client.py
        client = MilvusClient()
        
        # Log collection information
        collection = client.collection
        logger.info(f"Collection name: {collection.name}")
        logger.info(f"Collection description: {collection.description}")
        
        # Get collection statistics
        collection.flush()  # Flush to get accurate count
        num_entities = collection.num_entities
        logger.info(f"Number of entities in collection: {num_entities}")
        
        # List all collections to verify creation
        from pymilvus import utility
        collections = utility.list_collections()
        logger.info(f"Existing collections: {collections}")
        
        # Close the connection
        client.close()
        
        print("Database initialization completed successfully!")
        print(f"- Collection '{collection.name}' is ready")
        print(f"- Total entities in collection: {num_entities}")
        print(f"- Available collections: {collections}")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    print("Initializing Milvus database...")
    print("=" * 50)
    init_database()