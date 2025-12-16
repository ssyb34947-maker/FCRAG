import os
import uuid
from typing import List, Dict, Any, Union
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentLoaderManager:
    """Manage loading documents from various sources and formats."""
    
    def __init__(self):
        """Initialize the document loader manager."""
        self.loader_mapping = {
            '.txt': lambda path: TextLoader(path, encoding='utf-8'),
            '.pdf': lambda path: PyPDFLoader(path),
            '.docx': Docx2txtLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
        }
    
    def load_document(self, file_path: Union[str, Path], domain: str = "") -> List[Dict[str, Any]]:
        """
        Load a single document file.
        
        Args:
            file_path (Union[str, Path]): Path to the document file
            domain (str): Domain/category for the document
            
        Returns:
            List[Dict[str, Any]]: List of loaded documents with standardized format
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        extension = file_path.suffix.lower()
        if extension not in self.loader_mapping:
            raise ValueError(f"Unsupported file type: {extension}")
            
        try:
            loader_factory = self.loader_mapping[extension]
            loader = loader_factory(str(file_path))
            documents = loader.load()
            
            standardized_docs = []
            for doc in documents:
                standardized_doc = {
                    "id": str(uuid.uuid4()),
                    "domain": domain,
                    "source": str(file_path.absolute()),
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                standardized_docs.append(standardized_doc)
                
            logger.info(f"Loaded {len(standardized_docs)} documents from {file_path}")
            return standardized_docs
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: Union[str, Path], domain: str = "", 
                      recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path (Union[str, Path]): Path to the directory
            domain (str): Domain/category for the documents
            recursive (bool): Whether to load recursively from subdirectories
            
        Returns:
            List[Dict[str, Any]]: List of loaded documents with standardized format
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.loader_mapping:
                try:
                    docs = self.load_document(file_path, domain)
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
                    
        logger.info(f"Loaded {len(documents)} documents from directory {directory_path}")
        return documents