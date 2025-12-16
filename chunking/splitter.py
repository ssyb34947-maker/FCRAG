from typing import List, Dict, Any
import re
import copy
from utils.tokenizer import Tokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

class TextSplitter:
    """Split text into chunks according to different strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text splitter with configuration.
        
        Args:
            config (Dict[str, Any]): Chunking configuration
        """
        self.config = config
        self.strategy = config.get('strategy', 'sliding_token')
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 64)
    
    def split(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split document into chunks based on configured strategy.
        
        Args:
            document (Dict[str, Any]): Document to split
            
        Returns:
            List[Dict[str, Any]]: List of chunked documents
        """
        text = document.get('text', '')
        if not text.strip():
            return []
            
        if self.strategy == 'sliding_token':
            chunks = self._sliding_token_split(text)
        elif self.strategy == 'sentence':
            chunks = self._sentence_split(text)
        elif self.strategy == 'paragraph':
            chunks = self._paragraph_split(text)
        elif self.strategy == 'hybrid':
            chunks = self._hybrid_split(text)
        else:
            raise ValueError(f"Unknown splitting strategy: {self.strategy}")
            
        # Create document chunks with metadata
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Deep copy the document to avoid reference issues
            chunk_doc = copy.deepcopy(document)
            chunk_doc['text'] = chunk_text
            chunk_doc['metadata']['chunk_index'] = i
            chunk_doc['metadata']['chunk_total'] = len(chunks)
            document_chunks.append(chunk_doc)
            
        logger.info(f"Split document into {len(document_chunks)} chunks using {self.strategy} strategy")
        return document_chunks
    
    def _sliding_token_split(self, text: str) -> List[str]:
        """
        Split text using sliding window approach based on tokens.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        tokens = Tokenizer.tokenize(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = ''.join(chunk_tokens)
            chunks.append(chunk_text)
            
            if end == len(tokens):
                break
                
            start += self.chunk_size - self.chunk_overlap
            
        return chunks
    
    def _sentence_split(self, text: str) -> List[str]:
        """
        Split text by sentences.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Simple sentence splitting by punctuation
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = Tokenizer.count_tokens(sentence)
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _paragraph_split(self, text: str) -> List[str]:
        """
        Split text by paragraphs.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = Tokenizer.count_tokens(paragraph)
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_tokens += paragraph_tokens
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _hybrid_split(self, text: str) -> List[str]:
        """
        Hybrid approach combining paragraph and sentence splitting.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # First split by paragraphs
        paragraphs = self._paragraph_split(text)
        
        chunks = []
        for paragraph in paragraphs:
            if Tokenizer.count_tokens(paragraph) > self.chunk_size:
                # If paragraph is still too long, split by sentences
                sentence_chunks = self._sentence_split(paragraph)
                for sentence_chunk in sentence_chunks:
                    if Tokenizer.count_tokens(sentence_chunk) > self.chunk_size:
                        # If sentence is still too long, use sliding token split
                        token_chunks = self._sliding_token_split(sentence_chunk)
                        chunks.extend(token_chunks)
                    else:
                        chunks.append(sentence_chunk)
            else:
                chunks.append(paragraph)
                
        # Merge small chunks to optimize size
        merged_chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = Tokenizer.count_tokens(chunk)
            if current_tokens + chunk_tokens <= self.chunk_size:
                current_chunk += "\n\n" + chunk if current_chunk else chunk
                current_tokens += chunk_tokens
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
                current_tokens = chunk_tokens
                
        if current_chunk:
            merged_chunks.append(current_chunk)
            
        return merged_chunks