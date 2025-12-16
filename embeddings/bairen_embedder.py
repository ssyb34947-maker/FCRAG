import time
import numpy as np
from typing import List, Dict, Any
import requests
import yaml
import os
from utils.logger import get_logger

logger = get_logger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class BairenEmbedder:
    """Aliyun Bailian Embedding API client for text embedding."""
    
    def __init__(self):
        """Initialize the embedder with configuration."""
        self.api_key = config['embedding']['api_key']
        self.model_name = config['embedding']['model_name']
        self.batch_size = config['embedding']['batch_size']
        self.timeout = config['embedding']['timeout']
        self.max_retries = config['embedding']['max_retries']
        # API endpoint for multimodal embedding
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            print(f"Processing batch of {len(batch_texts)} texts")
            embeddings = self._embed_batch_with_retry(batch_texts)
            all_embeddings.extend(embeddings)
            print(f"ALL_Embedding dimension: {len(all_embeddings[0])}")
            
        # Normalize embeddings
        normalized_embeddings = [self._l2_normalize(emb) for emb in all_embeddings]
        return normalized_embeddings
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        for attempt in range(self.max_retries):
            try:
                return self._embed_batch(texts)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to embed texts after {self.max_retries} attempts: {str(e)}")
                    raise
                    
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Embedding failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
                
        # This should never be reached
        raise RuntimeError("Unexpected error in retry loop")
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # For multimodal embedding with text inputs, we need to format the input correctly
        # Each text should be sent as a separate content item with type "text"
        contents = []
        for text in texts:
            contents.append({"text": text})
        
        payload = {
            "model": self.model_name,
            "input": {
                "contents": contents
            }
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status {response.status_code}: {response.text}")
            
        result = response.json()
        if "output" not in result or "embeddings" not in result["output"]:
            raise RuntimeError(f"Invalid API response: {result}")
            
        # Sort embeddings by index to ensure correct order
        embeddings_data = sorted(result["output"]["embeddings"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in embeddings_data]
        print(f"Embeddings generated successfully for {len(embeddings)} texts")
        print(f"Embedding dimension: {len(embeddings[0])}")
        return embeddings
    
    def _l2_normalize(self, vector: List[float]) -> List[float]:
        """
        L2 normalize a vector.
        
        Args:
            vector (List[float]): Vector to normalize
            
        Returns:
            List[float]: Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return [v / norm for v in vector]