from typing import List

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

class Tokenizer:
    """Simple tokenizer wrapper for text processing."""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        if JIEBA_AVAILABLE:
            return list(jieba.cut(text))
        else:
            # Fallback to simple split if jieba is not available
            return text.split()
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count the number of tokens in text.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of tokens
        """
        return len(Tokenizer.tokenize(text))