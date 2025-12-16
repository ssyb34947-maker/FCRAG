import logging
import yaml
import os

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger based on the configuration file.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(config['logging']['level'])
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(config['logging']['format'])
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if specified
        if config['logging'].get('file'):
            file_handler = logging.FileHandler(config['logging']['file'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger