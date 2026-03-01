import logging
import logging.handlers
import sys
import os
from pathlib import Path

def setup_logging(log_file_path=None, log_level=logging.INFO):
    """
    Setup centralized logging configuration.
    
    Args:
        log_file_path (str or Path): Path to the log file. If None, uses default.
        log_level (int): Logging level (default: logging.INFO)
    """
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()
        
    # Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_file_path:
        # Ensure directory exists
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, 
            maxBytes=10*1024*1024, # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        
    return logger
