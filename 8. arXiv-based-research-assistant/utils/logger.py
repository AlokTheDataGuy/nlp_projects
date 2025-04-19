"""
Logging configuration for the arXiv Research Assistant.
"""
import logging
import sys
from pathlib import Path
from .config import LOG_LEVEL, BASE_DIR

# Create logs directory
logs_dir = BASE_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logger(name, log_file=None, level=LOG_LEVEL):
    """Set up logger with console and file handlers."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.propagate = False
    
    # Remove existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(logs_dir / log_file)
        file_handler.setLevel(getattr(logging, level))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger("arxiv_assistant", "arxiv_assistant.log")
