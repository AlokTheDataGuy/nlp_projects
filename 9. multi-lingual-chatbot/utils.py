"""
Utility functions for the chatbot.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Any
from config import SUPPORTED_LANGUAGES

# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/chatbot_{time.strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Reduce verbosity of some loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level {log_level}")


def get_language_name(lang_code: str) -> str:
    """
    Get the human-readable name of a language from its code.
    
    Args:
        lang_code: Language code
        
    Returns:
        Human-readable language name
    """
    return SUPPORTED_LANGUAGES.get(lang_code, "Unknown")


def get_language_code(lang_name: str) -> Optional[str]:
    """
    Get the language code from its human-readable name.
    
    Args:
        lang_name: Human-readable language name
        
    Returns:
        Language code or None if not found
    """
    lang_name_lower = lang_name.lower()
    for code, name in SUPPORTED_LANGUAGES.items():
        if name.lower() == lang_name_lower:
            return code
    return None


def format_conversation_history(conversation: List[Dict]) -> str:
    """
    Format conversation history for display.
    
    Args:
        conversation: List of conversation messages
        
    Returns:
        Formatted conversation history
    """
    formatted = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        language = get_language_name(msg.get("language", "eng_Latn"))
        
        if role == "user":
            formatted.append(f"User ({language}): {content}")
        elif role == "assistant":
            formatted.append(f"Assistant ({language}): {content}")
        elif role == "system":
            formatted.append(f"System: {content}")
    
    return "\n".join(formatted)


def save_conversation(conversation: List[Dict], filename: str) -> bool:
    """
    Save conversation history to a file.
    
    Args:
        conversation: List of conversation messages
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save conversation: {e}")
        return False


def load_conversation(filename: str) -> Optional[List[Dict]]:
    """
    Load conversation history from a file.
    
    Args:
        filename: Input filename
        
    Returns:
        Conversation history or None if file not found
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load conversation: {e}")
        return None


def estimate_memory_usage() -> Dict[str, float]:
    """
    Estimate memory usage of the application.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
    }


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and memory.
    
    Returns:
        Dictionary with GPU information
    """
    try:
        import torch
        
        gpu_available = torch.cuda.is_available()
        gpu_info = {
            "available": gpu_available,
            "device_count": torch.cuda.device_count() if gpu_available else 0,
            "device_name": torch.cuda.get_device_name(0) if gpu_available else None,
        }
        
        if gpu_available:
            # Get memory information
            gpu_info["total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_info["allocated_memory_mb"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
            gpu_info["cached_memory_mb"] = torch.cuda.memory_reserved(0) / (1024 * 1024)
            gpu_info["free_memory_mb"] = gpu_info["total_memory_mb"] - gpu_info["allocated_memory_mb"]
        
        return gpu_info
    except Exception as e:
        logging.error(f"Failed to check GPU availability: {e}")
        return {"available": False, "error": str(e)}
