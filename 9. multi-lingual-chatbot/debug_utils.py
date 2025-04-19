"""
Debug utilities for the chatbot.
"""

import logging
import inspect
import json
from typing import Any, Dict, List, Optional

# Configure debug logger
debug_logger = logging.getLogger("debug")
debug_logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
import os
os.makedirs("logs", exist_ok=True)
debug_file_handler = logging.FileHandler("logs/debug.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
debug_file_handler.setFormatter(debug_formatter)
debug_logger.addHandler(debug_file_handler)

# Add console handler for debug logs
debug_console_handler = logging.StreamHandler()
debug_console_handler.setLevel(logging.DEBUG)
debug_console_handler.setFormatter(debug_formatter)
debug_logger.addHandler(debug_console_handler)

def debug_checkpoint(message: str, data: Any = None, module: str = None, function: str = None, line: int = None):
    """
    Log a debug checkpoint with detailed information.
    
    Args:
        message: Debug message
        data: Data to log (will be converted to string)
        module: Module name (auto-detected if None)
        function: Function name (auto-detected if None)
        line: Line number (auto-detected if None)
    """
    if module is None or function is None or line is None:
        # Get caller information
        frame = inspect.currentframe().f_back
        if module is None:
            module = frame.f_globals["__name__"]
        if function is None:
            function = frame.f_code.co_name
        if line is None:
            line = frame.f_lineno
    
    # Format data for logging
    data_str = ""
    if data is not None:
        if isinstance(data, (dict, list)):
            try:
                data_str = json.dumps(data, ensure_ascii=False, indent=2)
            except:
                data_str = str(data)
        else:
            data_str = str(data)
    
    # Log the checkpoint
    log_message = f"CHECKPOINT: {module}.{function}:{line} - {message}"
    if data_str:
        log_message += f"\nDATA: {data_str}"
    
    debug_logger.debug(log_message)
    
    return True  # Return True to allow use in conditional statements
