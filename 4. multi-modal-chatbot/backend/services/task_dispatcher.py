from typing import Dict, Any, List, Optional
import asyncio

# Import model processors
import sys
import os

# Add the parent directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_processor import process_text
from models.image_processor import process_image

async def dispatch_tasks(
    input_type: Dict[str, bool],
    message: str,
    image_data: Optional[Dict[str, Any]] = None,
    history: Optional[List] = None
) -> Dict[str, Any]:
    """
    Dispatch tasks to appropriate models based on input classification.

    Args:
        input_type: Dictionary with processing flags
        message: The text message from the user
        image_data: Optional image data if an image was provided
        history: Optional conversation history

    Returns:
        Dictionary with results from all tasks
    """
    tasks = {}
    results = {}

    # If image processing is needed and we have image data, prioritize that
    if input_type["needs_image_processing"] and image_data:
        print("Prioritizing image processing task")
        tasks["image_processing"] = process_image(
            image_data=image_data,
            prompt=f"Describe this image and respond to: {message}",
            history=history
        )
    # Otherwise, dispatch text processing task
    elif input_type["needs_text_processing"]:
        print("Dispatching text processing task")
        tasks["text_processing"] = process_text(
            message=message,
            history=history
        )

    # Execute all tasks concurrently
    for task_name, task in tasks.items():
        results[task_name] = await task

    # No image generation in this version

    return results
