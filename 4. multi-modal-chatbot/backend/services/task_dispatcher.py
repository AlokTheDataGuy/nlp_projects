from typing import Dict, Any, List, Optional
import asyncio

# Import model processors
from models.text_processor import process_text
from models.image_processor import process_image
from models.image_generator import generate_image

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
    
    # Dispatch text processing task
    if input_type["needs_text_processing"]:
        tasks["text_processing"] = process_text(
            message=message,
            history=history,
            generate_image=input_type["needs_image_generation"]
        )
    
    # Dispatch image processing task if needed
    if input_type["needs_image_processing"] and image_data:
        tasks["image_processing"] = process_image(
            image_data=image_data,
            prompt=f"Describe this image and respond to: {message}",
            history=history
        )
    
    # Execute all tasks concurrently
    for task_name, task in tasks.items():
        results[task_name] = await task
    
    # Check if we need to generate an image based on text processing result
    if input_type["needs_image_generation"] and "text_processing" in results:
        text_result = results["text_processing"]
        if "image_prompt" in text_result and text_result["image_prompt"]:
            try:
                # Generate image using the provided prompt
                image_base64 = await generate_image(text_result["image_prompt"])
                results["image_generation"] = {
                    "image": image_base64,
                    "prompt": text_result["image_prompt"]
                }
            except Exception as e:
                print(f"Error in image generation: {e}")
                results["image_generation"] = {
                    "error": True,
                    "message": f"Failed to generate image: {str(e)}"
                }
    
    return results
