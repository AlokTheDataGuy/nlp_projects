from typing import Dict, Any, List, Tuple

def classify_input(message: str, has_image: bool) -> Dict[str, bool]:
    """
    Classify the input to determine what processing is needed.
    
    Args:
        message: The text message from the user
        has_image: Boolean indicating if an image was provided
        
    Returns:
        Dictionary with processing flags
    """
    # Initialize processing flags
    processing_flags = {
        "needs_text_processing": True,  # Always process text
        "needs_image_processing": False,
        "needs_image_generation": False
    }
    
    # Check if image processing is needed
    if has_image:
        processing_flags["needs_image_processing"] = True
    
    # Check if image generation might be needed
    image_generation_keywords = [
        "generate", "create", "draw", "show", "picture", "image", "photo", "visualization",
        "visualize", "illustrate", "illustration", "render", "sketch", "painting", "depict"
    ]
    
    # Check for image generation keywords in the message
    message_lower = message.lower()
    for keyword in image_generation_keywords:
        if keyword in message_lower:
            processing_flags["needs_image_generation"] = True
            break
    
    return processing_flags
