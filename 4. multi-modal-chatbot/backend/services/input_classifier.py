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
        "needs_image_processing": False
    }

    # Check if image processing is needed
    if has_image:
        processing_flags["needs_image_processing"] = True

    return processing_flags
