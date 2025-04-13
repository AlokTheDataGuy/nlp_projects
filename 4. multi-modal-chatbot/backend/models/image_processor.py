import ollama
import base64
from typing import Dict, Any, List, Optional

async def process_image(image_data: Dict[str, Any], prompt: str, history: Optional[List] = None) -> Dict[str, Any]:
    """
    Process an image using llama3.2-vision:11b model via Ollama.
    
    Args:
        image_data: Dictionary containing image path and base64 data
        prompt: Text prompt to guide image analysis
        history: Optional conversation history
        
    Returns:
        Dictionary with the model's response
    """
    try:
        # Prepare the message with image
        messages = []
        
        # Add history if provided
        if history:
            messages.extend(history)
        
        # Create the user message with image
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image", 
                    "image": image_data["base64"]
                }
            ]
        }
        
        messages.append(user_message)
        
        # Call the Ollama API with the vision model
        response = ollama.chat(
            model="llama3.2-vision:11b",
            messages=messages,
            stream=False
        )
        
        # Extract and return the response
        return {
            "text": response["message"]["content"],
            "model": "llama3.2-vision:11b"
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            "text": f"Error analyzing image: {str(e)}",
            "model": "llama3.2-vision:11b",
            "error": True
        }
