import ollama
from typing import Dict, Any, List, Optional

async def process_text(message: str, history: Optional[List] = None, generate_image: bool = False) -> Dict[str, Any]:
    """
    Process text using qwen2.5:7b model via Ollama.
    
    Args:
        message: Text message to process
        history: Optional conversation history
        generate_image: Flag indicating if image generation is requested
        
    Returns:
        Dictionary with the model's response
    """
    try:
        # Prepare messages
        messages = []
        
        # Add history if provided
        if history:
            messages.extend(history)
        
        # Add system message if image generation is requested
        if generate_image:
            system_message = {
                "role": "system",
                "content": "You are a helpful assistant that can generate text responses and suggest image prompts. "
                           "When the user asks for an image or when it would be helpful to show an image, "
                           "include a section at the end of your response with '[IMAGE_PROMPT]: <detailed prompt for image generation>'."
            }
            messages.insert(0, system_message)
        
        # Add the user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Call the Ollama API
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages,
            stream=False
        )
        
        # Extract the response text
        response_text = response["message"]["content"]
        
        # Check if there's an image prompt in the response
        image_prompt = None
        if generate_image and "[IMAGE_PROMPT]:" in response_text:
            parts = response_text.split("[IMAGE_PROMPT]:")
            response_text = parts[0].strip()
            image_prompt = parts[1].strip()
        
        return {
            "text": response_text,
            "model": "qwen2.5:7b",
            "image_prompt": image_prompt
        }
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "text": f"Error processing text: {str(e)}",
            "model": "qwen2.5:7b",
            "error": True
        }
