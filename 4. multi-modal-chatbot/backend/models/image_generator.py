import torch
from diffusers import StableDiffusionXLPipeline
import base64
from io import BytesIO
import os
from typing import Optional

# Cache for the model to avoid reloading
model_cache = None

async def generate_image(prompt: str, negative_prompt: Optional[str] = None) -> str:
    """
    Generate an image using Stable Diffusion 3 Medium model.
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Optional negative prompt
        
    Returns:
        Base64 encoded image
    """
    global model_cache
    
    try:
        # Load the model if not already loaded
        if model_cache is None:
            print("Loading Stable Diffusion 3 Medium model...")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load the model
            model_cache = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None,
            )
            
            model_cache = model_cache.to(device)
            print(f"Model loaded on {device}")
        
        # Generate the image
        image = model_cache(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        
        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str
        
    except Exception as e:
        print(f"Error generating image: {e}")
        raise Exception(f"Failed to generate image: {str(e)}")
