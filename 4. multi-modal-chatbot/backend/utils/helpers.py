import os
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional

def save_uploaded_image(image_data: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Save an uploaded image and return its path and base64 representation.
    
    Args:
        image_data: Binary image data
        filename: Optional filename
        
    Returns:
        Dictionary with image path and base64 data
    """
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        import uuid
        filename = f"{uuid.uuid4()}.jpg"
    
    # Full path to save the image
    filepath = os.path.join("uploads", filename)
    
    # Save the image
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    # Open the image and convert to base64
    with Image.open(filepath) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "path": filepath,
        "base64": img_base64
    }

def cleanup_old_images(max_age_hours: int = 24):
    """
    Clean up images older than the specified age.
    
    Args:
        max_age_hours: Maximum age of images in hours
    """
    import time
    
    # Get current time
    current_time = time.time()
    
    # Check if uploads directory exists
    if not os.path.exists("uploads"):
        return
    
    # Iterate through files in uploads directory
    for filename in os.listdir("uploads"):
        filepath = os.path.join("uploads", filename)
        
        # Check if file is a regular file (not a directory)
        if os.path.isfile(filepath):
            # Get file modification time
            file_time = os.path.getmtime(filepath)
            
            # Calculate age in hours
            age_hours = (current_time - file_time) / 3600
            
            # Delete if older than max age
            if age_hours > max_age_hours:
                try:
                    os.remove(filepath)
                    print(f"Deleted old file: {filepath}")
                except Exception as e:
                    print(f"Error deleting file {filepath}: {e}")
