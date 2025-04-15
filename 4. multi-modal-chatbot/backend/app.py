from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import base64
from io import BytesIO
from PIL import Image
import os
from typing import Optional, List, Dict, Any
import uuid

# Import services
import sys
import os

# Add the current directory to the path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.input_classifier import classify_input
from services.task_dispatcher import dispatch_tasks
from services.response_aggregator import aggregate_responses

app = FastAPI(title="Multi-Modal Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Multi-Modal Chatbot API is running"}

@app.post("/chat")
async def chat(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    conversation_history: Optional[str] = Form(None)
):
    try:
        # Process image if provided
        image_data = None
        if image:
            # Save the uploaded image
            img_path = f"uploads/{uuid.uuid4()}.jpg"
            with open(img_path, "wb") as f:
                f.write(await image.read())

            # Open the image for processing
            with Image.open(img_path) as img:
                # Convert to base64 for response
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                image_data = {
                    "path": img_path,
                    "base64": base64.b64encode(buffered.getvalue()).decode("utf-8")
                }

        # Parse conversation history
        history = []
        if conversation_history:
            try:
                import json
                history = json.loads(conversation_history)
            except Exception as e:
                print(f"Error parsing conversation history: {e}")

        # Classify input
        input_type = classify_input(message, image_data is not None)

        # Dispatch tasks to appropriate models
        tasks_results = await dispatch_tasks(input_type, message, image_data, history)

        # Aggregate responses
        response = aggregate_responses(tasks_results)

        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Image generation endpoint removed

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
