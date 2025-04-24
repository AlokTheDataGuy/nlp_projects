from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can import our app modules
from app.core.init_app import init_app
from app.core.processing_queue import processing_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# Create data directory
os.makedirs("data", exist_ok=True)

app = FastAPI(title="ArXiv Expert Chatbot API",
              description="API for interacting with the ArXiv Expert Chatbot",
              version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to ArXiv Expert Chatbot API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "queue_status": processing_queue.get_queue_status()
    }

# Import and include routers
try:
    from app.api.endpoints import chat, papers, concepts, visualizations
except ImportError as e:
    logger.error(f"Error importing endpoints: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(concepts.router, prefix="/api/concepts", tags=["concepts"])
app.include_router(visualizations.router, prefix="/api/visualizations", tags=["visualizations"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup.
    """
    logger.info("Starting up ArXiv Expert Chatbot API")
    init_app()

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown.
    """
    logger.info("Shutting down ArXiv Expert Chatbot API")
    processing_queue.stop()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
