"""
API Server Module

This module sets up the FastAPI server for the arXiv research assistant.
"""

import os
import logging
import yaml
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routes import router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = "config/app_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dict containing configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Create FastAPI app
app = FastAPI(
    title="arXiv Research Assistant API",
    description="API for the arXiv-based research assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve static files (UI)
ui_dir = Path("ui/dist")
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    logger.info(f"Mounted UI from {ui_dir}")
else:
    logger.warning(f"UI directory {ui_dir} not found")

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting arXiv Research Assistant API")

    # Load configuration
    config = load_config()

    # Set log level
    log_level = config["logging"]["level"]
    logging.getLogger().setLevel(log_level)

    # Log configuration
    logger.info(f"Server configuration: {config['server']}")

    # Initialize services
    # This would normally be done via dependency injection
    # For now, we'll rely on the get_services dependency in routes.py

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down arXiv Research Assistant API")

    # Clean up resources
    # This would normally be done via dependency injection
    # For now, we'll just log the shutdown

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Root endpoint
@app.get("/api")
async def root():
    return {
        "name": "arXiv Research Assistant API",
        "version": "1.0.0",
        "description": "API for the arXiv-based research assistant"
    }

# Run the server
if __name__ == "__main__":
    import uvicorn

    # Load configuration
    config = load_config()
    server_config = config["server"]

    # Run server
    uvicorn.run(
        "src.api.server:app",
        host=server_config["host"],
        port=server_config["port"],
        reload=server_config["debug"],
        workers=server_config["workers"]
    )
