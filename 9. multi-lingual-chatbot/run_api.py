"""
Script to run the FastAPI server.
"""

import argparse
import uvicorn
from utils import setup_logging

# Set up logging
setup_logging()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Run the FastAPI app with Uvicorn
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
