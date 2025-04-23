"""
Run Server Script

This script runs the FastAPI server for the arXiv research assistant.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the FastAPI server.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the FastAPI server")
    parser.add_argument("--config", type=str, default="config/app_config.yaml", help="Path to configuration file")
    parser.add_argument("--host", type=str, default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get server configuration
        server_config = config["server"]
        
        # Override with command line arguments
        host = args.host or server_config["host"]
        port = args.port or server_config["port"]
        reload = args.reload or server_config["debug"]
        workers = args.workers or server_config["workers"]
        
        # Run server
        import uvicorn
        
        logger.info(f"Starting server on {host}:{port} with {workers} workers")
        
        uvicorn.run(
            "src.api.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers
        )
    
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
