"""
Script to run the API server.
"""
import argparse
import uvicorn

from utils.config import API_HOST, API_PORT, API_WORKERS

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run arXiv Research Assistant API server")
    parser.add_argument("--host", type=str, default=API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=API_PORT, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=API_WORKERS, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Run API server
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
