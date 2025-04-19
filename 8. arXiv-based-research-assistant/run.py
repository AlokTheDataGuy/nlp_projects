"""
Script to run the arXiv Research Assistant.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

from utils.config import API_HOST, API_PORT
from utils.logger import setup_logger

logger = setup_logger("run", "run.log")

def run_api(host=API_HOST, port=API_PORT, reload=True):
    """Run the API server."""
    logger.info(f"Starting API server on {host}:{port}...")
    
    # Run API server
    subprocess.run([
        sys.executable, "run_api.py",
        "--host", host,
        "--port", str(port),
        "--reload" if reload else ""
    ])

def run_cli():
    """Run the CLI."""
    logger.info("Starting CLI...")
    
    # Run CLI
    subprocess.run([sys.executable, "cli.py"])

def run_frontend():
    """Run the frontend development server."""
    logger.info("Starting frontend development server...")
    
    # Check if frontend directory exists
    if not os.path.exists("frontend"):
        logger.error("Frontend directory not found")
        logger.error("Please set up the frontend first with: python setup_frontend.py")
        return
    
    # Run frontend server
    os.chdir("frontend")
    subprocess.run(["npm", "run", "dev"])
    os.chdir("..")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run arXiv Research Assistant")
    parser.add_argument("--api", action="store_true", help="Run API server")
    parser.add_argument("--cli", action="store_true", help="Run CLI")
    parser.add_argument("--frontend", action="store_true", help="Run frontend development server")
    parser.add_argument("--host", type=str, default=API_HOST, help="API host")
    parser.add_argument("--port", type=int, default=API_PORT, help="API port")
    parser.add_argument("--no-reload", action="store_true", help="Disable API auto-reload")
    
    args = parser.parse_args()
    
    # Default to CLI if no specific mode is specified
    if not (args.api or args.cli or args.frontend):
        args.cli = True
    
    # Run components
    if args.api:
        run_api(host=args.host, port=args.port, reload=not args.no_reload)
    
    elif args.cli:
        run_cli()
    
    elif args.frontend:
        run_frontend()

if __name__ == "__main__":
    main()
