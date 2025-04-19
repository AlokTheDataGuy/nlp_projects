"""
Script to run the entire arXiv Research Assistant project.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger("run_all", "run_all.log")

def init_project(args):
    """Initialize the project."""
    logger.info("Initializing project...")
    
    cmd = [sys.executable, "init_project.py"]
    
    if args.no_model:
        cmd.append("--no-model")
    
    if args.no_frontend:
        cmd.append("--no-frontend")
    
    subprocess.run(cmd)

def run_pipeline(args):
    """Run the data pipeline."""
    logger.info("Running data pipeline...")
    
    cmd = [sys.executable, "run_pipeline.py"]
    
    if args.categories:
        cmd.extend(["--categories"] + args.categories)
    
    if args.papers_per_category:
        cmd.extend(["--papers-per-category", str(args.papers_per_category)])
    
    if args.paper_ids:
        cmd.extend(["--paper-ids"] + args.paper_ids)
    
    if args.steps:
        cmd.extend(["--steps"] + args.steps)
    
    if args.download_only:
        cmd.append("--download-only")
    
    if args.process_only:
        cmd.append("--process-only")
    
    if args.chunk_only:
        cmd.append("--chunk-only")
    
    if args.embed_only:
        cmd.append("--embed-only")
    
    if args.index_only:
        cmd.append("--index-only")
    
    subprocess.run(cmd)

def run_api(args):
    """Run the API server."""
    logger.info("Running API server...")
    
    cmd = [sys.executable, "run_api.py"]
    
    if args.host:
        cmd.extend(["--host", args.host])
    
    if args.port:
        cmd.extend(["--port", str(args.port)])
    
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
    
    if args.reload:
        cmd.append("--reload")
    
    subprocess.run(cmd)

def run_cli(args):
    """Run the CLI."""
    logger.info("Running CLI...")
    
    subprocess.run([sys.executable, "cli.py"])

def run_frontend(args):
    """Run the frontend."""
    logger.info("Running frontend...")
    
    # Check if frontend directory exists
    if not os.path.exists("frontend"):
        logger.error("Frontend directory not found")
        logger.error("Please set up the frontend first with: python setup_frontend.py")
        return
    
    # Run frontend server
    os.chdir("frontend")
    subprocess.run(["npm", "run", "dev"])
    os.chdir("..")

def run_tests(args):
    """Run tests."""
    logger.info("Running tests...")
    
    cmd = [sys.executable, "run_tests.py"]
    
    if args.system:
        cmd.append("--system")
    
    if args.unit:
        cmd.append("--unit")
    
    if args.llm:
        cmd.append("--llm")
    
    if args.retriever:
        cmd.append("--retriever")
    
    if args.generator:
        cmd.append("--generator")
    
    if args.all:
        cmd.append("--all")
    
    subprocess.run(cmd)

def run_docker(args):
    """Run Docker containers."""
    logger.info("Running Docker containers...")
    
    # Build and run containers
    subprocess.run(["docker-compose", "up", "--build"])

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run arXiv Research Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init parser
    init_parser = subparsers.add_parser("init", help="Initialize project")
    init_parser.add_argument("--no-model", action="store_true", help="Skip downloading the LLaMA model")
    init_parser.add_argument("--no-frontend", action="store_true", help="Skip setting up the React frontend")
    
    # Pipeline parser
    pipeline_parser = subparsers.add_parser("pipeline", help="Run data pipeline")
    pipeline_parser.add_argument("--categories", nargs="+", help="arXiv categories to download")
    pipeline_parser.add_argument("--papers-per-category", type=int, help="Number of papers per category")
    pipeline_parser.add_argument("--paper-ids", nargs="+", help="Specific paper IDs to process")
    pipeline_parser.add_argument("--steps", nargs="+", help="Pipeline steps to run")
    pipeline_parser.add_argument("--download-only", action="store_true", help="Only download papers")
    pipeline_parser.add_argument("--process-only", action="store_true", help="Only process papers")
    pipeline_parser.add_argument("--chunk-only", action="store_true", help="Only chunk documents")
    pipeline_parser.add_argument("--embed-only", action="store_true", help="Only generate embeddings")
    pipeline_parser.add_argument("--index-only", action="store_true", help="Only build index")
    
    # API parser
    api_parser = subparsers.add_parser("api", help="Run API server")
    api_parser.add_argument("--host", type=str, help="Host to bind to")
    api_parser.add_argument("--port", type=int, help="Port to bind to")
    api_parser.add_argument("--workers", type=int, help="Number of worker processes")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # CLI parser
    cli_parser = subparsers.add_parser("cli", help="Run CLI")
    
    # Frontend parser
    frontend_parser = subparsers.add_parser("frontend", help="Run frontend")
    
    # Tests parser
    tests_parser = subparsers.add_parser("tests", help="Run tests")
    tests_parser.add_argument("--system", action="store_true", help="Run system tests")
    tests_parser.add_argument("--unit", action="store_true", help="Run unit tests")
    tests_parser.add_argument("--llm", action="store_true", help="Test LLaMA model")
    tests_parser.add_argument("--retriever", action="store_true", help="Test hybrid retriever")
    tests_parser.add_argument("--generator", action="store_true", help="Test response generator")
    tests_parser.add_argument("--all", action="store_true", help="Test all components")
    
    # Docker parser
    docker_parser = subparsers.add_parser("docker", help="Run Docker containers")
    
    args = parser.parse_args()
    
    # Run command
    if args.command == "init":
        init_project(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    elif args.command == "api":
        run_api(args)
    elif args.command == "cli":
        run_cli(args)
    elif args.command == "frontend":
        run_frontend(args)
    elif args.command == "tests":
        run_tests(args)
    elif args.command == "docker":
        run_docker(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
