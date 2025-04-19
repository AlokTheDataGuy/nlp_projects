"""
Script to run tests for the arXiv Research Assistant.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger("run_tests", "run_tests.log")

def run_system_tests(llm=False, retriever=False, generator=False, all_tests=False):
    """Run system tests."""
    logger.info("Running system tests...")
    
    # Build command
    cmd = [sys.executable, "test_system.py"]
    
    if llm:
        cmd.append("--llm")
    
    if retriever:
        cmd.append("--retriever")
    
    if generator:
        cmd.append("--generator")
    
    if all_tests:
        cmd.append("--all")
    
    # Run tests
    subprocess.run(cmd)

def run_unit_tests(test_dir="tests"):
    """Run unit tests."""
    logger.info("Running unit tests...")
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        logger.error("Please create unit tests first")
        return
    
    # Run tests with pytest
    subprocess.run([sys.executable, "-m", "pytest", test_dir, "-v"])

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for arXiv Research Assistant")
    parser.add_argument("--system", action="store_true", help="Run system tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--llm", action="store_true", help="Test LLaMA model")
    parser.add_argument("--retriever", action="store_true", help="Test hybrid retriever")
    parser.add_argument("--generator", action="store_true", help="Test response generator")
    parser.add_argument("--all", action="store_true", help="Test all components")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test is specified
    if not (args.system or args.unit or args.llm or args.retriever or args.generator or args.all):
        args.system = True
        args.all = True
    
    # Run tests
    if args.system or args.llm or args.retriever or args.generator or args.all:
        run_system_tests(
            llm=args.llm,
            retriever=args.retriever,
            generator=args.generator,
            all_tests=args.all
        )
    
    if args.unit:
        run_unit_tests()

if __name__ == "__main__":
    main()
