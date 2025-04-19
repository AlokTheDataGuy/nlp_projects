"""
Script to run tests for the multilingual chatbot.
"""

import os
import sys
import unittest
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tests(test_module=None, verbose=False):
    """
    Run tests for the multilingual chatbot.
    
    Args:
        test_module: Specific test module to run (None for all)
        verbose: Whether to show verbose output
    """
    # Discover and run tests
    if test_module:
        # Run specific test module
        suite = unittest.defaultTestLoader.loadTestsFromName(test_module)
    else:
        # Run all tests
        suite = unittest.defaultTestLoader.discover(".")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tests for the multilingual chatbot")
    parser.add_argument("--module", help="Specific test module to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    # Run tests
    exit_code = run_tests(args.module, args.verbose)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
