"""
Script to test the language detection module.
"""

import logging
from language_detection import LanguageDetector
from utils import setup_logging

# Set up logging
setup_logging()

def main():
    """Run the language detection test."""
    print("Running language detection test...")
    result = LanguageDetector.test_detection()
    print(result)

if __name__ == "__main__":
    main()
