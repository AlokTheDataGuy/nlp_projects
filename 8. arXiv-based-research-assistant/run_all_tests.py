#!/usr/bin/env python
"""
Master test script for the ArXiv Expert Chatbot.
This script runs all test scripts and generates a comprehensive report.
"""

import subprocess
import argparse
import sys
import os
import time
from datetime import datetime
import logging
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("master_test.log")
    ]
)

logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000/api"
REPORTS_DIR = "test_reports"
TEST_SCRIPTS = [
    "test_chatbot.py",
    "evaluate_accuracy.py",
    "test_vector_search.py"
]

def run_test_script(script: str, api_url: str) -> bool:
    """
    Run a test script.
    
    Args:
        script: The script to run
        api_url: The API URL to use
        
    Returns:
        True if the script ran successfully, False otherwise
    """
    logger.info(f"Running test script: {script}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script, "--url", api_url],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check if the script ran successfully
        if result.returncode == 0:
            logger.info(f"Test script {script} ran successfully")
            return True
        else:
            logger.error(f"Test script {script} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running test script {script}: {e}")
        return False

def collect_test_results() -> Dict[str, Any]:
    """
    Collect test results from all test scripts.
    
    Returns:
        Dictionary of test results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_results": {}
    }
    
    # Collect results from test_chatbot.py
    try:
        latest_file = get_latest_file("test_results", "test_results_")
        if latest_file:
            with open(latest_file, 'r') as f:
                results["test_results"]["chatbot"] = json.load(f)
    except Exception as e:
        logger.error(f"Error collecting results from test_chatbot.py: {e}")
    
    # Collect results from evaluate_accuracy.py
    try:
        latest_file = get_latest_file("accuracy_results", "accuracy_results_")
        if latest_file:
            with open(latest_file, 'r') as f:
                results["test_results"]["accuracy"] = json.load(f)
    except Exception as e:
        logger.error(f"Error collecting results from evaluate_accuracy.py: {e}")
    
    # Collect results from test_vector_search.py
    try:
        latest_file = get_latest_file("vector_search_results", "vector_search_results_")
        if latest_file:
            with open(latest_file, 'r') as f:
                results["test_results"]["vector_search"] = json.load(f)
    except Exception as e:
        logger.error(f"Error collecting results from test_vector_search.py: {e}")
    
    return results

def get_latest_file(directory: str, prefix: str) -> Optional[str]:
    """
    Get the latest file in a directory with a given prefix.
    
    Args:
        directory: The directory to search
        prefix: The file prefix to match
        
    Returns:
        The path to the latest file, or None if no files found
    """
    if not os.path.exists(directory):
        return None
    
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    
    return max(files, key=os.path.getmtime)

def generate_comprehensive_report(results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive report from all test results.
    
    Args:
        results: Dictionary of test results
        
    Returns:
        Comprehensive report string
    """
    report = []
    report.append("=" * 80)
    report.append("ARXIV EXPERT CHATBOT COMPREHENSIVE TEST REPORT")
    report.append("=" * 80)
    report.append(f"Timestamp: {results['timestamp']}")
    report.append("")
    
    # Add chatbot test results
    if "chatbot" in results["test_results"]:
        chatbot_results = results["test_results"]["chatbot"]
        report.append("CHATBOT FUNCTIONALITY TEST RESULTS")
        report.append("-" * 80)
        
        overall = chatbot_results.get("overall_results", {})
        report.append(f"Success Rate: {overall.get('success_rate', 0):.2f}%")
        report.append(f"Successful Queries: {overall.get('successful_queries', 0)} / {overall.get('total_queries', 0)}")
        
        # Add category results
        categories = chatbot_results.get("category_results", {})
        report.append("\nCategory Results:")
        for category, cat_results in categories.items():
            report.append(f"  {category}: {cat_results.get('success_rate', 0):.2f}%")
        
        report.append("")
    
    # Add accuracy test results
    if "accuracy" in results["test_results"]:
        accuracy_results = results["test_results"]["accuracy"]
        report.append("FACTUAL ACCURACY TEST RESULTS")
        report.append("-" * 80)
        
        report.append(f"Overall Accuracy: {accuracy_results.get('overall_accuracy', 0):.2f}%")
        report.append(f"Fact Accuracy: {accuracy_results.get('fact_accuracy', 0):.2f}%")
        report.append(f"Paper Accuracy: {accuracy_results.get('paper_accuracy', 0):.2f}%")
        report.append(f"Correct Facts: {accuracy_results.get('correct_facts', 0)} / {accuracy_results.get('total_facts', 0)}")
        report.append(f"Correct Papers: {accuracy_results.get('correct_papers', 0)} / {accuracy_results.get('total_papers', 0)}")
        
        report.append("")
    
    # Add vector search test results
    if "vector_search" in results["test_results"]:
        vector_results = results["test_results"]["vector_search"]
        report.append("VECTOR SEARCH TEST RESULTS")
        report.append("-" * 80)
        
        overall = vector_results.get("overall_results", {})
        report.append(f"Average Relevance Score: {overall.get('average_relevance_score', 0):.2f}")
        report.append(f"Average Keyword Score: {overall.get('average_keyword_score', 0):.2f}")
        if 'average_semantic_score' in overall:
            report.append(f"Average Semantic Score: {overall.get('average_semantic_score', 0):.2f}")
        
        report.append("")
    
    # Add summary
    report.append("SUMMARY")
    report.append("-" * 80)
    
    # Calculate overall performance score
    overall_score = 0.0
    score_components = 0
    
    if "chatbot" in results["test_results"]:
        chatbot_score = results["test_results"]["chatbot"].get("overall_results", {}).get("success_rate", 0)
        overall_score += chatbot_score
        score_components += 1
        report.append(f"Chatbot Functionality Score: {chatbot_score:.2f}%")
    
    if "accuracy" in results["test_results"]:
        accuracy_score = results["test_results"]["accuracy"].get("overall_accuracy", 0)
        overall_score += accuracy_score
        score_components += 1
        report.append(f"Factual Accuracy Score: {accuracy_score:.2f}%")
    
    if "vector_search" in results["test_results"]:
        vector_score = results["test_results"]["vector_search"].get("overall_results", {}).get("average_relevance_score", 0) * 100
        overall_score += vector_score
        score_components += 1
        report.append(f"Vector Search Score: {vector_score:.2f}%")
    
    if score_components > 0:
        final_score = overall_score / score_components
        report.append(f"\nOVERALL PERFORMANCE SCORE: {final_score:.2f}%")
        
        # Add performance rating
        if final_score >= 90:
            rating = "Excellent"
        elif final_score >= 80:
            rating = "Very Good"
        elif final_score >= 70:
            rating = "Good"
        elif final_score >= 60:
            rating = "Satisfactory"
        elif final_score >= 50:
            rating = "Needs Improvement"
        else:
            rating = "Poor"
        
        report.append(f"Performance Rating: {rating}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Run all tests for the ArXiv Expert Chatbot")
    parser.add_argument("--url", default=API_URL, help="Base URL for the API")
    args = parser.parse_args()
    
    try:
        # Create reports directory if it doesn't exist
        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
        
        # Run all test scripts
        all_successful = True
        for script in TEST_SCRIPTS:
            if not run_test_script(script, args.url):
                all_successful = False
        
        # Collect test results
        results = collect_test_results()
        
        # Generate comprehensive report
        report = generate_comprehensive_report(results)
        print(report)
        
        # Save report and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_filename = os.path.join(REPORTS_DIR, f"comprehensive_report_{timestamp}.txt")
        with open(report_filename, 'w') as f:
            f.write(report)
        
        # Save results
        results_filename = os.path.join(REPORTS_DIR, f"comprehensive_results_{timestamp}.json")
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comprehensive report saved to {report_filename}")
        logger.info(f"Comprehensive results saved to {results_filename}")
        
        return 0 if all_successful else 1
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
