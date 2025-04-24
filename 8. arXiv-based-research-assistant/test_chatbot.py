#!/usr/bin/env python
"""
Comprehensive testing script for the ArXiv Expert Chatbot.
This script evaluates the chatbot's accuracy, response quality, and functionality.
"""

import requests
import json
import time
import random
import argparse
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot_test.log")
    ]
)

logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000/api"
TEST_RESULTS_DIR = "test_results"

# Test queries by category
TEST_QUERIES = {
    "neural_networks": [
        "What are neural networks?",
        "Explain how convolutional neural networks work",
        "What's the difference between CNNs and RNNs?",
        "How do transformers compare to traditional neural networks?",
        "What are the latest advancements in neural network architectures?"
    ],
    "machine_learning": [
        "Explain supervised vs unsupervised learning",
        "What is reinforcement learning?",
        "How does gradient descent work?",
        "What are support vector machines?",
        "Explain the bias-variance tradeoff"
    ],
    "nlp": [
        "What is natural language processing?",
        "How do transformers work in NLP?",
        "Explain the attention mechanism",
        "What is BERT and how does it work?",
        "What are the challenges in NLP research?"
    ],
    "algorithms": [
        "Explain the quicksort algorithm",
        "What is dynamic programming?",
        "How does the A* search algorithm work?",
        "What are the most efficient sorting algorithms?",
        "Explain the difference between BFS and DFS"
    ],
    "research_papers": [
        "Find papers about quantum computing",
        "What are the most cited papers in machine learning?",
        "Find recent papers on large language models",
        "Summarize the key findings in attention is all you need paper",
        "What are the breakthrough papers in computer vision?"
    ],
    "follow_up": [
        "Can you explain that in simpler terms?",
        "Why is that important?",
        "How does that compare to other approaches?",
        "Can you give me an example?",
        "What are the limitations of this approach?"
    ],
    "edge_cases": [
        "What is the meaning of life?",
        "",  # Empty query
        "你好，请问你能用中文回答吗？",  # Non-English query
        "a" * 1000,  # Very long query
        "<script>alert('XSS')</script>"  # Potential injection
    ]
}

class ChatbotTester:
    """Test the ArXiv Expert Chatbot's functionality and accuracy."""
    
    def __init__(self, api_url: str = API_URL, save_results: bool = True):
        """
        Initialize the chatbot tester.
        
        Args:
            api_url: Base URL for the API
            save_results: Whether to save test results to files
        """
        self.api_url = api_url
        self.save_results = save_results
        self.conversation_history = []
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_results": {},
            "category_results": {},
            "query_results": []
        }
        
        # Create results directory if it doesn't exist
        if self.save_results and not os.path.exists(TEST_RESULTS_DIR):
            os.makedirs(TEST_RESULTS_DIR)
    
    def test_chatbot(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test the chatbot with various queries.
        
        Args:
            categories: List of query categories to test, or None for all
            
        Returns:
            Test results dictionary
        """
        if categories is None:
            categories = list(TEST_QUERIES.keys())
        
        total_queries = 0
        successful_queries = 0
        
        # Test each category
        for category in categories:
            if category not in TEST_QUERIES:
                logger.warning(f"Unknown category: {category}")
                continue
            
            logger.info(f"Testing category: {category}")
            
            category_queries = TEST_QUERIES[category]
            category_success = 0
            category_results = []
            
            # Test each query in the category
            for query in category_queries:
                result = self._test_query(query, category)
                category_results.append(result)
                
                if result["success"]:
                    category_success += 1
                    successful_queries += 1
                
                total_queries += 1
                
                # Add to conversation history for follow-up queries
                if category != "follow_up" and result["success"]:
                    self.conversation_history.append({
                        "role": "user",
                        "content": query
                    })
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": result["response"]
                    })
                
                # Small delay to avoid overwhelming the API
                time.sleep(1)
            
            # Calculate category success rate
            category_success_rate = (category_success / len(category_queries)) * 100
            
            # Store category results
            self.test_results["category_results"][category] = {
                "success_rate": category_success_rate,
                "successful_queries": category_success,
                "total_queries": len(category_queries),
                "results": category_results
            }
            
            logger.info(f"Category {category} success rate: {category_success_rate:.2f}%")
        
        # Calculate overall success rate
        overall_success_rate = (successful_queries / total_queries) * 100
        
        # Store overall results
        self.test_results["overall_results"] = {
            "success_rate": overall_success_rate,
            "successful_queries": successful_queries,
            "total_queries": total_queries,
            "tested_categories": categories
        }
        
        logger.info(f"Overall success rate: {overall_success_rate:.2f}%")
        
        # Save results if enabled
        if self.save_results:
            self._save_results()
        
        return self.test_results
    
    def _test_query(self, query: str, category: str) -> Dict[str, Any]:
        """
        Test a single query.
        
        Args:
            query: The query to test
            category: The category of the query
            
        Returns:
            Query test result dictionary
        """
        logger.info(f"Testing query: {query}")
        
        start_time = time.time()
        success = False
        response = ""
        papers = []
        error = None
        
        try:
            # Send the query to the chatbot
            payload = {
                "message": query,
                "conversation_history": self.conversation_history if category == "follow_up" else []
            }
            
            response_obj = requests.post(
                f"{self.api_url}/chat/",
                json=payload,
                timeout=30
            )
            
            # Check if the request was successful
            if response_obj.status_code == 200:
                data = response_obj.json()
                response = data.get("response", "")
                papers = data.get("papers", [])
                
                # Evaluate the response
                evaluation = self._evaluate_response(query, response, papers, category)
                success = evaluation["success"]
                
                logger.info(f"Response evaluation: {'Success' if success else 'Failure'}")
            else:
                error = f"API returned status code {response_obj.status_code}"
                logger.error(error)
        except Exception as e:
            error = str(e)
            logger.error(f"Error testing query: {e}")
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Create result dictionary
        result = {
            "query": query,
            "category": category,
            "success": success,
            "response": response,
            "papers": papers,
            "response_time": response_time,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to query results
        self.test_results["query_results"].append(result)
        
        return result
    
    def _evaluate_response(self, query: str, response: str, papers: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a response.
        
        Args:
            query: The original query
            response: The chatbot's response
            papers: The recommended papers
            category: The category of the query
            
        Returns:
            Evaluation dictionary
        """
        # Basic validation
        if not response:
            return {"success": False, "reason": "Empty response"}
        
        # Check response length
        if len(response) < 50 and category not in ["edge_cases", "follow_up"]:
            return {"success": False, "reason": "Response too short"}
        
        # Check if papers are provided for relevant categories
        if category in ["neural_networks", "machine_learning", "nlp", "algorithms", "research_papers"] and not papers:
            return {"success": False, "reason": "No papers provided"}
        
        # Check for error messages in the response
        error_indicators = ["error", "sorry", "couldn't", "cannot", "failed"]
        if any(indicator in response.lower() for indicator in error_indicators) and category != "edge_cases":
            return {"success": False, "reason": "Error message in response"}
        
        # Check for query keywords in response (except for follow-up and edge cases)
        if category not in ["follow_up", "edge_cases"]:
            query_keywords = [word.lower() for word in query.split() if len(word) > 3]
            if not any(keyword in response.lower() for keyword in query_keywords):
                return {"success": False, "reason": "Response doesn't address query keywords"}
        
        # For follow-up queries, any non-error response is considered successful
        if category == "follow_up":
            return {"success": True, "reason": "Follow-up response provided"}
        
        # For edge cases, any reasonable response is considered successful
        if category == "edge_cases":
            return {"success": True, "reason": "Edge case handled"}
        
        # If we've made it this far, the response is considered successful
        return {"success": True, "reason": "Response meets criteria"}
    
    def _save_results(self):
        """Save test results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(TEST_RESULTS_DIR, f"test_results_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report of the test results.
        
        Returns:
            Report string
        """
        overall = self.test_results["overall_results"]
        categories = self.test_results["category_results"]
        
        report = []
        report.append("=" * 80)
        report.append("ARXIV EXPERT CHATBOT TEST REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.test_results['timestamp']}")
        report.append("")
        
        report.append("OVERALL RESULTS")
        report.append("-" * 80)
        report.append(f"Success Rate: {overall['success_rate']:.2f}%")
        report.append(f"Successful Queries: {overall['successful_queries']} / {overall['total_queries']}")
        report.append(f"Tested Categories: {', '.join(overall['tested_categories'])}")
        report.append("")
        
        report.append("CATEGORY RESULTS")
        report.append("-" * 80)
        for category, results in categories.items():
            report.append(f"Category: {category}")
            report.append(f"  Success Rate: {results['success_rate']:.2f}%")
            report.append(f"  Successful Queries: {results['successful_queries']} / {results['total_queries']}")
            report.append("")
        
        report.append("SAMPLE QUERIES AND RESPONSES")
        report.append("-" * 80)
        # Show a few sample queries and responses
        samples = random.sample(self.test_results["query_results"], min(5, len(self.test_results["query_results"])))
        for i, sample in enumerate(samples):
            report.append(f"Sample {i+1}:")
            report.append(f"  Category: {sample['category']}")
            report.append(f"  Query: {sample['query']}")
            report.append(f"  Success: {'Yes' if sample['success'] else 'No'}")
            report.append(f"  Response Time: {sample['response_time']:.2f} seconds")
            
            # Truncate response if it's too long
            response = sample['response']
            if len(response) > 200:
                response = response[:200] + "..."
            report.append(f"  Response: {response}")
            
            report.append(f"  Papers: {len(sample['papers'])}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function to run the chatbot tester."""
    parser = argparse.ArgumentParser(description="Test the ArXiv Expert Chatbot")
    parser.add_argument("--url", default=API_URL, help="Base URL for the API")
    parser.add_argument("--categories", nargs="+", help="Categories to test")
    parser.add_argument("--no-save", action="store_true", help="Don't save test results")
    args = parser.parse_args()
    
    try:
        # Check if the API is available
        try:
            response = requests.get(f"{args.url}/chat/")
            if response.status_code not in [200, 307, 404]:
                logger.error(f"API not available at {args.url}/chat/")
                return 1
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to API at {args.url}")
            return 1
        
        # Create and run the tester
        tester = ChatbotTester(api_url=args.url, save_results=not args.no_save)
        tester.test_chatbot(categories=args.categories)
        
        # Generate and print the report
        report = tester.generate_report()
        print(report)
        
        # Save the report
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = os.path.join(TEST_RESULTS_DIR, f"test_report_{timestamp}.txt")
            with open(report_filename, 'w') as f:
                f.write(report)
            logger.info(f"Test report saved to {report_filename}")
        
        return 0
    except Exception as e:
        logger.error(f"Error running chatbot tester: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
