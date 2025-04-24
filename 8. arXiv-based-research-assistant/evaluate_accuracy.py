#!/usr/bin/env python
"""
Accuracy evaluation script for the ArXiv Expert Chatbot.
This script evaluates the factual accuracy of the chatbot's responses.
"""

import requests
import json
import time
import argparse
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import csv
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("accuracy_evaluation.log")
    ]
)

logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000/api"
RESULTS_DIR = "accuracy_results"

# Ground truth data for factual accuracy testing
GROUND_TRUTH = [
    {
        "query": "When was the Transformer architecture introduced?",
        "expected_facts": [
            "2017",
            "Vaswani",
            "Attention Is All You Need"
        ],
        "expected_paper_id": "1706.03762"
    },
    {
        "query": "What is BERT?",
        "expected_facts": [
            "Bidirectional",
            "Encoder",
            "Representations",
            "Transformers",
            "pre-training",
            "Devlin"
        ],
        "expected_paper_id": "1810.04805"
    },
    {
        "query": "Explain the ResNet architecture",
        "expected_facts": [
            "residual",
            "skip connections",
            "He",
            "deep networks",
            "gradient vanishing"
        ],
        "expected_paper_id": "1512.03385"
    },
    {
        "query": "What is the Adam optimizer?",
        "expected_facts": [
            "adaptive",
            "moment",
            "estimation",
            "learning rate",
            "Kingma",
            "Ba"
        ],
        "expected_paper_id": "1412.6980"
    },
    {
        "query": "Explain the concept of attention in neural networks",
        "expected_facts": [
            "focus",
            "relevant parts",
            "weight",
            "importance",
            "context"
        ],
        "expected_paper_id": "1706.03762"
    },
    {
        "query": "What is transfer learning?",
        "expected_facts": [
            "pre-trained",
            "knowledge",
            "one task",
            "another task",
            "fine-tuning"
        ],
        "expected_paper_id": None  # Multiple relevant papers
    },
    {
        "query": "Explain the concept of word embeddings",
        "expected_facts": [
            "vector",
            "representation",
            "words",
            "semantic",
            "meaning",
            "space"
        ],
        "expected_paper_id": None  # Multiple relevant papers
    },
    {
        "query": "What is the difference between CNN and RNN?",
        "expected_facts": [
            "convolutional",
            "recurrent",
            "spatial",
            "sequential",
            "images",
            "time series"
        ],
        "expected_paper_id": None  # Multiple relevant papers
    },
    {
        "query": "What is the Transformer encoder-decoder architecture?",
        "expected_facts": [
            "encoder",
            "decoder",
            "self-attention",
            "feed-forward",
            "multi-head"
        ],
        "expected_paper_id": "1706.03762"
    },
    {
        "query": "What is the GPT architecture?",
        "expected_facts": [
            "Generative",
            "Pre-trained",
            "Transformer",
            "autoregressive",
            "OpenAI"
        ],
        "expected_paper_id": None  # Could be multiple GPT papers
    }
]

class AccuracyEvaluator:
    """Evaluate the factual accuracy of the ArXiv Expert Chatbot."""
    
    def __init__(self, api_url: str = API_URL, save_results: bool = True):
        """
        Initialize the accuracy evaluator.
        
        Args:
            api_url: Base URL for the API
            save_results: Whether to save evaluation results to files
        """
        self.api_url = api_url
        self.save_results = save_results
        self.evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_accuracy": 0.0,
            "query_results": []
        }
        
        # Create results directory if it doesn't exist
        if self.save_results and not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
    
    def evaluate_accuracy(self, ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate the factual accuracy of the chatbot.
        
        Args:
            ground_truth: List of ground truth data, or None to use default
            
        Returns:
            Evaluation results dictionary
        """
        if ground_truth is None:
            ground_truth = GROUND_TRUTH
        
        total_facts = 0
        correct_facts = 0
        total_papers = 0
        correct_papers = 0
        
        # Evaluate each query
        for item in ground_truth:
            query = item["query"]
            expected_facts = item["expected_facts"]
            expected_paper_id = item["expected_paper_id"]
            
            logger.info(f"Evaluating query: {query}")
            
            # Get response from chatbot
            response_text, papers = self._get_chatbot_response(query)
            
            # Evaluate factual accuracy
            fact_results = self._evaluate_facts(response_text, expected_facts)
            paper_result = self._evaluate_paper(papers, expected_paper_id)
            
            # Update counters
            total_facts += len(expected_facts)
            correct_facts += fact_results["correct_count"]
            
            if expected_paper_id is not None:
                total_papers += 1
                if paper_result["correct"]:
                    correct_papers += 1
            
            # Store query result
            query_result = {
                "query": query,
                "response": response_text,
                "papers": papers,
                "fact_results": fact_results,
                "paper_result": paper_result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.evaluation_results["query_results"].append(query_result)
            
            # Small delay to avoid overwhelming the API
            time.sleep(1)
        
        # Calculate overall accuracy
        fact_accuracy = (correct_facts / total_facts) * 100 if total_facts > 0 else 0
        paper_accuracy = (correct_papers / total_papers) * 100 if total_papers > 0 else 0
        overall_accuracy = (fact_accuracy + paper_accuracy) / 2 if total_papers > 0 else fact_accuracy
        
        # Store overall results
        self.evaluation_results["overall_accuracy"] = overall_accuracy
        self.evaluation_results["fact_accuracy"] = fact_accuracy
        self.evaluation_results["paper_accuracy"] = paper_accuracy
        self.evaluation_results["correct_facts"] = correct_facts
        self.evaluation_results["total_facts"] = total_facts
        self.evaluation_results["correct_papers"] = correct_papers
        self.evaluation_results["total_papers"] = total_papers
        
        logger.info(f"Overall accuracy: {overall_accuracy:.2f}%")
        logger.info(f"Fact accuracy: {fact_accuracy:.2f}%")
        logger.info(f"Paper accuracy: {paper_accuracy:.2f}%")
        
        # Save results if enabled
        if self.save_results:
            self._save_results()
        
        return self.evaluation_results
    
    def _get_chatbot_response(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get a response from the chatbot.
        
        Args:
            query: The query to send
            
        Returns:
            Tuple of (response text, papers)
        """
        try:
            # Send the query to the chatbot
            payload = {
                "message": query,
                "conversation_history": []
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
                return response, papers
            else:
                logger.error(f"API returned status code {response_obj.status_code}")
                return f"Error: API returned status code {response_obj.status_code}", []
        except Exception as e:
            logger.error(f"Error getting chatbot response: {e}")
            return f"Error: {str(e)}", []
    
    def _evaluate_facts(self, response: str, expected_facts: List[str]) -> Dict[str, Any]:
        """
        Evaluate the factual accuracy of a response.
        
        Args:
            response: The chatbot's response
            expected_facts: List of expected facts
            
        Returns:
            Fact evaluation dictionary
        """
        correct_facts = []
        missing_facts = []
        
        # Check for each expected fact
        for fact in expected_facts:
            if re.search(r'\b' + re.escape(fact) + r'\b', response, re.IGNORECASE):
                correct_facts.append(fact)
            else:
                missing_facts.append(fact)
        
        # Calculate accuracy
        accuracy = (len(correct_facts) / len(expected_facts)) * 100 if expected_facts else 0
        
        return {
            "accuracy": accuracy,
            "correct_facts": correct_facts,
            "missing_facts": missing_facts,
            "correct_count": len(correct_facts),
            "total_count": len(expected_facts)
        }
    
    def _evaluate_paper(self, papers: List[Dict[str, Any]], expected_paper_id: Optional[str]) -> Dict[str, Any]:
        """
        Evaluate if the expected paper is in the results.
        
        Args:
            papers: List of papers returned by the chatbot
            expected_paper_id: Expected paper ID, or None if not applicable
            
        Returns:
            Paper evaluation dictionary
        """
        if expected_paper_id is None:
            return {
                "correct": True,
                "reason": "No specific paper expected"
            }
        
        # Check if the expected paper is in the results
        paper_ids = [paper.get("paper_id", "") for paper in papers]
        correct = expected_paper_id in paper_ids
        
        return {
            "correct": correct,
            "reason": f"Paper {expected_paper_id} {'found' if correct else 'not found'} in results",
            "found_papers": paper_ids
        }
    
    def _save_results(self):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = os.path.join(RESULTS_DIR, f"accuracy_results_{timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Save CSV summary
        csv_filename = os.path.join(RESULTS_DIR, f"accuracy_summary_{timestamp}.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Query", "Fact Accuracy", "Paper Correct", "Missing Facts"])
            
            for result in self.evaluation_results["query_results"]:
                writer.writerow([
                    result["query"],
                    f"{result['fact_results']['accuracy']:.2f}%",
                    "Yes" if result["paper_result"]["correct"] else "No",
                    ", ".join(result["fact_results"]["missing_facts"])
                ])
        
        logger.info(f"Evaluation results saved to {json_filename} and {csv_filename}")
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report of the evaluation results.
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("ARXIV EXPERT CHATBOT ACCURACY EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.evaluation_results['timestamp']}")
        report.append("")
        
        report.append("OVERALL RESULTS")
        report.append("-" * 80)
        report.append(f"Overall Accuracy: {self.evaluation_results['overall_accuracy']:.2f}%")
        report.append(f"Fact Accuracy: {self.evaluation_results['fact_accuracy']:.2f}%")
        report.append(f"Paper Accuracy: {self.evaluation_results['paper_accuracy']:.2f}%")
        report.append(f"Correct Facts: {self.evaluation_results['correct_facts']} / {self.evaluation_results['total_facts']}")
        report.append(f"Correct Papers: {self.evaluation_results['correct_papers']} / {self.evaluation_results['total_papers']}")
        report.append("")
        
        report.append("QUERY RESULTS")
        report.append("-" * 80)
        for result in self.evaluation_results["query_results"]:
            report.append(f"Query: {result['query']}")
            report.append(f"Fact Accuracy: {result['fact_results']['accuracy']:.2f}%")
            report.append(f"Correct Facts: {result['fact_results']['correct_count']} / {result['fact_results']['total_count']}")
            
            if result["fact_results"]["missing_facts"]:
                report.append(f"Missing Facts: {', '.join(result['fact_results']['missing_facts'])}")
            
            report.append(f"Paper Correct: {'Yes' if result['paper_result']['correct'] else 'No'}")
            
            # Truncate response if it's too long
            response = result['response']
            if len(response) > 200:
                response = response[:200] + "..."
            report.append(f"Response: {response}")
            
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function to run the accuracy evaluator."""
    parser = argparse.ArgumentParser(description="Evaluate the factual accuracy of the ArXiv Expert Chatbot")
    parser.add_argument("--url", default=API_URL, help="Base URL for the API")
    parser.add_argument("--no-save", action="store_true", help="Don't save evaluation results")
    parser.add_argument("--custom-data", help="Path to custom ground truth data JSON file")
    args = parser.parse_args()
    
    try:
        # Load custom ground truth data if provided
        ground_truth = None
        if args.custom_data:
            try:
                with open(args.custom_data, 'r') as f:
                    ground_truth = json.load(f)
                logger.info(f"Loaded custom ground truth data from {args.custom_data}")
            except Exception as e:
                logger.error(f"Error loading custom ground truth data: {e}")
                return 1
        
        # Check if the API is available
        try:
            response = requests.get(f"{args.url}/chat/")
            if response.status_code not in [200, 307, 404]:
                logger.error(f"API not available at {args.url}/chat/")
                return 1
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to API at {args.url}")
            return 1
        
        # Create and run the evaluator
        evaluator = AccuracyEvaluator(api_url=args.url, save_results=not args.no_save)
        evaluator.evaluate_accuracy(ground_truth=ground_truth)
        
        # Generate and print the report
        report = evaluator.generate_report()
        print(report)
        
        # Save the report
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = os.path.join(RESULTS_DIR, f"accuracy_report_{timestamp}.txt")
            with open(report_filename, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {report_filename}")
        
        return 0
    except Exception as e:
        logger.error(f"Error running accuracy evaluator: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
