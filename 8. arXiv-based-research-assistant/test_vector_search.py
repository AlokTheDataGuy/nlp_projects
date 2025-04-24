#!/usr/bin/env python
"""
Test script for the vector search functionality of the ArXiv Expert Chatbot.
This script evaluates the relevance of search results for various queries.
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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vector_search_test.log")
    ]
)

logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000/api"
RESULTS_DIR = "vector_search_results"

# Test queries for vector search
TEST_QUERIES = [
    {
        "query": "How do transformers work in natural language processing?",
        "keywords": ["transformer", "attention", "nlp", "language", "model"]
    },
    {
        "query": "What are the latest advancements in deep learning?",
        "keywords": ["deep learning", "neural network", "advancement", "recent"]
    },
    {
        "query": "Explain the concept of reinforcement learning",
        "keywords": ["reinforcement learning", "reward", "agent", "environment", "policy"]
    },
    {
        "query": "How do convolutional neural networks process images?",
        "keywords": ["cnn", "convolutional", "image", "filter", "kernel"]
    },
    {
        "query": "What is the difference between supervised and unsupervised learning?",
        "keywords": ["supervised", "unsupervised", "learning", "labeled", "unlabeled"]
    },
    {
        "query": "Explain the concept of transfer learning in deep neural networks",
        "keywords": ["transfer learning", "pre-trained", "fine-tuning", "knowledge"]
    },
    {
        "query": "What are graph neural networks and how do they work?",
        "keywords": ["graph", "neural network", "node", "edge", "message passing"]
    },
    {
        "query": "How does BERT handle natural language understanding?",
        "keywords": ["bert", "bidirectional", "transformer", "language", "understanding"]
    },
    {
        "query": "What are the challenges in computer vision?",
        "keywords": ["computer vision", "challenge", "image", "recognition", "detection"]
    },
    {
        "query": "Explain the concept of generative adversarial networks",
        "keywords": ["gan", "generative", "adversarial", "generator", "discriminator"]
    }
]

class VectorSearchTester:
    """Test the vector search functionality of the ArXiv Expert Chatbot."""
    
    def __init__(self, api_url: str = API_URL, save_results: bool = True):
        """
        Initialize the vector search tester.
        
        Args:
            api_url: Base URL for the API
            save_results: Whether to save test results to files
        """
        self.api_url = api_url
        self.save_results = save_results
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_results": {},
            "query_results": []
        }
        
        # Load sentence transformer model for semantic similarity evaluation
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            self.model = None
        
        # Create results directory if it doesn't exist
        if self.save_results and not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
    
    def test_vector_search(self, queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Test the vector search functionality.
        
        Args:
            queries: List of test queries, or None to use default
            
        Returns:
            Test results dictionary
        """
        if queries is None:
            queries = TEST_QUERIES
        
        total_relevance_score = 0.0
        total_keyword_score = 0.0
        total_semantic_score = 0.0
        
        # Test each query
        for query_data in queries:
            query = query_data["query"]
            keywords = query_data["keywords"]
            
            logger.info(f"Testing query: {query}")
            
            # Get response from chatbot
            response_text, papers = self._get_chatbot_response(query)
            
            # Evaluate relevance of papers
            relevance_results = self._evaluate_relevance(query, papers, keywords)
            
            # Store query result
            query_result = {
                "query": query,
                "keywords": keywords,
                "response": response_text,
                "papers": papers,
                "relevance_results": relevance_results,
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results["query_results"].append(query_result)
            
            # Update total scores
            total_relevance_score += relevance_results["overall_relevance"]
            total_keyword_score += relevance_results["keyword_score"]
            if "semantic_score" in relevance_results:
                total_semantic_score += relevance_results["semantic_score"]
            
            # Small delay to avoid overwhelming the API
            time.sleep(1)
        
        # Calculate average scores
        avg_relevance_score = total_relevance_score / len(queries)
        avg_keyword_score = total_keyword_score / len(queries)
        avg_semantic_score = total_semantic_score / len(queries) if self.model else 0.0
        
        # Store overall results
        self.test_results["overall_results"] = {
            "average_relevance_score": avg_relevance_score,
            "average_keyword_score": avg_keyword_score,
            "average_semantic_score": avg_semantic_score,
            "total_queries": len(queries)
        }
        
        logger.info(f"Average relevance score: {avg_relevance_score:.2f}")
        logger.info(f"Average keyword score: {avg_keyword_score:.2f}")
        if self.model:
            logger.info(f"Average semantic score: {avg_semantic_score:.2f}")
        
        # Save results if enabled
        if self.save_results:
            self._save_results()
        
        return self.test_results
    
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
    
    def _evaluate_relevance(self, query: str, papers: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluate the relevance of papers to the query.
        
        Args:
            query: The original query
            papers: The papers returned by the chatbot
            keywords: Keywords related to the query
            
        Returns:
            Relevance evaluation dictionary
        """
        if not papers:
            return {
                "overall_relevance": 0.0,
                "keyword_score": 0.0,
                "semantic_score": 0.0 if self.model else None,
                "paper_scores": []
            }
        
        paper_scores = []
        total_keyword_score = 0.0
        total_semantic_score = 0.0
        
        # Evaluate each paper
        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            categories = paper.get("categories", "")
            
            # Calculate keyword relevance
            keyword_score = self._calculate_keyword_score(title, abstract, categories, keywords)
            
            # Calculate semantic similarity if model is available
            semantic_score = None
            if self.model:
                semantic_score = self._calculate_semantic_score(query, title, abstract)
            
            # Calculate overall score for this paper
            if semantic_score is not None:
                overall_score = (keyword_score + semantic_score) / 2
            else:
                overall_score = keyword_score
            
            # Store paper score
            paper_score = {
                "paper_id": paper.get("paper_id", ""),
                "title": title,
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
                "overall_score": overall_score
            }
            
            paper_scores.append(paper_score)
            
            # Update total scores
            total_keyword_score += keyword_score
            if semantic_score is not None:
                total_semantic_score += semantic_score
        
        # Calculate average scores
        avg_keyword_score = total_keyword_score / len(papers)
        avg_semantic_score = total_semantic_score / len(papers) if self.model else None
        
        # Calculate overall relevance
        if avg_semantic_score is not None:
            overall_relevance = (avg_keyword_score + avg_semantic_score) / 2
        else:
            overall_relevance = avg_keyword_score
        
        return {
            "overall_relevance": overall_relevance,
            "keyword_score": avg_keyword_score,
            "semantic_score": avg_semantic_score,
            "paper_scores": paper_scores
        }
    
    def _calculate_keyword_score(self, title: str, abstract: str, categories: str, keywords: List[str]) -> float:
        """
        Calculate keyword relevance score.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            categories: Paper categories
            keywords: Query keywords
            
        Returns:
            Keyword relevance score (0.0 to 1.0)
        """
        if not keywords:
            return 0.0
        
        # Combine text for keyword search
        text = (title + " " + abstract + " " + categories).lower()
        
        # Count keyword occurrences
        keyword_count = 0
        for keyword in keywords:
            if keyword.lower() in text:
                keyword_count += 1
        
        # Calculate score
        return keyword_count / len(keywords)
    
    def _calculate_semantic_score(self, query: str, title: str, abstract: str) -> float:
        """
        Calculate semantic similarity score.
        
        Args:
            query: Original query
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            Semantic similarity score (0.0 to 1.0)
        """
        if not self.model:
            return 0.0
        
        try:
            # Combine title and abstract
            paper_text = title + ". " + abstract
            
            # Encode query and paper text
            query_embedding = self.model.encode([query])[0]
            paper_embedding = self.model.encode([paper_text])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                paper_embedding.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic score: {e}")
            return 0.0
    
    def _save_results(self):
        """Save test results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"vector_search_results_{timestamp}.json")
        
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
        
        report = []
        report.append("=" * 80)
        report.append("ARXIV EXPERT CHATBOT VECTOR SEARCH TEST REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.test_results['timestamp']}")
        report.append("")
        
        report.append("OVERALL RESULTS")
        report.append("-" * 80)
        report.append(f"Average Relevance Score: {overall['average_relevance_score']:.2f}")
        report.append(f"Average Keyword Score: {overall['average_keyword_score']:.2f}")
        if 'average_semantic_score' in overall and overall['average_semantic_score'] > 0:
            report.append(f"Average Semantic Score: {overall['average_semantic_score']:.2f}")
        report.append(f"Total Queries: {overall['total_queries']}")
        report.append("")
        
        report.append("QUERY RESULTS")
        report.append("-" * 80)
        for result in self.test_results["query_results"]:
            report.append(f"Query: {result['query']}")
            report.append(f"Keywords: {', '.join(result['keywords'])}")
            report.append(f"Overall Relevance: {result['relevance_results']['overall_relevance']:.2f}")
            report.append(f"Keyword Score: {result['relevance_results']['keyword_score']:.2f}")
            if 'semantic_score' in result['relevance_results'] and result['relevance_results']['semantic_score'] is not None:
                report.append(f"Semantic Score: {result['relevance_results']['semantic_score']:.2f}")
            
            report.append("Top Papers:")
            for i, paper_score in enumerate(sorted(result['relevance_results']['paper_scores'], key=lambda x: x['overall_score'], reverse=True)[:3]):
                report.append(f"  {i+1}. {paper_score['title']} (Score: {paper_score['overall_score']:.2f})")
            
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function to run the vector search tester."""
    parser = argparse.ArgumentParser(description="Test the vector search functionality of the ArXiv Expert Chatbot")
    parser.add_argument("--url", default=API_URL, help="Base URL for the API")
    parser.add_argument("--no-save", action="store_true", help="Don't save test results")
    parser.add_argument("--custom-queries", help="Path to custom queries JSON file")
    args = parser.parse_args()
    
    try:
        # Load custom queries if provided
        queries = None
        if args.custom_queries:
            try:
                with open(args.custom_queries, 'r') as f:
                    queries = json.load(f)
                logger.info(f"Loaded custom queries from {args.custom_queries}")
            except Exception as e:
                logger.error(f"Error loading custom queries: {e}")
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
        
        # Create and run the tester
        tester = VectorSearchTester(api_url=args.url, save_results=not args.no_save)
        tester.test_vector_search(queries=queries)
        
        # Generate and print the report
        report = tester.generate_report()
        print(report)
        
        # Save the report
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = os.path.join(RESULTS_DIR, f"vector_search_report_{timestamp}.txt")
            with open(report_filename, 'w') as f:
                f.write(report)
            logger.info(f"Test report saved to {report_filename}")
        
        return 0
    except Exception as e:
        logger.error(f"Error running vector search tester: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
