"""
Query Analyzer Module

This module classifies questions and extracts key entities.
"""

import os
import logging
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """
    Class for classifying questions and extracting key entities.
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the QueryAnalyzer.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model: en_core_web_sm")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define question types
        self.question_types = {
            "definition": [
                r"what (is|are) (.+)\??",
                r"define (.+)\??",
                r"explain (.+)\??",
                r"describe (.+)\??"
            ],
            "comparison": [
                r"(compare|difference between|similarities between) (.+) and (.+)\??",
                r"how (does|do) (.+) (compare|differ|relate) to (.+)\??",
                r"what (is|are) the (difference|differences|similarity|similarities) between (.+) and (.+)\??"
            ],
            "application": [
                r"how (can|could|is|are) (.+) (used|applied|implemented|utilized)\??",
                r"what (is|are) the (application|applications|use|uses) of (.+)\??",
                r"how (does|do) (.+) work\??"
            ],
            "advantages": [
                r"what (is|are) the (advantage|advantages|benefit|benefits) of (.+)\??",
                r"why (use|choose|select|prefer) (.+)\??",
                r"what makes (.+) (better|good|effective|useful)\??"
            ],
            "disadvantages": [
                r"what (is|are) the (disadvantage|disadvantages|limitation|limitations|drawback|drawbacks) of (.+)\??",
                r"why not (use|choose|select|prefer) (.+)\??",
                r"what (is|are) the (problem|problems|issue|issues|challenge|challenges) with (.+)\??"
            ],
            "timeline": [
                r"when (was|were) (.+) (developed|created|introduced|discovered|invented)\??",
                r"what is the history of (.+)\??",
                r"how has (.+) evolved\??"
            ],
            "state_of_the_art": [
                r"what (is|are) the (state of the art|latest|newest|most recent|current) (.+)\??",
                r"what (is|are) the (best|top|leading) (.+)\??",
                r"what advances have been made in (.+)\??"
            ],
            "future": [
                r"what (is|are) the future of (.+)\??",
                r"how will (.+) (evolve|change|develop|improve) in the future\??",
                r"what (is|are) the (potential|possibilities) of (.+)\??"
            ],
            "factual": [
                r"who (created|developed|invented|discovered) (.+)\??",
                r"where (is|are|was|were) (.+) (used|developed|created|implemented)\??",
                r"how many (.+)\??"
            ]
        }
        
        # Load CS domain-specific terms
        self.cs_terms = self._load_cs_terms()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dict containing configuration.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_cs_terms(self) -> Set[str]:
        """
        Load Computer Science domain-specific terms.
        
        Returns:
            Set of CS terms.
        """
        # Common CS terms (this is a small sample, in a real system this would be much more comprehensive)
        cs_terms = {
            # AI/ML terms
            "machine learning", "deep learning", "neural network", "artificial intelligence",
            "supervised learning", "unsupervised learning", "reinforcement learning",
            "convolutional neural network", "recurrent neural network", "transformer",
            "attention mechanism", "backpropagation", "gradient descent", "overfitting",
            "underfitting", "regularization", "hyperparameter", "feature extraction",
            
            # NLP terms
            "natural language processing", "tokenization", "lemmatization", "stemming",
            "named entity recognition", "part-of-speech tagging", "sentiment analysis",
            "text classification", "language model", "word embedding", "sequence-to-sequence",
            
            # Computer Vision terms
            "computer vision", "image processing", "object detection", "image segmentation",
            "feature detection", "edge detection", "facial recognition", "pose estimation",
            
            # Systems/Architecture terms
            "distributed systems", "parallel computing", "cloud computing", "edge computing",
            "microservices", "serverless", "container", "virtualization", "scalability",
            
            # Programming terms
            "algorithm", "data structure", "complexity", "big o notation", "recursion",
            "dynamic programming", "greedy algorithm", "sorting", "searching", "hashing",
            
            # Database terms
            "database", "sql", "nosql", "indexing", "query optimization", "transaction",
            "acid", "sharding", "replication", "data warehouse", "data lake",
            
            # Security terms
            "cybersecurity", "encryption", "authentication", "authorization", "vulnerability",
            "exploit", "firewall", "intrusion detection", "penetration testing",
            
            # Networking terms
            "network", "protocol", "tcp/ip", "http", "dns", "routing", "load balancing",
            "cdn", "vpn", "bandwidth", "latency", "throughput"
        }
        
        return cs_terms
    
    def classify_question(self, query: str) -> Dict[str, Any]:
        """
        Classify the question type and extract key entities.
        
        Args:
            query: The query to classify.
            
        Returns:
            Dictionary containing classification results.
        """
        # Normalize query
        normalized_query = query.lower().strip()
        if not normalized_query.endswith("?"):
            normalized_query += "?"
        
        # Classify question type
        question_type = "general"
        matched_pattern = None
        
        for q_type, patterns in self.question_types.items():
            for pattern in patterns:
                match = re.match(pattern, normalized_query, re.IGNORECASE)
                if match:
                    question_type = q_type
                    matched_pattern = pattern
                    break
            if matched_pattern:
                break
        
        # Extract entities
        entities = self.extract_entities(query)
        
        # Extract key terms
        key_terms = self.extract_key_terms(query)
        
        # Determine domain relevance
        is_cs_relevant = self.is_cs_relevant(query)
        
        # Create classification result
        classification = {
            "query": query,
            "question_type": question_type,
            "matched_pattern": matched_pattern,
            "entities": entities,
            "key_terms": key_terms,
            "is_cs_relevant": is_cs_relevant
        }
        
        return classification
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: The text to extract entities from.
            
        Returns:
            Dictionary with entity types as keys and lists of entities as values.
        """
        doc = self.nlp(text)
        
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            # Normalize entity text
            entity_text = ent.text.lower().strip()
            
            # Add entity if it's not already in the list
            if entity_text and entity_text not in entities[ent.label_]:
                entities[ent.label_].append(entity_text)
        
        return entities
    
    def extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: The text to extract key terms from.
            
        Returns:
            List of key terms.
        """
        doc = self.nlp(text)
        
        # Extract noun chunks and filter by length
        key_terms = []
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            term = chunk.text.lower().strip()
            if term and term not in key_terms:
                key_terms.append(term)
        
        # Add technical terms
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in key_terms:
                key_terms.append(token.text.lower())
        
        return key_terms
    
    def is_cs_relevant(self, text: str) -> bool:
        """
        Determine if the text is relevant to Computer Science.
        
        Args:
            text: The text to check.
            
        Returns:
            True if the text is CS-relevant, False otherwise.
        """
        text_lower = text.lower()
        
        # Check if any CS terms are in the text
        for term in self.cs_terms:
            if term in text_lower:
                return True
        
        return False
    
    def expand_query(self, query: str) -> str:
        """
        Expand the query with related terms to improve retrieval.
        
        Args:
            query: The query to expand.
            
        Returns:
            Expanded query.
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated techniques like word embeddings
        
        # Extract key terms
        key_terms = self.extract_key_terms(query)
        
        # Add related terms for common CS concepts
        expanded_terms = []
        
        for term in key_terms:
            expanded_terms.append(term)
            
            # Add related terms for common concepts
            if "neural network" in term:
                expanded_terms.extend(["deep learning", "artificial neural network", "ANN"])
            elif "transformer" in term:
                expanded_terms.extend(["attention mechanism", "self-attention", "encoder-decoder"])
            elif "machine learning" in term:
                expanded_terms.extend(["ML", "AI", "artificial intelligence", "deep learning"])
            elif "natural language processing" in term:
                expanded_terms.extend(["NLP", "computational linguistics", "text processing"])
            elif "computer vision" in term:
                expanded_terms.extend(["CV", "image processing", "visual recognition"])
        
        # Remove duplicates and join
        expanded_query = " ".join(list(dict.fromkeys(expanded_terms)))
        
        return expanded_query


if __name__ == "__main__":
    # Example usage
    query_analyzer = QueryAnalyzer()
    
    # Test queries
    test_queries = [
        "What is a transformer in deep learning?",
        "Compare BERT and GPT models",
        "How are neural networks used in computer vision?",
        "What are the advantages of using attention mechanisms?",
        "What are the limitations of convolutional neural networks?",
        "When was the transformer architecture introduced?",
        "What is the state of the art in natural language processing?",
        "What is the future of reinforcement learning?",
        "Who invented the transformer architecture?"
    ]
    
    for query in test_queries:
        classification = query_analyzer.classify_question(query)
        
        print(f"Query: {query}")
        print(f"Question type: {classification['question_type']}")
        print(f"Key terms: {classification['key_terms']}")
        print(f"CS relevant: {classification['is_cs_relevant']}")
        
        expanded_query = query_analyzer.expand_query(query)
        print(f"Expanded query: {expanded_query}")
        print()
