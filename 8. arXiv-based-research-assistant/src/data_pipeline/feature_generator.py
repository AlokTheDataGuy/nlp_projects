"""
Feature Generator Module

This module identifies key entities, concepts, and relationships from processed papers.
"""

import os
import re
import logging
import yaml
import spacy
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Class for identifying key entities, concepts, and relationships from processed papers.
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the FeatureGenerator.
        
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
        
        # Create directories if they don't exist
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def extract_key_phrases(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract key phrases from text.
        
        Args:
            text: The text to extract key phrases from.
            top_n: Number of top phrases to return.
            
        Returns:
            List of key phrases.
        """
        doc = self.nlp(text)
        
        # Extract noun chunks and filter by length
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2]
        
        # Count occurrences
        phrase_counts = Counter(noun_chunks)
        
        # Filter by CS domain relevance
        cs_relevant_phrases = []
        for phrase, count in phrase_counts.items():
            # Check if the phrase is a CS term or contains CS terms
            is_cs_relevant = phrase in self.cs_terms or any(term in phrase for term in self.cs_terms)
            
            if is_cs_relevant:
                cs_relevant_phrases.append((phrase, count))
        
        # Sort by count and take top N
        top_phrases = [phrase for phrase, _ in sorted(cs_relevant_phrases, key=lambda x: x[1], reverse=True)[:top_n]]
        
        return top_phrases
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """
        Extract technical terms from text.
        
        Args:
            text: The text to extract technical terms from.
            
        Returns:
            List of technical terms.
        """
        # Pattern for technical terms (e.g., acronyms, camelCase, terms with numbers)
        patterns = [
            r'\b[A-Z][A-Za-z]*[0-9]+[A-Za-z0-9]*\b',  # Terms with numbers (e.g., GPT3, BERT2)
            r'\b[A-Z]{2,}\b',  # Acronyms (e.g., CNN, RNN, LSTM)
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # camelCase terms
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b'  # PascalCase terms
        ]
        
        technical_terms = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            technical_terms.update(matches)
        
        # Filter out common words and short terms
        filtered_terms = [term for term in technical_terms if len(term) > 2]
        
        return list(filtered_terms)
    
    def identify_relationships(self, paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify relationships between concepts in a paper.
        
        Args:
            paper_data: Dictionary containing paper data.
            
        Returns:
            List of relationships.
        """
        relationships = []
        
        # Extract key phrases from abstract and full text
        abstract_phrases = self.extract_key_phrases(paper_data["abstract"], top_n=10)
        full_text_phrases = self.extract_key_phrases(paper_data["full_text"], top_n=30)
        
        # Combine phrases
        all_phrases = list(set(abstract_phrases + full_text_phrases))
        
        # Identify co-occurrence relationships
        for i, phrase1 in enumerate(all_phrases):
            for phrase2 in all_phrases[i+1:]:
                # Check if phrases co-occur in the same sentences
                co_occurrence_count = 0
                
                # Split text into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paper_data["full_text"])
                
                for sentence in sentences:
                    if phrase1 in sentence.lower() and phrase2 in sentence.lower():
                        co_occurrence_count += 1
                
                if co_occurrence_count > 0:
                    relationships.append({
                        "source": phrase1,
                        "target": phrase2,
                        "type": "co-occurrence",
                        "weight": co_occurrence_count
                    })
        
        return relationships
    
    def process_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a paper and extract features.
        
        Args:
            paper_data: Dictionary containing paper data.
            
        Returns:
            Dictionary with paper data and extracted features.
        """
        # Extract entities from abstract and full text
        abstract_entities = self.extract_entities(paper_data["abstract"])
        full_text_entities = self.extract_entities(paper_data["full_text"])
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(paper_data["full_text"])
        
        # Extract technical terms
        technical_terms = self.extract_technical_terms(paper_data["full_text"])
        
        # Identify relationships
        relationships = self.identify_relationships(paper_data)
        
        # Add features to paper data
        paper_data["features"] = {
            "abstract_entities": abstract_entities,
            "full_text_entities": full_text_entities,
            "key_phrases": key_phrases,
            "technical_terms": technical_terms,
            "relationships": relationships
        }
        
        # Save updated paper data
        paper_id = paper_data["id"]
        output_path = self.processed_dir / f"{paper_id}.yaml"
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(paper_data, f, default_flow_style=False, allow_unicode=True)
        
        return paper_data
    
    def process_papers(self, papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple papers.
        
        Args:
            papers_data: List of dictionaries containing paper data.
            
        Returns:
            List of dictionaries with paper data and extracted features.
        """
        processed_papers = []
        
        for paper_data in tqdm(papers_data, desc="Extracting features"):
            try:
                processed_paper = self.process_paper(paper_data)
                processed_papers.append(processed_paper)
            except Exception as e:
                logger.error(f"Error extracting features from paper {paper_data['id']}: {str(e)}")
        
        return processed_papers


if __name__ == "__main__":
    # Example usage
    import glob
    
    # Load processed papers
    processed_dir = Path("data/processed")
    paper_files = glob.glob(str(processed_dir / "*.yaml"))
    
    papers_data = []
    for paper_file in paper_files:
        with open(paper_file, 'r', encoding='utf-8') as f:
            paper_data = yaml.safe_load(f)
            papers_data.append(paper_data)
    
    feature_generator = FeatureGenerator()
    processed_papers = feature_generator.process_papers(papers_data)
    
    print(f"Extracted features from {len(processed_papers)} papers")
