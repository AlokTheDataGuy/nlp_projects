"""
Knowledge Graph Module

This module provides functionality for building and querying a knowledge graph
of concepts and relationships extracted from papers.
"""

import os
import logging
import yaml
import json
import networkx as nx
from typing import Dict, List, Any, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Class for building and querying a knowledge graph of concepts and relationships.
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the KnowledgeGraph.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        
        # Create directories if they don't exist
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the graph
        self.graph = nx.DiGraph()
    
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
    
    def build_graph_from_papers(self, papers_data: List[Dict[str, Any]]) -> None:
        """
        Build a knowledge graph from paper data.
        
        Args:
            papers_data: List of dictionaries containing paper data with features.
        """
        # Clear the existing graph
        self.graph.clear()
        
        # Process each paper
        for paper in papers_data:
            paper_id = paper["id"]
            title = paper["title"]
            
            # Add paper node
            self.graph.add_node(paper_id, type="paper", title=title, authors=paper["authors"])
            
            # Process features if available
            if "features" in paper:
                features = paper["features"]
                
                # Add key phrases as concept nodes
                if "key_phrases" in features:
                    for phrase in features["key_phrases"]:
                        # Add concept node if it doesn't exist
                        if not self.graph.has_node(phrase):
                            self.graph.add_node(phrase, type="concept")
                        
                        # Add edge from paper to concept
                        self.graph.add_edge(paper_id, phrase, type="contains")
                
                # Add technical terms as concept nodes
                if "technical_terms" in features:
                    for term in features["technical_terms"]:
                        # Add concept node if it doesn't exist
                        if not self.graph.has_node(term):
                            self.graph.add_node(term, type="technical_term")
                        
                        # Add edge from paper to concept
                        self.graph.add_edge(paper_id, term, type="contains")
                
                # Add relationships between concepts
                if "relationships" in features:
                    for relationship in features["relationships"]:
                        source = relationship["source"]
                        target = relationship["target"]
                        rel_type = relationship["type"]
                        weight = relationship["weight"]
                        
                        # Add concept nodes if they don't exist
                        if not self.graph.has_node(source):
                            self.graph.add_node(source, type="concept")
                        if not self.graph.has_node(target):
                            self.graph.add_node(target, type="concept")
                        
                        # Add edge between concepts
                        if self.graph.has_edge(source, target):
                            # Update weight if edge already exists
                            self.graph[source][target]["weight"] += weight
                        else:
                            self.graph.add_edge(source, target, type=rel_type, weight=weight)
        
        logger.info(f"Built knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Save the graph
        self.save_graph()
    
    def save_graph(self, file_path: Optional[str] = None) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            file_path: Path to save the graph to. If None, uses the default path.
        """
        if file_path is None:
            file_path = self.embeddings_dir / "knowledge_graph.json"
        
        # Convert the graph to a dictionary
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {
                "id": node,
                **attrs
            }
            graph_data["nodes"].append(node_data)
        
        # Add edges
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                **attrs
            }
            graph_data["edges"].append(edge_data)
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Saved knowledge graph to {file_path}")
    
    def load_graph(self, file_path: Optional[str] = None) -> bool:
        """
        Load the knowledge graph from a file.
        
        Args:
            file_path: Path to load the graph from. If None, uses the default path.
            
        Returns:
            True if the graph was loaded successfully, False otherwise.
        """
        if file_path is None:
            file_path = self.embeddings_dir / "knowledge_graph.json"
        
        if not os.path.exists(file_path):
            logger.error(f"Graph file {file_path} not found")
            return False
        
        try:
            # Load the graph data
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Clear the existing graph
            self.graph.clear()
            
            # Add nodes
            for node_data in graph_data["nodes"]:
                node_id = node_data.pop("id")
                self.graph.add_node(node_id, **node_data)
            
            # Add edges
            for edge_data in graph_data["edges"]:
                source = edge_data.pop("source")
                target = edge_data.pop("target")
                self.graph.add_edge(source, target, **edge_data)
            
            logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            return False
    
    def get_related_concepts(self, concept: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept: The concept to find related concepts for.
            max_distance: Maximum distance in the graph to search for related concepts.
            
        Returns:
            List of dictionaries containing related concepts and their relationships.
        """
        if not self.graph.has_node(concept):
            logger.warning(f"Concept {concept} not found in the graph")
            return []
        
        related_concepts = []
        
        # Get all nodes within max_distance of the concept
        for node in nx.single_source_shortest_path_length(self.graph, concept, cutoff=max_distance):
            if node != concept and self.graph.nodes[node].get("type") in ["concept", "technical_term"]:
                # Find the shortest path
                path = nx.shortest_path(self.graph, concept, node)
                
                # Get the relationship type and weight
                relationship_type = None
                relationship_weight = 0
                
                if len(path) == 2:  # Direct relationship
                    relationship_type = self.graph[path[0]][path[1]].get("type")
                    relationship_weight = self.graph[path[0]][path[1]].get("weight", 1)
                
                related_concepts.append({
                    "concept": node,
                    "distance": len(path) - 1,
                    "relationship_type": relationship_type,
                    "relationship_weight": relationship_weight
                })
        
        # Sort by distance and then by weight
        related_concepts.sort(key=lambda x: (x["distance"], -x.get("relationship_weight", 0)))
        
        return related_concepts
    
    def get_papers_for_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Get papers related to a given concept.
        
        Args:
            concept: The concept to find papers for.
            
        Returns:
            List of dictionaries containing paper information.
        """
        if not self.graph.has_node(concept):
            logger.warning(f"Concept {concept} not found in the graph")
            return []
        
        papers = []
        
        # Find papers that contain this concept
        for node in self.graph.predecessors(concept):
            if self.graph.nodes[node].get("type") == "paper":
                papers.append({
                    "id": node,
                    "title": self.graph.nodes[node].get("title"),
                    "authors": self.graph.nodes[node].get("authors")
                })
        
        return papers
    
    def get_concept_subgraph(self, concepts: List[str], max_distance: int = 2) -> Dict[str, Any]:
        """
        Get a subgraph containing the given concepts and their related concepts.
        
        Args:
            concepts: List of concepts to include in the subgraph.
            max_distance: Maximum distance in the graph to search for related concepts.
            
        Returns:
            Dictionary containing nodes and edges of the subgraph.
        """
        # Create a set to store nodes in the subgraph
        subgraph_nodes = set()
        
        # Add the input concepts and their related concepts
        for concept in concepts:
            if self.graph.has_node(concept):
                subgraph_nodes.add(concept)
                
                # Add related concepts
                for node, distance in nx.single_source_shortest_path_length(self.graph, concept, cutoff=max_distance).items():
                    if self.graph.nodes[node].get("type") in ["concept", "technical_term"]:
                        subgraph_nodes.add(node)
        
        # Create the subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Convert to dictionary format
        subgraph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node, attrs in subgraph.nodes(data=True):
            node_data = {
                "id": node,
                **attrs
            }
            subgraph_data["nodes"].append(node_data)
        
        # Add edges
        for source, target, attrs in subgraph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                **attrs
            }
            subgraph_data["edges"].append(edge_data)
        
        return subgraph_data
    
    def get_top_concepts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the top concepts in the graph based on centrality.
        
        Args:
            limit: Maximum number of concepts to return.
            
        Returns:
            List of dictionaries containing concept information.
        """
        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)
        
        # Filter for concept nodes
        concept_centrality = []
        for node, score in centrality.items():
            if self.graph.nodes[node].get("type") in ["concept", "technical_term"]:
                concept_centrality.append({
                    "concept": node,
                    "centrality": score,
                    "type": self.graph.nodes[node].get("type")
                })
        
        # Sort by centrality score
        concept_centrality.sort(key=lambda x: x["centrality"], reverse=True)
        
        # Return top concepts
        return concept_centrality[:limit]


if __name__ == "__main__":
    # Example usage
    import glob
    import yaml
    
    # Load processed papers
    processed_dir = Path("data/processed")
    paper_files = glob.glob(str(processed_dir / "*.yaml"))
    
    papers_data = []
    for paper_file in paper_files:
        with open(paper_file, 'r', encoding='utf-8') as f:
            paper_data = yaml.safe_load(f)
            papers_data.append(paper_data)
    
    # Create knowledge graph
    kg = KnowledgeGraph()
    kg.build_graph_from_papers(papers_data)
    
    # Get top concepts
    top_concepts = kg.get_top_concepts(limit=10)
    print("Top concepts:")
    for concept in top_concepts:
        print(f"- {concept['concept']} (Centrality: {concept['centrality']:.4f})")
    
    # Get related concepts for a sample concept
    if top_concepts:
        sample_concept = top_concepts[0]["concept"]
        related_concepts = kg.get_related_concepts(sample_concept)
        print(f"\nConcepts related to '{sample_concept}':")
        for concept in related_concepts[:5]:
            print(f"- {concept['concept']} (Distance: {concept['distance']}, Relationship: {concept['relationship_type']})")
        
        # Get papers for the sample concept
        papers = kg.get_papers_for_concept(sample_concept)
        print(f"\nPapers related to '{sample_concept}':")
        for paper in papers[:5]:
            print(f"- {paper['title']} (ID: {paper['id']})")
