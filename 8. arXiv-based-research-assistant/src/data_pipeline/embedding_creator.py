"""
Embedding Creator Module

This module generates embeddings for paper content using Sentence-Transformers.
"""

import os
import logging
import yaml
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingCreator:
    """
    Class for generating embeddings for paper content.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the EmbeddingCreator.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.model_id = self.config["embedding"]["model_id"]
        self.dimension = self.config["embedding"]["dimension"]
        self.batch_size = self.config["embedding"]["batch_size"]
        self.normalize = self.config["embedding"]["normalize"]
        self.device = self.config["embedding"]["device"]
        
        # Create directories if they don't exist
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the embedding model
        self.model = self._load_model()
    
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
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the Sentence-Transformers model.
        
        Returns:
            Loaded model.
        """
        try:
            # Check if CUDA is available
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Load the model
            model = SentenceTransformer(self.model_id, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_id} on {self.device}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            NumPy array of embeddings.
        """
        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=self.normalize
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def create_paper_embeddings(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create embeddings for a paper.
        
        Args:
            paper_data: Dictionary containing paper data.
            
        Returns:
            Dictionary with paper data and embeddings.
        """
        paper_id = paper_data["id"]
        
        # Create embeddings for different parts of the paper
        embeddings = {}
        
        # Abstract embedding
        abstract = paper_data["abstract"]
        embeddings["abstract"] = self.generate_embeddings([abstract])[0].tolist()
        
        # Title embedding
        title = paper_data["title"]
        embeddings["title"] = self.generate_embeddings([title])[0].tolist()
        
        # Section embeddings
        if "sections" in paper_data and paper_data["sections"]:
            section_embeddings = {}
            for section_name, section_text in paper_data["sections"].items():
                section_embeddings[section_name] = self.generate_embeddings([section_text])[0].tolist()
            embeddings["sections"] = section_embeddings
        
        # Chunk embeddings
        if "chunked_sections" in paper_data and paper_data["chunked_sections"]:
            chunk_embeddings = {}
            for section_name, chunks in paper_data["chunked_sections"].items():
                if chunks:  # Check if there are chunks in this section
                    chunk_embeddings[section_name] = self.generate_embeddings(chunks).tolist()
            embeddings["chunks"] = chunk_embeddings
        
        # Add embeddings to paper data
        paper_data["embeddings"] = embeddings
        
        # Save embeddings separately for faster loading
        embeddings_path = self.embeddings_dir / f"{paper_id}_embeddings.yaml"
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            yaml.dump(embeddings, f, default_flow_style=False, allow_unicode=True)
        
        return paper_data
    
    def create_papers_embeddings(self, papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create embeddings for multiple papers.
        
        Args:
            papers_data: List of dictionaries containing paper data.
            
        Returns:
            List of dictionaries with paper data and embeddings.
        """
        processed_papers = []
        
        for paper_data in tqdm(papers_data, desc="Creating embeddings"):
            try:
                processed_paper = self.create_paper_embeddings(paper_data)
                processed_papers.append(processed_paper)
            except Exception as e:
                logger.error(f"Error creating embeddings for paper {paper_data['id']}: {str(e)}")
        
        return processed_papers
    
    def build_faiss_index(self, papers_data: List[Dict[str, Any]], index_type: str = "chunks") -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Build a FAISS index from paper embeddings.
        
        Args:
            papers_data: List of dictionaries containing paper data with embeddings.
            index_type: Type of embeddings to index ("chunks", "abstract", "title", "sections").
            
        Returns:
            Tuple of (FAISS index, list of metadata for indexed items).
        """
        # Collect embeddings and metadata
        all_embeddings = []
        all_metadata = []
        
        for paper in papers_data:
            paper_id = paper["id"]
            
            if index_type == "chunks" and "embeddings" in paper and "chunks" in paper["embeddings"]:
                # Add chunk embeddings
                for section_name, chunks_embeddings in paper["embeddings"]["chunks"].items():
                    for i, chunk_embedding in enumerate(chunks_embeddings):
                        all_embeddings.append(chunk_embedding)
                        all_metadata.append({
                            "paper_id": paper_id,
                            "title": paper["title"],
                            "section": section_name,
                            "chunk_index": i,
                            "text": paper["chunked_sections"][section_name][i]
                        })
            
            elif index_type == "abstract" and "embeddings" in paper and "abstract" in paper["embeddings"]:
                # Add abstract embedding
                all_embeddings.append(paper["embeddings"]["abstract"])
                all_metadata.append({
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "type": "abstract",
                    "text": paper["abstract"]
                })
            
            elif index_type == "title" and "embeddings" in paper and "title" in paper["embeddings"]:
                # Add title embedding
                all_embeddings.append(paper["embeddings"]["title"])
                all_metadata.append({
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "type": "title",
                    "text": paper["title"]
                })
            
            elif index_type == "sections" and "embeddings" in paper and "sections" in paper["embeddings"]:
                # Add section embeddings
                for section_name, section_embedding in paper["embeddings"]["sections"].items():
                    all_embeddings.append(section_embedding)
                    all_metadata.append({
                        "paper_id": paper_id,
                        "title": paper["title"],
                        "section": section_name,
                        "text": paper["sections"][section_name]
                    })
        
        if not all_embeddings:
            logger.error(f"No embeddings found for index type: {index_type}")
            return None, []
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create and train the index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        
        # Add vectors to the index
        index.add(embeddings_array)
        
        # Save the index
        index_path = self.embeddings_dir / f"faiss_index_{index_type}"
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = self.embeddings_dir / f"faiss_index_{index_type}_metadata.yaml"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Built FAISS index for {index_type} with {len(all_embeddings)} vectors")
        
        return index, all_metadata


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
    
    embedding_creator = EmbeddingCreator()
    processed_papers = embedding_creator.create_papers_embeddings(papers_data)
    
    # Build FAISS indices
    for index_type in ["chunks", "abstract", "title", "sections"]:
        embedding_creator.build_faiss_index(processed_papers, index_type=index_type)
    
    print(f"Created embeddings for {len(processed_papers)} papers")
