import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any

class VectorService:
    def __init__(self, index_path: str = "../data/vector_db"):
        self.index_path = index_path
        self.metadata_path = os.path.join(index_path, "metadata.pkl")
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = self._load_or_create_index()
        self.metadata = self._load_or_create_metadata()
        
        # Initialize Ollama for generating embeddings
        self.embed_model = "llama3.1:8b"  # Use same model for embeddings
    
    def _load_or_create_index(self):
        index_file = os.path.join(self.index_path, "faiss.index")
        if os.path.exists(index_file):
            return faiss.read_index(index_file)
        else:
            # Create a new index with 384 dimensions (adjust based on your embedding size)
            return faiss.IndexFlatL2(384)
    
    def _load_or_create_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                return pickle.load(f)
        else:
            return {"ids": [], "papers": {}}
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using Ollama"""
        # Using a simplified approach; in production, use a dedicated embedding service
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.embed_model,
            "prompt": text,
            "options": {"embedding": True}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate", 
                headers=headers, 
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return np.array(result["embedding"], dtype=np.float32)
                else:
                    # Return a zero vector if embedding fails
                    return np.zeros(384, dtype=np.float32)
    
    async def index_papers(self, papers: List[Dict[str, Any]]):
        """Index papers in the vector database"""
        for paper_data in papers:
            paper = paper_data["paper"]
            content = paper_data["content"]
            
            # Create a combined text representation
            text = f"{paper['title']} {paper['abstract']}"
            if 'introduction' in content:
                text += f" {content['introduction']}"
                
            # Get embedding
            embedding = await self._get_embedding(text)
            embedding = embedding.reshape(1, -1).astype('float32')
            
            # Add to index
            self.index.add(embedding)
            
            # Update metadata
            paper_id = paper["id"]
            index_id = len(self.metadata["ids"])
            self.metadata["ids"].append(paper_id)
            self.metadata["papers"][paper_id] = {
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["abstract"],
                "url": paper.get("pdf_url", ""),
                "index_id": index_id
            }
        
        # Save index and metadata
        self._save_index()
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for papers semantically similar to the query"""
        query_embedding = await self._get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search the index
        if self.index.ntotal == 0:
            return []  # No papers indexed yet
            
        D, I = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for i in range(len(I[0])):
            if I[0][i] < len(self.metadata["ids"]):
                paper_id = self.metadata["ids"][I[0][i]]
                paper_data = self.metadata["papers"][paper_id]
                results.append({
                    "id": paper_id,
                    "title": paper_data["title"],
                    "authors": paper_data["authors"],
                    "abstract": paper_data["abstract"],
                    "url": paper_data["url"],
                    "score": float(D[0][i])
                })
        
        return results
    
    def _save_index(self):
        """Save the index and metadata to disk"""
        index_file = os.path.join(self.index_path, "faiss.index")
        faiss.write_index(self.index, index_file)
        
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)