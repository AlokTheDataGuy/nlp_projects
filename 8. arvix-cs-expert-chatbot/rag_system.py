import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import uuid
from typing import List, Dict, Any
import yaml
import os
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.rag_config = self.config['rag']
        self.embedding_model = SentenceTransformer(self.config['nlp']['embedding_model'])
        
        # Initialize ChromaDB for vector storage
        self.setup_vector_db()
        
    def setup_vector_db(self):
        """Setup ChromaDB for efficient vector storage and retrieval"""
        try:
            # Initialize ChromaDB client with updated configuration
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

            # Create or get collection
            collection_name = "arxiv_papers"
            try:
                self.collection = self.chroma_client.get_collection(name=collection_name)
                print(f"Loaded existing ChromaDB collection: {collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "ArXiv CS papers for RAG"}
                )
                print(f"Created new ChromaDB collection: {collection_name}")

        except Exception as e:
            print(f"Error setting up ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def add_papers_to_vector_db(self, papers, embeddings):
        """Add papers and their embeddings to vector database"""
        if not self.collection:
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = [paper['id'] for paper in papers]
            documents = [f"{paper['title']} {paper['abstract']}" for paper in papers]
            metadatas = [{
                'title': paper['title'],
                'authors': ', '.join(paper['authors']),
                'categories': ', '.join(paper['categories']),
                'published': paper['published'],
                'primary_category': paper['primary_category']
            } for paper in papers]
            
            # Convert embeddings to list format for ChromaDB
            embeddings_list = embeddings.tolist()
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(papers), batch_size):
                end_idx = min(i + batch_size, len(papers))
                
                self.collection.add(
                    embeddings=embeddings_list[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                
                print(f"Added batch {i//batch_size + 1}/{(len(papers) + batch_size - 1)//batch_size}")
            
            print(f"Successfully added {len(papers)} papers to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding papers to vector DB: {e}")
            return False
    
    def retrieve_relevant_papers(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant papers using vector similarity search"""
        if top_k is None:
            top_k = self.rag_config['top_k_papers']

        if not self.collection:
            print("Error: ChromaDB collection not available")
            return []

        try:
            # Check collection count
            collection_count = self.collection.count()
            print(f"Collection has {collection_count} documents")

            if collection_count == 0:
                print("Warning: Collection is empty")
                return []

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            print(f"Generated query embedding with shape: {query_embedding.shape}")

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k, collection_count),  # Don't request more than available
                include=['documents', 'metadatas', 'distances']
            )

            print(f"ChromaDB returned {len(results['ids'][0])} results")

            # Process results
            relevant_papers = []

            # Find the range of distances to normalize properly
            distances = [results['distances'][0][i] for i in range(len(results['ids'][0]))]
            min_distance = min(distances)
            max_distance = max(distances)
            distance_range = max_distance - min_distance

            print(f"Distance range: {min_distance:.4f} to {max_distance:.4f}, range: {distance_range:.4f}")

            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]

                # Normalize distance to similarity score (0-1 range)
                # Lower distance = higher similarity
                if distance_range > 0.001:  # Avoid division by very small numbers
                    # Normalize to 0-1 where 0 = max_distance, 1 = min_distance
                    similarity = 1.0 - ((distance - min_distance) / distance_range)
                else:
                    # All distances are very similar, assign based on rank
                    similarity = 1.0 - (i * 0.05)  # Decreasing similarity by rank

                # Ensure similarity is in valid range
                similarity = max(0.0, min(1.0, similarity))

                print(f"Paper {i}: distance={distance:.4f}, similarity={similarity:.4f}, threshold={self.config['data']['min_similarity']}")

                paper_info = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,
                    'title': results['metadatas'][0][i]['title'],
                    'authors': results['metadatas'][0][i]['authors'].split(', '),
                    'categories': results['metadatas'][0][i]['categories'].split(', '),
                    'published': results['metadatas'][0][i]['published'],
                    'primary_category': results['metadatas'][0][i]['primary_category']
                }

                # Include all papers for now (remove similarity filtering)
                relevant_papers.append(paper_info)
                print(f"Added paper: {paper_info['title'][:50]}... (similarity: {similarity:.4f})")

            print(f"Returning {len(relevant_papers)} papers")
            return relevant_papers

        except Exception as e:
            print(f"Error retrieving papers: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text for better RAG performance"""
        chunk_size = self.rag_config['chunk_size']
        chunk_overlap = self.rag_config['chunk_overlap']
        
        # Simple sentence-based chunking
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks
    
    def create_context_for_query(self, query: str, max_context_length: int = 2000) -> Dict:
        """Create context from relevant papers for query"""
        relevant_papers = self.retrieve_relevant_papers(query)
        
        context_parts = []
        total_length = 0
        used_papers = []
        
        for paper in relevant_papers:
            # Create paper summary for context
            paper_context = f"Title: {paper['title']}\n"
            paper_context += f"Abstract: {paper['document'].split('Abstract:')[1] if 'Abstract:' in paper['document'] else paper['document']}\n"
            paper_context += f"Categories: {', '.join(paper['categories'])}\n\n"
            
            # Check if adding this paper would exceed context length
            if total_length + len(paper_context.split()) > max_context_length:
                break
                
            context_parts.append(paper_context)
            used_papers.append(paper)
            total_length += len(paper_context.split())
        
        context = {
            'text': '\n'.join(context_parts),
            'papers': used_papers,
            'total_papers': len(used_papers)
        }
        
        return context
    
    def rerank_papers(self, query: str, papers: List[Dict]) -> List[Dict]:
        """Rerank papers using cross-encoder for better relevance"""
        if not self.rag_config.get('use_reranking', False):
            return papers
        
        try:
            from sentence_transformers import CrossEncoder
            
            # Load cross-encoder for reranking
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Create query-document pairs
            pairs = []
            for paper in papers:
                doc_text = f"{paper['title']} {paper['document']}"
                pairs.append([query, doc_text])
            
            # Get reranking scores
            scores = cross_encoder.predict(pairs)
            
            # Update similarity scores and resort
            for i, paper in enumerate(papers):
                paper['rerank_score'] = scores[i]
            
            # Sort by rerank score
            papers.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return papers
            
        except Exception as e:
            print(f"Error in reranking: {e}")
            return papers
    
    def get_paper_stats(self) -> Dict:
        """Get statistics about the paper database"""
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            
            return {
                'total_papers': count,
                'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension(),
                'model_name': self.config['nlp']['embedding_model']
            }
        except:
            return {}
