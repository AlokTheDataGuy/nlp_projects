import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import faiss
import asyncio
from sentence_transformers import SentenceTransformer
from app.services.query_processor import QueryProcessor

class RetrievalService:
    """
    Service for retrieving relevant answers from the MedQuAD dataset
    """

    def __init__(self):
        # Define paths - adjust to look one level up from backend
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend directory
        project_dir = os.path.dirname(base_dir)  # project directory (one level up from backend)
        self.data_dir = os.path.join(project_dir, "data")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.embeddings_dir = os.path.join(self.data_dir, "embeddings")
        self.faiss_dir = os.path.join(self.data_dir, "faiss")

        print(f"Looking for data in: {self.processed_dir}")

        # Load the complete dataset
        self.df = pd.read_csv(os.path.join(self.processed_dir, "medquad_complete.csv"))

        # Initialize the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for quick inference

        # Initialize or load FAISS index
        self.index = None
        self.load_or_create_index()

        # Initialize query processor
        self.query_processor = QueryProcessor()

    def load_or_create_index(self):
        """
        Load existing FAISS index or create a new one if it doesn't exist
        """
        index_path = os.path.join(self.faiss_dir, "medquad.index")
        embeddings_path = os.path.join(self.embeddings_dir, "question_embeddings.npy")

        if os.path.exists(index_path) and os.path.exists(embeddings_path):
            # Load existing index and embeddings
            self.index = faiss.read_index(index_path)
            self.question_embeddings = np.load(embeddings_path)
            print("Loaded existing FAISS index and embeddings")
        else:
            # Create new index and embeddings
            print("Creating new FAISS index and embeddings...")
            self.create_index()

    def create_index(self):
        """
        Create FAISS index from the MedQuAD dataset
        """
        # Ensure directories exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.faiss_dir, exist_ok=True)

        # Get all questions from the dataset
        questions = self.df['question'].tolist()

        # Generate embeddings for all questions
        self.question_embeddings = self.model.encode(questions, show_progress_bar=True)

        # Create FAISS index
        dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.question_embeddings.astype('float32'))

        # Save embeddings and index
        np.save(os.path.join(self.embeddings_dir, "question_embeddings.npy"), self.question_embeddings)
        faiss.write_index(self.index, os.path.join(self.faiss_dir, "medquad.index"))

        print(f"Created FAISS index with {len(questions)} questions")

    def retrieve(self, query: str, max_results: int = 3, entities: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant answers for a given query
        """
        # Process the query
        processed_query = self.query_processor.process_query(query)

        # Generate embedding for the query
        query_embedding = self.model.encode([processed_query])[0].reshape(1, -1).astype('float32')

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, max_results)

        # Get the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.df):  # Ensure index is valid
                row = self.df.iloc[idx]

                # Calculate confidence score (1 - normalized distance)
                confidence = 1.0 - min(distances[0][i] / 100.0, 0.99)

                # Get related entities
                related_entities = []
                if entities:
                    for entity in entities:
                        if entity["text"].lower() in row["question"].lower() or entity["text"].lower() in row["answer"].lower():
                            related_entities.append(entity)

                results.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "source": f"{row['source']} - {row['url']}",
                    "score": float(confidence),
                    "related_entities": related_entities
                })

        return results

    def retrieve_by_type(self, query: str, question_type: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve answers based on question type
        """
        # Filter dataset by question type
        filtered_df = self.df[self.df['question_type'] == question_type]

        if filtered_df.empty:
            # Fallback to general retrieval if no questions of this type
            return self.retrieve(query, max_results)

        # Process the query
        processed_query = self.query_processor.process_query(query)

        # Generate embedding for the query
        query_embedding = self.model.encode([processed_query])[0]

        # Generate embeddings for filtered questions (this could be optimized in production)
        questions = filtered_df['question'].tolist()
        question_embeddings = self.model.encode(questions, show_progress_bar=True)

        # Calculate distances
        distances = np.linalg.norm(question_embeddings - query_embedding, axis=1)

        # Get top results
        top_indices = np.argsort(distances)[:max_results]

        # Format results
        results = []
        for idx in top_indices:
            row = filtered_df.iloc[idx]
            confidence = 1.0 - min(distances[idx] / 100.0, 0.99)

            results.append({
                "question": row["question"],
                "answer": row["answer"],
                "source": f"{row['source']} - {row['url']}",
                "score": float(confidence)
            })

        return results

    async def hybrid_retrieve(self, query: str, max_results: int = 3, entities: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using both FAISS and Meditron LLM
        """
        # First try to get results from FAISS
        faiss_results = self.retrieve(query, max_results, entities)

        # If we have good results (high confidence), return them
        if faiss_results and faiss_results[0]["score"] > 0.7:
            return faiss_results

        # Otherwise, try to get a response from Meditron
        try:
            from app.services.llm_service import MeditronService
            llm_service = MeditronService()

            # Get context from FAISS results if available
            context = ""
            if faiss_results:
                context = "\n\n".join([f"Q: {result['question']}\nA: {result['answer']}"
                                    for result in faiss_results[:2]])

            # Generate response from Meditron
            llm_response = await llm_service.generate_response(query, context, entities)

            # Create a result object
            result = {
                "question": query,
                "answer": llm_response,
                "source": "Meditron LLM",
                "score": 0.6,  # Default confidence for LLM responses
                "related_entities": entities if entities else []
            }

            # If we have FAISS results, combine them with the LLM response
            if faiss_results:
                return [result] + faiss_results
            else:
                return [result]

        except Exception as e:
            print(f"Error using Meditron LLM: {str(e)}")
            # Fallback to FAISS results if LLM fails
            return faiss_results if faiss_results else []
