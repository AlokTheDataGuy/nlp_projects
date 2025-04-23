# Update the path in backend/app/utils/create_index.py
import os
import sys
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

def create_faiss_index():
    """
    Create FAISS index from the MedQuAD dataset
    """
    print("Starting FAISS index creation...")
    start_time = time.time()
    
    # Define paths - adjust to look one level up from backend
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend directory
    project_dir = os.path.dirname(base_dir)  # project directory (one level up from backend)
    data_dir = os.path.join(project_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    embeddings_dir = os.path.join(data_dir, "embeddings")
    faiss_dir = os.path.join(data_dir, "faiss")
    
    print(f"Looking for data in: {processed_dir}")
    
    # Ensure directories exist
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(faiss_dir, exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(os.path.join(processed_dir, "medquad_complete.csv"))
    
    # Get all questions from the dataset
    questions = df['question'].tolist()
    print(f"Found {len(questions)} questions in the dataset")
    
    # Initialize the embedding model
    print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for all questions
    print("Generating embeddings for all questions...")
    question_embeddings = model.encode(questions, show_progress_bar=True)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(question_embeddings.astype('float32'))
    
    # Save embeddings and index
    print("Saving embeddings and index...")
    np.save(os.path.join(embeddings_dir, "question_embeddings.npy"), question_embeddings)
    faiss.write_index(index, os.path.join(faiss_dir, "medquad.index"))
    
    end_time = time.time()
    print(f"FAISS index creation completed in {end_time - start_time:.2f} seconds")
    print(f"Created index with {len(questions)} questions")

if __name__ == "__main__":
    create_faiss_index()