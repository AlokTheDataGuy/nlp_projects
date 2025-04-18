from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
import json
from pathlib import Path

# Import custom modules
from ner import MedicalNER
from embedding import TextEmbedder
from retrieval import FAISSRetriever
from utils import load_dataset

# Set this to True to enable more detailed logging
DEBUG = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global variables
ner_model = None
embedder = None
retriever = None
dataset = None

def initialize_models():
    """Initialize all models and load data"""
    global ner_model, embedder, retriever, dataset

    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset()
        logger.info(f"Dataset loaded with {len(dataset)} QA pairs")

        # Initialize NER model
        logger.info("Initializing NER model...")
        ner_model = MedicalNER()

        # Initialize embedder
        logger.info("Initializing text embedder...")
        embedder = TextEmbedder()

        # Initialize retriever
        logger.info("Initializing FAISS retriever...")
        retriever = FAISSRetriever()

        # Check if FAISS index exists
        index_path = 'data/faiss/question_index.faiss'
        mapping_path = 'data/faiss/id_mapping.json'

        if os.path.exists(index_path) and os.path.exists(mapping_path):
            logger.info("Loading existing FAISS index...")
            retriever.load_index(index_path, mapping_path, dataset)
        else:
            logger.info("Building new FAISS index...")
            # Check if embeddings exist
            embeddings_path = 'data/embeddings/question_embeddings.npy'
            if os.path.exists(embeddings_path):
                logger.info("Loading existing embeddings...")
                embeddings = np.load(embeddings_path)
            else:
                logger.info("Building new embeddings...")
                embeddings = embedder.build_question_embeddings(dataset)

            # Build index
            retriever.build_index(embeddings, dataset)

        logger.info("All models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/query', methods=['POST'])
def query():
    """Process a query and return the answer"""
    try:
        data = request.json
        query_text = data.get('query', '')

        if not query_text:
            return jsonify({'error': 'No query provided'}), 400

        # Extract entities using NER
        logger.info(f"Processing query: {query_text}")
        ner_results = ner_model.extract_entities(query_text)
        logger.info(f"Extracted entities: {ner_results['entities']}")
        if DEBUG:
            logger.info(f"Entity labels: {ner_results['entity_labels']}")
            logger.info(f"Abbreviations: {ner_results['abbreviations']}")

        # Embed the query
        query_embedding = embedder.embed_text(query_text)

        # Retrieve answers
        answers = retriever.retrieve_answers(query_embedding, ner_results)

        # Format response
        response = {
            'query': query_text,
            'entities': ner_results['entities'],
            'entity_labels': ner_results['entity_labels'],
            'abbreviations': ner_results['abbreviations'],
            'answers': answers
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Initialize models
    if initialize_models():
        # Run the app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize models. Exiting.")
