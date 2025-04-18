import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalChatbot: 
    def __init__(self):
        self.model = None
        self.index = None
        self.dataset = None
        self.id_map = None

        # Initialize components
        self.load_dataset()
        self.load_model()
        self.build_or_load_index()

    def load_dataset(self):
        """Load the MedQuAD dataset"""
        try:
            logger.info("Loading dataset...")
            self.dataset = pd.read_csv('data/processed/medquad_complete.csv')
            # Clean the dataset
            self.dataset = self.dataset.dropna(subset=['question', 'answer'])
            logger.info(f"Dataset loaded with {len(self.dataset)} QA pairs")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info("Loading Sentence Transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def build_or_load_index(self):
        """Build or load the FAISS index"""
        index_path = 'data/faiss/question_index.faiss'
        mapping_path = 'data/faiss/id_mapping.json'

        if os.path.exists(index_path) and os.path.exists(mapping_path):
            logger.info("Loading existing FAISS index...")
            self.load_index(index_path, mapping_path)
        else:
            logger.info("Building new FAISS index...")
            self.build_index()

    def build_index(self):
        """Build a FAISS index from the dataset"""
        try:
            # Create directories if they don't exist
            os.makedirs('data/faiss', exist_ok=True)

            # Generate embeddings
            logger.info("Generating embeddings...")
            questions = self.dataset['question'].tolist()
            embeddings = self.model.encode(questions, show_progress_bar=True)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)

            # Create ID mapping
            self.id_map = {str(i): int(i) for i in range(len(self.dataset))}

            # Save index and mapping
            index_path = 'data/faiss/question_index.faiss'
            mapping_path = 'data/faiss/id_mapping.json'

            faiss.write_index(self.index, index_path)
            with open(mapping_path, 'w') as f:
                json.dump(self.id_map, f)

            logger.info("FAISS index built and saved successfully")
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise

    def load_index(self, index_path, mapping_path):
        """Load a FAISS index from disk"""
        try:
            self.index = faiss.read_index(index_path)

            with open(mapping_path, 'r') as f:
                self.id_map = json.load(f)

            logger.info("FAISS index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def search(self, query, top_k=20):
        """Search for similar questions"""
        try:
            # Encode the query
            query_embedding = self.model.encode(query)

            # Reshape and normalize
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            # Search for more candidates than needed to filter later
            scores, indices = self.index.search(query_embedding, top_k * 2)

            # Get results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # -1 indicates no result
                    try:
                        # Try to get the index from the id_map
                        if str(idx) in self.id_map:
                            df_idx = self.id_map[str(idx)]
                        elif idx in self.id_map:
                            df_idx = self.id_map[idx]
                        else:
                            # If not in id_map, use the index directly (if within bounds)
                            if 0 <= idx < len(self.dataset):
                                df_idx = idx
                            else:
                                logger.warning(f"Index {idx} out of bounds, skipping")
                                continue

                        # Get the row from the dataset
                        row = self.dataset.iloc[df_idx]

                        # Calculate text similarity score for better ranking
                        from difflib import SequenceMatcher
                        question_similarity = SequenceMatcher(None, query.lower(), row['question'].lower()).ratio()

                        # Combine embedding similarity with text similarity
                        combined_score = float(scores[0][i]) * 0.7 + question_similarity * 0.3

                        results.append({
                            'question': row['question'],
                            'answer': row['answer'],
                            'score': combined_score,
                            'source': row.get('source', 'MedQuAD'),
                            'text_similarity': question_similarity
                        })
                    except Exception as e:
                        logger.warning(f"Error processing result at index {idx}: {e}")
                        continue

            # Sort by combined score
            results.sort(key=lambda x: x['score'], reverse=True)

            # Return top_k results
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def generate_response(self, query):
        """Generate a response to the query"""
        try:
            # Handle greetings and simple queries
            query_lower = query.lower().strip()
            if query_lower in ['hi', 'hello', 'hey', 'hii', 'hiii', 'hiiii']:
                return {
                    "answer": "Hello! I'm a medical Q&A chatbot. How can I help you with your medical questions today?",
                    "sources": []
                }
            elif query_lower in ['how are you', 'how are you?', 'how are you doing', 'how are you doing?']:
                return {
                    "answer": "I'm functioning well, thank you for asking! I'm here to help answer your medical questions. What would you like to know?",
                    "sources": []
                }
            elif query_lower in ['thank you', 'thanks', 'thx', 'ty']:
                return {
                    "answer": "You're welcome! If you have any more medical questions, feel free to ask.",
                    "sources": []
                }

            # Search for relevant QA pairs
            search_results = self.search(query)

            if not search_results:
                return {
                    "answer": "I'm sorry, I couldn't find information related to your question in my medical knowledge base. Could you try rephrasing your question or asking about a different medical topic?",
                    "sources": []
                }

            # Filter results by minimum text similarity
            min_similarity = 0.5  # Minimum text similarity threshold
            filtered_results = [r for r in search_results if r['text_similarity'] >= min_similarity]

            # If no results meet the threshold, check if any contain key terms from the query
            if not filtered_results:
                # Extract key terms (words with 4+ characters)
                key_terms = [word.lower() for word in query.split() if len(word) >= 4]

                if key_terms:
                    for result in search_results:
                        question_lower = result['question'].lower()
                        # Check if any key term is in the question
                        if any(term in question_lower for term in key_terms):
                            filtered_results.append(result)

            # If still no results, use the original results but with a warning
            if not filtered_results:
                best_match = search_results[0]
                answer = best_match['answer']
                answer = "Note: I couldn't find an exact match for your question, but here's the closest information I have:\n\n" + answer
            else:
                # Use the filtered results
                best_match = filtered_results[0]
                answer = best_match['answer']

            # Format the response
            response = {
                "answer": answer,
                "sources": [
                    {
                        "question": result['question'],
                        "source": result['source'],
                        "score": result['score'],
                        "similarity": result.get('text_similarity', 0)
                    } for result in (filtered_results or search_results)[:3]  # Include top 3 sources
                ]
            }

            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try asking a different medical question.",
                "sources": []
            }

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Initialize chatbot
chatbot = None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('llm_index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message and return a response"""
    try:
        data = request.json
        query = data.get('message', '')

        if not query:
            return jsonify({'error': 'No message provided'}), 400

        # Get chatbot instance from app config
        chatbot = app.config.get('chatbot')
        if not chatbot:
            return jsonify({'error': 'Chatbot not initialized'}), 500

        # Generate response
        logger.info(f"Processing query: {query}")
        response = chatbot.generate_response(query)

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

# This file is imported by run_llm_chatbot.py
