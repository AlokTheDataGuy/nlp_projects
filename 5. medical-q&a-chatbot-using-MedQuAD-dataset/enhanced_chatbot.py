import pandas as pd
import numpy as np
import os
import json
import logging
import requests
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for structured output
class Entity(BaseModel):
    name: str = Field(description="Name of the medical entity")
    type: Optional[str] = Field(description="Type of the entity (e.g., disease, symptom, treatment)")

class RephrasedQuery(BaseModel):
    rephrased_query: str = Field(description="The rephrased medical query")
    entities: List[Entity] = Field(description="List of medical entities identified in the query")
    search_terms: List[str] = Field(description="Key terms to search for")

class RankedResult(BaseModel):
    relevance_score: float = Field(description="Score from 0-1 indicating relevance to the query")
    explanation: str = Field(description="Explanation of why this result is relevant or not")

class EnhancedMedicalChatbot:
    def __init__(self, ollama_base_url="http://localhost:11434"):
        """Initialize the enhanced medical chatbot with LangChain and Ollama integration"""
        self.dataset = None
        self.embedding_model = None
        self.index = None
        self.id_map = None
        self.ollama_base_url = ollama_base_url

        # Initialize components
        self.load_dataset()
        self.load_embedding_model()
        self.build_or_load_index()
        self.setup_langchain()

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

    def load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info("Loading Sentence Transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def setup_langchain(self):
        """Set up LangChain with Ollama"""
        try:
            logger.info("Setting up LangChain with Ollama...")

            # Check if Ollama is running
            try:
                response = requests.get(f"{self.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    available_models = response.json().get("models", [])
                    logger.info(f"Available Ollama models: {[m.get('name') for m in available_models]}")
                    self.llm_model_name = "meditron:7b"
                else:
                    logger.warning("Could not get available models from Ollama")
                    self.llm_model_name = "meditron:7b"
            except Exception as e:
                logger.warning(f"Error checking Ollama: {e}")
                self.llm_model_name = "meditron:7b"

            logger.info(f"Using Ollama model: {self.llm_model_name}")

            # Initialize Ollama LLMs - one for understanding/search (LLaMA) and one for direct answers (Meditron)
            self.search_llm = Ollama(
                base_url=self.ollama_base_url,
                model="llama3.1:8b",
                system="You are an AI assistant helping to understand medical questions and find relevant information."
            )

            # Initialize Meditron for direct medical answers
            self.medical_llm = Ollama(
                base_url=self.ollama_base_url,
                model="meditron:7b",
                system="You are Meditron, a specialized medical AI assistant. Provide concise, accurate medical information."
            )

            # Set the main LLM to the search LLM for compatibility with existing code
            self.llm = self.search_llm

            # Set up query rephrasing chain
            self.setup_query_rephrasing_chain()

            # Set up reranking chain
            self.setup_reranking_chain()

            # Set up simplification chain
            self.setup_simplification_chain()

            logger.info("LangChain setup completed")
        except Exception as e:
            logger.error(f"Error setting up LangChain: {e}")
            logger.warning("Continuing without LangChain functionality")
            self.llm = None

    def setup_query_rephrasing_chain(self):
        """Set up the query rephrasing chain"""
        try:
            # Define the prompt template
            rephrasing_template = """You are a medical expert helping to rephrase and enhance user queries for a medical question answering system.

            Given the user's query, please:
            1. Rephrase it into a clear, medically precise question
            2. Identify key medical entities (diseases, symptoms, treatments, etc.)
            3. Extract key search terms that would help find relevant information

            User Query: {query}

            {format_instructions}
            """

            # Set up the parser
            parser = PydanticOutputParser(pydantic_object=RephrasedQuery)

            # Create the prompt
            self.rephrasing_prompt = PromptTemplate(
                template=rephrasing_template,
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # Create the chain
            self.rephrasing_chain = LLMChain(
                llm=self.llm,
                prompt=self.rephrasing_prompt,
                output_parser=parser,
                verbose=True
            )

            logger.info("Query rephrasing chain set up successfully")
        except Exception as e:
            logger.error(f"Error setting up query rephrasing chain: {e}")
            self.rephrasing_chain = None

    def setup_reranking_chain(self):
        """Set up the reranking chain"""
        try:
            # Define the prompt template
            reranking_template = """You are a medical expert helping to evaluate the relevance of potential answers to a medical question.

            User Query: {query}

            Potential Answer:
            Question: {candidate_question}
            Answer: {candidate_answer}

            Please evaluate how relevant this answer is to the user's query on a scale from 0 to 1, where:
            - 0 means completely irrelevant
            - 1 means perfectly relevant and directly answers the query

            Consider:
            - Does the candidate answer address the specific medical condition, treatment, or concept asked about?
            - Does it provide the type of information the user is seeking?
            - Is it accurate and comprehensive for the query?

            {format_instructions}
            """

            # Set up the parser
            parser = PydanticOutputParser(pydantic_object=RankedResult)

            # Create the prompt
            self.reranking_prompt = PromptTemplate(
                template=reranking_template,
                input_variables=["query", "candidate_question", "candidate_answer"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # Create the chain
            self.reranking_chain = LLMChain(
                llm=self.llm,
                prompt=self.reranking_prompt,
                output_parser=parser,
                verbose=True
            )

            logger.info("Reranking chain set up successfully")
        except Exception as e:
            logger.error(f"Error setting up reranking chain: {e}")
            self.reranking_chain = None

    def setup_simplification_chain(self):
        """Set up the answer simplification chain"""
        # We're removing the simplification feature as requested
        self.simplification_chain = None
        logger.info("Simplification feature disabled as requested")

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
            embeddings = self.embedding_model.encode(questions, show_progress_bar=True)

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

    def rephrase_query(self, query):
        """Rephrase the query using LangChain and Ollama"""
        if not self.llm or not self.rephrasing_chain:
            logger.warning("LLM or rephrasing chain not available, using original query")
            return {
                "rephrased_query": query,
                "entities": [],
                "search_terms": [term.lower() for term in query.split() if len(term) >= 4]
            }

        try:
            logger.info(f"Rephrasing query: {query}")
            # Use invoke instead of run to avoid deprecation warning
            result = self.rephrasing_chain.invoke({"query": query})
            logger.info(f"Rephrased query: {result.rephrased_query}")
            return result
        except Exception as e:
            logger.error(f"Error rephrasing query: {e}")
            return {
                "rephrased_query": query,
                "entities": [],
                "search_terms": [term.lower() for term in query.split() if len(term) >= 4]
            }

    def search(self, query, top_k=20):
        """Search for similar questions"""
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode(query)

            # Reshape and normalize
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            # Search for more candidates than needed to filter later
            scores, indices = self.index.search(query_embedding, top_k * 2)

            # Get results
            results = []
            dataset_length = len(self.dataset)

            # Log dataset size for debugging
            logger.info(f"Dataset size: {dataset_length}")

            for i, idx in enumerate(indices[0]):
                if idx == -1:  # -1 indicates no result
                    continue

                try:
                    # Determine the actual dataframe index to use
                    df_idx = None

                    # Try to get the index from the id_map
                    if str(idx) in self.id_map:
                        df_idx = self.id_map[str(idx)]
                    elif idx in self.id_map:
                        df_idx = self.id_map[idx]
                    else:
                        # If not in id_map, use the index directly (if within bounds)
                        if 0 <= idx < dataset_length:
                            df_idx = idx
                        else:
                            logger.warning(f"Index {idx} out of bounds (dataset size: {dataset_length}), skipping")
                            continue

                    # Verify df_idx is valid
                    if df_idx is None or df_idx < 0 or df_idx >= dataset_length:
                        logger.warning(f"Mapped index {df_idx} out of bounds (dataset size: {dataset_length}), skipping")
                        continue

                    # Get the row from the dataset
                    row = self.dataset.iloc[df_idx]

                    # Calculate text similarity score for better ranking
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

    def rerank_results(self, query, results, top_k=3):
        """Rerank results using LangChain and Ollama"""
        if not self.llm or not self.reranking_chain or not results:
            logger.warning("LLM or reranking chain not available, or no results to rerank")
            return results[:top_k]

        try:
            logger.info(f"Reranking {len(results)} results")

            # Make sure we have results to rerank
            if not results or len(results) == 0:
                logger.warning("No results to rerank")
                return []

            # Only rerank the top candidates to save time
            candidates_to_rerank = min(len(results), 5)
            reranked_results = []

            for i in range(candidates_to_rerank):
                # Safety check to make sure we don't go out of bounds
                if i >= len(results):
                    logger.warning(f"Index {i} out of bounds for results list of length {len(results)}")
                    break

                result = results[i]
                try:
                    reranking_result = self.reranking_chain.invoke({
                        "query": query,
                        "candidate_question": result['question'],
                        "candidate_answer": result['answer']
                    })

                    # Update the score with the LLM-assigned relevance
                    result['llm_relevance'] = reranking_result.relevance_score
                    result['explanation'] = reranking_result.explanation

                    # Combine the original score with the LLM relevance
                    result['final_score'] = result['score'] * 0.3 + reranking_result.relevance_score * 0.7

                    reranked_results.append(result)
                except Exception as e:
                    logger.warning(f"Error reranking result {i}: {e}")
                    # Keep the original result if reranking fails
                    result['llm_relevance'] = 0
                    result['explanation'] = "Reranking failed"
                    result['final_score'] = result['score']
                    reranked_results.append(result)

            # Sort by final score
            reranked_results.sort(key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)

            # If we didn't rerank all results, add the remaining ones
            if candidates_to_rerank < len(results):
                for i in range(candidates_to_rerank, len(results)):
                    if len(reranked_results) >= top_k:
                        break

                    # Safety check to make sure we don't go out of bounds
                    if i >= len(results):
                        logger.warning(f"Index {i} out of bounds for results list of length {len(results)}")
                        break

                    result = results[i]
                    result['llm_relevance'] = 0
                    result['explanation'] = "Not reranked"
                    result['final_score'] = result['score']
                    reranked_results.append(result)

            logger.info(f"Reranking complete, returning top {top_k} results")
            return reranked_results[:top_k]
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results[:top_k]

    def simplify_answer(self, answer):
        """Simplify the answer using LangChain and Ollama"""
        if not self.llm or not self.simplification_chain:
            logger.warning("LLM or simplification chain not available, returning original answer")
            return answer

        try:
            logger.info("Simplifying answer")
            simplified = self.simplification_chain.invoke({"answer": answer})
            logger.info("Answer simplified successfully")
            return simplified
        except Exception as e:
            logger.error(f"Error simplifying answer: {e}")
            return answer

    def is_common_medical_question(self, query):
        """Check if this is a common medical question that LLM should handle directly"""
        query_lower = query.lower()

        # List of common medical topics that LLM should handle directly
        common_topics = [
            "cancer", "diabetes", "heart disease", "stroke", "alzheimer", "parkinson",
            "asthma", "copd", "arthritis", "depression", "anxiety", "hypertension",
            "high blood pressure", "cholesterol", "obesity", "pneumonia", "bronchitis",
            "flu", "influenza", "covid", "coronavirus", "hiv", "aids", "hepatitis",
            "cirrhosis", "kidney disease", "thyroid", "lupus", "multiple sclerosis",
            "epilepsy", "migraine", "headache", "allergy", "eczema", "psoriasis",
            "ulcer", "ibs", "crohn", "colitis", "gerd", "acid reflux", "osteoporosis",
            "anemia", "leukemia", "lymphoma", "melanoma", "carcinoma", "tumor",
            "symptoms", "treatment", "diagnosis", "prevention", "causes", "risk factors"
        ]

        # Check if query contains any common medical topics
        if any(topic in query_lower for topic in common_topics):
            return True

        # Check for question patterns about symptoms, causes, treatments
        symptom_patterns = ["symptom", "sign", "how do you know", "how to tell"]
        cause_patterns = ["cause", "why do people get", "risk factor", "how do you get"]
        treatment_patterns = ["treatment", "cure", "therapy", "medication", "drug", "how to treat"]

        if any(pattern in query_lower for pattern in symptom_patterns + cause_patterns + treatment_patterns):
            return True

        return False

    def generate_llm_answer(self, query):
        """Generate a direct answer using Meditron for medical questions"""
        if not self.medical_llm:
            return None

        try:
            # Create a prompt for Meditron that requests concise answers
            prompt = f"""Please answer the following medical question with factual, evidence-based information.
            Keep your answer very concise (2-3 sentences maximum).
            Focus only on the most important information.
            Do NOT repeat the question in your answer.
            Do NOT include any thank you messages or pleasantries.

            Question: {query}

            Answer:"""

            # Get response from Meditron
            response = self.medical_llm.invoke(prompt)

            if response and len(response) > 10:  # Ensure we got a meaningful response
                # Clean the response to remove unwanted patterns
                response = self.clean_response(response, query)

                # Limit response length if it's too long
                if len(response) > 300:
                    # Try to find a good sentence break to truncate at
                    truncate_points = [response.rfind('. ', 0, 300), response.rfind('! ', 0, 300), response.rfind('? ', 0, 300)]
                    truncate_point = max(truncate_points)
                    if truncate_point > 0:
                        response = response[:truncate_point+1]
                    else:
                        response = response[:300] + '...'
                return response
            return None
        except Exception as e:
            logger.warning(f"Error generating Meditron answer: {e}")
            return None

    def generate_related_questions(self, query):
        """Generate related questions using search LLM"""
        if not self.search_llm:
            return []

        try:
            # Create a prompt for generating related questions
            prompt = f"""Based on the medical question below, generate 2 short, related follow-up questions.
            Make the questions specific, relevant, and concise (under 10 words each).
            Each question should be on a new line and start with a number (1, 2).

            Original question: {query}

            Related questions:"""

            # Get response from search LLM
            response = self.search_llm.invoke(prompt)

            # Parse the response to extract questions
            questions = []
            if response:
                # Split by newlines and look for numbered items
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Check if line starts with a number followed by a dot or parenthesis
                    if (line and (line[0].isdigit() or
                                 (len(line) > 1 and line[0] in '123' and line[1] in '.)'))):
                        # Clean up the question
                        question = line
                        # Remove leading numbers, dots, parentheses, etc.
                        for char in '0123456789.):- ':
                            if question.startswith(char):
                                question = question[1:].strip()
                            else:
                                break
                        if question and len(question) > 10:
                            questions.append(question)

            # If we couldn't parse questions properly, return empty list
            return questions[:3] if questions else []
        except Exception as e:
            logger.warning(f"Error generating related questions: {e}")
            return []

    def clean_response(self, response_text, query):
        """Clean up response text to remove unwanted patterns"""
        if not response_text:
            return response_text

        # Remove the query if it appears at the beginning of the response
        query_lower = query.lower().strip()
        response_lower = response_text.lower().strip()

        if response_lower.startswith(query_lower):
            response_text = response_text[len(query):].strip()

        # Remove common pleasantries
        pleasantries = [
            "thank you for your question", "thanks for asking", "thank you for asking",
            "i hope this helps", "i hope that helps", "hope this helps", "hope that helps",
            "let me know if you have any other questions", "feel free to ask"
        ]

        for phrase in pleasantries:
            if phrase in response_lower:
                # Find the phrase and remove the sentence containing it
                idx = response_lower.find(phrase)
                if idx > 0:
                    # Find the start of the sentence
                    start = max(0, response_lower.rfind('.', 0, idx) + 1)
                    # Find the end of the sentence
                    end = response_lower.find('.', idx)
                    if end == -1:
                        end = len(response_lower)
                    else:
                        end += 1  # Include the period

                    # Remove the sentence
                    response_text = response_text[:start] + response_text[end:]

        return response_text.strip()

    def generate_response(self, query):
        """Generate a response to the query"""
        try:
            # Handle greetings and simple queries
            query_lower = query.lower().strip()
            if query_lower in ['hi', 'hello', 'hey', 'hii', 'hiii', 'hiiii']:
                return {
                    "answer": "Hello! I'm a medical Q&A chatbot. How can I help you with your medical questions today?",
                    "sources": [],
                    "related_questions": []
                }
            elif query_lower in ['how are you', 'how are you?', 'how are you doing', 'how are you doing?']:
                return {
                    "answer": "I'm functioning well, thank you for asking! I'm here to help answer your medical questions. What would you like to know?",
                    "sources": [],
                    "related_questions": []
                }
            elif query_lower in ['thank you', 'thanks', 'thx', 'ty']:
                return {
                    "answer": "You're welcome! If you have any more medical questions, feel free to ask.",
                    "sources": [],
                    "related_questions": []
                }

            # Step 1: Use LLaMA to understand the query and rephrase it
            rephrased = self.rephrase_query(query)

            # Handle both dictionary and Pydantic object return types
            if hasattr(rephrased, 'rephrased_query'):
                search_query = rephrased.rephrased_query
            elif isinstance(rephrased, dict) and "rephrased_query" in rephrased:
                search_query = rephrased["rephrased_query"]
            else:
                search_query = query

            logger.info(f"Original query: {query}")
            logger.info(f"Rephrased query: {search_query}")

            # Step 2: Search for relevant QA pairs in the dataset
            search_results = self.search(search_query)

            # Step 3: Check if we have good search results (score > 0.7)
            if search_results and len(search_results) > 0 and search_results[0]['score'] > 0.7:
                # We found a good match in the dataset
                best_match = search_results[0]
                answer = best_match['answer']

                # Format the answer to be more concise using LLaMA
                format_prompt = f"""Summarize the following medical information in 2-3 concise sentences.
                Do NOT include any thank you messages or pleasantries.
                Do NOT repeat the question in your summary.
                Focus only on the key medical facts.

                {answer}

                Concise summary:"""

                try:
                    formatted_answer = self.search_llm.invoke(format_prompt)
                    if formatted_answer and len(formatted_answer) > 20:
                        # Clean the formatted answer
                        formatted_answer = self.clean_response(formatted_answer, query)
                        answer = formatted_answer
                except Exception as e:
                    logger.warning(f"Error formatting answer: {e}")

                # Generate related questions
                related_questions = self.generate_related_questions(query)

                # Return the formatted dataset answer
                return {
                    "answer": answer,
                    "sources": [
                        {
                            "name": result['source'],
                            "url": f"https://www.niddk.nih.gov/health-information/health-topics/search?query={result['question'].replace(' ', '+')}"
                        } for result in search_results[:2]
                    ],
                    "related_questions": related_questions
                }

            # Step 4: If no good match in dataset, use Meditron for direct answer
            meditron_answer = self.generate_llm_answer(query)

            if meditron_answer:
                # Generate related questions
                related_questions = self.generate_related_questions(query)

                # Return Meditron's answer
                return {
                    "answer": meditron_answer,
                    "sources": [
                        {
                            "name": "Meditron Medical Knowledge",
                            "url": "https://www.nih.gov/health-information"
                        }
                    ],
                    "related_questions": related_questions
                }

            # Step 5: Fallback if everything else fails
            return {
                "answer": "I'm sorry, I don't have a clear answer for that medical question. Please try rephrasing or asking about a different topic.",
                "sources": [],
                "related_questions": [
                    "What are common symptoms of diabetes?",
                    "How can I maintain a healthy heart?"
                ]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try asking a different medical question.",
                "sources": [],
                "related_questions": [
                    "What are common symptoms of diabetes?",
                    "How can I maintain a healthy heart?",
                    "What are the warning signs of a stroke?"
                ]
            }

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('enhanced_index.html')

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

# Simplify endpoint removed as requested

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

# This file is imported by run_enhanced_chatbot.py
