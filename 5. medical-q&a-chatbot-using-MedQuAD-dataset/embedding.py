from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the text embedding module
        
        Args:
            model_name: Name of the Sentence Transformers model to use
        """
        logger.info(f"Loading Sentence Transformers model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Sentence Transformers model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Sentence Transformers model: {e}")
            raise
    
    def embed_text(self, text):
        """
        Generate embeddings for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(text, show_progress_bar=False)
    
    def embed_batch(self, texts, batch_size=32):
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def build_question_embeddings(self, df, output_path='data/embeddings/question_embeddings.npy'):
        """
        Build and save embeddings for all questions in the dataset
        
        Args:
            df: DataFrame containing the dataset
            output_path: Path to save the embeddings
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Building embeddings for {len(df)} questions")
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate embeddings
        embeddings = self.embed_batch(df['question'].tolist())
        
        # Save embeddings
        np.save(output_path, embeddings)
        logger.info(f"Embeddings saved to {output_path}")
        
        return embeddings
