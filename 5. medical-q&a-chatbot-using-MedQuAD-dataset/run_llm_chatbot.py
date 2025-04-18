import logging
from llm_chatbot import app, MedicalChatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the LLM-based Medical Chatbot"""
    try:
        # Initialize chatbot
        logger.info("Initializing chatbot...")
        app.config['chatbot'] = MedicalChatbot()
        logger.info("Chatbot initialized successfully")
        
        # Run the app
        logger.info("Starting the Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
        
        return 0
    except Exception as e:
        logger.error(f"Error running the application: {e}")
        return 1

if __name__ == "__main__":
    main()
