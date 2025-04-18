import logging
import argparse
from enhanced_chatbot import app, EnhancedMedicalChatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the Enhanced Medical Chatbot"""
    parser = argparse.ArgumentParser(description='Run the Enhanced Medical Chatbot')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='URL for the Ollama API (default: http://localhost:11434)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the Flask app on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the Flask app on (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Run Flask in debug mode')
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        logger.info(f"Initializing Enhanced Medical Chatbot with Ollama URL: {args.ollama_url}")
        chatbot = EnhancedMedicalChatbot(ollama_base_url=args.ollama_url)
        app.config['chatbot'] = chatbot
        logger.info("Chatbot initialized successfully")
        
        # Run the app
        logger.info(f"Starting the Flask application on {args.host}:{args.port}")
        app.run(debug=args.debug, host=args.host, port=args.port)
        
        return 0
    except Exception as e:
        logger.error(f"Error running the application: {e}")
        return 1

if __name__ == "__main__":
    main()
