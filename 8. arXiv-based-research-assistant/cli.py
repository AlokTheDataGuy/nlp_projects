"""
Interactive CLI for the arXiv Research Assistant.
"""
import os
import argparse
import uuid
from typing import List, Dict, Any, Optional

from rag.system import RAGSystem
from utils.logger import setup_logger

logger = setup_logger("cli", "cli.log")

class ArxivCLI:
    """Interactive CLI for the arXiv Research Assistant."""
    
    def __init__(self):
        """Initialize the CLI."""
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        self.rag_system = RAGSystem()
        
        # Initialize session
        self.session_id = str(uuid.uuid4())
        logger.info(f"Session ID: {self.session_id}")
        
        # Welcome message
        self.welcome_message = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║               arXiv Research Assistant CLI                   ║
║                                                              ║
║  Ask questions about computer science research papers.       ║
║  Type 'help' for commands or 'exit' to quit.                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        # Help message
        self.help_message = """
Available commands:
  help                 - Show this help message
  exit                 - Exit the CLI
  clear                - Clear the screen
  new                  - Start a new session
  sessions             - List all sessions
  load <session_id>    - Load a session
  delete <session_id>  - Delete a session
  
Any other input will be treated as a query to the research assistant.
"""
    
    def run(self):
        """Run the CLI."""
        print(self.welcome_message)
        
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()
                
                # Process commands
                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == "help":
                    print(self.help_message)
                
                elif user_input.lower() == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    print(self.welcome_message)
                
                elif user_input.lower() == "new":
                    self.session_id = str(uuid.uuid4())
                    print(f"Started new session with ID: {self.session_id}")
                
                elif user_input.lower() == "sessions":
                    sessions = self.rag_system.list_conversations()
                    if sessions:
                        print("Available sessions:")
                        for i, session in enumerate(sessions):
                            print(f"  {i+1}. {session}")
                    else:
                        print("No sessions found")
                
                elif user_input.lower().startswith("load "):
                    session_id = user_input[5:].strip()
                    if session_id in self.rag_system.list_conversations():
                        self.session_id = session_id
                        print(f"Loaded session: {self.session_id}")
                    else:
                        print(f"Session not found: {session_id}")
                
                elif user_input.lower().startswith("delete "):
                    session_id = user_input[7:].strip()
                    if session_id in self.rag_system.list_conversations():
                        self.rag_system.delete_conversation(session_id)
                        print(f"Deleted session: {session_id}")
                        if session_id == self.session_id:
                            self.session_id = str(uuid.uuid4())
                            print(f"Started new session with ID: {self.session_id}")
                    else:
                        print(f"Session not found: {session_id}")
                
                elif user_input.strip():
                    # Process query
                    print("Thinking...")
                    response = self.rag_system.process_chat(user_input, self.session_id)
                    
                    # Print response
                    print("\nResponse:")
                    print(response["response"])
                    
                    # Print context if available
                    if "context" in response and response["context"]:
                        print("\nSources:")
                        for i, doc in enumerate(response["context"]):
                            print(f"  [{i+1}] {doc['title']} by {', '.join(doc['authors'])}")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="arXiv Research Assistant CLI")
    parser.parse_args()
    
    # Run CLI
    cli = ArxivCLI()
    cli.run()

if __name__ == "__main__":
    main()
