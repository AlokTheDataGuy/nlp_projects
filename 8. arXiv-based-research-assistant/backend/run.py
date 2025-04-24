import uvicorn
import os
import sys

# Add the current directory to the path so that the main module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Print debugging information
print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path}")

if __name__ == "__main__":
    """
    Run the FastAPI backend server.

    This script is used to run only the backend server for development or testing.
    For running both frontend and backend together, use the run.py script in the root directory.
    """
    print("Starting ArXiv Expert Chatbot backend server...")

    # Get the absolute path to the main.py file
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    main_module = "main"

    # Check if main.py exists
    if not os.path.exists(main_path):
        print(f"Error: {main_path} not found.")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir())
        sys.exit(1)

    # Run the server
    uvicorn.run(main_module + ":app", host="0.0.0.0", port=8000, reload=True)
