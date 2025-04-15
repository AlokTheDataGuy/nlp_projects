import uvicorn
import os
import sys

def run_backend():
    """Run the FastAPI backend server."""
    # Get the current directory (should be the backend directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Add the current directory to the Python path
    sys.path.insert(0, current_dir)

    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run_backend()
