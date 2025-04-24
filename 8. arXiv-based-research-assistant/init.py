import os
import sys
import subprocess
import importlib

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    # Check if we can import key libraries
    required_libs = [
        "fastapi", "sqlalchemy", "transformers", "torch",
        "faiss", "arxiv", "sentence_transformers"
    ]

    missing_libs = []
    for lib in required_libs:
        try:
            importlib.import_module(lib)
            print(f"✅ {lib} is installed")
        except ImportError:
            print(f"❌ {lib} is not installed")
            missing_libs.append(lib)

    if missing_libs:
        print("\n❌ Some required libraries are missing. Please install them using:")
        print(f"pip install {' '.join(missing_libs)}")
        print("\nOr run: pip install -r backend/requirements.txt")
        return False

    return True

def main():
    print("Initializing ArXiv Expert Chatbot...")

    # Check dependencies
    if not check_dependencies():
        print("\n⚠️ Initialization aborted due to missing dependencies.")
        print("Please install the required dependencies and run this script again.")
        return

    # Create necessary directories
    print("\nCreating necessary directories...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("backend/data", exist_ok=True)

    # Initialize database
    print("Initializing database...")
    os.chdir("backend")
    subprocess.run([sys.executable, "-m", "app.db.init_db"])
    os.chdir("..")

    # Check GPU availability
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        else:
            print("⚠️ GPU not available. The LLM will run on CPU, which may be slow.")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")

    print("\nInitialization complete!")
    print("You can now run the application using 'python run.py'")
    print("For a more detailed dependency check, run 'python check_dependencies.py'")

if __name__ == "__main__":
    main()
