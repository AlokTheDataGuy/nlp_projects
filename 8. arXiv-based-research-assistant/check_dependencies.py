import importlib
import sys
import subprocess
import pkg_resources

def check_library(library_name):
    """Check if a library is installed and get its version"""
    try:
        module = importlib.import_module(library_name)
        try:
            version = pkg_resources.get_distribution(library_name).version
            return True, version
        except pkg_resources.DistributionNotFound:
            return True, "Unknown version"
    except ImportError:
        return False, None

def main():
    print("Checking required libraries for ArXiv Expert Chatbot...")
    
    # Core dependencies
    core_libs = [
        "fastapi", "sqlalchemy", "faiss", "transformers", 
        "sentence_transformers", "torch", "bitsandbytes", "accelerate"
    ]
    
    # ArXiv integration
    arxiv_libs = ["arxiv", "fitz", "pdf2image"]  # fitz is PyMuPDF
    
    # NLP and processing
    nlp_libs = ["nltk", "spacy", "sklearn"]
    
    # Visualization
    viz_libs = ["networkx", "plotly", "matplotlib"]
    
    # Utility libraries
    util_libs = ["psutil", "aiohttp", "requests", "tqdm", "numpy"]
    
    # Check all libraries
    all_libs = {
        "Core Dependencies": core_libs,
        "ArXiv Integration": arxiv_libs,
        "NLP and Processing": nlp_libs,
        "Visualization": viz_libs,
        "Utility Libraries": util_libs
    }
    
    missing_libs = []
    
    print("\nLibrary Status:")
    print("-" * 60)
    print(f"{'Library':<25} {'Status':<10} {'Version':<15}")
    print("-" * 60)
    
    for category, libs in all_libs.items():
        print(f"\n{category}:")
        for lib in libs:
            installed, version = check_library(lib)
            status = "✅ Installed" if installed else "❌ Missing"
            version_str = version if version else "N/A"
            print(f"{lib:<25} {status:<10} {version_str:<15}")
            
            if not installed:
                missing_libs.append(lib)
    
    print("\n" + "-" * 60)
    
    if missing_libs:
        print(f"\n❌ Missing libraries: {', '.join(missing_libs)}")
        print("\nTo install missing libraries, run:")
        print("pip install " + " ".join(missing_libs))
    else:
        print("\n✅ All required libraries are installed!")
    
    # Check GPU availability for PyTorch
    print("\nChecking GPU availability for PyTorch...")
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

if __name__ == "__main__":
    main()
