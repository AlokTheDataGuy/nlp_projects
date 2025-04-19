"""
Script to run the data pipeline with GPU optimizations.
"""
import argparse
import torch
from data.pipeline import run_pipeline
from utils.config import ARXIV_CATEGORIES, PAPERS_PER_CATEGORY, EMBEDDING_BATCH_SIZE

def main():
    """Main function."""
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"GPU detected: {gpu_name} with {gpu_memory:.2f} GB memory")
    else:
        print("No GPU detected. Using CPU only.")
    
    parser = argparse.ArgumentParser(description="Run arXiv data pipeline with GPU optimizations")
    parser.add_argument("--categories", nargs="+", default=ARXIV_CATEGORIES, help="arXiv categories to download")
    parser.add_argument("--papers-per-category", type=int, default=PAPERS_PER_CATEGORY, help="Number of papers per category")
    parser.add_argument("--paper-ids", nargs="+", help="Specific paper IDs to process")
    parser.add_argument("--steps", nargs="+", default=["download", "process", "chunk", "embed", "index"], 
                        help="Pipeline steps to run")
    parser.add_argument("--download-only", action="store_true", help="Only download papers")
    parser.add_argument("--process-only", action="store_true", help="Only process papers")
    parser.add_argument("--chunk-only", action="store_true", help="Only chunk documents")
    parser.add_argument("--embed-only", action="store_true", help="Only generate embeddings")
    parser.add_argument("--index-only", action="store_true", help="Only build index")
    
    # GPU optimization parameters
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE, 
                        help="Batch size for embedding generation")
    parser.add_argument("--use-gpu", action="store_true", default=gpu_available,
                        help="Use GPU for processing if available")
    parser.add_argument("--chunk-batch-size", type=int, default=50,
                        help="Number of papers to chunk in each batch")
    parser.add_argument("--chunk-delay", type=int, default=2,
                        help="Delay in seconds between chunking batches")
    
    args = parser.parse_args()
    
    # Handle specific step flags
    if args.download_only:
        args.steps = ["download"]
    elif args.process_only:
        args.steps = ["process"]
    elif args.chunk_only:
        args.steps = ["chunk"]
    elif args.embed_only:
        args.steps = ["embed"]
    elif args.index_only:
        args.steps = ["index"]
    
    # Set environment variables for GPU usage
    import os
    if args.use_gpu and gpu_available:
        os.environ["EMBEDDING_DEVICE"] = "cuda"
        os.environ["EMBEDDING_BATCH_SIZE"] = str(args.batch_size)
        print(f"Using GPU for embeddings with batch size {args.batch_size}")
    else:
        os.environ["EMBEDDING_DEVICE"] = "cpu"
        print("Using CPU for embeddings")
    
    # Special handling for chunking to avoid memory issues
    if "chunk" in args.steps:
        print(f"Chunking documents in batches of {args.chunk_batch_size} with {args.chunk_delay}s delay between batches")
        from chunk_in_batches import chunk_in_batches
        chunk_in_batches(batch_size=args.chunk_batch_size, delay=args.chunk_delay)
        # Remove chunking from steps as we've handled it separately
        args.steps.remove("chunk")
        
        if not args.steps:  # If no steps left, exit
            print("Chunking completed. Exiting.")
            return
    
    # Special handling for embedding to optimize GPU usage
    if "embed" in args.steps:
        print(f"Generating embeddings with batch size {args.batch_size}")
        from embed_in_batches import embed_in_batches
        embed_in_batches(batch_size=args.batch_size, delay=5)
        # Remove embedding from steps as we've handled it separately
        args.steps.remove("embed")
        
        if not args.steps:  # If no steps left, exit
            print("Embedding completed. Exiting.")
            return
    
    # Run remaining pipeline steps
    if args.steps:
        print(f"Running remaining pipeline steps: {args.steps}")
        run_pipeline(
            categories=args.categories,
            papers_per_category=args.papers_per_category,
            paper_ids=args.paper_ids,
            steps=args.steps
        )

if __name__ == "__main__":
    main()
