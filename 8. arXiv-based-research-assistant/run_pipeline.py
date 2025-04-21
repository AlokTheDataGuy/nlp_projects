"""
Script to run the data pipeline with GPU optimizations.
"""
import argparse
import os
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
    parser.add_argument("--batch-size", type=int, default=32 if gpu_available else 16, 
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
        
        # Import here to avoid circular imports
        import time
        from data.chunker import DocumentChunker
        from utils.config import DATA_DIR
        
        # Initialize chunker
        chunker = DocumentChunker()
        
        # Get list of all processed papers
        input_dir = DATA_DIR / "processed" / "text"
        all_paper_ids = [f.stem for f in input_dir.glob("*.json")]
        
        # Filter out already chunked papers
        chunked_papers = chunker.get_chunked_papers()
        papers_to_chunk = [pid for pid in all_paper_ids if pid not in chunked_papers]
        
        print(f"Found {len(papers_to_chunk)} papers to chunk")
        
        # Process in batches
        total_batches = (len(papers_to_chunk) + args.chunk_batch_size - 1) // args.chunk_batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * args.chunk_batch_size
            end_idx = min(start_idx + args.chunk_batch_size, len(papers_to_chunk))
            
            batch = papers_to_chunk[start_idx:end_idx]
            
            print(f"Processing batch {batch_num+1}/{total_batches} with {len(batch)} papers")
            
            # Process batch
            successful = chunker.chunk_documents(batch)
            
            print(f"Batch {batch_num+1}/{total_batches} completed. Processed {len(successful)}/{len(batch)} papers successfully")
            
            # Delay between batches to allow memory to be freed
            if batch_num < total_batches - 1:
                print(f"Waiting {args.chunk_delay} seconds before next batch...")
                time.sleep(args.chunk_delay)
        
        print("Chunking completed")
        
        # Remove chunking from steps as we've handled it separately
        args.steps.remove("chunk")
    
    # Special handling for embedding to optimize GPU usage
    if "embed" in args.steps and args.use_gpu:
        print(f"Generating embeddings with GPU acceleration and batch size {args.batch_size}")
        
        # Import here to avoid circular imports
        import time
        from data.embedder import DocumentEmbedder
        from utils.config import DATA_DIR
        
        # Initialize embedder with GPU settings
        embedder = DocumentEmbedder(
            embedding_device="cuda" if gpu_available else "cpu",
            batch_size=args.batch_size
        )
        
        # Get list of all chunked papers
        chunks_dir = DATA_DIR / "chunks"
        all_paper_ids = [f.stem.replace("_chunks", "") for f in chunks_dir.glob("*_chunks.json")]
        
        # Filter out already embedded papers
        embedded_papers = embedder.get_processed_papers()
        papers_to_embed = [pid for pid in all_paper_ids if pid not in embedded_papers]
        
        print(f"Found {len(papers_to_embed)} papers to embed")
        
        # Process in batches
        embed_batch_size = 10  # Smaller batch size for embedding to avoid CUDA OOM
        total_batches = (len(papers_to_embed) + embed_batch_size - 1) // embed_batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * embed_batch_size
            end_idx = min(start_idx + embed_batch_size, len(papers_to_embed))
            
            batch = papers_to_embed[start_idx:end_idx]
            
            print(f"Processing embedding batch {batch_num+1}/{total_batches} with {len(batch)} papers")
            
            # Process batch
            successful = embedder.generate_embeddings(batch)
            
            print(f"Embedding batch {batch_num+1}/{total_batches} completed. Processed {len(successful)}/{len(batch)} papers successfully")
            
            # Delay between batches to allow memory to be freed
            if batch_num < total_batches - 1:
                print(f"Waiting 5 seconds before next batch...")
                time.sleep(5)
        
        print("Embedding completed")
        
        # Remove embedding from steps as we've handled it separately
        args.steps.remove("embed")
    
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