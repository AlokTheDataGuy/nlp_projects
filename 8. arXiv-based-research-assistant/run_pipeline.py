"""
Script to run the data pipeline.
"""
import argparse
from data.pipeline import run_pipeline
from utils.config import ARXIV_CATEGORIES, PAPERS_PER_CATEGORY

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run arXiv data pipeline")
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
    
    # Run pipeline
    run_pipeline(
        categories=args.categories,
        papers_per_category=args.papers_per_category,
        paper_ids=args.paper_ids,
        steps=args.steps
    )

if __name__ == "__main__":
    main()
