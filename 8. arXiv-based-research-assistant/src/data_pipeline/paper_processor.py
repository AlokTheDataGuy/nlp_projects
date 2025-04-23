#src/data_pipeline/paper_processor.py

"""
Paper Processor Module

This module handles downloading and processing arXiv papers.
"""

import time
import logging
import arxiv
import requests
import urllib.request
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivPaperProcessor:
    """
    Class for downloading and processing arXiv papers.
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the ArxivPaperProcessor.

        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.categories = self.config["data_pipeline"]["arxiv"]["categories"]
        self.max_papers = self.config["data_pipeline"]["arxiv"]["max_papers"]
        self.start_date = self.config["data_pipeline"]["arxiv"]["start_date"]

        # Create directories if they don't exist
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Dict containing configuration.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def download_papers(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Download papers from arXiv based on configured categories.

        Args:
            limit: Maximum number of papers to download. If None, uses the value from config.

        Returns:
            List of dictionaries containing paper metadata.
        """
        if limit is None:
            limit = self.max_papers

        logger.info(f"Downloading up to {limit} papers from categories: {', '.join(self.categories)}")

        # Initialize the arXiv client
        client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=5
        )

        # Download papers
        papers_metadata = []

        # Try a simpler approach - search each category without date filters
        logger.info("Using simplified search approach")
        results = []

        for category in self.categories:
            # Simple category query
            simple_query = f"cat:{category}"

            simple_search = arxiv.Search(
                query=simple_query,
                max_results=limit // len(self.categories) + 1,  # Distribute limit across categories
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            try:
                category_results = list(client.results(simple_search))
                logger.info(f"Found {len(category_results)} papers in category {category}")
                results.extend(category_results)
            except Exception as e:
                logger.error(f"Error searching category {category}: {str(e)}")

        # Process results
        for result in tqdm(results, total=len(results), desc="Downloading papers"):
            paper_id = result.get_short_id()
            pdf_path = self.raw_dir / f"{paper_id}.pdf"

            # Skip if already downloaded
            if pdf_path.exists():
                logger.info(f"Paper {paper_id} already downloaded, skipping.")
            else:
                try:
                    # Create the raw directory if it doesn't exist
                    self.raw_dir.mkdir(parents=True, exist_ok=True)

                    # Download the PDF manually instead of using the built-in method
                    pdf_url = result.pdf_url

                    # Use a custom download method
                    try:
                        response = requests.get(pdf_url, stream=True)
                        response.raise_for_status()

                        with open(pdf_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        logger.info(f"Downloaded paper {paper_id}")
                    except Exception as inner_e:
                        logger.error(f"Error downloading PDF for {paper_id}: {str(inner_e)}")
                        # Try alternative method as fallback
                        try:
                            urllib.request.urlretrieve(pdf_url, str(pdf_path))
                            logger.info(f"Downloaded paper {paper_id} using fallback method")
                        except Exception as fallback_e:
                            logger.error(f"Fallback download failed for {paper_id}: {str(fallback_e)}")
                            raise

                    # Add a small delay to avoid overloading the server
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error downloading paper {paper_id}: {str(e)}")
                    continue

            # Extract metadata
            metadata = {
                "id": paper_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "categories": result.categories,
                "published": result.published.strftime("%Y-%m-%d"),
                "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "pdf_path": str(pdf_path)
            }

            papers_metadata.append(metadata)

            # Break if we've reached the limit
            if len(papers_metadata) >= limit:
                break

        logger.info(f"Downloaded {len(papers_metadata)} papers")
        return papers_metadata

    def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """
        Get a specific paper by its ID.

        Args:
            paper_id: The arXiv ID of the paper.

        Returns:
            Dictionary containing paper metadata.
        """
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])

        try:
            result = next(client.results(search))
            pdf_path = self.raw_dir / f"{paper_id}.pdf"

            # Download if not already downloaded
            if not pdf_path.exists():
                result.download_pdf(str(pdf_path))
                logger.info(f"Downloaded paper {paper_id}")

            # Extract metadata
            metadata = {
                "id": paper_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "categories": result.categories,
                "published": result.published.strftime("%Y-%m-%d"),
                "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "pdf_path": str(pdf_path)
            }

            return metadata
        except Exception as e:
            logger.error(f"Error retrieving paper {paper_id}: {str(e)}")
            return None


if __name__ == "__main__":
    # Example usage
    processor = ArxivPaperProcessor()
    papers = processor.download_papers(limit=10)
    print(f"Downloaded {len(papers)} papers")
