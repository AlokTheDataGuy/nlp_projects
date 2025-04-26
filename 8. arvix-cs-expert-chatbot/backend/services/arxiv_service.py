import arxiv
import asyncio
from typing import List, Dict, Any, Optional

class ArxivService:
    def __init__(self, cache_dir: str = "../data/paper_cache"):
        self.cache_dir = cache_dir

    async def search_papers(self, query: str, concepts: Optional[List[str]] = None,
                           max_results: int = 10, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for papers on arXiv based on query and concepts

        Args:
            query: The search query
            concepts: List of key concepts extracted from the query
            max_results: Maximum number of results to return
            categories: List of arXiv categories to search in (e.g., cs.AI, cs.CL)

        Returns:
            List of paper metadata dictionaries
        """
        # Enhance query with concepts if provided
        enhanced_query = query
        if concepts and len(concepts) > 0:
            # Add concepts to query with OR operator
            concept_query = " OR ".join(concepts)
            enhanced_query = f"{query} {concept_query}"

        # Add category filters if provided
        if categories and len(categories) > 0:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            enhanced_query = f"{enhanced_query} AND ({category_filter})"
        else:
            # Default to CS categories if none specified
            enhanced_query = f"{enhanced_query} AND (cat:cs.*)"

        # Execute search using arxiv API
        search = arxiv.Search(
            query=enhanced_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        # Process results
        papers = []

        # Use asyncio to run the potentially blocking operation in a thread pool
        def fetch_results():
            results = list(search.results())
            return results

        results = await asyncio.to_thread(fetch_results)

        for result in results:
            paper = {
                'id': result.entry_id.split('/')[-1],  # Extract ID from URL
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'categories': result.categories,
                'pdf_url': result.pdf_url,
                'published': result.published.isoformat() if result.published else None,
                'updated': result.updated.isoformat() if result.updated else None,
                'doi': result.doi
            }
            papers.append(paper)

        return papers

    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific paper by its arXiv ID

        Args:
            paper_id: The arXiv ID of the paper

        Returns:
            Paper metadata dictionary or None if not found
        """
        search = arxiv.Search(
            id_list=[paper_id],
            max_results=1
        )

        # Use asyncio to run the potentially blocking operation in a thread pool
        def fetch_result():
            results = list(search.results())
            return results[0] if results else None

        result = await asyncio.to_thread(fetch_result)

        if not result:
            return None

        return {
            'id': result.entry_id.split('/')[-1],
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'abstract': result.summary,
            'categories': result.categories,
            'pdf_url': result.pdf_url,
            'published': result.published.isoformat() if result.published else None,
            'updated': result.updated.isoformat() if result.updated else None,
            'doi': result.doi
        }

    def map_concepts_to_categories(self, concepts: List[str]) -> List[str]:
        """
        Map extracted concepts to arXiv categories

        Args:
            concepts: List of concepts extracted from query

        Returns:
            List of relevant arXiv categories
        """
        # This is a simplified mapping - could be enhanced with ML/NLP
        category_keywords = {
            "cs.AI": ["ai", "artificial intelligence", "machine learning", "neural", "deep learning"],
            "cs.CL": ["nlp", "natural language", "language model", "text", "linguistics", "translation"],
            "cs.CV": ["computer vision", "image", "object detection", "recognition", "segmentation"],
            "cs.LG": ["learning", "neural network", "deep learning", "reinforcement", "supervised"],
            "cs.IR": ["information retrieval", "search", "recommendation", "ranking"],
            "cs.SE": ["software", "engineering", "development", "testing", "verification"],
            "cs.CY": ["security", "privacy", "cryptography", "encryption"],
            "cs.DB": ["database", "data management", "sql", "nosql", "query"],
            "cs.DC": ["distributed", "parallel", "concurrency", "cloud"],
            "cs.NE": ["neural", "evolutionary", "genetic algorithm", "optimization"]
        }

        matched_categories = set()
        for concept in concepts:
            concept_lower = concept.lower()
            for category, keywords in category_keywords.items():
                if any(keyword in concept_lower for keyword in keywords):
                    matched_categories.add(category)

        # If no matches, return general CS category
        return list(matched_categories) if matched_categories else ["cs.AI", "cs.CL", "cs.LG"]