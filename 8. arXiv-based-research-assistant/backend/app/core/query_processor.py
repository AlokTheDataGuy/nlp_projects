import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from app.models.models import QueryCache, Paper
from app.core.vector_store import VectorIndex
import arxiv
from datetime import datetime

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, vector_index: VectorIndex):
        """
        Initialize the query processor.
        
        Args:
            vector_index: Vector index for semantic search
        """
        self.vector_index = vector_index
    
    def process_query(self, query: str, db: Session) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: User query
            db: Database session
            
        Returns:
            Dictionary with query intent and relevant information
        """
        try:
            # Check cache first
            cached_response = self._check_cache(query, db)
            if cached_response:
                logger.info(f"Cache hit for query: {query}")
                return cached_response
            
            # Analyze query intent
            intent = self._analyze_intent(query)
            
            # Get relevant papers based on intent
            relevant_papers = self._get_relevant_papers(query, intent, db)
            
            # Extract context from papers
            context = self._extract_context(relevant_papers)
            
            # Prepare response
            response = {
                'query': query,
                'intent': intent,
                'relevant_papers': relevant_papers,
                'context': context,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache the response
            self._cache_response(query, response, db)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _analyze_intent(self, query: str) -> str:
        """
        Analyze the intent of a query.
        
        Args:
            query: User query
            
        Returns:
            Query intent (search, explanation, summary, etc.)
        """
        # Simple rule-based intent classification
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['find', 'search', 'look for', 'papers on']):
            return 'search'
        elif any(keyword in query_lower for keyword in ['explain', 'what is', 'how does', 'definition']):
            return 'explanation'
        elif any(keyword in query_lower for keyword in ['summarize', 'summary', 'overview']):
            return 'summary'
        elif any(keyword in query_lower for keyword in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        else:
            return 'general'
    
    def _get_relevant_papers(self, query: str, intent: str, db: Session) -> List[Dict[str, Any]]:
        """
        Get relevant papers based on the query and intent.
        
        Args:
            query: User query
            intent: Query intent
            db: Database session
            
        Returns:
            List of relevant paper dictionaries
        """
        # Try vector search first if we have vectors
        vector_results = self.vector_index.search(query, k=5)
        
        # If we have vector results, use them
        if vector_results:
            papers = []
            for result in vector_results:
                # Check if paper exists in database
                paper = db.query(Paper).filter(Paper.paper_id == result['paper_id']).first()
                
                if paper:
                    papers.append({
                        'paper_id': paper.paper_id,
                        'title': paper.title,
                        'abstract': paper.abstract,
                        'authors': paper.authors,
                        'categories': paper.categories,
                        'published_date': paper.published_date,
                        'url': f"https://arxiv.org/abs/{paper.paper_id}"
                    })
                else:
                    # Use metadata from vector result
                    papers.append({
                        'paper_id': result['paper_id'],
                        'title': result['metadata'].get('title', 'Unknown Title'),
                        'abstract': result['text'],
                        'authors': result['metadata'].get('authors', 'Unknown Authors'),
                        'categories': result['metadata'].get('categories', ''),
                        'published_date': result['metadata'].get('published_date', ''),
                        'url': f"https://arxiv.org/abs/{result['paper_id']}"
                    })
            
            return papers
        
        # Fallback to ArXiv search
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in client.results(search):
                paper = {
                    'paper_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': ', '.join(author.name for author in result.authors),
                    'categories': ', '.join(result.categories),
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'url': f"https://arxiv.org/abs/{result.entry_id.split('/')[-1]}"
                }
                papers.append(paper)
                
                # Save paper to database for future use
                self._save_paper_to_db(paper, db)
            
            return papers
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _save_paper_to_db(self, paper_data: Dict[str, Any], db: Session) -> Optional[Paper]:
        """
        Save a paper to the database.
        
        Args:
            paper_data: Paper data dictionary
            db: Database session
            
        Returns:
            Paper model instance or None if failed
        """
        try:
            # Check if paper already exists
            existing_paper = db.query(Paper).filter(Paper.paper_id == paper_data['paper_id']).first()
            if existing_paper:
                # Update last_accessed timestamp
                existing_paper.last_accessed = datetime.utcnow()
                db.commit()
                return existing_paper
            
            # Create new paper
            paper = Paper(
                paper_id=paper_data['paper_id'],
                title=paper_data['title'],
                abstract=paper_data['abstract'],
                authors=paper_data['authors'],
                categories=paper_data['categories'],
                published_date=paper_data['published_date'],
                processed=False,
                last_accessed=datetime.utcnow()
            )
            
            db.add(paper)
            db.commit()
            db.refresh(paper)
            
            # Add to vector index
            self.vector_index.add_paper(
                paper.paper_id,
                paper.abstract,
                {
                    'title': paper.title,
                    'authors': paper.authors,
                    'categories': paper.categories,
                    'published_date': paper.published_date
                }
            )
            
            return paper
        except Exception as e:
            logger.error(f"Error saving paper to database: {e}")
            db.rollback()
            return None
    
    def _extract_context(self, papers: List[Dict[str, Any]]) -> str:
        """
        Extract context from papers for the LLM.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Context string
        """
        context_parts = []
        
        for i, paper in enumerate(papers):
            # Extract paper information
            title = paper.get('title', 'Untitled')
            authors = paper.get('authors', 'Unknown authors')
            abstract = paper.get('abstract', '')
            
            # Format paper information
            paper_context = f"[{i+1}] {title} by {authors}\n\nAbstract: {abstract}\n\n"
            context_parts.append(paper_context)
        
        # Join all context parts
        return "\n".join(context_parts)
    
    def _check_cache(self, query: str, db: Session) -> Optional[Dict[str, Any]]:
        """
        Check if a query is in the cache.
        
        Args:
            query: User query
            db: Database session
            
        Returns:
            Cached response or None
        """
        query_hash = self._hash_query(query)
        cached_query = db.query(QueryCache).filter(QueryCache.query_hash == query_hash).first()
        
        if cached_query:
            # Update usage count and timestamp
            cached_query.usage_count += 1
            cached_query.timestamp = datetime.utcnow()
            db.commit()
            
            # Return cached response
            import json
            return json.loads(cached_query.response)
        
        return None
    
    def _cache_response(self, query: str, response: Dict[str, Any], db: Session) -> bool:
        """
        Cache a query response.
        
        Args:
            query: User query
            response: Response to cache
            db: Database session
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query_hash = self._hash_query(query)
            
            import json
            cache_entry = QueryCache(
                query_hash=query_hash,
                query_text=query,
                response=json.dumps(response),
                timestamp=datetime.utcnow(),
                usage_count=1
            )
            
            db.add(cache_entry)
            db.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            db.rollback()
            return False
    
    def _hash_query(self, query: str) -> str:
        """
        Create a hash of a query for caching.
        
        Args:
            query: User query
            
        Returns:
            Query hash
        """
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Create hash
        return hashlib.md5(normalized_query.encode()).hexdigest()
