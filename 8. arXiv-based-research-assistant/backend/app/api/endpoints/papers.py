from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.db.database import get_db
import logging
import arxiv

logger = logging.getLogger(__name__)

router = APIRouter()

class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = 10

class PaperResponse(BaseModel):
    papers: List[Dict[str, Any]]

@router.post("/search", response_model=PaperResponse)
async def search_papers(request: PaperSearchRequest, db: Session = Depends(get_db)):
    """
    Search for papers on ArXiv.
    """
    try:
        # Use the ArXiv API to search for papers
        client = arxiv.Client()
        search = arxiv.Search(
            query=request.query,
            max_results=request.max_results,
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
                'pdf_url': result.pdf_url,
                'entry_id': result.entry_id
            }
            papers.append(paper)
        
        return {"papers": papers}
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{paper_id}", response_model=Dict[str, Any])
async def get_paper(paper_id: str, db: Session = Depends(get_db)):
    """
    Get a paper by ID.
    """
    try:
        # Fetch paper from ArXiv
        client = arxiv.Client()
        search = arxiv.Search(
            id_list=[paper_id],
            max_results=1
        )
        
        results = list(client.results(search))
        if not results:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
        
        result = results[0]
        paper = {
            'paper_id': result.entry_id.split('/')[-1],
            'title': result.title,
            'abstract': result.summary,
            'authors': ', '.join(author.name for author in result.authors),
            'categories': ', '.join(result.categories),
            'published_date': result.published.strftime('%Y-%m-%d'),
            'pdf_url': result.pdf_url,
            'entry_id': result.entry_id
        }
        
        return paper
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))
