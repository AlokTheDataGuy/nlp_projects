from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PaperSearchRequest(BaseModel):
    """Request model for searching papers"""
    query: str = Field(..., description="The search query")
    max_results: int = Field(10, description="Maximum number of results to return")
    categories: Optional[List[str]] = Field(None, description="List of arXiv categories to search in")

class PaperModel(BaseModel):
    """Model for a paper"""
    id: str = Field(..., description="The arXiv ID")
    title: str = Field(..., description="The paper title")
    authors: List[str] = Field(..., description="List of authors")
    abstract: str = Field(..., description="The paper abstract")
    categories: List[str] = Field(..., description="List of arXiv categories")
    pdf_url: str = Field(..., description="URL to the PDF")
    published: Optional[str] = Field(None, description="Publication date")
    updated: Optional[str] = Field(None, description="Last update date")
    doi: Optional[str] = Field(None, description="DOI if available")
    score: Optional[float] = Field(None, description="Relevance score (for search results)")

class PaperSearchResponse(BaseModel):
    """Response model for paper search"""
    papers: List[PaperModel] = Field([], description="List of papers")
    total: int = Field(0, description="Total number of results")
    query: str = Field(..., description="The search query")

class PaperContentRequest(BaseModel):
    """Request model for getting paper content"""
    paper_id: str = Field(..., description="The arXiv ID")
    sections: Optional[List[str]] = Field(None, description="List of sections to extract")

class PaperContentResponse(BaseModel):
    """Response model for paper content"""
    paper: PaperModel = Field(..., description="The paper metadata")
    content: Dict[str, Any] = Field(..., description="The extracted content")
    key_points: Optional[List[str]] = Field(None, description="Key points extracted from the paper")
    citations: Optional[List[Dict[str, str]]] = Field(None, description="Citations extracted from the paper")