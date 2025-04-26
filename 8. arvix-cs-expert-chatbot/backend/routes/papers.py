from fastapi import APIRouter, HTTPException

from schemas.paper import (
    PaperSearchRequest,
    PaperSearchResponse,
    PaperContentRequest,
    PaperContentResponse,
    PaperModel
)
from services.arxiv_service import ArxivService
from services.paper_processor import PaperProcessor
from services.vector_service import VectorService

router = APIRouter()
arxiv_service = ArxivService()
paper_processor = PaperProcessor()
vector_service = VectorService()

@router.post("/search", response_model=PaperSearchResponse)
async def search_papers(request: PaperSearchRequest):
    """
    Search for papers on arXiv
    """
    try:
        # Extract concepts from query
        concepts = request.query.lower().split()
        stopwords = {"the", "a", "an", "in", "of", "and", "or", "to", "is", "are", "what", "how"}
        concepts = [word for word in concepts if word not in stopwords and len(word) > 2]

        # Map concepts to categories if none provided
        categories = request.categories
        if not categories and concepts:
            categories = arxiv_service.map_concepts_to_categories(concepts)

        # Search papers
        papers = await arxiv_service.search_papers(
            query=request.query,
            concepts=concepts,
            max_results=request.max_results,
            categories=categories
        )

        # Also search in vector DB if available
        vector_results = await vector_service.semantic_search(request.query, top_k=5)

        # Combine results (simple approach - could be improved)
        paper_dict = {paper['id']: paper for paper in papers}
        for paper in vector_results:
            if paper['id'] not in paper_dict:
                paper_dict[paper['id']] = paper

        combined_papers = list(paper_dict.values())

        return PaperSearchResponse(
            papers=combined_papers,
            total=len(combined_papers),
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching papers: {str(e)}")

@router.get("/paper/{paper_id}", response_model=PaperModel)
async def get_paper(paper_id: str):
    """
    Get a specific paper by ID
    """
    try:
        paper = await arxiv_service.get_paper_by_id(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
        return paper
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting paper: {str(e)}")

@router.post("/content", response_model=PaperContentResponse)
async def get_paper_content(request: PaperContentRequest):
    """
    Get the content of a paper
    """
    try:
        # Get paper metadata
        paper = await arxiv_service.get_paper_by_id(request.paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {request.paper_id} not found")

        # Extract content
        content = await paper_processor.extract_content(request.paper_id, request.sections)

        # Extract key points and citations
        key_points = paper_processor.extract_key_points(content)
        citations = paper_processor.extract_citations(content)

        return PaperContentResponse(
            paper=paper,
            content=content,
            key_points=key_points,
            citations=citations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting paper content: {str(e)}")

@router.get("/categories")
async def get_categories():
    """
    Get a list of arXiv CS categories
    """
    # Static list of CS categories
    cs_categories = {
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Computation and Language",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering",
        "cs.CG": "Computational Geometry",
        "cs.GT": "Computer Science and Game Theory",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.CR": "Cryptography and Security",
        "cs.DS": "Data Structures and Algorithms",
        "cs.DB": "Databases",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.AR": "Hardware Architecture",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MS": "Mathematical Software",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.NI": "Networking and Internet Architecture",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NA": "Numerical Analysis",
        "cs.OS": "Operating Systems",
        "cs.OH": "Other Computer Science",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SI": "Social and Information Networks",
        "cs.SE": "Software Engineering",
        "cs.SD": "Sound",
        "cs.SC": "Symbolic Computation",
        "cs.SY": "Systems and Control"
    }

    return {"categories": [{"id": k, "name": v} for k, v in cs_categories.items()]}