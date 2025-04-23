from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import re

from app.services.query_processor import QueryProcessor
from app.services.retrieval_service import RetrievalService
from app.services.entity_recognition import EntityRecognizer
from app.services.llm_service import MeditronService
from app.services.llama_service import LlamaService
from app.services.conversation_service import ConversationService

logger = logging.getLogger(__name__)
router = APIRouter()

class QuestionRequest(BaseModel):
    question: str
    max_results: Optional[int] = 3

class CurationRequest(BaseModel):
    question: str
    originalAnswer: str
    model: Optional[str] = "llama3.1:8b"

class CurationResponse(BaseModel):
    curatedAnswer: str

class AnswerResponse(BaseModel):
    answer: str
    source: str
    confidence: float
    entities: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    answers: List[AnswerResponse]
    entities_detected: List[dict]

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """
    Process a medical question and return relevant answers
    """
    try:
        # Initialize services
        entity_recognizer = EntityRecognizer()
        query_processor = QueryProcessor()
        retrieval_service = RetrievalService()
        conversation_service = ConversationService()

        # Check if this is a conversational query
        conv_type = conversation_service.detect_conversation_type(request.question)
        if conv_type:
            # This is a conversational query, return a conversational response
            response = conversation_service.get_response(conv_type)

            # Create a single answer with the conversational response
            answers = [
                AnswerResponse(
                    answer=response,
                    source="Conversation",
                    confidence=1.0,
                    entities=[]
                )
            ]

            return ChatResponse(
                answers=answers,
                entities_detected=[]
            )

        # Check if this is a very simple medical question that Llama can answer directly
        # These are typically definition questions that don't need complex retrieval
        simple_medical_patterns = [
            r"^what is ([a-zA-Z\s]+)\??$",
            r"^define ([a-zA-Z\s]+)\??$",
            r"^meaning of ([a-zA-Z\s]+)\??$",
            r"^tell me about ([a-zA-Z\s]+)\??$"
        ]

        is_simple_question = False
        for pattern in simple_medical_patterns:
            if re.match(pattern, request.question.lower()):
                is_simple_question = True
                break

        if is_simple_question and len(request.question.split()) <= 5:
            # Use Llama directly for very simple questions
            llama_service = LlamaService()
            try:
                direct_answer = llama_service.answer_simple_question(request.question)

                # Create a single answer with the direct response
                answers = [
                    AnswerResponse(
                        answer=direct_answer,
                        source="Llama 3.1 (Direct)",
                        confidence=0.9,
                        entities=[]
                    )
                ]

                return ChatResponse(
                    answers=answers,
                    entities_detected=[]
                )
            except Exception as e:
                logger.error(f"Error using Llama for simple question: {str(e)}")
                # Continue with normal processing if Llama fails

        # Process the medical question
        processed_query = query_processor.process_query(request.question)

        # Extract medical entities
        entities = entity_recognizer.extract_entities(request.question)

        # Use hybrid retrieval (FAISS + Meditron)
        results = await retrieval_service.hybrid_retrieve(
            processed_query,
            max_results=request.max_results,
            entities=entities
        )

        # Format response
        answers = [
            AnswerResponse(
                answer=result["answer"],
                source=result["source"],
                confidence=result["score"],
                entities=result.get("related_entities")
            )
            for result in results
        ]

        return ChatResponse(
            answers=answers,
            entities_detected=entities
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/curate", response_model=CurationResponse)
async def curate_answer(request: CurationRequest):
    """
    Curate and enhance a medical answer using Llama 3.1
    """
    try:
        # Initialize Llama service
        llama_service = LlamaService(model_name=request.model)

        # Curate the answer
        curated_answer = llama_service.curate_response(
            question=request.question,
            original_answer=request.originalAnswer
        )

        return CurationResponse(curatedAnswer=curated_answer)

    except Exception as e:
        # If there's an error, return the original answer
        logger.error(f"Error in curation: {str(e)}")
        return CurationResponse(curatedAnswer=request.originalAnswer)

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if Ollama is running
        meditron_service = MeditronService()
        llama_service = LlamaService()

        meditron_status = meditron_service.health_check()
        llama_status = llama_service.health_check()

        return {
            "status": "healthy",
            "ollama": {
                "meditron": "online" if meditron_status else "offline",
                "llama": "online" if llama_status else "offline"
            }
        }
    except Exception as e:
        return {
            "status": "healthy",
            "ollama": {
                "meditron": "offline",
                "llama": "offline"
            },
            "error": str(e)
        }
