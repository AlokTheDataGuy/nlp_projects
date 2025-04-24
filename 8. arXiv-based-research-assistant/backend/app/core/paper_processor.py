import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.models import Paper, Concept, Summary, ConceptRelation
from app.core.llm import llm_manager
from app.core.vector_store import VectorIndex
from app.db.database import SessionLocal
import re

logger = logging.getLogger(__name__)

class PaperProcessor:
    def __init__(self, vector_index: VectorIndex):
        """
        Initialize the paper processor.
        
        Args:
            vector_index: Vector index for storing embeddings
        """
        self.vector_index = vector_index
    
    def process_paper(self, paper_id: str):
        """
        Process a paper.
        
        Args:
            paper_id: Paper ID
        """
        try:
            # Create a new database session
            db = SessionLocal()
            
            # Get the paper from the database
            paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
            
            if not paper:
                logger.error(f"Paper with ID {paper_id} not found")
                return
            
            # Check if the paper has already been processed
            if paper.processed:
                logger.info(f"Paper with ID {paper_id} has already been processed")
                return
            
            # Generate summary
            self._generate_summary(paper, db)
            
            # Extract concepts
            self._extract_concepts(paper, db)
            
            # Add to vector index
            self._add_to_vector_index(paper)
            
            # Mark paper as processed
            paper.processed = True
            db.commit()
            
            logger.info(f"Paper with ID {paper_id} processed successfully")
        except Exception as e:
            logger.error(f"Error processing paper with ID {paper_id}: {e}")
            if db:
                db.rollback()
        finally:
            if db:
                db.close()
    
    def _generate_summary(self, paper: Paper, db: Session):
        """
        Generate a summary for a paper.
        
        Args:
            paper: Paper model instance
            db: Database session
        """
        try:
            # Check if summary already exists
            existing_summary = db.query(Summary).filter(Summary.paper_id == paper.paper_id).first()
            if existing_summary:
                logger.info(f"Summary for paper with ID {paper.paper_id} already exists")
                return
            
            # Prepare prompt for summary generation
            prompt = f"""
            Title: {paper.title}
            Authors: {paper.authors}
            Abstract: {paper.abstract}
            
            Please generate a comprehensive summary of this research paper with the following sections:
            1. Executive Summary (2-3 sentences overview)
            2. Methodology Summary (key methods used)
            3. Findings Summary (main results)
            4. Implications (significance of the work)
            
            Format your response as:
            Executive Summary: [your summary]
            Methodology Summary: [your summary]
            Findings Summary: [your summary]
            Implications: [your summary]
            """
            
            # Generate summary using LLM
            summary_text = llm_manager.generate_response("", prompt)
            
            # Parse the summary
            executive_summary = self._extract_section(summary_text, "Executive Summary")
            methodology_summary = self._extract_section(summary_text, "Methodology Summary")
            findings_summary = self._extract_section(summary_text, "Findings Summary")
            implications = self._extract_section(summary_text, "Implications")
            
            # Create summary
            summary = Summary(
                paper_id=paper.paper_id,
                executive_summary=executive_summary,
                methodology_summary=methodology_summary,
                findings_summary=findings_summary,
                implications=implications
            )
            
            db.add(summary)
            db.commit()
            
            logger.info(f"Summary generated for paper with ID {paper.paper_id}")
        except Exception as e:
            logger.error(f"Error generating summary for paper with ID {paper.paper_id}: {e}")
            db.rollback()
    
    def _extract_concepts(self, paper: Paper, db: Session):
        """
        Extract concepts from a paper.
        
        Args:
            paper: Paper model instance
            db: Database session
        """
        try:
            # Prepare prompt for concept extraction
            prompt = f"""
            Title: {paper.title}
            Abstract: {paper.abstract}
            
            Please extract the key technical concepts from this research paper. For each concept:
            1. Provide the concept name
            2. Give a clear definition
            
            Format your response as a list:
            Concept: [concept name]
            Definition: [definition]
            
            Concept: [concept name]
            Definition: [definition]
            
            ...and so on.
            """
            
            # Generate concepts using LLM
            concepts_text = llm_manager.generate_response("", prompt)
            
            # Parse the concepts
            concept_pattern = r"Concept: (.*?)\nDefinition: (.*?)(?=\n\nConcept:|$)"
            concept_matches = re.findall(concept_pattern, concepts_text, re.DOTALL)
            
            # Add concepts to database
            for name, definition in concept_matches:
                name = name.strip()
                definition = definition.strip()
                
                # Check if concept already exists
                existing_concept = db.query(Concept).filter(Concept.name == name).first()
                if existing_concept:
                    logger.info(f"Concept '{name}' already exists")
                    continue
                
                # Create concept
                concept = Concept(
                    name=name,
                    definition=definition,
                    paper_id=paper.paper_id
                )
                
                db.add(concept)
            
            db.commit()
            
            # Extract concept relations
            self._extract_concept_relations(paper, db)
            
            logger.info(f"Concepts extracted for paper with ID {paper.paper_id}")
        except Exception as e:
            logger.error(f"Error extracting concepts for paper with ID {paper.paper_id}: {e}")
            db.rollback()
    
    def _extract_concept_relations(self, paper: Paper, db: Session):
        """
        Extract relations between concepts.
        
        Args:
            paper: Paper model instance
            db: Database session
        """
        try:
            # Get all concepts for this paper
            concepts = db.query(Concept).filter(Concept.paper_id == paper.paper_id).all()
            
            if len(concepts) < 2:
                logger.info(f"Not enough concepts to extract relations for paper with ID {paper.paper_id}")
                return
            
            # Prepare concept names
            concept_names = [concept.name for concept in concepts]
            
            # Prepare prompt for relation extraction
            prompt = f"""
            Title: {paper.title}
            Abstract: {paper.abstract}
            
            Concepts extracted from this paper:
            {', '.join(concept_names)}
            
            Please identify relationships between these concepts. For each relationship:
            1. Specify the source concept
            2. Specify the target concept
            3. Describe the relationship type (e.g., "is a type of", "is used in", "depends on", etc.)
            
            Format your response as a list:
            Source: [source concept]
            Target: [target concept]
            Relation: [relation type]
            
            Source: [source concept]
            Target: [target concept]
            Relation: [relation type]
            
            ...and so on.
            """
            
            # Generate relations using LLM
            relations_text = llm_manager.generate_response("", prompt)
            
            # Parse the relations
            relation_pattern = r"Source: (.*?)\nTarget: (.*?)\nRelation: (.*?)(?=\n\nSource:|$)"
            relation_matches = re.findall(relation_pattern, relations_text, re.DOTALL)
            
            # Add relations to database
            for source_name, target_name, relation_type in relation_matches:
                source_name = source_name.strip()
                target_name = target_name.strip()
                relation_type = relation_type.strip()
                
                # Find source and target concepts
                source_concept = db.query(Concept).filter(Concept.name == source_name).first()
                target_concept = db.query(Concept).filter(Concept.name == target_name).first()
                
                if not source_concept or not target_concept:
                    logger.warning(f"Source or target concept not found: {source_name} -> {target_name}")
                    continue
                
                # Check if relation already exists
                existing_relation = db.query(ConceptRelation).filter(
                    ConceptRelation.source_concept_id == source_concept.concept_id,
                    ConceptRelation.target_concept_id == target_concept.concept_id
                ).first()
                
                if existing_relation:
                    logger.info(f"Relation between '{source_name}' and '{target_name}' already exists")
                    continue
                
                # Create relation
                relation = ConceptRelation(
                    source_concept_id=source_concept.concept_id,
                    target_concept_id=target_concept.concept_id,
                    relation_type=relation_type,
                    evidence_paper_id=paper.paper_id
                )
                
                db.add(relation)
            
            db.commit()
            
            logger.info(f"Concept relations extracted for paper with ID {paper.paper_id}")
        except Exception as e:
            logger.error(f"Error extracting concept relations for paper with ID {paper.paper_id}: {e}")
            db.rollback()
    
    def _add_to_vector_index(self, paper: Paper):
        """
        Add a paper to the vector index.
        
        Args:
            paper: Paper model instance
        """
        try:
            # Add abstract to vector index
            self.vector_index.add_paper(
                paper.paper_id,
                paper.abstract,
                {
                    'title': paper.title,
                    'authors': paper.authors,
                    'categories': paper.categories,
                    'published_date': paper.published_date,
                    'type': 'abstract'
                }
            )
            
            # If full text is available, add chunks
            if paper.full_text:
                # Split full text into chunks
                chunks = self._split_text(paper.full_text, chunk_size=1000, overlap=200)
                
                # Add chunks to vector index
                for i, chunk in enumerate(chunks):
                    self.vector_index.add_paper(
                        f"{paper.paper_id}_chunk_{i}",
                        chunk,
                        {
                            'paper_id': paper.paper_id,
                            'title': paper.title,
                            'authors': paper.authors,
                            'categories': paper.categories,
                            'published_date': paper.published_date,
                            'type': 'full_text',
                            'chunk_index': i
                        }
                    )
            
            logger.info(f"Paper with ID {paper.paper_id} added to vector index")
        except Exception as e:
            logger.error(f"Error adding paper with ID {paper.paper_id} to vector index: {e}")
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Extract a section from text.
        
        Args:
            text: Text to extract from
            section_name: Name of the section
            
        Returns:
            Extracted section text
        """
        pattern = f"{section_name}: (.*?)(?=\n\n|$)"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to find a sentence boundary
            if end < len(text):
                # Look for sentence boundary within 100 characters
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in ['.', '!', '?', '\n'] and i + 1 < len(text) and text[i + 1] == ' ':
                        end = i + 1
                        break
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks

# Create a singleton instance with a placeholder vector index
# This will be properly initialized in the application startup
paper_processor = PaperProcessor(VectorIndex())
