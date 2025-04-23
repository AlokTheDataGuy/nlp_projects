"""
Text Extractor Module

This module handles extracting text from PDF files and cleaning the extracted text.
"""

import os
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import yaml
import PyPDF2
from pdfminer.high_level import extract_text
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextExtractor:
    """
    Class for extracting and cleaning text from PDF files.
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the TextExtractor.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.chunk_size = self.config["data_pipeline"]["processing"]["chunk_size"]
        self.chunk_overlap = self.config["data_pipeline"]["processing"]["chunk_overlap"]
        
        # Create directories if they don't exist
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Extracted text as a string.
        """
        try:
            # First try with PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            
            # If PyPDF2 fails to extract meaningful text, try with pdfminer
            if not text.strip() or len(text) < 100:
                text = extract_text(pdf_path)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean the extracted text.
        
        Args:
            text: The text to clean.
            
        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers and footers (common patterns)
        text = re.sub(r'(?i)(arxiv:|submitted to|appears in|proceedings of|copyright|all rights reserved)', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
        
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', '', text)
        
        # Remove references section (often starts with "References" or "Bibliography")
        text = re.sub(r'(?i)(References|Bibliography)[\s\S]*$', '', text)
        
        return text.strip()
    
    def segment_into_sections(self, text: str) -> Dict[str, str]:
        """
        Segment the text into sections.
        
        Args:
            text: The text to segment.
            
        Returns:
            Dictionary with section names as keys and section text as values.
        """
        # Common section headers in research papers
        section_patterns = [
            r'(?i)abstract',
            r'(?i)introduction',
            r'(?i)related work',
            r'(?i)background',
            r'(?i)methodology',
            r'(?i)method',
            r'(?i)approach',
            r'(?i)experiment',
            r'(?i)evaluation',
            r'(?i)result',
            r'(?i)discussion',
            r'(?i)conclusion',
            r'(?i)future work',
            r'(?i)acknowledgment'
        ]
        
        # Combine patterns
        combined_pattern = '|'.join([f'({pattern})' for pattern in section_patterns])
        
        # Find all section headers
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))
        
        sections = {}
        
        # If no sections found, return the whole text as "content"
        if not matches:
            sections["content"] = text
            return sections
        
        # Extract sections
        for i, match in enumerate(matches):
            section_name = match.group(0).strip()
            start_pos = match.start()
            
            # End position is the start of the next section or the end of the text
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            # Extract section text
            section_text = text[start_pos:end_pos].strip()
            
            # Remove the section header from the text
            section_text = re.sub(f'^{re.escape(section_name)}', '', section_text, flags=re.IGNORECASE).strip()
            
            sections[section_name.lower()] = section_text
        
        return sections
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk.
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between chunks in characters.
            
        Returns:
            List of text chunks.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed the chunk size, save the current chunk and start a new one
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep the last few sentences for overlap
                overlap_size = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1  # +1 for the space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for the space
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_paper(self, paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a paper and extract its text.
        
        Args:
            paper_metadata: Dictionary containing paper metadata.
            
        Returns:
            Dictionary with paper metadata and extracted text.
        """
        pdf_path = paper_metadata["pdf_path"]
        paper_id = paper_metadata["id"]
        
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Segment into sections
        sections = self.segment_into_sections(cleaned_text)
        
        # Chunk each section
        chunked_sections = {}
        for section_name, section_text in sections.items():
            chunked_sections[section_name] = self.chunk_text(section_text)
        
        # Create processed paper data
        processed_paper = {
            **paper_metadata,
            "full_text": cleaned_text,
            "sections": sections,
            "chunked_sections": chunked_sections
        }
        
        # Save processed data
        output_path = self.processed_dir / f"{paper_id}.yaml"
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(processed_paper, f, default_flow_style=False, allow_unicode=True)
        
        return processed_paper
    
    def process_papers(self, papers_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple papers.
        
        Args:
            papers_metadata: List of dictionaries containing paper metadata.
            
        Returns:
            List of dictionaries with paper metadata and extracted text.
        """
        processed_papers = []
        
        for paper_metadata in tqdm(papers_metadata, desc="Processing papers"):
            try:
                processed_paper = self.process_paper(paper_metadata)
                processed_papers.append(processed_paper)
            except Exception as e:
                logger.error(f"Error processing paper {paper_metadata['id']}: {str(e)}")
        
        return processed_papers


if __name__ == "__main__":
    # Example usage
    from paper_processor import ArxivPaperProcessor
    
    processor = ArxivPaperProcessor()
    papers = processor.download_papers(limit=5)
    
    extractor = TextExtractor()
    processed_papers = extractor.process_papers(papers)
    
    print(f"Processed {len(processed_papers)} papers")
