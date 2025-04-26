import os
import aiohttp
import asyncio
import fitz  # PyMuPDF
import re
import json
from typing import Dict, Any, Optional, List

class PaperProcessor:
    def __init__(self, cache_dir: str = "../data/paper_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    async def extract_content(self, paper_id: str, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract content from a paper PDF

        Args:
            paper_id: The arXiv ID of the paper
            sections: Optional list of sections to extract (e.g., ["abstract", "introduction"])

        Returns:
            Dictionary with extracted content
        """
        # Check cache first
        cache_path = os.path.join(self.cache_dir, f"{paper_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Download PDF if not in cache
        pdf_path = await self._download_paper(paper_id)
        if not pdf_path:
            return {"error": "Failed to download paper"}

        # Extract text using PyMuPDF
        content = await self._extract_text_from_pdf(pdf_path, sections)

        # Cache the results
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(content, f)

        return content

    async def _download_paper(self, paper_id: str) -> Optional[str]:
        """
        Download a paper PDF from arXiv

        Args:
            paper_id: The arXiv ID of the paper

        Returns:
            Path to downloaded PDF or None if download failed
        """
        pdf_path = os.path.join(self.cache_dir, f"{paper_id}.pdf")

        # Check if already downloaded
        if os.path.exists(pdf_path):
            return pdf_path

        # Download from arXiv
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(pdf_path, 'wb') as f:
                            f.write(await response.read())
                        return pdf_path
                    else:
                        return None
        except Exception as e:
            print(f"Error downloading paper {paper_id}: {str(e)}")
            return None

    async def _extract_text_from_pdf(self, pdf_path: str, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract text from a PDF file

        Args:
            pdf_path: Path to the PDF file
            sections: Optional list of sections to extract

        Returns:
            Dictionary with extracted content
        """
        # Use asyncio to run the potentially blocking operation in a thread pool
        def extract():
            try:
                doc = fitz.open(pdf_path)

                # Extract full text
                full_text = ""
                for page in doc:
                    full_text += page.get_text()

                # Process and segment the text
                content = {"full_text": full_text}

                # Extract sections
                section_patterns = {
                    "abstract": r"(?i)abstract\s*(.*?)(?=\n\s*(?:introduction|keywords|related work|\d\.|conclusion))",
                    "introduction": r"(?i)(?:introduction|1\s+introduction)\s*(.*?)(?=\n\s*(?:\d\.|related work|background|preliminaries))",
                    "methodology": r"(?i)(?:methodology|method|approach|3\s+method|3\s+methodology)\s*(.*?)(?=\n\s*(?:\d\.|experiments|results|evaluation))",
                    "results": r"(?i)(?:results|experiments|evaluation|4\s+results|4\s+experiments|4\s+evaluation)\s*(.*?)(?=\n\s*(?:\d\.|discussion|conclusion|limitations))",
                    "conclusion": r"(?i)(?:conclusion|conclusions|5\s+conclusion|5\s+conclusions)\s*(.*?)(?=\n\s*(?:references|acknowledgements|appendix|$))"
                }

                # If specific sections requested, filter the patterns
                if sections:
                    section_patterns = {k: v for k, v in section_patterns.items() if k in sections}

                # Extract each section
                for section_name, pattern in section_patterns.items():
                    match = re.search(pattern, full_text, re.DOTALL)
                    if match:
                        content[section_name] = match.group(1).strip()
                    else:
                        content[section_name] = ""

                return content
            except Exception as e:
                return {"error": f"Error extracting text: {str(e)}"}

        return await asyncio.to_thread(extract)

    def extract_key_points(self, content: Dict[str, Any]) -> List[str]:
        """
        Extract key points from paper content

        Args:
            content: Dictionary with extracted content

        Returns:
            List of key points
        """
        key_points = []

        # Extract from abstract
        if "abstract" in content and content["abstract"]:
            # Simple heuristic: sentences with keywords like "propose", "novel", "approach"
            abstract_sentences = re.split(r'(?<=[.!?])\s+', content["abstract"])
            for sentence in abstract_sentences:
                if any(keyword in sentence.lower() for keyword in ["propose", "novel", "approach", "method", "algorithm", "framework", "contribution"]):
                    key_points.append(sentence.strip())

        # Extract from conclusion
        if "conclusion" in content and content["conclusion"]:
            conclusion_sentences = re.split(r'(?<=[.!?])\s+', content["conclusion"])
            for sentence in conclusion_sentences:
                if any(keyword in sentence.lower() for keyword in ["result", "show", "demonstrate", "achieve", "outperform", "improve", "future"]):
                    key_points.append(sentence.strip())

        # Limit to top 5 key points
        return key_points[:5] if key_points else ["No key points extracted"]

    def extract_citations(self, content: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract citations from paper content

        Args:
            content: Dictionary with extracted content

        Returns:
            List of citation dictionaries
        """
        citations = []

        # Simple regex-based citation extraction
        if "full_text" in content:
            # Look for citation patterns like [1], [2], etc.
            citation_matches = re.finditer(r'\[(\d+)\](.*?)(?=\[\d+\]|\n\n|$)', content["full_text"])
            for match in citation_matches:
                citation_num = match.group(1)
                citation_text = match.group(2).strip()
                if citation_text:
                    citations.append({
                        "number": citation_num,
                        "text": citation_text
                    })

        return citations