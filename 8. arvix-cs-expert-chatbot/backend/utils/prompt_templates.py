from typing import List, Dict, Any

def create_system_message() -> str:
    """
    Create the system message for the LLM

    Returns:
        System message string
    """
    return """You are an expert Computer Science research assistant with deep knowledge of arXiv papers and academic research.
Your role is to help users understand complex CS concepts, explain research papers, and provide accurate information based on scientific literature.

Guidelines:
1. Provide accurate, factual information based on the research papers provided.
2. Explain complex concepts clearly, breaking them down into understandable parts.
3. When discussing papers, cite them properly and summarize their key contributions.
4. If you're uncertain about something, acknowledge the limitations of your knowledge.
5. Maintain a helpful, educational tone throughout the conversation.
6. When appropriate, suggest related papers or concepts that might interest the user.

You have access to information from arXiv papers to support your responses. Use this information to provide detailed, accurate answers.
"""

def create_chat_prompt(query: str, papers: List[Dict[str, Any]],
                      paper_contents: List[Dict[str, Any]],
                      conversation_history: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for the LLM based on the user query and retrieved papers

    Args:
        query: The user's query
        papers: List of paper metadata
        paper_contents: List of paper content dictionaries
        conversation_history: Previous conversation history

    Returns:
        Formatted prompt string
    """
    # Format conversation history
    history_text = ""
    if conversation_history:
        for i, message in enumerate(conversation_history[-3:]):  # Include last 3 exchanges
            history_text += f"User: {message['user']}\n"
            history_text += f"Assistant: {message['assistant']}\n\n"

    # Format paper information
    papers_text = ""
    for i, paper in enumerate(papers[:5]):  # Include top 5 papers
        papers_text += f"[{i+1}] {paper['title']}\n"
        papers_text += f"Authors: {', '.join(paper['authors'])}\n"
        papers_text += f"Categories: {', '.join(paper['categories'])}\n"
        papers_text += f"Abstract: {paper['abstract'][:300]}...\n\n"

    # Format paper contents (more detailed information)
    content_text = ""
    for i, paper_data in enumerate(paper_contents):
        paper = paper_data["paper"]
        content = paper_data["content"]

        content_text += f"--- PAPER {i+1}: {paper['title']} ---\n"

        # Include abstract
        if "abstract" in content and content["abstract"]:
            content_text += f"ABSTRACT:\n{content['abstract'][:500]}...\n\n"

        # Include introduction if available
        if "introduction" in content and content["introduction"]:
            content_text += f"INTRODUCTION:\n{content['introduction'][:500]}...\n\n"

        # Include methodology if available
        if "methodology" in content and content["methodology"]:
            content_text += f"METHODOLOGY:\n{content['methodology'][:500]}...\n\n"

        # Include conclusion if available
        if "conclusion" in content and content["conclusion"]:
            content_text += f"CONCLUSION:\n{content['conclusion'][:300]}...\n\n"

    # Construct the final prompt
    prompt = f"""
{'CONVERSATION HISTORY:' + history_text if history_text else ''}

RELEVANT PAPERS:
{papers_text}

PAPER CONTENTS:
{content_text}

USER QUERY: {query}

Please provide a comprehensive response to the user's query based on the information from the papers.
Cite specific papers when referencing their findings or methods.
If the papers don't contain enough information to fully answer the query, acknowledge this and provide the best response possible based on the available information.
"""

    return prompt

def create_paper_summary_prompt(paper: Dict[str, Any], content: Dict[str, Any]) -> str:
    """
    Create a prompt for generating a paper summary

    Args:
        paper: Paper metadata
        content: Paper content

    Returns:
        Formatted prompt string
    """
    prompt = f"""
PAPER TITLE: {paper['title']}
AUTHORS: {', '.join(paper['authors'])}
CATEGORIES: {', '.join(paper['categories'])}

ABSTRACT:
{content.get('abstract', 'No abstract available')}

"""

    if 'introduction' in content and content['introduction']:
        prompt += f"""
INTRODUCTION:
{content['introduction'][:1000]}...
"""

    if 'methodology' in content and content['methodology']:
        prompt += f"""
METHODOLOGY:
{content['methodology'][:1000]}...
"""

    if 'results' in content and content['results']:
        prompt += f"""
RESULTS:
{content['results'][:1000]}...
"""

    if 'conclusion' in content and content['conclusion']:
        prompt += f"""
CONCLUSION:
{content['conclusion'][:500]}...
"""

    prompt += """
Please provide a comprehensive summary of this research paper, including:
1. The main research problem or question addressed
2. The key methodology or approach used
3. The most significant findings or results
4. The main contributions to the field
5. Limitations or future work mentioned

Format the summary in a clear, structured way that would help someone quickly understand the paper's importance and findings.
"""

    return prompt

def create_concept_explanation_prompt(concept: str, papers: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for explaining a CS concept

    Args:
        concept: The concept to explain
        papers: List of relevant papers

    Returns:
        Formatted prompt string
    """
    # Format paper information
    papers_text = ""
    for i, paper in enumerate(papers[:3]):  # Include top 3 papers
        papers_text += f"[{i+1}] {paper['title']}\n"
        papers_text += f"Abstract: {paper['abstract'][:200]}...\n\n"

    prompt = f"""
CONCEPT TO EXPLAIN: {concept}

RELEVANT PAPERS:
{papers_text}

Please provide a comprehensive explanation of the concept "{concept}" in computer science. Your explanation should:
1. Define the concept clearly and concisely
2. Explain its importance and applications in computer science
3. Describe how it relates to other fundamental concepts
4. Provide examples to illustrate the concept
5. Reference relevant information from the papers provided

Your explanation should be accessible to someone with a basic understanding of computer science but should also include enough depth for more advanced readers.
"""

    return prompt