"""
Prompt templates for different query types.
"""
from typing import List, Dict, Any, Optional
from string import Template

# System prompts
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer science. 
Your task is to provide accurate, helpful, and scientifically grounded responses based on the research papers provided.
Always cite your sources by referring to the paper titles and authors when providing information.
If the provided context doesn't contain enough information to answer the question, acknowledge the limitations and suggest what additional information might be needed.
"""

# RAG prompts
RAG_PROMPT_TEMPLATE = Template("""
I'll provide you with information from relevant research papers to help answer the following question:

$query

Here are the most relevant excerpts from research papers:

$context

Based on these research papers, please provide a comprehensive answer to the question. 
Cite specific papers when referencing information from them.
If the information provided is insufficient to fully answer the question, please acknowledge this and explain what additional information would be needed.
""")

# Summarization prompts
SUMMARIZATION_PROMPT_TEMPLATE = Template("""
I need a summary of the following research paper:

Title: $title
Authors: $authors
Content:
$content

Please provide:
1. A concise abstract (2-3 sentences)
2. The key contributions and findings (3-5 bullet points)
3. The methodology used
4. The main results and their implications
5. Limitations mentioned in the paper

Your summary should be comprehensive yet concise, focusing on the most important aspects of the paper.
""")

# Concept explanation prompts
CONCEPT_EXPLANATION_PROMPT_TEMPLATE = Template("""
I need an explanation of the following concept from computer science research:

$concept

Here are relevant excerpts from research papers that discuss this concept:

$context

Please provide:
1. A clear definition of the concept
2. The historical development and key papers that introduced or refined it
3. How it works (with simple examples if possible)
4. Its importance and applications in the field
5. Current research directions or open questions related to this concept

Your explanation should be accessible to someone with a basic understanding of computer science but may not be familiar with this specific concept.
""")

# Chain-of-thought prompts
CHAIN_OF_THOUGHT_PROMPT_TEMPLATE = Template("""
I need to solve a complex research question that requires step-by-step reasoning:

$query

Here are relevant excerpts from research papers that might help:

$context

Please approach this question using chain-of-thought reasoning:
1. First, break down the question into smaller sub-problems
2. For each sub-problem:
   a. Identify the relevant information from the provided research papers
   b. Reason through the sub-problem step by step
   c. Reach a conclusion for the sub-problem
3. Combine the solutions to the sub-problems to answer the original question
4. Reflect on your answer, considering any limitations or assumptions

Make your reasoning explicit at each step, and cite specific papers when using information from them.
""")

# Comparison prompts
COMPARISON_PROMPT_TEMPLATE = Template("""
I need a comparison between the following concepts/methods/models in computer science research:

$items_to_compare

Here are relevant excerpts from research papers that discuss these items:

$context

Please provide a comprehensive comparison that includes:
1. Brief definitions of each item
2. Key similarities between them
3. Important differences and distinguishing features
4. Strengths and weaknesses of each
5. Typical use cases or applications
6. A summary table that highlights the main points of comparison

Base your comparison on the information provided in the research papers, and cite specific papers when referencing information from them.
""")

# Literature review prompts
LITERATURE_REVIEW_PROMPT_TEMPLATE = Template("""
I need a brief literature review on the following topic in computer science research:

$topic

Here are relevant excerpts from research papers on this topic:

$context

Please provide a structured literature review that includes:
1. An overview of the topic and its importance
2. The historical development of research in this area
3. Key papers and their contributions
4. Major themes or approaches in the research
5. Current state of the art
6. Open questions and future research directions

Organize the review chronologically or thematically as appropriate, and cite specific papers when discussing their contributions.
""")

# Implementation guidance prompts
IMPLEMENTATION_GUIDANCE_PROMPT_TEMPLATE = Template("""
I need guidance on implementing the following algorithm/method/model from computer science research:

$implementation_target

Here are relevant excerpts from research papers that describe this implementation:

$context

Please provide:
1. A clear explanation of how the implementation works
2. The key components or steps involved
3. Any important parameters or configurations
4. Potential challenges or pitfalls to be aware of
5. Pseudocode or high-level implementation steps
6. Evaluation methods to verify correct implementation

Base your guidance on the information provided in the research papers, and cite specific papers when referencing information from them.
""")

def get_rag_prompt(query: str, context: List[Dict[str, Any]]) -> str:
    """
    Generate a RAG prompt.
    
    Args:
        query: User query
        context: List of context documents
    
    Returns:
        Formatted prompt
    """
    # Format context
    formatted_context = ""
    
    for i, doc in enumerate(context):
        title = doc.get("title", "Untitled")
        authors = ", ".join(doc.get("authors", ["Unknown"]))
        content = doc.get("content", "")
        
        formatted_context += f"[{i+1}] Paper: \"{title}\" by {authors}\n"
        formatted_context += f"Excerpt: {content}\n\n"
    
    # Generate prompt
    return RAG_PROMPT_TEMPLATE.substitute(
        query=query,
        context=formatted_context
    )

def get_summarization_prompt(title: str, authors: List[str], content: str) -> str:
    """
    Generate a summarization prompt.
    
    Args:
        title: Paper title
        authors: Paper authors
        content: Paper content
    
    Returns:
        Formatted prompt
    """
    return SUMMARIZATION_PROMPT_TEMPLATE.substitute(
        title=title,
        authors=", ".join(authors),
        content=content
    )

def get_concept_explanation_prompt(concept: str, context: List[Dict[str, Any]]) -> str:
    """
    Generate a concept explanation prompt.
    
    Args:
        concept: Concept to explain
        context: List of context documents
    
    Returns:
        Formatted prompt
    """
    # Format context
    formatted_context = ""
    
    for i, doc in enumerate(context):
        title = doc.get("title", "Untitled")
        authors = ", ".join(doc.get("authors", ["Unknown"]))
        content = doc.get("content", "")
        
        formatted_context += f"[{i+1}] Paper: \"{title}\" by {authors}\n"
        formatted_context += f"Excerpt: {content}\n\n"
    
    # Generate prompt
    return CONCEPT_EXPLANATION_PROMPT_TEMPLATE.substitute(
        concept=concept,
        context=formatted_context
    )

def get_chain_of_thought_prompt(query: str, context: List[Dict[str, Any]]) -> str:
    """
    Generate a chain-of-thought prompt.
    
    Args:
        query: User query
        context: List of context documents
    
    Returns:
        Formatted prompt
    """
    # Format context
    formatted_context = ""
    
    for i, doc in enumerate(context):
        title = doc.get("title", "Untitled")
        authors = ", ".join(doc.get("authors", ["Unknown"]))
        content = doc.get("content", "")
        
        formatted_context += f"[{i+1}] Paper: \"{title}\" by {authors}\n"
        formatted_context += f"Excerpt: {content}\n\n"
    
    # Generate prompt
    return CHAIN_OF_THOUGHT_PROMPT_TEMPLATE.substitute(
        query=query,
        context=formatted_context
    )
