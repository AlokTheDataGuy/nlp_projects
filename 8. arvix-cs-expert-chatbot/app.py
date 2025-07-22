"""
CS Expert AI Assistant - Main Streamlit Application
Advanced Computer Science Research Assistant with Foundation LLM + RAG
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import os
import warnings
import time
import json
import torch
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Suppress PyTorch-Streamlit warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Fix PyTorch path issue
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
except (AttributeError, TypeError):
    pass

# Set environment variable to disable Streamlit file watcher for torch
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Import custom modules
from data_processor import EnhancedArxivProcessor
from llm_engine import FoundationLLMEngine, QueryType
from nlp_pipeline import AdvancedNLPPipeline
from concept_visualizer import ConceptVisualizer
from rag_system import RAGSystem


class CSExpertApp:
    """Main application class for CS Expert AI Assistant"""
    
    def __init__(self):
        self.config = None
        self.processor = None
        self.nlp_pipeline = None
        self.llm_engine = None
        self.visualizer = None
        
        # Configure Streamlit page
        self._configure_page()
        self._load_custom_styles()
    
    def _configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="CS Expert AI Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'hhttps://github.com/AlokTheDataGuy/arXiv-cs-expert-chatbot',
                'Report a bug': 'https://github.com/AlokTheDataGuy/arXiv-cs-expert-chatbot/issues',
                'About': "CS Expert AI Assistant - Powered by Llama 3 + RAG"
            }
        )
    
    def _load_custom_styles(self):
        """Load custom CSS styles"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                font-weight: bold;
                background: linear-gradient(45deg, #1e3a8a, #7c3aed);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 2rem;
                margin-top: 0rem;
                padding: 0rem;
            }
            .chat-message {
                padding: 1rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .user-message {
                background: linear-gradient(135deg, #e0f2fe, #b3e5fc);
                border-left: 4px solid #0288d1;
                color: black;
            }
            .assistant-message {
                background: linear-gradient(135deg, #f3e5f5, #e1bee7);
                border-left: 4px solid #7b1fa2;
            }
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .source-paper {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            .concept-tag {
                background: #ddd6fe;
                color: #5b21b6;
                padding: 0.2rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                margin: 0.2rem;
                display: inline-block;
            }
            .error-container {
                background: #fee2e2;
                border: 1px solid #fca5a5;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            .success-container {
                background: #d1fae5;
                border: 1px solid #6ee7b7;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @st.cache_resource
    def load_configuration(_self):
        """Load application configuration"""
        config_path = "config.yaml"
        
        if not os.path.exists(config_path):
            st.error("❌ Configuration file not found. Please ensure config.yaml exists.")
            st.info("📝 Create a config.yaml file with the required settings.")
            st.stop()
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            st.error(f"❌ Error loading configuration: {str(e)}")
            st.stop()
    
    @st.cache_resource
    def initialize_system(_self, _config):
        """Initialize all system components with progress tracking"""
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize data processor
                status_text.text("🔧 Initializing data processor...")
                progress_bar.progress(10)
                processor = EnhancedArxivProcessor()
                
                # Step 2: Load research papers
                status_text.text("📚 Loading research papers...")
                progress_bar.progress(30)
                processor.fetch_arxiv_papers()
                
                # Step 3: Create semantic embeddings
                status_text.text("🧠 Creating semantic embeddings...")
                progress_bar.progress(50)
                processor.create_enhanced_embeddings()
                
                # Step 4: Initialize NLP pipeline
                status_text.text("🔤 Setting up NLP pipeline...")
                progress_bar.progress(70)
                nlp_pipeline = AdvancedNLPPipeline()
                
                # Step 5: Initialize LLM engine
                status_text.text("🤖 Loading AI language model...")
                progress_bar.progress(85)
                llm_engine = FoundationLLMEngine()
                
                # Step 6: Setup RAG system
                status_text.text("🗄️ Setting up knowledge retrieval...")
                progress_bar.progress(90)
                _self._setup_rag_system(llm_engine, processor)
                
                # Step 7: Initialize visualizer
                visualizer = ConceptVisualizer()
                
                # Complete initialization
                progress_bar.progress(100)
                status_text.text("✅ System ready!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                return processor, nlp_pipeline, llm_engine, visualizer
                
            except Exception as e:
                st.error(f"❌ System initialization failed: {str(e)}")
                st.error("Please check your configuration and try again.")
                st.stop()
    
    def _setup_rag_system(self, llm_engine, processor):
        """Setup RAG system with vector database"""
        try:
            if llm_engine.rag_system.collection:
                count = llm_engine.rag_system.collection.count()
                if count == 0:
                    llm_engine.rag_system.add_papers_to_vector_db(
                        processor.papers, processor.embeddings
                    )
        except Exception as e:
            st.warning(f"⚠️ RAG system setup warning: {str(e)}")
            # Attempt to add papers anyway
            try:
                llm_engine.rag_system.add_papers_to_vector_db(
                    processor.papers, processor.embeddings
                )
            except:
                st.error("❌ Failed to setup vector database")
    
    def display_system_status(self):
        """Display comprehensive system status in sidebar"""
        with st.sidebar:
            st.header("🔧 System Status")
            
            # Data status
            st.subheader("📊 Dataset")
            if self.processor and self.processor.papers:
                st.success(f"✅ {len(self.processor.papers)} papers loaded")
                categories = len(set(p['primary_category'] for p in self.processor.papers))
                st.info(f"📂 {categories} categories")
                
                # Show data range
                df = self.processor.get_papers_dataframe()
                date_range = f"{df['year'].min()} - {df['year'].max()}"
                st.info(f"📅 {date_range}")
            else:
                st.error("❌ No data loaded")
            
            # AI Model status
            st.subheader("🤖 AI Model")
            if self.llm_engine:
                model_info = self.llm_engine.get_model_info()
                if model_info.get('available'):
                    st.success(f"✅ {model_info['model_name']}")
                else:
                    st.error("❌ Model unavailable")
                    st.info("💡 Make sure Ollama is running")
            else:
                st.error("❌ LLM engine not initialized")
            
            # Knowledge Base status
            st.subheader("🧠 Knowledge Base")
            if self.llm_engine and self.llm_engine.rag_system:
                rag_stats = self.llm_engine.rag_system.get_paper_stats()
                if rag_stats:
                    st.success(f"✅ {rag_stats['total_papers']} papers indexed")
                    st.info(f"📏 {rag_stats['embedding_dimension']}D embeddings")
                else:
                    st.warning("⚠️ Vector DB not available")
            else:
                st.error("❌ RAG system not available")
            
            # Conversation status
            st.subheader("💬 Conversation")
            if self.llm_engine and hasattr(self.llm_engine, 'conversation_history'):
                conv_count = len(self.llm_engine.conversation_history)
                st.info(f"💭 {conv_count} conversation turns")
                
                if conv_count > 0 and st.button("🗑️ Clear History", type="secondary"):
                    self.llm_engine.clear_conversation_history()
                    st.success("History cleared!")
                    st.rerun()
            else:
                st.info("💭 No conversation history")
    
    def create_chat_interface(self):
        """Create the main chat interface"""
        st.header("💬 CS Expert AI Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            welcome_msg = self._get_welcome_message()
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        # Display chat history
        self._display_chat_history()
        
        # Quick start questions
        self._display_quick_questions()
        
        # Chat input
        self._handle_chat_input()
    
    def _get_welcome_message(self) -> str:
        """Get welcome message for new users"""
        return """
        🎓 **Welcome to the CS Expert AI Assistant!**
        
        I'm powered by advanced language models and have access to a comprehensive database of computer science research papers. I can help you with:
        
        - **📚 Explaining fundamental CS concepts** with clear, educational explanations
        - **🔬 Discussing cutting-edge research** and recent developments  
        - **📝 Summarizing research papers** based on your specific interests
        - **💡 Providing detailed concept explanations** with practical examples
        - **❓ Answering follow-up questions** to deepen your understanding
                
        Ask me anything about computer science, machine learning, AI, algorithms, or any research topic!
        """
    
    def _display_chat_history(self):
        """Display chat message history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message">{message["content"]}</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(message["content"])
    
    def _display_quick_questions(self):
        """Display quick start question buttons"""
        st.markdown("### 🚀 Quick Start Questions")
        
        quick_questions = [
            "What is machine learning?",
            "Explain transformer architecture",
            "Latest developments in AI",
            "Data structures vs algorithms"
        ]
        
        cols = st.columns(4)
        for i, (col, question) in enumerate(zip(cols, quick_questions)):
            with col:
                if st.button(question, key=f"quick_{i}"):
                    self._process_user_input(question)
    
    def _handle_chat_input(self):
        """Handle chat input from user"""
        if prompt := st.chat_input("Ask me anything about computer science..."):
            self._process_user_input(prompt)
    
    def _process_user_input(self, prompt: str):
        """Process user input and generate response"""
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(
                f'<div class="chat-message user-message">{prompt}</div>', 
                unsafe_allow_html=True
            )
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            self._generate_assistant_response(prompt)
        
        # Rerun to update the interface
        st.rerun()
    
    def _generate_assistant_response(self, prompt: str):
        """Generate comprehensive assistant response"""
        with st.spinner("🧠 Analyzing your question and consulting research database..."):
            try:
                # Generate response using LLM engine
                llm_response = self.llm_engine.generate_response(prompt)
                
                # Display main response
                st.markdown(llm_response.content)
                
                # Display metadata
                self._display_response_metadata(llm_response)
                
                # Show relevant papers
                self._display_relevant_papers(llm_response)
                
                # Show follow-up suggestions
                self._display_followup_suggestions(llm_response)
                
                # Add to conversation history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": llm_response.content
                })
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
    
    def _display_response_metadata(self, llm_response):
        """Display response confidence and query type"""
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_color = (
                "🟢" if llm_response.confidence > 0.8 else 
                "🟡" if llm_response.confidence > 0.6 else "🔴"
            )
            st.caption(f"Confidence: {confidence_color} {llm_response.confidence:.1%}")
        
        with col2:
            st.caption(f"Query type: {llm_response.query_type.value}")
    
    def _display_relevant_papers(self, llm_response):
        """Display relevant research papers"""
        if not llm_response.sources:
            return
        
        with st.expander(
            f"📚 {len(llm_response.sources)} Relevant Research Papers", 
            expanded=False
        ):
            for i, paper in enumerate(llm_response.sources[:5], 1):
                self._display_paper_card(paper, i)
    
    def _display_paper_card(self, paper: Dict, index: int):
        """Display individual paper card"""
        with st.container():
            st.markdown('<div class="source-paper">', unsafe_allow_html=True)
            st.markdown(f"**{index}. {paper['title']}**")
            
            # Authors and metadata
            authors = paper.get('authors', [])[:3]
            authors_str = ", ".join(authors)
            if len(paper.get('authors', [])) > 3:
                authors_str += " et al."
            
            pub_date = paper.get('published', 'Unknown')[:10]
            st.markdown(f"*{authors_str}* • Published: {pub_date}")
            
            # Categories as tags
            categories = paper.get('categories', [])[:3]
            if categories:
                category_tags = " ".join([
                    f'<span class="concept-tag">{cat}</span>' 
                    for cat in categories
                ])
                st.markdown(category_tags, unsafe_allow_html=True)
            
            # Abstract excerpt
            abstract = paper.get('document', '')
            if 'Abstract:' in abstract:
                abstract = abstract.split('Abstract:', 1)[1].strip()
            
            summary_text = abstract[:300] + ('...' if len(abstract) > 300 else '')
            st.markdown(f"**Summary:** {summary_text}")
            
            # Similarity score
            similarity = paper.get('similarity', 0)
            st.progress(similarity, text=f"Relevance: {similarity:.1%}")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🔍 Deep Analysis", key=f"analysis_{index}_{len(st.session_state.messages)}"):
                    self._perform_paper_analysis(paper, abstract)
            
            with col2:
                pdf_url = paper.get('pdf_url')
                if pdf_url:
                    st.link_button("📄 View PDF", pdf_url)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _perform_paper_analysis(self, paper: Dict, abstract: str):
        """Perform deep analysis of a paper"""
        with st.spinner("Performing advanced analysis..."):
            try:
                full_text = f"{paper['title']} {abstract}"
                
                # Extract keywords and entities
                keywords = self.nlp_pipeline.extract_advanced_keywords(full_text, num_keywords=8)
                entities = self.nlp_pipeline.extract_enhanced_entities(full_text)
                
                # Display analysis results
                st.markdown("**🔑 Key Concepts:**")
                if keywords:
                    concept_tags = " ".join([
                        f'<span class="concept-tag">{kw}</span>' 
                        for kw in keywords
                    ])
                    st.markdown(concept_tags, unsafe_allow_html=True)
                
                if entities:
                    st.markdown("**🏷️ Technical Entities:**")
                    entity_info = [
                        f"• {ent['text']} ({ent['label']})" 
                        for ent in entities[:5]
                    ]
                    st.markdown("\n".join(entity_info))
                
                # Complexity analysis
                complexity = self.nlp_pipeline.analyze_text_complexity(abstract)
                st.markdown(
                    f"**📊 Text Analysis:** {complexity['reading_level']} complexity, "
                    f"{complexity['technical_density']:.1%} technical density"
                )
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    def _display_followup_suggestions(self, llm_response):
        """Display follow-up question suggestions"""
        if not llm_response.follow_up_suggestions:
            return
        
        st.markdown("### 💡 Suggested Follow-up Questions:")
        for j, suggestion in enumerate(llm_response.follow_up_suggestions[:3]):
            if st.button(
                f"💭 {suggestion}", 
                key=f"followup_{j}_{len(st.session_state.messages)}"
            ):
                self._process_user_input(suggestion)
    
    def create_paper_search_interface(self):
        """Create simplified paper search interface"""
        st.header("🔍Paper Search")

        # Search controls
        search_query = st.text_input(
            "Search research papers:",
            placeholder="e.g., transformer, machine learning, neural networks"
        )

        # Process search
        if search_query:
            self._process_simple_search_query(search_query)

    def _process_simple_search_query(self, search_query: str):
        """Process search query with simplified logic"""
        with st.spinner("🔍 Searching through research papers..."):
            try:
                # Check if RAG system is available
                if not self.llm_engine or not self.llm_engine.rag_system:
                    st.error("❌ Search system not available. Please check system initialization.")
                    return

                # Check if vector database is available
                if not self.llm_engine.rag_system.collection:
                    st.error("❌ Vector database not available. Please check ChromaDB setup.")
                    return

                # Perform search - get more results
                results = self.llm_engine.rag_system.retrieve_relevant_papers(
                    search_query, top_k=50
                )

                # Also do simple keyword search as fallback
                keyword_results = self._simple_keyword_search(search_query)

                # Combine results
                all_results = results + keyword_results

                # Remove duplicates based on title
                seen_titles = set()
                unique_results = []
                for result in all_results:
                    title = result.get('title', '').lower()
                    if title not in seen_titles:
                        seen_titles.add(title)
                        unique_results.append(result)

                # Display results
                st.markdown(f"### 📊 Found {len(unique_results)} relevant papers")

                if unique_results:
                    self._display_simple_search_results(unique_results[:20])  # Show top 20
                else:
                    st.info("No papers found. Try different keywords like 'neural', 'learning', 'algorithm', etc.")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    def _simple_keyword_search(self, search_query: str):
        """Simple keyword-based search through papers"""
        results = []
        search_terms = search_query.lower().split()

        try:
            for paper in self.processor.papers:
                # Search in title and abstract
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                text_to_search = f"{title} {abstract}"

                # Count matching terms
                matches = sum(1 for term in search_terms if term in text_to_search)

                if matches > 0:
                    # Calculate simple similarity based on term matches
                    similarity = matches / len(search_terms)

                    result = {
                        'id': paper.get('id', ''),
                        'title': paper.get('title', ''),
                        'authors': paper.get('authors', []),
                        'abstract': paper.get('abstract', ''),
                        'categories': paper.get('categories', []),
                        'published': paper.get('published', ''),
                        'primary_category': paper.get('primary_category', ''),
                        'similarity': similarity,
                        'document': f"{paper.get('title', '')} {paper.get('abstract', '')}",
                        'pdf_url': paper.get('pdf_url', '')
                    }
                    results.append(result)

            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:10]  # Return top 10 keyword matches

        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    def _display_simple_search_results(self, results):
        """Display search results in a simple format"""
        for i, result in enumerate(results, 1):
            with st.expander(
                f"{i}. {result['title']}",
                expanded=False
            ):
                # Authors
                authors = result.get('authors', [])
                if isinstance(authors, list):
                    authors_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_str += " et al."
                else:
                    authors_str = str(authors)

                st.write(f"**Authors:** {authors_str}")
                st.write(f"**Published:** {result.get('published', 'Unknown')[:10]}")

                # Categories
                categories = result.get('categories', [])
                if categories:
                    st.write(f"**Categories:** {', '.join(categories[:3])}")

                # Abstract
                abstract = result.get('abstract', result.get('document', ''))
                if abstract:
                    # Clean up abstract
                    if 'Abstract:' in abstract:
                        abstract = abstract.split('Abstract:', 1)[1].strip()

                    # Truncate if too long
                    if len(abstract) > 500:
                        abstract = abstract[:500] + "..."

                    st.write(f"**Abstract:** {abstract}")

                # PDF link if available
                pdf_url = result.get('pdf_url')
                if pdf_url:
                    st.link_button("📄 View PDF", pdf_url)
    
    def _create_search_controls(self) -> Tuple[str, str]:
        """Create search input controls"""
        col1, col2 = st.columns(2)
        
        with col1:
            search_query = st.text_input(
                "Search research papers:",
                placeholder="e.g., transformer attention mechanism, deep learning optimization"
            )
        
        with col2:
            search_type = st.selectbox(
                "Search Type:",
                ["Semantic Search", "Keyword Search", "Author Search", "Category Search"]
            )
        
        return search_query, search_type
    
    def _create_search_filters(self) -> Tuple[Tuple[int, int], List[str], float]:
        """Create advanced search filters"""
        with st.expander("🔧 Advanced Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                year_filter = st.slider("Publication Year", 2018, 2024, (2020, 2024))
            
            with col2:
                category_options = list(set(
                    p['primary_category'] for p in self.processor.papers
                ))
                category_filter = st.multiselect("Categories", category_options)
            
            with col3:
                min_similarity = st.slider("Minimum Relevance", 0.0, 1.0, 0.3, 0.05)
        
        return year_filter, category_filter, min_similarity
    
    def _process_search_query(self, search_query: str, search_type: str,
                            year_filter: Tuple[int, int], category_filter: List[str],
                            min_similarity: float):
        """Process search query and display results"""
        with st.spinner("🔍 Searching through research papers..."):
            try:
                # Check if RAG system is available
                if not self.llm_engine or not self.llm_engine.rag_system:
                    st.error("❌ Search system not available. Please check system initialization.")
                    return

                # Check if vector database is available
                if not self.llm_engine.rag_system.collection:
                    st.error("❌ Vector database not available. Please check ChromaDB setup.")
                    return

                # Perform search
                results = self.llm_engine.rag_system.retrieve_relevant_papers(
                    search_query, top_k=20
                )

                # Debug information
                st.info(f"🔍 Raw search returned {len(results)} results")

                # Apply filters
                filtered_results = self._apply_search_filters(
                    results, year_filter, category_filter, min_similarity
                )

                # Display results
                st.markdown(f"### 📊 Found {len(filtered_results)} relevant papers")
                self._display_search_results(filtered_results)

            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                st.error(f"Error details: {type(e).__name__}")
                # Show more detailed error for debugging
                import traceback
                st.code(traceback.format_exc())
    
    def _apply_search_filters(self, results: List[Dict], year_filter: Tuple[int, int], 
                            category_filter: List[str], min_similarity: float) -> List[Dict]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            # Year filter
            try:
                pub_year = int(result.get('published', '2020')[:4])
                if not (year_filter[0] <= pub_year <= year_filter[1]):
                    continue
            except (ValueError, TypeError):
                continue
            
            # Category filter
            if category_filter and result.get('primary_category') not in category_filter:
                continue
            
            # Similarity filter
            if result.get('similarity', 0) < min_similarity:
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _display_search_results(self, results: List[Dict]):
        """Display search results with enhanced information"""
        if not results:
            st.info("No papers found matching your criteria. Try adjusting the filters.")
            return
        
        for i, result in enumerate(results[:15], 1):
            with st.expander(
                f"{i}. {result['title']} (Relevance: {result.get('similarity', 0):.2%})", 
                expanded=False
            ):
                self._display_search_result_details(result, i)
    
    def _display_search_result_details(self, result: Dict, index: int):
        """Display detailed information for a search result"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Paper details
            authors_list = result['authors'][:5]
            st.write(f"**Authors:** {', '.join(authors_list)}")
            st.write(f"**Published:** {result['published'][:10]}")
            st.write(f"**Categories:** {', '.join(result['categories'])}")
            
            # Abstract
            abstract = result['document']
            if 'Abstract:' in abstract:
                abstract = abstract.split('Abstract:', 1)[1].strip()
            st.write(f"**Abstract:** {abstract}")
        
        with col2:
            # Action buttons
            st.metric("Relevance", f"{result.get('similarity', 0):.1%}")
            
            if st.button(f"📝 Summarize", key=f"summarize_{index}"):
                with st.spinner("Generating summary..."):
                    summary = self.nlp_pipeline.generate_advanced_summary(abstract)
                    st.success(f"**Summary:** {summary}")
            
            # PDF link
            pdf_url = result.get('pdf_url')
            if pdf_url:
                st.link_button("📄 View PDF", pdf_url)
    
    def create_visualization_dashboard(self):
        """Create comprehensive visualization dashboard"""
        st.header("📊 Research Analytics Dashboard")
        
        # Get data
        papers_df = self.processor.get_papers_dataframe()
        
        # Create dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", "🕸️ Concept Networks", 
            "📊 Trends", "🔍 Deep Analysis"
        ])
        
        with tab1:
            self._create_overview_tab(papers_df)
        
        with tab2:
            self._create_concept_networks_tab(papers_df)
        
        with tab3:
            self._create_trends_tab(papers_df)
        
        with tab4:
            self._create_deep_analysis_tab(papers_df)
    
    def _create_overview_tab(self, papers_df: pd.DataFrame):
        """Create overview tab content"""
        st.subheader("📋 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", len(papers_df))
        
        with col2:
            unique_authors = papers_df['authors'].explode().nunique()
            st.metric("Unique Authors", f"{unique_authors:,}")
        
        with col3:
            avg_abstract_len = papers_df['abstract_length'].mean()
            st.metric("Avg Abstract Length", f"{avg_abstract_len:.0f}")
        
        with col4:
            latest_year = papers_df['year'].max()
            st.metric("Latest Papers", latest_year)
        
        # Visualizations
        try:
            fig_dashboard = self.visualizer.create_paper_metrics_dashboard(papers_df)
            st.plotly_chart(fig_dashboard, use_container_width=True)
            
            fig_categories = self.visualizer.create_category_distribution(papers_df)
            st.plotly_chart(fig_categories, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    def _create_concept_networks_tab(self, papers_df: pd.DataFrame):
        """Create concept networks tab content"""
        st.subheader("🕸️ Concept Co-occurrence Networks")
        
        if st.button("🔄 Generate Concept Network"):
            with st.spinner("Extracting concepts and building network..."):
                try:
                    # Sample papers for performance
                    sample_size = min(200, len(papers_df))
                    sample_papers = papers_df.sample(sample_size)
                    
                    # Extract text for concept analysis
                    texts = (sample_papers['title'] + " " + sample_papers['abstract']).tolist()
                    
                    # Build concept graph
                    concept_graph = self.nlp_pipeline.extract_concept_graph(
                        texts, min_cooccurrence=3
                    )
                    
                    if concept_graph.nodes():
                        # Create network visualization
                        fig_network = self.visualizer.create_concept_network(concept_graph)
                        st.plotly_chart(fig_network, use_container_width=True)
                        
                        # Network statistics
                        self._display_network_stats(concept_graph)
                    else:
                        st.info("No significant concept relationships found. Try increasing the sample size.")
                        
                except Exception as e:
                    st.error(f"Network generation failed: {str(e)}")
    
    def _display_network_stats(self, concept_graph):
        """Display network statistics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Concepts", len(concept_graph.nodes()))
        
        with col2:
            st.metric("Connections", len(concept_graph.edges()))
        
        with col3:
            if len(concept_graph.nodes()) > 1:
                max_edges = len(concept_graph.nodes()) * (len(concept_graph.nodes()) - 1) / 2
                density = len(concept_graph.edges()) / max_edges
                st.metric("Network Density", f"{density:.3f}")
    
    def _create_trends_tab(self, papers_df: pd.DataFrame):
        """Create trends analysis tab"""
        st.subheader("📈 Research Trends Analysis")
        
        try:
            # Topic evolution over time
            fig_evolution = self.visualizer.create_topic_evolution(papers_df)
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Keyword trends
            self._create_keyword_trends_section(papers_df)
            
            # Author collaboration network
            if st.checkbox("Show Author Collaboration Network"):
                with st.spinner("Building collaboration network..."):
                    fig_collab = self.visualizer.create_author_collaboration(papers_df)
                    st.plotly_chart(fig_collab, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Trends analysis failed: {str(e)}")
    
    def _create_keyword_trends_section(self, papers_df: pd.DataFrame):
        """Create keyword trends section"""
        st.subheader("🔤 Keyword Trends")
        default_keywords = ["deep learning", "neural network", "transformer", "attention", "bert", "gpt"]
        
        keyword_input = st.text_input(
            "Enter keywords (comma-separated):", 
            ", ".join(default_keywords)
        )
        keywords = [kw.strip() for kw in keyword_input.split(",") if kw.strip()]
        
        if keywords:
            try:
                fig_trends = self.visualizer.create_keyword_trends(papers_df, keywords)
                st.plotly_chart(fig_trends, use_container_width=True)
            except Exception as e:
                st.error(f"Keyword trends failed: {str(e)}")
    
    def _create_deep_analysis_tab(self, papers_df: pd.DataFrame):
        """Create deep analysis tab"""
        st.subheader("🔍 Deep Analysis")
        
        # Embedding visualization
        if hasattr(self.processor, 'embeddings') and self.processor.embeddings is not None:
            self._create_embedding_visualization(papers_df)
        
        # Text complexity analysis
        self._create_complexity_analysis(papers_df)
    
    def _create_embedding_visualization(self, papers_df: pd.DataFrame):
        """Create embedding visualization section"""
        st.subheader("🎯 Paper Similarity Visualization")
        
        viz_method = st.selectbox("Visualization Method:", ["t-SNE", "PCA"])
        
        if st.button("🔄 Generate Embedding Visualization"):
            with st.spinner(f"Creating {viz_method} visualization..."):
                try:
                    fig_embed = self.visualizer.create_embedding_visualization(
                        self.processor.embeddings, papers_df, method=viz_method
                    )
                    st.plotly_chart(fig_embed, use_container_width=True)
                except Exception as e:
                    st.error(f"Embedding visualization failed: {str(e)}")
    
    def _create_complexity_analysis(self, papers_df: pd.DataFrame):
        """Create text complexity analysis section"""
        st.subheader("📝 Text Complexity Analysis")
        
        if st.button("📊 Analyze Abstract Complexity"):
            with st.spinner("Analyzing text complexity..."):
                try:
                    complexity_scores = []
                    
                    for _, paper in papers_df.iterrows():
                        complexity = self.nlp_pipeline.analyze_text_complexity(paper['abstract'])
                        complexity_scores.append({
                            'title': paper['title'][:50] + '...',
                            'reading_level': complexity['reading_level'],
                            'lexical_diversity': complexity['lexical_diversity'],
                            'technical_density': complexity['technical_density'],
                            'word_count': complexity['word_count']
                        })
                    
                    complexity_df = pd.DataFrame(complexity_scores)
                    
                    # Display complexity distribution
                    fig_complexity = px.scatter(
                        complexity_df,
                        x='lexical_diversity',
                        y='technical_density',
                        color='reading_level',
                        size='word_count',
                        hover_data=['title'],
                        title="Paper Complexity Analysis"
                    )
                    st.plotly_chart(fig_complexity, use_container_width=True)
                    
                    # Show complexity stats
                    st.dataframe(complexity_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Complexity analysis failed: {str(e)}")
    
    def create_system_info_tab(self):
        """Create system information tab"""
        st.subheader("⚙️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_model_configuration()
            self._display_dataset_statistics()
        
        with col2:
            self._display_rag_status()
            self._display_system_resources()
        
        # Data export options
        self._create_export_section()
    
    def _display_model_configuration(self):
        """Display AI model configuration"""
        st.markdown("**🤖 AI Model Configuration:**")
        if self.llm_engine:
            model_info = self.llm_engine.get_model_info()
            st.json(model_info)
        else:
            st.error("Model information not available")
    
    def _display_dataset_statistics(self):
        """Display dataset statistics"""
        st.markdown("**📊 Dataset Statistics:**")
        if self.processor and self.processor.papers:
            df = self.processor.get_papers_dataframe()
            dataset_stats = {
                "total_papers": len(self.processor.papers),
                "categories": len(set(p['primary_category'] for p in self.processor.papers)),
                "average_abstract_length": df['abstract_length'].mean(),
                "date_range": f"{df['year'].min()} - {df['year'].max()}"
            }
            st.json(dataset_stats)
        else:
            st.error("Dataset information not available")
    
    def _display_rag_status(self):
        """Display RAG system status"""
        st.markdown("**🧠 RAG System Status:**")
        if self.llm_engine and self.llm_engine.rag_system:
            rag_stats = self.llm_engine.rag_system.get_paper_stats()
            st.json(rag_stats)
        else:
            st.error("RAG system information not available")
    
    def _display_system_resources(self):
        """Display system resource usage"""
        st.markdown("**💾 System Resources:**")
        try:
            memory_info = {
                "ram_usage_percent": psutil.virtual_memory().percent,
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "cpu_percent": psutil.cpu_percent()
            }
            st.json(memory_info)
        except Exception as e:
            st.error(f"Resource information unavailable: {str(e)}")
    
    def _create_export_section(self):
        """Create data export section"""
        st.subheader("📥 Data Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Paper Data"):
                try:
                    csv_data = self.processor.get_papers_dataframe().to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        "research_papers.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            if st.button("💬 Export Chat History"):
                try:
                    if self.llm_engine and self.llm_engine.conversation_history:
                        chat_json = json.dumps(self.llm_engine.conversation_history, indent=2)
                        st.download_button(
                            "Download Chat JSON",
                            chat_json,
                            "chat_history.json",
                            "application/json"
                        )
                    else:
                        st.info("No chat history available")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col3:
            if st.button("🔧 Export Configuration"):
                try:
                    config_json = yaml.dump(self.config, default_flow_style=False)
                    st.download_button(
                        "Download Config",
                        config_json,
                        "system_config.yaml",
                        "text/yaml"
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown(
            '<h1 class="main-header">🤖 CS Expert AI Assistant</h1>', 
            unsafe_allow_html=True
        )
        st.markdown("### CS Expert AI Assistant with Foundation LLM + RAG")
        
        # Load configuration and initialize system
        try:
            self.config = self.load_configuration()
            
            with st.spinner("🚀 Initializing CS Expert AI System..."):
                (self.processor, self.nlp_pipeline, 
                 self.llm_engine, self.visualizer) = self.initialize_system(self.config)
            
            # Display system status in sidebar
            self.display_system_status()
            
            # Main navigation tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "💬 AI Assistant",
                "🔍 Paper Search",
                "📊 Analytics",
                "⚙️ System Info"
            ])
            
            with tab1:
                self.create_chat_interface()
            
            with tab2:
                self.create_paper_search_interface()
            
            with tab3:
                self.create_visualization_dashboard()
            
            with tab4:
                self.create_system_info_tab()
            
            # Footer
            st.markdown("---")
            st.markdown(
                "Built with ❤️ using Streamlit, Ollama (Llama 3), ChromaDB, and advanced NLP techniques | "
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
        except Exception as e:
            st.error(f"❌ Application failed to start: {str(e)}")
            st.info("Please check your configuration and dependencies.")


# Application entry point
def main():
    """Application entry point"""
    app = CSExpertApp()
    app.run()


if __name__ == "__main__":
    main()
