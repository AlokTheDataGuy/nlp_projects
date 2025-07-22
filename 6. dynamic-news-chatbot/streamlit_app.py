# streamlit_app.py
import streamlit as st
import asyncio
import threading
import time
from news_scraper import NewsScraperWithAPI
from vector_database import NewsVectorDatabase
from llama_chat import LlamaNewsBot
import schedule
from datetime import datetime

# Page config
st.set_page_config(
    page_title="📰 Llama News Chatbot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_system():
    """Initialize all components - cached to avoid reloading"""
    print("🚀 Initializing News Chatbot System...")
    
    news_scraper = NewsScraperWithAPI()
    vector_db = NewsVectorDatabase()
    llama_bot = LlamaNewsBot(vector_db)
    
    return news_scraper, vector_db, llama_bot

def update_knowledge_base(news_scraper, vector_db):
    """Update the knowledge base with fresh news"""
    with st.spinner("🔄 Updating knowledge base with fresh news..."):
        articles = news_scraper.collect_and_scrape_all_news()
        
        if articles:
            vector_db.add_articles(articles)
            st.success(f"✅ Updated with {len(articles)} fresh articles!")
            st.session_state['last_update'] = datetime.now()
        else:
            st.error("❌ Failed to collect articles")

def main():
    st.title("🤖 Llama 3.1 News Chatbot")
    st.markdown("### *Powered by NewsAPI, Vector Search & Llama 3.1 8B*")
    
    # Initialize system
    news_scraper, vector_db, llama_bot = initialize_system()
    
    # Sidebar with controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Manual update button
        if st.button("🔄 Update News Database", type="primary"):
            update_knowledge_base(news_scraper, vector_db)

        # Clear database button
        if st.button("🗑️ Clear Database", help="Clear all articles from the database"):
            if vector_db.clear_database():
                st.success("✅ Database cleared successfully!")
                st.rerun()
        
        # Database stats
        st.header("📊 Database Stats")
        stats = vector_db.get_database_stats()
        if 'error' not in stats:
            st.metric("Total Articles", stats['total_chunks'])
            
            if stats.get('sectors'):
                st.write("**Articles by Sector:**")
                for sector, count in stats['sectors'].items():
                    st.write(f"• {sector.title()}: {count}")
        
        # Bot stats
        bot_stats = llama_bot.get_conversation_stats()
        st.metric("Conversations", bot_stats['total_conversations'])
        st.write(f"**Model:** {bot_stats['model']}")
        
        # Last update info
        if 'last_update' in st.session_state:
            st.write(f"**Last Update:** {st.session_state['last_update'].strftime('%H:%M:%S')}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm your news assistant powered by Llama 3.1 8B. I can help you with the latest news from technology, business, health, science, sports, entertainment, and more. What would you like to know?"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about recent news..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = llama_bot.generate_response(prompt)
            st.markdown(response)
        
        # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Auto-update section
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🔄 **Auto-Update:** The system can be configured to automatically update every 12 hours")
    
    with col2:
        st.info("💡 **Usage:** Ask about recent news in any sector - technology, business, health, science, sports, entertainment")

if __name__ == "__main__":
    main()
