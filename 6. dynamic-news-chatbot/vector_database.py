# vector_database.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
import hashlib
import time
import random
import uuid

class NewsVectorDatabase:
    def __init__(self, db_path="./chroma_news_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="news_articles_llama",
            metadata={"description": "Scraped news articles for Llama 3.1 8B"}
        )
        
        # Use a good sentence transformer for embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("🔧 Vector database initialized")
    
    def create_document_chunks(self, article):
        """Split article into meaningful chunks for better retrieval"""
        full_text = f"Title: {article['title']}\n\nContent: {article['full_content']}"
        
        # Simple chunking - split by paragraphs and combine small ones
        paragraphs = full_text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk + para) < 500:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [full_text[:1000]]
    
    def add_articles(self, articles):
        """Add scraped articles to vector database"""
        documents = []
        metadatas = []
        ids = []

        print("📊 Processing articles for vector database...")

        for i, article in enumerate(articles):
            if not article.get('full_content'):
                continue

            # Create chunks from article
            chunks = self.create_document_chunks(article)

            for j, chunk in enumerate(chunks):
                # Create unique ID for each chunk using UUID to ensure uniqueness
                chunk_id = str(uuid.uuid4())

                # Ensure uniqueness within current batch (should not be needed with UUID4, but just in case)
                counter = 0
                while chunk_id in ids:
                    counter += 1
                    chunk_id = str(uuid.uuid4())

                documents.append(chunk)
                metadatas.append({
                    'title': article['title'],
                    'sector': article['sector'],
                    'source': article['source'],
                    'published_at': article['published_at'],
                    'url': article['url'],
                    'chunk_index': j,
                    'scraped_successfully': article['scraped_successfully'],
                    'article_index': i
                })
                ids.append(chunk_id)

        if documents:
            # Check for existing IDs in the database and remove duplicates
            try:
                existing_data = self.collection.get()
                existing_ids = set(existing_data['ids'])

                # Also check for duplicates within the current batch
                current_batch_ids = set()

                # Filter out documents that already exist or are duplicates in current batch
                filtered_docs = []
                filtered_meta = []
                filtered_ids = []

                for doc, meta, doc_id in zip(documents, metadatas, ids):
                    if doc_id not in existing_ids and doc_id not in current_batch_ids:
                        filtered_docs.append(doc)
                        filtered_meta.append(meta)
                        filtered_ids.append(doc_id)
                        current_batch_ids.add(doc_id)

                documents = filtered_docs
                metadatas = filtered_meta
                ids = filtered_ids

            except Exception as e:
                print(f"⚠️  Could not check for existing documents: {e}")
                # If we can't check existing, at least remove duplicates within current batch
                unique_docs = []
                unique_meta = []
                unique_ids = []
                seen_ids = set()

                for doc, meta, doc_id in zip(documents, metadatas, ids):
                    if doc_id not in seen_ids:
                        unique_docs.append(doc)
                        unique_meta.append(meta)
                        unique_ids.append(doc_id)
                        seen_ids.add(doc_id)

                documents = unique_docs
                metadatas = unique_meta
                ids = unique_ids

            if documents:
                # Add to ChromaDB in batches
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_meta = metadatas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]

                    try:
                        self.collection.add(
                            documents=batch_docs,
                            metadatas=batch_meta,
                            ids=batch_ids
                        )
                    except Exception as e:
                        print(f"❌ Error adding batch to database: {e}")
                        continue

                print(f"✅ Added {len(documents)} document chunks to vector database")
            else:
                print("ℹ️  All articles already exist in database")
        else:
            print("⚠️  No valid articles to add to database")
    
    def search_relevant_articles(self, query, n_results=5):
        """Search for articles relevant to user query"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return {'documents': [[]], 'metadatas': [[]]}
    
    def get_database_stats(self):
        """Get statistics about the database"""
        try:
            all_data = self.collection.get()
            total_chunks = len(all_data['ids'])

            sectors = {}
            for meta in all_data['metadatas']:
                sector = meta.get('sector', 'unknown')
                sectors[sector] = sectors.get(sector, 0) + 1

            return {
                'total_chunks': total_chunks,
                'sectors': sectors,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

    def clear_database(self):
        """Clear all articles from the database"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name="news_articles_llama")
            self.collection = self.client.get_or_create_collection(
                name="news_articles_llama",
                metadata={"description": "Scraped news articles for Llama 3.1 8B"}
            )
            print("🗑️  Database cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return False
