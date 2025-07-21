import pandas as pd
import numpy as np
import arxiv
import json
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import os
import yaml
import time
import random

class EnhancedArxivProcessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_config = self.config['data']
        self.max_results = self.data_config['max_papers']
        self.categories = self.data_config['categories']
        
        self.papers = []
        self.embeddings = None
        self.metadata = {}
        
        # Initialize models
        embedding_model = self.config['nlp']['embedding_model']
        self.sentence_model = SentenceTransformer(embedding_model)
        
    def fetch_arxiv_papers(self, save_path="arxiv_papers.json", chunk_size=100):
        """
        Fetch papers from arXiv with improved error handling and chunking
        """
        if os.path.exists(save_path):
            print(f"Loading existing papers from {save_path}")
            with open(save_path, 'r') as f:
                data = json.load(f)
                self.papers = data['papers']
                self.metadata = data.get('metadata', {})
            return
        
        print(f"Fetching {self.max_results} papers from arXiv...")
        
        # Create query for multiple categories
        query_parts = [f"cat:{cat}" for cat in self.categories]
        query = " OR ".join(query_parts)
        
        papers_data = []
        failed_attempts = 0
        max_failures = 3
        
        while len(papers_data) < self.max_results and failed_attempts < max_failures:
            try:
                remaining = self.max_results - len(papers_data)
                current_chunk = min(chunk_size, remaining)
                
                search = arxiv.Search(
                    query=query,
                    max_results=current_chunk,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                chunk_papers = []
                for result in search.results():
                    paper = {
                        'id': result.entry_id.split('/')[-1],
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary,
                        'categories': result.categories,
                        'published': result.published.isoformat(),
                        'pdf_url': result.pdf_url,
                        'primary_category': result.primary_category,
                        'updated': result.updated.isoformat()
                    }
                    chunk_papers.append(paper)
                
                if chunk_papers:
                    papers_data.extend(chunk_papers)
                    print(f"Fetched {len(chunk_papers)} papers, total: {len(papers_data)}")
                    failed_attempts = 0  # Reset failure count on success
                else:
                    failed_attempts += 1
                    print(f"Empty response, attempt {failed_attempts}/{max_failures}")
                
                # Rate limiting
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                failed_attempts += 1
                print(f"Error fetching papers (attempt {failed_attempts}): {str(e)}")
                if failed_attempts < max_failures:
                    time.sleep(random.uniform(2, 4))
        
        self.papers = papers_data
        
        # Create metadata
        self.metadata = {
            'total_papers': len(papers_data),
            'categories': list(set([paper['primary_category'] for paper in papers_data])),
            'fetch_date': datetime.now().isoformat(),
            'query_used': query
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump({
                'papers': papers_data,
                'metadata': self.metadata
            }, f, indent=2)
        
        print(f"Successfully fetched and saved {len(papers_data)} papers")
    
    def create_enhanced_embeddings(self, save_path="enhanced_embeddings.pkl"):
        """
        Create enhanced embeddings with metadata
        """
        if os.path.exists(save_path):
            print(f"Loading existing embeddings from {save_path}")
            with open(save_path, 'rb') as f:
                embedding_data = pickle.load(f)
                self.embeddings = embedding_data['embeddings']
                self.embedding_metadata = embedding_data.get('metadata', {})
            return
        
        print("Creating enhanced embeddings...")
        
        # Prepare texts with title and abstract
        texts = []
        for paper in self.papers:
            # Combine title, abstract, and categories for richer embeddings
            combined_text = f"Title: {paper['title']} Abstract: {paper['abstract']} Categories: {' '.join(paper['categories'])}"
            texts.append(combined_text)
        
        # Create embeddings in batches to handle memory
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.sentence_model.encode(
                batch_texts,
                show_progress_bar=True,
                batch_size=batch_size
            )
            all_embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 100 == 0:
                print(f"Processed {i + batch_size}/{len(texts)} embeddings")
        
        self.embeddings = np.array(all_embeddings)
        
        # Create embedding metadata
        self.embedding_metadata = {
            'model_name': self.config['nlp']['embedding_model'],
            'embedding_dimension': self.embeddings.shape[1],
            'total_papers': len(self.papers),
            'created_date': datetime.now().isoformat()
        }
        
        # Save embeddings with metadata
        embedding_data = {
            'embeddings': self.embeddings,
            'metadata': self.embedding_metadata
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"Enhanced embeddings created and saved: {self.embeddings.shape}")
    
    def get_papers_dataframe(self):
        """
        Convert papers to enhanced pandas DataFrame
        """
        df = pd.DataFrame(self.papers)
        
        # Add derived columns
        df['year'] = pd.to_datetime(df['published']).dt.year
        df['month'] = pd.to_datetime(df['published']).dt.month
        df['author_count'] = df['authors'].apply(len)
        df['abstract_length'] = df['abstract'].apply(len)
        df['title_length'] = df['title'].apply(len)
        df['category_count'] = df['categories'].apply(len)
        
        # Extract first author
        df['first_author'] = df['authors'].apply(lambda x: x[0] if x else '')
        
        return df
    
    def get_category_distribution(self):
        """Get distribution of papers by category"""
        df = self.get_papers_dataframe()
        return df['primary_category'].value_counts()
    
    def get_temporal_distribution(self):
        """Get temporal distribution of papers"""
        df = self.get_papers_dataframe()
        return df.groupby(['year', 'month']).size()
    
    def search_papers_by_metadata(self, author=None, category=None, year=None, title_keywords=None):
        """Search papers by metadata criteria"""
        df = self.get_papers_dataframe()
        
        if author:
            df = df[df['authors'].apply(lambda x: any(author.lower() in a.lower() for a in x))]
        
        if category:
            df = df[df['categories'].apply(lambda x: any(category.lower() in c.lower() for c in x))]
        
        if year:
            df = df[df['year'] == year]
        
        if title_keywords:
            df = df[df['title'].str.contains(title_keywords, case=False, na=False)]
        
        return df.to_dict('records')
