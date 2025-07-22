# news_scraper.py
import requests
import json
from newspaper import Article
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
from bs4 import BeautifulSoup

load_dotenv()

class NewsScraperWithAPI:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')  # Free from newsapi.org
        self.base_url = "https://newsapi.org/v2"
        
        # News sectors - targeting 5 articles each
        self.sectors = [
            'technology', 'business', 'health', 'science',
            'sports', 'entertainment', 'general'
        ]
    
    def fetch_article_urls_by_sector(self, sector, articles_needed=5):
        """Fetch article URLs from NewsAPI for a specific sector"""
        from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        # Use /everything endpoint with 'q' parameter instead of 'category'
        url = f"{self.base_url}/everything"
        
        # Map sectors to search queries since /everything doesn't support categories
        sector_queries = {
            'technology': 'technology OR tech OR software OR AI OR startup',
            'business': 'business OR finance OR economy OR market OR company',
            'health': 'health OR medical OR healthcare OR medicine OR wellness',
            'science': 'science OR research OR study OR discovery OR scientific',
            'sports': 'sports OR football OR basketball OR soccer OR athletics',
            'entertainment': 'entertainment OR movie OR music OR celebrity OR hollywood',
            'general': 'news OR breaking OR latest OR today'
        }
        
        params = {
            'apiKey': self.api_key,
            'q': sector_queries.get(sector, sector),
            'from': from_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(articles_needed * 2, 20)
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            
            # Filter articles with valid URLs and content - FIXED validation
            valid_articles = []
            for article in articles[:articles_needed * 2]:
                # Better validation to handle None values
                title = article.get('title')
                description = article.get('description')
                url = article.get('url')
                
                # Check if all required fields exist and are valid
                if (url and title and 
                    description is not None and 
                    len(str(description).strip()) > 20 and  # Convert to string and check length
                    'source' in article and 
                    article['source'] and 
                    article['source'].get('name')):
                    
                    valid_articles.append({
                        'title': title,
                        'description': description,
                        'url': url,
                        'source': article['source']['name'],
                        'published_at': article.get('publishedAt', ''),
                        'sector': sector
                    })
            
            print(f"📊 Found {len(valid_articles)} valid articles for {sector}")
            return valid_articles[:articles_needed]
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {sector} news: {e}")
            return []

    
    def scrape_article_content(self, article_data):
        """Scrape full content from article URL using newspaper3k"""
        try:
            # Ensure we have valid data
            if not article_data.get('url'):
                article_data['full_content'] = article_data.get('description', 'No content available')
                article_data['scraped_successfully'] = False
                return article_data
                
            # Try newspaper3k first
            article = Article(article_data['url'])
            article.download()
            article.parse()
            
            if len(article.text) > 200:  # Ensure we got substantial content
                article_data['full_content'] = article.text[:3000]  # Limit length
                article_data['scraped_successfully'] = True
                return article_data
            
            # Fallback to BeautifulSoup
            response = requests.get(article_data['url'], timeout=10, 
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) > 200:
                article_data['full_content'] = text[:3000]  # Limit to reasonable length
                article_data['scraped_successfully'] = True
            else:
                article_data['full_content'] = article_data.get('description', 'Limited content available')
                article_data['scraped_successfully'] = False
            
            return article_data
            
        except Exception as e:
            print(f"⚠️  Failed to scrape {article_data.get('url', 'unknown URL')}: {e}")
            article_data['full_content'] = article_data.get('description', 'Content unavailable')
            article_data['scraped_successfully'] = False
            return article_data

    
    def collect_and_scrape_all_news(self):
        """Collect URLs and scrape full content for all sectors"""
        all_articles = []
        total_api_calls = 0
        
        print("🔄 Starting news collection and scraping...")
        
        for sector in self.sectors:
            print(f"📰 Collecting {sector} news...")
            
            # Fetch URLs from NewsAPI
            article_urls = self.fetch_article_urls_by_sector(sector, 5)
            total_api_calls += 1
            
            print(f"📊 Found {len(article_urls)} articles for {sector}")
            
            # Scrape full content for each article
            scraped_articles = []
            for i, article_data in enumerate(article_urls):
                print(f"🔍 Scraping article {i+1}/{len(article_urls)} from {sector}...")
                scraped_article = self.scrape_article_content(article_data)
                scraped_articles.append(scraped_article)
                time.sleep(1)  # Be respectful to websites
            
            successful_scrapes = sum(1 for a in scraped_articles if a['scraped_successfully'])
            print(f"✅ Successfully scraped {successful_scrapes}/{len(scraped_articles)} articles from {sector}")
            
            all_articles.extend(scraped_articles)
        
        print(f"🎉 Collection complete! Total API calls used: {total_api_calls}/1000")
        print(f"📚 Total articles collected: {len(all_articles)}")
        
        return all_articles
