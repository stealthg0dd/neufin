"""
News recommendation engine for personalized financial news.
Uses content-based filtering and user interaction data to recommend
relevant financial news articles to users.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import Counter
import streamlit as st
from textblob import TextBlob
from data_fetcher import fetch_market_news
from sentiment_analyzer import analyze_text_sentiment
from database import get_db_session, User, UserSettings
import re
from scipy.spatial.distance import cosine

# Try to import more advanced NLP models if available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class NewsRecommender:
    """Class to handle news recommendation logic"""
    
    def __init__(self, user_id=None):
        """
        Initialize the news recommender.
        
        Args:
            user_id (int, optional): The ID of the user to get recommendations for.
                If None, will provide general recommendations.
        """
        self.user_id = user_id
        self.user_interests = self._get_user_interests() if user_id else None
        self.news_cache = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) if SKLEARN_AVAILABLE else None
        self.vectors = None
    
    def _get_user_interests(self):
        """Get user interests from their favorites and settings"""
        if not self.user_id:
            return []
        
        interests = []
        
        # Get user's favorite stocks
        with get_db_session() as session:
            try:
                # Get user's favorite stocks/sectors
                favorites = session.query(User).filter_by(id=self.user_id).first().favorites
                for fav in favorites:
                    if fav.favorite_type == 'stock':
                        interests.append(fav.item_name)
                
                # Get user settings for default stocks
                settings = session.query(UserSettings).filter_by(user_id=self.user_id).first()
                if settings and settings.default_stocks:
                    default_stocks = settings.default_stocks.split(',')
                    interests.extend(default_stocks)
            except Exception as e:
                st.warning(f"Could not fetch user interests: {str(e)}")
                return []
        
        # Remove duplicates and return
        return list(set(interests))
    
    def _fetch_recent_news(self, limit=30):
        """Fetch recent news and cache locally"""
        if self.news_cache is None:
            self.news_cache = fetch_market_news(limit=limit)
        return self.news_cache
    
    def _extract_keywords(self, text):
        """Extract important keywords from text"""
        # Simple keyword extraction using frequency and stopwords
        # More advanced models would use NER or keyword extraction algorithms
        stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
                    'is', 'was', 'were', 'be', 'been', 'being', 'as', 'of', 'from', 'this', 'that', 'these', 'those']
        
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Split into words and remove stopwords
        words = [word for word in text.split() if word not in stopwords and len(word) > 2]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Return most common words
        return [word for word, freq in word_freq.most_common(10)]
    
    def _calculate_relevance_score(self, news_item, user_interests):
        """Calculate relevance score for a news item based on user interests"""
        if not user_interests:
            return 0
        
        title = news_item.get('title', '')
        summary = news_item.get('summary', '')
        full_text = f"{title} {summary}"
        
        # Calculate score based on interest keyword matches
        score = 0
        for interest in user_interests:
            if interest.lower() in full_text.lower():
                score += 1
                # Give extra weight if it's in the title
                if interest.lower() in title.lower():
                    score += 0.5
        
        # Normalize score by the number of interests
        if user_interests:
            score = score / len(user_interests)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_content_similarity(self, news_items, feature='content_vector'):
        """Calculate content similarity between news items using TF-IDF vectors"""
        if not SKLEARN_AVAILABLE or len(news_items) < 2:
            return
        
        # Extract titles and summaries
        texts = [f"{item.get('title', '')} {item.get('summary', '')}" for item in news_items]
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.vectors = tfidf_matrix
        
        # Add vectors to news items
        for i, item in enumerate(news_items):
            item[feature] = tfidf_matrix[i]

    def get_recommendations(self, limit=10, include_sentiment=True):
        """
        Get personalized news recommendations for the user.
        
        Args:
            limit (int): Maximum number of recommendations to return
            include_sentiment (bool): Whether to include sentiment analysis
            
        Returns:
            list: List of recommended news items with scores
        """
        news_items = self._fetch_recent_news(limit=30)  # Fetch more than needed for filtering
        
        if not news_items:
            return []
        
        # If sklearn is available, calculate content vectors
        if SKLEARN_AVAILABLE:
            self._calculate_content_similarity(news_items)
        
        # Calculate relevance scores
        for item in news_items:
            # Calculate content-based relevance score
            if self.user_interests:
                item['relevance_score'] = self._calculate_relevance_score(item, self.user_interests)
            else:
                item['relevance_score'] = 0.5  # Neutral score for non-logged-in users
            
            # Extract keywords from title and summary
            title = item.get('title', '')
            summary = item.get('summary', '')
            item['keywords'] = self._extract_keywords(f"{title} {summary}")
            
            # Add sentiment score if requested
            if include_sentiment:
                sentiment_score = analyze_text_sentiment(title)
                item['sentiment_score'] = sentiment_score
        
        # Sort by relevance score (descending)
        sorted_items = sorted(news_items, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Return the top N items
        return sorted_items[:limit]
    
    def get_related_news(self, article_id, limit=5):
        """
        Get news related to a specific article based on content similarity
        
        Args:
            article_id (str): ID of the article to find related news for
            limit (int): Maximum number of related articles to return
            
        Returns:
            list: List of related news items
        """
        news_items = self._fetch_recent_news(limit=30)
        
        if not news_items:
            return []
        
        # Find the target article
        target_article = None
        for item in news_items:
            if item.get('id') == article_id:
                target_article = item
                break
        
        if not target_article:
            return []
        
        # If sklearn is available, use TF-IDF similarity
        if SKLEARN_AVAILABLE and self.vectors is None:
            self._calculate_content_similarity(news_items)
            
        if SKLEARN_AVAILABLE and self.vectors is not None:
            # Get index of target article
            target_idx = news_items.index(target_article)
            target_vector = self.vectors[target_idx]
            
            # Calculate similarities
            similarities = []
            for i, item in enumerate(news_items):
                if i == target_idx:
                    continue
                sim = cosine_similarity(target_vector, self.vectors[i])[0][0]
                similarities.append((item, sim))
            
            # Sort by similarity (descending)
            sorted_items = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Return the top N items
            return [item for item, sim in sorted_items[:limit]]
        else:
            # Fallback to keyword-based similarity
            target_keywords = self._extract_keywords(f"{target_article.get('title', '')} {target_article.get('summary', '')}")
            
            if not target_keywords:
                return []
            
            # Calculate keyword overlap for each article
            similarities = []
            for item in news_items:
                if item.get('id') == article_id:
                    continue
                
                item_keywords = self._extract_keywords(f"{item.get('title', '')} {item.get('summary', '')}")
                overlap = len(set(target_keywords) & set(item_keywords))
                similarity = overlap / max(len(target_keywords), 1)
                similarities.append((item, similarity))
            
            # Sort by similarity (descending)
            sorted_items = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Return the top N items
            return [item for item, sim in sorted_items[:limit]]

def format_news_card(news_item, show_relevance=False, show_sentiment=True):
    """
    Format a news item as a stylish card with relevance and sentiment indicators
    
    Args:
        news_item (dict): News item to format
        show_relevance (bool): Whether to show relevance score
        show_sentiment (bool): Whether to show sentiment indicator
        
    Returns:
        str: HTML for the formatted card
    """
    title = news_item.get('title', 'No title')
    link = news_item.get('link', '#')
    source = news_item.get('source', 'Unknown source')
    
    # Format the published time
    published_time = news_item.get('published', None)
    if published_time:
        # If it's a string, parse it
        if isinstance(published_time, str):
            try:
                published_time = datetime.strptime(published_time, '%Y-%m-%dT%H:%M:%SZ')
            except:
                published_time = datetime.now()
    else:
        published_time = datetime.now()
    
    # Calculate how long ago it was published
    time_diff = datetime.now() - published_time
    if time_diff.days > 0:
        time_ago = f"{time_diff.days} days ago"
    elif time_diff.seconds // 3600 > 0:
        time_ago = f"{time_diff.seconds // 3600} hours ago"
    else:
        time_ago = f"{time_diff.seconds // 60} minutes ago"
    
    # Get sentiment if available
    sentiment_html = ""
    if show_sentiment and 'sentiment_score' in news_item:
        sentiment_score = news_item['sentiment_score']
        
        # Determine color based on sentiment
        if sentiment_score > 0.2:
            sentiment_color = "#4CAF50"  # Green for positive
            sentiment_icon = "📈"
        elif sentiment_score < -0.2:
            sentiment_color = "#FF5252"  # Red for negative
            sentiment_icon = "📉"
        else:
            sentiment_color = "#FFC107"  # Yellow for neutral
            sentiment_icon = "➖"
        
        sentiment_html = f"""
        <div style="display: inline-block; margin-left: 10px; padding: 3px 8px; 
                    background-color: rgba({sentiment_color.replace('#', '').upper()}, 0.1); 
                    border: 1px solid {sentiment_color}; border-radius: 4px; font-size: 12px; color: {sentiment_color};">
            {sentiment_icon} {sentiment_score:.2f}
        </div>
        """
    
    # Add relevance indicator if requested
    relevance_html = ""
    if show_relevance and 'relevance_score' in news_item:
        relevance = news_item['relevance_score']
        # Use stars to indicate relevance
        if relevance > 0.8:
            stars = "★★★★★"
        elif relevance > 0.6:
            stars = "★★★★☆"
        elif relevance > 0.4:
            stars = "★★★☆☆"
        elif relevance > 0.2:
            stars = "★★☆☆☆"
        else:
            stars = "★☆☆☆☆"
            
        relevance_html = f"""
        <div style="display: inline-block; margin-left: 10px; font-size: 12px; color: #7B68EE;">
            Relevance: {stars}
        </div>
        """
    
    # Create keywords pills if available
    keywords_html = ""
    if 'keywords' in news_item and news_item['keywords']:
        keywords = news_item['keywords'][:5]  # Limit to 5 keywords
        pills = []
        for keyword in keywords:
            pills.append(f'<span style="display: inline-block; margin-right: 5px; margin-bottom: 5px; padding: 2px 8px; background-color: rgba(123, 104, 238, 0.1); border-radius: 12px; font-size: 11px; color: #7B68EE;">{keyword}</span>')
        
        keywords_html = f"""
        <div style="margin-top: 8px;">
            {''.join(pills)}
        </div>
        """
    
    # Create the card HTML
    card_html = f"""
    <div style="padding: 15px; margin-bottom: 15px; border-radius: 10px; 
                border: 1px solid rgba(123, 104, 238, 0.2);
                background: linear-gradient(to bottom right, rgba(30, 30, 46, 0.8), rgba(17, 17, 17, 0.9));
                transition: transform 0.2s, box-shadow 0.2s;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <a href="{link}" target="_blank" style="color: #E0E0E0; text-decoration: none; font-weight: bold; font-size: 16px;
                                                      display: block; margin-bottom: 5px;">{title}</a>
                <div style="color: #999; font-size: 12px;">
                    {source} · {time_ago}
                    {sentiment_html}
                    {relevance_html}
                </div>
            </div>
        </div>
        {keywords_html}
    </div>
    """
    
    return card_html