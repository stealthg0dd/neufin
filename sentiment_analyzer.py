import pandas as pd
import numpy as np
from textblob import TextBlob
import streamlit as st
import os
import json
from datetime import datetime
import re

# Try to import more advanced NLP models
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from scipy.special import softmax
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    
# Try to import Anthropic for AI-based sentiment analysis
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Try to import OpenAI for AI-based sentiment analysis if configured
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
except ImportError:
    OPENAI_AVAILABLE = False

# Initialize AI clients
anthropic_client = None
openai_client = None

if ANTHROPIC_AVAILABLE:
    try:
        anthropic_client = Anthropic()
    except Exception as e:
        st.warning(f"Could not initialize Anthropic client: {str(e)}")

if OPENAI_AVAILABLE:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"Could not initialize OpenAI client: {str(e)}")

# Financial keywords with sentiment scores
FINANCIAL_SENTIMENT_KEYWORDS = {
    'positive': {
        'bullish': 0.8,
        'surge': 0.7,
        'rally': 0.7,
        'outperform': 0.6,
        'beat': 0.6,
        'growth': 0.6,
        'profit': 0.6,
        'gain': 0.5,
        'upgrade': 0.5,
        'strong': 0.5,
        'positive': 0.5,
        'rise': 0.4,
        'up': 0.3,
        'higher': 0.3,
        'increase': 0.3,
        'improved': 0.3,
        'boost': 0.3,
    },
    'negative': {
        'bearish': -0.8,
        'crash': -0.8,
        'plunge': -0.7,
        'tumble': -0.7,
        'downgrade': -0.6,
        'miss': -0.6,
        'loss': -0.6,
        'weak': -0.5,
        'negative': -0.5,
        'decline': -0.5,
        'fall': -0.4,
        'down': -0.3,
        'lower': -0.3,
        'decrease': -0.3,
        'worsen': -0.3,
        'drop': -0.3,
    }
}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def analyze_text_sentiment(text, use_advanced_nlp=True, use_ai=True):
    """
    Analyze the sentiment of a text string using multiple NLP models.
    
    Args:
        text (str): The text to analyze
        use_advanced_nlp (bool): Whether to use advanced NLP models if available
        use_ai (bool): Whether to use AI-based sentiment analysis if available
    
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Combine scores from multiple models
    scores = []
    weights = []
    
    try:
        # 1. Basic TextBlob sentiment (always available)
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        scores.append(textblob_score)
        weights.append(1.0)  # Base weight
        
        # 2. Use financial keyword dictionary for domain-specific sentiment
        financial_score = 0
        text_lower = text.lower()
        
        # Count matches for positive and negative financial terms
        matches = 0
        for term, score in FINANCIAL_SENTIMENT_KEYWORDS['positive'].items():
            if term in text_lower:
                financial_score += score
                matches += 1
                
        for term, score in FINANCIAL_SENTIMENT_KEYWORDS['negative'].items():
            if term in text_lower:
                financial_score += score
                matches += 1
        
        # Only add if we found matches
        if matches > 0:
            financial_score = financial_score / matches  # Average the scores
            scores.append(financial_score)
            weights.append(1.5)  # Higher weight for financial terms
        
        # 3. Use advanced NLP if available and requested
        if ADVANCED_NLP_AVAILABLE and use_advanced_nlp:
            # For now, we'll use a simplified approach based on TF-IDF and context
            # In a real implementation, this would use a pre-trained model
            advanced_score = analyze_advanced_nlp_sentiment(text)
            scores.append(advanced_score)
            weights.append(2.0)  # Higher weight for advanced NLP
            
        # 4. Use AI-based sentiment analysis if available and requested
        if use_ai:
            ai_score = None
            
            # Try OpenAI first if available (it's faster)
            if OPENAI_AVAILABLE and openai_client:
                ai_score = analyze_openai_sentiment(text)
                
            # If OpenAI failed or isn't available, try Anthropic
            if ai_score is None and ANTHROPIC_AVAILABLE and anthropic_client:
                ai_score = analyze_anthropic_sentiment(text)
                
            if ai_score is not None:
                scores.append(ai_score)
                weights.append(3.0)  # Highest weight for AI analysis
        
        # Calculate weighted average
        if scores:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return np.clip(weighted_score, -1, 1)  # Ensure it's in the range [-1, 1]
        else:
            return 0.0
            
    except Exception as e:
        st.error(f"Error analyzing text sentiment: {str(e)}")
        # Fallback to TextBlob if there was an error
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0

def analyze_advanced_nlp_sentiment(text):
    """
    Use more advanced NLP techniques for sentiment analysis
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Sentiment score between -1 and 1
    """
    if not ADVANCED_NLP_AVAILABLE:
        return None
        
    try:
        # This is a simplified implementation
        # In a production environment, you would use a pre-trained model
        
        # Extract features that indicate financial sentiment
        features = []
        
        # 1. Count financial terms by sentiment
        positive_terms = sum(1 for term in FINANCIAL_SENTIMENT_KEYWORDS['positive'] if term in text.lower())
        negative_terms = sum(1 for term in FINANCIAL_SENTIMENT_KEYWORDS['negative'] if term in text.lower())
        
        # 2. Look for percentage changes with regex
        percent_changes = re.findall(r'(up|down|increase|decrease|rose|fell|gained|lost|climb|drop).*?(\d+(?:\.\d+)?)%', text.lower())
        percent_sentiment = 0
        for direction, amount in percent_changes:
            amount = float(amount)
            if direction in ['up', 'increase', 'rose', 'gained', 'climb']:
                percent_sentiment += min(amount / 10, 1.0)  # Cap at 1.0
            else:
                percent_sentiment -= min(amount / 10, 1.0)  # Cap at -1.0
                
        # 3. Look for financial metrics
        has_earnings = 1 if re.search(r'earnings|revenue|profit|income', text.lower()) else 0
        has_guidance = 1 if re.search(r'guidance|forecast|outlook|expect', text.lower()) else 0
        
        # Combine features
        if positive_terms + negative_terms > 0:
            term_ratio = (positive_terms - negative_terms) / (positive_terms + negative_terms)
        else:
            term_ratio = 0
            
        # Simple model: weighted combination of features
        score = (
            term_ratio * 0.5 + 
            np.clip(percent_sentiment, -1, 1) * 0.3 +
            (has_earnings * 0.1) + 
            (has_guidance * 0.1)
        )
        
        return np.clip(score, -1, 1)
        
    except Exception as e:
        st.warning(f"Advanced NLP analysis failed: {str(e)}")
        return None

def analyze_anthropic_sentiment(text):
    """
    Use Anthropic's Claude model for sentiment analysis
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Sentiment score between -1 and 1, or None if unavailable
    """
    if not ANTHROPIC_AVAILABLE or not anthropic_client:
        return None
        
    try:
        # Skip very short texts
        if len(text.split()) < 3:
            return None
            
        prompt = f"""Please analyze the financial sentiment of the following text on a scale from -1 (very negative) to 1 (very positive), with 0 being neutral. Return only a single number between -1 and 1.

Text: "{text}"

Sentiment score (-1 to 1):"""

        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=5,
            temperature=0,
            system="You are a financial sentiment analysis expert. Analyze the sentiment of financial text on a scale from -1 (very negative) to 1 (very positive). Respond with only a number between -1 and 1.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract just the number from the response
        response_text = response.content[0].text.strip()
        score_match = re.search(r'(-?\d+(\.\d+)?)', response_text)
        
        if score_match:
            score = float(score_match.group(1))
            # Ensure it's within range
            return np.clip(score, -1, 1)
        else:
            return None
            
    except Exception as e:
        st.warning(f"Anthropic sentiment analysis failed: {str(e)}")
        return None

def analyze_openai_sentiment(text):
    """
    Use OpenAI's GPT model for sentiment analysis
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Sentiment score between -1 and 1, or None if unavailable
    """
    if not OPENAI_AVAILABLE or not openai_client:
        return None
        
    try:
        # Skip very short texts
        if len(text.split()) < 3:
            return None
            
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial sentiment analysis expert. Analyze the sentiment of financial text on a scale from -1 (very negative) to 1 (very positive). Respond with JSON containing a single key 'sentiment_score' with a number value between -1 and 1."},
                {"role": "user", "content": f"Analyze the financial sentiment of this text: \"{text}\""}
            ]
        )
        
        # Parse the JSON response
        response_text = response.choices[0].message.content
        response_json = json.loads(response_text)
        
        if 'sentiment_score' in response_json:
            score = float(response_json['sentiment_score'])
            # Ensure it's within range
            return np.clip(score, -1, 1)
        else:
            return None
            
    except Exception as e:
        st.warning(f"OpenAI sentiment analysis failed: {str(e)}")
        return None

@st.cache_data
def analyze_stock_sentiment(stock_data):
    """
    Analyze stock sentiment based on price action and technical indicators.
    
    Args:
        stock_data (pandas.DataFrame): DataFrame containing stock price data
    
    Returns:
        float: Sentiment score between -1 (bearish) and 1 (bullish)
    """
    if stock_data is None or stock_data.empty:
        return 0.0
    
    try:
        # Calculate various technical indicators and derive sentiment
        
        # 1. Price trend
        price_change = stock_data['Close'].pct_change().dropna()
        price_trend = price_change.mean() * 20  # Scale to contribute to sentiment
        
        # 2. Volatility (lower is better)
        volatility = price_change.std()
        volatility_sentiment = -np.clip(volatility * 10, -0.5, 0.5)  # Invert and scale
        
        # 3. Volume trend
        volume_change = stock_data['Volume'].pct_change().dropna()
        volume_trend = volume_change.mean() * 10  # Scale to contribute to sentiment
        
        # 4. Recent momentum (last 5 days vs previous 5)
        if len(stock_data) >= 10:
            recent_period = stock_data['Close'][-5:].mean()
            previous_period = stock_data['Close'][-10:-5].mean()
            momentum = (recent_period - previous_period) / previous_period
            momentum_sentiment = np.clip(momentum * 5, -0.5, 0.5)  # Scale to sentiment range
        else:
            momentum_sentiment = 0
        
        # Combine all factors with weights
        sentiment_score = (
            price_trend * 0.4 +
            volatility_sentiment * 0.2 +
            volume_trend * 0.1 +
            momentum_sentiment * 0.3
        )
        
        # Ensure the final score is within the range of -1 to 1
        sentiment_score = np.clip(sentiment_score, -1, 1)
        
        return sentiment_score
    except Exception as e:
        st.error(f"Error analyzing stock sentiment: {str(e)}")
        return 0.0

@st.cache_data
def analyze_news_sentiment_batch(news_items):
    """
    Analyze sentiment for a batch of news items.
    
    Args:
        news_items (list): List of news dictionaries with 'title' keys
    
    Returns:
        list: List of sentiment scores corresponding to each news item
    """
    if not news_items:
        return []
    
    try:
        sentiments = []
        for item in news_items:
            title = item.get('title', '')
            sentiment = analyze_text_sentiment(title)
            sentiments.append(sentiment)
        
        return sentiments
    except Exception as e:
        st.error(f"Error analyzing batch news sentiment: {str(e)}")
        return [0.0] * len(news_items)
