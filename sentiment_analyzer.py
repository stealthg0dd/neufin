import pandas as pd
import numpy as np
from textblob import TextBlob
import streamlit as st

@st.cache_data
def analyze_text_sentiment(text):
    """
    Analyze the sentiment of a text string using TextBlob.
    
    Args:
        text (str): The text to analyze
    
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    try:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        
        # TextBlob returns polarity between -1 and 1
        return blob.sentiment.polarity
    except Exception as e:
        st.error(f"Error analyzing text sentiment: {str(e)}")
        return 0.0

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
