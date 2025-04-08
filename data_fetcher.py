import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import streamlit as st
import random
from database import cache_stock_data, get_cached_stock_data

def fetch_stock_data(ticker, period='1mo', use_cache=True):
    """
    Fetch stock data for a given ticker and period from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, etc.)
        use_cache (bool): Whether to use database cache
    
    Returns:
        pandas.DataFrame: DataFrame containing stock data
    """
    try:
        # Check if we can get data from cache
        if use_cache:
            cached_data = get_cached_stock_data(ticker, period, max_age_hours=1)
            if cached_data is not None:
                return cached_data
        
        # If not in cache or cache disabled, fetch from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            st.warning(f"No data available for {ticker} in the selected period.")
            return None
        
        # Cache the data for future use
        if use_cache:
            cache_stock_data(ticker, period, data)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache news for 1 hour
def fetch_market_news(limit=10):
    """
    Fetch recent financial news from Yahoo Finance API.
    
    Args:
        limit (int): Maximum number of news items to return
    
    Returns:
        list: List of news items with title, link, and summary
    """
    try:
        # Yahoo Finance doesn't have a direct news API that's easily accessible
        # So we'll fetch news data from the ticker directly
        ticker = yf.Ticker("^GSPC")  # S&P 500 index
        news = ticker.news
        
        if not news:
            return []
        
        processed_news = []
        for item in news[:limit]:
            processed_news.append({
                'title': item.get('title', 'No title'),
                'link': item.get('link', '#'),
                'publisher': item.get('publisher', 'Unknown'),
                'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
                if 'providerPublishTime' in item else 'Unknown',
            })
        
        return processed_news
    except Exception as e:
        st.error(f"Error fetching market news: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache sector data for 1 hour
def fetch_sector_performance():
    """
    Fetch sector performance data.
    
    Returns:
        list: List of sectors with performance data
    """
    try:
        # This typically requires parsing from a webpage or using a premium API
        # For demonstration purposes, we'll get some data from major sector ETFs
        sectors = [
            {"ticker": "XLK", "name": "Technology"},
            {"ticker": "XLF", "name": "Financial"},
            {"ticker": "XLE", "name": "Energy"},
            {"ticker": "XLV", "name": "Healthcare"},
            {"ticker": "XLI", "name": "Industrial"},
            {"ticker": "XLP", "name": "Consumer Staples"},
            {"ticker": "XLY", "name": "Consumer Discretionary"},
            {"ticker": "XLB", "name": "Materials"},
            {"ticker": "XLU", "name": "Utilities"},
            {"ticker": "XLRE", "name": "Real Estate"}
        ]
        
        sector_data = []
        for sector in sectors:
            ticker_data = fetch_stock_data(sector["ticker"], period="1mo")
            if ticker_data is not None and not ticker_data.empty:
                start_price = ticker_data.iloc[0]["Close"]
                end_price = ticker_data.iloc[-1]["Close"]
                performance = ((end_price - start_price) / start_price) * 100
                
                sector_data.append({
                    "Sector": sector["name"],
                    "Performance": performance,
                    "Ticker": sector["ticker"]
                })
        
        return sector_data
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
        return []
