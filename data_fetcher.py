import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import streamlit as st
import random
from database import cache_stock_data, get_cached_stock_data

# Dict to track which exchange a ticker is from
TICKER_EXCHANGE_MAP = {}

# Simple compatibility function for any code that still calls format_indian_ticker
def format_indian_ticker(ticker):
    """
    For backwards compatibility only - returns ticker unchanged
    """
    return ticker

def fetch_stock_data(ticker, period='1mo', use_cache=True, exchange=None):
    """
    Fetch stock data for a given ticker and period from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, etc.)
        use_cache (bool): Whether to use database cache
        exchange (str, optional): Not used - kept for backward compatibility
        
    Returns:
        pandas.DataFrame: DataFrame containing stock data
    """
    try:
        # Store the original ticker for cache lookups
        original_ticker = ticker
        
        # Check if we can get data from cache
        if use_cache:
            cached_data = get_cached_stock_data(ticker, period, max_age_hours=1)
            if cached_data is not None:
                return cached_data
        
        # If not in cache or cache disabled, fetch from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            st.warning(f"No data available for {original_ticker} in the selected period.")
            return None
        
        # Cache the data for future use
        if use_cache:
            cache_stock_data(ticker, period, data)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache news for 1 hour
def fetch_market_news(limit=10, market='global'):
    """
    Fetch recent financial news from Yahoo Finance API.
    
    Args:
        limit (int): Maximum number of news items to return
        market (str): Market to fetch news for - 'global' or 'us'
    
    Returns:
        list: List of news items with title, link, and summary
    """
    try:
        # Select the appropriate ticker based on market
        if market.lower() == 'us':
            # S&P 500 for US market news
            ticker = yf.Ticker("^GSPC")
        else:
            # Default to S&P 500 (global market news often comes from US sources)
            ticker = yf.Ticker("^GSPC")
            
        news = ticker.news
        
        if not news:
            return []
        
        processed_news = []
        for item in news[:limit]:
            processed_news.append({
                'id': item.get('uuid', ''),  # Unique ID for relating news
                'title': item.get('title', 'No title'),
                'link': item.get('link', '#'),
                'publisher': item.get('publisher', 'Unknown'),
                'summary': item.get('summary', ''),
                'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
                if 'providerPublishTime' in item else 'Unknown',
                'source': f"{market.title()} Market" if market.lower() != 'global' else 'Global Market'
            })
        
        return processed_news
    except Exception as e:
        st.error(f"Error fetching market news: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache sector data for 1 hour
def fetch_sector_performance(market='global'):
    """
    Fetch sector performance data.
    
    Args:
        market (str): The market to fetch sectors for - 'global' or 'us'
    
    Returns:
        list: List of sectors with performance data
    """
    try:
        # Global/US sectors using major sector ETFs
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
                    "Ticker": sector["ticker"],
                    "Market": market.title()
                })
        
        return sector_data
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
        return []

def get_market_indices(market='global'):
    """
    Get a list of major market indices based on market.
    
    Args:
        market (str): The market to get indices for - 'global' or 'us'
        
    Returns:
        list: List of index ticker symbols
    """
    if market.lower() == 'us':
        return [
            "^GSPC",     # S&P 500
            "^DJI",      # Dow Jones Industrial Average
            "^IXIC",     # NASDAQ Composite
            "^RUT",      # Russell 2000
            "^VIX"       # CBOE Volatility Index
        ]
    else:
        # Global indices
        return [
            "^GSPC",     # S&P 500 (US)
            "^STOXX50E", # EURO STOXX 50 (Europe)
            "^N225",     # Nikkei 225 (Japan)
            "^HSI",      # Hang Seng Index (Hong Kong)
            "^FTSE"      # FTSE 100 (UK)
        ]
