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

def format_indian_ticker(ticker):
    """
    Format a ticker symbol for Indian stock exchanges.
    
    Indian stocks on Yahoo Finance have specific suffixes:
    - .NS for National Stock Exchange (NSE)
    - .BO for Bombay Stock Exchange (BSE)
    
    Args:
        ticker (str): The ticker symbol to format
        
    Returns:
        str: Formatted ticker symbol
    """
    # Skip formatting for indices (which start with ^)
    if ticker.startswith('^'):
        return ticker
        
    # If ticker already has a suffix, return as is
    if ticker.endswith(('.NS', '.BO')):
        return ticker
    
    # If we've seen this ticker before, use the same exchange
    if ticker in TICKER_EXCHANGE_MAP:
        return f"{ticker}{TICKER_EXCHANGE_MAP[ticker]}"
    
    # By default, try NSE first, then BSE if NSE data is not available
    return f"{ticker}.NS"

def fetch_stock_data(ticker, period='1mo', use_cache=True, exchange=None):
    """
    Fetch stock data for a given ticker and period from Yahoo Finance.
    Supports Indian stock exchanges (NSE and BSE).
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, etc.)
        use_cache (bool): Whether to use database cache
        exchange (str, optional): Preferred exchange - 'NSE' or 'BSE' for Indian stocks
        
    Returns:
        pandas.DataFrame: DataFrame containing stock data
    """
    try:
        # Skip formatting for indices (which start with ^)
        if ticker.startswith('^'):
            formatted_ticker = ticker
        # If specifically requesting Indian exchange
        elif exchange == 'NSE':
            formatted_ticker = f"{ticker}.NS"
        elif exchange == 'BSE':
            formatted_ticker = f"{ticker}.BO"
        else:
            formatted_ticker = format_indian_ticker(ticker)
        
        # Store the original ticker for cache lookups
        original_ticker = ticker
        ticker = formatted_ticker
        
        # Check if we can get data from cache
        if use_cache:
            cached_data = get_cached_stock_data(ticker, period, max_age_hours=1)
            if cached_data is not None:
                return cached_data
        
        # If not in cache or cache disabled, fetch from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        # For Indian stocks, try alternate exchange if first attempt returns empty data
        if data.empty and ticker.endswith('.NS'):
            # Try BSE if NSE data is not available
            ticker = ticker.replace('.NS', '.BO')
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty:
                # Remember this ticker's exchange for future use
                TICKER_EXCHANGE_MAP[original_ticker] = '.BO'
        
        elif data.empty and ticker.endswith('.BO'):
            # Try NSE if BSE data is not available
            ticker = ticker.replace('.BO', '.NS')
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty:
                # Remember this ticker's exchange for future use
                TICKER_EXCHANGE_MAP[original_ticker] = '.NS'
        
        elif not data.empty:
            # Remember successful exchange for this ticker
            if ticker.endswith('.NS'):
                TICKER_EXCHANGE_MAP[original_ticker] = '.NS'
            elif ticker.endswith('.BO'):
                TICKER_EXCHANGE_MAP[original_ticker] = '.BO'
        
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
        market (str): Market to fetch news for - 'global', 'us', or 'india'
    
    Returns:
        list: List of news items with title, link, and summary
    """
    try:
        # Select the appropriate ticker based on market
        if market.lower() == 'india':
            # Use NIFTY 50 index for Indian market news
            ticker = yf.Ticker("^NSEI")
        elif market.lower() == 'us':
            # S&P 500 for US market news
            ticker = yf.Ticker("^GSPC")
        else:
            # Default to S&P 500 (global market news often comes from US sources)
            ticker = yf.Ticker("^GSPC")
            
        news = ticker.news
        
        # If no news found for the specified market, try another index
        if not news and market.lower() == 'india':
            # Try BSE SENSEX if NIFTY doesn't have news
            ticker = yf.Ticker("^BSESN")
            news = ticker.news
            
        # If still no news, try global index as fallback
        if not news and market.lower() == 'india':
            ticker = yf.Ticker("^GSPC")
            news = ticker.news
            st.warning("Limited Indian market news available, showing global news instead.")
        
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
        market (str): The market to fetch sectors for - 'global' (US), 'india'
    
    Returns:
        list: List of sectors with performance data
    """
    try:
        # Define sectors based on market
        if market.lower() == 'india':
            # Indian sectors with their respective indexes or ETFs
            sectors = [
                {"ticker": "NIFTYIT.NS", "name": "IT"},
                {"ticker": "NIFTYFMCG.NS", "name": "FMCG"},
                {"ticker": "NIFTYBANK.NS", "name": "Banking"},
                {"ticker": "NIFTYPHARMA.NS", "name": "Pharma"},
                {"ticker": "NIFTYAUTO.NS", "name": "Auto"},
                {"ticker": "NIFTYMETAL.NS", "name": "Metal"},
                {"ticker": "NIFTYENERGY.NS", "name": "Energy"},
                {"ticker": "CNXREALTY.NS", "name": "Realty"},
                {"ticker": "NIFTYPSUBANK.NS", "name": "PSU Bank"},
                {"ticker": "NIFTYPVTBANK.NS", "name": "Private Bank"}
            ]
        else:
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
        
        # If no data could be fetched for Indian market, fall back to global 
        if market.lower() == 'india' and len(sector_data) < 3:
            st.warning("Limited Indian sector data available, supplementing with global sectors.")
            # Try to fetch global data
            global_sectors = fetch_sector_performance(market='global')
            # Add only sectors not already in the data
            existing_sectors = [s["Sector"] for s in sector_data]
            for sector in global_sectors:
                if sector["Sector"] not in existing_sectors:
                    sector_data.append(sector)
        
        return sector_data
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
        return []

def get_market_indices(market='global'):
    """
    Get a list of major market indices based on market.
    
    Args:
        market (str): The market to get indices for - 'global', 'us', 'india'
        
    Returns:
        list: List of index ticker symbols
    """
    if market.lower() == 'india':
        return [
            "^NSEI",     # NIFTY 50 (India's benchmark index)
            "^BSESN",    # BSE SENSEX
            "^NSEBANK",  # NIFTY Bank
            "^CNXIT",    # NIFTY IT
            "^CNXPHARMA" # NIFTY Pharma
        ]
    elif market.lower() == 'us':
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
            "^NSEI"      # NIFTY 50 (India)
        ]
