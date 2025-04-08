import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time
import streamlit as st
import random
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from database import cache_stock_data, get_cached_stock_data

# Initialize Alpha Vantage client
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    st.error("Alpha Vantage API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")

# Initialize the Alpha Vantage time series client
try:
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    print("Alpha Vantage client initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize Alpha Vantage client: {str(e)}")
    print(f"Error initializing Alpha Vantage client: {str(e)}")

# Dict to track which exchange a ticker is from
TICKER_EXCHANGE_MAP = {}

# Simple compatibility function for any code that still calls format_indian_ticker
def format_indian_ticker(ticker):
    """
    For backwards compatibility only - returns ticker unchanged
    """
    return ticker

def generate_fallback_stock_data(ticker, period='1mo'):
    """
    Generate fallback stock data when Yahoo Finance is unavailable.
    This is only for UI demonstration when external API is down.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, etc.)
    
    Returns:
        pandas.DataFrame: Synthetic DataFrame containing demo stock data
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create date range based on period
    end_date = datetime.now()
    if period == '1d':
        start_date = end_date - timedelta(days=1)
        periods = 24
        freq = 'H'
    elif period == '5d':
        start_date = end_date - timedelta(days=5)
        periods = 5
        freq = 'D'
    elif period == '1mo':
        start_date = end_date - timedelta(days=30)
        periods = 30
        freq = 'D'
    elif period == '3mo':
        start_date = end_date - timedelta(days=90)
        periods = 90
        freq = 'D'
    elif period == '6mo':
        start_date = end_date - timedelta(days=180)
        periods = 180
        freq = 'D'
    elif period == '1y':
        start_date = end_date - timedelta(days=365)
        periods = 52
        freq = 'W'
    else:
        start_date = end_date - timedelta(days=30)
        periods = 30
        freq = 'D'
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Initial price depends on the ticker (just to make different tickers look different)
    if ticker.startswith('^'):  # Index
        base_price = 3000 if ticker == '^GSPC' else 30000 if ticker == '^DJI' else 15000
    else:  # Stock
        # Hash the ticker to get a consistent starting price
        base_price = sum(ord(c) for c in ticker) % 900 + 100
    
    # Generate prices with some randomness but overall trend
    np.random.seed(sum(ord(c) for c in ticker))  # Seed based on ticker for consistency
    
    # Trend direction: positive or negative based on ticker
    trend = 0.001 * (sum(ord(c) for c in ticker) % 20 - 10)
    
    # Generate price data
    price_data = []
    current_price = base_price
    for i in range(len(dates)):
        # Add some daily volatility
        daily_change = np.random.normal(trend, 0.01)
        current_price *= (1 + daily_change)
        
        # Generate OHLC data
        daily_volatility = current_price * 0.01
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, daily_volatility))
        low_price = open_price - abs(np.random.normal(0, daily_volatility))
        close_price = open_price + np.random.normal(0, daily_volatility)
        
        # Ensure high is highest and low is lowest
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Volume depends on volatility
        volume = int(np.random.normal(1000000, 200000) * (1 + abs(daily_change) * 10))
        
        price_data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
        
        current_price = close_price
    
    # Create DataFrame
    df = pd.DataFrame(price_data, index=dates)
    df.index.name = 'Date'
    
    return df

def fetch_stock_data(ticker, period='1mo', use_cache=True, exchange=None):
    """
    Fetch stock data for a given ticker and period from Alpha Vantage.
    
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
            if cached_data is not None and not cached_data.empty:
                return cached_data
        
        # Ensure ticker doesn't have any invalid suffix (cleaning up any potential .NS suffix)
        if ticker.endswith('.NS') and not ticker.startswith('^'):
            ticker = ticker.replace('.NS', '')
            
        # Add more logging for debugging
        print(f"Fetching data for ticker: {ticker}, period: {period}")
        
        # Map period string to Alpha Vantage output size and function
        if period == '1d':
            outputsize = 'compact'  # Last 100 data points (1 day for minute data)
            function = 'TIME_SERIES_INTRADAY'
            interval = '5min'
        elif period == '5d':
            outputsize = 'full'  # Full data (up to 5 days for minute data)
            function = 'TIME_SERIES_INTRADAY'
            interval = '15min'
        else:
            # For longer periods, use daily data
            if period in ['1mo', '3mo']:
                outputsize = 'compact'  # Last 100 data points (about 3 months of trading days)
            else:  # 6mo, 1y, etc.
                outputsize = 'full'  # Full data (up to 20 years of daily data)
            function = 'TIME_SERIES_DAILY'
            interval = None
        
        # Handle index tickers specially (they start with ^)
        if ticker.startswith('^'):
            # For indices, we'll use SPY as a proxy for ^GSPC (S&P 500), etc.
            if ticker == '^GSPC':
                ticker = 'SPY'  # S&P 500 ETF
            elif ticker == '^DJI':
                ticker = 'DIA'  # Dow Jones Industrial Average ETF
            elif ticker == '^IXIC':
                ticker = 'QQQ'  # NASDAQ-100 ETF
            elif ticker == '^RUT':
                ticker = 'IWM'  # Russell 2000 ETF
            elif ticker == '^VIX':
                ticker = 'VIXY'  # VIX ETF
            elif ticker == '^STOXX50E':
                ticker = 'FEZ'  # EURO STOXX 50 ETF
            elif ticker == '^N225':
                ticker = 'EWJ'  # Japan ETF (proxy for Nikkei 225)
            elif ticker == '^HSI':
                ticker = 'EWH'  # Hong Kong ETF (proxy for Hang Seng)
            elif ticker == '^FTSE':
                ticker = 'EWU'  # UK ETF (proxy for FTSE 100)
            print(f"Using ETF {ticker} as a proxy for index")
        
        # If ALPHA_VANTAGE_API_KEY is not available, use fallback data
        if not ALPHA_VANTAGE_API_KEY:
            print("Alpha Vantage API key not found, using fallback data")
            return generate_fallback_stock_data(original_ticker, period)
            
        # Fetch data from Alpha Vantage
        try:
            if function == 'TIME_SERIES_INTRADAY':
                data, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize=outputsize)
            else:  # 'TIME_SERIES_DAILY'
                data, meta_data = ts.get_daily(symbol=ticker, outputsize=outputsize)
                
            # Alpha Vantage returns data with most recent first, so reverse it
            data = data.sort_index()
            
            # Limit data based on period
            if period == '1mo':
                data = data.tail(30)  # Approximately 1 month of trading days
            elif period == '3mo':
                data = data.tail(90)  # Approximately 3 months of trading days
            elif period == '6mo':
                data = data.tail(180)  # Approximately 6 months of trading days
            elif period == '1y':
                data = data.tail(252)  # Approximately 1 year of trading days
            
            # Rename columns to match Yahoo Finance format
            data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }, inplace=True)
            
            # Check if data is empty
            if data.empty:
                print(f"Empty data returned for {ticker} (original: {original_ticker})")
                st.warning(f"No data available for {original_ticker} in the selected period.")
                return generate_fallback_stock_data(original_ticker, period)
                
            # Cache the data for future use
            if use_cache:
                cache_stock_data(original_ticker, period, data)
            
            return data
            
        except Exception as e:
            print(f"Alpha Vantage API error for {ticker}: {str(e)}")
            st.warning(f"Unable to fetch data from Alpha Vantage for {original_ticker}. Using fallback data.")
            return generate_fallback_stock_data(original_ticker, period)
            
    except Exception as e:
        print(f"Exception fetching data for {ticker}: {str(e)}")
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        
        # Use fallback data for demonstration purposes when deployed
        print(f"Using fallback data for {ticker} after exception")
        return generate_fallback_stock_data(original_ticker, period)

def generate_fallback_market_news(limit=10, market='global'):
    """
    Generate fallback market news when Yahoo Finance is unavailable.
    
    Args:
        limit (int): Maximum number of news items to return
        market (str): Market to fetch news for - 'global' or 'us'
    
    Returns:
        list: List of news items with title, link, and summary
    """
    import uuid
    from datetime import datetime, timedelta
    
    # Sample news data
    news_templates = [
        {
            'title': "Market Volatility Continues Amid Economic Uncertainty",
            'summary': "Financial markets experienced fluctuations today as investors assessed the latest economic indicators and central bank signals. Analysts suggest caution while economic conditions stabilize.",
            'publisher': "Market Watch Daily"
        },
        {
            'title': "Tech Stocks Lead Market Rally on Innovation Announcements",
            'summary': "Technology sector shares posted significant gains following announcements of breakthrough innovations and strong quarterly results from industry leaders. Analysts predict continued growth potential.",
            'publisher': "Tech Finance News"
        },
        {
            'title': "Energy Sector Faces Challenges with Shifting Global Demands",
            'summary': "Energy companies are navigating complex market conditions as global demand patterns shift and renewable alternatives gain traction. Industry experts recommend diversification strategies.",
            'publisher': "Energy Market Report"
        },
        {
            'title': "Retail Stocks Respond to Changing Consumer Behavior",
            'summary': "Consumer spending patterns show resilience despite economic pressures, with retail stocks responding positively to adaptive business strategies and digital transformation efforts.",
            'publisher': "Retail Investor Daily"
        },
        {
            'title': "Financial Sector Stabilizes After Regulatory Announcements",
            'summary': "Banking and financial services stocks found stability following clarification of regulatory frameworks. Analysts note improved market confidence in the sector's growth prospects.",
            'publisher': "Financial Times"
        },
        {
            'title': "Healthcare Innovation Drives Sector Performance",
            'summary': "Healthcare companies reported strong performance metrics driven by innovation in treatment methodologies and technological integration. Investor interest continues to grow in this resilient sector.",
            'publisher': "Health Market Insights"
        },
        {
            'title': "Manufacturing Data Signals Economic Resilience",
            'summary': "Recent manufacturing indices exceeded expectations, suggesting underlying economic strength despite broader market concerns. Industrial stocks responded positively to the encouraging data.",
            'publisher': "Industrial Market News"
        },
        {
            'title': "Global Trade Developments Impact Market Sentiment",
            'summary': "International trade negotiations and policy shifts created market ripples across multiple sectors. Analysts recommend portfolios balanced with domestic and international exposure.",
            'publisher': "Global Trade Monitor"
        },
        {
            'title': "Real Estate Market Trends Reflect Economic Conditions",
            'summary': "Property markets show regional variations in performance as interest rate expectations and urban migration patterns evolve. REIT performance metrics indicate selective investment opportunities.",
            'publisher': "Real Estate Investor"
        },
        {
            'title': "Sustainability Focus Reshapes Investment Landscapes",
            'summary': "Companies with strong environmental, social, and governance profiles attracted significant investment flows. Market analysts note the growing importance of sustainability metrics in valuation models.",
            'publisher': "Sustainable Finance Report"
        },
        {
            'title': "Dividend Stocks Gain Attention in Uncertain Markets",
            'summary': "Investors seeking stability increased allocations to dividend-paying stocks across multiple sectors. Financial advisors highlight the importance of income generation in balanced portfolios.",
            'publisher': "Dividend Investor Weekly"
        },
        {
            'title': "Small Cap Stocks Show Growth Potential Despite Volatility",
            'summary': "Smaller companies demonstrated resilience and innovation, attracting investor interest despite market fluctuations. Analysts identify selective opportunities in the small cap segment.",
            'publisher': "Small Cap Insights"
        }
    ]
    
    # Create randomized news items with appropriate timestamps
    processed_news = []
    now = datetime.now()
    
    for i in range(min(limit, len(news_templates))):
        # Create a random timestamp within the last 3 days
        hours_ago = i * 4 + (i % 3)  # Spread news over time
        timestamp = now - timedelta(hours=hours_ago)
        
        # Select news template
        template = news_templates[i]
        
        # Add market-specific modifier if needed
        market_prefix = f"{market.title()} markets: " if market.lower() != 'global' else ""
        
        processed_news.append({
            'id': str(uuid.uuid4()),
            'title': market_prefix + template['title'],
            'link': '#',
            'publisher': template['publisher'],
            'summary': template['summary'],
            'published': timestamp.strftime('%Y-%m-%d %H:%M'),
            'source': f"{market.title()} Market" if market.lower() != 'global' else 'Global Market'
        })
    
    return processed_news

@st.cache_data(ttl=3600)  # Cache news for 1 hour
def fetch_market_news(limit=10, market='global'):
    """
    Fetch recent financial news from Alpha Vantage API.
    
    Args:
        limit (int): Maximum number of news items to return
        market (str): Market to fetch news for - 'global' or 'us'
    
    Returns:
        list: List of news items with title, link, and summary
    """
    try:
        # If Alpha Vantage API key is not available, use fallback news
        if not ALPHA_VANTAGE_API_KEY:
            print("Alpha Vantage API key not found, using fallback news")
            return generate_fallback_market_news(limit, market)
            
        # Select the appropriate topics based on market
        if market.lower() == 'us':
            topics = 'us_market'
        else:
            topics = 'global_market'
            
        # Fetch news from Alpha Vantage API
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={topics}&apikey={ALPHA_VANTAGE_API_KEY}&limit={limit}"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Alpha Vantage API error: {response.status_code}, using fallback news")
            return generate_fallback_market_news(limit, market)
            
        data = response.json()
        
        # Check if we got valid data
        if 'feed' not in data or not data['feed']:
            print("No news data available from Alpha Vantage, using fallback news")
            return generate_fallback_market_news(limit, market)
            
        # Process the news items
        processed_news = []
        for item in data['feed'][:limit]:
            # Generate a UUID based on the title for consistency
            import hashlib
            item_id = hashlib.md5(item.get('title', '').encode()).hexdigest()
            
            # Extract the summary from the content if available
            summary = item.get('summary', '')
            if not summary and 'content' in item:
                # Use the first 200 characters of content as summary
                summary = item['content'][:200] + '...' if len(item['content']) > 200 else item['content']
                
            # Format the published date
            published = item.get('time_published', '')
            if published:
                # Format is YYYYMMDDTHHMMSS, we need YYYY-MM-DD HH:MM
                try:
                    published = f"{published[0:4]}-{published[4:6]}-{published[6:8]} {published[9:11]}:{published[11:13]}"
                except:
                    published = "Unknown"
                    
            processed_news.append({
                'id': item_id,
                'title': item.get('title', 'No title'),
                'link': item.get('url', '#'),
                'publisher': item.get('source', 'Unknown'),
                'summary': summary,
                'published': published,
                'source': f"{market.title()} Market" if market.lower() != 'global' else 'Global Market'
            })
        
        return processed_news
    except Exception as e:
        st.error(f"Error fetching market news: {str(e)}")
        print(f"Error fetching market news: {str(e)}, using fallback news")
        return generate_fallback_market_news(limit, market)

def generate_fallback_sector_performance(market='global'):
    """
    Generate fallback sector performance data when Yahoo Finance is unavailable.
    
    Args:
        market (str): The market to generate sectors for - 'global' or 'us'
    
    Returns:
        list: List of sectors with performance data
    """
    import numpy as np
    
    # Define sectors
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
    
    # Generate performance data with some realistic values
    np.random.seed(42)  # For reproducible results
    
    sector_data = []
    for sector in sectors:
        # Generate a performance value between -15% and +15%
        if sector["name"] == "Technology":
            # Tech tends to be more bullish on average
            performance = np.random.normal(5.0, 8.0)
        elif sector["name"] == "Energy":
            # Energy can be volatile
            performance = np.random.normal(0.0, 12.0)
        elif sector["name"] == "Utilities":
            # Utilities tend to be more stable
            performance = np.random.normal(2.0, 4.0)
        else:
            # Other sectors
            performance = np.random.normal(1.0, 6.0)
        
        sector_data.append({
            "Sector": sector["name"],
            "Performance": performance,
            "Ticker": sector["ticker"],
            "Market": market.title()
        })
    
    # Sort by performance for better display
    sector_data.sort(key=lambda x: x["Performance"], reverse=True)
    
    return sector_data

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
        
        # If we couldn't get data for any sectors, use fallback data
        if not sector_data:
            print("No sector data available, using fallback sector performance")
            return generate_fallback_sector_performance(market)
            
        return sector_data
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
        print(f"Error fetching sector performance: {str(e)}, using fallback data")
        return generate_fallback_sector_performance(market)

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
