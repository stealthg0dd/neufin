import streamlit as st

def format_sentiment_score(score):
    """
    Convert a numerical sentiment score to a descriptive label.
    
    Args:
        score (float): Sentiment score between -1 and 1
    
    Returns:
        str: Descriptive sentiment label
    """
    if score >= 0.5:
        return "Very Bullish"
    elif score >= 0.2:
        return "Bullish"
    elif score > -0.2:
        return "Neutral"
    elif score > -0.5:
        return "Bearish"
    else:
        return "Very Bearish"

def get_market_preferences():
    """
    Get market preferences from session state or create defaults.
    
    Returns:
        dict: Market preferences including selected market and settings
    """
    if 'market_preferences' not in st.session_state:
        st.session_state.market_preferences = {
            'selected_market': 'global',
            'show_global_insights': True,
            'show_us_insights': True,
            'show_india_insights': True,
            'default_tickers': {
                'global': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'us': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'india': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            }
        }
    return st.session_state.market_preferences

def update_market_preferences(key, value):
    """
    Update a specific market preference.
    
    Args:
        key (str): The preference key to update
        value: The new value for the preference
    """
    prefs = get_market_preferences()
    prefs[key] = value

def get_sentiment_color(score):
    """
    Get a color representation for a sentiment score.
    
    Args:
        score (float): Sentiment score between -1 and 1
    
    Returns:
        str: Hex color code
    """
    if score >= 0.5:
        return "#1B5E20"  # Dark green
    elif score >= 0.2:
        return "#4CAF50"  # Green
    elif score > -0.2:
        return "#FFC107"  # Amber
    elif score > -0.5:
        return "#FF9800"  # Orange
    else:
        return "#D32F2F"  # Red

def get_market_indices():
    """
    Get a list of major market indices based on the current selected market.
    
    Returns:
        list: List of index ticker symbols
    """
    from data_fetcher import get_market_indices as fetch_indices
    
    # Get current market selection
    prefs = get_market_preferences()
    selected_market = prefs['selected_market']
    
    # Use the new function from data_fetcher
    return fetch_indices(market=selected_market)

def format_large_number(num):
    """
    Format large numbers in a human-readable format (K, M, B).
    
    Args:
        num (int or float): Number to format
    
    Returns:
        str: Formatted number string
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num:.1f}"

def calculate_moving_average(data, window):
    """
    Calculate moving average of a data series.
    
    Args:
        data (pandas.Series): Data series
        window (int): Moving average window size
    
    Returns:
        pandas.Series: Moving average series
    """
    return data.rolling(window=window).mean()
