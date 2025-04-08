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
    Get a list of major market indices.
    
    Returns:
        list: List of index ticker symbols
    """
    return [
        "^GSPC",    # S&P 500
        "^DJI",     # Dow Jones Industrial Average
        "^IXIC",    # NASDAQ Composite
        "^RUT",     # Russell 2000
        "^VIX",     # Volatility Index
        "^FTSE",    # FTSE 100
        "^N225",    # Nikkei 225
    ]

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
