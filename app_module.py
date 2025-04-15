import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
import re

# Import auth modules
from auth_manager import (
    init_auth_session, is_authenticated, get_current_user, 
    login_user, register_user, logout_user, init_google_oauth,
    handle_google_callback, handle_facebook_callback, init_facebook_oauth,
    user_has_permission, require_login, require_permission, 
    show_login_ui, show_user_profile_ui, start_free_trial
)

# Import database modules
from database import (
    get_user_subscription, update_subscription, get_historical_sentiment, 
    store_market_sentiment
)

# Import subscription modules
from subscription_manager import (
    create_checkout_session, handle_subscription_webhook,
    show_payment_ui, show_payment_success_ui, show_subscription_management
)

# Import sentiment analysis modules
from sentiment_analyzer import analyze_text_sentiment, analyze_stock_sentiment, analyze_news_sentiment_batch

# Import data fetching modules
from data_fetcher import fetch_stock_data, fetch_market_news, fetch_sector_performance, get_market_indices

# Import investment advisor modules
from investment_advisor import get_stock_recommendations, analyze_global_trade_impact, get_sector_insights

# Import AI analysis modules
from ai_analyst import generate_investment_thesis, generate_sector_outlook, analyze_global_trade_conditions

# Import utility modules
from utils import (
    format_sentiment_score, get_sentiment_color, get_market_preferences, 
    update_market_preferences, format_large_number, get_market_indices
)

# Import predictive analytics modules
from predictive_analytics import (
    predict_future_sentiment, generate_prediction_chart, get_sentiment_insights, 
    forecast_sentiment_impact, integrate_price_prediction
)

# Import news recommendation module
from news_recommender import NewsRecommender, format_news_card

# Import the required functions from app.py
from app import (
    check_feature_access, 
    load_dashboard, 
    handle_landing_redirect,
    apply_dashboard_styling,
    add_footer
)

# Check if OpenAI API key is available for advanced sentiment analysis
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_ADVANCED_AI = bool(OPENAI_API_KEY)

# Function to run the dashboard
def run_dashboard():
    """Main dashboard function that can be called from main.py"""
    # Initialize session state if not already done
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "dashboard"
    if "selected_stock" not in st.session_state:
        st.session_state["selected_stock"] = "AAPL"
    if "time_period" not in st.session_state:
        st.session_state["time_period"] = "1mo"
    if "market_region" not in st.session_state:
        st.session_state["market_region"] = "US"
    if "refresh_data" not in st.session_state:
        st.session_state["refresh_data"] = False
    if "last_update" not in st.session_state:
        st.session_state["last_update"] = datetime.now()
    if "show_demo" not in st.session_state:
        st.session_state["show_demo"] = False
    if "auto_refresh" not in st.session_state:
        st.session_state["auto_refresh"] = False
    if "refresh_interval" not in st.session_state:
        st.session_state["refresh_interval"] = 60  # Default to 60 seconds
    if "last_auto_refresh" not in st.session_state:
        st.session_state["last_auto_refresh"] = datetime.now()
        
    # Apply dashboard styling
    apply_dashboard_styling()
    
    # Check if redirect from landing page
    if handle_landing_redirect():
        st.rerun()
    
    # Check if demo showcase mode is enabled
    if st.session_state.get("show_demo", False):
        # Show the demo showcase instead of the regular dashboard
        from app import load_demo_showcase
        load_demo_showcase()
    else:
        # Regular dashboard flow
        if st.session_state.get("refresh_data", False):
            with st.spinner("Refreshing data in real-time..."):
                load_dashboard()
                # Reset the refresh flag after loading
                st.session_state.refresh_data = False
                # Update the last update timestamp
                st.session_state.last_update = datetime.now()
        else:
            load_dashboard()
            
    # Add automatic refresh behavior
    if st.session_state.get("auto_refresh", False):
        # Calculate time to wait before next refresh
        seconds_since_refresh = (datetime.now() - st.session_state.last_auto_refresh).total_seconds()
        seconds_until_refresh = max(1, min(10, st.session_state.refresh_interval - seconds_since_refresh))
        
        # Only rerun if we're more than 80% of the way to the next refresh
        # This prevents too frequent reruns while still maintaining real-time updates
        if seconds_since_refresh > (st.session_state.refresh_interval * 0.8):
            time.sleep(1)  # Small delay to prevent excessive reruns
            st.rerun()
    
    # Add footer content
    add_footer()