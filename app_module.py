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

# Import financial snapshot generator
from financial_snapshot import create_financial_snapshot

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

# These functions were previously imported from app.py
# Now implementing them directly to avoid circular imports

def apply_dashboard_styling():
    """Apply custom styling for the dashboard"""
    pass  # This is now handled directly in run_dashboard

def handle_landing_redirect():
    """Handle redirect from landing page with email parameter"""
    # Check for email in query params
    if 'email' in st.query_params:
        # If email is in params, set it in session state and clear params
        email = st.query_params.get('email')
        if email and email.strip():
            st.session_state["prefilled_email"] = email
            st.query_params.clear()
            # Set the show_auth flag to redirect to login
            st.session_state["show_auth"] = True
            return True
    return False

def check_feature_access(feature):
    """
    Check if current user has access to a specific feature based on subscription level.
    
    Args:
        feature (str): Feature name to check ('basic', 'premium', 'free', etc.)
    
    Returns:
        bool: True if user has access, False otherwise
    """
    # Get current user
    user = get_current_user()
    if not user:
        return False
    
    # Check user subscription level
    subscription = get_user_subscription(user['id'])
    if not subscription:
        return False
    
    # Map feature to required subscription level
    feature_levels = {
        'basic': ['basic', 'premium', 'enterprise'],
        'premium': ['premium', 'enterprise'],
        'enterprise': ['enterprise'],
        'free': ['free', 'basic', 'premium', 'enterprise']
    }
    
    # Check if user's subscription level has access to this feature
    required_levels = feature_levels.get(feature, ['premium'])
    return subscription.get('level', 'free') in required_levels

def load_dashboard():
    """Placeholder for the original dashboard loader"""
    # This is now handled directly in run_dashboard
    pass

def add_footer():
    """Add the branding footer to the application"""
    # This is now handled directly in run_dashboard
    pass

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
        
    # Apply dashboard styling - simplified version
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }
        .neufin-card {
            background-color: rgba(30, 30, 30, 0.7);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(123, 104, 238, 0.3);
        }
        .sentiment-gauge {
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .market-pulse {
            font-size: 1.2rem; 
            font-weight: bold; 
            margin-bottom: 15px;
            background: linear-gradient(90deg, #7B68EE 0%, #9370DB 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Check if it's demo mode
    if st.session_state.get("show_demo", False):
        st.markdown("## Neufin AI Demo Mode")
        st.markdown("This is a demonstration of the Neufin AI platform capabilities.")
        
        # Create tabs for different demo features
        demo_tab1, demo_tab2, demo_tab3, demo_tab4 = st.tabs(["Market Sentiment", "Stock Analysis", "Financial Snapshot", "AI Assistant"])
        
        with demo_tab1:
            st.markdown("### Market Sentiment Analysis")
            st.markdown("Our advanced AI algorithms analyze market data and news to provide real-time sentiment insights.")
            
            # Demo sentiment gauge
            sentiment_score = 0.72  # Demo value
            sentiment_color = "#00C853" if sentiment_score > 0.66 else "#FFD600" if sentiment_score > 0.33 else "#FF3D00"
            st.markdown(f"""
                <div class="sentiment-gauge" style="background-color: rgba({int(255 * (1-sentiment_score))}, {int(255 * sentiment_score)}, 0, 0.2);">
                    <div style="font-size: 1rem;">Overall Market Sentiment</div>
                    <div style="font-size: 2rem; color: {sentiment_color};">{sentiment_score:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with demo_tab2:
            st.markdown("### Stock Analysis")
            st.markdown("AI-powered analysis of individual stocks with technical and fundamental insights.")
            
            # Demo stock recommendation
            st.info("DEMO: Based on current market conditions, our AI recommends: Buy AAPL, Hold MSFT, Sell FB")
            
        with demo_tab3:
            st.markdown("### Financial Snapshot Generator")
            st.markdown("Create a comprehensive financial snapshot with just one click, featuring interactive elements and AI-powered insights.")
            
            # Add a simplified version of the financial snapshot
            st.markdown("""
            <div style="background: rgba(30, 30, 30, 0.7); border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(123, 104, 238, 0.3);">
                <h4 style="margin-top: 0;">📊 One-Click Financial Snapshot</h4>
                <div style="margin-bottom: 15px;">
                    <strong>Stock Symbol:</strong> AAPL &nbsp; | &nbsp; <strong>Time Period:</strong> 1mo
                </div>
                <button style="background-color: #7B68EE; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; width: 100%;">
                    📸 Generate Snapshot
                </button>
            </div>
            """, unsafe_allow_html=True)
            
            # Show features of the financial snapshot
            st.markdown("#### ✨ Features")
            feature_cols = st.columns(2)
            
            with feature_cols[0]:
                st.markdown("""
                * 📈 Interactive stock charts
                * 🔍 Micro-interactions for data points
                * 🧠 AI-powered insights
                """)
                
            with feature_cols[1]:
                st.markdown("""
                * 💹 Personalized wellness score
                * 🚀 Financial mood emoji tracker
                * 🏢 Sector performance context
                """)
                
            # Show a preview of the financial wellness score
            st.markdown("#### 🧘‍♂️ Preview: Financial Wellness Score")
            st.markdown("""
            <div style="background: rgba(0, 0, 0, 0.1); padding: 15px; border-radius: 5px; margin-bottom: 15px; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 10px;">Financial Wellness Score</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: #FFD600;">78<span style="font-size: 1.2rem;">/100</span></div>
                <div style="font-size: 0.9rem; opacity: 0.7; margin-top: 10px;">Based on diversification, risk management, and market timing</div>
            </div>
            """, unsafe_allow_html=True)
            
        with demo_tab4:
            st.markdown("### AI Assistant")
            st.markdown("Ask our AI anything about markets, stocks, or investment strategies.")
            
            st.text_input("Ask a question:", key="demo_question")
            if st.button("Get Answer"):
                st.markdown("""
                **DEMO Response:**
                
                Based on current market indicators, technology stocks are showing strong momentum due to:
                
                1. Increased enterprise spending on digital transformation
                2. Favorable regulatory environment
                3. Strong earnings growth in the sector
                
                Consider allocating 30-40% of your portfolio to quality tech stocks with solid earnings.
                """)
                
    else:
        # Offer options to access different sections
        st.title("Neufin AI Dashboard")
        
        # Create tabs for different sections
        main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
            "Market Sentiment", 
            "Investment Recommendations", 
            "Financial Snapshot", 
            "AI Assistant"
        ])
        
        with main_tab1:
            st.header("Market Sentiment Analysis")
            st.write("This section would display the market sentiment analysis dashboard.")
            
        with main_tab2:
            st.header("Investment Recommendations")
            st.write("This section would display investment recommendations based on market sentiment.")
            
        with main_tab3:
            st.header("Financial Snapshot Generator")
            # Add financial snapshot generator
            create_financial_snapshot()
            
        with main_tab4:
            st.header("AI Assistant")
            st.write("This section would display the AI assistant for natural language queries.")
    
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; background-color: rgba(18, 18, 18, 0.8); padding: 10px; text-align: center; border-top: 1px solid rgba(123, 104, 238, 0.3);">
        <p style="margin: 0; font-size: 0.8rem; color: #888;">
            Neufin AI © 2025 | Neural powered finance unlocked - Financial Superintelligence in your reach<br>
            Neufin OÜ registered in Estonia (Järvevana tee 9, 11314, Tallinn) | A unit of Ctech Ventures
        </p>
    </div>
    """, unsafe_allow_html=True)