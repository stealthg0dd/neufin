import streamlit as st
import os
from landing import landing_page
from app_module import run_dashboard
from auth_manager import is_authenticated, show_login_ui, init_auth_session

# Initialize auth session
init_auth_session()

# Set page config
st.set_page_config(
    page_title="Neufin AI - Neural Powered Finance Unlocked",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# Neufin AI\nNeural powered finance unlocked. Cutting-edge market sentiment analysis using advanced AI."
    }
)

# Add SEO meta tags
st.markdown("""
    <meta name="description" content="Neufin AI - Neural powered finance unlocked. Market sentiment analysis platform using advanced AI and real-time data.">
    <meta name="keywords" content="financial AI, market sentiment, neural networks, finance, investment, stock analysis, AI investing, financial dashboard, neural powered finance">
    <meta name="author" content="Neufin OÜ">
    <meta property="og:title" content="Neufin AI - Neural Powered Finance Unlocked">
    <meta property="og:description" content="Cutting-edge market sentiment analysis platform powered by advanced AI and real-time data for smart investment decisions.">
    <meta property="og:type" content="website">
    <!-- Google Analytics tracking code would go here -->
""", unsafe_allow_html=True)

# Main application flow
if "show_auth" in st.session_state and st.session_state["show_auth"]:
    # Show authentication UI
    show_login_ui()
elif is_authenticated():
    # User is authenticated, show dashboard
    run_dashboard()
else:
    # User is not authenticated, show landing page
    landing_page()