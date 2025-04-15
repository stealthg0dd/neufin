import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Neufin AI - Neural Powered Finance Unlocked",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# Neufin AI\nNeural powered finance unlocked. Cutting-edge market sentiment analysis using advanced AI."
    }
)

import os
from landing import landing_page
from app_module import run_dashboard
from auth_manager import is_authenticated, show_login_ui, init_auth_session

# Initialize auth session
init_auth_session()

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

# Check query parameters for demo mode
if 'demo' in st.query_params and st.query_params.get('demo') == 'true':
    # Set the demo flag from query parameters
    st.session_state["show_demo"] = True
    st.session_state.show_demo = True
    print("Main flow: Setting demo mode from query parameters")
    # Clear the parameter to prevent reuse on refresh
    st.query_params.clear()

# Main application flow
if "show_auth" in st.session_state and st.session_state["show_auth"]:
    # Show authentication UI
    print("Main flow: Redirecting to auth page")
    show_login_ui()
elif "show_demo" in st.session_state and st.session_state["show_demo"]:
    # Show demo showcase
    print("Main flow: Launching demo showcase")
    # Import the dashboard function and run in demo mode
    st.session_state["demo_mode"] = True
    run_dashboard()
elif "show_ai_assistant" in st.session_state and st.session_state["show_ai_assistant"]:
    # Show AI Assistant directly
    print("Main flow: Redirecting to AI Assistant")
    st.session_state["current_page"] = "ai_assistant"
    run_dashboard()
elif is_authenticated():
    # User is authenticated, show dashboard
    run_dashboard()
else:
    # User is not authenticated, show landing page
    print("Main flow: Showing landing page")
    landing_page()