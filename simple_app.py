import streamlit as st
import os
import time
from datetime import datetime

# Set page config first before any other Streamlit commands
st.set_page_config(
    page_title="Neufin AI - Neural Powered Finance Unlocked",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# Neufin AI\nNeural powered finance unlocked. Cutting-edge market sentiment analysis using advanced AI."
    }
)

# Import landing page
from landing import landing_page
# Import auth modules
from auth_manager import is_authenticated, show_login_ui, init_auth_session

# Initialize session state with default values
def init_session_state():
    """Initialize session state with default values"""
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
    if "show_auth" not in st.session_state:
        st.session_state["show_auth"] = False
    
    # Landing page specific state
    if "email_input" not in st.session_state:
        st.session_state["email_input"] = ""
    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False
    if "valid_email" not in st.session_state:
        st.session_state["valid_email"] = True
    if "sentiment_value" not in st.session_state:
        st.session_state["sentiment_value"] = 0.65  # Default positive sentiment
    if "sentiment_trend" not in st.session_state:
        # Create a sample trend for demonstration
        st.session_state["sentiment_trend"] = [0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7, 0.65, 0.7, 0.75]

def main():
    """Main entry point for the application"""
    # Initialize auth session
    init_auth_session()
    
    # Initialize other session state
    init_session_state()
    
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
        from app import load_dashboard
        
        # Apply custom dashboard styling
        st.markdown("""
        <style>
            .dark-ui {
                background-color: #121212;
                color: #e0e0e0;
            }
            
            .block-container {
                padding-top: 2rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Load the dashboard
        load_dashboard()
        
        # Add spacer and footer
        st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
        
        # Add company footer
        st.markdown("""
        <div style="font-size:0.8em; color:#999; text-align:center; margin-top:30px; padding: 0 20px;">
            <p><strong>Disclaimer:</strong> All sentiment analysis and recommendations are powered by AI and should be used for informational purposes only.
            Neufin does not provide financial advice. Investment decisions should be made in consultation with financial professionals.</p>
            <p>Data source: Alpha Vantage | Last updated: {}</p>
            <div style="margin-top: 15px; opacity: 0.7;">
                <img src="data:image/png;base64,""" + open('neufin_new_logo_base64.txt', 'r').read() + """" style="height: 40px; vertical-align: middle; margin-right: 8px;">
                © 2025 Neufin OÜ | A Unit of Ctech Ventures | Järvevana tee 9, 11314, Tallinn, Estonia
            </div>
        </div>
        """.format(st.session_state.get("last_update", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    else:
        # User is not authenticated, show landing page
        landing_page()
        
# Run the main function if this script is run directly
if __name__ == "__main__":
    main()