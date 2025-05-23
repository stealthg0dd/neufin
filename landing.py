import streamlit as st
import time
from datetime import datetime
import os
import re

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

# Custom CSS for Mercury-inspired landing page
st.markdown("""
<style>
/* Reset some default styling */
div[data-testid="stAppViewBlockContainer"] {
    padding-top: 0 !important;
}

div[data-testid="stToolbar"] {
    display: none !important;
}

div[data-testid="stSidebarContent"] {
    background-color: #0c0c14 !important;
}

/* Landing page container */
.landing-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 20px;
    position: relative;
    font-family: 'Inter', sans-serif;
    color: #E0E0E0;
}

/* Logo styling */
.landing-logo {
    max-width: 180px;
    margin: 0 auto 30px auto;
    display: block;
}

/* Headings */
.landing-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
    color: #7B68EE;
}

.landing-subtitle {
    font-size: 1.2rem;
    text-align: center;
    margin-bottom: 40px;
    color: #ADB3C9;
}

/* Form styling */
.landing-form {
    max-width: 500px;
    margin: 0 auto 30px auto;
    background: rgba(30, 30, 45, 0.7);
    border-radius: 8px;
    padding: 30px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
}

.input-container input {
    background: rgba(20, 20, 35, 0.7) !important;
    border: 1px solid rgba(123, 104, 238, 0.3) !important;
    color: #E0E0E0 !important;
    border-radius: 4px !important;
    padding: 12px !important;
    margin-bottom: 5px !important;
}

/* Decorative elements */
.stars {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(ellipse at center, rgba(30, 30, 50, 0.4) 0%, rgba(12, 12, 20, 0.8) 100%);
    z-index: -3;
}

.decoration {
    position: fixed;
    border-radius: 50%;
    opacity: 0.1;
    z-index: -2;
    background: linear-gradient(45deg, #7B68EE, #4F3EC8);
}

.decoration-1 {
    width: 500px;
    height: 500px;
    top: -250px;
    left: -150px;
}

.decoration-2 {
    width: 400px;
    height: 400px;
    bottom: -200px;
    right: -100px;
}

.decoration-3 {
    width: 300px;
    height: 300px;
    bottom: 30%;
    left: 10%;
    opacity: 0.05;
}

.landing-footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    color: #676B7E;
    font-size: 0.8rem;
}

.error-message {
    color: #FF5252;
    font-size: 0.9rem;
    margin-top: -5px;
    margin-bottom: 15px;
    display: none;
}
</style>
""", unsafe_allow_html=True)

# Function to validate email
def is_valid_email(email):
    """Validate email address format"""
    if not email or not isinstance(email, str) or len(email.strip()) == 0:
        return False
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# Initialize session state variables for landing page
def init_landing_session_state():
    """Initialize all session state variables needed for the landing page"""
    if 'email_input' not in st.session_state:
        st.session_state.email_input = ""
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'valid_email' not in st.session_state:
        st.session_state.valid_email = True
    if 'redirect_user' not in st.session_state:
        st.session_state.redirect_user = False
    if 'redirect_email' not in st.session_state:
        st.session_state.redirect_email = ""
    if 'show_demo' not in st.session_state:
        st.session_state.show_demo = False
    if 'show_ai_assistant' not in st.session_state:
        st.session_state.show_ai_assistant = False
    if 'sentiment_value' not in st.session_state:
        st.session_state.sentiment_value = 0.65  # Default positive sentiment
    if 'sentiment_trend' not in st.session_state:
        # Create a sample trend for demonstration
        st.session_state.sentiment_trend = [0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7, 0.65, 0.7, 0.75]
    # Set a flag to show login UI directly
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False

# Handle form submission
def handle_submit():
    """Handle email form submission with validation and redirection"""
    email = st.session_state.email_input
    if is_valid_email(email):
        # Set required session state variables for redirection
        st.session_state.submitted = True
        st.session_state.valid_email = True
        st.session_state.redirect_user = True
        st.session_state.email_prefill = email
        st.session_state.redirect_email = email
        st.session_state.show_auth = True  # This is the key flag needed by simple_app.py
        
        # Log the redirection for debugging
        print(f"Email validation successful. Redirecting with email: {email}")
        
        # Force an immediate rerun to trigger the redirect logic
        st.rerun()
    else:
        # Simply mark the email as invalid - we'll show an error message
        st.session_state.valid_email = False
        print(f"Email validation failed for: {email}")

# Function to format sentiment score
def format_sentiment_score(score):
    """Format a sentiment score with a label and color"""
    if score >= 0.7:
        return "Very Bullish", "#00C853"
    elif score >= 0.5:
        return "Bullish", "#64DD17"
    elif score >= 0.3:
        return "Neutral", "#FFD600"
    elif score >= 0.1:
        return "Bearish", "#FF9100"
    else:
        return "Very Bearish", "#FF3D00"

# Function to create a sentiment gauge
def create_sentiment_gauge(sentiment_value):
    """Create an HTML gauge for displaying sentiment"""
    sentiment_label, color = format_sentiment_score(sentiment_value)
    
    # Calculate rotation based on sentiment (from -90 to 90 degrees)
    rotation = -90 + (sentiment_value * 180)
    
    gauge_html = f"""
    <div style="width: 200px; margin: 0 auto 30px auto;">
        <div style="text-align: center; margin-bottom: 5px; font-size: 1.2rem; font-weight: 600; color: {color};">
            Market Sentiment: {sentiment_label}
        </div>
        <div style="width: 200px; height: 100px; position: relative; margin: 0 auto; overflow: hidden;">
            <div style="width: 200px; height: 200px; border-radius: 100%; background: conic-gradient(
                #FF3D00 0% 20%, 
                #FF9100 20% 40%, 
                #FFD600 40% 60%, 
                #64DD17 60% 80%, 
                #00C853 80% 100%
            ); position: absolute; bottom: 0;"></div>
            <div style="width: 180px; height: 180px; border-radius: 100%; background: #121212; position: absolute; bottom: 0; left: 10px;"></div>
            <div style="width: 4px; height: 90px; background: white; position: absolute; bottom: 0; left: 98px; transform-origin: bottom; transform: rotate({rotation}deg); z-index: 1;"></div>
            <div style="width: 20px; height: 20px; border-radius: 100%; background: white; position: absolute; bottom: 0; left: 90px; z-index: 2;"></div>
        </div>
    </div>
    """
    return gauge_html

# Function to create a sentiment trend
def create_sentiment_trend(values):
    """Create a simple HTML chart showing sentiment trend"""
    max_height = 50
    num_points = len(values)
    point_width = 100 / (num_points - 1)
    
    # Create SVG path for the line
    path_points = []
    for i, value in enumerate(values):
        x = i * point_width
        y = max_height - (value * max_height)
        path_points.append(f"{x},{y}")
    
    path = " ".join(path_points)
    
    # Determine color based on recent trend
    recent_trend = values[-1] - values[-3]
    if recent_trend > 0.05:
        line_color = "#00C853"  # Strong up trend
    elif recent_trend > 0:
        line_color = "#64DD17"  # Slight up trend
    elif recent_trend > -0.05:
        line_color = "#FFD600"  # Slight down trend
    else:
        line_color = "#FF3D00"  # Strong down trend
    
    trend_html = f"""
    <div style="width: 200px; margin: 0 auto 20px auto;">
        <div style="text-align: center; margin-bottom: 5px; font-size: 1rem; color: #E0E0E0;">
            7-Day Trend
        </div>
        <svg width="200" height="{max_height}" style="overflow: visible;">
            <polyline
                fill="none"
                stroke="{line_color}"
                stroke-width="3"
                points="{path}"
            />
            <!-- Add dots for each data point -->
            {' '.join([f'<circle cx="{i * point_width}" cy="{max_height - (val * max_height)}" r="3" fill="white" />' for i, val in enumerate(values)])}
        </svg>
    </div>
    """
    return trend_html

# Function to show demo showcase
def show_demo_showcase():
    """Show a demo showcase of the platform features"""
    from app import create_animated_sentiment_trend, show_predictive_analytics
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="font-size: 2rem; color: #7B68EE; margin-bottom: 10px;">Neufin Demo Showcase</h2>
        <p style="color: #ADB3C9; max-width: 600px; margin: 0 auto;">
            Experience the power of Neufin's AI-driven financial insights without registration.
            This demo showcases our core features with sample data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for the demo layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Market Sentiment Analysis")
        # Show animated sentiment trend
        create_animated_sentiment_trend(days=10, with_ai_insights=True)
        
    with col2:
        st.subheader("Sector Performance")
        # Create a sample sector performance chart
        sector_data = {
            "Technology": 7.2,
            "Healthcare": 3.5,
            "Financial": 1.8,
            "Energy": -2.3,
            "Consumer": 4.1
        }
        
        # Sort sectors by performance
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
        
        # Create bar chart with conditional colors
        for sector, performance in sorted_sectors:
            if performance > 5:
                color = "#00C853"  # Strong positive
            elif performance > 0:
                color = "#64DD17"  # Positive
            elif performance > -3:
                color = "#FF9100"  # Slight negative
            else:
                color = "#FF3D00"  # Strong negative
                
            # Calculate width percentage (normalize to 0-100%)
            max_perf = max([abs(p) for _, p in sorted_sectors])
            width_pct = abs(performance) / max_perf * 100
            
            # Create the bar
            st.markdown(f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: 500;">{sector}</span>
                    <span style="color: {'#ADB3C9' if performance < 0 else '#E0E0E0'};">{performance:+.1f}%</span>
                </div>
                <div style="background: rgba(30, 30, 40, 0.4); border-radius: 4px; height: 10px; width: 100%; overflow: hidden;">
                    <div style="background: {color}; height: 100%; width: {width_pct}%; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show predictive analytics section
    show_predictive_analytics()
    
    # Add a button to register now
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <h3 style="color: #E0E0E0; margin-bottom: 15px;">Ready to unlock all features?</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the registration button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📝 Sign Up Now", key="register_from_demo", type="primary", use_container_width=True):
            st.session_state.show_demo = False
            st.session_state.show_auth = True
            print("Register from demo clicked! Redirecting to auth page")
            st.rerun()

# Function to redirect to login
def show_login():
    """Redirect to login page by setting session state"""
    st.session_state.show_auth = True
    print("show_login() called! Setting show_auth to True")
    st.rerun()
    
# Create the landing page layout
def landing_page():
    # Initialize session state first
    init_landing_session_state()
    
    # If show_login is set, redirect to login page
    if st.session_state.show_login:
        # Use the auth_manager's show_login_ui
        from auth_manager import show_login_ui
        show_login_ui()
        return
    
    # Handle redirect if user has submitted a valid email
    if st.session_state.redirect_user and st.session_state.redirect_email:
        # Set query parameters for redirection to sign up page
        st.session_state.show_auth = True  # This will cause simple_app.py to show login UI
        st.session_state.email_prefill = st.session_state.redirect_email  # For prefilling the email
        
        # Clear the flag to avoid infinite redirects
        st.session_state.redirect_user = False
        st.rerun()
    
    # If demo is active, show the demo showcase
    if st.session_state.show_demo:
        show_demo_showcase()
        return
        
    # Decorative elements
    st.markdown('<div class="stars"></div>', unsafe_allow_html=True)
    st.markdown('<div class="decoration decoration-1"></div>', unsafe_allow_html=True)
    st.markdown('<div class="decoration decoration-2"></div>', unsafe_allow_html=True)
    st.markdown('<div class="decoration decoration-3"></div>', unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    
    # Logo
    st.markdown(f'<img src="data:image/png;base64,{open("neufin_new_logo_base64.txt", "r").read()}" class="landing-logo" alt="Neufin AI">', unsafe_allow_html=True)
    
    # Main heading
    st.markdown('<h1 class="landing-title">Neural Powered Finance Unlocked</h1>', unsafe_allow_html=True)
    
    # Subtitle
    st.markdown('<p class="landing-subtitle">Financial Superintelligence in your reach</p>', unsafe_allow_html=True)
    
    # Add sentiment gauge widget
    sentiment_gauge = create_sentiment_gauge(st.session_state.sentiment_value)
    st.markdown(sentiment_gauge, unsafe_allow_html=True)
    
    # Add sentiment trend chart
    sentiment_trend = create_sentiment_trend(st.session_state.sentiment_trend)
    st.markdown(sentiment_trend, unsafe_allow_html=True)
    
    # Add feature highlights
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin: 0 auto 30px auto; max-width: 600px; text-align: center;">
        <div style="flex: 1; padding: 0 10px;">
            <div style="font-size: 2rem; color: #7B68EE; margin-bottom: 5px;">✨</div>
            <div style="font-weight: 600; margin-bottom: 5px; color: #E0E0E0;">AI-Powered Insights</div>
            <div style="font-size: 0.9rem; color: #ADB3C9;">Advanced sentiment analysis using neural networks</div>
        </div>
        <div style="flex: 1; padding: 0 10px;">
            <div style="font-size: 2rem; color: #7B68EE; margin-bottom: 5px;">📊</div>
            <div style="font-weight: 600; margin-bottom: 5px; color: #E0E0E0;">Real-time Data</div>
            <div style="font-size: 0.9rem; color: #ADB3C9;">Live market data and predictive analytics</div>
        </div>
        <div style="flex: 1; padding: 0 10px;">
            <div style="font-size: 2rem; color: #7B68EE; margin-bottom: 5px;">🔒</div>
            <div style="font-weight: 600; margin-bottom: 5px; color: #E0E0E0;">Premium Insights</div>
            <div style="font-size: 0.9rem; color: #ADB3C9;">Exclusive analysis for subscribers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Email form
    st.markdown('<div class="landing-form">', unsafe_allow_html=True)
    
    # Email input field with unique key
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    unique_email_key = f"landing_email_input_{int(time.time())}"
    email = st.text_input("Email address", 
                         key=unique_email_key, 
                         placeholder="Enter your email",
                         label_visibility="collapsed")
    st.session_state.email_input = email  # Sync with the expected session state variable
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show error message if email is invalid
    if not st.session_state.valid_email:
        st.markdown('<div class="error-message" style="display: block;">Please enter a valid email address</div>', unsafe_allow_html=True)
    
    # Submit button with unique key - Get Started comes first now
    unique_submit_key = f"submit_button_{int(time.time())}"
    if st.button("🚀 Get Started", key=unique_submit_key, type="primary", use_container_width=True):
        # Override the on_click approach with direct implementation
        email = st.session_state.email_input
        if not email or not isinstance(email, str) or len(email.strip()) == 0:
            st.session_state.valid_email = False
            print("Email validation failed: Empty email")
        else:
            pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if re.match(pattern, email):
                st.session_state.valid_email = True
                st.session_state.submitted = True
                st.session_state.email_prefill = email
                st.session_state.redirect_email = email
                st.session_state.show_auth = True
                print(f"Email validation successful: {email}")
                st.rerun()
            else:
                st.session_state.valid_email = False
                print(f"Email validation failed: Invalid format: {email}")
        
    st.markdown('<div style="margin-bottom: 15px;"></div>', unsafe_allow_html=True)
    
    # Add AI Assistant option
    ai_assistant_col1, ai_assistant_col2 = st.columns([3, 1])
    with ai_assistant_col1:
        st.markdown('<div style="display: flex; align-items: center;">', unsafe_allow_html=True)
        st.markdown('<span style="margin-right: 10px;">✨ Try our AI Assistant</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ai_assistant_col2:
        unique_assistant_key = f"ai_assistant_button_{int(time.time())}"
        if st.button("AI Assistant", key=unique_assistant_key, use_container_width=True):
            # Set session state variables for AI Assistant
            st.session_state["show_ai_assistant"] = True
            print("AI Assistant button clicked! Setting show_ai_assistant=True")
            st.rerun()
    
    # Add options section with demo and login
    st.markdown('<div style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        unique_demo_key = f"try_demo_button_{int(time.time())}"
        if st.button("👀 Try a Demo", key=unique_demo_key, use_container_width=True):
            # Set flag and print a debug message
            st.session_state["show_demo"] = True
            st.session_state.show_demo = True  # Set both formats for compatibility
            print("Demo button clicked! Setting show_demo=True")
            # Force redirect to main.py with demo mode
            st.query_params['demo'] = 'true'
            st.rerun()
            
    with col2:
        unique_login_key = f"login_button_{int(time.time())}"
        if st.button("🔑 Login", key=unique_login_key, use_container_width=True):
            # Direct implementation instead of using on_click
            st.session_state["show_auth"] = True
            print("Login button clicked! Setting show_auth=True")
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add marketplace advantages
    st.markdown("""
    <div style="max-width: 600px; margin: 40px auto 0 auto; text-align: center;">
        <h3 style="color: #e0e0e0; margin-bottom: 15px; font-size: 1.3rem;">Why Choose Neufin</h3>
        <div style="display: flex; justify-content: space-between; text-align: left;">
            <div style="flex: 1; padding: 0 10px;">
                <div style="margin-bottom: 10px;">✅ <span style="font-weight: 500;">Real-time sentiment analysis</span></div>
                <div style="margin-bottom: 10px;">✅ <span style="font-weight: 500;">Advanced AI predictive models</span></div>
            </div>
            <div style="flex: 1; padding: 0 10px;">
                <div style="margin-bottom: 10px;">✅ <span style="font-weight: 500;">Sector performance tracking</span></div>
                <div style="margin-bottom: 10px;">✅ <span style="font-weight: 500;">Personalized investment insights</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="landing-footer">Neufin OÜ · A Unit of Ctech Ventures · info@ctechventure.com</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)