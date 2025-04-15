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
    /* Base styling */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0F1117 0%, #171924 100%);
        color: #E0E0E0;
        line-height: 1.6;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F1117 0%, #171924 100%);
    }
    
    /* Hide default elements we don't need on the landing page */
    #MainMenu, header, footer {
        visibility: hidden;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    
    /* Landing page container */
    .landing-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 90vh;
        padding: 0 20px;
        text-align: center;
    }
    
    /* Logo styling */
    .landing-logo {
        height: 120px;
        margin-bottom: 30px;
    }
    
    /* Typography */
    .landing-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7B68EE, #5D4FD3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -0.03em;
        line-height: 1.1;
        text-align: center;
    }
    
    .landing-subtitle {
        font-size: 1.5rem;
        color: #ADB3C9;
        margin-bottom: 2rem;
        max-width: 750px;
        text-align: center;
        letter-spacing: -0.01em;
    }
    
    /* Cards and feature panels */
    .feature-card {
        background: rgba(30, 30, 40, 0.4);
        border-radius: 10px;
        padding: 30px;
        margin-bottom: 30px;
        border: 1px solid rgba(123, 104, 238, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        border-color: rgba(123, 104, 238, 0.4);
    }
    
    /* Email input styling */
    .landing-form {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        max-width: 500px;
        margin-bottom: 1.5rem;
        background: rgba(30, 30, 40, 0.4);
        padding: 30px;
        border-radius: 12px;
        border: 1px solid rgba(123, 104, 238, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .input-container {
        position: relative;
        width: 100%;
        margin-bottom: 1.5rem;
    }
    
    .landing-input {
        width: 100%;
        padding: 16px 20px;
        font-size: 16px;
        background-color: rgba(20, 20, 30, 0.6);
        border: 1px solid rgba(123, 104, 238, 0.4);
        border-radius: 8px;
        color: #E0E0E0;
        transition: all 0.3s ease;
    }
    
    .landing-input:focus {
        border-color: #7B68EE;
        outline: none;
        box-shadow: 0 0 0 3px rgba(123, 104, 238, 0.3);
    }
    
    .landing-input::placeholder {
        color: #6c757d;
    }
    
    /* Button styling */
    .landing-button {
        display: inline-block;
        padding: 14px 32px;
        font-size: 16px;
        font-weight: 600;
        color: white;
        background: linear-gradient(90deg, #7B68EE, #5D4FD3);
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        margin-top: 5px;
        width: 100%;
    }
    
    .landing-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(123, 104, 238, 0.4);
    }
    
    /* Footer */
    .landing-footer {
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.9rem;
        text-align: center;
        padding: 15px;
        border-top: 1px solid rgba(123, 104, 238, 0.1);
        width: 100%;
        max-width: 800px;
    }
    
    /* Pulsing animation for button */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(123, 104, 238, 0.5);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(123, 104, 238, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(123, 104, 238, 0);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom shape decorations */
    .decoration {
        position: fixed;
        z-index: -1;
        opacity: 0.3;
    }
    
    .decoration-1 {
        top: 10%;
        left: 5%;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(123, 104, 238, 0.8) 0%, rgba(123, 104, 238, 0) 70%);
        filter: blur(70px);
    }
    
    .decoration-2 {
        bottom: 10%;
        right: 5%;
        width: 400px;
        height: 400px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(93, 79, 211, 0.8) 0%, rgba(93, 79, 211, 0) 70%);
        filter: blur(80px);
    }
    
    /* Add a third decoration */
    .decoration-3 {
        top: 40%;
        right: 20%;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(60, 50, 180, 0.6) 0%, rgba(60, 50, 180, 0) 70%);
        filter: blur(60px);
    }
    
    /* Star background */
    .stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -2;
        background-image: radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
                        radial-gradient(2px 2px at 40px 70px, #fff, rgba(0,0,0,0)),
                        radial-gradient(1px 1px at 90px 40px, #fff, rgba(0,0,0,0)),
                        radial-gradient(1px 1px at 130px 80px, #fff, rgba(0,0,0,0)),
                        radial-gradient(2px 2px at 160px 120px, #fff, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 200px 200px;
        opacity: 0.05;
    }
    
    /* Success message */
    .success-message {
        background-color: rgba(0, 200, 83, 0.1);
        border-left: 4px solid #00C853;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
        color: #00C853;
        font-weight: 500;
        text-align: center;
        display: none;
    }
    
    /* Error message */
    .error-message {
        background-color: rgba(255, 82, 82, 0.1);
        border-left: 4px solid #FF5252;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
        color: #FF5252;
        font-weight: 500;
        text-align: center;
        display: none;
    }
    
    /* Panel for sentiment widgets */
    .sentiment-panel {
        background: rgba(30, 30, 40, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(123, 104, 238, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        max-width: 500px;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* Feature box styling */
    .feature-box {
        background: rgba(20, 20, 30, 0.6);
        border-radius: 10px;
        padding: 15px;
        transition: all 0.3s ease;
        border: 1px solid rgba(123, 104, 238, 0.1);
    }
    
    .feature-box:hover {
        background: rgba(30, 30, 45, 0.6);
        border-color: rgba(123, 104, 238, 0.3);
        transform: translateY(-5px);
    }
    
    /* Responsive styling */
    @media (max-width: 768px) {
        .landing-title {
            font-size: 3rem;
        }
        
        .landing-subtitle {
            font-size: 1.2rem;
        }
    }
    
    @media (max-width: 480px) {
        .landing-title {
            font-size: 2.5rem;
        }
        
        .landing-subtitle {
            font-size: 1rem;
        }
    }
    
    /* Style for the Streamlit default elements to match our design */
    .stTextInput > div > div > input {
        background-color: rgba(20, 20, 30, 0.6) !important;
        color: #E0E0E0 !important;
        border: 1px solid rgba(123, 104, 238, 0.4) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7B68EE !important;
        box-shadow: 0 0 0 3px rgba(123, 104, 238, 0.3) !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #7B68EE, #5D4FD3) !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(123, 104, 238, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to validate email
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# Initialize session state variables
# Note: we'll now rely on init_session_state() from simple_app.py for this
# to avoid duplicate initializations

# Handle form submission
def handle_submit():
    email = st.session_state.email_input
    if is_valid_email(email):
        st.session_state.submitted = True
        st.session_state.valid_email = True
        # Redirect to the main app with email parameter
        # In a production app, we would use a proper auth flow here
        st.query_params.redirect_to = "signup"
        st.query_params.email = email
        st.rerun()
    else:
        st.session_state.valid_email = False

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

# Create the landing page layout
def landing_page():
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
    st.markdown('<p class="landing-subtitle">Apply in 10 minutes for an account that transforms how you operate.</p>', unsafe_allow_html=True)
    
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
    
    # Email input field
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    email = st.text_input("Email address", 
                         key="email_input", 
                         placeholder="Enter your email",
                         label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show error message if email is invalid
    if not st.session_state.valid_email:
        st.markdown('<div class="error-message" style="display: block;">Please enter a valid email address</div>', unsafe_allow_html=True)
    
    # Submit button
    if st.button("🚀 Get Started", key="submit_button", on_click=handle_submit, type="primary", use_container_width=True):
        # The logic is handled in the handle_submit function
        pass
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="landing-footer">Neufin OÜ · A Unit of Ctech Ventures · info@ctechventure.com</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main entry point
landing_page()