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
        margin-bottom: 40px;
    }
    
    /* Typography */
    .landing-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7B68EE, #5D4FD3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        letter-spacing: -0.03em;
        line-height: 1.1;
        text-align: center;
    }
    
    .landing-subtitle {
        font-size: 1.5rem;
        color: #ADB3C9;
        margin-bottom: 3rem;
        max-width: 750px;
        text-align: center;
        letter-spacing: -0.01em;
    }
    
    /* Email input styling */
    .landing-form {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        max-width: 500px;
        margin-bottom: 1.5rem;
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
        background-color: rgba(30, 30, 40, 0.6);
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
        margin-top: 4rem;
        color: #6c757d;
        font-size: 0.9rem;
        text-align: center;
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
        position: absolute;
        z-index: -1;
        opacity: 0.2;
    }
    
    .decoration-1 {
        top: 15%;
        left: 10%;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(123, 104, 238, 0.8) 0%, rgba(123, 104, 238, 0) 70%);
        filter: blur(50px);
    }
    
    .decoration-2 {
        bottom: 20%;
        right: 15%;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(93, 79, 211, 0.8) 0%, rgba(93, 79, 211, 0) 70%);
        filter: blur(60px);
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
</style>
""", unsafe_allow_html=True)

# Function to validate email
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# Create session state for handling form data
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'valid_email' not in st.session_state:
    st.session_state.valid_email = True

# Handle form submission
def handle_submit():
    email = st.session_state.email
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

# Create the landing page layout
def landing_page():
    # Decorative elements
    st.markdown('<div class="decoration decoration-1"></div>', unsafe_allow_html=True)
    st.markdown('<div class="decoration decoration-2"></div>', unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    
    # Logo
    st.markdown(f'<img src="data:image/png;base64,{open("neufin_new_logo_base64.txt", "r").read()}" class="landing-logo" alt="Neufin AI">', unsafe_allow_html=True)
    
    # Main heading
    st.markdown('<h1 class="landing-title">Neural Powered Finance Unlocked</h1>', unsafe_allow_html=True)
    
    # Subtitle
    st.markdown('<p class="landing-subtitle">Apply in 10 minutes for an account that transforms how you operate.</p>', unsafe_allow_html=True)
    
    # Email form
    st.markdown('<div class="landing-form">', unsafe_allow_html=True)
    
    # Email input field
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    email = st.text_input("Email address", 
                         key="email", 
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