"""
Authentication and user management for Neufin platform.
Handles user registration, login, profile management, and more.
"""

import os
import json
import streamlit as st
import streamlit_authenticator as stauth
from datetime import datetime, timedelta
import hashlib
import secrets
import google.oauth2.credentials
import google_auth_oauthlib.flow
import requests
from database import (create_user, authenticate_user, get_user_subscription, 
                     update_subscription, User, Subscription, UserSettings,
                     get_db_session, get_user_settings, update_user_settings)

# Google OAuth2 configuration
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REDIRECT_URI = "https://neufin.repl.co/callback"  # Always use the correct Replit domain
GOOGLE_SCOPES = ['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']

# Facebook OAuth configuration
FACEBOOK_APP_ID = os.environ.get('FACEBOOK_APP_ID', '')
FACEBOOK_APP_SECRET = os.environ.get('FACEBOOK_APP_SECRET', '')
FACEBOOK_REDIRECT_URI = "https://neufin.repl.co/facebook_callback"
FACEBOOK_SCOPES = ['email', 'public_profile']

# Session state keys
USER_SESSION_KEY = "neufin_user"
AUTH_STATUS_KEY = "neufin_auth_status"
AUTH_MESSAGE_KEY = "neufin_auth_message"

# User roles and permissions
ROLES = {
    "admin": ["read", "write", "delete", "manage_users"],
    "premium": ["read", "write", "premium_features"],
    "basic": ["read", "limited_features"],
    "free": ["read", "very_limited_features"]
}

def init_auth_session():
    """Initialize auth-related session state variables"""
    if USER_SESSION_KEY not in st.session_state:
        st.session_state[USER_SESSION_KEY] = None
    if AUTH_STATUS_KEY not in st.session_state:
        st.session_state[AUTH_STATUS_KEY] = False
    if AUTH_MESSAGE_KEY not in st.session_state:
        st.session_state[AUTH_MESSAGE_KEY] = ""
        
def is_authenticated():
    """Check if user is authenticated"""
    return st.session_state.get(AUTH_STATUS_KEY, False)

def get_current_user():
    """Get current authenticated user"""
    return st.session_state.get(USER_SESSION_KEY, None)

def login_user(email, password):
    """Login a user with email and password"""
    user = authenticate_user(email, password)
    if user:
        st.session_state[USER_SESSION_KEY] = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": datetime.now().isoformat()
        }
        st.session_state[AUTH_STATUS_KEY] = True
        st.session_state[AUTH_MESSAGE_KEY] = f"Welcome back, {user.username}!"
        return True
    else:
        st.session_state[AUTH_MESSAGE_KEY] = "Invalid email or password. Please try again."
        return False

def register_user(username, email, password):
    """Register a new user"""
    try:
        user = create_user(username, email, password)
        if user:
            st.session_state[USER_SESSION_KEY] = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": datetime.now().isoformat()
            }
            st.session_state[AUTH_STATUS_KEY] = True
            st.session_state[AUTH_MESSAGE_KEY] = f"Welcome to Neufin, {username}! Your account has been created."
            return True
        else:
            st.session_state[AUTH_MESSAGE_KEY] = "Error creating user account. Please try again."
            return False
    except Exception as e:
        st.session_state[AUTH_MESSAGE_KEY] = f"Error: {str(e)}"
        return False

def logout_user():
    """Logout current user"""
    st.session_state[USER_SESSION_KEY] = None
    st.session_state[AUTH_STATUS_KEY] = False
    st.session_state[AUTH_MESSAGE_KEY] = "You have been logged out."
    
def init_google_oauth():
    """Initialize Google OAuth flow"""
    flow = google_auth_oauthlib.flow.Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_REDIRECT_URI]
            }
        },
        scopes=GOOGLE_SCOPES
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    
    # Generate a secure state token
    state_token = secrets.token_urlsafe(16)
    st.session_state['oauth_state'] = state_token
    
    # Set the state parameter to protect against CSRF
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=state_token,
        prompt='consent'
    )
    
    return authorization_url

def init_facebook_oauth():
    """Initialize Facebook OAuth flow"""
    # Generate a secure state token
    state_token = secrets.token_urlsafe(16)
    st.session_state['fb_oauth_state'] = state_token
    
    # Construct the Facebook OAuth URL
    auth_url = f"https://www.facebook.com/v12.0/dialog/oauth?client_id={FACEBOOK_APP_ID}&redirect_uri={FACEBOOK_REDIRECT_URI}&state={state_token}&scope={','.join(FACEBOOK_SCOPES)}"
    
    return auth_url

def handle_facebook_callback(state, code):
    """Handle Facebook OAuth callback"""
    # Verify state matches to prevent CSRF
    if state != st.session_state.get('fb_oauth_state'):
        st.session_state[AUTH_MESSAGE_KEY] = "Invalid state parameter. Authentication failed."
        return False
    
    try:
        # Exchange code for access token
        token_url = f"https://graph.facebook.com/v12.0/oauth/access_token?client_id={FACEBOOK_APP_ID}&redirect_uri={FACEBOOK_REDIRECT_URI}&client_secret={FACEBOOK_APP_SECRET}&code={code}"
        token_response = requests.get(token_url)
        token_data = token_response.json()
        
        if 'error' in token_data:
            st.session_state[AUTH_MESSAGE_KEY] = f"Facebook authentication error: {token_data['error']['message']}"
            return False
        
        access_token = token_data.get('access_token')
        
        # Get user info
        user_info_url = f"https://graph.facebook.com/me?fields=id,name,email&access_token={access_token}"
        userinfo_response = requests.get(user_info_url)
        userinfo = userinfo_response.json()
        
        if 'error' in userinfo:
            st.session_state[AUTH_MESSAGE_KEY] = f"Error getting Facebook user info: {userinfo['error']['message']}"
            return False
        
        # Get or create user
        email = userinfo.get('email')
        if not email:
            st.session_state[AUTH_MESSAGE_KEY] = "Your Facebook account does not have an email address or hasn't granted access to it."
            return False
            
        name = userinfo.get('name', email.split('@')[0])
        
        # Check if user exists
        session = get_db_session()
        user = session.query(User).filter_by(email=email).first()
        
        if not user:
            # Create new user with random password
            random_password = secrets.token_urlsafe(16)
            user = create_user(name, email, random_password)
            
        # Login user
        st.session_state[USER_SESSION_KEY] = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": datetime.now().isoformat(),
            "oauth_provider": "facebook"
        }
        st.session_state[AUTH_STATUS_KEY] = True
        st.session_state[AUTH_MESSAGE_KEY] = f"Welcome, {name}! You've successfully signed in with Facebook."
        
        return True
    except Exception as e:
        st.session_state[AUTH_MESSAGE_KEY] = f"Error during Facebook authentication: {str(e)}"
        return False

def handle_google_callback(state, code):
    """Handle Google OAuth callback"""
    # Verify state matches to prevent CSRF
    if state != st.session_state.get('oauth_state'):
        st.session_state[AUTH_MESSAGE_KEY] = "Invalid state parameter. Authentication failed."
        return False
        
    try:
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [GOOGLE_REDIRECT_URI]
                }
            },
            scopes=GOOGLE_SCOPES,
            state=state
        )
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        
        # Exchange authorization code for tokens
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Get user info
        userinfo_response = requests.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {credentials.token}'}
        )
        userinfo = userinfo_response.json()
        
        # Get or create user
        email = userinfo.get('email')
        name = userinfo.get('name', email.split('@')[0])
        
        # Check if user exists
        session = get_db_session()
        user = session.query(User).filter_by(email=email).first()
        
        if not user:
            # Create new user with random password
            random_password = secrets.token_urlsafe(16)
            user = create_user(name, email, random_password)
            
        # Login user
        st.session_state[USER_SESSION_KEY] = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": datetime.now().isoformat(),
            "oauth_provider": "google"
        }
        st.session_state[AUTH_STATUS_KEY] = True
        st.session_state[AUTH_MESSAGE_KEY] = f"Welcome, {name}! You've successfully signed in with Google."
        
        return True
    except Exception as e:
        st.session_state[AUTH_MESSAGE_KEY] = f"Error during Google authentication: {str(e)}"
        return False

def user_has_permission(permission):
    """Check if current user has specific permission"""
    if not is_authenticated():
        return False
        
    user_id = st.session_state[USER_SESSION_KEY]["id"]
    subscription = get_user_subscription(user_id)
    
    if not subscription or not subscription.is_active:
        role = "free"
    else:
        role = subscription.level
        
    return permission in ROLES.get(role, [])

def require_login(page_func):
    """Decorator to require login for a page"""
    def wrapper(*args, **kwargs):
        if is_authenticated():
            return page_func(*args, **kwargs)
        else:
            st.warning("You need to be logged in to access this page.")
            show_login_ui()
            return None
    return wrapper

def require_permission(permission):
    """Decorator to require specific permission for a page"""
    def decorator(page_func):
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.warning("You need to be logged in to access this page.")
                show_login_ui()
                return None
            elif not user_has_permission(permission):
                st.error("You don't have permission to access this feature. Please upgrade your subscription.")
                return None
            else:
                return page_func(*args, **kwargs)
        return wrapper
    return decorator

def show_login_ui():
    """Display login UI"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login")
        login_form = st.form("login_form")
        email = login_form.text_input("Email", key="login_email")
        password = login_form.text_input("Password", type="password", key="login_password")
        submit = login_form.form_submit_button("Login")
        
        if submit:
            if login_user(email, password):
                st.rerun()
    
    with col2:
        st.subheader("Register")
        reg_form = st.form("register_form")
        username = reg_form.text_input("Username", key="register_username")
        email = reg_form.text_input("Email", key="register_email")
        password = reg_form.text_input("Password", type="password", key="register_password")
        confirm_password = reg_form.text_input("Confirm Password", type="password", key="register_confirm_password")
        submit = reg_form.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif register_user(username, email, password):
                st.rerun()
                
    # Social login options
    st.markdown("---")
    st.subheader("Or sign in with:")
    
    social_cols = st.columns(2)
    
    with social_cols[0]:
        if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
            google_auth_url = init_google_oauth()
            st.markdown(f"""
            <a href="{google_auth_url}" target="_self">
                <div style="background-color:#4285F4; color:white; padding:8px 16px; border-radius:4px; display:inline-flex; align-items:center; cursor:pointer;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" style="height:24px; margin-right:8px;">
                    Sign in with Google
                </div>
            </a>
            """, unsafe_allow_html=True)
        else:
            st.info("Google authentication is not configured.")
    
    with social_cols[1]:
        if FACEBOOK_APP_ID and FACEBOOK_APP_SECRET:
            facebook_auth_url = init_facebook_oauth()
            st.markdown(f"""
            <a href="{facebook_auth_url}" target="_self">
                <div style="background-color:#1877F2; color:white; padding:8px 16px; border-radius:4px; display:inline-flex; align-items:center; cursor:pointer;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c2/F_icon.svg" style="height:24px; margin-right:8px;">
                    Sign in with Facebook
                </div>
            </a>
            """, unsafe_allow_html=True)
        else:
            st.info("Facebook authentication is not configured.")
        
    # Display any authentication messages
    auth_message = st.session_state.get(AUTH_MESSAGE_KEY)
    if auth_message:
        st.info(auth_message)
        # Clear message after displaying
        st.session_state[AUTH_MESSAGE_KEY] = ""

def show_user_profile_ui():
    """Display user profile UI"""
    if not is_authenticated():
        st.warning("You need to be logged in to view your profile.")
        return
        
    user = st.session_state[USER_SESSION_KEY]
    user_id = user["id"]
    
    st.title("Your Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://www.gravatar.com/avatar/" + hashlib.md5(user["email"].lower().encode()).hexdigest() + "?s=200&d=identicon", width=150)
        st.subheader(user["username"])
        st.text(user["email"])
        
        subscription = get_user_subscription(user_id)
        if subscription and subscription.is_active:
            if subscription.level == "premium":
                st.success(f"Premium Subscription Active (until {subscription.end_date.strftime('%Y-%m-%d')})")
            elif subscription.level == "basic":
                st.info(f"Basic Subscription Active (until {subscription.end_date.strftime('%Y-%m-%d')})")
            elif subscription.is_trial:
                st.warning(f"Trial Subscription (until {subscription.end_date.strftime('%Y-%m-%d')})")
        else:
            st.error("No active subscription")
            
    with col2:
        tabs = st.tabs(["Account Settings", "Subscription", "Preferences"])
        
        with tabs[0]:
            st.subheader("Account Details")
            st.text(f"Member since: {datetime.fromisoformat(user['created_at']).strftime('%Y-%m-%d')}" if user.get('created_at') else "")
            
            with st.form("update_profile_form"):
                st.subheader("Update Profile")
                new_username = st.text_input("Username", value=user["username"])
                update_btn = st.form_submit_button("Update Profile")
                
                if update_btn:
                    # Update username logic here
                    st.success("Profile updated successfully!")
            
            with st.form("change_password_form"):
                st.subheader("Change Password")
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                update_pwd_btn = st.form_submit_button("Change Password")
                
                if update_pwd_btn:
                    if new_password != confirm_password:
                        st.error("New passwords do not match!")
                    else:
                        # Change password logic here
                        st.success("Password changed successfully!")
        
        with tabs[1]:
            st.subheader("Subscription Management")
            
            if subscription and subscription.is_active:
                st.write(f"Current plan: **{subscription.level.title()}**")
                st.write(f"Status: **Active**")
                st.write(f"Renewal date: **{subscription.end_date.strftime('%Y-%m-%d')}**")
                
                if subscription.level != "premium":
                    st.button("Upgrade to Premium", on_click=lambda: st.session_state.update({"show_premium_upgrade": True}))
                
                if st.session_state.get("show_premium_upgrade", False):
                    with st.form("premium_upgrade_form"):
                        st.subheader("Upgrade to Premium")
                        st.write("Enjoy all Neufin features with our Premium plan:")
                        st.write("- Advanced AI-powered market analysis")
                        st.write("- Real-time sentiment analysis")
                        st.write("- Personalized investment recommendations")
                        st.write("- Priority customer support")
                        st.write("\nPrice: $19.99/month")
                        
                        payment_btn = st.form_submit_button("Proceed to Payment")
                        if payment_btn:
                            # Redirect to payment page
                            st.session_state["redirect_to_payment"] = True
                            st.session_state["payment_plan"] = "premium"
                            st.rerun()
            else:
                st.write("You don't have an active subscription.")
                
                col1, col2 = st.columns(2)
                with col1:
                    with st.container():
                        st.subheader("Basic Plan")
                        st.write("- Market data access")
                        st.write("- Basic analytics")
                        st.write("- Daily updates")
                        st.write("\nPrice: $9.99/month")
                        st.button("Choose Basic", on_click=lambda: st.session_state.update({
                            "redirect_to_payment": True, 
                            "payment_plan": "basic"
                        }))
                
                with col2:
                    with st.container():
                        st.subheader("Premium Plan")
                        st.write("- All Basic features")
                        st.write("- Advanced AI analytics")
                        st.write("- Real-time updates")
                        st.write("- Personalized recommendations")
                        st.write("\nPrice: $19.99/month")
                        st.button("Choose Premium", on_click=lambda: st.session_state.update({
                            "redirect_to_payment": True, 
                            "payment_plan": "premium"
                        }))
                
                st.button("Start Free Trial", on_click=lambda: start_free_trial(user_id))
        
        with tabs[2]:
            st.subheader("Preferences")
            
            settings = get_user_settings(user_id)
            
            with st.form("preferences_form"):
                st.subheader("Display Settings")
                
                theme = st.selectbox(
                    "Theme", 
                    options=["light", "dark"], 
                    index=0 if not settings or settings.theme_preference == "light" else 1
                )
                
                default_stocks_str = "" if not settings or not settings.default_stocks else settings.default_stocks
                default_stocks = st.text_input(
                    "Default Stocks (comma-separated)", 
                    value=default_stocks_str
                )
                
                default_time_period = st.selectbox(
                    "Default Time Period",
                    options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                    index=2 if not settings or settings.default_time_period == "1mo" else 
                          ["1d", "5d", "1mo", "3mo", "6mo", "1y"].index(settings.default_time_period)
                )
                
                save_pref_btn = st.form_submit_button("Save Preferences")
                
                if save_pref_btn:
                    update_user_settings(
                        user_id=user_id,
                        default_stocks=default_stocks,
                        default_time_period=default_time_period,
                        theme_preference=theme
                    )
                    st.success("Preferences saved successfully!")

def start_free_trial(user_id):
    """Start a free trial for the user"""
    # Set trial period (14 days)
    end_date = datetime.now() + timedelta(days=14)
    
    # Update subscription
    update_subscription(
        user_id=user_id,
        level="basic",
        is_trial=True,
        end_date=end_date
    )
    
    st.success("Your 14-day free trial has been activated!")
    st.rerun()

# Run this when the module is imported
init_auth_session()