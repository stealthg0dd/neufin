import streamlit as st
from datetime import datetime, timedelta
import json
import uuid

def initialize_subscription_state():
    """Initialize session state variables for subscription management"""
    if 'user_logged_in' not in st.session_state:
        st.session_state.user_logged_in = False
    
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    
    if 'subscription_level' not in st.session_state:
        st.session_state.subscription_level = "free"  # Options: free, basic, premium
    
    if 'subscription_expiry' not in st.session_state:
        st.session_state.subscription_expiry = None
    
    if 'trial_active' not in st.session_state:
        st.session_state.trial_active = False
    
    if 'trial_end_date' not in st.session_state:
        st.session_state.trial_end_date = None
    
    if 'login_error' not in st.session_state:
        st.session_state.login_error = ""
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'payment_processing' not in st.session_state:
        st.session_state.payment_processing = False
    
    if 'selected_plan' not in st.session_state:
        st.session_state.selected_plan = None

def show_login_form():
    """Display login form for users"""
    if not st.session_state.user_logged_in:
        with st.sidebar.form("login_form"):
            st.subheader("Login / Sign Up")
            
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                login_submitted = st.form_submit_button("Login")
            
            with col2:
                signup_submitted = st.form_submit_button("Sign Up")
            
            if st.session_state.login_error:
                st.error(st.session_state.login_error)
                st.session_state.login_error = ""
            
            if login_submitted:
                # In a real app, this would validate against a database
                if email and password:
                    # Simulate successful login
                    st.session_state.user_logged_in = True
                    st.session_state.user_email = email
                    st.session_state.user_name = email.split('@')[0]
                    st.session_state.user_id = str(uuid.uuid4())
                    st.session_state.subscription_level = "free"
                    
                    # Check if user should get a trial
                    activate_premium_trial()
                    
                    st.rerun()
                else:
                    st.session_state.login_error = "Please enter both email and password"
            
            elif signup_submitted:
                if email and password:
                    # Simulate successful signup
                    st.session_state.user_logged_in = True
                    st.session_state.user_email = email
                    st.session_state.user_name = email.split('@')[0]
                    st.session_state.user_id = str(uuid.uuid4())
                    st.session_state.subscription_level = "free"
                    
                    # Activate premium trial for new users
                    activate_premium_trial()
                    
                    st.rerun()
                else:
                    st.session_state.login_error = "Please enter both email and password"
    else:
        # Show user info
        st.sidebar.markdown(f"**Welcome, {st.session_state.user_name}!**")
        
        # Show subscription status
        subscription_label = {
            "free": "Free Tier",
            "basic": "Basic Subscription",
            "premium": "Premium Subscription"
        }.get(st.session_state.subscription_level, "Unknown")
        
        status_color = {
            "free": "gray",
            "basic": "blue",
            "premium": "green"
        }.get(st.session_state.subscription_level, "gray")
        
        st.sidebar.markdown(f"""
        <div style='background-color:{status_color}20; padding:10px; border-radius:5px;'>
        <strong>Current Plan:</strong> {subscription_label}
        </div>
        """, unsafe_allow_html=True)
        
        # Show trial information if active
        if st.session_state.trial_active and st.session_state.trial_end_date:
            days_left = (st.session_state.trial_end_date - datetime.now()).days
            
            if days_left > 0:
                st.sidebar.markdown(f"""
                <div style='background-color:#FFC10720; padding:10px; border-radius:5px; margin-top:10px;'>
                <strong>Premium Trial:</strong> {days_left} days remaining
                </div>
                """, unsafe_allow_html=True)
        
        # Logout button
        if st.sidebar.button("Logout"):
            for key in ['user_logged_in', 'user_name', 'user_email', 'subscription_level', 
                        'subscription_expiry', 'trial_active', 'trial_end_date', 'user_id']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()

def show_subscription_options():
    """Display subscription options for users"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Subscription Options")
    
    # Define subscription tiers
    subscription_tiers = {
        "free": {
            "name": "Free Tier",
            "price": "$0/month",
            "features": [
                "Basic market sentiment analysis",
                "Limited historical data (7 days)",
                "Standard charts and visualizations"
            ]
        },
        "basic": {
            "name": "Basic Plan",
            "price": "$9.99/month",
            "features": [
                "Advanced market sentiment analysis",
                "Investment recommendations",
                "Sector insights",
                "Extended historical data (30 days)",
                "Priority support"
            ]
        },
        "premium": {
            "name": "Premium Plan",
            "price": "$19.99/month",
            "features": [
                "All Basic Plan features",
                "AI-powered detailed stock analysis",
                "Global trade impact analysis",
                "Comprehensive sector reports",
                "Extended historical data (90 days)",
                "Premium customer support"
            ]
        }
    }
    
    # Only show upgrade options
    for tier, details in subscription_tiers.items():
        # Skip current tier or lower tiers
        if tier == "free" or subscription_level_value(tier) <= subscription_level_value(st.session_state.subscription_level):
            continue
        
        with st.sidebar.expander(f"✨ {details['name']} - {details['price']}"):
            for feature in details['features']:
                st.markdown(f"- {feature}")
            
            if st.button(f"Upgrade to {details['name']}", key=f"upgrade_{tier}"):
                st.session_state.selected_plan = tier
                show_payment_modal(tier)
    
    # Show downgrade option if on a paid plan
    if st.session_state.subscription_level != "free":
        st.sidebar.markdown("---")
        if st.sidebar.button("Downgrade to Free Tier"):
            st.session_state.subscription_level = "free"
            st.sidebar.success("Subscription downgraded to Free Tier")
            st.rerun()

def show_payment_modal(plan_type):
    """Show a simulated payment modal for subscription upgrade"""
    st.session_state.payment_processing = True
    st.session_state.selected_plan = plan_type

def process_payment():
    """Process payment for subscription upgrade (simulated)"""
    if st.session_state.payment_processing and st.session_state.selected_plan:
        plan = st.session_state.selected_plan
        
        # Create payment form
        with st.sidebar.form("payment_form"):
            st.subheader(f"Upgrade to {plan.title()} Plan")
            
            # Payment details
            st.text_input("Card Number", value="4242 4242 4242 4242")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Expiry", value="12/25")
            with col2:
                st.text_input("CVC", value="123")
            
            st.text_input("Cardholder Name", value=st.session_state.user_name)
            
            # Subscription price display
            price = "$9.99/month" if plan == "basic" else "$19.99/month"
            st.markdown(f"**Total: {price}**")
            
            # Submit payment
            if st.form_submit_button("Complete Payment"):
                # Process would happen here in a real app
                st.session_state.subscription_level = plan
                st.session_state.subscription_expiry = datetime.now() + timedelta(days=30)
                st.session_state.payment_processing = False
                st.session_state.selected_plan = None
                
                # Deactivate trial if active
                st.session_state.trial_active = False
                st.session_state.trial_end_date = None
                
                st.sidebar.success(f"🎉 Successfully upgraded to {plan.title()} Plan!")
                st.rerun()
        
        # Cancel button
        if st.sidebar.button("Cancel"):
            st.session_state.payment_processing = False
            st.session_state.selected_plan = None
            st.rerun()

def activate_premium_trial():
    """Activate premium trial for new users"""
    # Only activate if user is on free tier and doesn't already have an active trial
    if (st.session_state.subscription_level == "free" and 
        not st.session_state.trial_active and 
        not st.session_state.trial_end_date):
        
        st.session_state.trial_active = True
        st.session_state.trial_end_date = datetime.now() + timedelta(days=14)  # 14-day trial

def check_feature_access(feature_level):
    """
    Check if user has access to a particular feature
    
    Args:
        feature_level (str): 'free', 'basic', or 'premium'
    
    Returns:
        bool: Whether user has access
    """
    # Trial users get access to premium features
    if st.session_state.trial_active and st.session_state.trial_end_date and st.session_state.trial_end_date > datetime.now():
        return True
    
    user_level = st.session_state.subscription_level
    
    # Convert levels to numeric for comparison
    user_value = subscription_level_value(user_level)
    feature_value = subscription_level_value(feature_level)
    
    return user_value >= feature_value

def subscription_level_value(level):
    """Convert subscription level to numeric value for comparison"""
    levels = {
        "free": 0,
        "basic": 1,
        "premium": 2
    }
    return levels.get(level, 0)

def show_upgrade_prompt(feature_name, required_plan):
    """Show an upgrade prompt for premium features"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"⭐ Upgrade to the {required_plan.title()} Plan to unlock {feature_name}")
    
    with col2:
        if st.button("Upgrade Now"):
            st.session_state.selected_plan = required_plan
            show_payment_modal(required_plan)
            st.rerun()

def check_trial_status():
    """Check and update trial status"""
    if (st.session_state.trial_active and 
        st.session_state.trial_end_date and 
        st.session_state.trial_end_date < datetime.now()):
        
        # Trial has expired
        st.session_state.trial_active = False
        st.session_state.trial_end_date = None