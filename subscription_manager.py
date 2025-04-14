"""
Subscription management system for Neufin.
Handles subscription creation, updates, billing, and Stripe integration.
"""

import os
import stripe
import streamlit as st
from datetime import datetime, timedelta
from database import get_user_subscription, update_subscription

# Initialize Stripe with secret key
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

# Subscription plans
SUBSCRIPTION_PLANS = {
    "basic": {
        "name": "Basic Plan",
        "price_id": "price_basic",  # Replace with actual Stripe price ID
        "amount": 999,  # $9.99 in cents
        "interval": "month",
        "features": [
            "Market data access",
            "Basic financial news",
            "Daily market summaries"
        ]
    },
    "premium": {
        "name": "Premium Plan",
        "price_id": "price_premium",  # Replace with actual Stripe price ID
        "amount": 1999,  # $19.99 in cents
        "interval": "month",
        "features": [
            "Advanced AI market analysis",
            "Real-time sentiment tracking",
            "Personalized investment recommendations",
            "Sector performance analysis",
            "Global trade impact analysis",
            "Priority customer support"
        ]
    }
}

def create_checkout_session(user_id, plan_id):
    """
    Create a Stripe checkout session for subscription
    
    Args:
        user_id (int): User ID
        plan_id (str): Plan ID (basic or premium)
        
    Returns:
        str: Checkout session URL
    """
    if plan_id not in SUBSCRIPTION_PLANS:
        raise ValueError(f"Invalid plan ID: {plan_id}")
    
    plan = SUBSCRIPTION_PLANS[plan_id]
    
    try:
        # Create a product if it doesn't exist
        product = _ensure_product_exists(plan_id, plan)
        
        # Create a price for the product if it doesn't exist
        price = _ensure_price_exists(product.id, plan)
        
        # Create the checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price': price.id,
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url='http://localhost:5000/payment_success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='http://localhost:5000/payment_cancel',
            client_reference_id=str(user_id),
            metadata={
                'user_id': str(user_id),
                'plan_id': plan_id
            }
        )
        
        return checkout_session.url
    except Exception as e:
        st.error(f"Error creating checkout session: {str(e)}")
        return None

def _ensure_product_exists(plan_id, plan):
    """Ensure the Stripe product exists, create if not"""
    # First try to find existing product
    products = stripe.Product.list(limit=100)
    for product in products.data:
        if product.name == plan["name"]:
            return product
    
    # Create new product if not found
    return stripe.Product.create(
        name=plan["name"],
        description=f"Neufin {plan['name']} subscription",
        metadata={
            'plan_id': plan_id
        }
    )

def _ensure_price_exists(product_id, plan):
    """Ensure the Stripe price exists, create if not"""
    # First try to find existing price
    prices = stripe.Price.list(
        product=product_id,
        limit=100
    )
    
    for price in prices.data:
        if price.recurring.interval == plan["interval"] and price.unit_amount == plan["amount"]:
            return price
    
    # Create new price if not found
    return stripe.Price.create(
        product=product_id,
        unit_amount=plan["amount"],
        currency="usd",
        recurring={"interval": plan["interval"]},
        metadata={
            'plan_id': product_id
        }
    )

def handle_subscription_webhook(event):
    """
    Handle Stripe webhook events for subscription management
    
    Args:
        event (dict): Stripe webhook event
        
    Returns:
        bool: True if handled successfully, False otherwise
    """
    try:
        event_type = event['type']
        data = event['data']['object']
        
        if event_type == 'checkout.session.completed':
            # Customer completed checkout
            user_id = int(data['metadata']['user_id'])
            plan_id = data['metadata']['plan_id']
            payment_id = data['id']
            
            # Set subscription end date (1 month from now)
            end_date = datetime.now() + timedelta(days=30)
            
            # Update subscription in database
            update_subscription(
                user_id=user_id,
                level=plan_id,
                is_trial=False,
                end_date=end_date,
                payment_id=payment_id
            )
            return True
            
        elif event_type == 'invoice.payment_succeeded':
            # Subscription renewal payment succeeded
            subscription_id = data['subscription']
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Get customer ID
            customer_id = subscription.customer
            customer = stripe.Customer.retrieve(customer_id)
            
            # Get user_id from customer metadata
            user_id = int(customer.metadata.get('user_id', 0))
            if user_id:
                # Get plan ID from subscription
                plan_id = subscription.metadata.get('plan_id')
                
                # Set subscription end date (1 month from now)
                end_date = datetime.now() + timedelta(days=30)
                
                # Update subscription in database
                update_subscription(
                    user_id=user_id,
                    level=plan_id,
                    is_trial=False,
                    end_date=end_date,
                    payment_id=subscription_id
                )
                
            return True
            
        elif event_type == 'customer.subscription.deleted':
            # Subscription was canceled
            subscription_id = data['id']
            
            # Get customer ID
            customer_id = data['customer']
            customer = stripe.Customer.retrieve(customer_id)
            
            # Get user_id from customer metadata
            user_id = int(customer.metadata.get('user_id', 0))
            if user_id:
                # Update subscription status in database
                current_subscription = get_user_subscription(user_id)
                if current_subscription and current_subscription.payment_id == subscription_id:
                    update_subscription(
                        user_id=user_id,
                        level='free',
                        is_active=False
                    )
                    
            return True
            
        return False  # Event not handled
        
    except Exception as e:
        st.error(f"Error handling webhook: {str(e)}")
        return False

def show_payment_ui(user_id, plan="basic"):
    """
    Display payment UI for subscription purchase
    
    Args:
        user_id (int): User ID
        plan (str): Plan ID (basic or premium)
    """
    if plan not in SUBSCRIPTION_PLANS:
        st.error(f"Invalid plan selected: {plan}")
        return
        
    plan_details = SUBSCRIPTION_PLANS[plan]
    
    st.title("Subscribe to Neufin")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(plan_details["name"])
        st.subheader(f"${plan_details['amount']/100:.2f} per month")
        
        st.write("### Features:")
        for feature in plan_details["features"]:
            st.write(f"✓ {feature}")
            
        st.write("### Payment Details")
        st.write("Secure payment processing powered by Stripe.")
        st.write("You can cancel your subscription at any time.")
        
    with col2:
        st.write("### Order Summary")
        st.write(f"Plan: **{plan_details['name']}**")
        st.write(f"Price: **${plan_details['amount']/100:.2f}/month**")
        st.write(f"Billing: **Monthly**")
        
        checkout_url = create_checkout_session(user_id, plan)
        
        if checkout_url:
            st.markdown(f'''
            <a href="{checkout_url}" target="_self">
                <button style="background-color:#7B68EE; color:white; border:none; 
                               padding:10px 20px; border-radius:4px; cursor:pointer;
                               width:100%; font-weight:bold; font-size:16px;">
                    Proceed to Payment
                </button>
            </a>
            ''', unsafe_allow_html=True)
        else:
            st.error("Unable to create checkout session. Please try again later.")

def show_payment_success_ui(session_id=None):
    """Display payment success message"""
    st.success("🎉 Payment successful! Your subscription has been activated.")
    st.write("Thank you for subscribing to Neufin!")
    st.write("You now have access to all premium features.")
    
    if session_id:
        try:
            # Verify the session status with Stripe
            session = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status == "paid":
                st.write(f"Transaction ID: {session.payment_intent}")
        except Exception as e:
            st.warning(f"Could not verify payment: {str(e)}")
            
    if st.button("Go to Dashboard"):
        st.session_state["page"] = "dashboard"
        st.rerun()

def show_subscription_management(user_id):
    """
    Display subscription management UI
    
    Args:
        user_id (int): User ID
    """
    subscription = get_user_subscription(user_id)
    
    st.title("Manage Your Subscription")
    
    if subscription and subscription.is_active:
        st.write(f"Current plan: **{subscription.level.title()}**")
        
        if subscription.is_trial:
            st.write("**Trial Subscription**")
            days_left = (subscription.end_date - datetime.now()).days
            st.write(f"Your trial ends in **{days_left} days**")
            
            if st.button("Upgrade to Paid Plan"):
                st.session_state["redirect_to_payment"] = True
                st.session_state["payment_plan"] = subscription.level
                st.rerun()
        else:
            st.write(f"Status: **Active**")
            if subscription.end_date:
                st.write(f"Renewal date: **{subscription.end_date.strftime('%Y-%m-%d')}**")
            else:
                st.write("Renewal date: **Lifetime access**")
            
            if subscription.level != "premium":
                if st.button("Upgrade to Premium"):
                    st.session_state["redirect_to_payment"] = True
                    st.session_state["payment_plan"] = "premium"
                    st.rerun()
            
            if st.button("Cancel Subscription"):
                st.warning("Are you sure you want to cancel your subscription?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Cancel"):
                        # Cancel subscription logic here
                        try:
                            # If there's a payment ID, cancel in Stripe
                            if subscription.payment_id:
                                stripe.Subscription.delete(subscription.payment_id)
                                
                            # Update local database
                            update_subscription(
                                user_id=user_id,
                                level='free',
                                is_active=False
                            )
                            
                            st.success("Your subscription has been canceled.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error canceling subscription: {str(e)}")
                with col2:
                    if st.button("No, Keep Subscription"):
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
                if st.button("Choose Basic"):
                    st.session_state["redirect_to_payment"] = True
                    st.session_state["payment_plan"] = "basic"
                    st.rerun()
        
        with col2:
            with st.container():
                st.subheader("Premium Plan")
                st.write("- All Basic features")
                st.write("- Advanced AI analytics")
                st.write("- Real-time updates")
                st.write("- Personalized recommendations")
                st.write("\nPrice: $19.99/month")
                if st.button("Choose Premium"):
                    st.session_state["redirect_to_payment"] = True
                    st.session_state["payment_plan"] = "premium"
                    st.rerun()
        
        if st.button("Start Free Trial"):
            # Start trial logic
            end_date = datetime.now() + timedelta(days=14)
            update_subscription(
                user_id=user_id,
                level="basic",
                is_trial=True,
                end_date=end_date
            )
            st.success("Your 14-day free trial has been activated!")
            st.rerun()