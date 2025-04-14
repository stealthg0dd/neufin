"""
Utility script to upgrade a specific user to premium subscription.
"""

import os
import sys
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import User, Subscription

# Get database connection string
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable is not set.")
    sys.exit(1)

# Create database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def upgrade_user_to_premium(email, lifetime=True):
    """
    Upgrade a user to premium subscription.
    
    Args:
        email (str): Email of the user to upgrade
        lifetime (bool): If True, no end date will be set (lifetime access)
    """
    try:
        # Find the user
        user = session.query(User).filter_by(email=email).first()
        if not user:
            print(f"User with email {email} not found.")
            return False
            
        # Check if user already has a subscription
        subscription = session.query(Subscription).filter_by(user_id=user.id).first()
        
        # Set end date (None for lifetime access, or 1 year from now)
        end_date = None if lifetime else datetime.now() + timedelta(days=365)
        
        if subscription:
            # Update existing subscription
            subscription.level = "premium"
            subscription.is_active = True
            subscription.is_trial = False
            subscription.end_date = end_date
            print(f"Updated subscription for {email} to premium.")
        else:
            # Create new subscription
            new_subscription = Subscription(
                user_id=user.id,
                level="premium",
                start_date=datetime.now(),
                end_date=end_date,
                is_active=True,
                is_trial=False
            )
            session.add(new_subscription)
            print(f"Created new premium subscription for {email}.")
            
        # Commit changes
        session.commit()
        return True
        
    except Exception as e:
        print(f"Error upgrading user: {str(e)}")
        session.rollback()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    # Upgrade the specific user
    email = "info@ctechventure.com"
    result = upgrade_user_to_premium(email, lifetime=True)
    
    if result:
        print(f"Successfully upgraded {email} to premium with lifetime access.")
    else:
        print(f"Failed to upgrade {email}.")