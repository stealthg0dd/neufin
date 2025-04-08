import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import streamlit as st

# Get DB connection string from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create database engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define database models
class User(Base):
    """User account information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(100), nullable=False)  # In production, use proper password hashing
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    saved_analyses = relationship("SavedAnalysis", back_populates="user")
    favorites = relationship("Favorite", back_populates="user")

class Subscription(Base):
    """User subscription information"""
    __tablename__ = 'subscriptions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    level = Column(String(20), nullable=False, default='free')  # free, basic, premium
    start_date = Column(DateTime, default=datetime.now)
    end_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_trial = Column(Boolean, default=False)
    payment_id = Column(String(100))
    
    # Relationships
    user = relationship("User", back_populates="subscription")

class StockData(Base):
    """Cached stock data to reduce API calls"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    time_period = Column(String(10), nullable=False)  # 1d, 5d, 1mo, etc.
    last_updated = Column(DateTime, default=datetime.now)
    data_json = Column(Text)  # JSON string of DataFrame
    
    # Composite unique constraint
    __table_args__ = (
        {},
    )

class MarketSentiment(Base):
    """Overall market sentiment analysis results"""
    __tablename__ = 'market_sentiment'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.now)
    sentiment_score = Column(Float)
    market_indices = Column(String(100))  # Comma-separated list of indices used
    news_sentiment = Column(Float)
    technical_sentiment = Column(Float)
    
    # Composite unique constraint
    __table_args__ = (
        {},
    )

class StockAnalysis(Base):
    """AI-generated stock analysis for caching"""
    __tablename__ = 'stock_analysis'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    analysis_date = Column(DateTime, default=datetime.now)
    analysis_json = Column(Text)  # JSON string of analysis result
    
    # Composite unique constraint
    __table_args__ = (
        {},
    )

class SectorAnalysis(Base):
    """AI-generated sector analysis for caching"""
    __tablename__ = 'sector_analysis'
    
    id = Column(Integer, primary_key=True)
    sector_name = Column(String(50), nullable=False)
    analysis_date = Column(DateTime, default=datetime.now)
    analysis_json = Column(Text)  # JSON string of analysis result
    
    # Composite unique constraint
    __table_args__ = (
        {},
    )

class TradeAnalysis(Base):
    """AI-generated global trade analysis for caching"""
    __tablename__ = 'trade_analysis'
    
    id = Column(Integer, primary_key=True)
    analysis_date = Column(DateTime, default=datetime.now)
    analysis_json = Column(Text)  # JSON string of analysis result

class SavedAnalysis(Base):
    """User saved analyses and reports"""
    __tablename__ = 'saved_analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    analysis_type = Column(String(20), nullable=False)  # stock, sector, trade
    reference_id = Column(Integer, nullable=False)  # ID from respective analysis table
    saved_date = Column(DateTime, default=datetime.now)
    notes = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="saved_analyses")

class Favorite(Base):
    """User favorite stocks/sectors"""
    __tablename__ = 'favorites'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    favorite_type = Column(String(20), nullable=False)  # stock, sector
    item_name = Column(String(50), nullable=False)  # Ticker or sector name
    added_date = Column(DateTime, default=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="favorites")

class UserSettings(Base):
    """User preferences and settings"""
    __tablename__ = 'user_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, unique=True)
    default_stocks = Column(String(200))  # Comma-separated list of tickers
    default_time_period = Column(String(10), default='1mo')
    theme_preference = Column(String(20), default='light')  # light, dark
    notification_preferences = Column(JSON, default={})
    
    # Relationships
    user = relationship("User")

# Create database tables if they don't exist
Base.metadata.create_all(engine)

# Create a session class
Session = sessionmaker(bind=engine)

# Database helper functions
def get_db_session():
    """Get a database session"""
    return Session()

def create_user(username, email, password):
    """Create a new user in the database"""
    with get_db_session() as session:
        # Check if user already exists
        existing_user = session.query(User).filter(User.email == email).first()
        if existing_user:
            return None, "User with this email already exists"
        
        # In production, use a proper password hashing library
        password_hash = password  # DEMO ONLY - Insecure!
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            last_login=datetime.now()
        )
        session.add(new_user)
        session.commit()
        
        # Create default subscription (free tier)
        new_subscription = Subscription(
            user_id=new_user.id,
            level='free',
            is_active=True,
            end_date=None  # Free doesn't expire
        )
        session.add(new_subscription)
        session.commit()
        
        return new_user, None

def authenticate_user(email, password):
    """Authenticate a user with email and password"""
    with get_db_session() as session:
        # Find user by email
        user = session.query(User).filter(User.email == email).first()
        
        if not user:
            return None, "User not found"
        
        # In production, use a proper password verification
        if user.password_hash != password:  # DEMO ONLY - Insecure!
            return None, "Invalid password"
        
        # Update last login
        user.last_login = datetime.now()
        session.commit()
        
        return user, None

def get_user_subscription(user_id):
    """Get user subscription details"""
    with get_db_session() as session:
        subscription = session.query(Subscription).filter(Subscription.user_id == user_id).first()
        return subscription

def update_subscription(user_id, level, is_trial=False, end_date=None):
    """Update user subscription level"""
    with get_db_session() as session:
        subscription = session.query(Subscription).filter(Subscription.user_id == user_id).first()
        
        if not subscription:
            # Create new subscription if it doesn't exist
            subscription = Subscription(
                user_id=user_id,
                level=level,
                is_trial=is_trial,
                end_date=end_date,
                is_active=True
            )
            session.add(subscription)
        else:
            # Update existing subscription
            subscription.level = level
            subscription.is_trial = is_trial
            subscription.end_date = end_date
            subscription.is_active = True
            subscription.start_date = datetime.now()
        
        session.commit()
        return subscription

def cache_stock_data(ticker, time_period, data_df):
    """Cache stock data to reduce API calls"""
    with get_db_session() as session:
        # Check if we already have this data cached
        cached_data = session.query(StockData).filter(
            StockData.ticker == ticker,
            StockData.time_period == time_period
        ).first()
        
        # Convert DataFrame to JSON string
        data_json = data_df.to_json(orient='split', date_format='iso')
        
        if cached_data:
            # Update existing cache
            cached_data.data_json = data_json
            cached_data.last_updated = datetime.now()
        else:
            # Create new cache entry
            new_cache = StockData(
                ticker=ticker,
                time_period=time_period,
                data_json=data_json
            )
            session.add(new_cache)
        
        session.commit()

def get_cached_stock_data(ticker, time_period, max_age_hours=1):
    """Get cached stock data if available and not too old"""
    with get_db_session() as session:
        cached_data = session.query(StockData).filter(
            StockData.ticker == ticker,
            StockData.time_period == time_period
        ).first()
        
        if not cached_data:
            return None
        
        # Check if data is too old
        time_diff = datetime.now() - cached_data.last_updated
        if time_diff.total_seconds() > (max_age_hours * 3600):
            return None
        
        # Convert JSON string back to DataFrame
        import pandas as pd
        return pd.read_json(cached_data.data_json, orient='split')

def cache_ai_analysis(analysis_type, identifier, analysis_data):
    """Cache AI-generated analysis"""
    with get_db_session() as session:
        analysis_json = json.dumps(analysis_data)
        
        if analysis_type == 'stock':
            # Check if we already have this analysis cached
            cached = session.query(StockAnalysis).filter(
                StockAnalysis.ticker == identifier
            ).first()
            
            if cached:
                # Update existing cache
                cached.analysis_json = analysis_json
                cached.analysis_date = datetime.now()
            else:
                # Create new cache entry
                new_cache = StockAnalysis(
                    ticker=identifier,
                    analysis_json=analysis_json
                )
                session.add(new_cache)
                
        elif analysis_type == 'sector':
            # Check if we already have this analysis cached
            cached = session.query(SectorAnalysis).filter(
                SectorAnalysis.sector_name == identifier
            ).first()
            
            if cached:
                # Update existing cache
                cached.analysis_json = analysis_json
                cached.analysis_date = datetime.now()
            else:
                # Create new cache entry
                new_cache = SectorAnalysis(
                    sector_name=identifier,
                    analysis_json=analysis_json
                )
                session.add(new_cache)
                
        elif analysis_type == 'trade':
            # Only keep one global trade analysis, always update
            cached = session.query(TradeAnalysis).first()
            
            if cached:
                # Update existing cache
                cached.analysis_json = analysis_json
                cached.analysis_date = datetime.now()
            else:
                # Create new cache entry
                new_cache = TradeAnalysis(
                    analysis_json=analysis_json
                )
                session.add(new_cache)
        
        session.commit()

def get_cached_ai_analysis(analysis_type, identifier=None, max_age_hours=24):
    """Get cached AI analysis if available and not too old"""
    with get_db_session() as session:
        if analysis_type == 'stock':
            cached = session.query(StockAnalysis).filter(
                StockAnalysis.ticker == identifier
            ).first()
            
        elif analysis_type == 'sector':
            cached = session.query(SectorAnalysis).filter(
                SectorAnalysis.sector_name == identifier
            ).first()
            
        elif analysis_type == 'trade':
            cached = session.query(TradeAnalysis).first()
            
        else:
            return None
        
        if not cached:
            return None
        
        # Check if data is too old
        time_diff = datetime.now() - cached.analysis_date
        if time_diff.total_seconds() > (max_age_hours * 3600):
            return None
        
        # Parse JSON string back to dict
        return json.loads(cached.analysis_json)

def save_user_analysis(user_id, analysis_type, reference_id, notes=None):
    """Save analysis for a user"""
    with get_db_session() as session:
        saved = SavedAnalysis(
            user_id=user_id,
            analysis_type=analysis_type,
            reference_id=reference_id,
            notes=notes
        )
        session.add(saved)
        session.commit()
        return saved.id

def get_user_saved_analyses(user_id):
    """Get all analyses saved by a user"""
    with get_db_session() as session:
        saved = session.query(SavedAnalysis).filter(
            SavedAnalysis.user_id == user_id
        ).all()
        return saved

def add_favorite(user_id, favorite_type, item_name):
    """Add favorite stock or sector for a user"""
    with get_db_session() as session:
        # Check if already favorited
        existing = session.query(Favorite).filter(
            Favorite.user_id == user_id,
            Favorite.favorite_type == favorite_type,
            Favorite.item_name == item_name
        ).first()
        
        if existing:
            return existing.id
        
        # Add new favorite
        favorite = Favorite(
            user_id=user_id,
            favorite_type=favorite_type,
            item_name=item_name
        )
        session.add(favorite)
        session.commit()
        return favorite.id

def remove_favorite(user_id, favorite_type, item_name):
    """Remove favorite stock or sector for a user"""
    with get_db_session() as session:
        favorite = session.query(Favorite).filter(
            Favorite.user_id == user_id,
            Favorite.favorite_type == favorite_type,
            Favorite.item_name == item_name
        ).first()
        
        if favorite:
            session.delete(favorite)
            session.commit()
            return True
        return False

def get_user_favorites(user_id, favorite_type=None):
    """Get all favorite stocks or sectors for a user"""
    with get_db_session() as session:
        query = session.query(Favorite).filter(Favorite.user_id == user_id)
        
        if favorite_type:
            query = query.filter(Favorite.favorite_type == favorite_type)
            
        favorites = query.all()
        return favorites

def get_user_settings(user_id):
    """Get user settings"""
    with get_db_session() as session:
        settings = session.query(UserSettings).filter(
            UserSettings.user_id == user_id
        ).first()
        
        if not settings:
            # Create default settings
            settings = UserSettings(user_id=user_id)
            session.add(settings)
            session.commit()
            
        return settings

def update_user_settings(user_id, **kwargs):
    """Update user settings"""
    with get_db_session() as session:
        settings = session.query(UserSettings).filter(
            UserSettings.user_id == user_id
        ).first()
        
        if not settings:
            # Create settings with provided values
            settings_data = {"user_id": user_id}
            settings_data.update(kwargs)
            settings = UserSettings(**settings_data)
            session.add(settings)
        else:
            # Update existing settings
            for key, value in kwargs.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
                    
        session.commit()
        return settings