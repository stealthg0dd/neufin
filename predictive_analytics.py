"""
Predictive Analytics Module for Neufin Platform

This module provides predictive analytics capabilities based on historical
sentiment data and market trends. It uses machine learning models to forecast
future market sentiment and price movements.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from database import get_historical_sentiment

def create_features(df, target_col):
    """
    Create time series features based on sentiment data
    
    Args:
        df (DataFrame): Historical sentiment data
        target_col (str): Target column to predict
        
    Returns:
        DataFrame: DataFrame with features for model training
    """
    df = df.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create lagged features (previous days sentiment)
    for lag in range(1, 8):  # 1-week lags
        df[f'sentiment_lag_{lag}'] = df[target_col].shift(lag)
    
    # Calculate rolling mean for different windows
    for window in [3, 7, 14]:
        df[f'sentiment_roll_mean_{window}'] = df[target_col].rolling(window=window).mean()
    
    # Calculate rolling std for volatility
    for window in [7, 14]:
        df[f'sentiment_roll_std_{window}'] = df[target_col].rolling(window=window).std()
    
    # Calculate rate of change
    df['sentiment_roc_1'] = df[target_col].pct_change(1)
    df['sentiment_roc_3'] = df[target_col].pct_change(3)
    df['sentiment_roc_7'] = df[target_col].pct_change(7)
    
    # Add date-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Drop rows with NaN values (due to lagging/rolling operations)
    df = df.dropna()
    
    return df

def train_sentiment_prediction_model(days_history=60, forecast_days=10):
    """
    Train a model to predict future sentiment based on historical data
    
    Args:
        days_history (int): Number of days of historical data to use
        forecast_days (int): Number of days to forecast
        
    Returns:
        tuple: (model, scaler, feature_names, test_score)
    """
    # Get historical sentiment data
    historical_data = get_historical_sentiment(days=days_history)
    
    if not historical_data or len(historical_data) < 30:
        # Not enough data to build a reliable model
        return None, None, None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Create features
    feature_df = create_features(df, 'sentiment_score')
    
    # Prepare for training
    X = feature_df.drop(['date', 'sentiment_score', 'market_indices', 'news_sentiment', 'technical_sentiment'], axis=1)
    y = feature_df['sentiment_score']
    
    # Save feature names
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create pipeline with scaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Get test score
    test_score = pipeline.score(X_test, y_test)
    
    # Extract scaler for later use
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    
    return model, scaler, feature_names, test_score

def predict_future_sentiment(days_ahead=10):
    """
    Predict future sentiment for specified number of days ahead
    
    Args:
        days_ahead (int): Number of days to predict ahead
        
    Returns:
        DataFrame: Predictions with dates
    """
    # Get historical data for the last 60 days
    historical_data = get_historical_sentiment(days=60)
    
    if not historical_data or len(historical_data) < 30:
        # Not enough historical data for a reliable forecast
        return None
    
    # Train model
    model, scaler, feature_names, score = train_sentiment_prediction_model()
    
    if model is None:
        return None
    
    # Convert historical data to DataFrame
    hist_df = pd.DataFrame(historical_data)
    
    # Process features
    feature_df = create_features(hist_df, 'sentiment_score')
    
    # Make a copy of the latest available data for forecasting
    latest_data = feature_df.iloc[-1:].copy()
    forecast_dates = []
    forecast_values = []
    
    # Append the last known actual value
    last_known_date = feature_df['date'].iloc[-1]
    last_known_value = feature_df['sentiment_score'].iloc[-1]
    
    forecast_dates.append(last_known_date)
    forecast_values.append(last_known_value)
    
    # Predict one day at a time, using previous predictions for lagged features
    current_data = latest_data.copy()
    
    for i in range(1, days_ahead + 1):
        # Calculate next date
        next_date = last_known_date + timedelta(days=i)
        forecast_dates.append(next_date)
        
        # Update date-based features
        current_data['day_of_week'] = next_date.dayofweek
        current_data['month'] = next_date.month
        current_data['quarter'] = next_date.quarter
        
        # Select features for prediction
        X_predict = current_data[feature_names]
        
        # Make prediction
        prediction = model.predict(scaler.transform(X_predict))[0]
        forecast_values.append(prediction)
        
        # Update current data for next iteration
        current_data['sentiment_score'] = prediction
        
        # Update lagged features
        for lag in range(7, 0, -1):
            if f'sentiment_lag_{lag}' in current_data.columns:
                if lag == 1:
                    current_data[f'sentiment_lag_{lag}'] = prediction
                else:
                    current_data[f'sentiment_lag_{lag}'] = current_data[f'sentiment_lag_{lag-1}']
        
        # Update rolling means and std
        # (simplification: just use the prediction as an approximation)
        for window in [3, 7, 14]:
            current_data[f'sentiment_roll_mean_{window}'] = prediction
        
        for window in [7, 14]:
            current_data[f'sentiment_roll_std_{window}'] = feature_df['sentiment_roll_std_{window}'].iloc[-1]
        
        # Update rate of change (simplified)
        last_value = forecast_values[-2]
        current_data['sentiment_roc_1'] = (prediction - last_value) / last_value if last_value != 0 else 0
        current_data['sentiment_roc_3'] = current_data['sentiment_roc_1']
        current_data['sentiment_roc_7'] = current_data['sentiment_roc_1']
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'sentiment_score': forecast_values,
        'is_prediction': [False] + [True] * days_ahead
    })
    
    return forecast_df, score

def generate_prediction_chart(forecast_df, historical_df=None, confidence=0.9):
    """
    Generate a plotly chart showing historical and predicted sentiment
    
    Args:
        forecast_df (DataFrame): DataFrame with predictions
        historical_df (DataFrame, optional): Additional historical data to show
        confidence (float): Confidence interval (0-1)
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if forecast_df is None:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Calculate confidence intervals
    std_dev = 0.05  # Estimate of standard deviation in predictions
    z_score = 1.96  # ~95% confidence interval
    margin = z_score * std_dev
    
    # Extract actual and predicted values
    actual_data = forecast_df[forecast_df['is_prediction'] == False]
    predicted_data = forecast_df[forecast_df['is_prediction'] == True]
    
    # Add additional historical data if provided
    if historical_df is not None:
        hist_dates = historical_df['date'].tolist()
        hist_values = historical_df['sentiment_score'].tolist()
        
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines',
            name='Historical Sentiment',
            line=dict(color='rgba(123, 104, 238, 0.8)', width=2)
        ))
    
    # Add traces for actual data points
    fig.add_trace(go.Scatter(
        x=actual_data['date'],
        y=actual_data['sentiment_score'],
        mode='lines+markers',
        name='Actual Sentiment',
        line=dict(color='#7B68EE', width=3)
    ))
    
    # Add traces for predicted data points
    fig.add_trace(go.Scatter(
        x=predicted_data['date'],
        y=predicted_data['sentiment_score'],
        mode='lines+markers',
        name='Predicted Sentiment',
        line=dict(color='#00C853', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=predicted_data['date'].tolist() + predicted_data['date'].tolist()[::-1],
        y=(predicted_data['sentiment_score'] + margin).tolist() + 
           (predicted_data['sentiment_score'] - margin).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 200, 83, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title='Market Sentiment Forecast',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        yaxis=dict(
            range=[-1.1, 1.1],
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish']
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(255, 255, 255, 0.2)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(255, 255, 255, 0.2)',
        zerolinecolor='rgba(255, 255, 255, 0.2)'
    )
    
    return fig

def forecast_sentiment_impact(ticker, forecast_df):
    """
    Forecast potential price impact based on sentiment prediction
    
    Args:
        ticker (str): Stock ticker symbol
        forecast_df (DataFrame): Sentiment forecast data
        
    Returns:
        dict: Forecast impact analysis
    """
    if forecast_df is None or len(forecast_df) < 2:
        return None
    
    # Extract just the predictions
    predictions = forecast_df[forecast_df['is_prediction']]['sentiment_score'].tolist()
    
    # Calculate average predicted sentiment
    avg_sentiment = sum(predictions) / len(predictions)
    
    # Calculate trend direction (positive = upward trend)
    trend = predictions[-1] - predictions[0]
    
    # Calculate volatility
    volatility = np.std(predictions)
    
    # Estimate potential price impact based on sentiment
    # (This is a simplified model)
    price_impact_percentage = avg_sentiment * 2.5  # Simplified assumption: 2.5% change per 1.0 sentiment
    
    return {
        'ticker': ticker,
        'avg_sentiment': avg_sentiment,
        'trend': trend,
        'volatility': volatility,
        'price_impact_percentage': price_impact_percentage,
        'forecast_days': len(predictions),
        'sentiment_direction': 'bullish' if avg_sentiment > 0.2 else ('bearish' if avg_sentiment < -0.2 else 'neutral'),
        'confidence': max(0.5, min(0.95, 0.75 - volatility)),  # Higher volatility = lower confidence
        'analysis_date': datetime.now().strftime('%Y-%m-%d')
    }

def get_sentiment_insights(forecast_df):
    """
    Generate insights text based on sentiment predictions
    
    Args:
        forecast_df (DataFrame): Forecast DataFrame with predictions
        
    Returns:
        str: HTML-formatted insights text
    """
    if forecast_df is None or len(forecast_df) < 2:
        return "<p>Insufficient data for sentiment prediction.</p>"
    
    # Extract just the predictions
    predictions = forecast_df[forecast_df['is_prediction']]['sentiment_score'].tolist()
    
    # Calculate metrics
    avg_sentiment = sum(predictions) / len(predictions)
    trend = predictions[-1] - predictions[0]
    days_ahead = len(predictions)
    
    # Determine sentiment descriptions
    if avg_sentiment > 0.7:
        sentiment_desc = "very bullish"
        sentiment_color = "#00C853"
    elif avg_sentiment > 0.3:
        sentiment_desc = "bullish"
        sentiment_color = "#4CAF50"
    elif avg_sentiment > -0.3:
        sentiment_desc = "neutral"
        sentiment_color = "#FFD54F"
    elif avg_sentiment > -0.7:
        sentiment_desc = "bearish"
        sentiment_color = "#FF9800"
    else:
        sentiment_desc = "very bearish"
        sentiment_color = "#FF5252"
    
    # Determine trend descriptions
    if trend > 0.3:
        trend_desc = "strongly improving"
        trend_color = "#00C853"
    elif trend > 0.1:
        trend_desc = "improving"
        trend_color = "#4CAF50"
    elif trend > -0.1:
        trend_desc = "stable"
        trend_color = "#FFD54F"
    elif trend > -0.3:
        trend_desc = "declining"
        trend_color = "#FF9800"
    else:
        trend_desc = "strongly declining"
        trend_color = "#FF5252"
    
    # Generate insights text
    insights_html = f"""
    <div style="margin-bottom:15px;">
        <p>Our predictive model projects <span style="color:{sentiment_color};font-weight:bold;">{sentiment_desc}</span> 
        market sentiment over the next {days_ahead} days, with a 
        <span style="color:{trend_color};font-weight:bold;">{trend_desc}</span> trend.</p>
        
        <p style="margin-top:10px;">Forecast Summary:</p>
        <ul style="margin-top:5px;">
            <li>Average Sentiment: <span style="color:{sentiment_color};font-weight:bold;">{avg_sentiment:.2f}</span></li>
            <li>Trend Direction: <span style="color:{trend_color};font-weight:bold;">{trend:.2f}</span></li>
            <li>Forecast Period: {days_ahead} days</li>
        </ul>
    </div>
    """
    
    return insights_html

def integrate_price_prediction(sentiment_prediction, historical_prices):
    """
    Integrate sentiment prediction with price data to create price forecast
    
    Args:
        sentiment_prediction (DataFrame): Predicted sentiment values
        historical_prices (DataFrame): Historical price data with 'Date' and 'Close' columns
        
    Returns:
        DataFrame: Price forecast data
    """
    if sentiment_prediction is None or historical_prices is None:
        return None
    
    if len(historical_prices) < 10:
        return None
    
    # Ensure data is in the right format
    historical_prices = historical_prices.copy()
    
    # Make sure date is datetime
    if 'Date' in historical_prices.columns:
        historical_prices['Date'] = pd.to_datetime(historical_prices['Date'])
    elif 'date' in historical_prices.columns:
        historical_prices['Date'] = pd.to_datetime(historical_prices['date'])
        historical_prices = historical_prices.drop('date', axis=1)
    
    # Get just the prediction part
    predictions = sentiment_prediction[sentiment_prediction['is_prediction']]
    
    # Get the last known price
    last_price = historical_prices['Close'].iloc[-1]
    
    # Create forecast
    dates = predictions['date'].tolist()
    
    # Simple model: price change proportional to sentiment
    # Positive sentiment -> price goes up, negative -> price goes down
    price_changes = []
    cumulative_price = last_price
    
    for sentiment in predictions['sentiment_score']:
        # Daily change as a percentage of price, based on sentiment
        # Example: sentiment of 0.5 might mean 0.5% price increase
        daily_change_pct = sentiment * 0.01  # Convert to percentage (adjustable factor)
        daily_change = cumulative_price * daily_change_pct
        cumulative_price += daily_change
        price_changes.append(cumulative_price)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': dates,
        'Close': price_changes,
        'is_forecast': True
    })
    
    # Format the last part of historical data
    historical_subset = historical_prices.iloc[-10:].copy()
    historical_subset['is_forecast'] = False
    
    # Combine historical and forecast
    combined_df = pd.concat([historical_subset, forecast_df], ignore_index=True)
    
    return combined_df