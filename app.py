import streamlit as st

# This will be imported from main.py, so don't set page config here

# Apply custom styling for the dashboard
def apply_dashboard_styling():
    """Apply custom styling for the dashboard"""
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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
import re
from urllib.parse import parse_qs, urlparse

# Import custom modules
# Import auth modules
from auth_manager import (
    init_auth_session, is_authenticated, get_current_user, 
    login_user, register_user, logout_user, init_google_oauth,
    handle_google_callback, handle_facebook_callback, init_facebook_oauth,
    user_has_permission, require_login, require_permission, 
    show_login_ui, show_user_profile_ui, start_free_trial
)

# Import database modules
from database import (
    get_user_subscription, update_subscription, get_historical_sentiment, 
    store_market_sentiment
)

# Import subscription modules
from subscription_manager import (
    create_checkout_session, handle_subscription_webhook,
    show_payment_ui, show_payment_success_ui, show_subscription_management
)

# Import sentiment analysis modules
from sentiment_analyzer import analyze_text_sentiment, analyze_stock_sentiment, analyze_news_sentiment_batch

# Import data fetching modules
from data_fetcher import fetch_stock_data, fetch_market_news, fetch_sector_performance, get_market_indices

# Import investment advisor modules
from investment_advisor import get_stock_recommendations, analyze_global_trade_impact, get_sector_insights

# Import AI analysis modules
from ai_analyst import generate_investment_thesis, generate_sector_outlook, analyze_global_trade_conditions

# Import utility modules
from utils import (
    format_sentiment_score, get_sentiment_color, get_market_preferences, 
    update_market_preferences, format_large_number, get_market_indices
)

# Import predictive analytics modules
from predictive_analytics import (
    predict_future_sentiment, generate_prediction_chart, get_sentiment_insights, 
    forecast_sentiment_impact, integrate_price_prediction
)

# Import news recommendation module
from news_recommender import NewsRecommender, format_news_card

# Check if OpenAI API key is available for advanced sentiment analysis
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_ADVANCED_AI = bool(OPENAI_API_KEY)

# Initialize authentication session state
init_auth_session()

# Function to handle redirect from landing page
def handle_landing_redirect():
    """Handle redirect from landing page with email parameter"""
    try:
        # Use the latest Streamlit API for query parameters
        if hasattr(st, 'query_params'):
            query_params = st.query_params
            if "redirect_to" in query_params and query_params["redirect_to"] == "signup" and "email" in query_params:
                email = query_params["email"]
                # Set up registration flow with prefilled email
                st.session_state["show_auth"] = True
                st.session_state["auth_mode"] = "register"
                st.session_state["prefill_email"] = email
                # Clear the query params to avoid infinite redirects
                st.query_params.clear()
                return True
        return False
    except Exception as e:
        print(f"Error handling landing redirect: {str(e)}")
        return False

# Check if user has access to specific features based on subscription level
def check_feature_access(feature):
    """
    Check if current user has access to a specific feature based on subscription level.
    
    Args:
        feature (str): Feature name to check ('basic', 'premium', 'free', etc.)
    
    Returns:
        bool: True if user has access, False otherwise
    """
    if not is_authenticated():
        return feature == 'free'  # Only free features for non-authenticated users
        
    if feature == 'free':
        return True  # Everyone has access to free features
        
    # Get current user information
    user = get_current_user()
    if not user:
        return False
        
    # Check subscription permissions
    return user_has_permission(feature + '_features')

def create_animated_sentiment_trend(days=14, with_ai_insights=True):
    """
    Create an animated sentiment trend visualization using historical sentiment data.
    
    Args:
        days (int): Number of days to show in the trend
        with_ai_insights (bool): Whether to include AI-generated insights
        
    Returns:
        plotly.graph_objects.Figure: Animated trend visualization
    """
    # Get historical sentiment data
    sentiment_history = get_historical_sentiment(days)
    
    # If no historical data is available, create an empty chart with a message
    if not sentiment_history:
        fig = go.Figure()
        fig.update_layout(
            title="No historical sentiment data available",
            title_font_color="#7B68EE",
            font_color="#E0E0E0",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300
        )
        return fig
    
    # Convert list of dict to DataFrame
    df = pd.DataFrame(sentiment_history)
    
    # Create a date range index to make sure we have entries for all days
    date_range = pd.date_range(
        start=df['date'].min(), 
        end=df['date'].max()
    )
    
    # Reindex the DataFrame and interpolate missing values
    df.set_index('date', inplace=True)
    df = df.reindex(date_range)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    df.interpolate(method='linear', inplace=True)
    
    # Generate AI insights for each data point if requested
    if with_ai_insights and 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = {}
    
    # Check if we need to generate new insights
    if with_ai_insights:
        insights_needed = False
        for i, row in df.iterrows():
            date_key = row['date'].strftime('%Y-%m-%d')
            if date_key not in st.session_state.ai_insights:
                insights_needed = True
                break
        
        # Generate all needed insights at once to be more efficient
        if insights_needed and check_feature_access('premium'):  # Make this a premium feature
            try:
                with st.spinner("Generating AI insights for data points..."):
                    # Generate AI insights for all data points at once
                    generate_sentiment_insights(df)
            except Exception as e:
                st.warning(f"Could not generate AI insights: {str(e)}")
    
    # Prepare hover templates with insights if available
    hover_templates = []
    for i, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        sentiment_score = row['sentiment_score']
        sentiment_label = format_sentiment_score(sentiment_score)
        
        # Format the hover text
        hover_text = f"<b>Date:</b> {date_str}<br><b>Sentiment:</b> {sentiment_label} ({sentiment_score:.2f})"
        
        # Add AI insight if available
        if with_ai_insights and check_feature_access('premium'):
            if date_str in st.session_state.ai_insights:
                insight = st.session_state.ai_insights[date_str]
                hover_text += f"<br><br><b>AI Insight:</b><br>{insight}"
            else:
                hover_text += "<br><br><i>Hover over other points for AI insights</i>"
        
        hover_templates.append(hover_text)
    
    # Add hover_templates to the DataFrame
    df['hover_text'] = hover_templates
    
    # Create animation frames
    frames = []
    for i in range(1, len(df) + 1):
        subset = df.iloc[:i]
        
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=subset['date'],
                    y=subset['sentiment_score'],
                    mode='lines+markers',
                    line=dict(width=3, color='#7B68EE'),
                    marker=dict(
                        size=10, 
                        color='#7B68EE',
                        symbol='circle',
                        line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
                    ),
                    hoverinfo='text',
                    hovertext=subset['hover_text'],
                    hoverlabel=dict(
                        bgcolor="rgba(20, 20, 30, 0.9)",
                        bordercolor="#7B68EE",
                        font=dict(color="white", size=12),
                        align="left"
                    )
                )
            ],
            name=f'frame{i}'
        )
        frames.append(frame)
    
    # Create the initial figure with enhanced tooltips
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df['date'][:1],
                y=df['sentiment_score'][:1],
                mode='lines+markers',
                line=dict(width=3, color='#7B68EE'),
                marker=dict(
                    size=10, 
                    color='#7B68EE',
                    symbol='circle',
                    line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
                ),
                name='Market Sentiment',
                hoverinfo='text',
                hovertext=df['hover_text'][:1],
                hoverlabel=dict(
                    bgcolor="rgba(20, 20, 30, 0.9)",
                    bordercolor="#7B68EE",
                    font=dict(color="white", size=12),
                    align="left"
                )
            )
        ],
        frames=frames
    )
    
    # Add play button and slider
    fig.update_layout(
        title="Market Sentiment Trend with AI Insights",
        title_font_color="#7B68EE",
        font_color="#E0E0E0",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,  # Increased height for better visualization
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(123, 104, 238, 0.2)',
            title_font=dict(color='#E0E0E0'),
            tickfont=dict(color='#E0E0E0')
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='rgba(123, 104, 238, 0.1)',
            showline=True,
            linecolor='rgba(123, 104, 238, 0.2)',
            title="Sentiment Score",
            range=[-1, 1],
            title_font=dict(color='#E0E0E0'),
            tickfont=dict(color='#E0E0E0'),
            zeroline=True,
            zerolinecolor='rgba(123, 104, 238, 0.2)'
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                mode='immediate'
                            )
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode='immediate',
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ],
                x=0.1,
                y=0,
                bgcolor="rgba(10, 10, 20, 0.8)",
                font=dict(color="#E0E0E0")
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"frame{k+1}"],
                            dict(
                                frame=dict(duration=100, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ],
                        label=df['date'].iloc[k].strftime('%b %d')
                    )
                    for k in range(min(len(df), 10))  # Only show 10 steps max
                ],
                y=0,
                x=0.1,
                len=0.9,
                xanchor="left",
                currentvalue=dict(
                    visible=True,
                    prefix="Date: ",
                    font=dict(color="#E0E0E0")
                ),
                font=dict(color="#E0E0E0"),
                bgcolor="rgba(10, 10, 20, 0.8)"
            )
        ],
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 30, 0.95)",
            bordercolor="#7B68EE",
            font=dict(family="Arial", size=12, color="white")
        ),
        hovermode="closest"
    )
    
    # Add annotations for sentiment trend
    if len(df) > 1:
        start_sentiment = df['sentiment_score'].iloc[0]
        end_sentiment = df['sentiment_score'].iloc[-1]
        change = end_sentiment - start_sentiment
        change_pct = (change / abs(start_sentiment)) * 100 if start_sentiment != 0 else 0
        
        # Determine color and symbol based on trend
        if change > 0:
            trend_color = "rgba(0, 255, 0, 0.8)"  # Green
            trend_symbol = "📈"
            trend_text = "Improving"
        elif change < 0:
            trend_color = "rgba(255, 0, 0, 0.8)"  # Red
            trend_symbol = "📉"
            trend_text = "Declining"
        else:
            trend_color = "rgba(255, 255, 0, 0.8)"  # Yellow
            trend_symbol = "➡️"
            trend_text = "Stable"
        
        # Add trend annotation
        fig.add_annotation(
            x=df['date'].iloc[-1],
            y=end_sentiment,
            text=f"{trend_symbol} {trend_text} ({change_pct:.1f}%)",
            showarrow=True,
            arrowhead=1,
            arrowcolor=trend_color,
            arrowsize=1,
            arrowwidth=2,
            ax=-50,
            ay=-40,
            font=dict(color=trend_color, size=12),
            bordercolor=trend_color,
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(10, 10, 20, 0.8)",
            opacity=0.8
        )
    
    return fig

def generate_sentiment_insights(sentiment_df):
    """
    Generate AI insights for sentiment data points using Anthropic API
    
    Args:
        sentiment_df (DataFrame): DataFrame containing sentiment data with dates and scores
        
    Returns:
        dict: Dictionary of insights keyed by date string
    """
    from ai_analyst import get_anthropic_client
    
    # Check if we have the Anthropic API key available
    try:
        client = get_anthropic_client()
        if client is None:
            st.warning("AI insights require Anthropic API key")
            return {}
    except Exception as e:
        st.warning(f"Error initializing Anthropic client: {str(e)}")
        return {}
    
    # Create a cache for insights if not already present
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = {}
    
    # Prepare market context by getting recent market data
    market_context = ""
    try:
        # Get data for market indices to provide context
        indices = ['SPY', 'QQQ', 'DIA']
        for idx in indices:
            data = fetch_stock_data(idx, '1mo')
            if data is not None:
                change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                market_context += f"{idx} {'up' if change > 0 else 'down'} {abs(change):.2f}% in the last month. "
    except:
        # If we can't get market data, proceed with limited context
        market_context = "Limited market context available."
    
    # Generate insights for each data point
    for i, row in sentiment_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        
        # Skip if we already have this insight cached
        if date_str in st.session_state.ai_insights:
            continue
        
        # Generate explanation for the sentiment score
        score = row['sentiment_score']
        sentiment_word = format_sentiment_score(score)
        
        # Prepare data from nearby days to show trend
        nearby_days = sentiment_df[(sentiment_df['date'] >= row['date'] - pd.Timedelta(days=3)) & 
                                  (sentiment_df['date'] <= row['date'] + pd.Timedelta(days=3))]
        trend_data = []
        for _, trend_row in nearby_days.iterrows():
            trend_date = trend_row['date'].strftime('%Y-%m-%d')
            trend_score = trend_row['sentiment_score']
            trend_data.append(f"{trend_date}: {trend_score:.2f}")
        
        trend_context = ", ".join(trend_data)
            
        # Create prompt for Anthropic
        prompt = f"""
        As a market sentiment analysis AI, provide a concise, insightful explanation (max 60 words) for the 
        {sentiment_word} sentiment score of {score:.2f} on {date_str}.
        
        Nearby sentiment scores: {trend_context}
        
        Market context: {market_context}
        
        Your explanation should be insightful, mention possible causes, and implications for investors. 
        Keep it very concise, under 60 words, and make it sound professional but accessible.
        Do not mention the word "sentiment" since that's implied. Focus on explaining likely causes.
        
        Example format: "Fed policy signals and stronger-than-expected jobs report drove market optimism. 
        Tech sector showing resilience with increased institutional buying."
        """
        
        try:
            # Call Anthropic API
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and store insight
            insight = message.content[0].text.strip()
            
            # Limit length if it's too verbose
            if len(insight) > 200:
                insight = insight[:197] + "..."
                
            st.session_state.ai_insights[date_str] = insight
            
        except Exception as e:
            # If API call fails, use a generic insight
            st.session_state.ai_insights[date_str] = f"Analysis for {sentiment_word} market condition with score {score:.2f}."
            continue
    
    return st.session_state.ai_insights

# Helper function to show upgrade prompt for premium features
def show_upgrade_prompt(feature_name, subscription_level="premium", location_id=None):
    """
    Display an upgrade prompt for premium features
    
    Args:
        feature_name (str): Name of the feature requiring upgrade
        subscription_level (str): Subscription level needed ('basic', 'premium', etc.)
        location_id (str, optional): Additional identifier to ensure unique button keys
                                     when the same feature appears in different UI sections
    """
    # Check if user already has access to this feature level
    if is_authenticated():
        user = get_current_user()
        if user and user_has_permission(f"{subscription_level}_features"):
            # User already has access to this feature, don't show upgrade prompt
            return False
    
    # Generate unique button keys based on feature name and location to avoid duplicate IDs
    feature_key = feature_name.lower().replace(" ", "_").replace("-", "_")
    if location_id:
        feature_key = f"{feature_key}_{location_id}"
    
    st.info(f"💎 **{feature_name}** is a {subscription_level.capitalize()} feature. Please subscribe to access it.")
    
    if not is_authenticated():
        st.warning("You need to log in first to access subscription options.")
        if st.button("Log In / Register", key=f"login_for_{feature_key}"):
            st.session_state["show_auth"] = True
            st.rerun()
    else:
        if st.button("Upgrade Subscription", key=f"upgrade_for_{feature_key}"):
            st.session_state["show_subscription"] = True
            st.rerun()
    
    return True


def show_predictive_analytics(real_time_indicator=""):
    """
    Display the predictive analytics section with sentiment forecasting
    
    Args:
        real_time_indicator (str): HTML string for real-time indicator icon
    """
    # Start the predictive analytics card
    st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">🔮 Predictive Analytics {real_time_indicator}</h3>', unsafe_allow_html=True)
    
    try:
        # Create tabs for different prediction views
        pred_tabs = st.tabs(["Sentiment Forecast", "Price Impact", "AI Insights"])
        
        with pred_tabs[0]:
            # Sentiment forecast tab
            st.markdown("### Sentiment Forecast")
            st.markdown("Forecasting market sentiment for the next 10 days based on historical patterns and current market conditions.")
            
            days_ahead = st.slider("Forecast Days", min_value=5, max_value=30, value=10, step=5, 
                                   help="Number of days to forecast ahead")
            
            with st.spinner("Generating sentiment forecast..."):
                # Get sentiment forecast using predictive model
                forecast_df = predict_future_sentiment(days_ahead=days_ahead)
                
                # Get historical sentiment for context
                historical_df = pd.DataFrame(get_historical_sentiment(30))
                
                if not forecast_df.empty:
                    # Generate forecast visualization
                    fig = generate_prediction_chart(forecast_df, historical_df, confidence=0.9)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation about the model
                    with st.expander("About the Forecast Model"):
                        st.markdown("""
                        <div style="color: #CCCCCC;">
                            <p><strong>How Our Forecasting Works:</strong></p>
                            <p>Our sentiment forecasting model uses machine learning to predict future market sentiment based on:</p>
                            <ul>
                                <li>Historical sentiment patterns over time</li>
                                <li>Market volatility and trend indicators</li>
                                <li>Seasonality and cyclical patterns in market behavior</li>
                                <li>Correlation between sentiment and external market factors</li>
                            </ul>
                            <p>The model is trained on historical data and is updated regularly to improve accuracy. The shaded area represents the confidence interval, indicating the range of potential outcomes.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Insufficient historical data to generate a forecast. More market data is needed.")
        
        with pred_tabs[1]:
            # Price impact tab
            st.markdown("### Price Impact Prediction")
            st.markdown("Forecast how sentiment changes might affect stock prices")
            
            # Stock selection for prediction
            selected_stock = st.selectbox(
                "Select stock for price impact prediction",
                st.session_state.selected_stocks,
                key="price_impact_stock"
            )
            
            with st.spinner("Analyzing potential price impact..."):
                # Get forecast data if not already generated
                if 'forecast_df' not in locals():
                    forecast_df = predict_future_sentiment(days_ahead=10)
                
                if not forecast_df.empty:
                    # Get impact analysis for selected stock
                    impact_analysis = forecast_sentiment_impact(selected_stock, forecast_df)
                    
                    # Create metrics row
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Potential Price Impact",
                            f"{impact_analysis['price_impact_pct']:.2f}%",
                            delta=f"{impact_analysis['price_impact_pct']:.2f}%",
                            delta_color="normal"
                        )
                    
                    with col2:
                        st.metric(
                            "Confidence Level",
                            f"{impact_analysis['confidence_level']:.0%}",
                            help="Model's confidence in the prediction"
                        )
                        
                    with col3:
                        st.metric(
                            "Time Horizon",
                            f"{impact_analysis['time_horizon']} days",
                            help="Expected time for impact to materialize"
                        )
                    
                    # Get historical stock data to integrate with prediction
                    stock_data = fetch_stock_data(selected_stock, period='1mo')
                    
                    if stock_data is not None and not stock_data.empty:
                        # Integrate price prediction with sentiment forecast
                        price_forecast = integrate_price_prediction(forecast_df, stock_data)
                        
                        # Plot the integrated price forecast
                        fig = go.Figure()
                        
                        # Historical prices
                        fig.add_trace(go.Scatter(
                            x=price_forecast['Date'][price_forecast['Type'] == 'Historical'],
                            y=price_forecast['Price'][price_forecast['Type'] == 'Historical'],
                            name='Historical Price',
                            line=dict(color='#4CAF50', width=2)
                        ))
                        
                        # Forecasted prices
                        fig.add_trace(go.Scatter(
                            x=price_forecast['Date'][price_forecast['Type'] == 'Forecast'],
                            y=price_forecast['Price'][price_forecast['Type'] == 'Forecast'],
                            name='Forecasted Price',
                            line=dict(color='#7B68EE', width=2, dash='dash')
                        ))
                        
                        # Add confidence intervals
                        fig.add_trace(go.Scatter(
                            x=price_forecast['Date'][price_forecast['Type'] == 'Forecast'],
                            y=price_forecast['Upper Bound'][price_forecast['Type'] == 'Forecast'],
                            fill=None,
                            mode='lines',
                            line=dict(color='rgba(123, 104, 238, 0)'),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=price_forecast['Date'][price_forecast['Type'] == 'Forecast'],
                            y=price_forecast['Lower Bound'][price_forecast['Type'] == 'Forecast'],
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(123, 104, 238, 0)'),
                            fillcolor='rgba(123, 104, 238, 0.2)',
                            name='Confidence Interval'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{selected_stock} - Price Forecast Based on Sentiment",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(30,30,46,0.3)",
                            font=dict(color="#E0E0E0"),
                            title_font_color="#7B68EE",
                            xaxis=dict(
                                gridcolor="rgba(123,104,238,0.15)",
                                zerolinecolor="rgba(123,104,238,0.15)"
                            ),
                            yaxis=dict(
                                gridcolor="rgba(123,104,238,0.15)",
                                zerolinecolor="rgba(123,104,238,0.15)"
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                bgcolor="rgba(30,30,46,0.5)",
                                bordercolor="rgba(123,104,238,0.5)"
                            ),
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display impact factors
                        st.subheader("Impact Factors")
                        
                        for factor in impact_analysis['impact_factors']:
                            st.markdown(f"""
                            <div style="padding: 10px; background-color: rgba(123, 104, 238, 0.1); 
                                        border-radius: 5px; margin-bottom: 10px; border-left: 3px solid #7B68EE">
                                <p style="margin: 0; color: #E0E0E0;"><strong>{factor['name']}:</strong> {factor['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"Unable to retrieve historical data for {selected_stock}")
                else:
                    st.warning("Insufficient historical data to generate price impact prediction.")
        
        with pred_tabs[2]:
            # AI insights tab
            st.markdown("### AI-Generated Market Insights")
            st.markdown("Advanced analysis of predicted market trends and potential investment strategies")
            
            with st.spinner("Generating AI insights..."):
                # Get forecast data if not already generated
                if 'forecast_df' not in locals():
                    forecast_df = predict_future_sentiment(days_ahead=10)
                
                if not forecast_df.empty:
                    # Get AI-generated insights text
                    insights_html = get_sentiment_insights(forecast_df)
                    
                    st.markdown(insights_html, unsafe_allow_html=True)
                    
                    # Add action suggestions based on insights
                    st.subheader("Suggested Actions")
                    
                    actions_col1, actions_col2 = st.columns(2)
                    
                    with actions_col1:
                        st.markdown("""
                        <div style="padding: 15px; background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(46, 125, 50, 0.1) 100%); 
                                     border-radius: 5px; height: 100%;">
                            <h4 style="color: #4CAF50; margin-top: 0;">📈 If Bullish Scenario Materializes</h4>
                            <ul style="color: #CCCCCC; padding-left: 20px;">
                                <li>Consider increasing positions in cyclical stocks</li>
                                <li>Review portfolio for potential profit-taking opportunities</li>
                                <li>Monitor for momentum breakouts in key sectors</li>
                                <li>Consider reduced hedging positions</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with actions_col2:
                        st.markdown("""
                        <div style="padding: 15px; background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(183, 28, 28, 0.1) 100%); 
                                     border-radius: 5px; height: 100%;">
                            <h4 style="color: #F44336; margin-top: 0;">📉 If Bearish Scenario Materializes</h4>
                            <ul style="color: #CCCCCC; padding-left: 20px;">
                                <li>Consider defensive positioning in utilities and consumer staples</li>
                                <li>Review stop-loss levels on more volatile positions</li>
                                <li>Evaluate increasing cash allocations</li>
                                <li>Consider hedging strategies for key positions</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Disclaimer
                    st.markdown("""
                    <div style="margin-top: 20px; padding: 10px; background-color: rgba(33, 33, 33, 0.4); border-radius: 5px; font-size: 12px;">
                        <p style="margin: 0; color: #AAAAAA;"><strong>Disclaimer:</strong> These insights are generated by AI based on historical patterns and current market sentiment. 
                        They should not be considered financial advice. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Insufficient historical data to generate AI insights.")
    
    except Exception as e:
        st.error(f"Error generating predictive analytics: {str(e)}")
        
    # Close the predictive analytics card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacer between cards
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

# Initialize page navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "dashboard"
    
if "show_auth" not in st.session_state:
    st.session_state["show_auth"] = False
    
if "show_subscription" not in st.session_state:
    st.session_state["show_subscription"] = False
    
if "show_profile" not in st.session_state:
    st.session_state["show_profile"] = False
    
if "show_payment" not in st.session_state:
    st.session_state["show_payment"] = False
    
if "payment_plan" not in st.session_state:
    st.session_state["payment_plan"] = "basic"
    
if "payment_success" not in st.session_state:
    st.session_state["payment_success"] = False
    
if "payment_session_id" not in st.session_state:
    st.session_state["payment_session_id"] = None

# Parse URL for OAuth callbacks
if "code" in st.query_params and "state" in st.query_params:
    # Handle OAuth callback
    code = st.query_params["code"]
    state = st.query_params["state"]
    
    # Determine which OAuth provider to use based on the current URL path
    # Try to parse the current URL or use other indicators to determine callback type
    current_url = st.query_params.get('_stcore_permalink', '')
    
    # If we still can't determine, fall back to a simple check of query parameters
    if not current_url:
        current_url = ""
    
    # Store the callback path in session state
    if 'facebook_callback' in current_url:
        st.session_state['oauth_callback_path'] = 'facebook_callback'
    else:
        # Default to Google callback
        st.session_state['oauth_callback_path'] = 'callback'  # Google uses /callback
    
    callback_path = st.session_state.get('oauth_callback_path', '')
    
    if callback_path == 'facebook_callback':
        # Process Facebook callback
        success = handle_facebook_callback(state, code)
    else:
        # Use Google callback (default)
        success = handle_google_callback(state, code)
        
    # Log the authentication process for debugging
    print(f"OAuth callback processed: {callback_path}, Success: {success}")
    
    # Clear query parameters to avoid processing callback twice
    st.query_params.clear()
    
    if success:
        st.session_state["show_auth"] = False
        st.rerun()
        
# Check for payment success callback
if "session_id" in st.query_params and "payment_success" in st.session_state and st.session_state["payment_success"]:
    st.session_state["payment_session_id"] = st.query_params["session_id"]
    st.query_params.clear()

# Navigation bar - always show at the top
def show_navigation():
    """Display navigation bar with user info"""
    cols = st.columns([8, 2])
    
    with cols[0]:
        st.title("Neufin Market Intelligence")
    
    with cols[1]:
        if is_authenticated():
            user = get_current_user()
            
            # User dropdown menu
            user_menu = st.selectbox(
                "👤 " + user["username"],
                ["Dashboard", "My Profile", "Subscription", "Logout"],
                label_visibility="collapsed"
            )
            
            if user_menu == "My Profile":
                st.session_state["show_profile"] = True
                st.rerun()
            elif user_menu == "Subscription":
                st.session_state["show_subscription"] = True
                st.rerun()
            elif user_menu == "Logout":
                logout_user()
                st.rerun()
        else:
            if st.button("Login / Signup", key="nav_login_button"):
                st.session_state["show_auth"] = True
                st.rerun()
    
    st.markdown("---")

# Mercury-inspired navigation at the top with Neufin logo
st.markdown("""
<div class="mercury-nav">
    <div class="mercury-nav-logo">
        <img src="data:image/png;base64,""" + open('neufin_new_logo_base64.txt', 'r').read() + """" class="neufin-main-logo" alt="Neufin AI Logo">
    </div>
    <div class="mercury-nav-menu">
        <div class="mercury-nav-menu-item active" onclick="navClick('dashboard')" id="nav-dashboard">Dashboard</div>
        <div class="mercury-nav-menu-item" onclick="navClick('markets')" id="nav-markets">Markets</div>
        <div class="mercury-nav-menu-item" onclick="navClick('portfolio')" id="nav-portfolio">Portfolio</div>
        <div class="mercury-nav-menu-item" onclick="navClick('insights')" id="nav-insights">Insights</div>
        <div class="mercury-nav-menu-item" onclick="navClick('about')" id="nav-about">About</div>
    </div>
</div>

<script>
function navClick(page) {
    // Set all menu items to inactive
    document.querySelectorAll('.mercury-nav-menu-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Set clicked item as active
    document.getElementById('nav-' + page).classList.add('active');
    
    // Store the selected page in localStorage
    localStorage.setItem('neufin_selected_page', page);
    
    // In a real app, we'd navigate to the page
    // For demo purposes, we'll just update the UI
    if (page !== 'dashboard') {
        // Show "coming soon" message for unimplemented pages
        const main = document.querySelector('main');
        if (main) {
            const existingMsg = document.getElementById('coming-soon-msg');
            if (!existingMsg) {
                const msg = document.createElement('div');
                msg.id = 'coming-soon-msg';
                msg.style.position = 'fixed';
                msg.style.top = '50%';
                msg.style.left = '50%';
                msg.style.transform = 'translate(-50%, -50%)';
                msg.style.backgroundColor = 'rgba(123, 104, 238, 0.9)';
                msg.style.color = 'white';
                msg.style.padding = '20px';
                msg.style.borderRadius = '10px';
                msg.style.zIndex = '1000';
                msg.style.textAlign = 'center';
                msg.innerHTML = '<h3>' + page.charAt(0).toUpperCase() + page.slice(1) + ' section coming soon!</h3><p>This feature is under development.</p><button onclick="this.parentNode.remove()">Close</button>';
                main.appendChild(msg);
            }
        }
    }
}

// Restore selected page on load
document.addEventListener('DOMContentLoaded', function() {
    const selectedPage = localStorage.getItem('neufin_selected_page');
    if (selectedPage) {
        navClick(selectedPage);
    }
});
</script>
""", unsafe_allow_html=True)

# Show auth UI if requested
if st.session_state["show_auth"]:
    show_login_ui()
    
    # Don't show main content when auth UI is visible
    st.stop()

# Show profile UI if requested
if st.session_state["show_profile"]:
    show_user_profile_ui()
    
    if st.button("Back to Dashboard", key="back_from_profile"):
        st.session_state["show_profile"] = False
        st.rerun()
        
    # Don't show main content when profile UI is visible
    st.stop()

# Show subscription management if requested
if st.session_state["show_subscription"]:
    user = get_current_user()
    if user:
        show_subscription_management(user["id"])
    else:
        st.error("User information not available. Please log in again.")
        if st.button("Back to Dashboard", key="back_from_sub_error"):
            st.session_state["show_subscription"] = False
            st.rerun()
    
    if st.button("Back to Dashboard", key="back_from_sub"):
        st.session_state["show_subscription"] = False
        st.rerun()
        
    # Don't show main content when subscription UI is visible
    st.stop()

# Show payment UI if requested
if st.session_state.get("redirect_to_payment", False):
    user = get_current_user()
    if user:
        plan = st.session_state["payment_plan"]
        show_payment_ui(user["id"], plan)
    else:
        st.error("User information not available. Please log in again.")
    
    if st.button("Back to Dashboard", key="back_from_payment"):
        st.session_state["redirect_to_payment"] = False
        st.rerun()
        
    # Don't show main content when payment UI is visible
    st.stop()

# Show payment success UI if needed
if "payment_success" in st.query_params:
    session_id = st.session_state.get("payment_session_id")
    show_payment_success_ui(session_id)
    
    # Don't show main content when payment success UI is visible
    st.stop()

# Custom CSS for Mercury-inspired Neufin design
st.markdown("""
<style>
    /* Base Styling - Mercury inspired */
    .stApp {
        background-color: #0F1117;
        color: #E0E0E0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    p {
        color: #ADB3C9;
    }
    
    /* Mercury-inspired card styling */
    .neufin-card {
        background: #171924;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #2A2D3A;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        margin-bottom: 20px;
        transition: all 0.2s ease;
    }
    
    .neufin-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border: 1px solid #3A3D4A;
    }
    
    .premium-features {
        background: #171924;
        border-top: 1px solid #2A2D3A;
        border-left: 1px solid #2A2D3A;
    }
    
    .neufin-headline {
        background: #171924;
        color: white;
        padding: 30px;
        border-radius: 8px;
        text-align: left;
        margin-bottom: 30px;
        border: 1px solid #2A2D3A;
    }
    
    .glow-text {
        color: #7B68EE;
    }
    
    /* Mercury-inspired navigation */
    .mercury-nav {
        display: flex;
        align-items: center;
        background-color: #171924;
        padding: 16px 20px;
        border-radius: 8px;
        border-bottom: 1px solid #2A2D3A;
        margin-bottom: 24px;
    }
    
    .mercury-nav-logo {
        display: flex;
        align-items: center;
    }
    
    .neufin-main-logo {
        height: 60px;
        margin-right: 15px;
    }
    
    .mercury-nav-menu {
        display: flex;
        gap: 24px;
        margin-left: 40px;
    }
    
    .mercury-nav-menu-item {
        color: #ADB3C9;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 12px;
        cursor: pointer;
        position: relative;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    
    .mercury-nav-menu-item.active {
        color: #7B68EE;
        background-color: rgba(123, 104, 238, 0.1);
    }
    
    .mercury-nav-menu-item.active:after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background-color: #7B68EE;
        border-radius: 1px;
    }
    
    .mercury-nav-menu-item:hover {
        color: #FFFFFF;
        background-color: rgba(123, 104, 238, 0.05);
    }
    
    /* Data metrics styling */
    .data-metric {
        background: #171924;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2A2D3A;
        transition: transform 0.2s;
    }
    
    .data-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .data-metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #E0E0E0;
    }
    
    .data-metric-label {
        font-size: 13px;
        color: #ADB3C9;
        margin-bottom: 4px;
    }
    
    /* Mercury-style dashboard sections */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 16px;
    }
    
    .metric-card {
        flex: 1;
        background: #171924;
        border-radius: 8px;
        border: 1px solid #2A2D3A;
        padding: 16px;
    }
    
    .metric-card-title {
        font-size: 13px;
        color: #ADB3C9;
        margin-bottom: 8px;
    }
    
    .metric-card-value {
        font-size: 20px;
        font-weight: 600;
        color: #E0E0E0;
    }
    
    /* Sidebar styling */
    .css-1cypcdb, .css-d1kyf5, .css-z5fcl4 {
        background-color: #171924 !important;
        border-right: 1px solid #2A2D3A !important;
    }
    
    /* Mercury-styled buttons */
    .stButton > button {
        background-color: #7B68EE;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 16px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #6B58DE;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(123, 104, 238, 0.2);
    }
    
    /* Feature pills in the footer */
    .feature-pill {
        text-align: center;
        padding: 8px 12px;
        background-color: #171924;
        border: 1px solid #2A2D3A;
        border-radius: 6px;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .feature-pill:hover {
        background-color: rgba(123, 104, 238, 0.1);
        border: 1px solid #7B68EE;
        transform: translateY(-1px);
    }
    
    .feature-icon {
        margin-right: 8px;
        font-size: 16px;
        color: #7B68EE;
    }
    
    .feature-text {
        color: #E0E0E0;
        font-weight: 500;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: #171924 !important;
        color: #7B68EE !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #2A2D3A !important;
        text-align: left !important;
        padding: 12px 16px !important;
    }
    
    .dataframe td {
        background-color: #171924 !important;
        color: #E0E0E0 !important;
        border: none !important;
        border-bottom: 1px solid #2A2D3A !important;
        padding: 12px 16px !important;
    }
    
    /* Mercury-style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #2A2D3A;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0;
        padding: 8px 16px;
        margin-right: 0;
        color: #ADB3C9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #7B68EE;
        color: #E0E0E0;
    }
    
    /* Chart background */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
    
    /* Real-time update styling */
    .real-time-badge {
        display: inline-block;
        background-color: rgba(0, 200, 83, 0.1);
        color: #00C853;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 4px;
        margin-left: 10px;
    }
    
    /* Toast styling */
    .stToast {
        background-color: #171924 !important;
        color: #E0E0E0 !important;
        border: 1px solid #2A2D3A !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    /* Mercury sidebar menu */
    .sidebar-menu {
        margin-top: 24px;
    }
    
    .sidebar-menu-item {
        padding: 10px 20px;
        display: flex;
        align-items: center;
        color: #ADB3C9;
        font-weight: 500;
        margin-bottom: 2px;
        cursor: pointer;
        border-left: 3px solid transparent;
    }
    
    .sidebar-menu-item:hover {
        background-color: rgba(123, 104, 238, 0.1);
        color: #E0E0E0;
    }
    
    .sidebar-menu-item.active {
        background-color: rgba(123, 104, 238, 0.1);
        color: #7B68EE;
        border-left: 3px solid #7B68EE;
    }
    
    .sidebar-menu-icon {
        margin-right: 12px;
        width: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration already set at the top of the file

# Initialize session state variables
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
if 'time_period' not in st.session_state:
    st.session_state.time_period = '1mo'
if 'refresh_data' not in st.session_state:
    st.session_state.refresh_data = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'ai_analyses' not in st.session_state:
    st.session_state.ai_analyses = {}
# Real-time update settings
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 60  # seconds
if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = datetime.now()
# Demo showcase mode toggle
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = False

# Mercury-inspired headline with clean design
# Add real-time badge if auto-refresh is enabled
real_time_badge = """<span class="real-time-badge">🔄 Real-time</span>""" if st.session_state.auto_refresh else ""
# Add headline with real-time badge if enabled
headline_html = f"""
<div class="neufin-headline">
    <div class="mercury-header-content">
        <h1 class="glow-text">🔮 Neufin Dashboard {real_time_badge}</h1>
        <p class="mercury-header-subtitle">AI-Powered Market Analytics Platform</p>
    </div>
</div>
"""

# Define CSS separately to avoid f-string issues
css_styles = """
<style>
    .neufin-headline {
        padding: 24px 30px;
        margin-bottom: 24px;
        position: relative;
    }
    
    .mercury-header-content {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    
    .mercury-header-subtitle {
        color: #ADB3C9;
        font-size: 14px;
        margin-top: 0;
    }
    
    .glow-text {
        font-weight: 700;
        margin: 0;
        font-size: 24px;
    }
</style>
"""

# Render both parts
st.markdown(headline_html, unsafe_allow_html=True)
st.markdown(css_styles, unsafe_allow_html=True)

# Mercury-inspired introduction with feature cards
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-card-title">Market Overview</div>
        <div class="metric-card-value">
            <span style="color:#7B68EE; font-weight:600;">Neufin</span> Neural Powered Finance Unlocked
        </div>
        <p style="margin-top:12px; color:#ADB3C9; font-size:14px;">
            Our platform combines AI analysis with real-time financial data to provide you with the most accurate market insights.
        </p>
    </div>
    <div class="metric-card">
        <div class="metric-card-title">Today's Date</div>
        <div class="metric-card-value">April 14, 2025</div>
        <p style="margin-top:12px; color:#ADB3C9; font-size:14px;">
            Welcome to your personal financial dashboard.
        </p>
    </div>
</div>

<div class="mercury-section-title">
    <div class="mercury-section-title-icon">⚡</div>
    Featured Insights
</div>

<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-card-icon">📊</div>
        <div class="feature-card-title">AI-Powered Analysis</div>
        <div class="feature-card-description">Advanced neural networks analyze market patterns and trends</div>
    </div>
    <div class="feature-card">
        <div class="feature-card-icon">🔍</div>
        <div class="feature-card-title">Sentiment Tracking</div>
        <div class="feature-card-description">Real-time analysis of market sentiment across global markets</div>
    </div>
    <div class="feature-card">
        <div class="feature-card-icon">💰</div>
        <div class="feature-card-title">Investment Insights</div>
        <div class="feature-card-description">Personalized investment recommendations based on market conditions</div>
    </div>
    <div class="feature-card">
        <div class="feature-card-icon">🌐</div>
        <div class="feature-card-title">Global Markets</div>
        <div class="feature-card-description">Comprehensive coverage of financial markets worldwide</div>
    </div>
</div>

<style>
    .mercury-section-title {
        font-size: 16px;
        font-weight: 600;
        color: #E0E0E0;
        margin: 24px 0 16px 0;
        display: flex;
        align-items: center;
    }

    .mercury-section-title-icon {
        color: #7B68EE;
        margin-right: 8px;
        font-size: 18px;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .feature-card {
        background: #171924;
        border: 1px solid #2A2D3A;
        border-radius: 8px;
        padding: 20px;
        transition: all 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid #3A3D4A;
    }

    .feature-card-icon {
        font-size: 24px;
        margin-bottom: 12px;
        color: #7B68EE;
    }

    .feature-card-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #E0E0E0;
    }

    .feature-card-description {
        font-size: 14px;
        color: #ADB3C9;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for filters, inputs, and account features
with st.sidebar:
    # Mercury-style branding with Neufin logo
    st.markdown("""
    <div class="mercury-sidebar-brand">
        <img src="data:image/png;base64,""" + open('neufin_new_logo_base64.txt', 'r').read() + """" class="neufin-sidebar-logo" alt="Neufin AI Logo">
        <div class="mercury-brand-text">
            <div class="mercury-brand-tagline">NEURAL POWERED FINANCE UNLOCKED</div>
        </div>
    </div>
    
    <style>
        .mercury-sidebar-brand {
            display: flex;
            align-items: center;
            padding: 0 10px 20px 10px;
            margin-bottom: 20px;
            border-bottom: 1px solid #2A2D3A;
        }
        
        .neufin-sidebar-logo {
            width: 80px;
            margin-right: 10px;
        }
        
        .mercury-brand-text {
            display: flex;
            flex-direction: column;
        }
        
        .mercury-brand-name {
            font-weight: 700;
            font-size: 18px;
            color: #7B68EE;
            line-height: 1.2;
        }
        
        .mercury-brand-tagline {
            font-size: 10px;
            color: #ADB3C9;
            letter-spacing: 0.5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Mercury-style sidebar navigation
    st.markdown("""
    <div class="sidebar-menu">
        <div class="sidebar-menu-item active">
            <div class="sidebar-menu-icon">📊</div>
            Dashboard
        </div>
        <div class="sidebar-menu-item">
            <div class="sidebar-menu-icon">💰</div>
            Markets
        </div>
        <div class="sidebar-menu-item">
            <div class="sidebar-menu-icon">📈</div>
            Portfolio
        </div>
        <div class="sidebar-menu-item">
            <div class="sidebar-menu-icon">🧠</div>
            AI Analysis
        </div>
        <div class="sidebar-menu-item">
            <div class="sidebar-menu-icon">⚙️</div>
            Settings
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom styled login/account container
    st.markdown('<div class="neufin-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
    
    # Show user info if logged in, otherwise show login button
    if is_authenticated():
        user = get_current_user()
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 15px;">
            <h3 style="color: #E0E0E0; margin-bottom: 5px;">👤 {user['username']}</h3>
            <p style="color: #AAAAAA; font-size: 12px;">{user['email']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show account management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("My Profile", use_container_width=True):
                st.session_state["show_profile"] = True
                st.rerun()
        with col2:
            if st.button("Subscription", use_container_width=True):
                st.session_state["show_subscription"] = True
                st.rerun()
                
        # Logout button
        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()
    else:
        st.markdown("""
        <h3 style="color: #7B68EE; text-align: center; margin-bottom: 15px;">Account Access</h3>
        <p style="text-align: center; color: #AAAAAA; margin-bottom: 15px;">
            Log in to access premium features and personalized insights.
        </p>
        """, unsafe_allow_html=True)
        
        if st.button("Login / Sign Up", use_container_width=True, key="sidebar_login_button"):
            st.session_state["show_auth"] = True
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="neufin-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #7B68EE; margin-bottom: 15px;">Dashboard Settings</h3>', unsafe_allow_html=True)
    
    # Market Region Selection
    market_prefs = get_market_preferences()
    st.markdown('<p style="color: #AAA; font-size: 14px; margin-bottom: 5px;">Market Region</p>', unsafe_allow_html=True)
    
    market_options = {
        'global': '🌎 Global Markets',
        'us': '🇺🇸 United States'
    }
    
    selected_market = st.selectbox(
        "Select market region",
        options=list(market_options.keys()),
        format_func=lambda x: market_options[x],
        index=list(market_options.keys()).index(market_prefs['selected_market']),
        key="market_region_selector",
        label_visibility="collapsed"
    )
    
    # Update preferences if changed
    if selected_market != market_prefs['selected_market']:
        # Clear selected_stocks when switching markets to avoid 
        # using stocks from one market with another market's exchange suffix
        if 'selected_stocks' in st.session_state:
            del st.session_state.selected_stocks
            
        update_market_preferences('selected_market', selected_market)
        st.session_state.refresh_data = True
        st.rerun()  # Refresh to update all components with new market
    
    st.markdown('<hr style="margin: 15px 0px; border-color: rgba(123, 104, 238, 0.2);">', unsafe_allow_html=True)
    
    # Stock selection based on selected market
    st.markdown('<p style="color: #AAA; font-size: 14px; margin-bottom: 5px;">Stocks & Indices</p>', unsafe_allow_html=True)
    
    # Get appropriate stocks based on market selection
    if selected_market == 'us':
        default_stocks = market_prefs['default_tickers']['us']
        available_stocks = get_market_indices() + default_stocks
    else:  # global
        default_stocks = market_prefs['default_tickers']['global']
        available_stocks = get_market_indices() + default_stocks
    
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = default_stocks[:3]  # Start with first 3 default stocks
    
    selected_stocks = st.multiselect(
        "Select stocks/indices to analyze",
        options=available_stocks,
        default=st.session_state.selected_stocks,
        label_visibility="collapsed"
    )
    
    if selected_stocks != st.session_state.selected_stocks:
        st.session_state.selected_stocks = selected_stocks
        st.session_state.refresh_data = True
    
    # Time period selection
    time_period = st.selectbox(
        "Select time period",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y'],
        index=2  # Default to 1 month
    )
    
    if time_period != st.session_state.time_period:
        st.session_state.time_period = time_period
        st.session_state.refresh_data = True
    
    # Real-time data updates section
    st.markdown('<div style="margin-top: 20px; border-top: 1px solid rgba(123, 104, 238, 0.3); padding-top: 15px;">', unsafe_allow_html=True)
    st.markdown('<h4 style="color: #7B68EE; margin-bottom: 10px;">Real-Time Updates</h4>', unsafe_allow_html=True)
    
    # Auto-refresh toggle
    auto_refresh = st.toggle("Enable Real-Time Updates", 
                             value=st.session_state.auto_refresh,
                             help="Automatically refresh data at regular intervals")
                             
    # Add warning if real-time updates are disabled
    if not auto_refresh:
        st.markdown("""
        <div style="background-color: rgba(255, 82, 82, 0.2); border-left: 4px solid #FF5252; padding: 10px; margin: 10px 0; border-radius: 4px;">
            <p style="margin: 0; color: #FFF; font-size: 13px;">
                <strong>Real-time updates disabled.</strong> The data won't update automatically.
                Click "Refresh Now" to manually update the dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if auto_refresh != st.session_state.auto_refresh:
        st.session_state.auto_refresh = auto_refresh
        st.session_state.last_auto_refresh = datetime.now()
    
    # Refresh interval selection (only shown if auto-refresh is enabled)
    if st.session_state.auto_refresh:
        refresh_interval = st.select_slider(
            "Refresh Interval",
            options=[30, 60, 120, 300, 600],
            value=st.session_state.refresh_interval,
            format_func=lambda x: f"{x} seconds" if x < 60 else f"{x//60} minute{'s' if x//60 > 1 else ''}"
        )
        
        if refresh_interval != st.session_state.refresh_interval:
            st.session_state.refresh_interval = refresh_interval
            st.session_state.last_auto_refresh = datetime.now()
    
    # Manual refresh button with custom styling
    st.markdown("""
    <style>
    .refresh-button {
        background-color: rgba(123, 104, 238, 0.2);
        color: #7B68EE;
        padding: 8px 15px;
        font-weight: bold;
        border: 1px solid rgba(123, 104, 238, 0.3);
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: 8px;
        margin-bottom: 8px;
        width: 100%;
    }
    .refresh-button:hover {
        background-color: rgba(123, 104, 238, 0.4);
        box-shadow: 0 0 10px rgba(123, 104, 238, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom refresh button
    if st.button("🔄 Refresh Data Now", key="manual_refresh", use_container_width=True):
        st.session_state.refresh_data = True
        st.session_state.last_auto_refresh = datetime.now()
    
    # Last updated indicator
    if st.session_state.auto_refresh:
        seconds_since_refresh = (datetime.now() - st.session_state.last_auto_refresh).total_seconds()
        progress_value = min(seconds_since_refresh / st.session_state.refresh_interval, 1.0)
        
        # Display time until next refresh
        seconds_until_refresh = max(st.session_state.refresh_interval - seconds_since_refresh, 0)
        time_until_refresh = f"{int(seconds_until_refresh)}s" if seconds_until_refresh < 60 else f"{int(seconds_until_refresh/60)}m {int(seconds_until_refresh%60)}s"
        
        st.markdown(f"""
        <div style="margin-top: 10px; font-size: 12px; color: #888;">
            Next refresh in: {time_until_refresh}
        </div>
        """, unsafe_allow_html=True)
        
        # Visual progress bar for time until next refresh
        st.progress(progress_value, "Refreshing soon...")
    
    # Always show last updated timestamp
    st.markdown(f"""
    <div style="margin-top: 10px; font-size: 12px; color: #888;">
        Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close the settings card
    
    # Demo showcase toggle card with futuristic design
    st.markdown('<div class="neufin-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #7B68EE; margin-bottom: 15px;">Try the Demo Showcase</h3>', unsafe_allow_html=True)
    
    # Demo mode toggle with unique key
    demo_mode = st.toggle("Show Interactive Demo", 
                         value=st.session_state.show_demo,
                         help="Explore all Neufin features in demo mode",
                         key="demo_toggle")
                         
    if demo_mode != st.session_state.show_demo:
        st.session_state.show_demo = demo_mode
        # Force a refresh when changing between modes
        st.rerun()
    
    st.markdown("""
    <p style="font-size: 13px; color: #AAA; margin-top: 10px;">
    Experience the full power of Neufin with our interactive demo showcase. 
    Explore AI-powered investment insights and advanced market analytics.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About section with futuristic design
    st.markdown('<div class="neufin-card" style="padding: 15px; margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #7B68EE; margin-bottom: 15px;">About Neufin</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 14px; margin-bottom: 10px;">
    Neufin is a cutting-edge financial intelligence platform powered by AI and designed to transform the way individuals and institutions interact with markets. Built to deliver real-time sentiment insights, predictive analytics, and personalized investment intelligence, Neufin empowers users to make smarter, data-driven financial decisions.
    </p>
    
    <p style="font-size: 14px; margin-bottom: 15px;">
    With an intuitive dark-themed interface and a vision to democratize advanced market tools, Neufin stands at the intersection of technology, finance, and user-centric design. It is a unit of Ctech Ventures, and officially operates under Neufin OÜ, a legally registered entity in Estonia.
    </p>

    <div style="margin-top: 15px; border-left: 3px solid #7B68EE; padding-left: 10px;">
        <p style="font-size: 13px; color: #AAA;">
        <strong style="color: #7B68EE;">Premium Features:</strong><br>
        • AI-generated investment recommendations<br>
        • Global trade impact analysis<br>
        • Detailed sector forecasting<br>
        • Personalized portfolio insights
        </p>
    </div>
    
    <div style="margin-top: 20px; font-size: 12px; color: #888; border-top: 1px solid #2A2D3A; padding-top: 15px;">
        <p>The registered address for Neufin OÜ – A Unit of Ctech Ventures is Järvevana tee 9, 11314, Tallinn, Estonia.</p>
        <p style="margin-top: 8px;">© 2025 Neufin OÜ. All rights reserved. Registered in Estonia.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Demo showcase section with previews of all features
def load_demo_showcase():
    # Add special styling for demo mode
    st.markdown("""
    <style>
    /* Demo mode banner */
    .demo-mode-banner {
        background: linear-gradient(90deg, rgba(123,104,238,0.15) 0%, rgba(123,104,238,0.3) 50%, rgba(123,104,238,0.15) 100%);
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(123,104,238,0.4);
        text-align: center;
        animation: pulse 3s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(123,104,238,0.4); }
        70% { box-shadow: 0 0 0 10px rgba(123,104,238,0); }
        100% { box-shadow: 0 0 0 0 rgba(123,104,238,0); }
    }
    
    /* Special styles for demo cards */
    .demo-card {
        border: 1px solid rgba(123,104,238,0.3);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background: linear-gradient(180deg, rgba(30,30,30,0.6) 0%, rgba(50,50,50,0.3) 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .demo-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(123,104,238,0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Demo mode banner
    st.markdown("""
    <div class="demo-mode-banner">
        <h2 style="color: #7B68EE; margin: 0;">🎮 DEMO MODE</h2>
        <p style="margin: 5px 0 0 0; color: #e0e0e0;">Exploring Neufin's Interactive Demo Showcase</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #7B68EE; margin-bottom: 15px;">🎮 Interactive Demo Showcase</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 14px; margin-bottom: 20px;">
    Experience all of Neufin's powerful features in this interactive demo. Explore our AI-powered insights, 
    market analysis, and investment tools without signing up.
    </p>
    <div style="background-color: rgba(123, 104, 238, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <p style="margin: 0; font-style: italic; color: #CCC;">
        Note: This is a simulated demonstration with sample data. Switch back to the regular dashboard
        using the toggle in the sidebar for real-time market analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for each feature demo
    demo_tabs = st.tabs([
        "🔮 Market Sentiment", 
        "📈 Stock Analysis", 
        "📊 Sector Performance", 
        "🤖 AI Investment Advisor",
        "🌎 Global Trade Impact"
    ])
    
    # 1. Market Sentiment Demo Tab
    with demo_tabs[0]:
        st.subheader("Market Sentiment Analysis")
        
        # Sample sentiment visualization
        sentiment_score = 0.35  # Sample positive sentiment
        
        # Create sentiment gauge
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Sentiment meter visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Market Sentiment", 'font': {'color': 'white', 'size': 16}},
                gauge = {
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#7B68EE"},
                    'bgcolor': "rgba(50, 50, 50, 0.8)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-1, -0.5], 'color': 'rgba(255, 65, 54, 0.5)'},
                        {'range': [-0.5, 0], 'color': 'rgba(255, 144, 14, 0.5)'},
                        {'range': [0, 0.5], 'color': 'rgba(44, 160, 101, 0.5)'},
                        {'range': [0.5, 1], 'color': 'rgba(44, 160, 44, 0.5)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_score
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor = 'rgba(0,0,0,0)',
                plot_bgcolor = 'rgba(0,0,0,0)',
                font = {'color': 'white'},
                height = 250,
                margin = dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: rgba(123, 104, 238, 0.1); border-radius: 10px; padding: 15px; height: 250px; display: flex; flex-direction: column; justify-content: center;">
                <h4 style="color: #7B68EE; margin-bottom: 15px;">Sentiment Factors</h4>
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>News Sentiment:</span>
                        <span style="color: #4CAF50;">+0.42</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: 71%; background-color: #4CAF50;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Technical Analysis:</span>
                        <span style="color: #FFA726;">+0.15</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: 58%; background-color: #FFA726;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Market Indices:</span>
                        <span style="color: #42A5F5;">+0.38</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: 69%; background-color: #42A5F5;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Demo market news
        st.subheader("Recent Market News")
        demo_news = [
            {"title": "Fed signals potential rate cuts in upcoming meeting", "sentiment": 0.45},
            {"title": "Tech sector leads market rally amid strong earnings", "sentiment": 0.78},
            {"title": "Supply chain issues persist for manufacturing sector", "sentiment": -0.32},
            {"title": "New fiscal policy expected to boost economic growth", "sentiment": 0.52}
        ]
        
        for news in demo_news:
            sentiment_color = get_sentiment_color(news["sentiment"])
            sentiment_label = format_sentiment_score(news["sentiment"])
            
            st.markdown(f"""
            <div style="padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: rgba(60, 60, 60, 0.3); border-left: 4px solid {sentiment_color};">
                <div style="display: flex; justify-content: space-between;">
                    <div style="flex-grow: 1;">{news["title"]}</div>
                    <div style="color: {sentiment_color}; margin-left: 15px;">{sentiment_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 2. Stock Analysis Demo Tab
    with demo_tabs[1]:
        st.subheader("Stock Performance Analysis")
        
        # Demo stock chart
        chart_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=90),
            'Price': [150 + (i * 0.5 + i**1.1 * 0.05 * (-1 if i % 10 == 0 else 1)) for i in range(90)],
            'MA7': [150 + (i * 0.5) for i in range(90)],
            'MA20': [150 + (i * 0.45) for i in range(90)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data['Date'], 
            y=chart_data['Price'],
            mode='lines',
            name='AAPL',
            line=dict(color='#7B68EE', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=chart_data['Date'], 
            y=chart_data['MA7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='#4CAF50', width=1.5, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=chart_data['Date'], 
            y=chart_data['MA20'],
            mode='lines',
            name='20-Day MA',
            line=dict(color='#FFA726', width=1.5, dash='dot')
        ))
        
        fig.update_layout(
            title="Apple Inc. (AAPL) - Demo Data",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend_title="Indicators",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,30,0.3)',
            font=dict(color='white'),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(80,80,80,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(80,80,80,0.2)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Investment Analysis
        st.subheader("AI Investment Analysis")
        
        demo_analysis = {
            "strengths": [
                "Strong earnings growth (22% YoY)",
                "Increasing market share in premium segment",
                "Robust cash reserves and dividend growth",
                "Innovative product pipeline",
            ],
            "risks": [
                "Supply chain vulnerabilities",
                "Increasing regulatory scrutiny",
                "Competitive pressure in key markets",
                "Currency fluctuation exposure",
            ],
            "recommendation": "Moderate Buy",
            "target_price": "$185.50",
            "confidence": 0.72,
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: rgba(76, 175, 80, 0.1); border-radius: 10px; padding: 15px; height: 100%;">
                <h4 style="color: #4CAF50; margin-bottom: 15px;">Strengths</h4>
                <ul style="list-style-type: none; padding-left: 0;">
            """, unsafe_allow_html=True)
            
            for strength in demo_analysis["strengths"]:
                st.markdown(f"""
                <li style="margin-bottom: 8px; display: flex;">
                    <span style="color: #4CAF50; margin-right: 8px;">✓</span>
                    <span>{strength}</span>
                </li>
                """, unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: rgba(244, 67, 54, 0.1); border-radius: 10px; padding: 15px; height: 100%;">
                <h4 style="color: #F44336; margin-bottom: 15px;">Risks</h4>
                <ul style="list-style-type: none; padding-left: 0;">
            """, unsafe_allow_html=True)
            
            for risk in demo_analysis["risks"]:
                st.markdown(f"""
                <li style="margin-bottom: 8px; display: flex;">
                    <span style="color: #F44336; margin-right: 8px;">!</span>
                    <span>{risk}</span>
                </li>
                """, unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Recommendation summary
        st.markdown(f"""
        <div style="margin-top: 20px; background-color: rgba(123, 104, 238, 0.1); border-radius: 10px; padding: 15px; text-align: center;">
            <h4 style="color: #7B68EE; margin-bottom: 10px;">AI Recommendation</h4>
            <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{demo_analysis["recommendation"]}</div>
            <div style="font-size: 18px; margin-bottom: 10px;">Target Price: {demo_analysis["target_price"]}</div>
            <div style="color: #AAA; font-size: 12px;">AI Confidence Score: {demo_analysis["confidence"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 3. Sector Performance Demo Tab
    with demo_tabs[2]:
        st.subheader("Sector Performance Analysis")
        
        # Sample sector data
        sectors = [
            {"Sector": "Technology", "Performance": 12.4, "Trend": "Bullish", "Sentiment": 0.68},
            {"Sector": "Healthcare", "Performance": 8.7, "Trend": "Bullish", "Sentiment": 0.52},
            {"Sector": "Finance", "Performance": 5.3, "Trend": "Neutral", "Sentiment": 0.21},
            {"Sector": "Energy", "Performance": -3.2, "Trend": "Bearish", "Sentiment": -0.35},
            {"Sector": "Consumer Cyclical", "Performance": 7.5, "Trend": "Bullish", "Sentiment": 0.43},
            {"Sector": "Real Estate", "Performance": -1.8, "Trend": "Neutral", "Sentiment": -0.12},
            {"Sector": "Utilities", "Performance": 2.1, "Trend": "Neutral", "Sentiment": 0.18},
            {"Sector": "Materials", "Performance": 4.6, "Trend": "Neutral", "Sentiment": 0.25},
        ]
        
        # Performance chart
        sectors_df = pd.DataFrame(sectors)
        sectors_df = sectors_df.sort_values(by="Performance", ascending=False)
        
        sectors_df["Color"] = sectors_df["Performance"].apply(
            lambda x: "#4CAF50" if x > 5 else "#FFA726" if x > 0 else "#F44336"
        )
        
        fig = go.Figure(go.Bar(
            x=sectors_df["Performance"],
            y=sectors_df["Sector"],
            orientation='h',
            marker_color=sectors_df["Color"],
            text=sectors_df["Performance"].apply(lambda x: f"{x}%"),
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Sector Performance (Last 30 Days)",
            xaxis_title="Performance (%)",
            yaxis_title="Sector",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,30,0.3)',
            font=dict(color='white'),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(80,80,80,0.2)'
            ),
            yaxis=dict(
                showgrid=False,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector insights
        st.subheader("Top Sector AI Insights")
        
        top_sector = sectors_df.iloc[0]
        
        st.markdown(f"""
        <div style="background-color: rgba(123, 104, 238, 0.1); border-radius: 10px; padding: 15px; margin-top: 10px;">
            <h4 style="color: #7B68EE; margin-bottom: 15px;">{top_sector["Sector"]} Sector Outlook</h4>
            <p style="margin-bottom: 15px;">
                The Technology sector is showing strong momentum with a {top_sector["Performance"]}% gain over the past month. 
                This outperformance is driven by strong earnings reports from major companies and increased capital expenditure 
                on cloud infrastructure and AI technologies.
            </p>
            <div style="margin-top: 20px;">
                <h5 style="color: #7B68EE; margin-bottom: 10px;">Key Drivers</h5>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 8px; display: flex;">
                        <span style="color: #4CAF50; margin-right: 8px;">►</span>
                        <span>Accelerating digital transformation initiatives across industries</span>
                    </li>
                    <li style="margin-bottom: 8px; display: flex;">
                        <span style="color: #4CAF50; margin-right: 8px;">►</span>
                        <span>Increased spending on cybersecurity solutions</span>
                    </li>
                    <li style="margin-bottom: 8px; display: flex;">
                        <span style="color: #4CAF50; margin-right: 8px;">►</span>
                        <span>Expanded consumer spending on premium devices</span>
                    </li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 4. AI Investment Advisor Demo Tab
    with demo_tabs[3]:
        st.subheader("AI Investment Recommendations")
        
        # Sample stock recommendations
        recommendations = [
            {"Ticker": "AAPL", "Company": "Apple Inc.", "Score": 92, "Sector": "Technology", "Sentiment": 0.76},
            {"Ticker": "MSFT", "Company": "Microsoft Corp.", "Score": 89, "Sector": "Technology", "Sentiment": 0.72},
            {"Ticker": "AMZN", "Company": "Amazon.com Inc.", "Score": 87, "Sector": "Consumer Cyclical", "Sentiment": 0.68},
            {"Ticker": "UNH", "Company": "UnitedHealth Group", "Score": 84, "Sector": "Healthcare", "Sentiment": 0.65},
            {"Ticker": "V", "Company": "Visa Inc.", "Score": 82, "Sector": "Finance", "Sentiment": 0.63},
        ]
        
        # Create a dataframe for the recommendations
        rec_df = pd.DataFrame(recommendations)
        
        # Style the recommendations table
        st.markdown("""
        <div style="background-color: rgba(123, 104, 238, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h4 style="color: #7B68EE; margin-bottom: 15px;">Top Investment Opportunities</h4>
            <p style="margin-bottom: 15px;">
                Our AI has analyzed market trends, company fundamentals, and sentiment indicators to identify these top 
                investment opportunities with strong growth potential.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the recommendation table
        for i, row in rec_df.iterrows():
            score_color = "#4CAF50" if row["Score"] >= 85 else "#FFA726"
            
            st.markdown(f"""
            <div style="padding: 15px; margin-bottom: 10px; border-radius: 10px; background-color: rgba(40, 40, 40, 0.3); 
                 border-left: 4px solid {score_color}; display: flex; align-items: center;">
                <div style="flex: 0 0 80px; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold; color: {score_color};">{row["Score"]}</div>
                    <div style="font-size: 12px; color: #AAA;">Score</div>
                </div>
                <div style="flex-grow: 1; margin-left: 15px;">
                    <div style="font-size: 18px; font-weight: bold;">{row["Ticker"]}</div>
                    <div style="font-size: 14px; color: #CCC;">{row["Company"]}</div>
                </div>
                <div style="flex: 0 0 120px; text-align: right;">
                    <div style="font-size: 14px; color: #AAA;">{row["Sector"]}</div>
                    <div style="font-size: 14px; color: {get_sentiment_color(row["Sentiment"])};">
                        {format_sentiment_score(row["Sentiment"])}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Strategy recommendations
        st.subheader("Investment Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: rgba(60, 60, 60, 0.3); border-radius: 10px; padding: 15px; height: 100%;">
                <h4 style="color: #7B68EE; margin-bottom: 15px;">Short-Term Strategy (1-3 Months)</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 12px; display: flex;">
                        <span style="color: #7B68EE; margin-right: 8px;">▶</span>
                        <span>Focus on high-quality tech stocks with strong earnings</span>
                    </li>
                    <li style="margin-bottom: 12px; display: flex;">
                        <span style="color: #7B68EE; margin-right: 8px;">▶</span>
                        <span>Consider healthcare companies with innovative products</span>
                    </li>
                    <li style="margin-bottom: 12px; display: flex;">
                        <span style="color: #7B68EE; margin-right: 8px;">▶</span>
                        <span>Reduce exposure to interest-rate sensitive sectors</span>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: rgba(60, 60, 60, 0.3); border-radius: 10px; padding: 15px; height: 100%;">
                <h4 style="color: #7B68EE; margin-bottom: 15px;">Long-Term Outlook (6-12 Months)</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 12px; display: flex;">
                        <span style="color: #7B68EE; margin-right: 8px;">▶</span>
                        <span>Position for AI-driven growth across multiple sectors</span>
                    </li>
                    <li style="margin-bottom: 12px; display: flex;">
                        <span style="color: #7B68EE; margin-right: 8px;">▶</span>
                        <span>Add quality dividend payers for potential volatility</span>
                    </li>
                    <li style="margin-bottom: 12px; display: flex;">
                        <span style="color: #7B68EE; margin-right: 8px;">▶</span>
                        <span>Consider international diversification in select markets</span>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # 5. Global Trade Impact Demo Tab
    with demo_tabs[4]:
        st.subheader("Global Trade Impact Analysis")
        
        # Global trade impact visualization
        st.markdown("""
        <div style="background-color: rgba(123, 104, 238, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h4 style="color: #7B68EE; margin-bottom: 15px;">Current Global Trade Conditions</h4>
            <p style="margin-bottom: 15px;">
                Our AI analysis has identified the following key trade factors affecting global markets and specific 
                investment opportunities.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key trade factors
        factors = [
            {
                "title": "Technology Export Controls",
                "description": "Increasing restrictions on semiconductor and AI technology exports to specific regions",
                "impact": "High",
                "affected_sectors": ["Technology", "Communication Services"],
                "trend": "Increasing"
            },
            {
                "title": "Supply Chain Reshoring",
                "description": "Major corporations relocating manufacturing closer to consumer markets",
                "impact": "Medium",
                "affected_sectors": ["Industrials", "Consumer Goods", "Technology"],
                "trend": "Stable"
            },
            {
                "title": "Agricultural Tariffs",
                "description": "New tariffs on agricultural products between major trading partners",
                "impact": "Medium",
                "affected_sectors": ["Consumer Staples", "Agriculture"],
                "trend": "Decreasing"
            }
        ]
        
        for factor in factors:
            impact_color = {
                "High": "#F44336",
                "Medium": "#FFA726",
                "Low": "#4CAF50"
            }.get(factor["impact"], "#7B68EE")
            
            trend_icon = {
                "Increasing": "↗",
                "Stable": "→",
                "Decreasing": "↘"
            }.get(factor["trend"], "→")
            
            st.markdown(f"""
            <div style="padding: 15px; margin-bottom: 15px; border-radius: 10px; background-color: rgba(40, 40, 40, 0.3);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div style="font-size: 16px; font-weight: bold;">{factor["title"]}</div>
                    <div style="display: flex; align-items: center;">
                        <span style="color: {impact_color}; margin-right: 5px;">Impact: {factor["impact"]}</span>
                        <span style="color: #AAA; margin-left: 10px;">{trend_icon} {factor["trend"]}</span>
                    </div>
                </div>
                <p style="margin-bottom: 10px; color: #CCC; font-size: 14px;">
                    {factor["description"]}
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px;">
            """, unsafe_allow_html=True)
            
            for sector in factor["affected_sectors"]:
                st.markdown(f"""
                <div style="background-color: rgba(123, 104, 238, 0.2); border-radius: 50px; padding: 5px 12px; font-size: 12px;">
                    {sector}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Investment recommendations based on global trade
        st.subheader("Related Investment Opportunities")
        
        trade_investments = [
            {
                "title": "Domestic Semiconductor Production",
                "description": "Companies benefiting from government incentives for local chip manufacturing",
                "tickers": ["INTC", "TSM", "AMAT"],
                "potential": "High Growth"
            },
            {
                "title": "Supply Chain Analytics",
                "description": "Firms providing technology solutions for supply chain resilience and visibility",
                "tickers": ["CRM", "SAP", "DSGX"],
                "potential": "Steady Growth"
            }
        ]
        
        col1, col2 = st.columns(2)
        
        for i, inv in enumerate(trade_investments):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div style="background-color: rgba(60, 60, 60, 0.3); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                    <h4 style="color: #7B68EE; margin-bottom: 10px;">{inv["title"]}</h4>
                    <p style="margin-bottom: 10px; font-size: 14px; color: #CCC;">
                        {inv["description"]}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                        <div style="display: flex; gap: 5px;">
                """, unsafe_allow_html=True)
                
                for ticker in inv["tickers"]:
                    st.markdown(f"""
                    <div style="background-color: rgba(123, 104, 238, 0.3); border-radius: 5px; padding: 3px 8px; font-size: 12px; font-weight: bold;">
                        {ticker}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                        </div>
                        <div style="color: #4CAF50; font-size: 12px;">
                            {inv["potential"]}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Call-to-action for subscription
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(123, 104, 238, 0.3) 0%, rgba(60, 60, 90, 0.3) 100%); 
         border-radius: 10px; padding: 20px; margin-top: 20px; text-align: center;">
        <h3 style="color: #7B68EE; margin-bottom: 10px;">Ready to unlock the full power of Neufin?</h3>
        <p style="margin-bottom: 20px;">
            Upgrade your account to access all premium features, including AI-powered recommendations,
            detailed analysis, and personalized insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close the demo card

# Main content area with futuristic styling
def load_dashboard():
    if not st.session_state.selected_stocks:
        st.warning("Please select at least one stock or index to analyze.")
        return
    
    # Get current market preferences
    market_prefs = get_market_preferences()
    selected_market = market_prefs['selected_market']
    
    # Market region indicator
    market_indicators = {
        'global': '🌎 Global Markets',
        'us': '🇺🇸 US Market'
    }
    market_indicator = market_indicators.get(selected_market, '🌎 Global Markets')
    
    # Enhanced Market Sentiment Section - Card with multiple widgets
    st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
    
    # Enhanced header with market region and time period
    # Define CSS styles for market pulse separately to avoid f-string issues
    market_pulse_css = """
    <style>
        .header-with-badge {
            display: flex;
            flex-direction: column;
        }
        
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #7B68EE;
            border-radius: 50%;
            margin-left: 8px;
            position: relative;
            display: inline-block;
            animation: pulse-dot 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite;
        }
        
        @keyframes pulse-dot {
            0% { transform: scale(0.8); opacity: 0.7; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.8); opacity: 0.7; }
        }
        
        .market-badge {
            display: flex;
            align-items: center;
            background-color: rgba(123, 104, 238, 0.15);
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid rgba(123, 104, 238, 0.3);
        }
        
        .market-badge-icon {
            font-size: 16px;
            margin-right: 8px;
        }
        
        .market-badge-text {
            color: #E0E0E0;
            font-size: 14px;
            font-weight: 500;
        }
    </style>
    """
    
    # Apply CSS
    st.markdown(market_pulse_css, unsafe_allow_html=True)
    
    # Create header content with market info
    market_icon = market_indicator.split()[0]
    market_text = ' '.join(market_indicator.split()[1:])
    
    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <div class="header-with-badge">
            <h3 style="color: #7B68EE; margin: 0; display: inline-flex; align-items: center;">
                Market Pulse Dashboard
                <span class="pulse-dot"></span>
            </h3>
            <span style="font-size: 12px; color: #AAA; margin-left: 5px;">Sentiment | Trends | Analysis</span>
        </div>
        <div class="market-badge">
            <div class="market-badge-icon">{market_icon}</div>
            <div class="market-badge-text">{market_text}</div>
        </div>
    </div>
    """
    
    # Display header
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Create a tabbed interface for different sentiment views
    sentiment_view_tabs = st.tabs(["Market Overview", "Stock Sentiment", "Sector Analysis"])
    
    # Tab 1: Market Overview
    with sentiment_view_tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Add real-time indicator to headings if auto-refresh is enabled
            real_time_indicator = """<span class="real-time-badge" style="font-size:10px; background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; padding: 3px 6px; border-radius: 4px; margin-left: 8px; border: 1px solid rgba(76, 175, 80, 0.3);">🔄 LIVE</span>""" if st.session_state.auto_refresh else ""
        
        try:
            # Fetch major index data for overall market sentiment
            indices_data = {}
            market_sentiment_scores = []
            
            with st.spinner(f"Analyzing {selected_market} market sentiment..."):
                for index in get_market_indices()[:3]:  # Only use top 3 indices
                    index_data = fetch_stock_data(index, st.session_state.time_period)
                    if index_data is not None:
                        indices_data[index] = index_data
                        sentiment_score = analyze_stock_sentiment(index_data)
                        market_sentiment_scores.append(sentiment_score)
                
                if market_sentiment_scores:
                    overall_sentiment = sum(market_sentiment_scores) / len(market_sentiment_scores)
                    sentiment_text = format_sentiment_score(overall_sentiment)
                    sentiment_color = get_sentiment_color(overall_sentiment)
                    
                    # Display overall sentiment score
                    st.markdown(f"<h1 style='text-align: center; color: {sentiment_color};'>{sentiment_text}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; color: {sentiment_color};'>Score: {overall_sentiment:.2f}</h3>", unsafe_allow_html=True)
                    
                    # Display gauge chart for overall sentiment
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=overall_sentiment,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Market Sentiment Gauge"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': sentiment_color},
                            'steps': [
                                {'range': [-1, -0.5], 'color': "red"},
                                {'range': [-0.5, 0], 'color': "lightcoral"},
                                {'range': [0, 0.5], 'color': "lightgreen"},
                                {'range': [0.5, 1], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': overall_sentiment
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not fetch market index data to analyze overall sentiment.")
        except Exception as e:
            st.error(f"Error analyzing market sentiment: {str(e)}")
    
        with col2:
            # Add AI indicator to news heading for premium users
            ai_indicator = """<span style="font-size: 12px; background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; padding: 3px 6px; border-radius: 4px; margin-left: 8px; border: 1px solid rgba(76, 175, 80, 0.3);">AI POWERED</span>""" if check_feature_access('premium') else ""
        
            # Add real-time indicator to news sentiment heading if auto-refresh is enabled
            st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Personalized Financial News {real_time_indicator} {ai_indicator}</h3>', unsafe_allow_html=True)
            
            try:
                with st.spinner("Loading personalized news..."):
                    # Get user ID if logged in, otherwise use None for general recommendations
                    user_id = st.session_state.get('user_id') if 'user_id' in st.session_state else None
                    
                    # Create news recommender instance
                    news_recommender = NewsRecommender(user_id=user_id)
                    
                    # Get recommended news - use basic recommendations for free tier, 
                    # personalized with sentiment for premium
                    use_advanced_sentiment = check_feature_access('premium') and USE_ADVANCED_AI
                
                if check_feature_access('basic'):
                    # For basic or premium: Show recommended news with relevance for user
                    recommended_news = news_recommender.get_recommendations(
                        limit=5,
                        include_sentiment=True
                    )
                
                    if recommended_news and len(recommended_news) > 0:
                        # Display news with fancy cards
                        for i, news in enumerate(recommended_news):
                            # Show relevance score for premium users
                            show_relevance = check_feature_access('premium')
                            
                            # Render HTML for news card
                            news_html = format_news_card(
                                news,
                                show_relevance=show_relevance,
                                show_sentiment=True
                            )
                            st.markdown(news_html, unsafe_allow_html=True)
                            
                            # Add "Related News" expander for premium users
                            if check_feature_access('premium') and 'id' in news:
                                with st.expander("Related News"):
                                    related_news = news_recommender.get_related_news(news['id'], limit=3)
                                    if related_news and len(related_news) > 0:
                                        for related in related_news:
                                            related_html = format_news_card(
                                                related,
                                                show_relevance=False,
                                                show_sentiment=True
                                            )
                                            st.markdown(related_html, unsafe_allow_html=True)
                                    else:
                                        st.info("No related news found.")
                    else:
                        st.info("No financial news available at the moment.")
                    
                    # For premium users, show information about how news is personalized
                    if check_feature_access('premium'):
                        with st.expander("About Personalized News"):
                            st.markdown("""
                            <div style="color: #CCCCCC; font-size: 14px;">
                                <p><strong>How News Personalization Works:</strong></p>
                                <p>Our AI-driven news engine analyzes your favorites, watched stocks, and reading patterns 
                                to recommend the most relevant financial news. News relevance is calculated using:</p>
                                <ul>
                                    <li>Content matching with your investment interests</li>
                                    <li>Sentiment analysis for emotional tone</li>
                                    <li>Keyword extraction for topic relevance</li>
                                    <li>Reading patterns and similar users' interests</li>
                                </ul>
                                <p>The stars indicate how closely the news aligns with your investment profile.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # For free tier: Show basic market news without fancy styling
                        market_news = fetch_market_news(market=selected_market)
                        
                        if market_news and len(market_news) > 0:
                            # Analyze sentiment for each news headline
                            for i, news in enumerate(market_news[:5]):  # Show top 5 news items
                                title = news.get('title', 'No title')
                                sentiment_score = analyze_text_sentiment(title, use_advanced_nlp=False, use_ai=False)
                                sentiment_text = format_sentiment_score(sentiment_score)
                                sentiment_color = get_sentiment_color(sentiment_score)
                                
                                st.markdown(f"""
                                <div style='background: rgba(30, 30, 46, 0.6); 
                                            border-left: 3px solid {sentiment_color}; 
                                            padding: 10px; 
                                            border-radius: 5px; 
                                            margin-bottom: 10px;'>
                                    <h4 style="margin-top: 0;">{title}</h4>
                                    <div style="display: flex; align-items: center;">
                                        <div style="width: 12px; height: 12px; border-radius: 50%; 
                                                    background-color: {sentiment_color}; margin-right: 8px;"></div>
                                        <p style="margin: 0;">{sentiment_text} ({sentiment_score:.2f})</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No recent market news available.")
                        
                        # Show upgrade prompt for personalized news
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, rgba(123, 104, 238, 0.1) 0%, rgba(60, 50, 120, 0.1) 100%); 
                        border-left: 3px solid #7B68EE; padding: 10px; border-radius: 4px; margin-top: 10px;">
                            <p style="color: #E0E0E0; margin: 0;"><strong>✨ Premium Feature:</strong> 
                            Upgrade to Basic for personalized news recommendations and to Premium for AI-powered 
                            content matching and advanced sentiment analysis.</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading news: {str(e)}")
    
    # Close the market sentiment card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacer between cards
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Predictive Analytics Section - based on sentiment trends
    # Only show for premium/authenticated users or in demo mode
    if check_feature_access('premium') or st.session_state.show_demo:
        show_predictive_analytics(real_time_indicator)
    else:
        # Show premium feature teaser
        st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
        st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">🔮 Predictive Analytics {real_time_indicator}</h3>', unsafe_allow_html=True)
        
        # Premium feature message
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div style="padding: 20px; background-color: rgba(123, 104, 238, 0.1); border-radius: 8px; border-left: 4px solid #7B68EE;">
                <h4 style="color: #7B68EE; margin-top: 0;">Unlock Predictive Analytics</h4>
                <p style="margin-bottom: 10px;">
                    Gain access to AI-powered sentiment forecasting and price prediction models that help you anticipate market movements before they happen.
                </p>
                <ul style="list-style-type: none; padding-left: 5px; color: #ADB3C9;">
                    <li>✓ 10-day sentiment forecasting with confidence intervals</li>
                    <li>✓ Price impact predictions based on sentiment analysis</li>
                    <li>✓ AI-generated insights on future market trends</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            show_upgrade_prompt("predictive analytics", location_id="dashboard_predictive")
            
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Stock-specific sentiment analysis - styled with neufin card
    st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
    # Add real-time indicator to stock sentiment heading if auto-refresh is enabled
    st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Stock Sentiment Analysis {real_time_indicator}</h3>', unsafe_allow_html=True)
    
    try:
        stock_sentiments = []
        
        with st.spinner("Analyzing stock sentiment..."):
            for stock in st.session_state.selected_stocks:
                stock_data = fetch_stock_data(stock, st.session_state.time_period)
                if stock_data is not None:
                    sentiment_score = analyze_stock_sentiment(stock_data)
                    stock_sentiments.append({
                        'Symbol': stock,
                        'Sentiment Score': sentiment_score,
                        'Sentiment': format_sentiment_score(sentiment_score)
                    })
            
            if stock_sentiments:
                # Create a DataFrame for display
                sentiment_df = pd.DataFrame(stock_sentiments)
                
                # Create columns for display
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Bar chart for sentiment comparison with dark theme
                    fig = px.bar(
                        sentiment_df,
                        x='Symbol',
                        y='Sentiment Score',
                        color='Sentiment Score',
                        color_continuous_scale=["#FF5252", "#FFC107", "#4CAF50"],
                        range_color=[-1, 1],
                        title="Sentiment Score by Stock"
                    )
                    
                    # Update layout with dark theme
                    fig.update_layout(
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(30,30,46,0.3)",
                        font=dict(color="#E0E0E0"),
                        title_font_color="#7B68EE",
                        xaxis=dict(
                            gridcolor="rgba(123,104,238,0.15)",
                            zerolinecolor="rgba(123,104,238,0.15)"
                        ),
                        yaxis=dict(
                            gridcolor="rgba(123,104,238,0.15)",
                            zerolinecolor="rgba(123,104,238,0.15)"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Enhanced styling for dataframe
                    st.markdown('<div class="data-metric">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #7B68EE; margin-top: 0;">Stock Sentiment Scores</h4>', unsafe_allow_html=True)
                    
                    # Table of sentiment scores
                    st.dataframe(
                        sentiment_df.style.applymap(
                            lambda x: f"background-color: {get_sentiment_color(x)}; color: white; font-weight: bold;" if isinstance(x, float) else "",
                            subset=['Sentiment Score']
                        ),
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error analyzing stock sentiment: {str(e)}")
    
    # Close the stock sentiment card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacer between cards
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Animated Sentiment Trend Chart - styled with neufin card
    if check_feature_access('basic'):  # Make this a basic subscription feature
        st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
        # Add AI indicator to the title if premium
        ai_indicator = """<span style="font-size: 12px; background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; padding: 3px 6px; border-radius: 4px; margin-left: 8px; border: 1px solid rgba(76, 175, 80, 0.3);">AI POWERED</span>""" if check_feature_access('premium') else ""
        st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Market Sentiment Trend {real_time_indicator} {ai_indicator}</h3>', unsafe_allow_html=True)
        
        try:
            # Store current market sentiment in the database for historical tracking
            # This should happen only once per day or when sentiment is recalculated
            if 'sentiment_stored_today' not in st.session_state:
                st.session_state.sentiment_stored_today = False
                
            # Check if we should store today's sentiment
            if st.session_state.refresh_data or not st.session_state.sentiment_stored_today:
                current_date = datetime.now().date()
                
                # Store market sentiment if recalculated (use latest from overall sentiment section)
                # This assumes overall_sentiment is calculated earlier in the page
                if 'overall_sentiment' in locals():
                    try:
                        # Store in database with current date
                        store_market_sentiment(
                            sentiment_score=overall_sentiment,
                            market_indices=','.join(st.session_state.selected_stocks),
                            news_sentiment=None,  # Could store news_sentiment component if available
                            technical_sentiment=None  # Could store technical_sentiment component if available
                        )
                        st.session_state.sentiment_stored_today = True
                    except Exception as e:
                        st.error(f"Error storing market sentiment: {str(e)}")
            
            # Create and display animated sentiment trend visualization with AI insights
            # Only use AI insights if the user has premium access
            use_ai_insights = check_feature_access('premium')
            animated_trend = create_animated_sentiment_trend(days=14, with_ai_insights=use_ai_insights)
            st.plotly_chart(animated_trend, use_container_width=True)
            
            # Add informational notes based on subscription level
            if use_ai_insights:
                st.info("This chart shows the sentiment trend over the past 14 days with AI-powered insights. Hover over data points to see AI analysis, and use the Play button to animate the trend.")
                
                # Explain how the insights work
                with st.expander("About AI-Generated Insights"):
                    st.markdown("""
                    <div style="color: #CCCCCC;">
                        <p><strong>How AI Insights Work:</strong></p>
                        <p>Our advanced AI system analyzes market conditions, news sentiment, and technical factors 
                        to generate insights for each data point. When you hover over any point on the chart, you'll see:</p>
                        <ol>
                            <li>The date and sentiment score</li>
                            <li>An AI-generated analysis of market conditions</li>
                            <li>Potential factors that influenced the market sentiment that day</li>
                        </ol>
                        <p>These insights help you understand <em>why</em> market sentiment changed, not just <em>how</em> it changed.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("This chart shows the sentiment trend over the past 14 days. Use the Play button to animate the trend.")
                
                # Show upgrade prompt for AI insights
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(123, 104, 238, 0.1) 0%, rgba(60, 50, 120, 0.1) 100%); 
                border-left: 3px solid #7B68EE; padding: 10px; border-radius: 4px; margin-top: 10px;">
                    <p style="color: #E0E0E0; margin: 0;"><strong>✨ Premium Feature:</strong> 
                    Upgrade to Premium for AI-powered insights on every data point! Our AI analyzes market conditions 
                    to explain why sentiment changed on each date, helping you make more informed decisions.</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating sentiment trend: {str(e)}")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add spacer between cards
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    else:
        # Show upgrade prompt for non-premium users
        st.markdown('<div class="neufin-card premium-features">', unsafe_allow_html=True)
        show_upgrade_prompt(
            feature_name="Historical Sentiment Trend Analysis", 
            subscription_level="basic",
            location_id="sentiment_dashboard"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add spacer between cards
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Historical Price and Volume Charts - styled with neufin card
    st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
    # Add real-time indicator to historical price charts heading if auto-refresh is enabled
    st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Historical Price Charts {real_time_indicator}</h3>', unsafe_allow_html=True)
    
    try:
        # Create tabs for each stock
        if len(st.session_state.selected_stocks) > 0:
            stock_tabs = st.tabs([f"{stock}" for stock in st.session_state.selected_stocks[:4]])
            
            for i, stock in enumerate(st.session_state.selected_stocks[:4]):  # Limit to first 4 stocks to avoid clutter
                with stock_tabs[i]:
                    with st.spinner(f"Loading {stock} data..."):
                        stock_data = fetch_stock_data(stock, st.session_state.time_period)
                        
                        if stock_data is not None and not stock_data.empty:
                            # Create figure with secondary y-axis
                            fig = go.Figure()
                            
                            # Calculate the price change color
                            price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
                            price_color = "#4CAF50" if price_change >= 0 else "#FF5252"
                            
                            # Add price line with dynamic color
                            fig.add_trace(go.Scatter(
                                x=stock_data.index,
                                y=stock_data['Close'],
                                name='Close Price',
                                line=dict(color=price_color, width=2)
                            ))
                            
                            # Add volume as bar chart on secondary y-axis
                            fig.add_trace(go.Bar(
                                x=stock_data.index,
                                y=stock_data['Volume'],
                                name='Volume',
                                yaxis='y2',
                                opacity=0.3,
                                marker=dict(color="#7B68EE", opacity=0.3)
                            ))
                            
                            # Set titles and layout with dark theme
                            fig.update_layout(
                                title=f"{stock} - Price and Volume",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(30,30,46,0.3)",
                                font=dict(color="#E0E0E0"),
                                title_font_color="#7B68EE",
                                xaxis=dict(
                                    gridcolor="rgba(123,104,238,0.15)",
                                    zerolinecolor="rgba(123,104,238,0.15)"
                                ),
                                yaxis=dict(
                                    gridcolor="rgba(123,104,238,0.15)",
                                    zerolinecolor="rgba(123,104,238,0.15)"
                                ),
                                yaxis2=dict(
                                    title="Volume",
                                    overlaying="y",
                                    side="right",
                                    showgrid=False,
                                    gridcolor="rgba(123,104,238,0.15)",
                                    zerolinecolor="rgba(123,104,238,0.15)"
                                ),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    bgcolor="rgba(30,30,46,0.5)",
                                    bordercolor="rgba(123,104,238,0.5)"
                                ),
                                height=400
                            )
                            
                            # Add price info in a box
                            start_price = stock_data['Close'].iloc[0]
                            end_price = stock_data['Close'].iloc[-1]
                            pct_change = ((end_price - start_price) / start_price) * 100
                            
                            # Add price metrics in a stylish box at top
                            st.markdown(f"""
                            <div style="display: flex; gap: 20px; margin-bottom: 15px;">
                                <div class="data-metric">
                                    <div class="data-metric-value">${end_price:.2f}</div>
                                    <div class="data-metric-label">Current Price</div>
                                </div>
                                <div class="data-metric">
                                    <div class="data-metric-value" style="color: {('#4CAF50' if pct_change >= 0 else '#FF5252')}">
                                        {'+' if pct_change >= 0 else ''}{pct_change:.2f}%
                                    </div>
                                    <div class="data-metric-label">Period Change</div>
                                </div>
                                <div class="data-metric">
                                    <div class="data-metric-value">${stock_data['Volume'].iloc[-1]//1000000}M</div>
                                    <div class="data-metric-label">Latest Volume</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Could not fetch data for {stock}")
        else:
            st.info("Select stocks from the sidebar to view their historical price charts.")
    except Exception as e:
        st.error(f"Error loading historical charts: {str(e)}")
    
    # Close the historical price card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacer between cards
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Sector Performance Analysis - styled with neufin card
    st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
    # Add real-time indicator to sector performance heading if auto-refresh is enabled
    st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Sector Performance Analysis {real_time_indicator}</h3>', unsafe_allow_html=True)
    
    try:
        with st.spinner("Analyzing sector performance..."):
            sectors_data = fetch_sector_performance(market=selected_market)
            
            if sectors_data is not None and len(sectors_data) > 0:
                # Calculate sentiment based on performance
                for sector in sectors_data:
                    sector['Sentiment Score'] = min(max(sector['Performance'] / 3, -1), 1)  # Normalize to [-1, 1] range
                    sector['Sentiment'] = format_sentiment_score(sector['Sentiment Score'])
                
                sectors_df = pd.DataFrame(sectors_data)
                
                # Sort by performance
                sectors_df = sectors_df.sort_values(by='Performance', ascending=False)
                
                # Show top performing sector in a highlight box
                top_sector = sectors_df.iloc[0]
                worst_sector = sectors_df.iloc[-1]
                
                # Metrics display for top and bottom sectors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="data-metric" style="border-left-color: #4CAF50;">
                        <h4 style="color: #4CAF50; margin-top: 0; margin-bottom: 10px;">Top Performing Sector</h4>
                        <div class="data-metric-value" style="font-size: 28px;">{top_sector['Sector']}</div>
                        <div style="color: #4CAF50; font-size: 20px; font-weight: bold; margin-top: 5px;">+{top_sector['Performance']:.2f}%</div>
                        <div class="data-metric-label">Sentiment: {top_sector['Sentiment']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="data-metric" style="border-left-color: #FF5252;">
                        <h4 style="color: #FF5252; margin-top: 0; margin-bottom: 10px;">Weakest Performing Sector</h4>
                        <div class="data-metric-value" style="font-size: 28px;">{worst_sector['Sector']}</div>
                        <div style="color: #FF5252; font-size: 20px; font-weight: bold; margin-top: 5px;">{worst_sector['Performance']:.2f}%</div>
                        <div class="data-metric-label">Sentiment: {worst_sector['Sentiment']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create sector performance chart with dark theme
                fig = px.bar(
                    sectors_df,
                    x='Sector',
                    y='Performance',
                    color='Performance',
                    color_continuous_scale=["#FF5252", "#FFC107", "#4CAF50"],
                    range_color=[-3, 3],  # Assumes performance is in percent
                    title="Sector Performance (%)"
                )
                
                # Update layout with dark theme
                fig.update_layout(
                    height=500,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30,30,46,0.3)",
                    font=dict(color="#E0E0E0"),
                    title_font_color="#7B68EE",
                    xaxis=dict(
                        gridcolor="rgba(123,104,238,0.15)",
                        zerolinecolor="rgba(123,104,238,0.15)"
                    ),
                    yaxis=dict(
                        gridcolor="rgba(123,104,238,0.15)",
                        zerolinecolor="rgba(123,104,238,0.15)"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sector table with enhanced styling
                st.markdown('<div class="data-metric" style="padding: 15px;">', unsafe_allow_html=True)
                st.markdown('<h4 style="color: #7B68EE; margin-top: 0; margin-bottom: 15px;">Sector Performance Details</h4>', unsafe_allow_html=True)
                st.dataframe(
                    sectors_df[['Sector', 'Performance', 'Sentiment', 'Sentiment Score']].style.format({
                        'Performance': '{:.2f}%',
                        'Sentiment Score': '{:.2f}'
                    }).applymap(
                        lambda x: f"background-color: {get_sentiment_color(x)}; color: white; font-weight: bold;" if isinstance(x, float) else "",
                        subset=['Sentiment Score']
                    ),
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No sector performance data available.")
    except Exception as e:
        st.error(f"Error analyzing sector performance: {str(e)}")
    
    # Close the sector performance card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update last refresh time
    st.session_state.last_update = datetime.now()
    st.session_state.refresh_data = False

# Add premium features - Investment recommendations in a stylish container
st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="neufin-card premium-features">', unsafe_allow_html=True)

# Use the same real-time indicator style for Premium Features
premium_real_time_indicator = """<span class="real-time-badge" style="font-size:10px; background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; padding: 3px 6px; border-radius: 4px; margin-left: 8px; border: 1px solid rgba(76, 175, 80, 0.3);">🔄 LIVE</span>""" if st.session_state.auto_refresh else ""
st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Premium AI-Powered Features {premium_real_time_indicator}</h3>', unsafe_allow_html=True)

# Create a custom styled tabs container
st.markdown("""
<div class="custom-tabs">
    <div class="tabs-header">
        <button class="tab-button active" onclick="activateTab(event, 'tab-recommendations')">Investment Recommendations</button>
        <button class="tab-button" onclick="activateTab(event, 'tab-sectors')">Sector Insights</button>
        <button class="tab-button" onclick="activateTab(event, 'tab-global')">Global Trade Analysis</button>
        <button class="tab-button" onclick="activateTab(event, 'tab-assistant')">AI Assistant</button>
    </div>
</div>

<style>
.custom-tabs {
    margin-bottom: 20px;
}
.tabs-header {
    display: flex;
    border-bottom: 1px solid rgba(123,104,238,0.5);
    margin-bottom: 20px;
}
.tab-button {
    background: none;
    border: none;
    padding: 10px 20px;
    cursor: pointer;
    color: #E0E0E0;
    font-weight: bold;
    position: relative;
}
.tab-button:hover {
    color: #7B68EE;
}
.tab-button.active {
    color: #7B68EE;
}
.tab-button.active::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: #7B68EE;
}
</style>

<script>
function activateTab(evt, tabId) {
    // This is just for show - Streamlit will handle the actual tab switching
    // But this could work with additional JavaScript if needed in the future
    var tabButtons = document.getElementsByClassName("tab-button");
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].className = tabButtons[i].className.replace(" active", "");
    }
    evt.currentTarget.className += " active";
}
</script>
""", unsafe_allow_html=True)

recommendation_tab, sector_tab, global_tab, assistant_tab = st.tabs(["Investment Recommendations", "Sector Insights", "Global Trade Analysis", "AI Assistant"])

with recommendation_tab:
    st.markdown('<h3 style="color: #7B68EE;">AI-Powered Investment Recommendations</h3>', unsafe_allow_html=True)
    
    # Check if user has access to premium features
    need_upgrade = show_upgrade_prompt("AI-Powered Investment Recommendations", "basic", "investment_tab")
    if not need_upgrade:
        try:
            # Only execute this if we have overall sentiment and sector data
            with st.spinner("Generating AI-powered investment recommendations..."):
                # Calculate overall market sentiment
                market_sentiment = 0
                market_sentiment_scores = []
                for index in get_market_indices()[:3]:
                    index_data = fetch_stock_data(index, st.session_state.time_period)
                    if index_data is not None:
                        sentiment_score = analyze_stock_sentiment(index_data)
                        market_sentiment_scores.append(sentiment_score)
                
                if market_sentiment_scores:
                    market_sentiment = sum(market_sentiment_scores) / len(market_sentiment_scores)
                
                # Get sector data for recommendations
                sectors_data = fetch_sector_performance(market=selected_market)
                
                # Generate recommendations
                recommendations_df = get_stock_recommendations(
                    sectors_data, 
                    market_sentiment, 
                    available_stocks, 
                    st.session_state.time_period
                )
                
                if not recommendations_df.empty:
                    # Display top recommendations
                    st.markdown("### Top Investment Opportunities")
                    
                    # Filter to top recommendations
                    top_recommendations = recommendations_df[recommendations_df['Score'] >= 65].head(5)
                    
                    if not top_recommendations.empty:
                        for i, row in top_recommendations.iterrows():
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                score_color = get_sentiment_color(row['Sentiment'])
                                st.markdown(f"""
                                <div style='background-color:{score_color}40; padding:20px; border-radius:10px; text-align:center;'>
                                    <h1>{row['Ticker']}</h1>
                                    <h2>${row['Current Price']:.2f}</h2>
                                    <p>{row['Price Change (%)']:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"#### {row['Recommendation']} ({row['Score']:.1f}/100)")
                                st.markdown(f"**Sector:** {row['Sector']}")
                                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                                
                                # If premium, offer detailed analysis
                                if check_feature_access('premium') and not row['Ticker'].startswith('^'):
                                    if st.button(f"View Detailed Analysis for {row['Ticker']}", key=f"analysis_{row['Ticker']}"):
                                        with st.spinner(f"Generating detailed investment thesis for {row['Ticker']}..."):
                                            stock_data = fetch_stock_data(row['Ticker'], st.session_state.time_period)
                                            market_news = fetch_market_news(market=selected_market)
                                            
                                            # Generate or get cached analysis
                                            if row['Ticker'] not in st.session_state.ai_analyses:
                                                thesis = generate_investment_thesis(
                                                    row['Ticker'], stock_data, market_news, market_sentiment
                                                )
                                                st.session_state.ai_analyses[row['Ticker']] = thesis
                                            else:
                                                thesis = st.session_state.ai_analyses[row['Ticker']]
                                            
                                            if 'error' not in thesis:
                                                st.markdown(f"### Investment Thesis: {row['Ticker']}")
                                                st.markdown(f"**{thesis.get('company_overview', '')}**")
                                                
                                                # Display strengths and risks in columns
                                                strength_col, risk_col = st.columns(2)
                                                
                                                with strength_col:
                                                    st.markdown("#### Key Strengths")
                                                    strengths = thesis.get('key_strengths', [])
                                                    for strength in strengths:
                                                        st.markdown(f"- {strength}")
                                                
                                                with risk_col:
                                                    st.markdown("#### Risk Factors")
                                                    risks = thesis.get('risk_factors', [])
                                                    for risk in risks:
                                                        st.markdown(f"- {risk}")
                                                
                                                # Valuation and recommendation
                                                st.markdown(f"#### Valuation: {thesis.get('valuation_assessment', '')}")
                                                
                                                # Show recommendation with color
                                                rec = thesis.get('investment_recommendation', 'Hold')
                                                rec_color = {
                                                    'Strong Buy': '#1B5E20', 
                                                    'Buy': '#4CAF50',
                                                    'Hold': '#FFC107',
                                                    'Sell': '#FF9800',
                                                    'Strong Sell': '#D32F2F'
                                                }.get(rec, '#FFC107')
                                                
                                                st.markdown(f"""
                                                <div style='background-color:{rec_color}40; padding:10px; border-radius:5px;'>
                                                <h4>Recommendation: {rec}</h4>
                                                <p>Target Price Range: {thesis.get('target_price_range', 'N/A')}</p>
                                                <p>Confidence Level: {thesis.get('confidence_level', 'Moderate')}</p>
                                                <p>Investment Timeframe: {thesis.get('investment_timeframe', 'Medium-term')}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            else:
                                                st.error(f"Could not generate detailed analysis: {thesis.get('error', 'Unknown error')}")
                                
                            st.markdown("---")
                    else:
                        st.info("No strong recommendations available based on current market conditions.")
                    
                    # Show all recommendations in a table
                    with st.expander("View All Stock Ratings"):
                        # Format the dataframe for display
                        display_df = recommendations_df[['Ticker', 'Sector', 'Current Price', 'Price Change (%)', 'Sentiment', 'Score', 'Recommendation']]
                        
                        st.dataframe(
                            display_df.style.format({
                                'Current Price': '${:.2f}',
                                'Price Change (%)': '{:.2f}%',
                                'Score': '{:.1f}'
                            }),
                            use_container_width=True
                        )
                else:
                    st.error("Could not generate investment recommendations. Please try again later.")
        except Exception as e:
            st.error(f"Error generating investment recommendations: {str(e)}")
    # The else block is removed as we already show the upgrade prompt at the beginning

with sector_tab:
    st.markdown('<h3 style="color: #7B68EE;">Detailed Sector Insights</h3>', unsafe_allow_html=True)
    
    # Check if user has access to premium features
    need_upgrade = show_upgrade_prompt("Detailed Sector Insights", "basic", "sector_tab")
    if not need_upgrade:
        try:
            with st.spinner("Analyzing sector performance..."):
                # Get sector data
                sectors_data = fetch_sector_performance(market=selected_market)
                
                if sectors_data and len(sectors_data) > 0:
                    # Get insights for the best performing sectors
                    sector_insights = get_sector_insights(sectors_data)
                    
                    # Display top sectors
                    if 'top_sectors' in sector_insights and sector_insights['top_sectors']:
                        st.markdown("### Top Performing Sectors")
                        
                        # Create metrics row for top sectors
                        cols = st.columns(len(sector_insights['top_sectors'][:3]))
                        
                        for i, sector in enumerate(sector_insights['top_sectors'][:3]):
                            with cols[i]:
                                performance = sector['Performance']
                                sentiment_color = get_sentiment_color(performance / 3)  # Normalize to -1 to 1 range
                                
                                st.markdown(f"""
                                <div style='background-color:{sentiment_color}20; padding:10px; border-radius:5px; text-align:center;'>
                                <h3>{sector['Sector']}</h3>
                                <h2 style='color:{sentiment_color};'>{performance:.2f}%</h2>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Display detailed sector insights
                    st.markdown("### Sector Analysis")
                    
                    if 'insights' in sector_insights:
                        # Allow user to select a sector for detailed analysis
                        available_sectors = list(sector_insights['insights'].keys())
                        
                        if available_sectors:
                            selected_sector = st.selectbox(
                                "Select a sector for detailed analysis",
                                options=available_sectors
                            )
                            
                            # Get the insights for the selected sector
                            sector_detail = sector_insights['insights'].get(selected_sector, {})
                            
                            if sector_detail:
                                # Find the sector performance data
                                sector_perf_data = next((s for s in sectors_data if s['Sector'] == selected_sector), None)
                                
                                # Create columns for sector details
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.markdown(f"#### Key Drivers")
                                    st.markdown(f"{sector_detail.get('drivers', 'No data available')}")
                                    
                                    st.markdown(f"#### Stocks to Watch")
                                    st.markdown(f"{sector_detail.get('stocks_to_watch', 'No data available')}")
                                
                                with col2:
                                    st.markdown(f"#### Outlook")
                                    st.markdown(f"{sector_detail.get('outlook', 'No data available')}")
                                    
                                    st.markdown(f"#### Risks")
                                    st.markdown(f"{sector_detail.get('risks', 'No data available')}")
                                
                                # If premium, offer detailed AI analysis
                                if check_feature_access('premium') and sector_perf_data:
                                    if 'ai_sector_analyses' not in st.session_state:
                                        st.session_state.ai_sector_analyses = {}
                                        
                                    if st.button(f"Generate AI-Powered Sector Report for {selected_sector}", key=f"sector_ai_{selected_sector}"):
                                        with st.spinner(f"Generating comprehensive sector analysis for {selected_sector}..."):
                                            # Calculate market sentiment
                                            market_sentiment = 0
                                            market_sentiment_scores = []
                                            for index in get_market_indices()[:3]:
                                                index_data = fetch_stock_data(index, st.session_state.time_period)
                                                if index_data is not None:
                                                    sentiment_score = analyze_stock_sentiment(index_data)
                                                    market_sentiment_scores.append(sentiment_score)
                                            
                                            if market_sentiment_scores:
                                                market_sentiment = sum(market_sentiment_scores) / len(market_sentiment_scores)
                                            
                                            # Generate or get cached analysis
                                            if selected_sector not in st.session_state.ai_sector_analyses:
                                                outlook = generate_sector_outlook(
                                                    selected_sector, sector_perf_data, market_sentiment
                                                )
                                                st.session_state.ai_sector_analyses[selected_sector] = outlook
                                            else:
                                                outlook = st.session_state.ai_sector_analyses[selected_sector]
                                            
                                            if 'error' not in outlook:
                                                st.markdown(f"### AI Sector Report: {selected_sector}")
                                                st.markdown(f"**{outlook.get('sector_overview', '')}**")
                                                
                                                # Create sections for the report
                                                tabs = st.tabs(["Key Factors", "Competitive Landscape", "Outlook & Strategy"])
                                                
                                                with tabs[0]:
                                                    col1, col2 = st.columns(2)
                                                    
                                                    with col1:
                                                        st.markdown("#### Key Drivers")
                                                        drivers = outlook.get('key_drivers', [])
                                                        for driver in drivers:
                                                            st.markdown(f"- {driver}")
                                                            
                                                        st.markdown("#### Opportunities")
                                                        opportunities = outlook.get('opportunities', [])
                                                        for opportunity in opportunities:
                                                            st.markdown(f"- {opportunity}")
                                                    
                                                    with col2:
                                                        st.markdown("#### Challenges")
                                                        challenges = outlook.get('challenges', [])
                                                        for challenge in challenges:
                                                            st.markdown(f"- {challenge}")
                                                            
                                                        st.markdown("#### Innovation Trends")
                                                        trends = outlook.get('innovation_trends', [])
                                                        for trend in trends:
                                                            st.markdown(f"- {trend}")
                                                
                                                with tabs[1]:
                                                    st.markdown("#### Competitive Landscape")
                                                    st.markdown(outlook.get('competitive_landscape', 'No data available'))
                                                    
                                                    st.markdown("#### Regulatory Environment")
                                                    st.markdown(outlook.get('regulatory_environment', 'No data available'))
                                                    
                                                    st.markdown("#### Top Companies to Watch")
                                                    companies = outlook.get('top_companies_to_watch', [])
                                                    for company in companies:
                                                        st.markdown(f"- {company}")
                                                
                                                with tabs[2]:
                                                    st.markdown("#### Short-term Outlook")
                                                    if isinstance(outlook.get('outlook', {}), dict):
                                                        st.markdown(outlook.get('outlook', {}).get('short_term', 'No data available'))
                                                    
                                                    st.markdown("#### Long-term Outlook")
                                                    if isinstance(outlook.get('outlook', {}), dict):
                                                        st.markdown(outlook.get('outlook', {}).get('long_term', 'No data available'))
                                                    
                                                    st.markdown("#### Investment Strategy")
                                                    st.markdown(outlook.get('investment_strategy', 'No data available'))
                                            else:
                                                st.error(f"Could not generate detailed sector analysis: {outlook.get('error', 'Unknown error')}")
                    else:
                        st.error("No sector insights available.")
                else:
                    st.info("No sector performance data available.")
        except Exception as e:
            st.error(f"Error generating sector insights: {str(e)}")
    # The else block is removed as we already show the upgrade prompt at the beginning

with global_tab:
    st.markdown('<h3 style="color: #7B68EE;">Global Trade Impact Analysis</h3>', unsafe_allow_html=True)
    
    # Check if user has access to premium features
    need_upgrade = show_upgrade_prompt("Global Trade Impact Analysis", "premium", "global_tab")
    if not need_upgrade:
        try:
            # Get global trade analysis
            with st.spinner("Analyzing global trade conditions..."):
                global_analysis = analyze_global_trade_conditions()
                
                if 'error' not in global_analysis:
                    st.markdown(f"### Global Trade Landscape")
                    st.markdown(f"**{global_analysis.get('summary', '')}**")
                    
                    # Display trade policies
                    st.markdown("### Major Trade Policies & Tariffs")
                    
                    policies = global_analysis.get('trade_policies', [])
                    for i, policy in enumerate(policies):
                        with st.expander(f"{policy.get('name', f'Policy {i+1}')}"):
                            st.markdown(f"**Countries Involved:** {', '.join(policy.get('countries_involved', ['Unknown']))}")
                            st.markdown(f"**Summary:** {policy.get('summary', 'No details available')}")
                            st.markdown(f"**Economic Impact:** {policy.get('impact', 'No impact data available')}")
                            st.markdown(f"**Affected Sectors:** {', '.join(policy.get('affected_sectors', ['Unknown']))}")
                    
                    # Display sanctions
                    st.markdown("### Major Sanctions Programs")
                    
                    sanctions = global_analysis.get('sanctions_programs', [])
                    cols = st.columns(min(len(sanctions), 3))
                    
                    for i, sanction in enumerate(sanctions):
                        with cols[i % len(cols)]:
                            st.markdown(f"""
                            <div style='background-color:#f8f8f8; padding:10px; border-radius:5px; height:200px; overflow-y:auto;'>
                            <h4>🚫 {sanction.get('target', 'Unknown')}</h4>
                            <p><b>Imposed by:</b> {', '.join(sanction.get('imposed_by', ['Unknown']))}</p>
                            <p>{sanction.get('summary', 'No details available')}</p>
                            <p><b>Market Impact:</b> {sanction.get('market_impact', 'No impact data available')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display tension regions with a map
                    st.markdown("### Trade Tension Hotspots")
                    
                    tensions = global_analysis.get('trade_tensions', [])
                    for tension in tensions:
                        risk_color = {
                            'High': '#D32F2F',
                            'Medium': '#FF9800',
                            'Low': '#4CAF50'
                        }.get(tension.get('risk_level', 'Medium'), '#FFC107')
                        
                        st.markdown(f"""
                        <div style='border-left: 5px solid {risk_color}; padding-left: 10px; margin-bottom: 10px;'>
                        <h4>{tension.get('region', 'Unknown Region')}</h4>
                        <p><b>Risk Level:</b> <span style='color:{risk_color};'>{tension.get('risk_level', 'Medium')}</span></p>
                        <p>{tension.get('description', 'No details available')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display sector impacts
                    st.markdown("### Impact on Investment Sectors")
                    
                    impacts = global_analysis.get('sector_impacts', [])
                    impact_df = pd.DataFrame(impacts)
                    
                    if not impact_df.empty and 'sector' in impact_df.columns and 'impact' in impact_df.columns:
                        # Add color column
                        impact_df['color'] = impact_df['impact'].apply(lambda x: {
                            'Positive': '#4CAF50',
                            'Negative': '#D32F2F',
                            'Mixed': '#FFC107'
                        }.get(x, '#9E9E9E'))
                        
                        # Create bar chart
                        fig = px.bar(
                            impact_df,
                            x='sector',
                            y=[1] * len(impact_df),  # Equal height bars
                            color='impact',
                            color_discrete_map={
                                'Positive': '#4CAF50',
                                'Negative': '#D32F2F',
                                'Mixed': '#FFC107',
                                'Neutral': '#9E9E9E'
                            },
                            title="Trade Impact by Sector"
                        )
                        fig.update_layout(height=400, xaxis_title="", yaxis_title="")
                        fig.update_yaxes(showticklabels=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display investor recommendations
                    st.markdown("### Strategic Recommendations for Investors")
                    
                    recommendations = global_analysis.get('investor_recommendations', [])
                    for rec in recommendations:
                        suitable = rec.get('suitable_for', 'All investors')
                        strategy = rec.get('strategy', 'Strategy')
                        description = rec.get('description', 'No details available')
                        
                        suitable_color = {
                            'Conservative': '#4CAF50',
                            'Balanced': '#FFC107',
                            'Aggressive': '#FF5722'
                        }.get(suitable.split('/')[0] if '/' in suitable else suitable, '#9E9E9E')
                        
                        st.markdown(f"""
                        <div style='background-color:{suitable_color}15; border:1px solid {suitable_color}50; padding:15px; border-radius:5px; margin-bottom:10px;'>
                        <h4>{strategy}</h4>
                        <p><b>Suitable for:</b> {suitable}</p>
                        <p>{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Could not retrieve global trade analysis: {global_analysis.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error analyzing global trade conditions: {str(e)}")
    # The else block is removed as we already show the upgrade prompt at the beginning

# AI Assistant tab implementation
with assistant_tab:
    st.markdown('<h3 style="color: #7B68EE;">Neufin AI Assistant</h3>', unsafe_allow_html=True)
    
    # Check if user has access to premium features
    need_upgrade = show_upgrade_prompt("AI Assistant", "premium", "assistant_tab")
    if not need_upgrade:
        try:
            # Import agent functionality
            from agent import run_neufin_agent, reset_agent_memory
            
            # Create a container for the chat interface
            chat_container = st.container()
            
            with chat_container:
                # Initialize chat history in session state if not present
                if "ai_assistant_messages" not in st.session_state:
                    st.session_state.ai_assistant_messages = []
                
                # Display intro message
                if not st.session_state.ai_assistant_messages:
                    st.markdown("""
                    <div style="padding: 15px; border-radius: 10px; background-color: rgba(123, 104, 238, 0.1); margin-bottom: 20px;">
                        <h4 style="color: #7B68EE; margin-top: 0;">Welcome to Neufin AI Assistant</h4>
                        <p>I'm your personal financial intelligence assistant powered by advanced AI models. Ask me about:</p>
                        <ul>
                            <li>Current market sentiment for specific stocks</li>
                            <li>Investment recommendations based on real-time data</li>
                            <li>Analysis of sector performance</li>
                            <li>Insights on global trade factors affecting markets</li>
                            <li>Explanations of financial concepts and terms</li>
                        </ul>
                        <p>I can access real-time market data and use advanced AI to provide personalized financial insights.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display chat messages
                for message in st.session_state.ai_assistant_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Add a reset button for the conversation
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Reset Conversation", key="reset_conversation"):
                        reset_agent_memory()
                        st.session_state.ai_assistant_messages = []
                        st.rerun()
                
                # User input
                prompt = st.chat_input("Ask anything about markets, investments, or financial data...")
                
                if prompt:
                    # Add user message to chat history
                    st.session_state.ai_assistant_messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        response_container = st.empty()
                        ai_response = run_neufin_agent(prompt, response_container)
                        
                        if ai_response:
                            # Add assistant response to chat history
                            if "output" in ai_response:
                                st.session_state.ai_assistant_messages.append({"role": "assistant", "content": ai_response["output"]})
            
            # Add example queries
            with st.expander("Example Questions"):
                st.markdown("""
                - What's the current market sentiment for Apple (AAPL)?
                - Can you analyze the technology sector performance?
                - What global trade factors are impacting markets today?
                - Give me investment recommendations for growth stocks.
                - How does inflation affect stock valuations?
                - What are the best performing sectors this month?
                - What's your analysis of Microsoft (MSFT) stock?
                """)
                
        except Exception as e:
            st.error(f"Error with AI Assistant: {str(e)}")
    # The else block is removed as we already show the upgrade prompt at the beginning

# Close the premium features container
st.markdown('</div>', unsafe_allow_html=True)

# Handle data loading with real-time updates
if (st.session_state.auto_refresh and 
    (datetime.now() - st.session_state.last_auto_refresh).total_seconds() >= st.session_state.refresh_interval):
    # Automatically refresh data based on interval
    st.session_state.refresh_data = True
    st.session_state.last_auto_refresh = datetime.now()
    
    # Visual indicator for real-time refresh
    st.toast(f"🔄 Data refreshed automatically at {datetime.now().strftime('%H:%M:%S')}", icon="🔄")

# Function to run the dashboard from main.py
def run_dashboard():
    """Main dashboard function that can be called from main.py"""
    # Apply dashboard styling first
    apply_dashboard_styling()
    
    # Check if demo showcase mode is enabled
    if st.session_state.get("show_demo", False):
        # Show the demo showcase instead of the regular dashboard
        load_demo_showcase()
    else:
        # Regular dashboard flow
        if st.session_state.get("refresh_data", False):
            with st.spinner("Refreshing data in real-time..."):
                load_dashboard()
                # Reset the refresh flag after loading
                st.session_state.refresh_data = False
                # Update the last update timestamp
                st.session_state.last_update = datetime.now()
        else:
            load_dashboard()
            
    # Add automatic refresh behavior
    if st.session_state.get("auto_refresh", False):
        # Calculate time to wait before next refresh
        seconds_since_refresh = (datetime.now() - st.session_state.last_auto_refresh).total_seconds()
        seconds_until_refresh = max(1, min(10, st.session_state.refresh_interval - seconds_since_refresh))
        
        # Only rerun if we're more than 80% of the way to the next refresh
        # This prevents too frequent reruns while still maintaining real-time updates
        if seconds_since_refresh > (st.session_state.refresh_interval * 0.8):
            time.sleep(1)  # Small delay to prevent excessive reruns
            st.rerun()
    
    # Add footer content
    add_footer()
    
# Set up automatic rerun for real-time updates if enabled
if st.session_state.auto_refresh:
    # Calculate time to wait before next refresh
    seconds_since_refresh = (datetime.now() - st.session_state.last_auto_refresh).total_seconds()
    seconds_until_refresh = max(1, min(10, st.session_state.refresh_interval - seconds_since_refresh))
    
    # Only rerun if we're more than 80% of the way to the next refresh
    # This prevents too frequent reruns while still maintaining real-time updates
    if seconds_since_refresh > (st.session_state.refresh_interval * 0.8):
        time.sleep(1)  # Small delay to prevent excessive reruns
        st.rerun()

# Function to add the footer to the app
def add_footer():
    """Add the branding footer to the application"""
    # Add spacer and commercial footer with Neufin branding
    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    
    # Commercial footer with futuristic dark theme
    st.markdown("""
<div class="neufin-card premium-features">
    <div style="text-align:center; margin-bottom: 20px;">
        <img src="data:image/png;base64,""" + open('neufin_new_logo_base64.txt', 'r').read() + """" class="neufin-footer-logo" alt="Neufin AI Logo">
        <h3 style="color: #7B68EE; margin-top: 15px;">Neural Powered Finance Unlocked</h3>
    </div>
    <p style="text-align:center; color: #e0e0e0;">
        Unlock the power of AI-driven market analysis. Subscribe to our Premium Plan for advanced insights, personalized recommendations, and global trade impact analysis.
    </p>
    <div style="display:flex; justify-content:center; gap:15px; flex-wrap: wrap; margin-top: 25px;">
        <div class="feature-pill">
            <div class="feature-icon">🚀</div>
            <div class="feature-text">AI Analysis</div>
        </div>
        <div class="feature-pill">
            <div class="feature-icon">💰</div>
            <div class="feature-text">Investment Recommendations</div>
        </div>
        <div class="feature-pill">
            <div class="feature-icon">📊</div>
            <div class="feature-text">Sector Insights</div>
        </div>
        <div class="feature-pill">
            <div class="feature-icon">🌎</div>
            <div class="feature-text">Global Trade Impact</div>
        </div>
    </div>
</div>

<style>
.feature-pill {
    text-align: center;
    padding: 10px 15px;
    background-color: rgba(123, 104, 238, 0.2);
    border-radius: 50px;
    display: flex;
    align-items: center;
    transition: all 0.3s ease;
}

.feature-pill:hover {
    background-color: rgba(123, 104, 238, 0.4);
    transform: translateY(-2px);
}

.feature-icon {
    margin-right: 8px;
    font-size: 18px;
}

.feature-text {
    color: #e0e0e0;
    font-weight: 500;
}

.neufin-footer-logo {
    height: 70px;
    margin: 0 auto;
    display: block;
}

.neufin-small-logo {
    height: 40px;
    vertical-align: middle;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# Disclaimer footer with Neufin branding
st.markdown("""
<div style="font-size:0.8em; color:#999; text-align:center; margin-top:30px; padding: 0 20px;">
    <p><strong>Disclaimer:</strong> All sentiment analysis and recommendations are powered by AI and should be used for informational purposes only.
    Neufin does not provide financial advice. Investment decisions should be made in consultation with financial professionals.</p>
    <p>Data source: Alpha Vantage | Last updated: {}</p>
    <div style="margin-top: 15px; opacity: 0.7;">
        <img src="data:image/png;base64,""" + open('neufin_new_logo_base64.txt', 'r').read() + """" class="neufin-small-logo" alt="Neufin AI Logo" style="height: 40px; vertical-align: middle; margin-right: 8px;">
        © 2025 Neufin OÜ | A Unit of Ctech Ventures | Järvevana tee 9, 11314, Tallinn, Estonia
    </div>
</div>
""".format(st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
