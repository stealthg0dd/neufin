import streamlit as st

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Neufin",
    page_icon="🔮",
    layout="wide",
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json

# Import custom modules
from data_fetcher import fetch_stock_data, fetch_market_news, fetch_sector_performance
from sentiment_analyzer import analyze_text_sentiment, analyze_stock_sentiment, analyze_news_sentiment_batch
from utils import format_sentiment_score, get_sentiment_color, get_market_indices, format_large_number
from database import store_market_sentiment, get_historical_sentiment
from investment_advisor import get_stock_recommendations, analyze_global_trade_impact, get_sector_insights
from subscription_manager import initialize_subscription_state, show_login_form, show_subscription_options, process_payment, check_feature_access, show_upgrade_prompt, check_trial_status
from ai_analyst import analyze_global_trade_conditions, generate_investment_thesis, generate_sector_outlook

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

# Custom CSS for enhanced futuristic Neufin design
st.markdown("""
<style>
    /* Base Styling */
    .stApp {
        background-color: #121212;
        background-image: radial-gradient(circle at top right, rgba(123, 104, 238, 0.1), transparent 400px);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    p {
        color: #CCCCCC;
    }
    
    /* Custom Card Styling */
    .neufin-card {
        background: linear-gradient(to bottom right, rgba(30, 30, 46, 0.8), rgba(17, 17, 17, 0.9));
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(123, 104, 238, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .neufin-card:hover {
        box-shadow: 0 6px 24px rgba(123, 104, 238, 0.2);
    }
    
    .premium-features {
        background: linear-gradient(135deg, rgba(30, 30, 46, 0.9) 0%, rgba(20, 20, 35, 0.9) 100%);
        border-top: 1px solid rgba(123, 104, 238, 0.3);
        border-left: 1px solid rgba(123, 104, 238, 0.3);
    }
    
    .neufin-headline {
        background: linear-gradient(90deg, #7B68EE, #3A3A80);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    
    .neufin-headline:before {
        content: "";
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        background: linear-gradient(45deg, rgba(123, 104, 238, 0.2), transparent, rgba(123, 104, 238, 0.2));
        z-index: -1;
        filter: blur(20px);
    }
    
    .glow-text {
        text-shadow: 0 0 10px rgba(123, 104, 238, 0.7);
    }
    
    /* Data metrics styling */
    .data-metric {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 8px;
        padding: 15px;
        border-left: 3px solid #7B68EE;
        transition: transform 0.2s;
    }
    
    .data-metric:hover {
        transform: translateY(-2px);
    }
    
    .data-metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #E0E0E0;
    }
    
    .data-metric-label {
        font-size: 14px;
        color: #AAAAAA;
        margin-top: 3px;
    }
    
    /* Sidebar styling */
    .css-1cypcdb, .css-d1kyf5, .css-z5fcl4 {
        background-color: #1A1A2E !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #7B68EE, #5D4DC4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 15px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #9281F1, #7B68EE);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(123, 104, 238, 0.4);
    }
    
    /* Feature pills in the footer */
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
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: rgba(30, 30, 46, 0.8) !important;
        color: #7B68EE !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(123, 104, 238, 0.2) !important;
        text-align: left !important;
        padding: 8px 12px !important;
    }
    
    .dataframe td {
        background-color: rgba(20, 20, 30, 0.6) !important;
        color: #E0E0E0 !important;
        border: none !important;
        padding: 8px 12px !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 46, 0.4);
        border-radius: 4px 4px 0 0;
        color: #AAAAAA;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(123, 104, 238, 0.2);
        color: #7B68EE;
        border-bottom: 2px solid #7B68EE;
    }
    
    /* Chart background */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
    
    /* Real-time update styling */
    .real-time-badge {
        background-color: rgba(123, 104, 238, 0.2);
        color: #7B68EE;
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 12px;
        margin-left: 8px;
        display: inline-flex;
        align-items: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(123, 104, 238, 0.4);
        }
        70% {
            box-shadow: 0 0 0 6px rgba(123, 104, 238, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(123, 104, 238, 0);
        }
    }
    
    /* Toast styling */
    .stToast {
        background-color: rgba(30, 30, 46, 0.9) !important;
        color: #E0E0E0 !important;
        border: 1px solid rgba(123, 104, 238, 0.3) !important;
        border-radius: 8px !important;
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
    
# Initialize subscription state
initialize_subscription_state()
check_trial_status()

# Main title with futuristic styling
# Add real-time badge if auto-refresh is enabled
real_time_badge = """<span class="real-time-badge">🔄 REAL-TIME</span>""" if st.session_state.auto_refresh else ""
st.markdown(f"""
<div class="neufin-headline">
    <h1 class="glow-text">🔮 NEUFIN {real_time_badge}</h1>
    <h3>The Future of Financial Intelligence</h3>
    <p style="color:#CCC">Advanced AI-Powered Market Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# Introduction with futuristic design
st.markdown("""
<div class="neufin-card">
    <p style="font-size:18px; margin-bottom:20px;">
        <span style="color:#7B68EE; font-weight:bold;">Welcome to Neufin</span> - Where AI meets Finance.
    </p>
    <p>
        Our cutting-edge neural networks continuously analyze market data, global news, and trading patterns to deliver 
        predictive insights with unprecedented accuracy. Navigate the complexities of today's markets with 
        our advanced toolset designed for the modern investor.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for filters, inputs, and account features
with st.sidebar:
    # Custom Neufin logo and brand
    from pathlib import Path
    logo_path = Path("neufin-icon.svg")
    if logo_path.exists():
        st.image("neufin-icon.svg", width=100)
    else:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 5px;">
            <h1 style="color: #7B68EE; font-weight: 700;">🔮</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #7B68EE; font-weight: 700;">NEUFIN</h2>
        <p style="color: #AAA; font-size: 12px; margin-top: -10px;">FINANCIAL INTELLIGENCE</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom styled login container
    st.markdown('<div class="neufin-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
    # Add login/subscription management
    show_login_form()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show subscription options if logged in with custom styling
    if st.session_state.user_logged_in:
        st.markdown('<div class="neufin-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
        show_subscription_options()
        process_payment()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="neufin-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #7B68EE; margin-bottom: 15px;">Dashboard Settings</h3>', unsafe_allow_html=True)
    
    # Stock selection
    available_stocks = get_market_indices() + ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'DIS', 
                                             'JPM', 'BAC', 'XOM', 'CVX', 'JNJ', 'PFE', 'WMT', 'PG', 'BA', 'CAT', 'NEE']
    selected_stocks = st.multiselect(
        "Select stocks/indices to analyze",
        options=available_stocks,
        default=st.session_state.selected_stocks
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
    Neufin leverages quantum-inspired algorithms and neural networks to analyze financial markets 
    in real-time. Our sentiment analysis ranges from -1 (bearish) to +1 (bullish).
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
    
    # Overall Market Sentiment Card
    st.markdown('<div class="neufin-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add real-time indicator to headings if auto-refresh is enabled
        real_time_indicator = """<span class="real-time-badge" style="font-size:10px; background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; padding: 3px 6px; border-radius: 4px; margin-left: 8px; border: 1px solid rgba(76, 175, 80, 0.3);">🔄 LIVE</span>""" if st.session_state.auto_refresh else ""
        st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Overall Market Sentiment {real_time_indicator}</h3>', unsafe_allow_html=True)
        
        try:
            # Fetch major index data for overall market sentiment
            indices_data = {}
            market_sentiment_scores = []
            
            with st.spinner("Analyzing overall market sentiment..."):
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
        # Add real-time indicator to news sentiment heading if auto-refresh is enabled
        st.markdown(f'<h3 style="color: #7B68EE; margin-bottom: 15px;">Latest News Sentiment {real_time_indicator}</h3>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Analyzing news sentiment..."):
                # Fetch and analyze recent market news
                market_news = fetch_market_news()
                
                if market_news and len(market_news) > 0:
                    # Analyze sentiment for each news headline
                    for i, news in enumerate(market_news[:5]):  # Show top 5 news items
                        title = news.get('title', 'No title')
                        sentiment_score = analyze_text_sentiment(title)
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
        except Exception as e:
            st.error(f"Error analyzing news sentiment: {str(e)}")
    
    # Close the market sentiment card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacer between cards
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
            required_plan="basic"
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
                                    <div class="data-metric-value" style="color: {'#4CAF50' if pct_change >= 0 else '#FF5252'}">
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
            sectors_data = fetch_sector_performance()
            
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

recommendation_tab, sector_tab, global_tab = st.tabs(["Investment Recommendations", "Sector Insights", "Global Trade Analysis"])

with recommendation_tab:
    st.markdown('<h3 style="color: #7B68EE;">AI-Powered Investment Recommendations</h3>', unsafe_allow_html=True)
    
    # Check if user has access to premium features
    if check_feature_access('basic'):
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
                sectors_data = fetch_sector_performance()
                
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
                                            market_news = fetch_market_news()
                                            
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
    else:
        # Show upgrade message for premium feature
        show_upgrade_prompt("AI-Powered Investment Recommendations", "basic")

with sector_tab:
    st.markdown('<h3 style="color: #7B68EE;">Detailed Sector Insights</h3>', unsafe_allow_html=True)
    
    # Check if user has access to premium features
    if check_feature_access('basic'):
        try:
            with st.spinner("Analyzing sector performance..."):
                # Get sector data
                sectors_data = fetch_sector_performance()
                
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
    else:
        # Show upgrade message for premium feature
        show_upgrade_prompt("Detailed Sector Insights", "basic")

with global_tab:
    st.markdown('<h3 style="color: #7B68EE;">Global Trade Impact Analysis</h3>', unsafe_allow_html=True)
    
    # Check if user has access to premium features
    if check_feature_access('premium'):
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
    else:
        # Show upgrade message for premium feature
        show_upgrade_prompt("Global Trade Impact Analysis", "premium")

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

# Check if demo showcase mode is enabled
if st.session_state.show_demo:
    # Show the demo showcase instead of the regular dashboard
    load_demo_showcase()
else:
    # Regular dashboard flow
    if st.session_state.refresh_data:
        with st.spinner("Refreshing data in real-time..."):
            load_dashboard()
            # Reset the refresh flag after loading
            st.session_state.refresh_data = False
            # Update the last update timestamp
            st.session_state.last_update = datetime.now()
    else:
        load_dashboard()
    
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

# Add spacer and commercial footer with Neufin branding
st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)

# Commercial footer with futuristic dark theme
st.markdown("""
<div class="neufin-card premium-features">
    <h3 style="text-align:center; color: #7B68EE; margin-bottom: 20px;">Neufin - AI-Powered Financial Intelligence</h3>
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
</style>
""", unsafe_allow_html=True)

# Disclaimer footer with Neufin branding
st.markdown("""
<div style="font-size:0.8em; color:#999; text-align:center; margin-top:30px; padding: 0 20px;">
    <p><strong>Disclaimer:</strong> All sentiment analysis and recommendations are powered by AI and should be used for informational purposes only.
    Neufin does not provide financial advice. Investment decisions should be made in consultation with financial professionals.</p>
    <p>Data source: Yahoo Finance | Last updated: {}</p>
    <div style="margin-top: 15px; opacity: 0.7;">
        <img src="neufin-icon.svg" alt="Neufin" height="24" style="vertical-align: middle; margin-right: 8px;">
        © 2025 Neufin Financial Intelligence
    </div>
</div>
""".format(st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
