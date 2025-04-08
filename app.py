import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Import custom modules
from data_fetcher import fetch_stock_data, fetch_market_news, fetch_sector_performance
from sentiment_analyzer import analyze_text_sentiment, analyze_stock_sentiment
from utils import format_sentiment_score, get_sentiment_color, get_market_indices

# Page configuration
st.set_page_config(
    page_title="Market Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
)

# Initialize session state variables
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
if 'time_period' not in st.session_state:
    st.session_state.time_period = '1mo'
if 'refresh_data' not in st.session_state:
    st.session_state.refresh_data = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Title and introduction
st.title("📊 Market Sentiment Analysis Dashboard")

st.markdown("""
This dashboard analyzes market sentiment using financial data from Yahoo Finance.
It combines technical indicators with natural language processing of financial news to provide
a comprehensive view of market sentiment.
""")

# Sidebar for filters and inputs
with st.sidebar:
    st.header("Settings")
    
    # Stock selection
    available_stocks = get_market_indices() + ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'DIS']
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
    
    # Manual refresh button
    if st.button("Refresh Data"):
        st.session_state.refresh_data = True
    
    st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard uses natural language processing to analyze financial news 
    and market data from Yahoo Finance. The sentiment scores range from -1 (extremely 
    negative) to +1 (extremely positive).
    """)

# Main content
def load_dashboard():
    if not st.session_state.selected_stocks:
        st.warning("Please select at least one stock or index to analyze.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Overall Market Sentiment")
        
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
        st.subheader("Latest Market News Sentiment")
        
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
                        <div style='border-left: 5px solid {sentiment_color}; padding-left: 10px; margin-bottom: 10px;'>
                            <h4>{title}</h4>
                            <p>Sentiment: {sentiment_text} ({sentiment_score:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent market news available.")
        except Exception as e:
            st.error(f"Error analyzing news sentiment: {str(e)}")
    
    st.markdown("---")
    
    # Stock-specific sentiment analysis
    st.subheader("Stock-Specific Sentiment Analysis")
    
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
                    # Bar chart for sentiment comparison
                    fig = px.bar(
                        sentiment_df,
                        x='Symbol',
                        y='Sentiment Score',
                        color='Sentiment Score',
                        color_continuous_scale=["red", "orange", "green"],
                        range_color=[-1, 1],
                        title="Sentiment Score by Stock"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Table of sentiment scores
                    st.dataframe(
                        sentiment_df.style.applymap(
                            lambda x: f"background-color: {get_sentiment_color(x)}" if isinstance(x, float) else "",
                            subset=['Sentiment Score']
                        ),
                        use_container_width=True
                    )
    except Exception as e:
        st.error(f"Error analyzing stock sentiment: {str(e)}")
    
    st.markdown("---")
    
    # Historical Price and Volume Charts
    st.subheader("Historical Price & Volume Charts")
    
    try:
        for stock in st.session_state.selected_stocks[:4]:  # Limit to first 4 stocks to avoid clutter
            with st.spinner(f"Loading {stock} data..."):
                stock_data = fetch_stock_data(stock, st.session_state.time_period)
                
                if stock_data is not None and not stock_data.empty:
                    # Create figure with secondary y-axis
                    fig = go.Figure()
                    
                    # Add price line
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        name='Close Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add volume as bar chart on secondary y-axis
                    fig.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name='Volume',
                        yaxis='y2',
                        opacity=0.3,
                        marker=dict(color='gray')
                    ))
                    
                    # Set titles and layout
                    fig.update_layout(
                        title=f"{stock} - Price and Volume",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        yaxis2=dict(
                            title="Volume",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Could not fetch data for {stock}")
    except Exception as e:
        st.error(f"Error loading historical charts: {str(e)}")
    
    st.markdown("---")
    
    # Sector Performance Analysis
    st.subheader("Sector Performance & Sentiment")
    
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
                
                # Create sector performance chart
                fig = px.bar(
                    sectors_df,
                    x='Sector',
                    y='Performance',
                    color='Performance',
                    color_continuous_scale=["red", "orange", "green"],
                    range_color=[-3, 3],  # Assumes performance is in percent
                    title="Sector Performance (%)"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sector table
                st.dataframe(
                    sectors_df[['Sector', 'Performance', 'Sentiment', 'Sentiment Score']].style.format({
                        'Performance': '{:.2f}%',
                        'Sentiment Score': '{:.2f}'
                    }).applymap(
                        lambda x: f"background-color: {get_sentiment_color(x)}" if isinstance(x, float) else "",
                        subset=['Sentiment Score']
                    ),
                    use_container_width=True
                )
            else:
                st.info("No sector performance data available.")
    except Exception as e:
        st.error(f"Error analyzing sector performance: {str(e)}")
    
    # Update last refresh time
    st.session_state.last_update = datetime.now()
    st.session_state.refresh_data = False

# Handle data loading
if st.session_state.refresh_data or (datetime.now() - st.session_state.last_update > timedelta(minutes=15)):
    load_dashboard()
else:
    load_dashboard()

# Footer
st.markdown("---")
st.markdown("**Note**: All sentiment analysis is powered by AI and should be used for informational purposes only.")
st.markdown("Data source: Yahoo Finance. Last updated: " + st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S"))
