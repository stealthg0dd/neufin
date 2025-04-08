import streamlit as st
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
from investment_advisor import get_stock_recommendations, analyze_global_trade_impact, get_sector_insights
from subscription_manager import initialize_subscription_state, show_login_form, show_subscription_options, process_payment, check_feature_access, show_upgrade_prompt, check_trial_status
from ai_analyst import analyze_global_trade_conditions, generate_investment_thesis, generate_sector_outlook

# Page configuration
st.set_page_config(
    page_title="Market Sentinel Pro",
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
if 'ai_analyses' not in st.session_state:
    st.session_state.ai_analyses = {}
    
# Initialize subscription state
initialize_subscription_state()
check_trial_status()

# Main title with branded styling
st.markdown("""
<div style="background-color:#1E88E5; padding:10px; border-radius:10px; margin-bottom:10px;">
  <h1 style="color:white; text-align:center;">📊 Market Sentinel Pro</h1>
  <h3 style="color:white; text-align:center;">AI-Powered Market Intelligence Platform</h3>
</div>
""", unsafe_allow_html=True)

# Introduction with value proposition
st.markdown("""
<div style="background-color:#f8f8f8; padding:15px; border-radius:5px; margin-bottom:20px;">
This platform delivers AI-powered market sentiment analysis using real-time financial data. 
Our advanced algorithms analyze technical indicators, news sentiment, and global trade impacts 
to provide actionable investment insights.
</div>
""", unsafe_allow_html=True)

# Sidebar for filters, inputs, and account features
with st.sidebar:
    # Logo and brand
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/streamlit-mark-color.png", width=100)
    st.header("Market Sentinel Pro")
    
    # Add login/subscription management
    show_login_form()
    
    # Show subscription options if logged in
    if st.session_state.user_logged_in:
        show_subscription_options()
        process_payment()
    
    st.markdown("---")
    st.header("Dashboard Settings")
    
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
    
    # Manual refresh button
    if st.button("Refresh Data"):
        st.session_state.refresh_data = True
    
    st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This platform uses advanced AI and natural language processing to analyze financial news 
    and market data from Yahoo Finance. The sentiment scores range from -1 (extremely 
    negative) to +1 (extremely positive).
    
    **Premium features** include AI-generated investment recommendations, global trade impact analysis,
    and detailed sector insights.
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

# Add premium features - Investment recommendations
st.markdown("---")
recommendation_tab, sector_tab, global_tab = st.tabs(["Investment Recommendations", "Sector Insights", "Global Trade Analysis"])

with recommendation_tab:
    st.subheader("🌟 AI-Powered Investment Recommendations")
    
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
    st.subheader("🔍 Detailed Sector Insights")
    
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
    st.subheader("🌎 Global Trade Impact Analysis")
    
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

# Handle data loading
if st.session_state.refresh_data or (datetime.now() - st.session_state.last_update > timedelta(minutes=15)):
    load_dashboard()
else:
    load_dashboard()

# Add commercial info and footer
st.markdown("---")

# Commercial footer
st.markdown("""
<div style='background-color:#f8f8f8; padding:20px; border-radius:5px;'>
<h3 style='text-align:center;'>Market Sentinel Pro - Your AI-Powered Investment Platform</h3>
<p style='text-align:center;'>
Subscribe to our Premium Plan for advanced market insights, personalized stock recommendations, and global trade impact analysis.
</p>
<div style='display:flex; justify-content:center; gap:10px;'>
<div style='text-align:center; padding:10px; background-color:#1E88E520; border-radius:5px;'>🚀 AI-Powered Analysis</div>
<div style='text-align:center; padding:10px; background-color:#1E88E520; border-radius:5px;'>💰 Investment Recommendations</div>
<div style='text-align:center; padding:10px; background-color:#1E88E520; border-radius:5px;'>📊 Sector Insights</div>
<div style='text-align:center; padding:10px; background-color:#1E88E520; border-radius:5px;'>🌎 Global Trade Impact</div>
</div>
</div>
""", unsafe_allow_html=True)

# Disclaimer footer
st.markdown("""
<div style='font-size:0.8em; color:#666; text-align:center; margin-top:20px;'>
<p>**Note**: All sentiment analysis and recommendations are powered by AI and should be used for informational purposes only.
Market Sentinel Pro does not provide financial advice. Investment decisions should be made in consultation with financial professionals.</p>
<p>Data source: Yahoo Finance. Last updated: {}</p>
</div>
""".format(st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
