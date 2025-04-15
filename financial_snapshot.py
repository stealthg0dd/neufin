"""
Financial Snapshot Generator for Neufin platform.
Creates a comprehensive one-click financial snapshot with interactive data points,
personalized wellness score, and AI-powered mood tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import random
import time

# Import platform modules
from data_fetcher import fetch_stock_data, fetch_market_news, fetch_sector_performance
from sentiment_analyzer import analyze_text_sentiment as analyze_sentiment_for_text
from sentiment_analyzer import analyze_stock_sentiment as get_market_sentiment
from ai_analyst import get_anthropic_client
from agent import run_neufin_agent, StockData, get_stock_sentiment

# Emoji mapping for financial mood tracking
MOOD_EMOJIS = {
    "very_bullish": "🚀",  # Rocket (very bullish)
    "bullish": "📈",       # Chart up (bullish)
    "neutral": "💼",       # Briefcase (neutral)
    "bearish": "📉",       # Chart down (bearish)
    "very_bearish": "🧸",  # Teddy bear (very bearish)
}

# Financial wellness score categories
WELLNESS_CATEGORIES = [
    {"name": "Portfolio Diversification", "weight": 0.25},
    {"name": "Risk Management", "weight": 0.20},
    {"name": "Market Timing", "weight": 0.15},
    {"name": "Long-term Strategy", "weight": 0.25},
    {"name": "Cash Flow Management", "weight": 0.15}
]

def calculate_financial_wellness_score(user_data, market_sentiment):
    """
    Calculate a personalized financial wellness score based on user data and market conditions.
    
    Args:
        user_data (dict): User portfolio and preference data
        market_sentiment (float): Current market sentiment score
        
    Returns:
        dict: Scores for different wellness categories and an overall score
    """
    # Default scores if user data is not available
    if not user_data:
        # Generate scores that lean slightly toward the market sentiment
        base_score = 0.5 + (market_sentiment - 0.5) * 0.3
        
        # Generate slightly varied scores around the base
        scores = {
            "Portfolio Diversification": max(0.1, min(1.0, base_score + random.uniform(-0.1, 0.1))),
            "Risk Management": max(0.1, min(1.0, base_score + random.uniform(-0.1, 0.1))),
            "Market Timing": max(0.1, min(1.0, base_score + random.uniform(-0.1, 0.1))),
            "Long-term Strategy": max(0.1, min(1.0, base_score + random.uniform(-0.1, 0.1))),
            "Cash Flow Management": max(0.1, min(1.0, base_score + random.uniform(-0.1, 0.1)))
        }
    else:
        # Use actual user data to calculate real scores
        scores = {
            "Portfolio Diversification": user_data.get("diversification_score", 0.7),
            "Risk Management": user_data.get("risk_score", 0.6),
            "Market Timing": user_data.get("timing_score", 0.5),
            "Long-term Strategy": user_data.get("strategy_score", 0.7),
            "Cash Flow Management": user_data.get("cash_flow_score", 0.6)
        }
    
    # Calculate weighted overall score
    overall_score = 0
    for category in WELLNESS_CATEGORIES:
        overall_score += scores[category["name"]] * category["weight"]
        
    return {
        "categories": scores,
        "overall": overall_score
    }

def get_financial_mood_emoji(sentiment_score):
    """
    Convert a sentiment score into a corresponding mood emoji
    
    Args:
        sentiment_score (float): Sentiment score between 0 and 1
        
    Returns:
        str: Emoji representing the financial mood
    """
    if sentiment_score >= 0.8:
        return MOOD_EMOJIS["very_bullish"]
    elif sentiment_score >= 0.6:
        return MOOD_EMOJIS["bullish"]
    elif sentiment_score >= 0.4:
        return MOOD_EMOJIS["neutral"]
    elif sentiment_score >= 0.2:
        return MOOD_EMOJIS["bearish"]
    else:
        return MOOD_EMOJIS["very_bearish"]

def create_radar_chart(wellness_data):
    """
    Create a radar chart for the financial wellness scores
    
    Args:
        wellness_data (dict): Financial wellness score data
        
    Returns:
        plotly.graph_objects.Figure: Radar chart figure
    """
    categories = list(wellness_data["categories"].keys())
    values = list(wellness_data["categories"].values())
    
    # Add the first value again to close the loop
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    # Add trace for the current scores
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Score',
        line_color='#7B68EE',
        fillcolor='rgba(123, 104, 238, 0.3)'
    ))
    
    # Add a trace for the "ideal" score
    ideal_values = [1] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=ideal_values,
        theta=categories,
        fill='toself',
        name='Ideal Score',
        line_color='rgba(0, 200, 83, 0.5)',
        fillcolor='rgba(0, 200, 83, 0.1)'
    ))
    
    # Set layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(
            color='#E0E0E0'
        )
    )
    
    return fig

def create_interactive_stock_chart(stock_data, ticker):
    """
    Create an interactive stock chart with tooltips and annotations
    
    Args:
        stock_data (DataFrame): Historical stock data
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: Interactive stock chart
    """
    # Create the base chart
    fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['open'],
        high=stock_data['high'],
        low=stock_data['low'],
        close=stock_data['close'],
        name=ticker
    ))
    
    # Add volume bars at the bottom
    fig.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['volume'],
        name='Volume',
        marker_color='rgba(123, 104, 238, 0.3)',
        opacity=0.4,
        yaxis='y2'
    ))
    
    # Add a moving average
    ma20 = stock_data['close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=ma20,
        line=dict(color='rgba(255, 152, 0, 0.8)', width=2),
        name='20-Day MA'
    ))
    
    # Add notable events (price spikes, dips, earnings)
    events = []
    for i in range(1, len(stock_data) - 1):
        current = stock_data.iloc[i]
        prev = stock_data.iloc[i-1]
        next_day = stock_data.iloc[i+1]
        
        # Check for price spike
        if current['close'] > prev['close'] * 1.05 and current['close'] > next_day['close']:
            events.append({
                'date': stock_data.index[i],
                'price': current['close'],
                'type': 'spike',
                'text': 'Price spike'
            })
        
        # Check for significant drop
        elif current['close'] < prev['close'] * 0.95 and current['close'] < next_day['close']:
            events.append({
                'date': stock_data.index[i],
                'price': current['close'],
                'type': 'drop',
                'text': 'Price drop'
            })
    
    # Add annotations for events
    for event in events[:5]:  # Limit to 5 events to avoid clutter
        fig.add_annotation(
            x=event['date'],
            y=event['price'],
            text=event['text'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='#E0E0E0',
            font=dict(size=10, color='#E0E0E0'),
            bgcolor='rgba(0, 0, 0, 0.6)',
            bordercolor='#7B68EE',
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price & Volume',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        yaxis2=dict(
            title='Volume',
            titlefont=dict(color='rgba(123, 104, 238, 1)'),
            tickfont=dict(color='rgba(123, 104, 238, 1)'),
            anchor='x',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0.03)',
        font=dict(
            color='#E0E0E0'
        )
    )
    
    return fig

def generate_ai_insights(ticker, stock_data, market_sentiment):
    """
    Generate AI-powered insights for the financial snapshot using LangChain
    
    Args:
        ticker (str): Stock ticker symbol
        stock_data (DataFrame): Historical stock data
        market_sentiment (float): Current market sentiment score
        
    Returns:
        str: AI-generated insights
    """
    # Create a placeholder for the AI response
    insight_placeholder = st.empty()
    insight_placeholder.markdown("*Generating AI insights...*")
    
    # Prepare the container for streaming the response
    insight_container = st.container()
    
    # Create the query for the agent
    query = f"Provide a concise analysis of {ticker} based on recent price movement and market conditions. Current overall market sentiment is {market_sentiment:.2f} (0-1 scale). Highlight key points for investors in 3-4 bullet points."
    
    # Run the agent and stream the response
    run_neufin_agent(query, insight_container)
    
    # Clear the placeholder when done
    insight_placeholder.empty()
    
    return True

def create_financial_snapshot(ticker="AAPL", time_period="1mo", user_data=None):
    """
    Create a comprehensive financial snapshot with interactive elements
    
    Args:
        ticker (str): Stock ticker symbol
        time_period (str): Time period for analysis
        user_data (dict, optional): User portfolio and preference data
        
    Returns:
        None: Displays the snapshot in the Streamlit UI
    """
    st.markdown("## 📊 Financial Snapshot Generator")
    
    with st.expander("Customize Snapshot", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Stock Symbol", value=ticker).upper()
        
        with col2:
            time_period = st.selectbox(
                "Time Period", 
                options=["1w", "1mo", "3mo", "6mo", "1y"],
                index=1
            )
    
    if st.button("📸 Generate Snapshot", key="generate_snapshot", type="primary"):
        with st.spinner("Generating comprehensive financial snapshot..."):
            # Create multiple tabs for different aspects of the snapshot
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Data", "💹 Wellness Score", "🧠 AI Insights", "👁️ Overview"])
            
            # Fetch required data
            stock_data = fetch_stock_data(ticker, period=time_period)
            market_sentiment = get_market_sentiment("SPY")
            
            # Generate stock sentiment
            stock_sentiment_args = StockData(ticker=ticker, time_period=time_period)
            stock_sentiment_result = get_stock_sentiment(stock_sentiment_args)
            if isinstance(stock_sentiment_result, str):
                try:
                    stock_sentiment_data = json.loads(stock_sentiment_result)
                    stock_sentiment = stock_sentiment_data.get("sentiment_score", 0.5)
                except:
                    stock_sentiment = 0.5
            else:
                stock_sentiment = 0.5
            
            # Calculate financial wellness score
            wellness_data = calculate_financial_wellness_score(user_data, market_sentiment)
            
            # Get mood emoji based on stock sentiment
            mood_emoji = get_financial_mood_emoji(stock_sentiment)
            
            # ------------------- Tab 1: Market Data -------------------
            with tab1:
                st.markdown(f"### {ticker} Market Analysis")
                
                # Display stock chart
                interactive_chart = create_interactive_stock_chart(stock_data, ticker)
                st.plotly_chart(interactive_chart, use_container_width=True)
                
                # Add micro-interactions for data points
                st.markdown("#### 🔍 Key Data Points")
                st.markdown("*Click on any metric for a detailed explanation*")
                
                # Calculate key metrics
                latest_price = stock_data['close'].iloc[-1]
                price_change = (stock_data['close'].iloc[-1] - stock_data['close'].iloc[0])
                price_change_pct = (price_change / stock_data['close'].iloc[0]) * 100
                avg_volume = stock_data['volume'].mean()
                volatility = stock_data['close'].pct_change().std() * 100
                
                # Create metrics with micro-interactions
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric(
                        "Current Price",
                        f"${latest_price:.2f}",
                        f"{price_change_pct:.2f}%"
                    )
                    if st.button("ℹ️", key="info_price"):
                        st.info("Current price reflects the latest closing price available for the selected timeframe.")
                
                with metric_cols[1]:
                    st.metric(
                        "Volatility",
                        f"{volatility:.2f}%"
                    )
                    if st.button("ℹ️", key="info_volatility"):
                        st.info("Volatility is measured as the standard deviation of daily price changes, indicating how much the stock price fluctuates.")
                
                with metric_cols[2]:
                    st.metric(
                        "Avg. Volume",
                        f"{avg_volume/1000000:.1f}M"
                    )
                    if st.button("ℹ️", key="info_volume"):
                        st.info("Average trading volume represents the typical number of shares traded daily, indicating liquidity and interest in the stock.")
                
                with metric_cols[3]:
                    sentiment_label = "Bullish" if stock_sentiment > 0.5 else "Bearish"
                    sentiment_delta = f"{(stock_sentiment - 0.5) * 200:.1f}%" if stock_sentiment > 0.5 else f"{(stock_sentiment - 0.5) * 200:.1f}%"
                    st.metric(
                        "Sentiment",
                        sentiment_label,
                        sentiment_delta
                    )
                    if st.button("ℹ️", key="info_sentiment"):
                        st.info("Market sentiment indicates the overall attitude of investors toward this stock, based on news, social media, and technical indicators.")
            
            # ------------------- Tab 2: Wellness Score -------------------
            with tab2:
                st.markdown("### 🧘‍♂️ Financial Wellness Assessment")
                
                # Display overall wellness score
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 30px;">
                    <div style="text-align: center; background: rgba(123, 104, 238, 0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(123, 104, 238, 0.3);">
                        <div style="font-size: 1.2rem; margin-bottom: 10px;">Your Financial Wellness Score</div>
                        <div style="font-size: 3rem; font-weight: bold; color: {get_score_color(wellness_data['overall'])};">
                            {int(wellness_data['overall'] * 100)}
                            <span style="font-size: 1.5rem;">/100</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display radar chart
                radar_chart = create_radar_chart(wellness_data)
                st.plotly_chart(radar_chart, use_container_width=True)
                
                # Add recommendations based on lowest scores
                st.markdown("#### 💡 Improvement Recommendations")
                
                # Find the lowest scoring categories
                scores = wellness_data["categories"]
                sorted_categories = sorted(scores.items(), key=lambda x: x[1])
                
                for i, (category, score) in enumerate(sorted_categories[:2]):
                    st.markdown(f"""
                    <div style="background: rgba(0, 0, 0, 0.1); padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">{category} ({int(score * 100)}/100)</div>
                        <div>{get_recommendation_for_category(category, score)}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ------------------- Tab 3: AI Insights -------------------
            with tab3:
                st.markdown("### 🤖 AI-Powered Insights")
                st.markdown(f"##### Stock: {ticker} | Market Mood: {mood_emoji}")
                
                # Generate and display AI insights
                generate_ai_insights(ticker, stock_data, market_sentiment)
                
                # Show query examples
                st.markdown("#### Ask More Questions")
                st.markdown("You can ask the AI assistant more detailed questions about this stock or the market:")
                
                example_queries = [
                    f"What are the main risks for {ticker} investors right now?",
                    f"How does {ticker}'s performance compare to its sector?",
                    f"What technical indicators are important for {ticker} right now?",
                    "How might current economic conditions affect this stock?"
                ]
                
                selected_query = st.selectbox("Sample questions:", example_queries)
                
                if st.button("Ask AI", key="ask_ai_button"):
                    # Create a container for the response
                    response_container = st.container()
                    
                    # Run the agent with the selected query
                    run_neufin_agent(selected_query, response_container)
            
            # ------------------- Tab 4: Overview -------------------
            with tab4:
                st.markdown("### 📊 Financial Snapshot Overview")
                
                # Create columns for the overview
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown(f"#### {ticker} Summary")
                    
                    # Display key metrics and mood emoji
                    st.markdown(f"""
                    <div style="background: rgba(0, 0, 0, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="font-size: 3rem; margin-right: 15px;">{mood_emoji}</div>
                            <div>
                                <div style="font-size: 1.2rem; font-weight: bold;">{ticker} Market Mood</div>
                                <div style="opacity: 0.8;">AI-detected sentiment based on market data and news</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <div>Current Price:</div>
                                <div style="font-weight: bold;">${latest_price:.2f}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <div>Price Change:</div>
                                <div style="font-weight: bold; color: {'#00C853' if price_change_pct > 0 else '#FF3D00'}">{price_change_pct:.2f}%</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <div>Volatility:</div>
                                <div style="font-weight: bold;">{volatility:.2f}%</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <div>Wellness Score:</div>
                                <div style="font-weight: bold;">{int(wellness_data['overall'] * 100)}/100</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recent news section
                    st.markdown("#### 📰 Recent News")
                    
                    # Fetch recent news for the stock
                    news = fetch_market_news(limit=3)
                    
                    for item in news:
                        st.markdown(f"""
                        <div style="background: rgba(0, 0, 0, 0.1); padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                            <div style="font-weight: bold; margin-bottom: 5px;">{item['title']}</div>
                            <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 10px;">{item.get('published_date', 'Recent')}</div>
                            <div style="font-size: 0.95rem;">{item.get('summary', '')[:150]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Display mini financial wellness gauge
                    wellness_score = int(wellness_data['overall'] * 100)
                    st.markdown(f"""
                    <div style="background: rgba(0, 0, 0, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                        <div style="font-size: 1.1rem; margin-bottom: 10px;">Financial Wellness</div>
                        <div style="position: relative; height: 120px; width: 120px; margin: 0 auto;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1.8rem; font-weight: bold;">
                                {wellness_score}
                            </div>
                            <svg width="120" height="120" viewBox="0 0 120 120">
                                <circle cx="60" cy="60" r="54" fill="none" stroke="#333" stroke-width="12" />
                                <circle cx="60" cy="60" r="54" fill="none" stroke="{get_score_color(wellness_data['overall'])}" 
                                        stroke-width="12" stroke-dasharray="{wellness_score * 3.39} 339" stroke-dashoffset="84.75" />
                            </svg>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display simplified sector context
                    st.markdown("#### 🏢 Sector Context")
                    
                    # Fetch sector data
                    sectors = fetch_sector_performance()
                    
                    # Find this stock's sector (simplified example)
                    # In a real implementation, we would have a mapping of tickers to sectors
                    stock_sector = "Technology" if ticker in ["AAPL", "MSFT", "GOOGL"] else "Healthcare" if ticker in ["JNJ", "PFE", "UNH"] else "Finance" if ticker in ["JPM", "BAC", "GS"] else "Consumer Discretionary"
                    
                    # Display the sector performance
                    for sector in sectors[:5]:
                        is_stock_sector = sector['name'] == stock_sector
                        st.markdown(f"""
                        <div style="background: {'rgba(123, 104, 238, 0.1)' if is_stock_sector else 'rgba(0, 0, 0, 0.1)'}; 
                                    padding: 10px; border-radius: 5px; margin-bottom: 8px;
                                    border: {f'1px solid rgba(123, 104, 238, 0.5)' if is_stock_sector else 'none'}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="{'font-weight: bold;' if is_stock_sector else ''}">{sector['name']}</div>
                                <div style="color: {'#00C853' if sector['performance'] > 0 else '#FF3D00'}">
                                    {sector['performance']}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add export/share buttons
                    st.markdown("#### 📤 Share Snapshot")
                    share_cols = st.columns(2)
                    
                    with share_cols[0]:
                        if st.button("📱 Export as PDF", key="export_pdf"):
                            st.info("PDF export functionality would be implemented here.")
                    
                    with share_cols[1]:
                        if st.button("🔗 Share Link", key="share_link"):
                            st.info("Link sharing functionality would be implemented here.")

def get_score_color(score):
    """Get color for financial wellness score"""
    if score >= 0.75:
        return "#00C853"  # Green
    elif score >= 0.5:
        return "#FFD600"  # Yellow
    elif score >= 0.25:
        return "#FF9100"  # Orange
    else:
        return "#FF3D00"  # Red

def get_recommendation_for_category(category, score):
    """Generate personalized recommendations based on category and score"""
    if category == "Portfolio Diversification":
        if score < 0.5:
            return "Consider diversifying your holdings across different sectors and asset classes to reduce risk."
        else:
            return "Your portfolio is reasonably diversified. Consider fine-tuning with international exposure."
    
    elif category == "Risk Management":
        if score < 0.5:
            return "Review your position sizes and consider implementing stop-loss orders to protect your investments."
        else:
            return "Your risk management approach is solid. Consider periodic rebalancing to maintain your target allocation."
    
    elif category == "Market Timing":
        if score < 0.5:
            return "Avoid excessive trading based on short-term market movements. Consider a more systematic approach."
        else:
            return "Your market timing approach is balanced. Consider dollar-cost averaging for new investments."
    
    elif category == "Long-term Strategy":
        if score < 0.5:
            return "Develop a clearer long-term investment plan with specific goals and timelines."
        else:
            return "Your long-term strategy is well-defined. Consider reviewing it annually to ensure it still aligns with your goals."
    
    elif category == "Cash Flow Management":
        if score < 0.5:
            return "Review your investment-to-savings ratio and ensure you have adequate emergency funds before investing more."
        else:
            return "Your cash flow management is effective. Consider optimizing your investment contributions for tax efficiency."
    
    return "Focus on improving this aspect of your financial wellness through education and consistent practice."