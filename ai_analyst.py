import os
import sys
import anthropic
from anthropic import Anthropic
import json
import streamlit as st
import pandas as pd
from datetime import datetime

#the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024

def get_anthropic_client():
    """Initialize and return the Anthropic client"""
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not anthropic_key:
        st.error("ANTHROPIC_API_KEY environment variable is not set. Please provide your API key.")
        return None
    
    return Anthropic(api_key=anthropic_key)

def analyze_global_trade_conditions():
    """
    Use Anthropic to analyze current global trade conditions and their impacts on investments.
    
    Returns:
        dict: Analysis results including trends, impacts, and recommendations
    """
    client = get_anthropic_client()
    
    if not client:
        return {"error": "Anthropic API client could not be initialized"}
    
    try:
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""
        Today is {current_date}. I need you to analyze the current global trade conditions focusing on tariffs, sanctions, trade tensions, and their impacts on investment markets.
        
        Please provide a comprehensive analysis that includes:
        
        1. A concise summary of current global trade conditions
        2. Major ongoing trade tensions and geopolitical hotspots affecting markets
        3. Significant trade policies and tariffs currently in effect
        4. Active sanctions programs and their market impacts
        5. How these conditions are affecting different investment sectors (positive, negative, or mixed impact)
        6. Strategic recommendations for investors based on these conditions
        
        Structure your response as a JSON object with the following keys:
        - summary (string)
        - trade_tensions (array of objects with fields: region, risk_level, description)
        - trade_policies (array of objects with fields: name, countries_involved, summary, impact, affected_sectors)
        - sanctions_programs (array of objects with fields: target, imposed_by, summary, market_impact)
        - sector_impacts (array of objects with fields: sector, impact, description)
        - investor_recommendations (array of objects with fields: strategy, suitable_for, description)
        
        For risk_level, use "High", "Medium", or "Low". For sector impact, use "Positive", "Negative", "Mixed", or "Neutral".
        
        Return only the JSON object without any additional text.
        """
        
        # Simulate or actually call Anthropic API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.2,
            system="You are an expert financial analyst specializing in global trade impacts on financial markets. Provide accurate, insightful analysis based on real-world data and trends.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the result
        try:
            content = response.content[0].text
            # Find JSON content (might be wrapped in code blocks)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content
            
            analysis = json.loads(json_content)
            
            # Add reference date
            analysis["reference_date"] = current_date
            
            return analysis
            
        except json.JSONDecodeError:
            return {
                "error": "Could not parse response as JSON",
                "reference_date": current_date,
                "summary": "Analysis temporarily unavailable. Please try again later.",
            }
            
    except Exception as e:
        return {
            "error": f"Error analyzing global trade conditions: {str(e)}",
            "reference_date": current_date,
            "summary": "Analysis temporarily unavailable due to service error. Please try again later.",
        }

def generate_investment_thesis(ticker, stock_data, market_news, market_sentiment):
    """
    Generate an investment thesis for a specific stock using AI analysis.
    
    Args:
        ticker (str): Stock ticker symbol
        stock_data (DataFrame): Historical stock data
        market_news (list): Relevant market news
        market_sentiment (float): Overall market sentiment score
        
    Returns:
        dict: Investment thesis including strengths, weaknesses, and recommendation
    """
    client = get_anthropic_client()
    
    if not client:
        return {"error": "Anthropic API client could not be initialized"}
    
    try:
        # Prepare stock data summary
        data_summary = ""
        if stock_data is not None and not stock_data.empty:
            # Calculate price change
            start_price = stock_data['Close'].iloc[0]
            end_price = stock_data['Close'].iloc[-1]
            price_change_pct = ((end_price - start_price) / start_price) * 100
            
            # Calculate trading volume
            avg_volume = stock_data['Volume'].mean()
            
            # Calculate volatility
            volatility = stock_data['Close'].pct_change().std() * 100
            
            data_summary = f"""
            Stock: {ticker}
            Current Price: ${end_price:.2f}
            Price Change: {price_change_pct:.2f}%
            Average Trading Volume: {int(avg_volume):,}
            Volatility: {volatility:.2f}%
            """
        
        # Prepare news summary - filter for relevant news for this stock
        news_summary = ""
        relevant_news = []
        
        if market_news:
            for news in market_news:
                if ticker in news.get('title', '') or ticker in news.get('summary', ''):
                    relevant_news.append(news)
            
            if relevant_news:
                news_summary = "Recent relevant news:\n"
                for news in relevant_news[:3]:  # Limit to top 3 relevant news
                    news_summary += f"- {news.get('title', 'No title')}\n"
            else:
                news_summary = "No recent stock-specific news found."
        
        # Format the overall market sentiment
        sentiment_text = "neutral"
        if market_sentiment > 0.3:
            sentiment_text = "strongly bullish"
        elif market_sentiment > 0:
            sentiment_text = "mildly bullish"
        elif market_sentiment > -0.3:
            sentiment_text = "mildly bearish"
        else:
            sentiment_text = "strongly bearish"
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""
        Today is {current_date}. I need you to generate a comprehensive investment thesis for {ticker} based on the following information:
        
        {data_summary}
        
        {news_summary}
        
        Overall market sentiment is currently {sentiment_text} ({market_sentiment:.2f} on a scale from -1 to 1).
        
        Please provide a thorough investment thesis that includes:
        
        1. A brief company overview
        2. Key strengths and competitive advantages (list format)
        3. Risk factors and challenges (list format)
        4. Valuation assessment
        5. Investment recommendation (Strong Buy, Buy, Hold, Sell, or Strong Sell)
        6. Target price range
        7. Confidence level in the recommendation (High, Moderate, Low)
        8. Suggested investment timeframe (Short-term, Medium-term, Long-term)
        
        Structure your response as a JSON object with the following keys:
        - company_overview (string)
        - key_strengths (array of strings)
        - risk_factors (array of strings)
        - valuation_assessment (string)
        - investment_recommendation (string)
        - target_price_range (string)
        - confidence_level (string)
        - investment_timeframe (string)
        
        Return only the JSON object without any additional text.
        """
        
        # Call Anthropic API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.2,
            system="You are an expert financial analyst with 20 years of experience in equity research. Provide accurate, well-reasoned investment analysis based on the data provided. Focus on providing actionable insights and realistic assessments.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the result
        try:
            content = response.content[0].text
            # Find JSON content (might be wrapped in code blocks)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content
            
            thesis = json.loads(json_content)
            
            # Add reference data
            thesis["ticker"] = ticker
            thesis["analysis_date"] = current_date
            
            return thesis
            
        except json.JSONDecodeError:
            return {
                "error": "Could not parse response as JSON",
                "ticker": ticker,
                "analysis_date": current_date,
            }
            
    except Exception as e:
        return {
            "error": f"Error generating investment thesis: {str(e)}",
            "ticker": ticker,
            "analysis_date": current_date,
        }

def generate_sector_outlook(sector_name, sector_data, market_sentiment):
    """
    Generate a comprehensive sector outlook using AI analysis.
    
    Args:
        sector_name (str): Name of the sector
        sector_data (dict): Performance data for the sector
        market_sentiment (float): Overall market sentiment score
        
    Returns:
        dict: Sector outlook including trends, opportunities, and challenges
    """
    client = get_anthropic_client()
    
    if not client:
        return {"error": "Anthropic API client could not be initialized"}
    
    try:
        # Format the overall market sentiment
        sentiment_text = "neutral"
        if market_sentiment > 0.3:
            sentiment_text = "strongly bullish"
        elif market_sentiment > 0:
            sentiment_text = "mildly bullish"
        elif market_sentiment > -0.3:
            sentiment_text = "mildly bearish"
        else:
            sentiment_text = "strongly bearish"
        
        # Format sector performance
        sector_performance = sector_data.get('Performance', 0)
        performance_text = "flat"
        if sector_performance > 5:
            performance_text = "strongly outperforming"
        elif sector_performance > 0:
            performance_text = "slightly outperforming"
        elif sector_performance > -5:
            performance_text = "slightly underperforming"
        else:
            performance_text = "strongly underperforming"
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""
        Today is {current_date}. I need you to generate a comprehensive outlook for the {sector_name} sector based on the following information:
        
        Sector: {sector_name}
        Current Performance: {sector_performance:.2f}% ({performance_text} the broader market)
        Overall Market Sentiment: {sentiment_text} ({market_sentiment:.2f} on a scale from -1 to 1)
        
        Please provide a thorough sector analysis that includes:
        
        1. A sector overview and current positioning
        2. Key drivers and catalysts (list format)
        3. Major challenges and headwinds (list format)
        4. Key innovation trends in the sector (list format)
        5. Competitive landscape analysis
        6. Regulatory environment assessment
        7. Short-term and long-term outlook
        8. Investment strategy recommendations
        9. Top companies to watch in the sector (list format)
        10. Opportunities for investors (list format)
        
        Structure your response as a JSON object with the following keys:
        - sector_overview (string)
        - key_drivers (array of strings)
        - challenges (array of strings)
        - innovation_trends (array of strings)
        - competitive_landscape (string)
        - regulatory_environment (string)
        - outlook (object with keys: short_term, long_term)
        - investment_strategy (string)
        - top_companies_to_watch (array of strings)
        - opportunities (array of strings)
        
        Return only the JSON object without any additional text.
        """
        
        # Call Anthropic API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2500,
            temperature=0.2,
            system="You are an expert sector analyst with deep knowledge of industry trends, competitive dynamics, and investment implications. Provide insightful, accurate sector analysis based on the data provided.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the result
        try:
            content = response.content[0].text
            # Find JSON content (might be wrapped in code blocks)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content
            
            outlook = json.loads(json_content)
            
            # Add reference data
            outlook["sector"] = sector_name
            outlook["analysis_date"] = current_date
            outlook["performance"] = sector_performance
            
            return outlook
            
        except json.JSONDecodeError:
            return {
                "error": "Could not parse response as JSON",
                "sector": sector_name,
                "analysis_date": current_date,
            }
            
    except Exception as e:
        return {
            "error": f"Error generating sector outlook: {str(e)}",
            "sector": sector_name,
            "analysis_date": current_date,
        }