import pandas as pd
import numpy as np
import streamlit as st
import random
from datetime import datetime
from data_fetcher import fetch_stock_data
from sentiment_analyzer import analyze_stock_sentiment

def get_stock_recommendations(sectors_data, market_sentiment, available_stocks, time_period='1mo'):
    """
    Generate stock investment recommendations based on market sentiment and sector performance.
    
    Args:
        sectors_data (list): List of sector performance data
        market_sentiment (float): Overall market sentiment score
        available_stocks (list): List of available stocks to analyze
        time_period (str): Time period to analyze
        
    Returns:
        DataFrame: Recommended stocks with scores and reasons
    """
    # Initialize empty recommendations list
    recommendations = []
    
    # Get sector-to-stock mapping (simplified)
    sector_mapping = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA'],
        'Communication Services': ['NFLX', 'GOOGL', 'META'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'DIS'],
        'Consumer Staples': ['WMT', 'PG'],
        'Healthcare': ['JNJ', 'PFE'],
        'Financials': ['JPM', 'BAC'],
        'Energy': ['XOM', 'CVX'],
        'Industrials': ['BA', 'CAT'],
        'Utilities': ['NEE'],
        'Market Indices': ['^DJI', '^GSPC', '^IXIC']
    }
    
    # Extract available stocks (non-indices)
    stocks_to_analyze = [s for s in available_stocks if not s.startswith('^') and s in available_stocks]
    
    # Add indices for market benchmark
    indices_to_analyze = [s for s in available_stocks if s.startswith('^') and s in available_stocks]
    
    # Analyze each stock
    try:
        for stock in stocks_to_analyze + indices_to_analyze:
            # Fetch stock data
            stock_data = fetch_stock_data(stock, time_period)
            
            if stock_data is None or stock_data.empty:
                continue
                
            # Calculate stock sentiment
            sentiment_score = analyze_stock_sentiment(stock_data)
            
            # Calculate price change
            if 'Close' in stock_data.columns and len(stock_data) > 0:
                start_price = stock_data['Close'].iloc[0]
                end_price = stock_data['Close'].iloc[-1]
                price_change_pct = ((end_price - start_price) / start_price) * 100
            else:
                price_change_pct = 0
                
            # Determine stock sector
            stock_sector = "Unknown"
            for sector, stocks in sector_mapping.items():
                if stock in stocks:
                    stock_sector = sector
                    break
            
            # Find sector performance if available
            sector_performance = 0
            if sectors_data:
                for sector in sectors_data:
                    if sector['Sector'] == stock_sector:
                        sector_performance = sector['Performance']
                        break
            
            # Calculate recommendation score (0-100)
            # Factors: sentiment (40%), sector performance (30%), price momentum (30%)
            sentiment_component = (sentiment_score + 1) * 20  # Convert -1 to 1 scale to 0-40
            sector_component = min(max(sector_performance + 10, 0), 30)  # 0-30 scale
            momentum_component = min(max(price_change_pct + 15, 0), 30)  # 0-30 scale
            
            score = sentiment_component + sector_component + momentum_component
            
            # Determine recommendation
            if score >= 80:
                recommendation = "Strong Buy"
            elif score >= 65:
                recommendation = "Buy"
            elif score >= 45:
                recommendation = "Hold"
            elif score >= 30:
                recommendation = "Sell"
            else:
                recommendation = "Strong Sell"
                
            # Generate reasoning
            reasoning_components = []
            
            if sentiment_score > 0.3:
                reasoning_components.append("strong positive sentiment")
            elif sentiment_score > 0:
                reasoning_components.append("mildly positive sentiment")
            elif sentiment_score > -0.3:
                reasoning_components.append("mildly negative sentiment")
            else:
                reasoning_components.append("strong negative sentiment")
                
            if sector_performance > 2:
                reasoning_components.append("strong sector performance")
            elif sector_performance > 0:
                reasoning_components.append("positive sector trend")
            elif sector_performance > -2:
                reasoning_components.append("underperforming sector")
            else:
                reasoning_components.append("weak sector performance")
                
            if price_change_pct > 5:
                reasoning_components.append("strong upward momentum")
            elif price_change_pct > 0:
                reasoning_components.append("mild positive momentum")
            elif price_change_pct > -5:
                reasoning_components.append("slight downward trend")
            else:
                reasoning_components.append("significant price decline")
                
            # Combine reasoning
            reasoning = f"Based on {', '.join(reasoning_components)}"
            
            # Add recommendation
            recommendations.append({
                'Ticker': stock,
                'Sector': stock_sector,
                'Current Price': end_price,
                'Price Change (%)': price_change_pct,
                'Sentiment': sentiment_score,
                'Score': score,
                'Recommendation': recommendation,
                'Reasoning': reasoning
            })
        
        # Convert to DataFrame and sort by score
        if recommendations:
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df = recommendations_df.sort_values(by='Score', ascending=False)
            return recommendations_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame()

def analyze_global_trade_impact():
    """
    Analyze current global trade conditions, tariffs, and sanctions.
    
    Returns:
        dict: Analysis of global trade conditions and investment recommendations
    """
    # This is a placeholder that would be replaced with actual API data
    # in a production environment
    
    # Reference date for data
    current_date = datetime.now().strftime("%B %d, %Y")
    
    return {
        "reference_date": current_date,
        "summary": "Global trade conditions show mixed signals with ongoing tariff adjustments and strategic shifts in supply chains. Regional tensions and protectionist policies remain key concerns for investors.",
        "trade_tensions": [
            {
                "region": "US-China Relations",
                "risk_level": "High",
                "description": "Ongoing tensions with bilateral trade restrictions and technology competition affecting global supply chains."
            },
            {
                "region": "European Energy Markets",
                "risk_level": "Medium",
                "description": "Supply constraints and regulatory changes impacting energy markets and downstream industries."
            },
            {
                "region": "Southeast Asian Manufacturing",
                "risk_level": "Low",
                "description": "Emerging as alternative manufacturing hub with favorable trade agreements and incentives."
            }
        ],
        "trade_policies": [
            {
                "name": "Semiconductor Export Controls",
                "countries_involved": ["United States", "Netherlands", "Japan", "China"],
                "summary": "Restrictions on advanced semiconductor technology exports to specific markets",
                "impact": "Creating investment opportunities in domestic chip production and alternative supply chains",
                "affected_sectors": ["Technology", "Manufacturing", "Automotive"]
            },
            {
                "name": "Clean Energy Incentives",
                "countries_involved": ["United States", "European Union", "China"],
                "summary": "Significant subsidies and incentives for clean energy development and manufacturing",
                "impact": "Accelerating investment in renewable energy, battery technology, and related infrastructure",
                "affected_sectors": ["Energy", "Utilities", "Materials", "Automotive"]
            }
        ],
        "sanctions_programs": [
            {
                "target": "Russia",
                "imposed_by": ["United States", "European Union", "United Kingdom", "Japan"],
                "summary": "Comprehensive sanctions affecting financial, energy, and technology sectors",
                "market_impact": "Disruptions in energy markets, commodity prices, and European industrial sectors"
            },
            {
                "target": "Iran",
                "imposed_by": ["United States", "Others"],
                "summary": "Sanctions targeting energy exports and financial transactions",
                "market_impact": "Limited direct market impact but affecting global energy risk premiums"
            }
        ],
        "sector_impacts": [
            {"sector": "Technology", "impact": "Mixed", "description": "Supply chain diversification creating winners and losers"},
            {"sector": "Energy", "impact": "Positive", "description": "Higher prices and incentives for alternative energy"},
            {"sector": "Materials", "impact": "Positive", "description": "Increased demand for critical minerals and materials"},
            {"sector": "Consumer Goods", "impact": "Negative", "description": "Higher input costs and supply chain disruptions"},
            {"sector": "Healthcare", "impact": "Neutral", "description": "Limited direct impact from trade policies"}
        ],
        "investor_recommendations": [
            {
                "strategy": "Supply Chain Diversification",
                "suitable_for": "Conservative/Long-term",
                "description": "Invest in companies with diversified manufacturing footprints and resilient supply chains"
            },
            {
                "strategy": "Critical Materials Exposure",
                "suitable_for": "Balanced/Medium-term",
                "description": "Position in companies involved in production and processing of critical materials"
            },
            {
                "strategy": "Regional Rotation",
                "suitable_for": "Aggressive/Short-term",
                "description": "Tactically shift exposure between regional markets based on policy developments"
            }
        ]
    }

def get_sector_insights(sectors_data):
    """
    Generate insights about the best performing sectors.
    
    Args:
        sectors_data (list): List of sector performance data
        
    Returns:
        dict: Insights about the best performing sectors
    """
    if not sectors_data or len(sectors_data) == 0:
        return {
            "error": "No sector data available"
        }
    
    # Sort sectors by performance
    sorted_sectors = sorted(sectors_data, key=lambda x: x['Performance'], reverse=True)
    
    # Generate insights for top sectors
    top_sectors = sorted_sectors[:3]
    bottom_sectors = sorted_sectors[-3:]
    
    # Create mock insights (would be replaced with ML/AI analysis in production)
    sector_insights = {}
    
    for sector in sorted_sectors:
        sector_name = sector['Sector']
        performance = sector['Performance']
        
        # Generate sector-specific insights
        if sector_name == "Technology":
            sector_insights[sector_name] = {
                "drivers": "Semiconductor strength, cloud computing growth, and AI advancements driving sector performance.",
                "stocks_to_watch": "NVIDIA (NVDA), Advanced Micro Devices (AMD), Microsoft (MSFT)",
                "outlook": "Positive with continued digitalization trends, though valuations remain elevated.",
                "risks": "Regulatory pressures, geopolitical tensions affecting supply chains, high valuations"
            }
        elif sector_name == "Energy":
            sector_insights[sector_name] = {
                "drivers": "Supply constraints, global demand recovery, and geopolitical tensions supporting prices.",
                "stocks_to_watch": "Exxon Mobil (XOM), Chevron (CVX), ConocoPhillips (COP)",
                "outlook": "Positive near-term on tight supply, longer-term challenged by energy transition.",
                "risks": "Transition to renewable energy, potential demand destruction from high prices, regulatory changes"
            }
        elif sector_name == "Financials":
            sector_insights[sector_name] = {
                "drivers": "Rising interest rates improving margins, loan growth, and strong investment banking activity.",
                "stocks_to_watch": "JPMorgan Chase (JPM), Bank of America (BAC), Goldman Sachs (GS)",
                "outlook": "Positive with continued rate environment, though potential slowing economic growth a concern.",
                "risks": "Credit quality deterioration in economic slowdown, flattening yield curve, fintech disruption"
            }
        elif sector_name == "Healthcare":
            sector_insights[sector_name] = {
                "drivers": "Innovation in treatments, aging demographics, COVID-related products and services.",
                "stocks_to_watch": "UnitedHealth Group (UNH), Johnson & Johnson (JNJ), Eli Lilly (LLY)",
                "outlook": "Stable with defensive characteristics, innovation pipeline supporting growth.",
                "risks": "Drug pricing legislation, patent expirations, rising input and labor costs"
            }
        elif sector_name == "Consumer Discretionary":
            sector_insights[sector_name] = {
                "drivers": "Pent-up demand, shifts in consumer spending patterns, e-commerce growth.",
                "stocks_to_watch": "Amazon (AMZN), Home Depot (HD), McDonald's (MCD)",
                "outlook": "Mixed as inflation impacts consumer spending power, particularly in lower income demographics.",
                "risks": "Inflation pressures on margins and consumer spending, supply chain disruptions, labor costs"
            }
        elif sector_name == "Consumer Staples":
            sector_insights[sector_name] = {
                "drivers": "Defensive positioning, pricing power, stable demand patterns.",
                "stocks_to_watch": "Procter & Gamble (PG), Coca-Cola (KO), Walmart (WMT)",
                "outlook": "Stable with inflation pass-through capabilities, though growth remains modest.",
                "risks": "Private label competition, changing consumer preferences, margin pressure from input costs"
            }
        elif sector_name == "Industrials":
            sector_insights[sector_name] = {
                "drivers": "Infrastructure spending, supply chain reshoring, automation trends.",
                "stocks_to_watch": "Caterpillar (CAT), Deere (DE), Union Pacific (UNP)",
                "outlook": "Positive with infrastructure initiatives, though cyclical exposure remains a factor.",
                "risks": "Economic slowdown impacting capital expenditures, supply chain disruptions, labor challenges"
            }
        elif sector_name == "Materials":
            sector_insights[sector_name] = {
                "drivers": "Commodity price strength, infrastructure spending, supply constraints in key materials.",
                "stocks_to_watch": "Freeport-McMoRan (FCX), Nucor (NUE), Air Products (APD)",
                "outlook": "Positive for critical minerals and metals needed for energy transition and infrastructure.",
                "risks": "Cyclical demand patterns, substitution risks, high energy input costs"
            }
        elif sector_name == "Utilities":
            sector_insights[sector_name] = {
                "drivers": "Defensive characteristics, clean energy transition investments, regulated returns.",
                "stocks_to_watch": "NextEra Energy (NEE), Duke Energy (DUK), Southern Company (SO)",
                "outlook": "Stable with growth opportunities in renewables, though interest rate sensitivity a factor.",
                "risks": "Rising interest rates, regulatory changes, high capital expenditure requirements"
            }
        elif sector_name == "Real Estate":
            sector_insights[sector_name] = {
                "drivers": "Reopening benefiting certain segments, inflation hedge characteristics, rent growth.",
                "stocks_to_watch": "Prologis (PLD), American Tower (AMT), Equinix (EQIX)",
                "outlook": "Mixed with work-from-home trends disrupting office while supporting residential and industrial.",
                "risks": "Interest rate increases, changing work and shopping patterns, oversupply in certain markets"
            }
        elif sector_name == "Communication Services":
            sector_insights[sector_name] = {
                "drivers": "Digital advertising growth, streaming competition, 5G infrastructure deployment.",
                "stocks_to_watch": "Alphabet (GOOGL), Meta Platforms (META), T-Mobile (TMUS)",
                "outlook": "Positive for infrastructure providers, mixed for content and social media companies.",
                "risks": "Regulatory scrutiny, competition in streaming, high content costs"
            }
        else:
            # Generic insights for any other sector
            sector_insights[sector_name] = {
                "drivers": f"Multiple factors contributing to {performance:.1f}% performance.",
                "stocks_to_watch": "Key companies in this sector worth monitoring",
                "outlook": "Outlook depends on broader economic conditions and sector-specific factors.",
                "risks": "Economic uncertainty, competitive pressures, and regulatory changes"
            }
    
    return {
        "top_sectors": top_sectors,
        "bottom_sectors": bottom_sectors,
        "insights": sector_insights
    }