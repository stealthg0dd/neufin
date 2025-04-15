"""
LangChain-powered AI agent for Neufin financial platform.
Provides conversational financial intelligence, market analysis, and personalized recommendations.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Import LangChain components
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler

# OpenAI integration
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic

# Import Neufin components for agent tools
from data_fetcher import fetch_stock_data, fetch_market_news, fetch_sector_performance
from sentiment_analyzer import analyze_stock_sentiment, analyze_text_sentiment
from database import get_historical_sentiment
from ai_analyst import generate_investment_thesis, generate_sector_outlook, analyze_global_trade_conditions
from utils import format_sentiment_score, get_sentiment_color, get_market_indices

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Check if keys are available
if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    st.error("Neither OpenAI nor Anthropic API keys are available. Please set at least one API key.")

# Select the AI model to use based on available API keys
if OPENAI_API_KEY:
    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    # do not change this unless explicitly requested by the user
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )
    print("Using OpenAI GPT-4o model")
elif ANTHROPIC_API_KEY:
    # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024.
    # do not change this unless explicitly requested by the user
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022", 
        temperature=0.2,
        anthropic_api_key=ANTHROPIC_API_KEY
    )
    print("Using Anthropic Claude 3.5 Sonnet model")

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming to a Streamlit container."""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.stream_message = None
        
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        if self.stream_message is None:
            self.stream_message = self.container.markdown(self.text)
        else:
            self.stream_message.markdown(self.text)
    
    def on_llm_end(self, response, **kwargs):
        pass

# Define function schemas using pydantic models for agent tools
class StockData(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    time_period: str = Field(description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)")

class NewsQuery(BaseModel):
    limit: int = Field(description="Maximum number of news items to return", default=5)
    market: str = Field(description="Market to fetch news for (global or us)", default="global")

class SectorQuery(BaseModel):
    market: str = Field(description="Market to fetch sectors for (global or us)", default="global")

class StockAnalysisQuery(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to analyze")
    
class SectorAnalysisQuery(BaseModel):
    sector_name: str = Field(description="Name of the sector to analyze")

# Define agent tools
def get_stock_sentiment(args: StockData) -> str:
    """Fetch and analyze sentiment for a specific stock"""
    try:
        stock_data = fetch_stock_data(args.ticker, args.time_period)
        if stock_data is None or stock_data.empty:
            return f"Unable to fetch data for {args.ticker} over {args.time_period} period."
        
        sentiment_score = analyze_stock_sentiment(stock_data)
        sentiment_text = format_sentiment_score(sentiment_score)
        
        # Calculate price change
        start_price = stock_data['Close'].iloc[0]
        end_price = stock_data['Close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100
        
        result = (
            f"Stock: {args.ticker}\n"
            f"Time Period: {args.time_period}\n"
            f"Sentiment: {sentiment_text} ({sentiment_score:.2f})\n"
            f"Price Change: {price_change:.2f}%\n"
            f"Current Price: {end_price:.2f}\n"
        )
        return result
    except Exception as e:
        return f"Error analyzing stock sentiment for {args.ticker}: {str(e)}"

def get_market_news(args: NewsQuery) -> str:
    """Fetch and analyze recent financial news"""
    try:
        news_items = fetch_market_news(args.limit, args.market)
        if not news_items:
            return f"No recent news available for {args.market} market."
        
        result = f"Recent {args.market} market news:\n\n"
        for i, news in enumerate(news_items, 1):
            title = news.get('title', 'No title')
            sentiment_score = analyze_text_sentiment(title)
            sentiment_text = format_sentiment_score(sentiment_score)
            
            result += f"{i}. {title}\n"
            result += f"   Sentiment: {sentiment_text} ({sentiment_score:.2f})\n\n"
        
        return result
    except Exception as e:
        return f"Error fetching market news: {str(e)}"

def get_sector_performance(args: SectorQuery) -> str:
    """Fetch and analyze sector performance data"""
    try:
        sectors = fetch_sector_performance(args.market)
        if not sectors:
            return f"No sector data available for {args.market} market."
        
        result = f"Current {args.market} market sector performance:\n\n"
        for sector in sectors:
            sector_name = sector.get('Sector', 'Unknown')
            performance = sector.get('Performance', 0)
            result += f"{sector_name}: {performance:.2f}%\n"
        
        return result
    except Exception as e:
        return f"Error fetching sector performance: {str(e)}"

def get_market_sentiment() -> str:
    """Get overall market sentiment based on historical data"""
    try:
        sentiment_data = get_historical_sentiment(days=14)
        if not sentiment_data:
            return "No historical sentiment data available."
        
        # Calculate current and trend
        latest_sentiment = sentiment_data[-1]['sentiment_score']
        sentiment_text = format_sentiment_score(latest_sentiment)
        
        # Calculate trend
        if len(sentiment_data) > 5:
            prev_sentiment = sentiment_data[-6]['sentiment_score']
            trend = latest_sentiment - prev_sentiment
            trend_text = "up" if trend > 0 else "down"
        else:
            trend_text = "stable"
        
        result = (
            f"Current Market Sentiment: {sentiment_text} ({latest_sentiment:.2f})\n"
            f"Trend: {trend_text}\n\n"
            f"Recent daily sentiment scores:\n"
        )
        
        # Add last 7 days of sentiment
        for data in sentiment_data[-7:]:
            date_str = data['date'].strftime('%Y-%m-%d')
            score = data['sentiment_score']
            result += f"{date_str}: {score:.2f}\n"
        
        return result
    except Exception as e:
        return f"Error retrieving market sentiment: {str(e)}"

def get_investment_thesis(args: StockAnalysisQuery) -> str:
    """Generate an AI-powered investment thesis for a specific stock"""
    try:
        ticker = args.ticker
        
        # Fetch required data
        stock_data = fetch_stock_data(ticker, '3mo')
        if stock_data is None or stock_data.empty:
            return f"Unable to fetch data for {ticker}."
        
        market_news = fetch_market_news(limit=5)
        
        # Get market sentiment
        sentiment_data = get_historical_sentiment(days=7)
        if sentiment_data:
            market_sentiment = sentiment_data[-1]['sentiment_score']
        else:
            market_sentiment = 0
        
        # Generate thesis
        thesis = generate_investment_thesis(
            ticker, 
            stock_data, 
            market_news, 
            market_sentiment
        )
        
        if not thesis:
            return f"Unable to generate investment thesis for {ticker}."
        
        # Format the response
        result = f"INVESTMENT THESIS FOR {ticker}\n\n"
        
        if 'overview' in thesis:
            result += f"OVERVIEW:\n{thesis['overview']}\n\n"
        
        if 'strengths' in thesis:
            result += "STRENGTHS:\n"
            for point in thesis['strengths']:
                result += f"- {point}\n"
            result += "\n"
        
        if 'weaknesses' in thesis:
            result += "WEAKNESSES:\n"
            for point in thesis['weaknesses']:
                result += f"- {point}\n"
            result += "\n"
        
        if 'recommendation' in thesis:
            result += f"RECOMMENDATION:\n{thesis['recommendation']}\n\n"
        
        if 'target_price' in thesis:
            result += f"TARGET PRICE: {thesis['target_price']}\n"
        
        if 'confidence' in thesis:
            result += f"CONFIDENCE: {thesis['confidence']}/10\n"
        
        return result
    except Exception as e:
        return f"Error generating investment thesis: {str(e)}"

def get_sector_outlook(args: SectorAnalysisQuery) -> str:
    """Generate a comprehensive sector outlook using AI analysis"""
    try:
        sector_name = args.sector_name
        
        # Fetch sector data
        all_sectors = fetch_sector_performance()
        sector_data = next((s for s in all_sectors if s['Sector'].lower() == sector_name.lower()), None)
        
        if not sector_data:
            return f"Sector '{sector_name}' not found. Available sectors: " + ", ".join([s['Sector'] for s in all_sectors])
        
        # Get market sentiment
        sentiment_data = get_historical_sentiment(days=7)
        if sentiment_data:
            market_sentiment = sentiment_data[-1]['sentiment_score']
        else:
            market_sentiment = 0
        
        # Generate outlook
        outlook = generate_sector_outlook(
            sector_name,
            sector_data,
            market_sentiment
        )
        
        if not outlook:
            return f"Unable to generate outlook for {sector_name} sector."
        
        # Format the response
        result = f"SECTOR OUTLOOK: {sector_name.upper()}\n\n"
        
        if 'overview' in outlook:
            result += f"OVERVIEW:\n{outlook['overview']}\n\n"
        
        if 'trends' in outlook:
            result += "KEY TRENDS:\n"
            for trend in outlook['trends']:
                result += f"- {trend}\n"
            result += "\n"
        
        if 'opportunities' in outlook:
            result += "OPPORTUNITIES:\n"
            for opp in outlook['opportunities']:
                result += f"- {opp}\n"
            result += "\n"
        
        if 'challenges' in outlook:
            result += "CHALLENGES:\n"
            for challenge in outlook['challenges']:
                result += f"- {challenge}\n"
            result += "\n"
        
        if 'recommendation' in outlook:
            result += f"RECOMMENDATION:\n{outlook['recommendation']}\n"
        
        return result
    except Exception as e:
        return f"Error generating sector outlook: {str(e)}"

def get_global_trade_impact() -> str:
    """Analyze current global trade conditions and their impacts on investments"""
    try:
        trade_analysis = analyze_global_trade_conditions()
        
        if not trade_analysis:
            return "Unable to generate global trade impact analysis."
        
        # Format the response
        result = "GLOBAL TRADE CONDITIONS ANALYSIS\n\n"
        
        if 'overview' in trade_analysis:
            result += f"OVERVIEW:\n{trade_analysis['overview']}\n\n"
        
        if 'trends' in trade_analysis:
            result += "KEY TRENDS:\n"
            for trend in trade_analysis['trends']:
                result += f"- {trend}\n"
            result += "\n"
        
        if 'impacts' in trade_analysis:
            result += "MARKET IMPACTS:\n"
            for impact in trade_analysis['impacts']:
                result += f"- {impact}\n"
            result += "\n"
        
        if 'opportunities' in trade_analysis:
            result += "INVESTMENT OPPORTUNITIES:\n"
            for opp in trade_analysis['opportunities']:
                result += f"- {opp}\n"
            result += "\n"
        
        if 'risks' in trade_analysis:
            result += "RISKS:\n"
            for risk in trade_analysis['risks']:
                result += f"- {risk}\n"
            result += "\n"
        
        if 'recommendation' in trade_analysis:
            result += f"RECOMMENDATION:\n{trade_analysis['recommendation']}\n"
        
        return result
    except Exception as e:
        return f"Error generating global trade analysis: {str(e)}"

# Create the agent tools list
tools = [
    Tool(
        name="StockSentiment",
        func=get_stock_sentiment,
        description="Analyze sentiment for a specific stock. Input requires ticker (e.g., AAPL, MSFT) and time_period (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y).",
        args_schema=StockData
    ),
    Tool(
        name="MarketNews",
        func=get_market_news,
        description="Fetch and analyze recent financial news. Input requires limit (number of news items) and market (global or us).",
        args_schema=NewsQuery
    ),
    Tool(
        name="SectorPerformance",
        func=get_sector_performance,
        description="Get current sector performance data. Input requires market (global or us).",
        args_schema=SectorQuery
    ),
    Tool(
        name="MarketSentiment",
        func=get_market_sentiment,
        description="Get overall market sentiment based on historical data. No input required."
    ),
    Tool(
        name="InvestmentThesis",
        func=get_investment_thesis,
        description="Generate a detailed investment thesis for a specific stock. Input requires ticker (e.g., AAPL, MSFT).",
        args_schema=StockAnalysisQuery
    ),
    Tool(
        name="SectorOutlook",
        func=get_sector_outlook,
        description="Generate a comprehensive sector outlook with trends, opportunities, and challenges. Input requires sector_name (e.g., Technology, Healthcare).",
        args_schema=SectorAnalysisQuery
    ),
    Tool(
        name="GlobalTradeImpact",
        func=get_global_trade_impact,
        description="Analyze current global trade conditions and their impacts on investments. No input required."
    )
]

# Define the system message template for the agent
system_prompt = """You are NeufinAgent, an AI financial advisor specializing in market sentiment analysis, stock evaluation, and investment recommendations.

You have access to real-time market data, sentiment analysis, news headlines, and AI-powered investment insights through the Neufin platform.

As a financial advisor, you should:
1. Provide precise, data-driven answers to financial questions.
2. Analyze sentiment trends and explain their implications.
3. Recommend specific stocks or sectors based on current sentiment and performance data.
4. Always provide context and explain your reasoning.
5. Use appropriate financial terminology but explain complex concepts.
6. When the user asks about a specific stock, always check its sentiment first.
7. Always look at market news when analyzing market conditions.

If you don't know the answer or can't access necessary data, be honest and clear about limitations.

Remember that investing involves risk, and you should provide balanced perspectives, not guarantees of performance.

You're an expert in helping users understand market sentiment and make informed financial decisions based on data analysis.
"""

# Initialize the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the agent with tools
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

def run_neufin_agent(query, container):
    """Run the NeufinAgent with the given query and stream output to Streamlit"""
    try:
        # Create a streaming callback handler
        callback_handler = StreamlitCallbackHandler(container)
        
        # Execute the agent with the query
        response = agent_executor.invoke(
            {"input": query},
            config={"callbacks": [callback_handler]}
        )
        
        return response
    except Exception as e:
        container.error(f"Error running NeufinAgent: {str(e)}")
        return None

def reset_agent_memory():
    """Reset the agent's conversation memory"""
    global memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return "Memory reset successfully. Starting a new conversation."