# Market Sentiment Dashboard

A Streamlit-based dashboard that uses AI to analyze financial feeds from Yahoo Finance and provide market sentiment analysis.

## Features

- Integration with Yahoo Finance API to fetch financial data
- AI-powered sentiment analysis of financial news and data
- Interactive dashboard displaying market sentiment metrics
- Historical sentiment trend visualization
- Stock/sector-specific sentiment analysis
- Basic filtering options for different market segments
- Responsive data tables showing key financial metrics

## Technology Stack

- Streamlit: For the interactive dashboard interface
- yfinance: For fetching Yahoo Finance data
- TextBlob: For NLP-based sentiment analysis
- Pandas: For data manipulation and analysis
- Plotly: For interactive data visualizations

## Running the Application

To run the application, execute the following command:

```bash
streamlit run app.py --server.port 5000
