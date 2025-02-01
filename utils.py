import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

def get_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info, None
    except Exception as e:
        return None, None, str(e)

def create_stock_chart(df):
    """Create an interactive stock price chart"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA20'],
        name='20-day MA',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA50'],
        name='50-day MA',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='Stock Price Chart',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_title='Date',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def format_large_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num:.2f}"

def get_key_metrics(info):
    """Extract key metrics from stock info"""
    metrics = {
        'Market Cap': format_large_number(info.get('marketCap', 0)),
        'P/E Ratio': f"{info.get('trailingPE', 0):.2f}",
        '52 Week High': f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
        '52 Week Low': f"${info.get('fiftyTwoWeekLow', 0):.2f}",
        'Volume': format_large_number(info.get('volume', 0)),
        'Avg Volume': format_large_number(info.get('averageVolume', 0))
    }
    return metrics
