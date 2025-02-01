import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def get_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    """Calculate MACD"""
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def create_stock_chart(df, show_indicators=None):
    """Create an interactive stock price chart with technical indicators"""
    if show_indicators is None:
        show_indicators = {'rsi': False, 'macd': False, 'bollinger': False}

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Main candlestick chart
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

    # Add moving averages
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

    # Add technical indicators based on show_indicators
    if show_indicators['bollinger']:
        upper_band, lower_band = calculate_bollinger_bands(df)
        fig.add_trace(go.Scatter(
            x=df.index, y=upper_band,
            name='Upper Bollinger Band',
            line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=lower_band,
            name='Lower Bollinger Band',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ))

    if show_indicators['rsi']:
        rsi = calculate_rsi(df)
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            name='RSI',
            yaxis='y3',
            line=dict(color='purple')
        ))

    if show_indicators['macd']:
        macd, signal = calculate_macd(df)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd,
            name='MACD',
            yaxis='y4',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=signal,
            name='Signal Line',
            yaxis='y4',
            line=dict(color='orange')
        ))

    # Update layout based on enabled indicators
    layout_updates = {
        'title': 'Stock Price Chart',
        'yaxis': dict(title='Price'),
        'yaxis2': dict(title='Volume', overlaying='y', side='right'),
        'height': 800,
        'showlegend': True,
        'legend': dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    }

    if show_indicators['rsi']:
        layout_updates['yaxis3'] = dict(
            title='RSI',
            overlaying='y',
            side='right',
            position=0.97
        )

    if show_indicators['macd']:
        layout_updates['yaxis4'] = dict(
            title='MACD',
            overlaying='y',
            side='right',
            position=0.94
        )

    fig.update_layout(**layout_updates)
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