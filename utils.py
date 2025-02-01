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

def calculate_alligator(data):
    """Calculate Alligator indicator (Jaw, Teeth, Lips)"""
    # Calculate SMMA for each line
    jaw = data['Close'].ewm(span=13, adjust=False).mean().shift(8)  # Blue line
    teeth = data['Close'].ewm(span=8, adjust=False).mean().shift(5)  # Red line
    lips = data['Close'].ewm(span=5, adjust=False).mean().shift(3)  # Green line
    return jaw, teeth, lips

def calculate_ichimoku(data):
    """Calculate Ichimoku Cloud components"""
    # Convert period to 9, 26, 52 for Tenkan-sen, Kijun-sen, and Senkou Span B
    high_9 = data['High'].rolling(window=9).max()
    low_9 = data['Low'].rolling(window=9).min()
    high_26 = data['High'].rolling(window=26).max()
    low_26 = data['Low'].rolling(window=26).min()
    high_52 = data['High'].rolling(window=52).max()
    low_52 = data['Low'].rolling(window=52).min()

    # Calculate Components
    tenkan = (high_9 + low_9) / 2
    kijun = (high_26 + low_26) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    chikou = data['Close'].shift(-26)  # 26 periods behind

    return tenkan, kijun, senkou_a, senkou_b, chikou

def create_stock_chart(df, show_indicators=None):
    """Create an interactive stock price chart with technical indicators"""
    if show_indicators is None:
        show_indicators = {'rsi': False, 'macd': False, 'bollinger': False, 'alligator': False, 'ichimoku': False}

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

    # Add technical indicators based on show_indicators
    if show_indicators['ichimoku']:
        tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df)

        # Add Ichimoku components
        fig.add_trace(go.Scatter(
            x=df.index, y=tenkan,
            name='Tenkan-sen',
            line=dict(color='red', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=kijun,
            name='Kijun-sen',
            line=dict(color='blue', width=1)
        ))

        # Add Senkou Span A and B (Cloud)
        fig.add_trace(go.Scatter(
            x=df.index, y=senkou_a,
            name='Senkou Span A',
            line=dict(color='green', width=0.5),
            fill=None
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=senkou_b,
            name='Senkou Span B',
            line=dict(color='red', width=0.5),
            fill='tonexty'  # Fill between Senkou Span A and B
        ))

        # Add Chikou Span
        fig.add_trace(go.Scatter(
            x=df.index, y=chikou,
            name='Chikou Span',
            line=dict(color='purple', width=1)
        ))

    if show_indicators['alligator']:
        jaw, teeth, lips = calculate_alligator(df)
        fig.add_trace(go.Scatter(
            x=df.index, y=jaw,
            name='Alligator (Jaw)',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=teeth,
            name='Alligator (Teeth)',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=lips,
            name='Alligator (Lips)',
            line=dict(color='green', width=2)
        ))

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

    # Update layout
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