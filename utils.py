import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def format_large_number(num):
    """Format angka besar dengan akhiran Rb, Jt, M"""
    if not num:
        return "N/A"
    if num >= 1e9:
        return f"{num/1e9:.2f}M"
    elif num >= 1e6:
        return f"{num/1e6:.2f}Jt"
    elif num >= 1e3:
        return f"{num/1e3:.2f}Rb"
    return f"{num:.2f}"

def get_key_metrics(info):
    """Ekstrak metrik utama dari info saham"""
    metrics = {
        'Kapitalisasi Pasar': format_large_number(info.get('marketCap', 0)),
        'Rasio P/E': f"{info.get('trailingPE', 0):.2f}",
        'Tertinggi 52 Minggu': f"Rp{info.get('fiftyTwoWeekHigh', 0):.2f}",
        'Terendah 52 Minggu': f"Rp{info.get('fiftyTwoWeekLow', 0):.2f}",
        'Volume': format_large_number(info.get('volume', 0)),
        'Rata-rata Volume': format_large_number(info.get('averageVolume', 0))
    }
    return metrics

def get_stock_data(symbol, period='1y'):
    """Ambil data saham dari Yahoo Finance"""
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

def calculate_ichimoku(data):
    """Calculate Ichimoku Cloud components"""
    high_9 = data['High'].rolling(window=9).max()
    low_9 = data['Low'].rolling(window=9).min()
    high_26 = data['High'].rolling(window=26).max()
    low_26 = data['Low'].rolling(window=26).min()
    high_52 = data['High'].rolling(window=52).max()
    low_52 = data['Low'].rolling(window=52).min()

    tenkan = (high_9 + low_9) / 2
    kijun = (high_26 + low_26) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    chikou = data['Close'].shift(-26)

    return tenkan, kijun, senkou_a, senkou_b, chikou

def create_stock_chart(df, show_indicators=None):
    """Buat grafik harga saham interaktif dengan indikator teknikal"""
    if show_indicators is None:
        show_indicators = {'rsi': False, 'macd': False, 'bollinger': False, 'ichimoku': False}

    # Buat figure dengan sumbu y sekunder
    fig = go.Figure()

    # Grafik candlestick utama
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))

    # Tambahkan volume
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

    # Add technical indicators based on show_indicators
    if show_indicators['ichimoku']:
        tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df)

        # Add Ichimoku components
        fig.add_trace(go.Scatter(
            x=df.index, y=tenkan,
            name='Tenkan-sen (Garis Konversi)',
            line=dict(color='red', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=kijun,
            name='Kijun-sen (Garis Dasar)',
            line=dict(color='blue', width=1)
        ))

        # Create future dates array for cloud
        future_dates = pd.date_range(start=df.index[-1], periods=26)
        dates_for_cloud = pd.concat([pd.Series(df.index), pd.Series(future_dates)])

        # Extend senkou spans with NaN values for future dates
        senkou_a_extended = pd.concat([senkou_a, pd.Series([None] * 26)])
        senkou_b_extended = pd.concat([senkou_b, pd.Series([None] * 26)])

        # Add Senkou Span A and B (Cloud)
        fig.add_trace(go.Scatter(
            x=dates_for_cloud,
            y=senkou_a_extended,
            name='Senkou Span A (Garis Depan A)',
            line=dict(color='rgba(76, 175, 80, 0.3)'),
            fill=None
        ))

        fig.add_trace(go.Scatter(
            x=dates_for_cloud,
            y=senkou_b_extended,
            name='Senkou Span B (Garis Depan B)',
            line=dict(color='rgba(255, 82, 82, 0.3)'),
            fill='tonexty'
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=chikou,
            name='Chikou Span (Garis Tertinggal)',
            line=dict(color='purple', width=1)
        ))

    if show_indicators['bollinger']:
        upper_band, lower_band = calculate_bollinger_bands(df)
        fig.add_trace(go.Scatter(
            x=df.index, y=upper_band,
            name='Pita Bollinger Atas',
            line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=lower_band,
            name='Pita Bollinger Bawah',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ))

    # Update layout
    layout_updates = {
        'title': 'Grafik Harga Saham',
        'yaxis': dict(title='Harga', domain=[0.3, 1]),
        'yaxis2': dict(
            title='Volume',
            domain=[0, 0.2],
            anchor='x'
        ),
        'height': 800,
        'showlegend': True,
        'legend': dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        'margin': dict(t=30, l=60, r=60, b=30)
    }

    if show_indicators['rsi']:
        rsi = calculate_rsi(df)
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            name='RSI',
            yaxis='y3',
            line=dict(color='purple')
        ))
        layout_updates['yaxis3'] = dict(
            title='RSI',
            overlaying='y',
            side='right',
            position=0.97
        )

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
            name='Garis Sinyal',
            yaxis='y4',
            line=dict(color='orange')
        ))
        layout_updates['yaxis4'] = dict(
            title='MACD',
            overlaying='y',
            side='right',
            position=0.94
        )

    fig.update_layout(**layout_updates)
    fig.update_xaxes(rangeslider_visible=True)

    return fig