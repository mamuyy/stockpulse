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

def calculate_momentum(data, period=10):
    """Hitung indikator momentum"""
    return data['Close'].diff(period)

def analyze_volume_trend(data, period=20):
    """Analisis tren volume"""
    volume_ma = data['Volume'].rolling(window=period).mean()
    volume_ratio = data['Volume'] / volume_ma
    return volume_ratio

def calculate_market_correlation(symbol, market_index='^JKSE', period='1y'):
    """Hitung korelasi dengan indeks pasar"""
    try:
        # Ambil data saham dan IHSG secara bersamaan
        data = yf.download([symbol, market_index], period=period, progress=False)
        if data.empty:
            return None, "Data tidak tersedia"

        # Ambil harga penutupan
        closing_prices = data['Close']

        # Cek apakah ada data untuk kedua instrumen
        if len(closing_prices.columns) != 2:
            return None, "Data tidak lengkap untuk salah satu instrumen"

        # Hitung return harian
        returns = closing_prices.pct_change().dropna()

        # Cek apakah ada cukup data untuk korelasi
        if len(returns) < 2:
            return None, "Data tidak cukup untuk menghitung korelasi"

        # Hitung korelasi
        correlation = returns.iloc[:, 0].corr(returns.iloc[:, 1])

        if np.isnan(correlation):
            return None, "Tidak dapat menghitung korelasi (hasil NaN)"

        return correlation, None

    except Exception as e:
        return None, f"Error: {str(e)}"

def get_quantitative_signals(data):
    """Implementasi sinyal trading kuantitatif dengan metode James Simons"""
    signals = {}

    # 1. Momentum Analysis with Multiple Timeframes
    momentum_5d = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100
    momentum_10d = ((data['Close'].iloc[-1] / data['Close'].iloc[-10]) - 1) * 100
    momentum_20d = ((data['Close'].iloc[-1] / data['Close'].iloc[-20]) - 1) * 100

    momentum_score = 0
    if momentum_5d > 3 and momentum_10d > 5 and momentum_20d > 7:
        signals['momentum'] = 'Sangat Bullish'
        momentum_score = 2
    elif momentum_5d > 1 and momentum_10d > 2 and momentum_20d > 3:
        signals['momentum'] = 'Bullish'
        momentum_score = 1
    elif momentum_5d < -3 and momentum_10d < -5 and momentum_20d < -7:
        signals['momentum'] = 'Sangat Bearish'
        momentum_score = -2
    elif momentum_5d < -1 and momentum_10d < -2 and momentum_20d < -3:
        signals['momentum'] = 'Bearish'
        momentum_score = -1
    else:
        signals['momentum'] = 'Netral'
        momentum_score = 0

    # 2. Volume Analysis with Trend Confirmation
    volume_ma = data['Volume'].rolling(window=20).mean()
    recent_volume = data['Volume'].iloc[-5:].mean()
    volume_ratio = recent_volume / volume_ma.iloc[-1]

    volume_score = 0
    if volume_ratio > 2.0:
        signals['volume'] = 'Volume Sangat Tinggi'
        volume_score = 2 if momentum_score > 0 else -2
    elif volume_ratio > 1.5:
        signals['volume'] = 'Volume Tinggi'
        volume_score = 1 if momentum_score > 0 else -1
    elif volume_ratio < 0.5:
        signals['volume'] = 'Volume Sangat Rendah'
        volume_score = -1
    else:
        signals['volume'] = 'Volume Normal'

    # 3. Price Trend Strength
    ma20 = data['Close'].rolling(window=20).mean()
    ma50 = data['Close'].rolling(window=50).mean()
    ma200 = data['Close'].rolling(window=200).mean()

    trend_score = 0
    if data['Close'].iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
        signals['trend'] = 'Uptrend Kuat'
        trend_score = 2
    elif data['Close'].iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]:
        signals['trend'] = 'Uptrend'
        trend_score = 1
    elif data['Close'].iloc[-1] < ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
        signals['trend'] = 'Downtrend Kuat'
        trend_score = -2
    elif data['Close'].iloc[-1] < ma20.iloc[-1] < ma50.iloc[-1]:
        signals['trend'] = 'Downtrend'
        trend_score = -1
    else:
        signals['trend'] = 'Sideways'
        trend_score = 0

    # 4. Volatility Analysis
    returns = data['Close'].pct_change()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    signals['volatility'] = f"{volatility:.1f}%"

    volatility_score = 0
    if volatility > 50:
        signals['volatility_level'] = 'Sangat Tinggi - Hati-hati'
        volatility_score = -1
    elif volatility > 30:
        signals['volatility_level'] = 'Tinggi'
        volatility_score = -0.5
    elif volatility < 15:
        signals['volatility_level'] = 'Rendah'
        volatility_score = 0.5
    else:
        signals['volatility_level'] = 'Normal'
        volatility_score = 0

    # 5. Calculate Overall Signal Score
    total_score = momentum_score + volume_score + trend_score + volatility_score

    # Generate Trading Recommendations
    if total_score >= 3:
        signals['rekomendasi'] = 'BELI KUAT 🟢'
        signals['alasan'] = [
            '✅ Momentum sangat positif',
            '✅ Volume mendukung tren naik',
            '✅ Tren harga bullish',
            '✅ Volatilitas terkendali'
        ]
    elif total_score >= 1:
        signals['rekomendasi'] = 'BELI 🟢'
        signals['alasan'] = ['✅ Kondisi teknikal mendukung untuk beli']
    elif total_score <= -3:
        signals['rekomendasi'] = 'JUAL KUAT 🔴'
        signals['alasan'] = [
            '❌ Momentum sangat negatif',
            '❌ Volume mendukung tren turun',
            '❌ Tren harga bearish',
            '❌ Risiko volatilitas tinggi'
        ]
    elif total_score <= -1:
        signals['rekomendasi'] = 'JUAL 🔴'
        signals['alasan'] = ['❌ Kondisi teknikal mendukung untuk jual']
    else:
        signals['rekomendasi'] = 'TAHAN ⚪'
        signals['alasan'] = ['⚠️ Tunggu sinyal yang lebih jelas']

    signals['skor_total'] = total_score
    return signals

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

def get_dividend_info(symbol):
    """Dapatkan informasi dividen saham"""
    try:
        stock = yf.Ticker(symbol)
        dividends = stock.dividends
        if not dividends.empty:
            last_dividend = dividends.iloc[-1]
            dividend_yield = stock.info.get('dividendYield', 0)
            dividend_date = dividends.index[-1]
            return {
                'last_dividend': last_dividend,
                'dividend_yield': dividend_yield * 100 if dividend_yield else 0,
                'last_dividend_date': dividend_date.strftime('%Y-%m-%d'),
                'error': None
            }
        return {
            'last_dividend': 0,
            'dividend_yield': 0,
            'last_dividend_date': 'N/A',
            'error': 'Tidak ada data dividen'
        }
    except Exception as e:
        return {
            'last_dividend': 0,
            'dividend_yield': 0,
            'last_dividend_date': 'N/A',
            'error': str(e)
        }

def calculate_buy_sell_signals(data):
    """Hitung sinyal jual dan beli berdasarkan beberapa indikator"""
    signals = {
        'current_signal': None,
        'last_signal_date': None,
        'explanation': [],
        'strength': 0  # -3 (Jual Kuat) sampai +3 (Beli Kuat)
    }

    try:
        # 1. RSI Signal
        rsi = calculate_rsi(data)
        if rsi.iloc[-1] < 30:
            signals['strength'] += 1
            signals['explanation'].append('RSI menunjukkan kondisi oversold (jenuh jual)')
        elif rsi.iloc[-1] > 70:
            signals['strength'] -= 1
            signals['explanation'].append('RSI menunjukkan kondisi overbought (jenuh beli)')

        # 2. Moving Average Signal
        ma20 = data['Close'].rolling(window=20).mean()
        ma50 = data['Close'].rolling(window=50).mean()

        if data['Close'].iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]:
            signals['strength'] += 1
            signals['explanation'].append('Harga di atas MA20 dan MA50 (tren naik)')
        elif data['Close'].iloc[-1] < ma20.iloc[-1] < ma50.iloc[-1]:
            signals['strength'] -= 1
            signals['explanation'].append('Harga di bawah MA20 dan MA50 (tren turun)')

        # 3. Volume Signal
        volume_ma = data['Volume'].rolling(window=20).mean()
        if data['Volume'].iloc[-1] > 1.5 * volume_ma.iloc[-1]:
            if signals['strength'] > 0:
                signals['strength'] += 1
                signals['explanation'].append('Volume tinggi mendukung tren naik')
            elif signals['strength'] < 0:
                signals['strength'] -= 1
                signals['explanation'].append('Volume tinggi mendukung tren turun')

        # Determine final signal
        if signals['strength'] >= 2:
            signals['current_signal'] = 'Beli Kuat'
        elif signals['strength'] == 1:
            signals['current_signal'] = 'Beli'
        elif signals['strength'] == 0:
            signals['current_signal'] = 'Netral'
        elif signals['strength'] == -1:
            signals['current_signal'] = 'Jual'
        else:
            signals['current_signal'] = 'Jual Kuat'

        signals['last_signal_date'] = data.index[-1].strftime('%Y-%m-%d')

    except Exception as e:
        signals['error'] = str(e)

    return signals

def analyze_multibagger_potential(symbol):
    """Analisis potensi multibagger dari sebuah saham"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        financials = stock.financials

        analysis = {
            'potensi': 0,  # Skala 0-10
            'alasan': [],
            'metrics': {},
            'error': None
        }

        # 1. Analisis Pertumbuhan Pendapatan
        if not financials.empty and 'Total Revenue' in financials.index:
            rev_growth = ((financials.loc['Total Revenue'][0] / financials.loc['Total Revenue'][1]) - 1) * 100
            analysis['metrics']['pertumbuhan_pendapatan'] = f"{rev_growth:.1f}%"
            if rev_growth > 20:
                analysis['potensi'] += 2
                analysis['alasan'].append("✅ Pertumbuhan pendapatan yang kuat (>20%)")
            elif rev_growth > 10:
                analysis['potensi'] += 1
                analysis['alasan'].append("✅ Pertumbuhan pendapatan yang baik (>10%)")

        # 2. Analisis ROE
        if 'returnOnEquity' in info and info['returnOnEquity']:
            roe = info['returnOnEquity'] * 100
            analysis['metrics']['ROE'] = f"{roe:.1f}%"
            if roe > 20:
                analysis['potensi'] += 2
                analysis['alasan'].append("✅ ROE sangat baik (>20%)")
            elif roe > 15:
                analysis['potensi'] += 1
                analysis['alasan'].append("✅ ROE yang baik (>15%)")

        # 3. Analisis Valuasi
        if 'trailingPE' in info and info['trailingPE']:
            pe = info['trailingPE']
            analysis['metrics']['PE_Ratio'] = f"{pe:.1f}"
            if pe < 15:
                analysis['potensi'] += 2
                analysis['alasan'].append("✅ Valuasi menarik (PE < 15)")
            elif pe < 20:
                analysis['potensi'] += 1
                analysis['alasan'].append("✅ Valuasi wajar (PE < 20)")

        # 4. Analisis Hutang
        if 'totalDebt' in info and 'totalCash' in info:
            debt_to_cash = info['totalDebt'] / info['totalCash'] if info['totalCash'] > 0 else float('inf')
            analysis['metrics']['Debt_to_Cash'] = f"{debt_to_cash:.1f}"
            if debt_to_cash < 1:
                analysis['potensi'] += 2
                analysis['alasan'].append("✅ Posisi keuangan yang kuat (Hutang < Kas)")
            elif debt_to_cash < 2:
                analysis['potensi'] += 1
                analysis['alasan'].append("✅ Posisi keuangan yang sehat")

        # 5. Analisis Tren Harga
        hist = stock.history(period="1y")
        if not hist.empty:
            price_change = ((hist['Close'][-1] / hist['Close'][0]) - 1) * 100
            analysis['metrics']['perubahan_harga_1y'] = f"{price_change:.1f}%"

            # Volume trend
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'][-20:].mean()
            volume_change = ((recent_volume / avg_volume) - 1) * 100
            analysis['metrics']['tren_volume'] = "Meningkat" if volume_change > 20 else "Stabil" if volume_change > -20 else "Menurun"

            if volume_change > 20 and price_change > 0:
                analysis['potensi'] += 2
                analysis['alasan'].append("✅ Tren harga dan volume positif")

        # Kategorikan potensi
        if analysis['potensi'] >= 8:
            analysis['kategori'] = "Potensi Multibagger Tinggi 🌟"
        elif analysis['potensi'] >= 6:
            analysis['kategori'] = "Potensi Multibagger Moderat ⭐"
        elif analysis['potensi'] >= 4:
            analysis['kategori'] = "Potensi Multibagger Rendah 📊"
        else:
            analysis['kategori'] = "Belum Menunjukkan Potensi Multibagger 📉"

        return analysis

    except Exception as e:
        return {
            'potensi': 0,
            'alasan': [],
            'metrics': {},
            'error': str(e)
        }

def analyze_historical_performance(symbol, periods=['1y', '3y', '5y']):
    """Analyze historical performance across multiple time periods"""
    try:
        results = {}
        stock = yf.Ticker(symbol)

        for period in periods:
            hist = stock.history(period=period)
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                max_price = hist['High'].max()
                min_price = hist['Low'].min()

                # Calculate returns
                total_return = ((end_price / start_price) - 1) * 100

                # Calculate volatility (annualized)
                daily_returns = hist['Close'].pct_change()
                volatility = daily_returns.std() * np.sqrt(252) * 100

                # Calculate average volume
                avg_volume = hist['Volume'].mean()
                recent_volume = hist['Volume'][-20:].mean()
                volume_trend = ((recent_volume / avg_volume) - 1) * 100

                # Calculate maximum drawdown
                rolling_max = hist['Close'].expanding().max()
                drawdowns = ((hist['Close'] - rolling_max) / rolling_max) * 100
                max_drawdown = drawdowns.min()

                results[period] = {
                    'total_return': f"{total_return:.1f}%",
                    'volatility': f"{volatility:.1f}%",
                    'max_price': max_price,
                    'min_price': min_price,
                    'volume_trend': f"{volume_trend:.1f}%",
                    'max_drawdown': f"{max_drawdown:.1f}%"
                }
            else:
                results[period] = None

        return results, None
    except Exception as e:
        return None, str(e)

def identify_support_resistance(data, window=20):
    """Identify potential support and resistance levels"""
    try:
        levels = {
            'support': [],
            'resistance': []
        }

        # Use rolling min/max to identify potential levels
        rolling_low = data['Low'].rolling(window=window).min()
        rolling_high = data['High'].rolling(window=window).max()

        # Find local minima for support
        for i in range(window, len(data)-window):
            if (rolling_low.iloc[i] == rolling_low.iloc[i-window:i+window].min()):
                levels['support'].append({
                    'price': rolling_low.iloc[i],
                    'date': data.index[i].strftime('%Y-%m-%d')
                })

        # Find local maxima for resistance
        for i in range(window, len(data)-window):
            if (rolling_high.iloc[i] == rolling_high.iloc[i-window:i+window].max()):
                levels['resistance'].append({
                    'price': rolling_high.iloc[i],
                    'date': data.index[i].strftime('%Y-%m-%d')
                })

        # Sort levels and take most recent ones
        levels['support'] = sorted(levels['support'], key=lambda x: x['price'])[-3:]
        levels['resistance'] = sorted(levels['resistance'], key=lambda x: x['price'])[:3]

        return levels, None
    except Exception as e:
        return None, str(e)

def calculate_momentum_indicators(data):
    """Calculate various momentum indicators"""
    try:
        indicators = {}

        # Rate of Change (ROC)
        indicators['ROC_5'] = ((data['Close'] / data['Close'].shift(5)) - 1) * 100
        indicators['ROC_10'] = ((data['Close'] / data['Close'].shift(10)) - 1) * 100
        indicators['ROC_20'] = ((data['Close'] / data['Close'].shift(20)) - 1) * 100

        # Moving Average Convergence Divergence (MACD)
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['Signal_Line'] = indicators['MACD'].ewm(span=9, adjust=False).mean()

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))

        return indicators, None
    except Exception as e:
        return None, str(e)
    
def backtest_quantitative_strategy(data, initial_capital=100_000_000):
    """
    Backtest the James Simons quantitative strategy
    Returns performance metrics and trade history
    """
    results = {
        'trades': [],
        'metrics': {
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'avg_return_per_trade': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
        },
        'equity_curve': []
    }

    position = None
    entry_price = 0
    capital = initial_capital
    max_capital = initial_capital

    # Calculate indicators once
    ma20 = data['Close'].rolling(window=20).mean()
    ma50 = data['Close'].rolling(window=50).mean()
    ma200 = data['Close'].rolling(window=200).mean()
    volume_ma = data['Volume'].rolling(window=20).mean()

    for i in range(200, len(data)-1):  # Start after MA200 is available
        current_data = data.iloc[:i+1]  # Use data up to current point

        # Get signals based on our quantitative strategy
        signals = get_quantitative_signals(current_data)
        current_price = data['Close'].iloc[i]

        # Entry logic
        if position is None:  # Not in position
            if signals['rekomendasi'] in ['BELI KUAT 🟢', 'BELI 🟢']:
                position= 'long'
                entry_price = current_price
                shares = (capital * 0.95) // current_price  # Use 95% of capital
                capital -= shares * current_price

                results['trades'].append({
                    'type': 'entry',
                    'date': data.index[i].strftime('%Y-%m-%d'),
                    'price': current_price,
                    'shares': shares,
                    'signals': signals
                })

        # Exit logic
        elif position == 'long':
            if signals['rekomendasi'] in ['JUAL KUAT 🔴', 'JUAL 🔴']:
                exit_price = current_price
                shares = next(trade['shares'] for trade in reversed(results['trades']) 
                            if trade['type'] == 'entry')
                capital += shares * exit_price

                # Calculate trade profit/loss
                profit = (exit_price - entry_price) * shares
                profit_pct = (exit_price / entry_price - 1) * 100

                results['trades'].append({
                    'type': 'exit',
                    'date': data.index[i].strftime('%Y-%m-%d'),
                    'price': exit_price,
                    'shares': shares,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'signals': signals
                })

                if profit > 0:
                    results['metrics']['profitable_trades'] += 1
                else:
                    results['metrics']['losing_trades'] += 1

                position = None

        # Track equity curve
        total_value = capital
        if position == 'long':
            shares = next(trade['shares'] for trade in reversed(results['trades']) 
                        if trade['type'] == 'entry')
            total_value += shares * current_price

        results['equity_curve'].append({
            'date': data.index[i].strftime('%Y-%m-%d'),
            'value': total_value
        })

        # Update max capital for drawdown calculation
        max_capital = max(max_capital, total_value)
        current_drawdown = (max_capital - total_value) / max_capital * 100
        results['metrics']['max_drawdown'] = max(
            results['metrics']['max_drawdown'], 
            current_drawdown
        )

    # Calculate final metrics
    total_trades = len([t for t in results['trades'] if t['type'] == 'exit'])
    results['metrics']['total_trades'] = total_trades

    if total_trades > 0:
        results['metrics']['win_rate'] = (results['metrics']['profitable_trades'] / 
                                        total_trades * 100)

        total_profit_pct = sum(t.get('profit_pct', 0) for t in results['trades'] 
                              if t['type'] == 'exit')
        results['metrics']['avg_return_per_trade'] = total_profit_pct / total_trades

        final_value = results['equity_curve'][-1]['value']
        results['metrics']['total_return'] = ((final_value / initial_capital) - 1) * 100

    return results