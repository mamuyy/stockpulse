import streamlit as st
import pandas as pd
from utils import get_stock_data, create_stock_chart, get_key_metrics
import base64
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header
st.title('📈 Stock Analysis Dashboard')
st.markdown('Enter a stock symbol to view financial data and analysis')

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOGL)', '').upper()
with col2:
    period = st.selectbox(
        'Select Time Period',
        ('1mo', '3mo', '6mo', '1y', '2y', '5y'),
        index=3
    )

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(symbol, period):
    return get_stock_data(symbol, period)

if symbol:
    # Get stock data
    hist_data, info, error = load_data(symbol, period)

    if error:
        st.error(f"Error fetching data: {error}")
    elif hist_data is not None and info is not None:
        # Display company info
        st.subheader(f"{info.get('longName', symbol)} ({symbol})")
        st.markdown(f"*{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}*")

        # Key metrics
        metrics = get_key_metrics(info)
        cols = st.columns(3)
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.metric(metric, value)

        # Technical Indicators Selection
        st.subheader('Technical Indicators')
        indicator_cols = st.columns(3)
        with indicator_cols[0]:
            show_rsi = st.checkbox('Show RSI', value=False)
        with indicator_cols[1]:
            show_macd = st.checkbox('Show MACD', value=False)
        with indicator_cols[2]:
            show_bollinger = st.checkbox('Show Bollinger Bands', value=False)

        # Store indicator preferences
        indicators = {
            'rsi': show_rsi,
            'macd': show_macd,
            'bollinger': show_bollinger
        }

        # Price chart
        st.subheader('Price Chart')
        fig = create_stock_chart(hist_data, show_indicators=indicators)
        st.plotly_chart(fig, use_container_width=True)
        
        from utils import calculate_macd, calculate_bollinger_bands

        # Technical Analysis Summary
        if any(indicators.values()):
            st.subheader('Technical Analysis Summary')
            summary_cols = st.columns(len([x for x in indicators.values() if x]))
            col_idx = 0

            if show_rsi:
                with summary_cols[col_idx]:
                    rsi_value = hist_data['Close'].iloc[-1]
                    st.metric("RSI (14)", f"{rsi_value:.2f}")
                    if rsi_value > 70:
                        st.write("🔴 Overbought")
                    elif rsi_value < 30:
                        st.write("🟢 Oversold")
                    else:
                        st.write("⚪ Neutral")
                col_idx += 1

            if show_macd:
                with summary_cols[col_idx]:
                    st.write("MACD Signal")
                    macd, signal = calculate_macd(hist_data)
                    if macd.iloc[-1] > signal.iloc[-1]:
                        st.write("🟢 Bullish")
                    else:
                        st.write("🔴 Bearish")
                col_idx += 1

            if show_bollinger:
                with summary_cols[col_idx]:
                    st.write("Bollinger Bands")
                    upper, lower = calculate_bollinger_bands(hist_data)
                    current_price = hist_data['Close'].iloc[-1]
                    if current_price > upper.iloc[-1]:
                        st.write("🔴 Above Upper Band")
                    elif current_price < lower.iloc[-1]:
                        st.write("🟢 Below Lower Band")
                    else:
                        st.write("⚪ Within Bands")

        # Data table
        st.subheader('Historical Data')
        df_display = hist_data.copy()
        df_display.index = df_display.index.strftime('%Y-%m-%d')
        st.dataframe(df_display)

        # Download button
        csv = hist_data.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        filename = f"{symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv"
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Company description
        if info.get('longBusinessSummary'):
            st.subheader('Company Description')
            st.write(info['longBusinessSummary'])
    else:
        st.warning("Please enter a valid stock symbol")

# Footer
st.markdown('---')
st.markdown('Data provided by Yahoo Finance')