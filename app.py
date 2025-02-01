import streamlit as st
import pandas as pd
from utils import get_stock_data, create_stock_chart, get_key_metrics
import base64
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Saham",
    page_icon="📈",
    layout="wide"
)

# Load custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header
st.title('📈 Dashboard Analisis Saham')
st.markdown('Masukkan kode saham untuk melihat data dan analisis keuangan')

# Bagian input
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input('Masukkan Kode Saham (contoh: GOTO.JK, BBCA.JK)', '').upper()
with col2:
    period = st.selectbox(
        'Pilih Periode Waktu',
        ('1mo', '3mo', '6mo', '1y', '2y', '5y'),
        index=3,
        format_func=lambda x: {
            '1mo': '1 Bulan',
            '3mo': '3 Bulan',
            '6mo': '6 Bulan',
            '1y': '1 Tahun',
            '2y': '2 Tahun',
            '5y': '5 Tahun'
        }[x]
    )

@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def load_data(symbol, period):
    return get_stock_data(symbol, period)

if symbol:
    # Ambil data saham
    hist_data, info, error = load_data(symbol, period)

    if error:
        st.error(f"Error mengambil data: {error}")
    elif hist_data is not None and info is not None:
        # Tampilkan info perusahaan
        st.subheader(f"{info.get('longName', symbol)} ({symbol})")
        st.markdown(f"*{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}*")

        # Metrik utama
        metrics = get_key_metrics(info)
        cols = st.columns(3)
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.metric(metric, value)

        # Pemilihan Indikator Teknikal
        st.subheader('Indikator Teknikal')
        indicator_cols = st.columns(4)
        with indicator_cols[0]:
            show_rsi = st.checkbox('Tampilkan RSI', value=False)
        with indicator_cols[1]:
            show_macd = st.checkbox('Tampilkan MACD', value=False)
        with indicator_cols[2]:
            show_bollinger = st.checkbox('Tampilkan Pita Bollinger', value=False)
        with indicator_cols[3]:
            show_ichimoku = st.checkbox('Tampilkan Awan Ichimoku', value=False)

        # Simpan preferensi indikator
        indicators = {
            'rsi': show_rsi,
            'macd': show_macd,
            'bollinger': show_bollinger,
            'ichimoku': show_ichimoku
        }

        # Grafik harga
        st.subheader('Grafik Harga')
        fig = create_stock_chart(hist_data, show_indicators=indicators)
        st.plotly_chart(fig, use_container_width=True)

        from utils import calculate_macd, calculate_bollinger_bands, calculate_ichimoku

        # Ringkasan Analisis Teknikal
        if any(indicators.values()):
            st.subheader('Ringkasan Analisis Teknikal')
            summary_cols = st.columns(len([x for x in indicators.values() if x]))
            col_idx = 0

            if show_rsi:
                with summary_cols[col_idx]:
                    rsi_value = hist_data['Close'].iloc[-1]
                    st.metric("RSI (14)", f"{rsi_value:.2f}")
                    if rsi_value > 70:
                        st.write("🔴 Overbought (Jenuh Beli)")
                    elif rsi_value < 30:
                        st.write("🟢 Oversold (Jenuh Jual)")
                    else:
                        st.write("⚪ Netral")
                col_idx += 1

            if show_macd:
                with summary_cols[col_idx]:
                    st.write("Sinyal MACD")
                    macd, signal = calculate_macd(hist_data)
                    if macd.iloc[-1] > signal.iloc[-1]:
                        st.write("🟢 Bullish (Tren Naik)")
                    else:
                        st.write("🔴 Bearish (Tren Turun)")
                col_idx += 1

            if show_bollinger:
                with summary_cols[col_idx]:
                    st.write("Pita Bollinger")
                    upper, lower = calculate_bollinger_bands(hist_data)
                    current_price = hist_data['Close'].iloc[-1]
                    if current_price > upper.iloc[-1]:
                        st.write("🔴 Di Atas Pita Atas")
                    elif current_price < lower.iloc[-1]:
                        st.write("🟢 Di Bawah Pita Bawah")
                    else:
                        st.write("⚪ Di Dalam Pita")
                col_idx += 1

            if show_ichimoku:
                with summary_cols[col_idx]:
                    st.write("Awan Ichimoku")
                    tenkan, kijun, senkou_a, senkou_b, _ = calculate_ichimoku(hist_data)
                    current_price = hist_data['Close'].iloc[-1]
                    if current_price > senkou_a.iloc[-1] and current_price > senkou_b.iloc[-1]:
                        st.write("🟢 Sangat Bullish (Di Atas Awan)")
                    elif current_price < senkou_a.iloc[-1] and current_price < senkou_b.iloc[-1]:
                        st.write("🔴 Sangat Bearish (Di Bawah Awan)")
                    else:
                        st.write("⚪ Dalam Awan (Netral)")
                col_idx += 1

        # Tabel data historis
        st.subheader('Data Historis')
        df_display = hist_data.copy()
        df_display.index = df_display.index.strftime('%Y-%m-%d')
        st.dataframe(df_display)

        # Tombol unduh
        csv = hist_data.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        filename = f"{symbol}_data_saham_{datetime.now().strftime('%Y%m%d')}.csv"
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Unduh File CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Deskripsi perusahaan
        if info.get('longBusinessSummary'):
            st.subheader('Deskripsi Perusahaan')
            st.write(info['longBusinessSummary'])
    else:
        st.warning("Mohon masukkan kode saham yang valid")

# Footer
st.markdown('---')
st.markdown('Data disediakan oleh Yahoo Finance')