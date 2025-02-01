import streamlit as st
import pandas as pd
from utils import (get_stock_data, create_stock_chart, get_key_metrics,
                  calculate_macd, calculate_bollinger_bands, calculate_ichimoku,
                  get_quantitative_signals, calculate_market_correlation,
                  get_dividend_info, calculate_buy_sell_signals, analyze_multibagger_potential,
                  analyze_historical_performance, identify_support_resistance, calculate_momentum_indicators,
                  backtest_quantitative_strategy)
import base64
from datetime import datetime
import plotly.graph_objects as go

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

        # Setelah menampilkan metrik utama, tambahkan informasi dividen
        st.subheader('💰 Informasi Dividen')
        dividend_info = get_dividend_info(symbol)

        if dividend_info['error']:
            if dividend_info['error'] == 'Tidak ada data dividen':
                st.info('Tidak ada data dividen untuk saham ini')
            else:
                st.error(f"Error mengambil data dividen: {dividend_info['error']}")
        else:
            div_col1, div_col2, div_col3 = st.columns(3)
            with div_col1:
                st.metric("Dividen Terakhir", f"Rp{dividend_info['last_dividend']:,.2f}")
            with div_col2:
                st.metric("Dividend Yield", f"{dividend_info['dividend_yield']:.2f}%")
            with div_col3:
                st.metric("Tanggal Dividen Terakhir", dividend_info['last_dividend_date'])

        # Tambahkan sinyal jual/beli sebelum grafik
        st.subheader('🎯 Sinyal Trading')
        signals = calculate_buy_sell_signals(hist_data)

        if 'error' in signals:
            st.error(f"Error menghitung sinyal: {signals['error']}")
        else:
            signal_col1, signal_col2 = st.columns([1, 2])

            with signal_col1:
                signal_color = '🟢' if 'Beli' in signals['current_signal'] else '🔴' if 'Jual' in signals['current_signal'] else '⚪'
                st.metric("Sinyal Saat Ini", f"{signal_color} {signals['current_signal']}")
                st.text(f"Per tanggal: {signals['last_signal_date']}")

            with signal_col2:
                st.markdown("**Analisis:**")
                for explanation in signals['explanation']:
                    st.markdown(f"• {explanation}")

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

        # Analisis Kuantitatif (Metode James Simons)
        st.subheader('📊 Analisis Kuantitatif (Metode James Simons)')

        # Dapatkan sinyal kuantitatif
        quant_signals = get_quantitative_signals(hist_data)

        # Tampilkan Rekomendasi Utama
        st.markdown(f"## {quant_signals['rekomendasi']}")
        st.markdown(f"*Skor Total: {quant_signals['skor_total']}*")

        # Tampilkan alasan-alasan
        st.markdown("### Alasan Rekomendasi:")
        for alasan in quant_signals['alasan']:
            st.markdown(alasan)

        # Tampilkan detail analisis dalam 4 kolom
        st.markdown("### Detail Analisis:")
        quant_cols = st.columns(4)

        with quant_cols[0]:
            st.metric("Momentum", quant_signals['momentum'])
            if 'Bullish' in quant_signals['momentum']:
                st.markdown('🟢 Momentum Positif')
            elif 'Bearish' in quant_signals['momentum']:
                st.markdown('🔴 Momentum Negatif')
            else:
                st.markdown('⚪ Momentum Netral')

        with quant_cols[1]:
            st.metric("Volume", quant_signals['volume'])
            if 'Tinggi' in quant_signals['volume']:
                st.markdown('🟢 Volume Signifikan')
            elif 'Rendah' in quant_signals['volume']:
                st.markdown('🔴 Volume Lemah')
            else:
                st.markdown('⚪ Volume Normal')

        with quant_cols[2]:
            st.metric("Tren", quant_signals['trend'])
            if 'Uptrend' in quant_signals['trend']:
                st.markdown('🟢 Tren Naik')
            elif 'Downtrend' in quant_signals['trend']:
                st.markdown('🔴 Tren Turun')
            else:
                st.markdown('⚪ Tren Sideways')

        with quant_cols[3]:
            st.metric("Volatilitas", quant_signals['volatility'])
            st.markdown(f"ℹ️ {quant_signals['volatility_level']}")

        # Tambahkan catatan penting
        st.markdown("""
        > **Catatan Penting:**
        > - Analisis kuantitatif ini menggunakan metode yang terinspirasi dari pendekatan trading James Simons
        > - Rekomendasi berdasarkan kombinasi momentum, volume, tren, dan volatilitas
        > - Selalu lakukan analisis fundamental dan pertimbangkan faktor eksternal
        > - Gunakan stop loss dan manajemen risiko yang baik
        """)

        # Backtesting Analysis
        st.subheader('🔄 Backtesting Hasil Strategi')
        backtest_results = backtest_quantitative_strategy(hist_data)

        # Display key metrics
        metrics = backtest_results['metrics']
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("Total Return", f"{metrics['total_return']:.1f}%")
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

        with metric_cols[1]:
            st.metric("Total Trades", str(metrics['total_trades']))
            st.metric("Profitable Trades", str(metrics['profitable_trades']))

        with metric_cols[2]:
            st.metric("Avg Return per Trade", f"{metrics['avg_return_per_trade']:.1f}%")
            st.metric("Losing Trades", str(metrics['losing_trades']))

        with metric_cols[3]:
            st.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.1f}%")

        # Show trade history
        st.subheader("📊 Riwayat Trading")
        if backtest_results['trades']:
            trades_df = pd.DataFrame([
                {
                    'Tanggal': trade['date'],
                    'Tipe': 'Beli' if trade['type'] == 'entry' else 'Jual',
                    'Harga': f"Rp{trade['price']:,.2f}",
                    'Jumlah Saham': f"{trade['shares']:,}",
                    'Profit/Loss': f"Rp{trade.get('profit', 0):,.2f}" if trade['type'] == 'exit' else '-',
                    'Return': f"{trade.get('profit_pct', 0):.1f}%" if trade['type'] == 'exit' else '-',
                    'Sinyal': trade['signals']['rekomendasi']
                }
                for trade in backtest_results['trades']
            ])
            st.dataframe(trades_df)

            # Plot equity curve
            equity_df = pd.DataFrame(backtest_results['equity_curve'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=equity_df['value'],
                name='Portfolio Value',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Tanggal',
                yaxis_title='Nilai Portfolio (Rp)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Belum ada sinyal trading yang dihasilkan dalam periode ini")

        # Add warning note
        st.markdown("""
        > **Catatan Penting tentang Backtesting:**
        > - Hasil backtesting adalah simulasi dan tidak menjamin kinerja di masa depan
        > - Biaya transaksi dan slippage tidak diperhitungkan dalam simulasi
        > - Strategi mungkin perlu disesuaikan dengan kondisi pasar terkini
        > - Selalu gunakan manajemen risiko yang baik dalam trading sesungguhnya
        """)

        # Korelasi dengan IHSG
        st.subheader('📈 Korelasi dengan IHSG')
        correlation, corr_error = calculate_market_correlation(symbol, period=period)

        if corr_error:
            st.error(f"Error menghitung korelasi: {corr_error}")
        elif correlation is not None:
            corr_col1, corr_col2 = st.columns([1, 2])
            with corr_col1:
                st.metric("Korelasi dengan IHSG", f"{correlation:.2f}")
            with corr_col2:
                if correlation > 0.7:
                    st.markdown('🟢 Korelasi Kuat Positif - Saham cenderung bergerak searah dengan IHSG')
                elif correlation < -0.7:
                    st.markdown('🔴 Korelasi Kuat Negatif - Saham cenderung bergerak berlawanan dengan IHSG')
                elif -0.3 <= correlation <= 0.3:
                    st.markdown('⚪ Korelasi Lemah - Pergerakan relatif independen dari IHSG')
                else:
                    st.markdown('🟡 Korelasi Moderat - Ada pengaruh IHSG tapi tidak terlalu kuat')
        
        # Tambahkan setelah bagian korelasi IHSG
        st.subheader('🚀 Analisis Potensi Multibagger')
        multibagger = analyze_multibagger_potential(symbol)

        if multibagger['error']:
            st.error(f"Error dalam analisis multibagger: {multibagger['error']}")
        else:
            # Tampilkan kategori dan skor
            st.markdown(f"### {multibagger['kategori']}")
            st.progress(multibagger['potensi'] / 10)
            st.markdown(f"Skor Potensi: {multibagger['potensi']}/10")

            # Tampilkan metrik-metrik penting
            if multibagger['metrics']:
                st.subheader('📊 Metrik Utama')
                metric_cols = st.columns(len(multibagger['metrics']))
                for i, (metric, value) in enumerate(multibagger['metrics'].items()):
                    with metric_cols[i]:
                        metric_name = metric.replace('_', ' ').title()
                        st.metric(metric_name, value)

            # Tampilkan alasan-alasan
            if multibagger['alasan']:
                st.subheader('📝 Analisis Detail')
                for alasan in multibagger['alasan']:
                    st.markdown(alasan)

            # Tambahkan catatan penting
            st.markdown("""
            > **Catatan Penting:**
            > - Analisis ini hanya merupakan indikator awal dan bukan rekomendasi investasi
            > - Lakukan riset mendalam sebelum mengambil keputusan investasi
            > - Perhatikan juga faktor makro ekonomi dan kondisi industri
            """)
        
        # After multibagger section, add historical analysis
        st.markdown('---')
        st.subheader('📊 Analisis Historis')

        # Historical Performance Analysis
        historical_perf, hist_error = analyze_historical_performance(symbol)
        if hist_error:
            st.error(f"Error dalam analisis historis: {hist_error}")
        else:
            # Create tabs for different time periods
            period_tabs = st.tabs(['1 Tahun', '3 Tahun', '5 Tahun'])
            periods = ['1y', '3y', '5y']

            for tab, period in zip(period_tabs, periods):
                with tab:
                    if historical_perf[period]:
                        perf = historical_perf[period]

                        # Display metrics in columns
                        cols = st.columns(3)

                        with cols[0]:
                            st.metric("Total Return", perf['total_return'])
                            st.metric("Volatilitas", perf['volatility'])

                        with cols[1]:
                            st.metric("Harga Tertinggi", f"Rp{perf['max_price']:,.2f}")
                            st.metric("Harga Terendah", f"Rp{perf['min_price']:,.2f}")

                        with cols[2]:
                            st.metric("Tren Volume", perf['volume_trend'])
                            st.metric("Maximum Drawdown", perf['max_drawdown'])
                    else:
                        st.info(f"Data tidak tersedia untuk periode {period}")

        # Support & Resistance Analysis
        st.subheader('🎯 Level Support & Resistance')
        levels, level_error = identify_support_resistance(hist_data)

        if level_error:
            st.error(f"Error mengidentifikasi level: {level_error}")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Level Support")
                for level in levels['support']:
                    st.markdown(f"• Rp{level['price']:,.2f} ({level['date']})")

            with col2:
                st.markdown("### Level Resistance")
                for level in levels['resistance']:
                    st.markdown(f"• Rp{level['price']:,.2f} ({level['date']})")

        # Momentum Analysis
        st.subheader('📈 Analisis Momentum')
        momentum_indicators, mom_error = calculate_momentum_indicators(hist_data)

        if mom_error:
            st.error(f"Error menghitung indikator momentum: {mom_error}")
        else:
            # Display latest momentum values
            mom_cols = st.columns(3)

            with mom_cols[0]:
                latest_roc = momentum_indicators['ROC_20'].iloc[-1]
                st.metric("Rate of Change (20 hari)", 
                         f"{latest_roc:.1f}%",
                         delta=f"{latest_roc - momentum_indicators['ROC_20'].iloc[-2]:.1f}%")

            with mom_cols[1]:
                latest_macd = momentum_indicators['MACD'].iloc[-1]
                st.metric("MACD",
                         f"{latest_macd:.2f}",
                         delta=f"{latest_macd - momentum_indicators['MACD'].iloc[-2]:.2f}")

            with mom_cols[2]:
                latest_rsi = momentum_indicators['RSI'].iloc[-1]
                st.metric("RSI",
                         f"{latest_rsi:.1f}",
                         delta=f"{latest_rsi - momentum_indicators['RSI'].iloc[-2]:.1f}")

            # Add interpretation
            st.markdown("### Interpretasi")

            # ROC Interpretation
            if latest_roc > 10:
                st.markdown("🟢 Momentum sangat positif (ROC > 10%)")
            elif latest_roc > 5:
                st.markdown("🟡 Momentum positif moderat (ROC > 5%)")
            elif latest_roc < -10:
                st.markdown("🔴 Momentum sangat negatif (ROC < -10%)")
            elif latest_roc < -5:
                st.markdown("🟠 Momentum negatif moderat (ROC < -5%)")
            else:
                st.markdown("⚪ Momentum netral (-5% < ROC < 5%)")

            # RSI Interpretation
            if latest_rsi > 70:
                st.markdown("🔴 Overbought (RSI > 70)")
            elif latest_rsi < 30:
                st.markdown("🟢 Oversold (RSI < 30)")
            else:
                st.markdown("⚪ RSI dalam range normal (30-70)")

        st.markdown('---')


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