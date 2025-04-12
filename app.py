import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure page
st.set_page_config(layout="wide", page_title="Breakout Model Scanner")
plt.style.use('ggplot')

# Cache data for performance
@st.cache_data(ttl=3600)
def get_stock_data(symbol, period='3mo'):
    try:
        return yf.Ticker(symbol).history(period=period)
    except:
        return None

# Technical indicators
def calculate_atr(high, low, close, window=14):
    tr = np.maximum(high - low, 
                   np.maximum(np.abs(high - close.shift()), 
                             np.abs(low - close.shift())))
    return tr.rolling(window).mean()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Breakout model with customizable parameters
def breakout_signal(data, atr_multiplier=1.5, volume_multiplier=1.5, rsi_min=40, rsi_max=70):
    if len(data) < 21:  # Need at least 21 days for calculations
        return None
    
    # Calculate indicators
    data['20_high'] = data['High'].rolling(20).max()
    data['20_low'] = data['Low'].rolling(20).min()
    data['20_vol_avg'] = data['Volume'].rolling(20).mean()
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    data['RSI'] = calculate_rsi(data['Close'])
    
    latest = data.iloc[-1]
    
    # Long breakout condition with customizable parameters
    long_condition = (
        (latest['Close'] > latest['20_high'] + atr_multiplier * latest['ATR']) and
        (latest['Volume'] > volume_multiplier * latest['20_vol_avg']) and
        (rsi_min < latest['RSI'] < rsi_max)
    )
    
    # Short breakout condition with customizable parameters
    short_condition = (
        (latest['Close'] < latest['20_low'] - atr_multiplier * latest['ATR']) and
        (latest['Volume'] > volume_multiplier * latest['20_vol_avg']) and
        (100-rsi_max < latest['RSI'] < 100-rsi_min)
    )
    
    if long_condition:
        return {
            'signal': 'BUY',
            'breakout_level': latest['20_high'],
            'current_price': latest['Close'],
            'atr': latest['ATR'],
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume'] / latest['20_vol_avg']
        }
    elif short_condition:
        return {
            'signal': 'SELL',
            'breakout_level': latest['20_low'],
            'current_price': latest['Close'],
            'atr': latest['ATR'],
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume'] / latest['20_vol_avg']
        }
    return None

def main():
    st.title("🚀 Bulk Breakout Scanner (Excel Input)")
    st.markdown("""
    Scan multiple stocks from Excel file for breakout signals with customizable parameters.
    Excel file must contain a 'Symbol' column.
    """)

    # Parameter customization
    st.subheader("🔧 Customize Scan Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        atr_multiplier = st.slider("ATR Multiplier", 1.0, 3.0, 1.5, 0.1)
        days_back = st.slider("Analysis Period (Days)", 30, 365, 90)
    with col2:
        volume_multiplier = st.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)
        min_volume = st.number_input("Minimum Avg Volume (M)", value=5)
    with col3:
        rsi_min = st.slider("RSI Minimum (Long)", 30, 60, 40)
        rsi_max = st.slider("RSI Maximum (Long)", 50, 80, 70)

    # Load stock list from Excel
    try:
        stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
    except FileNotFoundError:
        st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
        return

    selected_sheet = st.selectbox("Select Sheet", stock_sheets)
    
    if st.button("🔍 Scan Stocks"):
        with st.spinner(f"Loading {selected_sheet} sheet..."):
            try:
                symbols_df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
                if 'Symbol' not in symbols_df.columns:
                    st.error("Selected sheet must contain 'Symbol' column")
                    return
                
                symbols = symbols_df['Symbol'].astype(str).tolist()
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, symbol in enumerate(symbols):
                    status_text.text(f"Scanning {symbol} ({i+1}/{len(symbols)})...")
                    data = get_stock_data(symbol, f"{days_back}d")
                    
                    if data is not None and not data.empty:
                        # Filter for liquid stocks
                        if data['Volume'].mean() > min_volume * 1e6:
                            signal = breakout_signal(
                                data, 
                                atr_multiplier=atr_multiplier,
                                volume_multiplier=volume_multiplier,
                                rsi_min=rsi_min,
                                rsi_max=rsi_max
                            )
                            if signal:
                                results.append({
                                    'Symbol': symbol,
                                    'Signal': signal['signal'],
                                    'Price': signal['current_price'],
                                    'Breakout Level': signal['breakout_level'],
                                    'ATR': signal['atr'],
                                    'RSI': signal['rsi'],
                                    'Volume Ratio': f"{signal['volume_ratio']:.1f}x",
                                    '20D Avg Vol (M)': f"{data['Volume'].rolling(20).mean().iloc[-1]/1e6:.1f}"
                                })
                    progress_bar.progress((i + 1) / len(symbols))
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.subheader(f"📊 Results ({len(results_df)} Signals Found)")
                    
                    # Display summary stats
                    buy_count = len(results_df[results_df['Signal'] == 'BUY'])
                    sell_count = len(results_df[results_df['Signal'] == 'SELL'])
                    st.markdown(f"""
                    **Summary:**
                    - BUY Signals: {buy_count}
                    - SELL Signals: {sell_count}
                    - Average RSI: {results_df['RSI'].mean():.1f}
                    """)
                    
                    # Show results table
                    st.dataframe(
                        results_df.style.background_gradient(
                            subset=['RSI'], 
                            cmap='RdYlGn', 
                            vmin=rsi_min, 
                            vmax=rsi_max
                        ),
                        height=500,
                        use_container_width=True
                    )
                    
                    # Download results
                    st.download_button(
                        label="📥 Download Results",
                        data=results_df.to_csv(index=False),
                        file_name=f"breakout_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No breakout signals found with current parameters")
            except Exception as e:
                st.error(f"Error scanning stocks: {str(e)}")

if __name__ == "__main__":
    main()
