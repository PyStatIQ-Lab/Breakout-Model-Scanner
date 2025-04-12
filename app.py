import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

# Breakout model
def breakout_signal(data):
    if len(data) < 21:  # Need at least 21 days for calculations
        return None
    
    # Calculate indicators
    data['20_high'] = data['High'].rolling(20).max()
    data['20_low'] = data['Low'].rolling(20).min()
    data['20_vol_avg'] = data['Volume'].rolling(20).mean()
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    data['RSI'] = calculate_rsi(data['Close'])
    
    latest = data.iloc[-1]
    prev_close = data['Close'].iloc[-2]
    
    # Long breakout condition
    long_condition = (
        (latest['Close'] > latest['20_high'] + 1.5 * latest['ATR']) and
        (latest['Volume'] > 1.5 * latest['20_vol_avg']) and
        (40 < latest['RSI'] < 70)
    )
    
    # Short breakout condition (inverse logic)
    short_condition = (
        (latest['Close'] < latest['20_low'] - 1.5 * latest['ATR']) and
        (latest['Volume'] > 1.5 * latest['20_vol_avg']) and
        (30 < latest['RSI'] < 60)
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

# Visualization
def plot_breakout(data, signal):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                           gridspec_kw={'height_ratios': [3, 1]})
    
    # Price chart
    data['Close'].plot(ax=ax1, label='Close Price', color='steelblue')
    
    if signal['signal'] == 'BUY':
        data['20_high'].plot(ax=ax1, label='20-day High', 
                            linestyle='--', color='darkred', alpha=0.7)
        ax1.axhline(y=signal['breakout_level'], color='green', 
                   linestyle=':', label='Breakout Level')
        ax1.scatter(data.index[-1], signal['current_price'], 
                   color='lime', s=100, label='Breakout Signal')
    else:
        data['20_low'].plot(ax=ax1, label='20-day Low', 
                           linestyle='--', color='darkgreen', alpha=0.7)
        ax1.axhline(y=signal['breakout_level'], color='red', 
                   linestyle=':', label='Breakdown Level')
        ax1.scatter(data.index[-1], signal['current_price'], 
                   color='red', s=100, label='Breakdown Signal')
    
    ax1.set_title(f"Breakout Pattern - {signal['signal']} Signal")
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Volume chart
    data['Volume'].plot(ax=ax2, label='Volume', color='royalblue', alpha=0.5)
    data['20_vol_avg'].plot(ax=ax2, label='20-day Avg Volume', 
                           color='navy', linestyle='--')
    ax2.axhline(y=data['Volume'].iloc[-1], color='green' if signal['signal'] == 'BUY' else 'red',
               linestyle=':', alpha=0.5)
    ax2.set_ylabel('Volume')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("🚀 Breakout Model Scanner")
    st.markdown("""
    Identifies stocks breaking key resistance/support levels with volume confirmation:
    - **Price** > 20-day high + 1.5× ATR (for long)
    - **Volume** > 1.5× 20-day average
    - **RSI(14)** between 40-70 (avoid overbought)
    """)
    
    # Input section
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Enter Stock Symbol (e.g. AAPL):", "AAPL").upper()
    with col2:
        days_back = st.slider("Analysis Period (Days):", 30, 365, 90)
    
    analyze_btn = st.button("Analyze Breakout")
    
    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            data = get_stock_data(symbol, f"{days_back}d")
            
            if data is None or data.empty:
                st.error("Could not fetch data for this symbol")
                return
            
            signal = breakout_signal(data)
            
            if signal:
                st.success(f"**{signal['signal']} Signal Detected**")
                
                # Display metrics
                cols = st.columns(4)
                cols[0].metric("Current Price", f"${signal['current_price']:.2f}")
                cols[1].metric("Breakout Level", f"${signal['breakout_level']:.2f}")
                cols[2].metric("ATR", f"${signal['atr']:.2f}")
                cols[3].metric("RSI", f"{signal['rsi']:.1f}")
                
                # Show chart
                st.pyplot(plot_breakout(data, signal))
                
                # Trading plan
                st.subheader("📝 Trading Plan")
                current_price = signal['current_price']
                stop_loss = signal['breakout_level'] - signal['atr'] if signal['signal'] == 'BUY' else signal['breakout_level'] + signal['atr']
                target = current_price + (3 * signal['atr']) if signal['signal'] == 'BUY' else current_price - (3 * signal['atr'])
                
                if signal['signal'] == 'BUY':
                    stop_loss_pct = ((current_price - stop_loss)/current_price)*100
                    target_pct = ((target - current_price)/current_price)*100
                    st.markdown(f"""
                    - **Entry**: ${current_price:.2f}
                    - **Stop Loss**: ${stop_loss:.2f} ({stop_loss_pct:.1f}% below)
                    - **Target**: ${target:.2f} ({target_pct:.1f}% above)
                    - **Risk-Reward**: 1:3
                    """)
                else:
                    stop_loss_pct = ((stop_loss - current_price)/current_price)*100
                    target_pct = ((current_price - target)/current_price)*100
                    st.markdown(f"""
                    - **Entry**: ${current_price:.2f}
                    - **Stop Loss**: ${stop_loss:.2f} ({stop_loss_pct:.1f}% above)
                    - **Target**: ${target:.2f} ({target_pct:.1f}% below)
                    - **Risk-Reward**: 1:3
                    """)
            else:
                st.warning("No breakout signal detected")
                
                # Show why no signal
                latest = data.iloc[-1]
                st.info("Current Status:")
                cols = st.columns(3)
                cols[0].metric("Price vs 20-high", 
                              f"${latest['Close']:.2f} / ${latest['20_high']:.2f}", 
                              f"{latest['Close']-latest['20_high']:.2f}")
                cols[1].metric("Volume vs Avg", 
                              f"{latest['Volume']/1e6:.1f}M / {latest['20_vol_avg']/1e6:.1f}M", 
                              f"{latest['Volume']/latest['20_vol_avg']:.1f}x")
                cols[2].metric("RSI(14)", f"{latest['RSI']:.1f}", 
                              "In Range" if 40<latest['RSI']<70 else "Out of Range")
    
    # Bulk scanner section
    st.divider()
    st.subheader("📈 Bulk Scan Multiple Stocks")
    
    # Load stock list from Excel
    try:
        stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
    except FileNotFoundError:
        st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
        return
    
    scan_type = st.radio("Select Scan Method:", 
                        ["Use Excel Sheet", "Upload Custom CSV"])
    
    if scan_type == "Use Excel Sheet":
        selected_sheet = st.selectbox("Select Sheet", stock_sheets)
        min_volume = st.number_input("Minimum Average Volume (Millions)", value=5)
        
        if st.button("Scan Excel List"):
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
                                signal = breakout_signal(data)
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
                        st.dataframe(
                            results_df.style.background_gradient(
                                subset=['RSI'], 
                                cmap='RdYlGn', 
                                vmin=40, 
                                vmax=70
                            ),
                            height=500
                        )
                        
                        # Download results
                        st.download_button(
                            label="Download Breakout Signals",
                            data=results_df.to_csv(index=False),
                            file_name=f"breakout_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No breakout signals found in the scanned stocks")
                except Exception as e:
                    st.error(f"Error scanning stocks: {str(e)}")
    
    else:  # CSV upload option
        uploaded_file = st.file_uploader("Upload CSV with 'Symbol' column", type="csv")
        min_volume = st.number_input("Minimum Average Volume (Millions)", value=5)
        
        if uploaded_file and st.button("Scan Custom List"):
            symbols_df = pd.read_csv(uploaded_file)
            if 'Symbol' not in symbols_df.columns:
                st.error("Uploaded file must contain 'Symbol' column")
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
                        signal = breakout_signal(data)
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
                st.dataframe(
                    results_df.style.background_gradient(
                        subset=['RSI'], 
                        cmap='RdYlGn', 
                        vmin=40, 
                        vmax=70
                    ),
                    height=500
                )
                
                # Download results
                st.download_button(
                    label="Download Breakout Signals",
                    data=results_df.to_csv(index=False),
                    file_name=f"breakout_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No breakout signals found in the scanned stocks")

if __name__ == "__main__":
    main()
