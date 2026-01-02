import streamlit as st
from binance.client import Client
import pandas as pd
import ta
import time

# --- PAGE CONFIGURATION (Browser Tab Title etc) ---
st.set_page_config(
    page_title="Binance RSI Scanner",
    page_icon="🚀",
    layout="wide"
)

# --- SIDEBAR SETTINGS (Control Panel) ---
st.sidebar.title("⚙️ Scanner Settings")

# Market Type Selection (Spot vs Futures)
MARKET_TYPE = st.sidebar.radio("Market Type", ["Spot", "Futures"])

# User se inputs lena
RSI_OVERBOUGHT = st.sidebar.number_input("RSI Alert Level", min_value=50, max_value=100, value=82)

TIMEFRAME_OPTIONS = {
    "4 Hours": Client.KLINE_INTERVAL_4HOUR,
    "1 Hour": Client.KLINE_INTERVAL_1HOUR,
    "15 Minutes": Client.KLINE_INTERVAL_15MINUTE,
    "1 Day": Client.KLINE_INTERVAL_1DAY
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()))
TIMEFRAME = TIMEFRAME_OPTIONS[selected_tf_label]

st.sidebar.info(f"Click 'Start Scan' to check all {MARKET_TYPE} USDT pairs.")

# --- MAIN FUNCTIONS ---
@st.cache_resource
def init_client():
    return Client()

def get_rsi(client, symbol, tf, market_type):
    try:
        # Market type ke hisaab se data fetch karna
        if market_type == "Spot":
            klines = client.get_klines(symbol=symbol, interval=tf, limit=100)
        else:
            klines = client.futures_klines(symbol=symbol, interval=tf, limit=100)
            
        df = pd.DataFrame(klines, columns=[
            'time','open','high','low','close','volume',
            'close_time','qav','num_trades',
            'taker_base_vol','taker_quote_vol','ignore'
        ])
        df['close'] = df['close'].astype(float)
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        return rsi.iloc[-2], rsi.iloc[-1] # Prev, Curr
    except Exception as e:
        return None, None

# --- UI LAYOUT ---
st.title(f"🚀 Crypto RSI Alert Dashboard ({MARKET_TYPE})")
st.markdown(f"**Scanning for coins crossing RSI {RSI_OVERBOUGHT} on {selected_tf_label} timeframe in {MARKET_TYPE} Market.**")

# Scan Button
if st.button("🔄 Start Market Scan", type="primary"):
    client = init_client()
    
    # UI Elements for progress
    status_text = st.empty()
    progress_bar = st.progress(0)
    results_container = st.container()
    
    try:
        status_text.text(f"Fetching {MARKET_TYPE} pairs from Binance...")
        
        # Market type ke hisaab se symbols fetch karna
        if MARKET_TYPE == "Spot":
            exchange_info = client.get_exchange_info()
        else:
            exchange_info = client.futures_exchange_info()
            
        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
        ]
        
        # Testing ke liye agar chaho to kam symbols kar lo:
        # symbols = symbols[:20] 
        
        alerts = []
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            # Update Progress Bar
            progress = (i + 1) / total_symbols
            progress_bar.progress(progress)
            status_text.text(f"Scanning {i+1}/{total_symbols}: {symbol}...")
            
            # Get RSI (Market type pass kar rahe hain)
            prev_rsi, curr_rsi = get_rsi(client, symbol, TIMEFRAME, MARKET_TYPE)
            
            if prev_rsi is not None:
                # Check Alert Condition
                if prev_rsi < RSI_OVERBOUGHT and curr_rsi >= RSI_OVERBOUGHT:
                    alerts.append({
                        "Symbol": symbol,
                        "Previous RSI": round(prev_rsi, 2),
                        "Current RSI": round(curr_rsi, 2),
                        "Status": "CROSS OVER 🔥"
                    })
        
        progress_bar.empty()
        status_text.success(f"✅ Scan Complete! Scanned {total_symbols} {MARKET_TYPE} coins.")
        
        # Display Results
        if alerts:
            st.error(f"🚨 Found {len(alerts)} Coins with High RSI!")
            df_results = pd.DataFrame(alerts)
            st.dataframe(df_results, use_container_width=True)
        else:
            st.success("👍 No coins found crossing the RSI limit right now.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.write("Waiting for scan...")