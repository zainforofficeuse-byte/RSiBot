import streamlit as st
from binance.client import Client
import pandas as pd
import ta
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Binance RSI Scanner",
    page_icon="🚀",
    layout="wide"
)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Scanner Settings")

# Market Type Selection
MARKET_TYPE = st.sidebar.radio("Market Type", ["Spot", "Futures"])

# Region Selection (Fix for USA/Streamlit Cloud)
USE_US_BINANCE = st.sidebar.checkbox("Use Binance.US (Check if on Cloud)", value=False, help="Enable this if you are running on Streamlit Cloud or in the USA to avoid API restrictions.")

# RSI Level Input
RSI_OVERBOUGHT = st.sidebar.number_input("RSI Alert Level", min_value=50, max_value=100, value=82)

# Timeframe Selection
TIMEFRAME_OPTIONS = {
    "4 Hours": Client.KLINE_INTERVAL_4HOUR,
    "1 Hour": Client.KLINE_INTERVAL_1HOUR,
    "15 Minutes": Client.KLINE_INTERVAL_15MINUTE,
    "1 Day": Client.KLINE_INTERVAL_1DAY
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()))
TIMEFRAME = TIMEFRAME_OPTIONS[selected_tf_label]

st.sidebar.info(f"Click 'Start Scan' to check all {MARKET_TYPE} USDT pairs.")

# --- FUNCTIONS ---
@st.cache_resource
def init_client(use_us):
    # Agar US checkbox tick hai to US server use karega
    if use_us:
        return Client(tld='us')
    return Client()

def get_rsi(client, symbol, tf, market_type):
    try:
        # Data fetching logic based on market type
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
        return rsi.iloc[-2], rsi.iloc[-1] # Previous, Current
    except Exception:
        return None, None

# --- MAIN UI ---
st.title(f"🚀 Crypto RSI Alert Dashboard ({MARKET_TYPE})")
st.markdown(f"**Scanning for coins crossing RSI {RSI_OVERBOUGHT} on {selected_tf_label} timeframe.**")

if USE_US_BINANCE:
    st.warning("🇺🇸 Using Binance.US servers (Limited pairs, but works on US Cloud).")

# Scan Button
if st.button("🔄 Start Market Scan", type="primary"):
    # Client initialize karte waqt US setting pass kar rahay hain
    client = init_client(USE_US_BINANCE)
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text(f"Fetching {MARKET_TYPE} pairs from Binance...")
        
        # Symbol fetching based on market type
        if MARKET_TYPE == "Spot":
            exchange_info = client.get_exchange_info()
        else:
            exchange_info = client.futures_exchange_info()
            
        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
        ]
        
        alerts = []
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            # Progress bar update
            progress = (i + 1) / total_symbols
            progress_bar.progress(progress)
            status_text.text(f"Scanning {i+1}/{total_symbols}: {symbol}...")
            
            prev_rsi, curr_rsi = get_rsi(client, symbol, TIMEFRAME, MARKET_TYPE)
            
            if prev_rsi is not None:
                if prev_rsi < RSI_OVERBOUGHT and curr_rsi >= RSI_OVERBOUGHT:
                    alerts.append({
                        "Symbol": symbol,
                        "Previous RSI": round(prev_rsi, 2),
                        "Current RSI": round(curr_rsi, 2),
                        "Status": "CROSS OVER 🔥"
                    })
        
        progress_bar.empty()
        status_text.success(f"✅ Scan Complete! Scanned {total_symbols} coins.")
        
        if alerts:
            st.error(f"🚨 Found {len(alerts)} Coins with High RSI!")
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.success("👍 No coins found crossing the RSI limit.")
            
    except Exception as e:
        st.error(f"Error: {e}. Try checking 'Use Binance.US' in the sidebar if you are on Streamlit Cloud.")

else:
    st.write("Waiting for scan...")
