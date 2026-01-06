import streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import ta
import time
import requests
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Binance RSI Auto-Trader",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- CLEANER APP STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} 
            footer {visibility: hidden;}
            .block-container {padding-top: 1rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Bot Settings")

# 1. Connection
st.sidebar.subheader("🔌 Connection")
MARKET_TYPE = st.sidebar.radio("Market Type", ["Spot", "Futures"], horizontal=True)
USE_US_BINANCE = st.sidebar.checkbox("Use Binance.US", value=False)

with st.sidebar.expander("🔐 API Keys (Optional)", expanded=True):
    st.info("Keys sirf data limits barhane ke liye hain. Real Trading disabled hai.")
    USER_API_KEY = st.text_input("API Key", type="password")
    USER_API_SECRET = st.text_input("API Secret", type="password")

PROXY_URL = st.sidebar.text_input("Proxy URL (Optional)", placeholder="http://user:pass@ip:port")

st.sidebar.divider()

# 2. AUTO TRADING SECTION (SIMULATION ONLY)
st.sidebar.subheader("🤖 Auto Trading (Simulation)")
ENABLE_AUTOTRADE = st.sidebar.checkbox("Enable Paper Trading (Test Mode)", value=False)

if ENABLE_AUTOTRADE:
    col_t1, col_t2 = st.sidebar.columns(2)
    TRADE_AMOUNT_USDT = col_t1.number_input("Simulated Amount ($)", min_value=10.0, value=15.0)
    MAX_OPEN_TRADES = col_t2.number_input("Max Trades per Scan", min_value=1, value=3)
    st.sidebar.info(f"ℹ️ Sirf Testing hogi. Asli paisay nahi lagenge.")

st.sidebar.divider()

# 3. Strategy
st.sidebar.subheader("🔍 Strategy")
SEARCH_MODE = st.sidebar.radio(
    "Select Strategy:",
    ["📊 All-in-One Report", "Crossover Alert", "RSI Range", "Sustained Trend (Days)"]
)

# Dynamic Inputs
if SEARCH_MODE == "Crossover Alert":
    st.sidebar.info("🔔 Buy when RSI crosses BELOW level.")
    RSI_ALERT_LEVEL = st.sidebar.number_input("RSI Cross Level", 1, 100, 30)
    
elif SEARCH_MODE == "RSI Range":
    st.sidebar.info("↔️ Trade coins inside a range.")
    col1, col2 = st.sidebar.columns(2)
    MIN_RSI = col1.number_input("Min RSI", 1, 100, 70)
    MAX_RSI = col2.number_input("Max RSI", 1, 100, 90)

elif SEARCH_MODE == "Sustained Trend (Days)":
    st.sidebar.info("📅 Find coins staying Above/Below RSI.")
    SUSTAINED_DAYS = st.sidebar.number_input("Duration (Days)", 1, 30, 3)
    TREND_TYPE = st.sidebar.selectbox("Condition", ["Always ABOVE", "Always BELOW"])
    TREND_RSI_LEVEL = st.sidebar.number_input("RSI Threshold", 1, 100, 70)

elif SEARCH_MODE == "📊 All-in-One Report":
    st.sidebar.info("📑 Auto-Trade primarily on OVERSOLD signals.")
    col1, col2 = st.sidebar.columns(2)
    OVERBOUGHT_VAL = col1.number_input("Overbought (>)", 50, 100, 70)
    OVERSOLD_VAL = col2.number_input("Oversold (<)", 1, 50, 30)
    REPORT_DAYS = st.sidebar.number_input("Sustained Trend Days", 1, 10, 3)

# Timeframe
TIMEFRAME_OPTIONS = {
    "15 Minutes": Client.KLINE_INTERVAL_15MINUTE,
    "1 Hour": Client.KLINE_INTERVAL_1HOUR,
    "4 Hours": Client.KLINE_INTERVAL_4HOUR,
    "12 Hours": Client.KLINE_INTERVAL_12HOUR,
    "1 Day": Client.KLINE_INTERVAL_1DAY
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=2)
TIMEFRAME = TIMEFRAME_OPTIONS[selected_tf_label]

# --- HELPER FUNCTIONS ---
@st.cache_resource
def init_client(use_us, proxy, api_key, api_secret):
    args = {}
    if use_us: args['tld'] = 'us'
    if proxy: args['requests_params'] = {'proxies': {'http': proxy, 'https': proxy}}
    
    if api_key and api_secret:
        args['api_key'] = api_key
        args['api_secret'] = api_secret
        
    return Client(**args)

def get_tf_in_minutes(tf_str):
    mapping = {'15m':15, '1h':60, '4h':240, '12h':720, '1d':1440}
    return mapping.get(tf_str, 240)

def place_order_simulation(symbol, side, amount_usdt, price):
    """
    Sirf Simulation Order generate karta hai.
    Real trading logic completely removed.
    """
    qty = amount_usdt / price
    return {
        "symbol": symbol,
        "orderId": f"SIM-{int(time.time())}",
        "status": "FILLED",
        "type": "MARKET (PAPER)",
        "side": side,
        "executedQty": f"{qty:.5f}",
        "cummulativeQuoteQty": f"{amount_usdt:.2f}"
    }

def get_data_with_rsi(client, symbol, tf, market_type, limit=500):
    try:
        if market_type == "Spot":
            klines = client.get_klines(symbol=symbol, interval=tf, limit=limit)
        else:
            klines = client.futures_klines(symbol=symbol, interval=tf, limit=limit)
            
        df = pd.DataFrame(klines, columns=[
            'time','open','high','low','close','volume','close_time','qav','num_trades','tbv','tqv','ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        return df
    except:
        return None

# --- MAIN LOGIC ---
st.title(f"🤖 Binance RSI Scanner ({MARKET_TYPE})")

if USE_US_BINANCE: st.warning("🇺🇸 Using Binance.US")
if PROXY_URL: st.info("🌐 Using Proxy")

# Trading Status Bar
if ENABLE_AUTOTRADE:
    st.info(f"🔵 **SIMULATION MODE ACTIVE:** Bot will 'pretend' to buy coins. (Max {MAX_OPEN_TRADES} trades)")
else:
    st.markdown("**Status:** 🟢 Scanner Only")

# Button
btn_label = "🔄 Scan & Simulate" if ENABLE_AUTOTRADE else "🔄 Start Scanner"

if st.button(btn_label, type="primary"):
    
    if USE_US_BINANCE and MARKET_TYPE == "Futures":
        st.error("❌ Binance.US does not support Futures."); st.stop()

    try:
        client = init_client(USE_US_BINANCE, PROXY_URL, USER_API_KEY, USER_API_SECRET)
    except Exception as e:
        st.error(f"❌ Connection/Auth Error: {e}"); st.stop()

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text(f"Fetching {MARKET_TYPE} pairs...")
        
        # --- 1. Fetch Symbols & Funding Rates (If Futures) ---
        funding_map = {}
        
        if MARKET_TYPE == "Spot": 
            exchange_info = client.get_exchange_info()
        else: 
            # Futures: Exchange Info + Funding Rates fetch karein
            exchange_info = client.futures_exchange_info()
            try:
                status_text.text("Fetching Funding Rates...")
                mark_prices = client.futures_mark_price()
                # Create a dictionary for fast lookup: {'BTCUSDT': 0.0001, ...}
                for item in mark_prices:
                    funding_map[item['symbol']] = float(item['lastFundingRate'])
            except Exception as e:
                st.warning(f"Could not fetch funding rates: {e}")

        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']
        
        # --- 2. Candle Setup ---
        candles_needed = 100 
        days_to_check = REPORT_DAYS if SEARCH_MODE == "📊 All-in-One Report" else 0
        if SEARCH_MODE == "Sustained Trend (Days)": days_to_check = SUSTAINED_DAYS

        if days_to_check > 0:
            tf_minutes = get_tf_in_minutes(TIMEFRAME)
            candles_needed = int((days_to_check * 1440) / tf_minutes) + 20

        alerts = []
        trades_executed = 0
        trade_logs = []

        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            progress_bar.progress((i + 1) / total_symbols)
            status_text.text(f"Scanning {i+1}/{total_symbols}: {symbol}...")
            
            df = get_data_with_rsi(client, symbol, TIMEFRAME, MARKET_TYPE, limit=max(100, candles_needed))
            
            if df is not None and len(df) > 20:
                curr_rsi = df['rsi'].iloc[-1]
                prev_rsi = df['rsi'].iloc[-2]
                curr_price = df['close'].iloc[-1]
                
                # Funding Rate Logic
                funding_rate_display = "N/A"
                if MARKET_TYPE == "Futures" and symbol in funding_map:
                    fr = funding_map[symbol]
                    # Display as percentage (e.g., 0.0100%)
                    funding_rate_display = f"{fr * 100:.4f}%"

                match_found = False
                status_msg = ""
                signal_type = "NEUTRAL"
                
                # --- LOGIC SELECTION ---
                if SEARCH_MODE == "📊 All-in-One Report":
                    # Priority for Trading: OVERSOLD (Buy)
                    if curr_rsi <= OVERSOLD_VAL:
                         match_found = True; status_msg = "Oversold Zone"; signal_type = "BUY"
                    elif prev_rsi > OVERSOLD_VAL and curr_rsi <= OVERSOLD_VAL:
                         match_found = True; status_msg = "📉 BREAKDOWN (Buy Dip)"; signal_type = "BUY"
                    # Just for reporting
                    elif curr_rsi >= OVERBOUGHT_VAL:
                         match_found = True; status_msg = "Overbought Zone"; signal_type = "SELL"
                
                elif SEARCH_MODE == "Crossover Alert":
                    if RSI_ALERT_LEVEL < 50:
                        if prev_rsi > RSI_ALERT_LEVEL and curr_rsi <= RSI_ALERT_LEVEL:
                            match_found = True; status_msg = "CROSS BELOW"; signal_type = "BUY"
                
                # --- AUTO TRADE EXECUTION (SIMULATION) ---
                trade_result = None
                if match_found and ENABLE_AUTOTRADE and trades_executed < MAX_OPEN_TRADES:
                    if signal_type == "BUY":
                        st.toast(f"⚡ Simulating BUY {symbol}...")
                        trade_result = place_order_simulation(symbol, "BUY", TRADE_AMOUNT_USDT, curr_price)
                        status_msg += " | ✅ SIMULATED"
                        trades_executed += 1
                        trade_logs.append(trade_result)

                if match_found:
                    alerts.append({
                        "Symbol": symbol,
                        "Price": curr_price,
                        "RSI": round(curr_rsi, 2),
                        "Signal": signal_type,
                        "Funding Rate": funding_rate_display, # New Column
                        "Status": status_msg
                    })
                    
            if ENABLE_AUTOTRADE and trades_executed >= MAX_OPEN_TRADES:
                status_text.warning(f"🛑 Max simulation trades ({MAX_OPEN_TRADES}) reached.")
                break

        progress_bar.empty()
        status_text.success(f"✅ Scan Complete!")
        
        # Display Trades
        if trade_logs:
            st.subheader(f"📜 Simulation Log")
            st.json(trade_logs)

        # Display Scan Results
        if alerts:
            st.subheader("📊 Scan Results")
            df_results = pd.DataFrame(alerts)
            st.dataframe(df_results, use_container_width=True)
        else:
            st.warning("No signals found.")
            
    except Exception as e:
        st.error(f"Error: {e}")
