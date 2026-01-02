import streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import ta
import time
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Binance RSI Pro Scanner",
    page_icon="📊",
    layout="wide"
)

# --- CLEANER APP STYLE (Hide Streamlit Defaults) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 1rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Scanner Settings")

# Market & Connection
MARKET_TYPE = st.sidebar.radio("Market Type", ["Spot", "Futures"], horizontal=True)
USE_US_BINANCE = st.sidebar.checkbox("Use Binance.US", value=False)
PROXY_URL = st.sidebar.text_input("Proxy URL (Optional)", placeholder="http://user:pass@ip:port")

st.sidebar.divider()

# --- SEARCH MODES ---
st.sidebar.subheader("🔍 Search Mode")
SEARCH_MODE = st.sidebar.radio(
    "Select Strategy:",
    ["📊 All-in-One Report", "Crossover Alert", "RSI Range", "Sustained Trend (Days)"]
)

# Dynamic Inputs based on Mode
if SEARCH_MODE == "Crossover Alert":
    st.sidebar.info("🔔 Alert when RSI crosses a specific level.")
    RSI_ALERT_LEVEL = st.sidebar.number_input("RSI Cross Level", 1, 100, 30)
    
elif SEARCH_MODE == "RSI Range":
    st.sidebar.info("↔️ Show coins inside a specific RSI Range.")
    col1, col2 = st.sidebar.columns(2)
    MIN_RSI = col1.number_input("Min RSI", 1, 100, 70)
    MAX_RSI = col2.number_input("Max RSI", 1, 100, 90)

elif SEARCH_MODE == "Sustained Trend (Days)":
    st.sidebar.info("📅 Find coins staying Above/Below RSI for days.")
    SUSTAINED_DAYS = st.sidebar.number_input("Duration (Days)", 1, 30, 3)
    TREND_TYPE = st.sidebar.selectbox("Condition", ["Always ABOVE", "Always BELOW"])
    TREND_RSI_LEVEL = st.sidebar.number_input("RSI Threshold", 1, 100, 70)

elif SEARCH_MODE == "📊 All-in-One Report":
    st.sidebar.info("📑 Classify Market into Signals, Overbought, and Oversold.")
    col1, col2 = st.sidebar.columns(2)
    OVERBOUGHT_VAL = col1.number_input("Overbought (>)", 50, 100, 70)
    OVERSOLD_VAL = col2.number_input("Oversold (<)", 1, 50, 30)
    REPORT_DAYS = st.sidebar.number_input("Sustained Trend Days", 1, 10, 3)

st.sidebar.divider()

# Timeframe Selection
TIMEFRAME_OPTIONS = {
    "15 Minutes": Client.KLINE_INTERVAL_15MINUTE,
    "1 Hour": Client.KLINE_INTERVAL_1HOUR,
    "4 Hours": Client.KLINE_INTERVAL_4HOUR,
    "12 Hours": Client.KLINE_INTERVAL_12HOUR,
    "1 Day": Client.KLINE_INTERVAL_1DAY,
    "3 Days": Client.KLINE_INTERVAL_3DAY,
    "1 Week": Client.KLINE_INTERVAL_1WEEK
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=2)
TIMEFRAME = TIMEFRAME_OPTIONS[selected_tf_label]

# --- HELPER FUNCTIONS ---
@st.cache_resource
def init_client(use_us, proxy):
    args = {}
    if use_us: args['tld'] = 'us'
    if proxy: args['requests_params'] = {'proxies': {'http': proxy, 'https': proxy}}
    return Client(**args)

def get_tf_in_minutes(tf_str):
    mapping = {'15m':15, '1h':60, '4h':240, '12h':720, '1d':1440, '3d':4320, '1w':10080}
    return mapping.get(tf_str, 240) # Default 4h

def format_volume(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

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
        df['volume'] = df['volume'].astype(float) # Ensure volume is float
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        return df
    except:
        return None

# --- MAIN LOGIC ---
st.title(f"🚀 Binance RSI Pro Scanner ({MARKET_TYPE})")

# Description Display
if SEARCH_MODE == "Crossover Alert":
    st.markdown(f"**Strategy:** Finding coins crossing **{RSI_ALERT_LEVEL}**.")
elif SEARCH_MODE == "RSI Range":
    st.markdown(f"**Strategy:** Finding coins with RSI between **{MIN_RSI}** and **{MAX_RSI}**.")
elif SEARCH_MODE == "📊 All-in-One Report":
    st.markdown(f"**Strategy:** Categorizing Signals, Overbought (> {OVERBOUGHT_VAL}) & Oversold (< {OVERSOLD_VAL}).")
else:
    cond = ">" if "ABOVE" in TREND_TYPE else "<"
    st.markdown(f"**Strategy:** Coins where RSI has been **{cond} {TREND_RSI_LEVEL}** for the last **{SUSTAINED_DAYS} Days**.")

if st.button("🔄 Start Market Scan", type="primary"):
    
    if USE_US_BINANCE and MARKET_TYPE == "Futures":
        st.error("❌ Binance.US does not support Futures.")
        st.stop()

    # --- Robust Client Initialization ---
    try:
        client = init_client(USE_US_BINANCE, PROXY_URL)
        # Quick check if connection works
        client.get_system_status() 
    except BinanceAPIException as e:
        st.error("🚨 **Connection Error:** Binance blocked the connection.")
        st.error(f"**Details:** {e}")
        st.warning("👉 **Fix:** If you are on Streamlit Cloud (US Server), you MUST provide a valid **Proxy URL** in the sidebar to access Binance Global.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Failed to connect:** {e}")
        st.info("Check your Proxy URL format. It should be: `http://user:pass@ip:port`")
        st.stop()
    # ------------------------------------

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text(f"Fetching {MARKET_TYPE} pairs...")
        if MARKET_TYPE == "Spot":
            exchange_info = client.get_exchange_info()
        else:
            exchange_info = client.futures_exchange_info()
            
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']
        
        # Determine candles needed
        candles_needed = 100  # Default base
        days_to_check = 0
        
        if SEARCH_MODE == "Sustained Trend (Days)":
            days_to_check = SUSTAINED_DAYS
        elif SEARCH_MODE == "📊 All-in-One Report":
            days_to_check = REPORT_DAYS

        if days_to_check > 0:
            tf_minutes = get_tf_in_minutes(TIMEFRAME)
            total_minutes = days_to_check * 24 * 60
            candles_needed = int(total_minutes / tf_minutes) + 20 # Buffer
            if candles_needed > 999: candles_needed = 999 

        alerts = []
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            progress_bar.progress((i + 1) / total_symbols)
            status_text.text(f"Scanning {i+1}/{total_symbols}: {symbol}...")
            
            # Fetch Data Frame
            df = get_data_with_rsi(client, symbol, TIMEFRAME, MARKET_TYPE, limit=max(100, candles_needed))
            
            if df is not None and len(df) > 20:
                rsi_series = df['rsi']
                curr_rsi = rsi_series.iloc[-1]
                prev_rsi = rsi_series.iloc[-2]
                curr_price = df['close'].iloc[-1]
                curr_vol = df['volume'].iloc[-1] # Volume
                
                match_found = False
                status_msg = ""
                group_tag = "Normal" # Used for separating tables

                # --- LOGIC 1: ALL-IN-ONE REPORT ---
                if SEARCH_MODE == "📊 All-in-One Report":
                    # Sustained Data slice
                    relevant_rsi = rsi_series.iloc[-(candles_needed-14):] if candles_needed > 20 else rsi_series.tail(10)
                    
                    if not relevant_rsi.empty:
                        # 1. Check Fresh Crossovers (Priority 1)
                        if prev_rsi < OVERBOUGHT_VAL and curr_rsi >= OVERBOUGHT_VAL:
                            match_found = True; status_msg = "🚀 BREAKOUT (Pump)"; group_tag = "Signal"
                        elif prev_rsi > OVERSOLD_VAL and curr_rsi <= OVERSOLD_VAL:
                            match_found = True; status_msg = "📉 BREAKDOWN (Dump)"; group_tag = "Signal"
                        
                        # 2. Check Sustained Trends (Priority 2)
                        elif relevant_rsi.min() > OVERBOUGHT_VAL:
                            match_found = True; status_msg = f"🔥 Sustained Bullish ({REPORT_DAYS}d)"; group_tag = "Overbought"
                        elif relevant_rsi.max() < OVERSOLD_VAL:
                            match_found = True; status_msg = f"❄️ Sustained Bearish ({REPORT_DAYS}d)"; group_tag = "Oversold"

                        # 3. Check Zones (Priority 3)
                        elif curr_rsi >= OVERBOUGHT_VAL:
                             match_found = True; status_msg = "Overbought Zone"; group_tag = "Overbought"
                        elif curr_rsi <= OVERSOLD_VAL:
                             match_found = True; status_msg = "Oversold Zone"; group_tag = "Oversold"

                # --- LOGIC 2: CROSSOVER ---
                elif SEARCH_MODE == "Crossover Alert":
                    if RSI_ALERT_LEVEL < 50: # Oversold Logic
                        if prev_rsi > RSI_ALERT_LEVEL and curr_rsi <= RSI_ALERT_LEVEL:
                            match_found = True; status_msg = "CROSS BELOW 📉"; group_tag = "Alert"
                    else: # Overbought Logic
                        if prev_rsi < RSI_ALERT_LEVEL and curr_rsi >= RSI_ALERT_LEVEL:
                            match_found = True; status_msg = "CROSS ABOVE 🚀"; group_tag = "Alert"

                # --- LOGIC 3: RANGE ---
                elif SEARCH_MODE == "RSI Range":
                    if MIN_RSI <= curr_rsi <= MAX_RSI:
                        match_found = True; status_msg = f"IN RANGE ({MIN_RSI}-{MAX_RSI})"; group_tag = "Range"

                # --- LOGIC 4: SUSTAINED TREND ---
                elif SEARCH_MODE == "Sustained Trend (Days)":
                    relevant_rsi = rsi_series.iloc[-(candles_needed-14):]
                    if not relevant_rsi.empty:
                        if "ABOVE" in TREND_TYPE:
                            if relevant_rsi.min() > TREND_RSI_LEVEL:
                                match_found = True; status_msg = f"STRONG BULLISH 🔥"; group_tag = "Trend"
                        else:
                            if relevant_rsi.max() < TREND_RSI_LEVEL:
                                match_found = True; status_msg = f"STRONG BEARISH ❄️"; group_tag = "Trend"

                if match_found:
                    # --- SMART SUGGESTION LOGIC (FIXED) ---
                    # Default thresholds (Safe Defaults)
                    s_ob = 70
                    s_os = 30
                    
                    # Update thresholds if in All-in-One mode
                    if SEARCH_MODE == "📊 All-in-One Report":
                        s_ob = OVERBOUGHT_VAL
                        s_os = OVERSOLD_VAL

                    suggestion_msg = "Wait / Monitor ✋"
                    
                    if curr_rsi >= s_ob:
                        suggestion_msg = f"Consider SHORT 📉 (Overbought)"
                    elif curr_rsi <= s_os:
                        suggestion_msg = f"Consider LONG 📈 (Oversold)"
                    elif 50 <= curr_rsi < s_ob:
                        suggestion_msg = "Uptrend Momentum 🐂 (Wait)"
                    elif s_os < curr_rsi < 50:
                        suggestion_msg = "Downtrend Momentum 🐻 (Wait)"

                    alerts.append({
                        "Symbol": symbol,
                        "Price": curr_price,
                        "Suggestion": suggestion_msg, 
                        "Prev RSI": round(prev_rsi, 2),
                        "Current RSI": round(curr_rsi, 2),
                        "Volume": format_volume(curr_vol),
                        "Status": status_msg,
                        "Group": group_tag
                    })

        progress_bar.empty()
        status_text.success(f"✅ Scan Complete! Scanned {total_symbols} coins.")
        
        if alerts:
            # --- DISCLAIMER ---
            st.warning("⚠️ **CAUTION:** The 'Suggestion' is based purely on RSI levels. **Neutral Zone (30-70)** usually indicates trend continuation, not entry. **This is NOT financial advice.**")

            df_res = pd.DataFrame(alerts)
            
            # --- DISPLAY LOGIC (SEPARATE BOXES) ---
            if SEARCH_MODE == "📊 All-in-One Report":
                
                # 1. FRESH SIGNALS
                df_signals = df_res[df_res['Group'] == 'Signal']
                if not df_signals.empty:
                    st.subheader("⚡ Fresh Signals (Breakouts/Breakdowns)")
                    st.dataframe(df_signals.drop(columns=['Group']), use_container_width=True)
                
                # 2. OVERSOLD BOX
                df_oversold = df_res[df_res['Group'] == 'Oversold']
                if not df_oversold.empty:
                    st.subheader(f"🟢 Oversold / Buying Zones (RSI < {OVERSOLD_VAL})")
                    st.dataframe(df_oversold.drop(columns=['Group']).sort_values("Current RSI"), use_container_width=True)

                # 3. OVERBOUGHT BOX
                df_overbought = df_res[df_res['Group'] == 'Overbought']
                if not df_overbought.empty:
                    st.subheader(f"🔴 Overbought / Selling Zones (RSI > {OVERBOUGHT_VAL})")
                    st.dataframe(df_overbought.drop(columns=['Group']).sort_values("Current RSI", ascending=False), use_container_width=True)
                
                if df_signals.empty and df_oversold.empty and df_overbought.empty:
                     st.warning("No significant market movements found.")

            # --- STANDARD DISPLAY FOR OTHER MODES ---
            else:
                st.success(f"🎯 Found {len(alerts)} Matches!")
                st.dataframe(df_res.drop(columns=['Group']), use_container_width=True)

        else:
            st.warning("No coins matched your criteria.")
            
    except Exception as e:
        st.error(f"Error: {e}")
