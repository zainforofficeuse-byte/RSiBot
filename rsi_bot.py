import streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import ta
import time
import requests
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Binance RSI Pro Scanner 3.2",
    page_icon="🪤",
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

# --- SESSION STATE INITIALIZATION ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'scan_performed' not in st.session_state:
    st.session_state.scan_performed = False

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Scanner Settings 3.2")

# 1. Connection
st.sidebar.subheader("🔌 Connection")
MARKET_TYPE = st.sidebar.radio("Market Type", ["Spot", "Futures"], index=1, horizontal=True) 
USE_US_BINANCE = st.sidebar.checkbox("Use Binance.US", value=False)

with st.sidebar.expander("🔐 API Keys (Optional)", expanded=False):
    st.info("Keys sirf data limits barhane ke liye hain.")
    USER_API_KEY = st.text_input("API Key", type="password")
    USER_API_SECRET = st.text_input("API Secret", type="password")

PROXY_URL = st.sidebar.text_input("Proxy URL (Optional)", placeholder="http://user:pass@ip:port")

st.sidebar.divider()

# 2. ALERTS & GOOGLE SHEETS
st.sidebar.subheader("🔔 Alerts & Exports")
ENABLE_SOUND = st.sidebar.checkbox("🔊 Enable Sound Alerts", value=True)

with st.sidebar.expander("💾 Google Sheets Setup", expanded=True):
    GSHEET_URL = st.text_input("Web App URL", placeholder="https://script.google.com/macros/s/...", help="Paste the Web App URL from Google Apps Script here.")
    AUTO_EXPORT = st.checkbox("Auto-Upload Results", value=False, help="Automatically send scan results to Google Sheet after every scan.")

st.sidebar.divider()

# 3. Strategy
st.sidebar.subheader("🔍 Strategy")
SEARCH_MODE = st.sidebar.radio(
    "Select Strategy:",
    ["📊 All-in-One Report", "Crossover Alert", "RSI Range", "Sustained Trend (Days)", "💸 Funding Flip Scanner (3 Days)", "🪤 Trap Master (Net Positions)"],
    index=5
)

# Dynamic Inputs
if SEARCH_MODE == "Crossover Alert":
    st.sidebar.info("🔔 Alert when RSI crosses BELOW level.")
    RSI_ALERT_LEVEL = st.sidebar.number_input("RSI Cross Level", 1, 100, 30)
elif SEARCH_MODE == "RSI Range":
    st.sidebar.info("↔️ Show coins inside a specific RSI Range.")
    col1, col2 = st.sidebar.columns(2)
    MIN_RSI = col1.number_input("Min RSI", 1, 100, 70)
    MAX_RSI = col2.number_input("Max RSI", 1, 100, 90)
elif SEARCH_MODE == "Sustained Trend (Days)":
    st.sidebar.info("📅 Find coins staying Above/Below RSI.")
    SUSTAINED_DAYS = st.sidebar.number_input("Duration (Days)", 1, 30, 3)
    TREND_TYPE = st.sidebar.selectbox("Condition", ["Always ABOVE", "Always BELOW"])
    TREND_RSI_LEVEL = st.sidebar.number_input("RSI Threshold", 1, 100, 70)
elif SEARCH_MODE == "📊 All-in-One Report":
    st.sidebar.info("📑 Analyze Market for Signals, Overbought & Oversold.")
    col1, col2 = st.sidebar.columns(2)
    OVERBOUGHT_VAL = col1.number_input("Overbought (>)", 50, 100, 70)
    OVERSOLD_VAL = col2.number_input("Oversold (<)", 1, 50, 30)
    REPORT_DAYS = st.sidebar.number_input("Sustained Trend Days", 1, 10, 3)
elif SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)":
    st.sidebar.info("🔎 Finds Funding Flips with Price Change Analysis.")
elif SEARCH_MODE == "🪤 Trap Master (Net Positions)":
    st.sidebar.info("🧠 Analyzes 'Net Positions' logic to find Overcrowded Shorts/Longs.")
    TRAP_RSI_THRESHOLD = st.sidebar.number_input("Overcrowded Threshold (RSI)", 50, 100, 70, help="Level above which positions are considered 'Trapped' or 'Heavy'.")

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

def play_sound_alert():
    sound_html = """
    <audio autoplay>
    <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
    </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)

def send_to_google_sheet(data, url):
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Google Sheet Error: {response.text}")
            return False
    except Exception as e:
        st.error(f"Failed to upload to Sheet: {e}")
        return False

def get_data_with_indicators(client, symbol, tf, market_type, limit=100, get_oi=False):
    try:
        # 1. Get Klines
        if market_type == "Spot":
            klines = client.get_klines(symbol=symbol, interval=tf, limit=limit)
        else:
            klines = client.futures_klines(symbol=symbol, interval=tf, limit=limit)
            
        df = pd.DataFrame(klines, columns=[
            'time','open','high','low','close','volume','close_time','qav','num_trades','tbv','tqv','ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Indicators
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / df['vol_sma']
        
        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # 2. Get Open Interest (For Trap Master Logic)
        if get_oi and market_type == "Futures":
            try:
                oi_data = client.futures_open_interest_hist(symbol=symbol, period=tf, limit=limit)
                df_oi = pd.DataFrame(oi_data)
                df_oi['sumOpenInterestValue'] = df_oi['sumOpenInterestValue'].astype(float)
                # Align data lengths
                min_len = min(len(df), len(df_oi))
                df = df.iloc[-min_len:].reset_index(drop=True)
                df_oi = df_oi.iloc[-min_len:].reset_index(drop=True)
                
                df['oi'] = df_oi['sumOpenInterestValue']
                df['delta_oi'] = df['oi'].diff()
                
                # --- PINE SCRIPT LOGIC IMPLEMENTATION ---
                # Conditions
                price_up = df['close'] > df['open']
                price_down = df['close'] < df['open']
                oi_up = df['delta_oi'] > 0
                oi_down = df['delta_oi'] < 0
                
                # Classify Moves
                new_longs = (oi_up & price_up)
                rekt_longs = (oi_down & price_down)
                new_shorts = (oi_up & price_down)
                rekt_shorts = (oi_down & price_up)
                
                # Calculate Values
                longs_entering = new_longs * df['delta_oi']
                longs_exiting = rekt_longs * df['delta_oi']
                shorts_entering = new_shorts * df['delta_oi']
                shorts_exiting = rekt_shorts * df['delta_oi']
                
                # Cumulative Net Positions
                df['net_longs'] = (longs_entering.fillna(0) - longs_exiting.fillna(0).abs()).cumsum()
                df['net_shorts'] = (shorts_entering.fillna(0) - shorts_exiting.fillna(0).abs()).cumsum()
                
                # Calculate RSI of Net Positions
                df['nl_rsi'] = ta.momentum.RSIIndicator(close=df['net_longs'], window=14).rsi()
                df['ns_rsi'] = ta.momentum.RSIIndicator(close=df['net_shorts'], window=14).rsi()
                
            except Exception:
                df['nl_rsi'] = 50
                df['ns_rsi'] = 50
        
        return df
    except:
        return None

def check_funding_flip(client, symbol):
    try:
        rates = client.futures_funding_rate(symbol=symbol, limit=2)
        if len(rates) < 2: return None
        curr = float(rates[-1]['fundingRate'])
        prev = float(rates[-2]['fundingRate'])
        flip_ts = rates[-1]['fundingTime']
        flip_time = datetime.utcfromtimestamp(flip_ts / 1000).strftime('%H:%M UTC')
        if prev < 0 and curr > 0: return f"Flip Pos 🟢 {flip_time}"
        if prev > 0 and curr < 0: return f"Flip Neg 🔴 {flip_time}"
        return None
    except: return None

def get_historical_funding_flip(client, symbol, days=3):
    try:
        rates = client.futures_funding_rate(symbol=symbol, limit=100) 
        if not rates or len(rates) < 2: return None
        cutoff_time = time.time() * 1000 - (days * 24 * 60 * 60 * 1000)
        for i in range(len(rates) - 1, 0, -1):
            curr_rate = float(rates[i]['fundingRate'])
            prev_rate = float(rates[i-1]['fundingRate'])
            timestamp = rates[i]['fundingTime']
            if timestamp < cutoff_time: break
            if (prev_rate < 0 and curr_rate > 0) or (prev_rate > 0 and curr_rate < 0):
                flip_time = datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
                return {
                    "Flip Time": flip_time,
                    "Old Funding": f"{prev_rate*100:.4f}%",
                    "Flip Rate": f"{curr_rate*100:.4f}%",
                    "Type": "Positive" if curr_rate > 0 else "Negative",
                    "Timestamp": timestamp
                }
        return None
    except: return None

def get_long_short_ratio(client, symbol):
    try:
        data = client.futures_top_longshort_account_ratio(symbol=symbol, period='4h', limit=1)
        if data:
            l_ratio = float(data[0]['longAccount']) * 100
            s_ratio = float(data[0]['shortAccount']) * 100
            return l_ratio, s_ratio
        return 0, 0
    except: return 0, 0

def plot_chart(client, symbol, tf, market_type):
    with st.spinner(f"Loading Chart for {symbol}..."):
        is_trap_mode = (SEARCH_MODE == "🪤 Trap Master (Net Positions)")
        df = get_data_with_indicators(client, symbol, tf, market_type, limit=150, get_oi=is_trap_mode)
        if df is None: st.error("Could not load chart data."); return

        rows = 3
        titles = (f'{symbol} Price', 'RSI (14)', 'MACD')
        
        if is_trap_mode and 'nl_rsi' in df.columns:
            titles = (f'{symbol} Price', 'RSI (14)', 'Net Positions RSI (Green=Longs, Red=Shorts)')
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            subplot_titles=titles, row_width=[0.2, 0.2, 0.6])

        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        if is_trap_mode and 'nl_rsi' in df.columns:
            # Trap Mode Chart
            fig.add_trace(go.Scatter(x=df['time'], y=df['nl_rsi'], name='Net Long RSI', line=dict(color='#00ff00', width=2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['ns_rsi'], name='Net Short RSI', line=dict(color='#ff0000', width=2)), row=3, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="white", row=3, col=1, annotation_text="Overcrowded")
        else:
            fig.add_trace(go.Scatter(x=df['time'], y=df['macd'], name='MACD', line=dict(color='blue', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], name='Signal', line=dict(color='orange', width=1.5)), row=3, col=1)
            colors = ['green' if val >= 0 else 'red' for val in df['macd_diff']]
            fig.add_trace(go.Bar(x=df['time'], y=df['macd_diff'], name='Hist', marker_color=colors), row=3, col=1)
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=800, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- WORKER FUNCTION ---
def analyze_symbol(symbol, client_args, scan_params):
    try:
        client = scan_params['client']
        TIMEFRAME = scan_params['TIMEFRAME']
        MARKET_TYPE = scan_params['MARKET_TYPE']
        SEARCH_MODE = scan_params['SEARCH_MODE']
        funding_map = scan_params['funding_map']
        candles_needed = scan_params['candles_needed']
        
        RSI_ALERT_LEVEL = scan_params.get('RSI_ALERT_LEVEL', 30)
        MIN_RSI = scan_params.get('MIN_RSI', 70)
        MAX_RSI = scan_params.get('MAX_RSI', 90)
        OVERBOUGHT_VAL = scan_params.get('OVERBOUGHT_VAL', 70)
        OVERSOLD_VAL = scan_params.get('OVERSOLD_VAL', 30)
        TREND_TYPE = scan_params.get('TREND_TYPE', "Always ABOVE")
        TREND_RSI_LEVEL = scan_params.get('TREND_RSI_LEVEL', 70)
        TRAP_RSI_THRESHOLD = scan_params.get('TRAP_RSI_THRESHOLD', 70)

        # 1. Fetch Data
        get_oi = (SEARCH_MODE == "🪤 Trap Master (Net Positions)")
        df = get_data_with_indicators(client, symbol, TIMEFRAME, MARKET_TYPE, limit=max(100, candles_needed), get_oi=get_oi)
        
        if df is None or len(df) < 30: return None

        curr_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        curr_price = df['close'].iloc[-1]
        curr_rvol = df['rvol'].iloc[-1]
        curr_macd = df['macd'].iloc[-1]
        curr_signal = df['macd_signal'].iloc[-1]
        macd_trend = "BULLISH 🟢" if curr_macd > curr_signal else "BEARISH 🔴"
        
        funding_rate_display = "N/A"
        if MARKET_TYPE == "Futures" and symbol in funding_map:
            fr = funding_map[symbol]
            funding_rate_display = f"{fr * 100:.4f}%"

        match_found = False
        status_msg = ""
        signal_type = "NEUTRAL"
        group_tag = "Normal"
        
        flip_time = "-"; old_funding = "-"; flip_rate = "-"; long_pct = 0.0; short_pct = 0.0; flip_type = "Neutral"
        flip_price_val = "N/A"; change_pct_str = "0.00%"
        
        # Trap Vars
        nl_rsi_val = 0; ns_rsi_val = 0; trap_message = ""

        # --- LOGIC SELECTION ---
        if SEARCH_MODE == "🪤 Trap Master (Net Positions)":
            if MARKET_TYPE != "Futures": return None 
            if 'nl_rsi' in df.columns:
                nl_rsi_val = df['nl_rsi'].iloc[-1]
                ns_rsi_val = df['ns_rsi'].iloc[-1]
                
                # Simplified Report Logic
                if nl_rsi_val > TRAP_RSI_THRESHOLD:
                    match_found = True
                    trap_message = "Crowded Longs (Dump Risk 📉)"
                    group_tag = "TrapLongs"
                    signal_type = "SELL"
                elif ns_rsi_val > TRAP_RSI_THRESHOLD:
                    match_found = True
                    trap_message = "Crowded Shorts (Squeeze Risk 🚀)"
                    group_tag = "TrapShorts"
                    signal_type = "BUY"

        elif SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)":
            if MARKET_TYPE == "Futures":
                hist_flip = get_historical_funding_flip(client, symbol, days=3)
                if hist_flip:
                    match_found = True
                    status_msg = f"Flip: {hist_flip['Type']}"
                    signal_type = "ALERT"
                    flip_type = hist_flip['Type']
                    flip_time = hist_flip['Flip Time']
                    old_funding = hist_flip['Old Funding']
                    flip_rate = hist_flip['Flip Rate']
                    funding_rate_display = flip_rate
                    long_pct, short_pct = get_long_short_ratio(client, symbol)
                    try:
                        flip_kline = client.futures_klines(symbol=symbol, interval='1h', startTime=hist_flip['Timestamp'], limit=1)
                        if flip_kline:
                            f_price = float(flip_kline[0][1])
                            flip_price_val = f"{f_price}"
                            pct_change = ((curr_price - f_price) / f_price) * 100
                            trend_icon = "📈" if pct_change >= 0 else "📉"
                            change_pct_str = f"{pct_change:.2f}% {trend_icon}"
                    except: pass

        elif SEARCH_MODE == "📊 All-in-One Report":
            if prev_rsi > OVERSOLD_VAL and curr_rsi <= OVERSOLD_VAL:
                    match_found = True; status_msg = "📉 BREAKDOWN (Buy Dip)"; signal_type = "BUY"; group_tag = "Signal"
            elif prev_rsi < OVERBOUGHT_VAL and curr_rsi >= OVERBOUGHT_VAL:
                    match_found = True; status_msg = "🚀 BREAKOUT (Pump)"; signal_type = "SELL"; group_tag = "Signal"
            elif curr_rsi <= OVERSOLD_VAL:
                    match_found = True; status_msg = "Oversold Zone"; signal_type = "BUY"; group_tag = "Oversold"
            elif curr_rsi >= OVERBOUGHT_VAL:
                    match_found = True; status_msg = "Overbought Zone"; signal_type = "SELL"; group_tag = "Overbought"
        
        elif SEARCH_MODE == "Crossover Alert":
            if RSI_ALERT_LEVEL < 50:
                if prev_rsi > RSI_ALERT_LEVEL and curr_rsi <= RSI_ALERT_LEVEL: match_found = True; status_msg = "CROSS BELOW"; signal_type = "BUY"; group_tag="Alert"
            else:
                if prev_rsi < RSI_ALERT_LEVEL and curr_rsi >= RSI_ALERT_LEVEL: match_found = True; status_msg = "CROSS ABOVE"; signal_type = "SELL"; group_tag="Alert"

        elif SEARCH_MODE == "RSI Range":
            if MIN_RSI <= curr_rsi <= MAX_RSI: match_found = True; status_msg = f"IN RANGE"; group_tag = "Range"

        elif SEARCH_MODE == "Sustained Trend (Days)":
            relevant_rsi = df['rsi'].iloc[-(candles_needed-14):]
            if not relevant_rsi.empty:
                if "ABOVE" in TREND_TYPE and relevant_rsi.min() > TREND_RSI_LEVEL: match_found = True; status_msg = f"STRONG BULLISH 🔥"; group_tag = "Trend"
                elif "BELOW" in TREND_TYPE and relevant_rsi.max() < TREND_RSI_LEVEL: match_found = True; status_msg = f"STRONG BEARISH ❄️"; group_tag = "Trend"

        if match_found and MARKET_TYPE == "Futures" and SEARCH_MODE != "💸 Funding Flip Scanner (3 Days)":
                flip_status = check_funding_flip(client, symbol)
                if flip_status: funding_rate_display += f" ({flip_status})"

        if match_found:
            return {
                "Symbol": symbol,
                "Price": curr_price,
                "RSI": round(curr_rsi, 2),
                "Trend (MACD)": macd_trend, 
                "Signal": signal_type,
                "Funding": funding_rate_display,
                "Status": status_msg,
                "Group": group_tag,
                "Flip Type": flip_type,
                "Flip Time": flip_time,
                "Old Funding": old_funding,
                "Current Funding": flip_rate if SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)" else funding_rate_display,
                "Long %": f"{long_pct:.1f}%",
                "Short %": f"{short_pct:.1f}%",
                "_long_val": long_pct,
                "_short_val": short_pct,
                "has_funding_flip_alert": "Flip" in funding_rate_display,
                "Flip Price": flip_price_val,
                "Change %": change_pct_str,
                # Easy Report Columns for Trap Master
                "Market State": trap_message,
                "NL RSI": round(nl_rsi_val, 2),
                "NS RSI": round(ns_rsi_val, 2)
            }
        return None
    except: return None

# --- MAIN LOGIC ---
st.title(f"🤖 Binance RSI Pro Scanner 3.2")

if USE_US_BINANCE: st.warning("🇺🇸 Using Binance.US")
if PROXY_URL: st.info("🌐 Using Proxy")

try:
    client = init_client(USE_US_BINANCE, PROXY_URL, USER_API_KEY, USER_API_SECRET)
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.stop()

if st.button("🔄 Start New Scan", type="primary"):
    st.session_state.scan_results = []
    st.session_state.scan_performed = False

    if USE_US_BINANCE and MARKET_TYPE == "Futures":
        st.error("❌ Binance.US does not support Futures."); st.stop()

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text(f"Fetching {MARKET_TYPE} pairs...")
        funding_map = {}
        if MARKET_TYPE == "Spot": 
            exchange_info = client.get_exchange_info()
        else: 
            exchange_info = client.futures_exchange_info()
            try:
                mark_prices = client.futures_mark_price()
                for item in mark_prices: funding_map[item['symbol']] = float(item['lastFundingRate'])
            except: pass

        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']
        candles_needed = 100 
        days_to_check = REPORT_DAYS if SEARCH_MODE == "📊 All-in-One Report" else 0
        if SEARCH_MODE == "Sustained Trend (Days)": days_to_check = SUSTAINED_DAYS
        if days_to_check > 0:
            tf_minutes = get_tf_in_minutes(TIMEFRAME)
            candles_needed = int((days_to_check * 1440) / tf_minutes) + 30 

        alerts = []
        total_symbols = len(symbols)
        sound_triggered = False
        
        scan_params = {
            'client': client,
            'TIMEFRAME': TIMEFRAME,
            'MARKET_TYPE': MARKET_TYPE,
            'SEARCH_MODE': SEARCH_MODE,
            'funding_map': funding_map,
            'candles_needed': candles_needed,
            'RSI_ALERT_LEVEL': locals().get('RSI_ALERT_LEVEL', 30),
            'MIN_RSI': locals().get('MIN_RSI', 70),
            'MAX_RSI': locals().get('MAX_RSI', 90),
            'OVERBOUGHT_VAL': locals().get('OVERBOUGHT_VAL', 70),
            'OVERSOLD_VAL': locals().get('OVERSOLD_VAL', 30),
            'TREND_TYPE': locals().get('TREND_TYPE', "Always ABOVE"),
            'TREND_RSI_LEVEL': locals().get('TREND_RSI_LEVEL', 70),
            'TRAP_RSI_THRESHOLD': locals().get('TRAP_RSI_THRESHOLD', 70)
        }

        status_text.text(f"Scanning {total_symbols} coins (Parallel Mode)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(analyze_symbol, sym, {}, scan_params): sym for sym in symbols}
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                completed_count += 1
                progress = completed_count / total_symbols
                progress_bar.progress(progress)
                try:
                    result = future.result()
                    if result:
                        alerts.append(result)
                        if result.get("has_funding_flip_alert") and ENABLE_SOUND and not sound_triggered:
                            play_sound_alert(); sound_triggered = True; st.toast(f"🔔 Funding Change: {result['Symbol']}")
                except Exception as exc: pass 

        progress_bar.empty()
        status_text.success(f"✅ Scan Complete! Scanned {total_symbols} coins.")
        
        if alerts and GSHEET_URL and AUTO_EXPORT:
            if send_to_google_sheet(alerts, GSHEET_URL): st.toast("✅ Google Sheet Updated!")

        st.session_state.scan_results = alerts
        st.session_state.scan_performed = True
            
    except Exception as e:
        st.error(f"Error: {e}")

def highlight_ls(row):
    styles = [''] * len(row)
    try:
        l_val = row['_long_val']; s_val = row['_short_val']
        l_idx = row.index.get_loc("Long %"); s_idx = row.index.get_loc("Short %")
        if l_val > s_val: styles[l_idx] = 'color: orange; font-weight: bold'
        elif s_val > l_val: styles[s_idx] = 'color: orange; font-weight: bold'
    except: pass
    return styles

if st.session_state.scan_performed:
    alerts = st.session_state.scan_results
    if not alerts:
        st.warning("No signals found in the last scan.")
    else:
        st.sidebar.divider()
        st.sidebar.subheader("📊 Chart Visualizer")
        coin_list = [item['Symbol'] for item in alerts]
        selected_coin = st.sidebar.selectbox("Select Coin to View:", coin_list)
        
        tab1, tab2 = st.tabs(["📋 Scan Report", "📈 Chart Analysis"])
        
        with tab1:
            st.subheader(f"Found {len(alerts)} Signals")
            df_res = pd.DataFrame(alerts)
            
            if SEARCH_MODE == "🪤 Trap Master (Net Positions)":
                st.info("🔎 **Report Guide:**\n- **Crowded Shorts (Squeeze Risk):** Shorts bohat zyada hain, price uper ja sakti hai.\n- **Crowded Longs (Dump Risk):** Longs bohat zyada hain, price gir sakti hai.")
                cols_to_show = ['Symbol', 'Price', 'Market State', 'NL RSI', 'NS RSI', 'Signal']
                st.dataframe(df_res[cols_to_show], use_container_width=True)

            elif SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)":
                cols_to_show = ['Symbol', 'Price', 'Flip Time', 'Flip Price', 'Change %', 'Old Funding', 'Current Funding', 'Long %', 'Short %', 'Trend (MACD)', '_long_val', '_short_val', 'Flip Type']
                cols_to_show = [c for c in cols_to_show if c in df_res.columns]
                df_pos = df_res[df_res['Flip Type'] == "Positive"]
                df_neg = df_res[df_res['Flip Type'] == "Negative"]
                if not df_pos.empty:
                    st.subheader("🟢 Positive Funding (Flipped +)")
                    st.dataframe(df_pos[cols_to_show].drop(columns=['_long_val', '_short_val', 'Flip Type']).style.apply(highlight_ls, axis=1), use_container_width=True)
                if not df_neg.empty:
                    st.subheader("🔴 Negative Funding (Flipped -)")
                    st.dataframe(df_neg[cols_to_show].drop(columns=['_long_val', '_short_val', 'Flip Type']).style.apply(highlight_ls, axis=1), use_container_width=True)
                
            elif SEARCH_MODE == "📊 All-in-One Report":
                df_sig = df_res[df_res['Group'] == 'Signal']
                if not df_sig.empty: st.subheader("⚡ Fresh Signals"); st.dataframe(df_sig.drop(columns=['Group']), use_container_width=True)
                df_os = df_res[df_res['Group'] == 'Oversold']
                if not df_os.empty: st.subheader(f"🟢 Oversold"); st.dataframe(df_os.drop(columns=['Group']).sort_values("RSI"), use_container_width=True)
                df_ob = df_res[df_res['Group'] == 'Overbought']
                if not df_ob.empty: st.subheader(f"🔴 Overbought"); st.dataframe(df_ob.drop(columns=['Group']).sort_values("RSI", ascending=False), use_container_width=True)
                if df_sig.empty and df_os.empty and df_ob.empty: st.dataframe(df_res.drop(columns=['Group']), use_container_width=True)
            else:
                st.dataframe(df_res.drop(columns=['Group']), use_container_width=True)

        with tab2:
            st.subheader(f"{selected_coin} Analysis ({TIMEFRAME})")
            coin_details = next((item for item in alerts if item['Symbol'] == selected_coin), None)
            if coin_details:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Price", f"${coin_details['Price']}")
                c2.metric("RSI", coin_details['RSI'])
                c3.metric("Trend", coin_details.get('Trend (MACD)', 'N/A'))
                c4.metric("Market State", coin_details.get('Market State', 'N/A'))
                funding_val = coin_details.get('Current Funding', coin_details.get('Funding', 'N/A'))
                c5.metric("Funding", funding_val)
            plot_chart(client, selected_coin, TIMEFRAME, MARKET_TYPE)
