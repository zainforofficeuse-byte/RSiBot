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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Binance RSI Auto-Trader 2.7",
    page_icon="📈",
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
st.sidebar.title("⚙️ Bot Settings 2.7")

# 1. Connection
st.sidebar.subheader("🔌 Connection")
MARKET_TYPE = st.sidebar.radio("Market Type", ["Spot", "Futures"], horizontal=True)
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

# Google Sheet Integration
with st.sidebar.expander("💾 Google Sheets Setup", expanded=True):
    GSHEET_URL = st.text_input("Web App URL", placeholder="https://script.google.com/macros/s/...", help="Paste the Web App URL from Google Apps Script here.")
    AUTO_EXPORT = st.checkbox("Auto-Upload Results", value=False, help="Automatically send scan results to Google Sheet after every scan.")

st.sidebar.divider()

# 3. AUTO TRADING
st.sidebar.subheader("🤖 Auto Trading (Simulation)")
ENABLE_AUTOTRADE = st.sidebar.checkbox("Enable Paper Trading", value=False)

if ENABLE_AUTOTRADE:
    col_t1, col_t2 = st.sidebar.columns(2)
    TRADE_AMOUNT_USDT = col_t1.number_input("Simulated Amount ($)", min_value=10.0, value=15.0)
    MAX_OPEN_TRADES = col_t2.number_input("Max Trades", min_value=1, value=3)

st.sidebar.divider()

# 4. Strategy
st.sidebar.subheader("🔍 Strategy")
SEARCH_MODE = st.sidebar.radio(
    "Select Strategy:",
    ["📊 All-in-One Report", "Crossover Alert", "RSI Range", "Sustained Trend (Days)", "💸 Funding Flip Scanner (3 Days)"]
)

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
elif SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)":
    st.sidebar.info("🔎 Finds coins where Funding Rate changed sign (+/-) in the last 3 days.")

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

def place_order_simulation(symbol, side, amount_usdt, price):
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

def get_data_with_indicators(client, symbol, tf, market_type, limit=500):
    try:
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
        
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / df['vol_sma']

        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
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
    except:
        return None

def get_historical_funding_flip(client, symbol, days=3):
    """
    Checks for ANY funding rate flip in the last X days.
    Returns details if found.
    """
    try:
        # Fetching enough history (3 days * 24 hours assuming worst case 1h intervals = ~72)
        rates = client.futures_funding_rate(symbol=symbol, limit=100) 
        if not rates or len(rates) < 2: return None
        
        cutoff_time = time.time() * 1000 - (days * 24 * 60 * 60 * 1000)
        
        # Iterate from newest to oldest
        for i in range(len(rates) - 1, 0, -1):
            curr_rate = float(rates[i]['fundingRate'])
            prev_rate = float(rates[i-1]['fundingRate'])
            timestamp = rates[i]['fundingTime']
            
            if timestamp < cutoff_time:
                break
                
            # Check sign change
            if (prev_rate < 0 and curr_rate > 0) or (prev_rate > 0 and curr_rate < 0):
                flip_time = datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
                return {
                    "Flip Time": flip_time,
                    "Old Funding": f"{prev_rate*100:.4f}%",
                    "Flip Rate": f"{curr_rate*100:.4f}%",
                    "Type": "Positive 🟢" if curr_rate > 0 else "Negative 🔴"
                }
        return None
    except:
        return None

def plot_chart(client, symbol, tf, market_type):
    with st.spinner(f"Loading Chart for {symbol}..."):
        df = get_data_with_indicators(client, symbol, tf, market_type, limit=150)
        if df is None:
            st.error("Could not load chart data.")
            return

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=(f'{symbol} Price', 'RSI (14)', 'MACD'),
                            row_width=[0.2, 0.2, 0.6])

        fig.add_trace(go.Candlestick(x=df['time'],
                        open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'], name='Price'), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.add_trace(go.Scatter(x=df['time'], y=df['macd'], name='MACD', line=dict(color='blue', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], name='Signal', line=dict(color='orange', width=1.5)), row=3, col=1)
        
        colors = ['green' if val >= 0 else 'red' for val in df['macd_diff']]
        fig.add_trace(go.Bar(x=df['time'], y=df['macd_diff'], name='Hist', marker_color=colors), row=3, col=1)
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=800, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- MAIN LOGIC ---
st.title(f"🤖 Binance RSI Pro Bot 2.7")

if USE_US_BINANCE: st.warning("🇺🇸 Using Binance.US")
if PROXY_URL: st.info("🌐 Using Proxy")

try:
    client = init_client(USE_US_BINANCE, PROXY_URL, USER_API_KEY, USER_API_SECRET)
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.stop()

# --- SCAN BUTTON ---
btn_label = "🔄 Scan & Simulate" if ENABLE_AUTOTRADE else "🔄 Start New Scan"

if st.button(btn_label, type="primary"):
    
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
        trades_executed = 0
        trade_logs = []
        total_symbols = len(symbols)
        sound_triggered = False

        for i, symbol in enumerate(symbols):
            progress_bar.progress((i + 1) / total_symbols)
            status_text.text(f"Scanning {i+1}/{total_symbols}: {symbol}...")
            
            # --- 1. Basic Data (Price, RSI, MACD) ---
            df = get_data_with_indicators(client, symbol, TIMEFRAME, MARKET_TYPE, limit=max(100, candles_needed))
            
            if df is not None and len(df) > 30:
                curr_rsi = df['rsi'].iloc[-1]
                prev_rsi = df['rsi'].iloc[-2]
                curr_price = df['close'].iloc[-1]
                curr_rvol = df['rvol'].iloc[-1]
                
                curr_macd = df['macd'].iloc[-1]
                curr_signal = df['macd_signal'].iloc[-1]
                macd_trend = "BULLISH 🟢" if curr_macd > curr_signal else "BEARISH 🔴"
                
                # Default Funding Display
                funding_rate_display = "N/A"
                if MARKET_TYPE == "Futures" and symbol in funding_map:
                    fr = funding_map[symbol]
                    funding_rate_display = f"{fr * 100:.4f}%"

                match_found = False
                status_msg = ""
                signal_type = "NEUTRAL"
                group_tag = "Normal"
                
                # --- Specific Fields for Funding Scanner ---
                flip_time = "-"
                old_funding = "-"
                flip_rate = "-"
                
                # --- LOGIC SELECTION ---
                if SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)":
                    if MARKET_TYPE == "Futures":
                        hist_flip = get_historical_funding_flip(client, symbol, days=3)
                        if hist_flip:
                            match_found = True
                            status_msg = f"Flip: {hist_flip['Type']}"
                            signal_type = "ALERT"
                            group_tag = "FundingFlip"
                            # Store extra details
                            flip_time = hist_flip['Flip Time']
                            old_funding = hist_flip['Old Funding']
                            flip_rate = hist_flip['Flip Rate']
                    else:
                        pass # Spot me funding nahi hoti

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
                        if prev_rsi > RSI_ALERT_LEVEL and curr_rsi <= RSI_ALERT_LEVEL:
                            match_found = True; status_msg = "CROSS BELOW"; signal_type = "BUY"; group_tag="Alert"
                
                # --- FUNDING CHANGE NOTIFICATION (Current) ---
                if match_found and MARKET_TYPE == "Futures" and SEARCH_MODE != "💸 Funding Flip Scanner (3 Days)":
                     flip_status = check_funding_flip(client, symbol)
                     if flip_status:
                         funding_rate_display += f" ({flip_status})"
                         if ENABLE_SOUND:
                             st.toast(f"💸 Funding Flip: {symbol} {flip_status}", icon="🔔")
                             if not sound_triggered: 
                                 play_sound_alert()
                                 sound_triggered = True

                # --- AUTO TRADE SIMULATION ---
                if match_found and ENABLE_AUTOTRADE and trades_executed < MAX_OPEN_TRADES:
                    if signal_type == "BUY":
                        st.toast(f"⚡ Simulating BUY {symbol}...")
                        trade_result = place_order_simulation(symbol, "BUY", TRADE_AMOUNT_USDT, curr_price)
                        status_msg += " | ✅ SIMULATED"
                        trades_executed += 1
                        trade_logs.append(trade_result)

                if match_found:
                    alert_data = {
                        "Symbol": symbol,
                        "Price": curr_price,
                        "RSI": round(curr_rsi, 2),
                        "RVOL": round(curr_rvol, 2),
                        "Trend (MACD)": macd_trend, 
                        "Signal": signal_type,
                        "Funding": funding_rate_display,
                        "Status": status_msg,
                        "Group": group_tag 
                    }
                    
                    # Add extra columns only for Funding Scanner
                    if SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)":
                        alert_data["Flip Time"] = flip_time
                        alert_data["Old Funding"] = old_funding
                        alert_data["Flip Rate"] = flip_rate
                        alert_data["Current Funding"] = funding_rate_display # Rename for clarity

                    alerts.append(alert_data)
            
            if ENABLE_AUTOTRADE and trades_executed >= MAX_OPEN_TRADES: break

        progress_bar.empty()
        status_text.success(f"✅ Scan Complete!")
        
        # --- GOOGLE SHEET EXPORT ---
        if alerts and GSHEET_URL and AUTO_EXPORT:
            status_text.text("Uploading to Google Sheet...")
            if send_to_google_sheet(alerts, GSHEET_URL):
                st.toast("✅ Data uploaded to Google Sheet!", icon="💾")
            else:
                st.toast("❌ Google Sheet Upload Failed", icon="⚠️")

        st.session_state.scan_results = alerts
        st.session_state.trade_logs = trade_logs
        st.session_state.scan_performed = True
            
    except Exception as e:
        st.error(f"Error: {e}")

# --- DISPLAY RESULTS & CHARTS ---
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
            
            # Manual Upload Button
            if GSHEET_URL and not AUTO_EXPORT:
                if st.button("💾 Upload Results to Google Sheet"):
                    if send_to_google_sheet(alerts, GSHEET_URL):
                        st.success("Uploaded successfully!")

            df_res = pd.DataFrame(alerts)
            
            # --- SPECIAL DISPLAY FOR FUNDING SCANNER ---
            if SEARCH_MODE == "💸 Funding Flip Scanner (3 Days)":
                st.subheader("💸 Recent Funding Rate Flips")
                # Showing specific columns for this mode
                cols_to_show = ['Symbol', 'Price', 'Flip Time', 'Old Funding', 'Flip Rate', 'Current Funding', 'RSI', 'Trend (MACD)']
                st.dataframe(df_res[cols_to_show], use_container_width=True)
            
            elif SEARCH_MODE == "📊 All-in-One Report":
                df_sig = df_res[df_res['Group'] == 'Signal']
                if not df_sig.empty:
                    st.subheader("⚡ Fresh Signals")
                    st.dataframe(df_sig.drop(columns=['Group']), use_container_width=True)

                df_os = df_res[df_res['Group'] == 'Oversold']
                if not df_os.empty:
                    st.subheader(f"🟢 Oversold (< {OVERSOLD_VAL})")
                    st.dataframe(df_os.drop(columns=['Group']).sort_values("RSI"), use_container_width=True)

                df_ob = df_res[df_res['Group'] == 'Overbought']
                if not df_ob.empty:
                    st.subheader(f"🔴 Overbought (> {OVERBOUGHT_VAL})")
                    st.dataframe(df_ob.drop(columns=['Group']).sort_values("RSI", ascending=False), use_container_width=True)
                
                if df_sig.empty and df_os.empty and df_ob.empty:
                    st.dataframe(df_res.drop(columns=['Group']), use_container_width=True)
            else:
                st.dataframe(df_res.drop(columns=['Group']), use_container_width=True)
            
            if 'trade_logs' in st.session_state and st.session_state.trade_logs:
                st.divider(); st.subheader("📜 Simulation Log"); st.json(st.session_state.trade_logs)

        with tab2:
            st.subheader(f"{selected_coin} Analysis ({TIMEFRAME})")
            coin_details = next((item for item in alerts if item['Symbol'] == selected_coin), None)
            if coin_details:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Price", f"${coin_details['Price']}")
                c2.metric("RSI", coin_details['RSI'])
                c3.metric("RVOL", coin_details['RVOL'])
                c4.metric("MACD", coin_details['Trend (MACD)'])
                # Handle Funding key depending on mode
                funding_val = coin_details.get('Current Funding', coin_details.get('Funding', 'N/A'))
                c5.metric("Funding", funding_val)
            plot_chart(client, selected_coin, TIMEFRAME, MARKET_TYPE)
