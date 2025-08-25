"""
Minimal Stock Market Visualizer — Streamlit App

Features:
- Streamlit UI with sidebar controls
- Fetches data via yfinance
- Interactive Plotly candlestick charts with overlays (SMA/EMA, Bollinger Bands, RSI)
- Customizable timeframe, intervals, thresholds, and colors
- Save/load user configurations (JSON)
- Helper instructions to push code to GitHub

Run:
  streamlit run app_minimal.py
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Market Visualizer", page_icon="📈", layout="wide")

# ------------------------------
# Helpers
# ------------------------------

def sma(series: pd.Series, window: int):
    return series.rolling(window).mean()

def ema(series: pd.Series, window: int):
    return series.ewm(span=window, adjust=False).mean()

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2):
    mid = sma(series, window)
    sd = series.rolling(window).std()
    return mid + num_std * sd, mid - num_std * sd

def rsi(series: pd.Series, window: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=60)
def fetch_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    return df

# ------------------------------
# Sidebar controls
# ------------------------------

st.sidebar.title("📊 Controls")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

ma_windows = st.sidebar.multiselect("Moving Averages", [20, 50, 200], default=[20, 50])
use_ema = st.sidebar.checkbox("Use EMA (off = SMA)", value=False)
bb_window = st.sidebar.number_input("Bollinger Window", 5, 50, 20)
rsi_window = st.sidebar.number_input("RSI Window", 5, 50, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 50, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 50, 30)

# Save/load config
with st.sidebar.expander("💾 Config"):
    cfg = dict(ticker=ticker, period=period, interval=interval, ma_windows=ma_windows, use_ema=use_ema,
               bb_window=bb_window, rsi_window=rsi_window, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold)
    st.download_button("Download Config", data=json.dumps(cfg).encode(), file_name="config.json")
    upl = st.file_uploader("Upload Config", type="json")
    if upl:
        new_cfg = json.load(upl)
        st.session_state.update(new_cfg)
        st.experimental_rerun()

# ------------------------------
# Main chart
# ------------------------------

df = fetch_data(ticker, period, interval)
if df.empty:
    st.error("No data found.")
    st.stop()

close = df["Close"]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.07,
                    subplot_titles=(f"{ticker} Price", "RSI"))

fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                             increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)

for w in ma_windows:
    ma_series = ema(close, w) if use_ema else sma(close, w)
    fig.add_trace(go.Scatter(x=df.index, y=ma_series, mode="lines", name=f"{'EMA' if use_ema else 'SMA'} {w}"), row=1, col=1)

bb_up, bb_low = bollinger_bands(close, bb_window)
fig.add_trace(go.Scatter(x=df.index, y=bb_up, line=dict(width=1), name="BB Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=bb_low, line=dict(width=1), name="BB Lower"), row=1, col=1)

rsi_vals = rsi(close, rsi_window)
fig.add_trace(go.Scatter(x=df.index, y=rsi_vals, mode="lines", name=f"RSI {rsi_window}"), row=2, col=1)
fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified", height=700)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# GitHub helper
# ------------------------------

with st.expander("🐙 Push to GitHub"):
    st.markdown("""
    ```bash
    git init
    git add app_minimal.py
    git commit -m "Add minimal Stock Market Visualizer"
    git branch -M main
    git remote add origin https://github.com/<user>/<repo>.git
    git push -u origin main
    ```
    """)

st.caption("Built with Streamlit, Plotly, and yfinance. For informational purposes only.")
