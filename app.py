"""
Stock Market Observer — Streamlit App

Requirements
- Streamlit UI with Plotly charts
- Data from yfinance (optional: Barchart via API key)

Key Features
- Choose index or any ticker (S&P 500, NYSE Composite, etc.)
- Interactive candlestick or line charts with overlays (SMA/EMA, RSI, MACD)
- Customizable timeframe, interval, and colors
- Breadth & Sentiment tab:
  1) Put/Call Ratio (^CPC or ^CPCE from Yahoo Finance)
  2) % of constituents above 20-DMA (S&P 500 auto, NYSE via upload or subset)
  3) Advance–Decline Line (computed from constituents)
  4) CNN Fear & Greed Index (best-effort fetch from CNN JSON)
- Save/Load configuration as JSON
- GitHub helper panel with commands to push the project

Suggested requirements.txt:
streamlit>=1.33
plotly>=5.18
pandas>=2.2
numpy>=1.26
yfinance>=0.2.50
openpyxl>=3.1  # if you want Excel upload for constituents
requests>=2.31

Run:
  streamlit run stock_market_observer.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import requests
import streamlit as st
import yfinance as yf

# ------------------------------
# Page Setup
# ------------------------------

st.set_page_config(
    page_title="Stock Market Observer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .stButton>button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helpers — indicators
# ------------------------------

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

# ------------------------------
# Data fetchers (cached)
# ------------------------------

@st.cache_data(show_spinner=False, ttl=90)
def fetch_price_history(symbol: str, period: str, interval: str, auto_adjust: bool = True) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=auto_adjust, progress=False)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(show_spinner=False, ttl=900)
def fetch_constituents_sp500() -> List[str]:
    # Wikipedia list of S&P 500 companies
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
        return tickers
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=900)
def fetch_cnn_fear_greed() -> pd.DataFrame:
    """Best-effort: CNN publishes a JSON with historical values.
    If unavailable, returns an empty DataFrame.
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Expect data like {"fear_and_greed_historical": [{"x": 1700000000, "y": 55}, ...]}
        nodes = data.get("fear_and_greed_historical", []) or data.get("fear_and_greed", [])
        if not nodes:
            return pd.DataFrame()
        df = pd.DataFrame(nodes)
        # x might be epoch seconds or ms; normalize
        x = df["x"].astype(float)
        if x.max() > 10_000_000_000:  # ms
            ts = pd.to_datetime(x, unit="ms")
        else:
            ts = pd.to_datetime(x, unit="s")
        out = pd.DataFrame({"Date": ts, "Value": df["y"].astype(float)})
        out.set_index("Date", inplace=True)
        return out.sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=900)
def batch_download(tickers: List[str], period: str = "6mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Download multiple tickers with yfinance and return dict of DataFrames (Close, Volume, etc.)."""
    if not tickers:
        return {}
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, group_by='ticker', progress=False)
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = data[t].copy()
                df.index = pd.to_datetime(df.index)
                out[t] = df
            except Exception:
                continue
    else:
        # Single column frame when len(tickers)==1
        t = tickers[0]
        df = data.copy()
        df.index = pd.to_datetime(df.index)
        out[t] = df
    return out

# ------------------------------
# Breadth calculations
# ------------------------------

def pct_above_20dma(price_dict: Dict[str, pd.DataFrame]) -> pd.Series:
    """Return timeseries of % of tickers whose Close > 20-DMA each day (aligned index)."""
    if not price_dict:
        return pd.Series(dtype=float)
    # Build aligned Close frames
    closes = []
    for t, df in price_dict.items():
        if "Close" in df:
            s = df["Close"].rename(t)
            closes.append(s)
    if not closes:
        return pd.Series(dtype=float)
    prices = pd.concat(closes, axis=1)
    ma20 = prices.rolling(20, min_periods=20).mean()
    above = (prices > ma20)
    pct = above.sum(axis=1) / above.shape[1] * 100.0
    return pct.dropna()

def advance_decline_line(price_dict: Dict[str, pd.DataFrame]) -> pd.Series:
    """Compute advance-decline line from constituents.
    Advance = Close > previous Close; Decline = Close < previous Close.
    AD Line = cumulative sum of (Advances - Declines).
    """
    if not price_dict:
        return pd.Series(dtype=float)
    closes = []
    for t, df in price_dict.items():
        if "Close" in df:
            closes.append(df["Close"].rename(t))
    if not closes:
        return pd.Series(dtype=float)
    prices = pd.concat(closes, axis=1).dropna(how='all')
    rets = prices.pct_change()
    adv = (rets > 0).sum(axis=1)
    dec = (rets < 0).sum(axis=1)
    ad = (adv - dec).cumsum()
    return ad

# ------------------------------
# Sidebar Controls
# ------------------------------

st.sidebar.title("📈 Stock Market Observer")
st.sidebar.caption("Analyze indices and market breadth.")

# Data source options
with st.sidebar.expander("Data Sources", expanded=False):
    use_barchart = st.toggle("Use Barchart (API key)", value=False)
    barchart_key = st.text_input("Barchart API Key (optional)", type="password") if use_barchart else ""

# Universe / Index selection
index_choice = st.sidebar.selectbox(
    "Market Index / Symbol",
    options=["S&P 500 (^GSPC)", "NYSE Composite (^NYA)", "Custom Ticker"],
    index=0,
)

custom_ticker = st.sidebar.text_input("Custom Ticker (e.g., MSFT)", value="AAPL") if index_choice == "Custom Ticker" else ""
primary_symbol = {"S&P 500 (^GSPC)": "^GSPC", "NYSE Composite (^NYA)": "^NYA", "Custom Ticker": custom_ticker or "AAPL"}[index_choice]

period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","ytd","1y","5y","max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
ma_windows = st.sidebar.multiselect("MAs", options=[5,10,20,50,100,200], default=[20,50,200])
use_ema = st.sidebar.toggle("Use EMA (off=SMA)", value=False)

rsi_window = st.sidebar.number_input("RSI Window", 5, 50, 14)
macd_fast = st.sidebar.number_input("MACD Fast", 5, 50, 12)
macd_slow = st.sidebar.number_input("MACD Slow", 10, 100, 26)
macd_signal = st.sidebar.number_input("MACD Signal", 3, 30, 9)

# Colors
with st.sidebar.expander("Colors", expanded=False):
    candle_up = st.color_picker("Candle Up", "#26a69a")
    candle_down = st.color_picker("Candle Down", "#ef5350")
    rsi_color = st.color_picker("RSI", "#3949ab")
    macd_color = st.color_picker("MACD", "#00897b")
    macd_signal_color = st.color_picker("Signal", "#6d4c41")

# Save/Load configuration
with st.sidebar.expander("💾 Config", expanded=False):
    cfg = dict(
        index_choice=index_choice,
        custom_ticker=custom_ticker,
        period=period,
        interval=interval,
        chart_type=chart_type,
        ma_windows=ma_windows,
        use_ema=use_ema,
        rsi_window=int(rsi_window),
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
    )
    st.download_button("Download Config JSON", data=json.dumps(cfg, indent=2).encode(), file_name="observer_config.json")
    upl = st.file_uploader("Upload Config JSON", type="json")
    if upl is not None:
        try:
            new_cfg = json.load(upl)
            st.session_state.update(new_cfg)
            st.success("Config loaded. Reloading UI…")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to load config: {e}")

# ------------------------------
# Main Tabs
# ------------------------------

t1, t2 = st.tabs(["📉 Price Charts", "🧭 Breadth & Sentiment"])

# ------------------------------
# Tab 1 — Price Charts with overlays
# ------------------------------

with t1:
    df = fetch_price_history(primary_symbol, period, interval)
    if df.empty:
        st.error("No data for selected symbol/period.")
    else:
        close = df["Close"].copy()
        macd_line, signal_line = macd(close, macd_fast, macd_slow, macd_signal)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3], subplot_titles=(f"{primary_symbol} Price", "RSI / MACD"))

        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    increasing_line_color=candle_up, decreasing_line_color=candle_down, name="Candles"
                ), row=1, col=1
            )
        else:
            fig.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name="Close"), row=1, col=1)

        # MAs
        for w in ma_windows:
            ma_series = ema(close, w) if use_ema else sma(close, w)
            fig.add_trace(go.Scatter(x=df.index, y=ma_series, mode="lines", name=f"{'EMA' if use_ema else 'SMA'} {w}"), row=1, col=1)

        # RSI
        rsi_vals = rsi(close, rsi_window)
        fig.add_trace(go.Scatter(x=df.index, y=rsi_vals, mode="lines", name=f"RSI {rsi_window}", line=dict(color=rsi_color)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#b71c1c", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#1b5e20", row=2, col=1)

        # MACD overlay in same lower panel
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, mode="lines", name="MACD", line=dict(color=macd_color)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, mode="lines", name="Signal", line=dict(color=macd_signal_color)), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw price data (tail)", expanded=False):
            st.dataframe(df.tail(300))

# ------------------------------
# Tab 2 — Breadth & Sentiment
# ------------------------------

with t2:
    st.subheader("1) Put/Call Ratio (CBOE)")
    pcr_symbol = st.selectbox("PCR Symbol", ["^CPC (Total)", "^CPCE (Equity Only)"], index=0)
    pcr_map = {"^CPC (Total)": "^CPC", "^CPCE (Equity Only)": "^CPCE"}
    pcr_df = fetch_price_history(pcr_map[pcr_symbol], period="1y", interval="1d")
    if pcr_df.empty:
        st.info("PCR data unavailable.")
    else:
        pcr_fig = go.Figure()
        pcr_fig.add_trace(go.Scatter(x=pcr_df.index, y=pcr_df["Close"], mode="lines", name=pcr_map[pcr_symbol]))
        pcr_fig.update_layout(title=f"Put/Call Ratio — {pcr_symbol}")
        st.plotly_chart(pcr_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("2) % of Stocks Above 20-DMA")
    breadth_index = st.selectbox("Universe for breadth", ["S&P 500", "NYSE (upload list)"])
    max_cons = st.slider("Max constituents (performance cap)", 50, 500, 200, step=50)
    tickers: List[str] = []
    if breadth_index == "S&P 500":
        tickers = fetch_constituents_sp500()[:max_cons]
    else:
        upl = st.file_uploader("Upload CSV with a 'Ticker' column", type=["csv","xlsx","xls"])
        if upl is not None:
            try:
                if upl.name.endswith(".csv"):
                    tdf = pd.read_csv(upl)
                else:
                    tdf = pd.read_excel(upl)
                if "Ticker" in tdf.columns:
                    tickers = tdf["Ticker"].astype(str).str.upper().tolist()[:max_cons]
                else:
                    st.error("No 'Ticker' column found.")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    if tickers:
        prices = batch_download(tickers, period=period if period != "max" else "1y", interval="1d")
        pct20 = pct_above_20dma(prices)
        if pct20.empty:
            st.info("Could not compute % above 20-DMA.")
        else:
            fig20 = go.Figure()
            fig20.add_trace(go.Scatter(x=pct20.index, y=pct20.values, mode="lines", name="%>20DMA"))
            fig20.update_layout(title=f"% of {breadth_index} Constituents Above 20-DMA (n≈{len(tickers)})", yaxis_title="Percent")
            st.plotly_chart(fig20, use_container_width=True)

    st.markdown("---")
    st.subheader("3) Advance–Decline Line")
    if tickers:
        prices = batch_download(tickers, period=period if period != "max" else "1y", interval="1d")
        ad_line = advance_decline_line(prices)
        if ad_line.empty:
            st.info("Could not compute AD Line.")
        else:
            fig_ad = go.Figure()
            fig_ad.add_trace(go.Scatter(x=ad_line.index, y=ad_line.values, mode="lines", name="AD Line"))
            fig_ad.update_layout(title=f"Advance–Decline Line — {breadth_index}")
            st.plotly_chart(fig_ad, use_container_width=True)
    else:
        st.info("Provide a universe above to compute breadth (S&P 500 auto or upload tickers).")

    st.markdown("---")
    st.subheader("4) CNN Fear & Greed Index (best-effort)")
    fg_df = fetch_cnn_fear_greed()
    if fg_df.empty:
        st.info("Fear & Greed series unavailable right now.")
    else:
        fg_fig = go.Figure()
        fg_fig.add_trace(go.Scatter(x=fg_df.index, y=fg_df["Value"], mode="lines", name="F&G"))
        fg_fig.update_layout(title="CNN Fear & Greed Index", yaxis_title="Index (0–100)")
        st.plotly_chart(fg_fig, use_container_width=True)

# ------------------------------
# GitHub helper
# ------------------------------

with st.expander("🐙 Push this app to GitHub", expanded=False):
    st.markdown(
        """
        **Quick steps**
        1. Create a new repo on GitHub.
        2. Save this file as `stock_market_observer.py` and create a `requirements.txt` with the packages listed at the top.
        3. In your project folder, run:
        ```bash
        git init
        git add stock_market_observer.py requirements.txt
        git commit -m "Add Stock Market Observer"
        git branch -M main
        git remote add origin https://github.com/<your-username>/<your-repo>.git
        git push -u origin main
        ```
        """
    )

# ------------------------------
# Footer
# ------------------------------

st.caption("Data is provided as-is for informational purposes only. Not investment advice.")
