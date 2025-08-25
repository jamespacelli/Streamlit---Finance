import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import json
from datetime import datetime

# --------------------------
# Helper functions
# --------------------------
def download_data(ticker, start, end, interval="1d"):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    return df

def add_moving_averages(df, windows=[20, 50]):
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(window=w).mean()
        df[f"EMA{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df

def add_bollinger_bands(df, window=20):
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_upper"] = sma + 2*std
    df["BB_lower"] = sma - 2*std
    return df

def add_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Stock Market Visualizer", layout="wide")
st.title("📈 Stock Market Visualizer")

# Sidebar config
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter stock ticker", "AAPL")
start_date = st.sidebar.date_input("Start date", datetime(2022,1,1))
end_date = st.sidebar.date_input("End date", datetime.today())
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# Save/load config
if st.sidebar.button("💾 Save Config"):
    config = {"ticker": ticker, "start_date": str(start_date), "end_date": str(end_date), "interval": interval}
    with open("config.json", "w") as f:
        json.dump(config, f)
    st.sidebar.success("Config saved!")

if st.sidebar.button("📂 Load Config"):
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            ticker = config["ticker"]
            start_date = pd.to_datetime(config["start_date"])
            end_date = pd.to_datetime(config["end_date"])
            interval = config["interval"]
        st.sidebar.success("Config loaded!")
    except FileNotFoundError:
        st.sidebar.error("No saved config found.")

# Fetch data
df = download_data(ticker, start_date, end_date, interval)

if df.empty:
    st.error("No data found. Try different dates or ticker.")
    st.stop()

# Indicators
df = add_moving_averages(df)
df = add_bollinger_bands(df)
df = add_rsi(df)

# --------------------------
# Candlestick chart
# --------------------------
st.subheader(f"Candlestick chart for {ticker}")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="Candlestick"
))

# Add SMA & EMA
for col in [c for c in df.columns if "SMA" in c or "EMA" in c]:
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

# Bollinger Bands
fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(color="gray", dash="dot"), name="BB Upper"))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(color="gray", dash="dot"), name="BB Lower"))

fig.update_layout(title=f"{ticker} Candlestick with Indicators", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# RSI
# --------------------------
st.subheader("Relative Strength Index (RSI)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"))
fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"))
fig_rsi.update_layout(title="RSI Indicator")
st.plotly_chart(fig_rsi, use_container_width=True)

# --------------------------
# Extra visualizations
# --------------------------
st.header("📊 Additional Visualizations")

# Daily Returns Histogram
df["Daily Return"] = df["Close"].pct_change()
fig_ret = go.Figure()
fig_ret.add_trace(go.Histogram(x=df["Daily Return"].dropna(), nbinsx=50))
fig_ret.update_layout(title="Daily Returns Distribution")
st.plotly_chart(fig_ret, use_container_width=True)

# Cumulative Returns
cum_returns = (1 + df["Daily Return"].fillna(0)).cumprod() - 1
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=df.index, y=cum_returns, mode="lines", name="Cumulative Return"))
fig_cum.update_layout(title="Cumulative Return Over Time")
st.plotly_chart(fig_cum, use_container_width=True)

# Volume Analysis
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
fig_vol.update_layout(title="Trading Volume")
st.plotly_chart(fig_vol, use_container_width=True)

# MACD
st.subheader("MACD (Moving Average Convergence Divergence)")
exp1 = df["Close"].ewm(span=12, adjust=False).mean()
exp2 = df["Close"].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=macd, mode="lines", name="MACD"))
fig_macd.add_trace(go.Scatter(x=df.index, y=signal, mode="lines", name="Signal Line"))
fig_macd.update_layout(title="MACD Indicator")
st.plotly_chart(fig_macd, use_container_width=True)

# Correlation Heatmap
st.subheader("Correlation Heatmap (Multiple Tickers)")
tickers = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN")
tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
if tickers_list:
    df_corr = yf.download(tickers_list, period="1y")["Close"].pct_change().dropna()
    corr_matrix = df_corr.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Stock Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------
# GitHub Instructions
# --------------------------
st.sidebar.markdown("### 🚀 Push to GitHub")
st.sidebar.code("""
git init
git add .
git commit -m "Initial commit - Stock Market Visualizer"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
""")
