"""
Stock Market Visualizer — Streamlit App

Features
- Streamlit UI with sidebar controls
- Fetches data via yfinance for real-time & historical analysis
- Interactive Plotly candlestick charts with overlays:
  - Moving Averages (SMA/EMA)
  - Bollinger Bands
  - RSI (separate panel)
- Portfolio tracker (upload CSV/Excel)
- Financial ratios summary
- Correlation analysis across tickers
- Customizable visuals (timeframes, thresholds, colors)
- Export charts as PNG (via kaleido) and HTML
- Save & load user configurations (JSON). Optional query-param sharing.
- GitHub helper panel with copyable commands to push this file to a repo

Suggested requirements.txt (create alongside this file):
streamlit>=1.33
plotly>=5.18
pandas>=2.2
numpy>=1.26
yfinance>=0.2.50
openpyxl>=3.1  # for Excel upload
tabulate>=0.9
kaleido>=0.2.1  # for PNG export from Plotly

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st
import yfinance as yf

# ------------------------------
# Utilities & Config
# ------------------------------

st.set_page_config(
    page_title="Stock Market Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light/glass aesthetic
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .stMetric label { font-weight: 600 !important; }
    .stButton>button { border-radius: 12px; padding: 0.5rem 0.8rem; }
    .css-1kyxreq { gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helper indicator functions (no external TA deps)
# ------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    mid = sma(series, window)
    sd = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return upper, lower

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

# ------------------------------
# Data fetch
# ------------------------------

@st.cache_data(show_spinner=False, ttl=60)
def fetch_history(ticker: str, period: str, interval: str, auto_adjust: bool = True) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=auto_adjust, progress=False)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(show_spinner=False, ttl=60)
def fetch_info(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    info = {}
    # yfinance API evolves; try multiple access patterns
    for attr in ("get_info", "info"):
        try:
            if hasattr(t, attr):
                data = getattr(t, attr)
                info = data() if callable(data) else data
                if info:
                    break
        except Exception:
            pass
    # Also try fast_info (prices/market cap)
    try:
        fi = t.fast_info
        if isinstance(fi, dict):
            info.update(fi)
        else:
            info.update(dict(fi.__dict__))
    except Exception:
        pass
    return info or {}

@st.cache_data(show_spinner=False, ttl=300)
def fetch_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    t = yf.Ticker(ticker)
    out = {}
    try:
        out["financials"] = t.financials
    except Exception:
        out["financials"] = pd.DataFrame()
    try:
        out["balance_sheet"] = t.balance_sheet
    except Exception:
        out["balance_sheet"] = pd.DataFrame()
    try:
        out["cashflow"] = t.cashflow
    except Exception:
        out["cashflow"] = pd.DataFrame()
    return out

# ------------------------------
# Ratios (best-effort using available data)
# ------------------------------

def compute_ratios(ticker: str) -> pd.DataFrame:
    info = fetch_info(ticker)
    fin = fetch_financials(ticker)

    price = info.get("last_price") or info.get("lastPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap") or info.get("market_cap")
    eps_ttm = info.get("epsTrailingTwelveMonths") or info.get("trailingEps")
    pe = (price / eps_ttm) if price and eps_ttm not in (None, 0) else info.get("trailingPE")
    peg = info.get("pegRatio")
    pb = info.get("priceToBook") or info.get("priceToBookRatio")
    dividend_yield = info.get("dividendYield")
    profit_margin = info.get("profitMargins") or info.get("profitMargin")
    revenue_ttm = info.get("totalRevenue")

    # Try to derive margins from financial statements when possible
    try:
        fin_df = fin.get("financials", pd.DataFrame())
        if isinstance(fin_df, pd.DataFrame) and not fin_df.empty:
            # latest column
            col = fin_df.columns[0]
            revenue = float(fin_df.loc.get("Total Revenue", np.nan)) if hasattr(fin_df, "loc") else np.nan
            net = float(fin_df.loc.get("Net Income", np.nan)) if hasattr(fin_df, "loc") else np.nan
            if (not revenue or math.isnan(revenue)) and "Total Revenue" in fin_df.index:
                revenue = float(fin_df.at["Total Revenue", col])
            if (not net or math.isnan(net)) and "Net Income" in fin_df.index:
                net = float(fin_df.at["Net Income", col])
            if revenue and revenue != 0 and not math.isnan(revenue) and not math.isnan(net):
                profit_margin = net / revenue
    except Exception:
        pass

    rows = [
        ("Price", price),
        ("Market Cap", market_cap),
        ("EPS (TTM)", eps_ttm),
        ("P/E (TTM)", pe),
        ("PEG Ratio", peg),
        ("Price/Book", pb),
        ("Dividend Yield", dividend_yield),
        ("Profit Margin", profit_margin),
        ("Revenue (TTM)", revenue_ttm),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return df

# ------------------------------
# Plotting
# ------------------------------

def build_chart(
    df: pd.DataFrame,
    ma_windows: List[int],
    use_ema: bool,
    bb_window: int,
    bb_std: float,
    rsi_window: int,
    rsi_overbought: int,
    rsi_oversold: int,
    colors: Dict[str, str],
    title: str,
) -> go.Figure:
    if df.empty:
        return go.Figure()

    # Indicators
    close = df["Close"].copy()
    ma_vals = {}
    for w in ma_windows:
        ma_vals[w] = ema(close, w) if use_ema else sma(close, w)
    bb_upper, bb_lower = bollinger_bands(close, window=bb_window, num_std=bb_std)
    rsi_vals = rsi(close, window=rsi_window)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "RSI"),
    )

    candle = go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=colors["candle_up"], decreasing_line_color=colors["candle_down"],
        name="Candles"
    )
    fig.add_trace(candle, row=1, col=1)

    # MAs
    for w, series in ma_vals.items():
        fig.add_trace(
            go.Scatter(
                x=df.index, y=series, name=f"{'EMA' if use_ema else 'SMA'} {w}",
                mode="lines",
                line=dict(width=2, color=colors.get(f"ma_{w}", None)),
            ),
            row=1, col=1
        )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=bb_upper, name="BB Upper", mode="lines", line=dict(width=1, color=colors["bb"])),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=bb_lower, name="BB Lower", mode="lines", line=dict(width=1, color=colors["bb"]), fill=None),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=bb_upper, mode="lines", line=dict(width=0), showlegend=False,
            fill="tonexty", fillcolor=colors.get("bb_fill", "rgba(160,160,160,0.15)")
        ),
        row=1, col=1,
    )

    # RSI panel
    fig.add_trace(
        go.Scatter(x=df.index, y=rsi_vals, name=f"RSI {rsi_window}", mode="lines", line=dict(width=2, color=colors["rsi"])),
        row=2, col=1,
    )
    # RSI threshold lines
    for thr, nm in [(rsi_overbought, "Overbought"), (rsi_oversold, "Oversold")]:
        fig.add_hline(y=thr, line_dash="dash", line_width=1, line_color=colors.get("rsi_threshold", "gray"), row=2, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
    )
    return fig

# ------------------------------
# Configuration dataclass
# ------------------------------

@dataclass
class AppConfig:
    tickers: List[str]
    primary_ticker: str
    period: str
    interval: str
    ma_windows: List[int]
    use_ema: bool
    bb_window: int
    bb_std: float
    rsi_window: int
    rsi_overbought: int
    rsi_oversold: int
    colors: Dict[str, str]

DEFAULT_COLORS = {
    "candle_up": "#26a69a",
    "candle_down": "#ef5350",
    "bb": "#757575",
    "bb_fill": "rgba(117,117,117,0.12)",
    "rsi": "#3949ab",
    "rsi_threshold": "#9e9e9e",
    # moving averages will fallback to Plotly palette unless specified
}

DEFAULT_CONFIG = AppConfig(
    tickers=["AAPL", "MSFT", "NVDA"],
    primary_ticker="AAPL",
    period="6mo",
    interval="1d",
    ma_windows=[20, 50, 200],
    use_ema=False,
    bb_window=20,
    bb_std=2.0,
    rsi_window=14,
    rsi_overbought=70,
    rsi_oversold=30,
    colors=DEFAULT_COLORS.copy(),
)

# ------------------------------
# Sidebar — Controls
# ------------------------------

st.sidebar.title("📈 Stock Market Visualizer")
st.sidebar.caption("Analyze, visualize, and compare stocks with interactive charts.")

# Config load/upload
with st.sidebar.expander("💾 Load / Save Configuration", expanded=False):
    uploaded_cfg = st.file_uploader("Load config JSON", type=["json"], key="cfg_upl")
    if uploaded_cfg is not None:
        try:
            cfg_json = json.load(uploaded_cfg)
            st.session_state["app_config"] = AppConfig(**cfg_json)
            st.success("Configuration loaded.")
        except Exception as e:
            st.error(f"Failed to load config: {e}")

    cfg: AppConfig = st.session_state.get("app_config", DEFAULT_CONFIG)

    # Save current config button
    cfg_dict = asdict(cfg)
    cfg_byt = json.dumps(cfg_dict, indent=2).encode()
    st.download_button("Download current config", data=cfg_byt, file_name="stockviz_config.json", mime="application/json")

    # Optional: share via query params (lightweight)
    if st.button("Update URL with tickers/period"):
        st.experimental_set_query_params(
            tickers=",".join(cfg.tickers),
            primary=cfg.primary_ticker,
            period=cfg.period,
            interval=cfg.interval,
        )
        st.toast("URL updated with basic parameters.")

# Basic selections
with st.sidebar:
    tick_str = st.text_input("Tickers (comma-separated)", ", ".join(DEFAULT_CONFIG.tickers))
    tickers = [t.strip().upper() for t in tick_str.split(",") if t.strip()]
    primary = st.selectbox("Primary ticker for chart", options=tickers, index=0 if tickers else 0)

    period = st.selectbox(
        "Period",
        options=["1mo", "3mo", "6mo", "ytd", "1y", "5y", "max"],
        index=["1mo","3mo","6mo","ytd","1y","5y","max"].index(DEFAULT_CONFIG.period)
    )

    # Choose interval adaptively but allow override
    interval_choices = {
        "1mo": ["30m", "1h", "1d"],
        "3mo": ["1h", "1d"],
        "6mo": ["1d"],
        "ytd": ["1d"],
        "1y": ["1d", "1wk"],
        "5y": ["1d", "1wk", "1mo"],
        "max": ["1wk", "1mo"],
    }
    interval = st.selectbox("Interval", options=interval_choices.get(period, ["1d"]))

    st.markdown("---")

    st.subheader("Overlays & Indicators")
    ma_choice = st.multiselect("Moving Average windows", options=[5,10,20,50,100,200], default=DEFAULT_CONFIG.ma_windows)
    use_ema = st.toggle("Use EMA (off=SMA)", value=DEFAULT_CONFIG.use_ema)
    bb_window = st.number_input("Bollinger window", 5, 50, DEFAULT_CONFIG.bb_window)
    bb_std = st.slider("Bollinger std dev", 1.0, 3.0, float(DEFAULT_CONFIG.bb_std), 0.1)
    rsi_window = st.number_input("RSI window", 5, 50, DEFAULT_CONFIG.rsi_window)
    rsi_overbought = st.slider("RSI Overbought", 50, 90, DEFAULT_CONFIG.rsi_overbought)
    rsi_oversold = st.slider("RSI Oversold", 10, 50, DEFAULT_CONFIG.rsi_oversold)

    st.markdown("---")
    st.subheader("Colors")
    candle_up = st.color_picker("Candle Up", DEFAULT_COLORS["candle_up"])
    candle_down = st.color_picker("Candle Down", DEFAULT_COLORS["candle_down"])
    rsi_color = st.color_picker("RSI Line", DEFAULT_COLORS["rsi"])
    bb_color = st.color_picker("Bollinger Line", DEFAULT_COLORS["bb"])
    bb_fill = st.color_picker("Bollinger Fill", "#D0D0D0")
    ma_color_map = {}
    for w in ma_choice:
        ma_color_map[f"ma_{w}"] = st.color_picker(f"MA {w} Color", "")

    colors = {
        "candle_up": candle_up,
        "candle_down": candle_down,
        "rsi": rsi_color,
        "bb": bb_color,
        "bb_fill": f"rgba({int(bb_fill[1:3],16)},{int(bb_fill[3:5],16)},{int(bb_fill[5:7],16)},0.18)" if len(bb_fill)==7 else DEFAULT_COLORS["bb_fill"],
        "rsi_threshold": DEFAULT_COLORS["rsi_threshold"],
        **ma_color_map,
    }

    # Update session config
    st.session_state["app_config"] = AppConfig(
        tickers=tickers,
        primary_ticker=primary,
        period=period,
        interval=interval,
        ma_windows=ma_choice,
        use_ema=use_ema,
        bb_window=int(bb_window),
        bb_std=float(bb_std),
        rsi_window=int(rsi_window),
        rsi_overbought=int(rsi_overbought),
        rsi_oversold=int(rsi_oversold),
        colors=colors,
    )

# ------------------------------
# Main Tabs
# ------------------------------

t1, t2, t3, t4 = st.tabs(["📊 Chart", "📁 Portfolio", "🧮 Ratios", "🔗 Correlation"])

# ------------------------------
# Tab 1: Charting
# ------------------------------

with t1:
    cfg: AppConfig = st.session_state["app_config"]

    if not cfg.tickers:
        st.info("Add at least one ticker in the sidebar to begin.")
        st.stop()

    df = fetch_history(cfg.primary_ticker, cfg.period, cfg.interval)
    if df.empty:
        st.error("No data returned. Try a different period/interval or check the ticker symbol.")
    else:
        title = f"{cfg.primary_ticker} — {cfg.period} @ {cfg.interval}"
        fig = build_chart(
            df=df,
            ma_windows=cfg.ma_windows,
            use_ema=cfg.use_ema,
            bb_window=cfg.bb_window,
            bb_std=cfg.bb_std,
            rsi_window=cfg.rsi_window,
            rsi_overbought=cfg.rsi_overbought,
            rsi_oversold=cfg.rsi_oversold,
            colors=cfg.colors,
            title=title,
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # Export controls
        colA, colB = st.columns(2)
        with colA:
            try:
                png_bytes = fig.to_image(format="png", scale=2)  # kaleido
                st.download_button("⬇️ Download PNG", data=png_bytes, file_name=f"{cfg.primary_ticker}_{cfg.period}_{cfg.interval}.png", mime="image/png")
            except Exception as e:
                st.warning("PNG export requires the 'kaleido' package. Add it to requirements.txt.")
        with colB:
            html_str = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            st.download_button("⬇️ Download HTML", data=html_str, file_name=f"{cfg.primary_ticker}_{cfg.period}_{cfg.interval}.html", mime="text/html")

        with st.expander("Raw data (preview)", expanded=False):
            st.dataframe(df.tail(500))

# ------------------------------
# Tab 2: Portfolio
# ------------------------------

with t2:
    st.subheader("Upload Portfolio File")
    st.caption("CSV or Excel with columns: Ticker, Shares, CostBasis (optional)")

    upl = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
    if upl is not None:
        try:
            if upl.name.lower().endswith(".csv"):
                pf = pd.read_csv(upl)
            else:
                pf = pd.read_excel(upl)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            pf = pd.DataFrame()

        if not pf.empty:
            # Normalize columns
            pf.columns = [c.strip().title() for c in pf.columns]
            if "Ticker" not in pf.columns or "Shares" not in pf.columns:
                st.error("Missing required columns: 'Ticker' and 'Shares'.")
            else:
                pf["Ticker"] = pf["Ticker"].astype(str).str.upper()
                pf["Shares"] = pd.to_numeric(pf["Shares"], errors="coerce").fillna(0.0)
                if "Costbasis" in [c.lower() for c in pf.columns]:
                    # Support both CostBasis and Costbasis capitalization
                    for c in pf.columns:
                        if c.lower() == "costbasis":
                            pf.rename(columns={c: "CostBasis"}, inplace=True)
                            break
                if "Costbasis" not in [c.lower() for c in pf.columns] and "CostBasis" not in pf.columns:
                    pf["CostBasis"] = np.nan

                rows = []
                for _, r in pf.iterrows():
                    tkr = r["Ticker"]
                    sh = float(r["Shares"])
                    cb = r.get("CostBasis", np.nan)
                    info = fetch_info(tkr)
                    price = info.get("last_price") or info.get("regularMarketPrice") or info.get("currentPrice")
                    if price is None:
                        # fallback to quick history
                        hist = fetch_history(tkr, period="5d", interval="1d")
                        price = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
                    mv = sh * price if price and not math.isnan(price) else np.nan
                    rows.append((tkr, sh, price, mv, cb))

                rpt = pd.DataFrame(rows, columns=["Ticker", "Shares", "Price", "Market Value", "CostBasis"])
                rpt["P/L ($)"] = rpt.apply(lambda x: (x["Price"] - x["CostBasis"]) * x["Shares"] if not np.isnan(x["CostBasis"]) else np.nan, axis=1)
                total_mv = rpt["Market Value"].sum(skipna=True)
                rpt["Weight"] = rpt["Market Value"] / total_mv if total_mv else np.nan

                st.markdown("### Portfolio Summary")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Positions", len(rpt))
                kpi2.metric("Market Value", f"${total_mv:,.0f}")
                pl_total = rpt["P/L ($)"].sum(skipna=True)
                kpi3.metric("Total P/L", f"${pl_total:,.0f}", delta=None)

                st.dataframe(rpt, use_container_width=True)

                # Download enriched portfolio
                out_csv = rpt.to_csv(index=False).encode()
                st.download_button("Download Enriched CSV", data=out_csv, file_name="portfolio_enriched.csv", mime="text/csv")
        else:
            st.info("Upload a portfolio file to see analytics.")
    else:
        st.info("Upload a portfolio file to see analytics.")

# ------------------------------
# Tab 3: Ratios
# ------------------------------

with t3:
    st.subheader("Financial Ratios (best-effort)")
    ticker_for_ratios = st.selectbox("Select ticker", options=st.session_state["app_config"].tickers, index=0)
    ratios_df = compute_ratios(ticker_for_ratios)
    st.dataframe(ratios_df, use_container_width=True)

# ------------------------------
# Tab 4: Correlation
# ------------------------------

with t4:
    st.subheader("Correlation Analysis")
    st.caption("Computes correlations of daily returns over the selected period.")
    cfg: AppConfig = st.session_state["app_config"]
    corr_tickers = st.multiselect("Tickers to include", options=cfg.tickers, default=cfg.tickers)

    if corr_tickers:
        prices = []
        valid = []
        for tkr in corr_tickers:
            h = fetch_history(tkr, cfg.period, cfg.interval)
            if not h.empty:
                prices.append(h["Close"].rename(tkr))
                valid.append(tkr)
        if prices:
            price_df = pd.concat(prices, axis=1).dropna(how="all")
            rets = price_df.pct_change().dropna(how="all")
            corr = rets.corr()

            fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorbar=dict(title="ρ")))
            fig_corr.update_layout(margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_corr, use_container_width=True)

            st.download_button(
                "Download correlation CSV",
                data=corr.to_csv().encode(),
                file_name="correlation_matrix.csv",
                mime="text/csv",
            )
        else:
            st.info("No valid price data for selected tickers.")
    else:
        st.info("Select at least one ticker.")

# ------------------------------
# GitHub helper (non-executing guidance within the app UI)
# ------------------------------

with st.expander("🐙 Push this app to GitHub", expanded=False):
    st.markdown(
        """
        **Quick steps**
        1. Create a new GitHub repo (public or private).
        2. Save this file as `streamlit_app.py` (and `requirements.txt` as noted at the top).
        3. Run the commands below in your project folder (replace placeholders):
        ```bash
        git init
        git add streamlit_app.py requirements.txt
        git commit -m "Add Stock Market Visualizer"
        git branch -M main
        git remote add origin https://github.com/<your-username>/<your-repo>.git
        git push -u origin main
        ```
        Optionally add a minimal **Streamlit Cloud** `packages` file or deploy to other platforms.
        """
    )

# ------------------------------
# Footer
# ------------------------------

st.caption("Built with ❤️ using Streamlit, Plotly, and yfinance. Data is provided as-is for informational purposes only.")
