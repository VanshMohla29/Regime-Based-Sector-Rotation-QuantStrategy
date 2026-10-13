"""
India HMM Regime Detector — Live Paper Trading Portfolio (Frozen Model)
========================================================================
Uses the FROZEN trained model artifacts (hmm_model.pkl, hmm_scaler.pkl,
label_map.pkl, learned_sector_mix.json) to predict regimes on new market
data and simulate a live portfolio with actual NSE ETF end-of-day prices.

NO model retraining occurs. All data after the freeze date is treated
as pure out-of-sample testing data.

The portfolio:
  - Starts with ₹10,00,000 (configurable)
  - Allocates across 8 NSE Sector ETFs per the frozen learned_sector_mix
  - Rebalances ONLY when the regime changes (with MIN_HOLD lockout)
  - Tracks positions, NAV, P&L, and drawdown daily
  - Applies realistic transaction costs (STT + slippage)

Run:
    python paper_portfolio.py

Outputs:
    fig9_paper_portfolio.png  — Portfolio NAV, positions, returns, drawdown
    paper_portfolio_history.csv — Daily portfolio state log
"""

import os
import sys
import json
import pickle
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import yfinance as yf

warnings.filterwarnings('ignore')

try:
    from regime_detector_v2 import fetch_india_macro, fetch_india_wpi, impute_macro_signals
except ImportError:
    fetch_india_macro = None
    fetch_india_wpi = None
    impute_macro_signals = None

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

PROJECT_DIR   = Path(__file__).parent.resolve()
INITIAL_CAPITAL = 10_00_000  # ₹10 lakh

# ── Indian Securities Transaction Costs (Equity Delivery, NSE) ──────
# All rates as per SEBI / NSE / Income Tax regulations.
# Assumes discount broker (Zerodha/Groww): ₹0 brokerage on delivery.

# STT (Securities Transaction Tax) — charged on BOTH buy & sell for delivery
TC_STT_BUY   = 0.001      # 0.10% of buy turnover
TC_STT_SELL  = 0.001      # 0.10% of sell turnover

# Exchange Transaction Charges (NSE equity delivery)
TC_EXCHANGE  = 0.0000297  # 0.00297% of turnover

# SEBI Turnover Fee
TC_SEBI      = 0.000001   # 0.0001% (₹10 per crore)

# GST — 18% on (brokerage + exchange charges + SEBI fee)
# With ₹0 brokerage: GST = 18% × (TC_EXCHANGE + TC_SEBI)
TC_GST_RATE  = 0.18

# Stamp Duty — charged on BUY side only (equity delivery, uniform since Jul 2020)
TC_STAMP     = 0.00015    # 0.015% of buy value

# DP (Depository Participant) Charges — flat per-scrip fee on sell side
DP_CHARGE_PER_SCRIP = 15.93  # ₹15.93 per scrip per sell txn (CDSL via Zerodha)

# Market impact / slippage estimate
TC_SLIPPAGE  = 0.0005     # 0.05% estimated execution slippage

# Capital Gains Tax Rates (Budget 2024 onwards)
STCG_RATE    = 0.20       # 20% on Short-Term Capital Gains (holding < 12 months)
LTCG_RATE    = 0.125      # 12.5% on Long-Term Capital Gains (holding ≥ 12 months)
LTCG_EXEMPT  = 1_25_000   # ₹1,25,000 annual LTCG exemption


def compute_buy_charges(turnover):
    """Total charges for buying equity delivery on NSE."""
    stt      = turnover * TC_STT_BUY
    exchange = turnover * TC_EXCHANGE
    sebi     = turnover * TC_SEBI
    gst      = (exchange + sebi) * TC_GST_RATE
    stamp    = turnover * TC_STAMP
    slippage = turnover * TC_SLIPPAGE
    total    = stt + exchange + sebi + gst + stamp + slippage
    return total


def compute_sell_charges(turnover, n_scrips=1):
    """Total charges for selling equity delivery on NSE."""
    stt      = turnover * TC_STT_SELL
    exchange = turnover * TC_EXCHANGE
    sebi     = turnover * TC_SEBI
    gst      = (exchange + sebi) * TC_GST_RATE
    dp       = DP_CHARGE_PER_SCRIP * n_scrips
    slippage = turnover * TC_SLIPPAGE
    total    = stt + exchange + sebi + gst + dp + slippage
    return total

# Anti-whipsaw minimum holding period
MIN_HOLD_DAYS = 5

# Capital protection parameters (-12% drawdown stop with T+0 execution + 20 SMA re-entry)
STOP_LOSS_DD       = -0.12   # -12% portfolio drawdown stop to 100% Cash / Liquid (T+0 execution)
STOP_REENTRY_MA    = 20      # 20-day Simple Moving Average on NIFTY 50
STOP_REENTRY_TYPE  = 'SMA'   # 'EMA' or 'SMA'
STOP_COOLDOWN_DAYS = 5       # Minimum 5 trading days in Cash before checking re-entry

REGIME_COLORS = {
    'Bull':     '#10b981',
    'Bear':     '#ef4444',
    'HighVol':  '#f59e0b',
    'Sideways': '#3b82f6',
}

# ETF tickers on NSE (yfinance format)
ETF_TICKERS = {
    'BANKBEES':  'BANKBEES.NS',
    'ITBEES':    'ITBEES.NS',
    'PHARMABEES':'PHARMABEES.NS',
    'AUTOBEES':  'AUTOBEES.NS',
    'METALIETF': 'METALIETF.NS',
    'MOREALTY':  'MOREALTY.NS',
    'CPSEETF':   'CPSEETF.NS',
    'INFRABEES': 'INFRABEES.NS',
}

ETF_DISPLAY = {
    'BANKBEES':  'Bank',
    'ITBEES':    'IT',
    'PHARMABEES':'Pharma',
    'AUTOBEES':  'Auto',
    'METALIETF': 'Metal',
    'MOREALTY':  'Realty',
    'CPSEETF':   'CPSE',
    'INFRABEES': 'Infra',
}

# HMM Feature columns (must match training pipeline exactly)
FEATURE_COLS = [
    'ret_5d_pct',
    'vol_60d',
    'price_vs_ma200',
    'drawdown',
    'drawdown_diff20d',
    'vix_vs_ma20',
    'yield_curve',
    'cpi_yoy',
    'iip_yoy',
    'wpi_yoy',
    'real_rate',
]


# ══════════════════════════════════════════════════════════════════════
# 1. LOAD FROZEN MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════

def load_frozen_artifacts():
    """Load the frozen HMM model, scaler, label map, and sector mix."""
    artifacts = {}
    for name, fname in [('model', 'hmm_model.pkl'), ('scaler', 'hmm_scaler.pkl'),
                         ('label_map', 'label_map.pkl')]:
        path = PROJECT_DIR / fname
        if not path.exists():
            print(f"❌ {fname} not found in {PROJECT_DIR}. Run regime_detector_v2.py first.")
            sys.exit(1)
        with open(path, 'rb') as f:
            artifacts[name] = pickle.load(f)

    mix_path = PROJECT_DIR / 'learned_sector_mix.json'
    if mix_path.exists():
        with open(mix_path) as f:
            artifacts['sector_mix'] = json.load(f)
    else:
        print("❌ learned_sector_mix.json not found.")
        sys.exit(1)

    # Determine freeze date (must be persistent so OOS portfolio accumulates over time)
    hist_path = PROJECT_DIR / 'regime_history_v2.csv'
    freeze_file = PROJECT_DIR / 'freeze_date.txt'
    env_freeze = os.getenv('PORTFOLIO_FREEZE_DATE')

    if freeze_file.exists():
        try:
            f_str = freeze_file.read_text().strip()
            artifacts['freeze_date'] = pd.Timestamp(f_str).normalize()
        except Exception:
            artifacts['freeze_date'] = pd.Timestamp('2026-09-11').normalize()
    elif env_freeze:
        artifacts['freeze_date'] = pd.Timestamp(env_freeze).normalize()
        try:
            freeze_file.write_text(artifacts['freeze_date'].strftime('%Y-%m-%d'))
        except Exception:
            pass
    elif hist_path.exists():
        # Default baseline training cutoff: 11 Sep 2026 (first OOS trading session: 15 Sep 2026)
        artifacts['freeze_date'] = pd.Timestamp('2026-09-11').normalize()
        try:
            freeze_file.write_text(artifacts['freeze_date'].strftime('%Y-%m-%d'))
        except Exception:
            pass
    else:
        artifacts['freeze_date'] = pd.Timestamp('2026-09-11').normalize()

    # Last trained regime at freeze date
    if hist_path.exists():
        hist = pd.read_csv(hist_path, index_col=0, parse_dates=True)
        pre_freeze = hist[hist.index <= artifacts['freeze_date']]
        if not pre_freeze.empty:
            artifacts['last_regime'] = pre_freeze.iloc[-1]['Regime']
        else:
            artifacts['last_regime'] = hist.iloc[-1]['Regime']
    else:
        artifacts['last_regime'] = 'Sideways'

    print(f"✓ Freeze date: {artifacts['freeze_date'].strftime('%d %b %Y')}")
    print(f"  Last trained regime: {artifacts['last_regime']}")
    print(f"✓ Loaded frozen model ({artifacts['model'].n_components}-state HMM)")
    return artifacts


# ══════════════════════════════════════════════════════════════════════
# 2. DATA FETCHING (Market + Macro + ETF Prices)
# ══════════════════════════════════════════════════════════════════════

def _load_env_file():
    """Load .env for any needed API keys."""
    env_path = PROJECT_DIR / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, val = line.partition('=')
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def fetch_macro_fred():
    """Fetch official India macro data from FRED."""
    try:
        import urllib.request, io
        FRED_KEY = os.getenv('FRED_API_KEY', '')
        series_ids = {
            'GSecYield10': 'INDIRLTLT01STM',
            'GSecYield2':  'INDIR3TIB01STM',
            'CPI_YoY':     'INDCPIALLMINMEI',
            'IIP_YoY':     'INDPRMNTO01GYSAM',
            'RepoRate':    'IRSTCI01INM156N',
        }
        frames = {}
        for name, series_id in series_ids.items():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            try:
                data = pd.read_csv(url, index_col=0, parse_dates=True)
                data.columns = [name]
                data[name] = pd.to_numeric(data[name], errors='coerce')
                frames[name] = data[name]
            except Exception:
                continue
        if frames:
            macro = pd.DataFrame(frames).ffill()
            return macro
    except Exception:
        pass
    return None


def fetch_wpi():
    """Fetch official India WPI from eaindustry.nic.in."""
    try:
        # Try 2022-23 base series first, then 2011-12
        for base, prefix in [('2223', '2223'), ('1112', '1112')]:
            try:
                now = datetime.now()
                year = now.year
                month = now.month
                wpi_values = {}
                for yr in range(2020, year + 1):
                    for mo in range(1, 13):
                        if yr == year and mo > month:
                            break
                        ym = f"{yr}{mo:02d}"
                        url = f"https://eaindustry.nic.in/download_data_{prefix}/monthly_index_{ym}.xls"
                        try:
                            tbl = pd.read_html(url, header=0)
                            if tbl:
                                df0 = tbl[0]
                                for col in df0.columns:
                                    if 'all commodities' in str(col).lower() or 'general index' in str(col).lower():
                                        vals = pd.to_numeric(df0[col], errors='coerce').dropna()
                                        if len(vals) > 0:
                                            wpi_values[pd.Timestamp(yr, mo, 1)] = float(vals.iloc[0])
                                            break
                        except Exception:
                            continue
                if wpi_values:
                    wpi_s = pd.Series(wpi_values).sort_index()
                    wpi_yoy = wpi_s.pct_change(12) * 100
                    return wpi_yoy.dropna()
            except Exception:
                continue
    except Exception:
        pass
    return None


def fetch_full_market_data(start_date="2015-01-01"):
    """
    Fetch the same market + macro dataset as the main pipeline.
    Required to compute rolling features (200-day MA, volatility, etc.)
    """
    print(f"\n[Paper Portfolio] Fetching market data (since {start_date})...")
    tickers = {
        'NIFTY': '^NSEI',
        'VIX':   '^INDIAVIX',
    }

    df_raw = yf.download(list(tickers.values()), start=start_date, auto_adjust=True, progress=False)['Close']
    df = pd.DataFrame(index=df_raw.index)
    df['NIFTY']   = df_raw['^NSEI']
    df['Returns'] = np.log(df['NIFTY'] / df['NIFTY'].shift(1))
    df['VIX']     = df_raw['^INDIAVIX'].ffill().bfill()

    # Official India Macro (MoSPI + RBI + FRED)
    macro = fetch_india_macro() if fetch_india_macro is not None else fetch_macro_fred()
    if macro is not None:
        macro.index = macro.index + pd.DateOffset(months=1)
        macro_daily = macro.reindex(df.index, method='ffill')
        df['GSecYield10'] = macro_daily['GSecYield10']
        df['GSecYield2']  = macro_daily.get('GSecYield2', macro_daily.get('GSecYield10', np.nan) - 1.5)
        df['CPI']         = macro_daily['CPI_YoY']
        df['IIP']         = macro_daily['IIP_YoY']
        df['RepoRate']    = macro_daily['RepoRate']
    else:
        df['GSecYield10'] = np.nan
        df['GSecYield2']  = np.nan
        df['CPI']         = np.nan
        df['IIP']         = np.nan
        df['RepoRate']    = np.nan

    wpi_data = fetch_india_wpi() if fetch_india_wpi is not None else fetch_wpi()
    if wpi_data is not None:
        wpi_data.index = wpi_data.index + pd.DateOffset(months=1)
        wpi_daily = wpi_data.reindex(df.index, method='ffill')
        df['WPI'] = wpi_daily
    else:
        df['WPI'] = np.nan

    # Dynamic KNN Imputation anchored on Ret_1d, Vix, drawdown (k=5)
    if impute_macro_signals is not None:
        df = impute_macro_signals(df)
    else:
        df['YieldCurve'] = df['GSecYield10'] - df['GSecYield2']
        df['RealRate']   = df['RepoRate'] - df['CPI']

    df.dropna(subset=['NIFTY', 'Returns'], inplace=True)
    print(f"✓ Loaded {len(df)} trading days ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Latest Price: {df['NIFTY'].iloc[-1]:,.2f} | VIX: {df['VIX'].iloc[-1]:.2f} | 10Y Yield: {df['GSecYield10'].iloc[-1]:.2f}% | CPI (MoSPI): {df['CPI'].iloc[-1]:.2f}% | IIP (MoSPI): {df['IIP'].iloc[-1]:.2f}% | Repo (RBI): {df['RepoRate'].iloc[-1]:.2f}% | WPI (DPIIT): {df['WPI'].iloc[-1]:.2f}%")
    return df


def fetch_etf_prices(start_date="2024-01-01"):
    """Fetch actual end-of-day ETF prices from NSE via yfinance."""
    print(f"\n[Paper Portfolio] Fetching actual ETF end-of-day prices...")
    tickers = list(ETF_TICKERS.values())
    raw = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)['Close']

    etf_prices = pd.DataFrame(index=raw.index)
    for etf_name, ticker in ETF_TICKERS.items():
        if ticker in raw.columns:
            etf_prices[etf_name] = raw[ticker]
        else:
            print(f"  ⚠ {etf_name} ({ticker}) not available — will use NaN")
            etf_prices[etf_name] = np.nan

    etf_prices = etf_prices.ffill().dropna(how='all')
    available = etf_prices.columns[etf_prices.notna().any()].tolist()
    print(f"✓ ETF prices loaded for: {', '.join(available)} ({len(etf_prices)} sessions)")
    return etf_prices


# ══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (Mirrors regime_detector_v2.engineer_features)
# ══════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Replicate the exact feature engineering from the training pipeline."""
    feat = pd.DataFrame(index=df.index)

    feat['ret_1d']  = df['Returns']
    feat['ret_5d']  = df['Returns'].rolling(5).sum()
    feat['ret_20d'] = df['Returns'].rolling(20).sum()
    feat['ret_1d_pct'] = (df['NIFTY'] / df['NIFTY'].shift(1) - 1) * 100
    feat['ret_5d_pct'] = (df['NIFTY'] / df['NIFTY'].shift(5) - 1) * 100
    feat['ret_20d_pct'] = (df['NIFTY'] / df['NIFTY'].shift(20) - 1) * 100

    feat['vol_5d']  = df['Returns'].rolling(5).std() * np.sqrt(252)
    feat['vol_20d'] = df['Returns'].rolling(20).std() * np.sqrt(252)
    feat['vol_60d'] = df['Returns'].rolling(60).std() * np.sqrt(252)
    feat['vol_ratio_5_20'] = feat['vol_5d'] / (feat['vol_20d'] + 1e-6)

    feat['parkinson_vol'] = df['VIX'] / 100.0

    for window in [20, 50, 200]:
        feat[f'price_vs_ma{window}'] = df['NIFTY'] / df['NIFTY'].rolling(window).mean() - 1

    feat['ma20_vs_ma50']  = (df['NIFTY'].rolling(20).mean() / df['NIFTY'].rolling(50).mean() - 1)
    feat['ma50_vs_ma200'] = (df['NIFTY'].rolling(50).mean() / df['NIFTY'].rolling(200).mean() - 1)

    delta = df['Returns']
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-6)
    feat['rsi14'] = 100 - (100 / (1 + rs))

    rolling_max      = df['NIFTY'].cummax()
    feat['drawdown'] = (df['NIFTY'] - rolling_max) / rolling_max
    feat['drawdown_diff20d'] = feat['drawdown'] - feat['drawdown'].shift(20)
    feat['drawdown_diff50d'] = feat['drawdown'] - feat['drawdown'].shift(50)
    feat['drawdown_diff100d'] = feat['drawdown'] - feat['drawdown'].shift(100)
    feat['drawdown_diff200d'] = feat['drawdown'] - feat['drawdown'].shift(200)
    feat['drawdown_diff300d'] = feat['drawdown'] - feat['drawdown'].shift(300)

    feat['vix']          = df['VIX']
    feat['vix_diff_20']  = df['VIX'] - df['VIX'].shift(20)
    feat['vix_vs_ma20']  = df['VIX'] / df['VIX'].rolling(20).mean() - 1
    feat['vix_roc_5']    = df['VIX'].pct_change(5)

    feat['yield_curve'] = df['YieldCurve']
    feat['rate_spread'] = df['GSecYield10'] - df['RepoRate']
    feat['cpi_yoy']     = df['CPI']
    feat['iip_yoy']     = df['IIP']
    feat['wpi_yoy']     = df['WPI']
    feat['real_rate']   = (df['RealRate'] - df['RealRate'].shift(252)).bfill()

    clean = feat.dropna()
    return clean


# ══════════════════════════════════════════════════════════════════════
# 4. REGIME PREDICTION WITH FROZEN MODEL
# ══════════════════════════════════════════════════════════════════════

def predict_regimes_frozen(model, scaler, label_map, feat):
    """
    Use the FROZEN model to predict regimes on new data.
    Uses online forward decoding (causal, no lookahead).
    """
    X = feat[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    # Online forward decode (day-by-day, causal)
    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat  = np.log(model.transmat_  + 1e-300)
    framelogprob  = model._compute_log_likelihood(X_scaled)

    n_samples = X_scaled.shape[0]
    states = np.zeros(n_samples, dtype=int)
    posteriors = np.zeros((n_samples, model.n_components))

    # Day 0
    log_alpha = log_startprob + framelogprob[0]
    log_alpha -= np.logaddexp.reduce(log_alpha)
    states[0] = np.argmax(log_alpha)
    posteriors[0] = np.exp(log_alpha - np.logaddexp.reduce(log_alpha))

    for t in range(1, n_samples):
        log_alpha = np.logaddexp.reduce(
            log_alpha[:, None] + log_transmat, axis=0
        ) + framelogprob[t]
        log_alpha -= np.logaddexp.reduce(log_alpha)
        states[t] = np.argmax(log_alpha)
        posteriors[t] = np.exp(log_alpha - np.logaddexp.reduce(log_alpha))

    # Map states to regime names
    decoded_labels = [label_map.get(s, {'name': 'Sideways'})['name'] for s in states]
    decoded_labels = smooth_regimes_causal(decoded_labels)

    # Build posteriors DataFrame with regime names
    regime_posteriors = pd.DataFrame(index=feat.index)
    for s, info in label_map.items():
        regime_posteriors[info['name']] = posteriors[:, s]
    for reg in ['Bull', 'Bear', 'HighVol', 'Sideways']:
        if reg not in regime_posteriors:
            regime_posteriors[reg] = 0.0

    return decoded_labels, regime_posteriors


def smooth_regimes_causal(regime_series, min_hold=MIN_HOLD_DAYS):
    """Causal anti-whipsaw filter (mirrors regime_detector_v2.py exactly)."""
    if len(regime_series) == 0:
        return []
    smoothed = [regime_series[0]]
    current = regime_series[0]
    hold_count = 1
    for i in range(1, len(regime_series)):
        if regime_series[i] != current:
            if hold_count >= min_hold:
                current = regime_series[i]
                hold_count = 1
            else:
                hold_count += 1
        else:
            hold_count += 1
        smoothed.append(current)
    return smoothed


# ══════════════════════════════════════════════════════════════════════
# 5. PAPER TRADING ENGINE
# ══════════════════════════════════════════════════════════════════════

def run_paper_trading(df, feat, regimes, regime_posteriors, etf_prices,
                      sector_mix, freeze_date, initial_capital=INITIAL_CAPITAL):
    """
    Simulate a paper trading portfolio from freeze_date onwards.

    Returns a DataFrame with daily portfolio state:
      Date, Regime, NAV, Cash, Equity_Value, Drawdown, Daily_Return,
      per-ETF positions (units & value), rebalance flags
    """
    # Filter to dates AFTER freeze_date
    oos_dates = feat.index[feat.index > freeze_date]
    if len(oos_dates) == 0:
        print("⚠ No out-of-sample dates found yet. Portfolio will initialize at freeze date.")
        oos_dates = feat.index[feat.index >= freeze_date]

    # Align ETF prices with OOS dates
    common_dates = oos_dates.intersection(etf_prices.index)
    if len(common_dates) == 0:
        print("❌ No overlapping dates between market data and ETF prices.")
        return pd.DataFrame()

    print(f"\n[Paper Trading] Simulating from {common_dates[0].strftime('%d %b %Y')} to {common_dates[-1].strftime('%d %b %Y')} ({len(common_dates)} sessions)")

    etf_names = list(ETF_TICKERS.keys())
    available_etfs = [e for e in etf_names if e in etf_prices.columns and etf_prices[e].notna().any()]

    # Initialize portfolio
    portfolio_nav = initial_capital
    cash = initial_capital
    holdings = {etf: 0 for etf in available_etfs}          # whole units only (int)
    cost_basis = {etf: 0.0 for etf in available_etfs}      # weighted avg buy price
    realized_gains = 0.0                                    # cumulative realized P&L
    total_charges_paid = 0.0                                # cumulative txn costs
    peak_nav = initial_capital

    current_regime = None
    hold_days_since_rebalance = 0

    # Capital protection stop tracking (-12% DD stop + 200 SMA re-entry)
    in_cash = False
    days_in_cash = 0
    exit_next_day = False
    reenter_next_day = False

    if STOP_REENTRY_TYPE == 'EMA':
        ma_filter = df['NIFTY'].ewm(span=STOP_REENTRY_MA, adjust=False).mean()
    else:
        ma_filter = df['NIFTY'].rolling(STOP_REENTRY_MA, min_periods=1).mean()

    ema_10 = df['NIFTY'].ewm(span=10, adjust=False).mean()
    sma_150 = df['NIFTY'].rolling(150, min_periods=1).mean()
    sma_200 = df['NIFTY'].rolling(200, min_periods=1).mean()

    records = []

    for date in common_dates:
        idx = list(feat.index).index(date) if date in feat.index else -1
        if idx < 0:
            continue

        day_regime = regimes[idx]
        prices_today = {etf: etf_prices.loc[date, etf] for etf in available_etfs
                        if date in etf_prices.index and not pd.isna(etf_prices.loc[date, etf])}

        if not prices_today:
            continue

        rebalanced = False
        day_charges = 0.0

        # ── 1. Check if Stop Exit was triggered on previous day ──
        if exit_next_day:
            in_cash = True
            exit_next_day = False
            days_in_cash = 0
            # Liquidate all remaining ETF holdings to cash
            n_scrips_sold = 0
            sell_turnover = 0.0
            for etf in available_etfs:
                if holdings[etf] > 0 and etf in prices_today:
                    sell_value = holdings[etf] * prices_today[etf]
                    gain = sell_value - (cost_basis[etf] * holdings[etf])
                    realized_gains += gain
                    sell_turnover += sell_value
                    n_scrips_sold += 1
                    holdings[etf] = 0
                    cost_basis[etf] = 0.0

            if sell_turnover > 0:
                sell_charges = compute_sell_charges(sell_turnover, n_scrips=n_scrips_sold)
                cash += sell_turnover - sell_charges
                day_charges += sell_charges
                total_charges_paid += sell_charges

            current_regime = "CASH (STOP -12%)"
            rebalanced = True

        # ── 2. Check if Re-entry was triggered on previous day ──
        elif reenter_next_day:
            in_cash = False
            reenter_next_day = False
            target_weights = sector_mix.get(day_regime, {})
            if target_weights:
                invest_amount = cash
                orders = []
                for etf in available_etfs:
                    w = target_weights.get(etf, 0.0)
                    if w > 0 and etf in prices_today and prices_today[etf] > 0:
                        target_alloc = invest_amount * w
                        price = prices_today[etf]
                        units = int(target_alloc // price)
                        if units > 0:
                            buy_value = units * price
                            buy_charges = compute_buy_charges(buy_value)
                            total_cost = buy_value + buy_charges
                            orders.append((etf, units, price, buy_value, buy_charges, total_cost))

                total_order_cost = sum(o[5] for o in orders)
                if total_order_cost > cash and orders:
                    while total_order_cost > cash and orders:
                        orders.sort(key=lambda o: o[3], reverse=True)
                        etf, units, price, bv, bc, tc = orders[0]
                        units -= 1
                        if units > 0:
                            bv = units * price
                            bc = compute_buy_charges(bv)
                            tc = bv + bc
                            orders[0] = (etf, units, price, bv, bc, tc)
                        else:
                            orders.pop(0)
                        total_order_cost = sum(o[5] for o in orders)

                for etf, units, price, buy_value, buy_charges, total_cost in orders:
                    holdings[etf] = units
                    cost_basis[etf] = total_cost / units
                    cash -= total_cost
                    day_charges += buy_charges

                total_charges_paid += day_charges
                current_regime = day_regime
                hold_days_since_rebalance = 0
                rebalanced = True
                peak_nav = cash + sum(holdings[e] * prices_today.get(e, 0) for e in available_etfs)

        # ── 3. Handle cash yield and re-entry condition while in cash ──
        if in_cash:
            repo_rate = float(df.loc[date, 'RepoRate']) if ('RepoRate' in df.columns and not pd.isna(df.loc[date, 'RepoRate'])) else 6.50
            daily_repo = repo_rate / 100.0 / 252.0
            cash += cash * daily_repo
            days_in_cash += 1

            nifty_price = float(df.loc[date, 'NIFTY']) if date in df.index else 0.0
            ma_val = float(ma_filter.loc[date]) if date in ma_filter.index else 0.0
            if nifty_price > ma_val and days_in_cash >= STOP_COOLDOWN_DAYS:
                reenter_next_day = True

        # ── 4. Regular Regime-driven rebalance (only when active / not in cash) ──
        elif current_regime is None or (day_regime != current_regime and hold_days_since_rebalance >= MIN_HOLD_DAYS):
            target_weights = sector_mix.get(day_regime, {})
            if target_weights:
                # ── SELL all current holdings ──
                n_scrips_sold = 0
                sell_turnover = 0.0
                for etf in available_etfs:
                    if holdings[etf] > 0 and etf in prices_today:
                        sell_value = holdings[etf] * prices_today[etf]
                        gain = sell_value - (cost_basis[etf] * holdings[etf])
                        realized_gains += gain
                        sell_turnover += sell_value
                        n_scrips_sold += 1
                        holdings[etf] = 0
                        cost_basis[etf] = 0.0

                if sell_turnover > 0:
                    sell_charges = compute_sell_charges(sell_turnover, n_scrips=n_scrips_sold)
                    cash += sell_turnover - sell_charges
                    day_charges += sell_charges
                elif current_regime is None:
                    pass  # initial allocation, no sells

                # ── BUY new positions (whole units only, floor) ──
                invest_amount = cash
                orders = []
                for etf in available_etfs:
                    w = target_weights.get(etf, 0.0)
                    if w > 0 and etf in prices_today and prices_today[etf] > 0:
                        target_alloc = invest_amount * w
                        price = prices_today[etf]
                        units = int(target_alloc // price)
                        if units > 0:
                            buy_value = units * price
                            buy_charges = compute_buy_charges(buy_value)
                            total_cost = buy_value + buy_charges
                            orders.append((etf, units, price, buy_value, buy_charges, total_cost))

                total_order_cost = sum(o[5] for o in orders)
                if total_order_cost > cash and orders:
                    while total_order_cost > cash and orders:
                        orders.sort(key=lambda o: o[3], reverse=True)
                        etf, units, price, bv, bc, tc = orders[0]
                        units -= 1
                        if units > 0:
                            bv = units * price
                            bc = compute_buy_charges(bv)
                            tc = bv + bc
                            orders[0] = (etf, units, price, bv, bc, tc)
                        else:
                            orders.pop(0)
                        total_order_cost = sum(o[5] for o in orders)

                for etf, units, price, buy_value, buy_charges, total_cost in orders:
                    holdings[etf] = units
                    cost_basis[etf] = total_cost / units
                    cash -= total_cost
                    day_charges += buy_charges

                total_charges_paid += day_charges
                current_regime = day_regime
                hold_days_since_rebalance = 0
                rebalanced = True
        else:
            hold_days_since_rebalance += 1

        # ── 5. Mark to market & Drawdown evaluation ──
        equity_value = sum(holdings[etf] * prices_today.get(etf, 0) for etf in available_etfs)
        portfolio_nav = cash + equity_value

        # Compute unrealized gain for tax tracking
        unrealized_gains = 0.0
        for etf in available_etfs:
            if holdings[etf] > 0 and etf in prices_today:
                unrealized_gains += (prices_today[etf] - cost_basis[etf]) * holdings[etf]

        if not in_cash:
            peak_nav = max(peak_nav, portfolio_nav)
            drawdown = (portfolio_nav - peak_nav) / peak_nav if peak_nav > 0 else 0.0
            if drawdown <= STOP_LOSS_DD:
                # ── T+0 Execution: execute as soon as dd = -12% on date itself ──
                n_scrips_sold = 0
                sell_turnover = 0.0
                for etf in available_etfs:
                    if holdings[etf] > 0 and etf in prices_today:
                        sell_value = holdings[etf] * prices_today[etf]
                        gain = sell_value - (cost_basis[etf] * holdings[etf])
                        realized_gains += gain
                        sell_turnover += sell_value
                        n_scrips_sold += 1
                        holdings[etf] = 0
                        cost_basis[etf] = 0.0

                if sell_turnover > 0:
                    sell_charges = compute_sell_charges(sell_turnover, n_scrips=n_scrips_sold)
                    cash += sell_turnover - sell_charges
                    day_charges += sell_charges
                    total_charges_paid += sell_charges

                in_cash = True
                exit_next_day = False
                current_regime = "CASH (STOP -12%)"
                rebalanced = True
                days_in_cash = 0
                equity_value = 0.0
                portfolio_nav = cash
                unrealized_gains = 0.0
        else:
            drawdown = (portfolio_nav - peak_nav) / peak_nav if peak_nav > 0 else 0.0

        # Posterior probabilities for this date
        bull_p = regime_posteriors.loc[date, 'Bull'] if date in regime_posteriors.index else 0
        bear_p = regime_posteriors.loc[date, 'Bear'] if date in regime_posteriors.index else 0
        hv_p   = regime_posteriors.loc[date, 'HighVol'] if date in regime_posteriors.index else 0
        sw_p   = regime_posteriors.loc[date, 'Sideways'] if date in regime_posteriors.index else 0

        nifty_price = float(df.loc[date, 'NIFTY']) if date in df.index else 0.0
        ma_val      = float(ma_filter.loc[date]) if date in ma_filter.index else 0.0
        vix_val     = float(df.loc[date, 'VIX']) if date in df.index else 0.0

        record = {
            'Date': date,
            'Regime': current_regime if in_cash else day_regime,
            'Rebalanced': rebalanced,
            'NAV': portfolio_nav,
            'Cash': cash,
            'Equity': equity_value,
            'Drawdown': drawdown,
            'In_Cash': in_cash,
            'NIFTY': nifty_price,
            'MA_Filter': ma_val,
            'EMA_10': float(ema_10.loc[date]) if date in ema_10.index else ma_val,
            'SMA_Filter': ma_val,
            'SMA_150': float(sma_150.loc[date]) if date in sma_150.index else ma_val,
            'SMA_200': float(sma_200.loc[date]) if date in sma_200.index else ma_val,
            'VIX': vix_val,
            'Bull': bull_p,
            'Bear': bear_p,
            'HighVol': hv_p,
            'Sideways': sw_p,
            'Realized_Gains': realized_gains,
            'Unrealized_Gains': unrealized_gains,
            'Total_Charges': total_charges_paid,
            'Day_Charges': day_charges,
        }

        # Per-ETF positions (whole units + cost basis)
        for etf in available_etfs:
            record[f'{etf}_units'] = holdings[etf]
            record[f'{etf}_value'] = holdings[etf] * prices_today.get(etf, 0)
            record[f'{etf}_price'] = prices_today.get(etf, 0)
            record[f'{etf}_cost_basis'] = cost_basis[etf]
            if holdings[etf] > 0 and cost_basis[etf] > 0:
                record[f'{etf}_pnl'] = (prices_today.get(etf, 0) - cost_basis[etf]) * holdings[etf]
            else:
                record[f'{etf}_pnl'] = 0.0

        records.append(record)

    history = pd.DataFrame(records)
    if len(history) > 0:
        history.set_index('Date', inplace=True)
        history['Daily_Return'] = history['NAV'].pct_change().fillna(0)
        history['Daily_Return'].iloc[0] = (history['NAV'].iloc[0] / initial_capital) - 1
        history['Cumulative_Return'] = (history['NAV'] / initial_capital - 1) * 100
        # NIFTY benchmark rebased
        if 'NIFTY' in history.columns:
            freeze_nifty = df.loc[freeze_date, 'NIFTY'] if (freeze_date in df.index and df.loc[freeze_date, 'NIFTY'] > 0) else history['NIFTY'].iloc[0]
            if freeze_nifty > 0:
                history['NIFTY_Rebased'] = (history['NIFTY'] / freeze_nifty) * initial_capital
            else:
                history['NIFTY_Rebased'] = initial_capital

    return history


# ══════════════════════════════════════════════════════════════════════
# 6. FIGURE 9: PAPER PORTFOLIO VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def plot_paper_portfolio(history, sector_mix, freeze_date, out_dir):
    """Generate Fig 9 — Paper Trading Portfolio Dashboard."""
    if len(history) == 0:
        print("⚠ No OOS data for Fig 9 yet.")
        fig, ax = plt.subplots(figsize=(20, 14))
        fig.patch.set_facecolor('#0a0a0d')
        ax.set_facecolor('#0f0f11')
        ax.text(0.5, 0.5,
                "PAPER PORTFOLIO INITIALIZED\n\n" +
                f"Model frozen on {freeze_date.strftime('%d %b %Y')}\n" +
                f"Starting Capital: ₹{INITIAL_CAPITAL:,.0f}\n\n" +
                "Awaiting new trading days for OOS simulation.\n" +
                "Run this script again after market close.",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, color='#e5e7eb', fontfamily='monospace')
        ax.set_xticks([])
        ax.set_yticks([])
        fig_path = os.path.join(out_dir, 'fig9_paper_portfolio.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {fig_path}")
        alt_dir = os.path.join(out_dir, 'output') if os.path.basename(out_dir) != 'output' else os.path.dirname(out_dir)
        if os.path.exists(alt_dir):
            shutil.copy2(fig_path, os.path.join(alt_dir, 'fig9_paper_portfolio.png'))
        return

    # If only 1 trading day has elapsed since freeze, prepend Day 0 (initial cash baseline)
    # so that time-series lines, allocation area, and drawdown render seamlessly alongside holdings
    if len(history) == 1:
        day0 = history.iloc[0].copy()
        day0_date = history.index[0] - pd.DateOffset(days=1)
        day0['NAV'] = INITIAL_CAPITAL
        day0['Cash'] = INITIAL_CAPITAL
        day0['Equity'] = 0.0
        day0['Drawdown'] = 0.0
        day0['Daily_Return'] = 0.0
        day0['Cumulative_Return'] = 0.0
        day0['Rebalanced'] = False
        day0['Realized_Gains'] = 0.0
        day0['Unrealized_Gains'] = 0.0
        day0['Total_Charges'] = 0.0
        day0['Day_Charges'] = 0.0
        for col in history.columns:
            if col.endswith('_units') or col.endswith('_value') or col.endswith('_pnl'):
                day0[col] = 0.0
        history = pd.concat([pd.DataFrame([day0], index=[day0_date]), history])

    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family':      'monospace',
        'axes.facecolor':   '#0f0f11',
        'figure.facecolor': '#0a0a0d',
        'axes.edgecolor':   '#2a2a35',
        'axes.labelcolor':  '#9ca3af',
        'xtick.color':      '#6b7280',
        'ytick.color':      '#6b7280',
        'grid.color':       '#1e1e28',
        'text.color':       '#e5e7eb',
    })

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('INDIA HMM REGIME DETECTOR  ·  LIVE PAPER TRADING PORTFOLIO (FROZEN MODEL — OOS)',
                 fontsize=13, fontweight='bold', color='#f9fafb', y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
                           height_ratios=[3, 2, 2, 1.5])

    dates = history.index

    # ── Panel 1: Portfolio NAV vs NIFTY Benchmark (top, full width) ──
    ax1 = fig.add_subplot(gs[0, :])

    # Regime shading
    for i in range(len(dates)):
        reg = history['Regime'].iloc[i]
        color = REGIME_COLORS.get(reg, '#3b82f6')
        if i < len(dates) - 1:
            ax1.axvspan(dates[i], dates[i+1], alpha=0.12, color=color, zorder=0)

    ax1.plot(dates, history['NAV'], color='#ffffff', lw=2.2, label='Paper Portfolio NAV', zorder=3)

    if 'NIFTY_Rebased' in history.columns:
        ax1.plot(dates, history['NIFTY_Rebased'], color='#6b7280', lw=1.5,
                 ls='--', alpha=0.7, label='NIFTY 50 (Rebased)', zorder=2)

    # Mark rebalance events
    rebal_dates = history.index[history['Rebalanced'] == True]
    if len(rebal_dates) > 0:
        rebal_navs = history.loc[rebal_dates, 'NAV']
        ax1.scatter(rebal_dates, rebal_navs, marker='v', s=80, color='#f59e0b',
                    zorder=5, label='Rebalance', edgecolors='white', linewidths=0.5)

    ax1.set_title(f'Portfolio NAV  ·  Start: ₹{INITIAL_CAPITAL:,.0f}  |  '
                  f'Current: ₹{history["NAV"].iloc[-1]:,.0f}  |  '
                  f'Return: {history["Cumulative_Return"].iloc[-1]:+.2f}%',
                  fontsize=11, color='#e5e7eb', pad=10)
    ax1.set_ylabel('Portfolio Value (₹)', fontsize=10)
    ax1.grid(True, alpha=0.15)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))

    # Regime legend
    patches = [mpatches.Patch(color=v, label=k, alpha=0.6) for k, v in REGIME_COLORS.items()]
    ax1.legend(handles=patches + ax1.get_legend_handles_labels()[0],
               loc='upper left', fontsize=8, framealpha=0.3, ncol=3)

    # ── Panel 2: Current Portfolio Allocation Pie Chart (middle-left) ──
    ax2 = fig.add_subplot(gs[1, 0])

    latest = history.iloc[-1]
    pie_labels = []
    pie_values = []
    etf_units_cols = [c for c in history.columns if c.endswith('_units')]
    for col in etf_units_cols:
        etf = col.replace('_units', '')
        val = latest.get(f'{etf}_value', 0)
        if val > 0:
            pie_labels.append(ETF_DISPLAY.get(etf, etf))
            pie_values.append(val)

    cash_val = latest.get('Cash', 0)
    if cash_val > 0:
        pie_labels.append('Cash')
        pie_values.append(cash_val)

    sorted_slices = sorted(zip(pie_labels, pie_values), key=lambda x: x[1], reverse=True)
    pie_labels = [s[0] for s in sorted_slices]
    pie_values = [s[1] for s in sorted_slices]
    total_pie = sum(pie_values) if sum(pie_values) > 0 else 1

    pie_colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#eab308', '#64748b']

    wedges, texts, *extra = ax2.pie(
        pie_values,
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p >= 2.5 else '',
        pctdistance=0.74,
        startangle=140,
        radius=1.22,
        colors=pie_colors[:len(pie_values)],
        wedgeprops=dict(width=0.48, edgecolor='#0f0f11', linewidth=2.5)
    )
    autotexts = extra[0] if extra else []

    for at in autotexts:
        at.set_color('#ffffff')
        at.set_fontsize(8)
        at.set_fontweight('bold')

    ax2.text(0, 0, f'PORTFOLIO\n₹{latest["NAV"]:,.0f}', ha='center', va='center',
             fontsize=11, fontweight='bold', color='#f9fafb', fontfamily='monospace')

    legend_labels = [f'{lbl}: ₹{val:,.0f} ({val/total_pie*100:.1f}%)' for lbl, val in zip(pie_labels, pie_values)]
    ax2.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize=9, framealpha=0.35, edgecolor='#2a2a35')

    ax2.set_title(f'Current Portfolio Allocation  ·  {latest.name.strftime("%d %b %Y")}',
                  fontsize=11, color='#e5e7eb', pad=12)

    # ── Panel 3: Drawdown Chart (middle-right) ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(dates, history['Drawdown'] * 100, 0, color='#ef4444', alpha=0.4)
    ax3.plot(dates, history['Drawdown'] * 100, color='#ef4444', lw=1.2)
    ax3.axhline(STOP_LOSS_DD * 100, color='#f59e0b', ls='--', lw=1.0, label=f'Stop Loss Threshold ({STOP_LOSS_DD*100:.0f}%)')
    ax3.set_ylabel('Drawdown (%)', fontsize=10)
    ax3.set_title(f'Portfolio Drawdown  ·  Max: {history["Drawdown"].min()*100:.2f}%  |  Stop Threshold: {STOP_LOSS_DD*100:.0f}%',
                  fontsize=11, color='#e5e7eb', pad=8)
    ax3.grid(True, alpha=0.15)
    ax3.legend(loc='lower left', fontsize=8, framealpha=0.3)

    # ── Panel 4: Daily Returns Distribution (bottom-left) ──
    ax4 = fig.add_subplot(gs[2, 0])
    daily_rets = history['Daily_Return'].dropna()
    if len(daily_rets) >= 1:
        colors_bar = ['#10b981' if r >= 0 else '#ef4444' for r in daily_rets]
        ax4.bar(daily_rets.index, daily_rets * 100, color=colors_bar, alpha=0.7, width=0.6)
        ax4.axhline(0, color='#4b5563', lw=0.5)
        ax4.set_ylabel('Daily Return (%)', fontsize=10)
        ax4.set_title('Daily P&L', fontsize=11, color='#e5e7eb', pad=8)
        ax4.grid(True, alpha=0.15)

    # ── Panel 5: Current Holdings Table (bottom-right) ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    latest = history.iloc[-1]
    table_data = []
    etf_units_cols = [c for c in history.columns if c.endswith('_units')]
    for col in etf_units_cols:
        etf = col.replace('_units', '')
        units = latest.get(f'{etf}_units', 0)
        price = latest.get(f'{etf}_price', 0)
        cost  = latest.get(f'{etf}_cost_basis', 0)
        value = latest.get(f'{etf}_value', 0)
        pnl   = latest.get(f'{etf}_pnl', 0)
        nav = latest['NAV']
        weight = (value / nav * 100) if nav > 0 else 0
        if units > 0:
            table_data.append([ETF_DISPLAY.get(etf, etf), f'{int(units)}',
                               f'₹{cost:,.2f}', f'₹{price:,.2f}',
                               f'₹{value:,.0f}', f'{pnl:+,.0f}', f'{weight:.1f}%'])

    if table_data:
        charges_str = f'₹{latest.get("Total_Charges", 0):,.0f}'
        table_data.append(['CASH', '—', '—', '—', f'₹{latest["Cash"]:,.0f}', '—',
                           f'{latest["Cash"]/latest["NAV"]*100:.1f}%'])
        table_data.append(['TOTAL', '—', '—', '—', f'₹{latest["NAV"]:,.0f}',
                           f'{latest.get("Unrealized_Gains", 0):+,.0f}', '100.0%'])

        table = ax5.table(cellText=table_data,
                          colLabels=['ETF', 'Units', 'Avg Cost', 'CMP', 'Value', 'P&L', 'Wt%'],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)

        for key, cell in table.get_celld().items():
            cell.set_edgecolor('#2a2a35')
            if key[0] == 0:  # header
                cell.set_facecolor('#1e293b')
                cell.set_text_props(color='#f9fafb', fontweight='bold')
            elif key[0] == len(table_data):  # total row
                cell.set_facecolor('#1e293b')
                cell.set_text_props(color='#10b981', fontweight='bold')
            else:
                cell.set_facecolor('#0f172a')
                cell.set_text_props(color='#e5e7eb')

        ax5.set_title(f'Current Holdings — {latest.name.strftime("%d %b %Y")}',
                      fontsize=11, color='#e5e7eb', pad=8)

    # ── Panel 6: Performance Summary (bottom full width) ──
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    n_days = len(history)
    total_ret = (history['NAV'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    ann_factor = 252 / max(n_days, 1)

    daily_rets_clean = history['Daily_Return'].dropna()
    if len(daily_rets_clean) > 1 and daily_rets_clean.std() > 0:
        sharpe = (daily_rets_clean.mean() / daily_rets_clean.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    ann_ret = (1 + total_ret / 100) ** ann_factor - 1 if n_days > 0 else 0
    ann_vol = daily_rets_clean.std() * np.sqrt(252) * 100 if len(daily_rets_clean) > 1 else 0
    max_dd = history['Drawdown'].min() * 100
    n_rebalances = history['Rebalanced'].sum()

    # NIFTY benchmark stats
    if 'NIFTY_Rebased' in history.columns:
        nifty_ret = (history['NIFTY_Rebased'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    else:
        nifty_ret = 0

    curr_regime = history['Regime'].iloc[-1]
    regime_emoji = {'Bull': '🟢', 'Bear': '🔴', 'HighVol': '🟡', 'Sideways': '🔵'}

    total_charges = history['Total_Charges'].iloc[-1] if 'Total_Charges' in history.columns else 0
    realized_g = history['Realized_Gains'].iloc[-1] if 'Realized_Gains' in history.columns else 0
    unrealized_g = history['Unrealized_Gains'].iloc[-1] if 'Unrealized_Gains' in history.columns else 0

    summary_text = (
        f"  OOS PERIOD: {history.index[0].strftime('%d %b %Y')} → {history.index[-1].strftime('%d %b %Y')} ({n_days} trading days)  |  "
        f"MODEL FROZEN: {freeze_date.strftime('%d %b %Y')}  |  ACTIVE REGIME: [{curr_regime.upper()}]\n\n"
        f"  Portfolio Return: {total_ret:+.2f}%    |    NIFTY 50 Return: {nifty_ret:+.2f}%    |    "
        f"Alpha: {total_ret - nifty_ret:+.2f}%    |    "
        f"Sharpe: {sharpe:.2f}    |    Ann. Vol: {ann_vol:.1f}%    |    "
        f"Max Drawdown: {max_dd:.2f}%    |    Rebalances: {int(n_rebalances)}\n"
        f"  Realized P&L: ₹{realized_g:+,.0f}    |    Unrealized P&L: ₹{unrealized_g:+,.0f}    |    "
        f"Total Charges: ₹{total_charges:,.0f} (STT+ExchTxn+GST+SEBI+Stamp+DP+Slippage)\n\n"
        f"  LIVE OOS PAPER TRADING — Costs as per Indian regulations (SEBI/NSE/IT Act). Not financial advice."
    )

    ax6.text(0.02, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=10, color='#e5e7eb', fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e293b', edgecolor='#334155', alpha=0.9))

    fig_path = os.path.join(out_dir, 'fig9_paper_portfolio.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {fig_path}")

    alt_dir = os.path.join(out_dir, 'output') if os.path.basename(out_dir) != 'output' else os.path.dirname(out_dir)
    if os.path.exists(alt_dir):
        shutil.copy2(fig_path, os.path.join(alt_dir, 'fig9_paper_portfolio.png'))



# ══════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  INDIA HMM REGIME DETECTOR  —  LIVE PAPER TRADING (FROZEN MODEL)")
    print("  No retraining. All new data = pure out-of-sample testing.")
    print("=" * 70)

    _load_env_file()

    OUT_DIR = str(PROJECT_DIR)

    # 1. Load frozen model artifacts
    print("\n[1] Loading frozen model artifacts...")
    artifacts = load_frozen_artifacts()
    model      = artifacts['model']
    scaler     = artifacts['scaler']
    label_map  = artifacts['label_map']
    sector_mix = artifacts['sector_mix']
    freeze_date = artifacts['freeze_date']

    # 2. Fetch full market data (need history for rolling features)
    print("\n[2] Fetching market + macro data...")
    from regime_detector_v2 import fetch_live_market_data
    df = fetch_live_market_data(start_date="2015-01-01")

    # 3. Engineer features
    print("\n[3] Engineering features (identical to training pipeline)...")
    feat = engineer_features(df)
    print(f"✓ {len(feat)} feature rows ({feat.index[0].strftime('%Y-%m-%d')} → {feat.index[-1].strftime('%Y-%m-%d')})")

    # 4. Predict regimes on ALL data using FROZEN model
    print("\n[4] Predicting regimes with frozen model (causal online forward decode)...")
    regimes, regime_posteriors = predict_regimes_frozen(model, scaler, label_map, feat)

    oos_count = (feat.index > freeze_date).sum()
    print(f"  In-sample: {len(feat) - oos_count} days | Out-of-sample (new): {oos_count} days")

    if oos_count > 0:
        oos_regimes = [regimes[i] for i in range(len(regimes)) if feat.index[i] > freeze_date]
        from collections import Counter
        reg_counts = Counter(oos_regimes)
        print(f"  OOS regime distribution: {dict(reg_counts)}")

    # 5. Fetch actual ETF prices
    print("\n[5] Fetching actual end-of-day ETF prices...")
    # Fetch from 1 month before freeze date for price continuity
    etf_start = (freeze_date - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    etf_prices = fetch_etf_prices(start_date=etf_start)

    # 6. Run paper trading simulation
    print("\n[6] Running paper trading simulation...")
    history = run_paper_trading(
        df, feat, regimes, regime_posteriors, etf_prices,
        sector_mix, freeze_date, initial_capital=INITIAL_CAPITAL
    )

    if len(history) > 0:
        # Save portfolio history
        history.to_csv(os.path.join(OUT_DIR, 'paper_portfolio_history.csv'))
        print(f"✓ Portfolio history saved ({len(history)} rows)")

        # Performance summary
        total_ret = (history['NAV'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
        max_dd = history['Drawdown'].min() * 100
        stop_status = "STOPPED OUT (100% Cash / Liquid)" if ('In_Cash' in history.columns and history['In_Cash'].iloc[-1]) else f"Active (Drawdown: {history['Drawdown'].iloc[-1]*100:.2f}%, Stop: {STOP_LOSS_DD*100:.0f}%)"
        print(f"\n{'='*60}")
        print(f"  PAPER PORTFOLIO SUMMARY (Out-of-Sample)")
        print(f"{'='*60}")
        print(f"  Period      : {history.index[0].strftime('%d %b %Y')} → {history.index[-1].strftime('%d %b %Y')}")
        print(f"  Starting    : ₹{INITIAL_CAPITAL:,.0f}")
        print(f"  Current NAV : ₹{history['NAV'].iloc[-1]:,.0f}")
        print(f"  Total Return: {total_ret:+.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Stop Status : {stop_status}")
        print(f"  Re-entry MA : {STOP_REENTRY_MA} {STOP_REENTRY_TYPE} (NIFTY 50)")
        print(f"  Regime      : {history['Regime'].iloc[-1]}")
        total_chg = history['Total_Charges'].iloc[-1] if 'Total_Charges' in history.columns else 0
        print(f"  Total Charges: ₹{total_chg:,.2f} (STT+Exchange+GST+SEBI+Stamp+DP+Slippage)")
        print(f"  Realized P&L : ₹{history['Realized_Gains'].iloc[-1]:+,.2f}")
        print(f"  Unrealized   : ₹{history['Unrealized_Gains'].iloc[-1]:+,.2f}")
        cash_held = history['Cash'].iloc[-1]
        print(f"  Cash held   : ₹{cash_held:,.2f} (from rounding to whole units)")
        print(f"{'='*60}")

    # 7. Generate Fig 9
    print("\n[7] Generating Fig 9 — Paper Portfolio Dashboard...")
    plot_paper_portfolio(history, sector_mix, freeze_date, OUT_DIR)

    print("\n✓ Paper portfolio update complete.")


if __name__ == '__main__':
    main()