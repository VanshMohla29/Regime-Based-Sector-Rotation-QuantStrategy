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

from regime_detector_v2 import (
    fetch_live_market_data,
    engineer_features,
    smooth_regimes_causal,
    online_forward_decode,
    apply_dark_figure_style,
)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

SRC_DIR       = Path(__file__).parent.resolve()
PROJECT_DIR   = SRC_DIR.parent if (SRC_DIR.parent / 'data').exists() or (SRC_DIR.parent / 'artifacts').exists() else SRC_DIR
ARTIFACT_DIR  = PROJECT_DIR / 'artifacts' if (PROJECT_DIR / 'artifacts').exists() else PROJECT_DIR
DATA_DIR      = PROJECT_DIR / 'data' if (PROJECT_DIR / 'data').exists() else PROJECT_DIR
FIGURES_DIR   = PROJECT_DIR / 'figures' if (PROJECT_DIR / 'figures').exists() else PROJECT_DIR
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

# Capital protection parameters (-6% drawdown stop with T+0 execution + 20 SMA re-entry)
STOP_LOSS_DD       = -0.06   # -6% portfolio drawdown stop to 100% Cash / Liquid (T+0 execution)
STOP_REENTRY_MA    = 20      # 20-day Simple Moving Average on NIFTY 50
STOP_REENTRY_TYPE  = 'SMA'   # 'EMA' or 'SMA'
STOP_COOLDOWN_DAYS = 3      # Minimum 3 trading days in Cash before checking re-entry

# Trade execution timing: 'T+0' (same-day execution) or 'T+1' (next-day execution)
# T+0 executes stop exit, re-entry, and regime rebalances on signal date itself
EXECUTION_MODE     = os.getenv('PORTFOLIO_EXECUTION_MODE', 'T+0')

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
    'wpi_mom_vs_ma3m',
    'real_rate_mom',
]


# ══════════════════════════════════════════════════════════════════════
# 1. LOAD FROZEN MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════

def load_frozen_artifacts():
    """Load the frozen HMM model, scaler, label map, and sector mix."""
    artifacts = {}
    for name, fname in [('model', 'hmm_model.pkl'), ('scaler', 'hmm_scaler.pkl'),
                         ('label_map', 'label_map.pkl')]:
        path = ARTIFACT_DIR / fname
        if not path.exists():
            path = PROJECT_DIR / fname
        if not path.exists():
            print(f"❌ {fname} not found in {ARTIFACT_DIR} or {PROJECT_DIR}. Run regime_detector_v2.py first.")
            sys.exit(1)
        with open(path, 'rb') as f:
            artifacts[name] = pickle.load(f)

    mix_path = ARTIFACT_DIR / 'learned_sector_mix.json'
    if not mix_path.exists():
        mix_path = PROJECT_DIR / 'learned_sector_mix.json'
    if mix_path.exists():
        with open(mix_path) as f:
            artifacts['sector_mix'] = json.load(f)
    else:
        print("❌ learned_sector_mix.json not found.")
        sys.exit(1)

    # Determine freeze date (must be persistent so OOS portfolio accumulates over time)
    hist_path = DATA_DIR / 'regime_history_v2.csv'
    if not hist_path.exists():
        hist_path = PROJECT_DIR / 'regime_history_v2.csv'
    freeze_file = ARTIFACT_DIR / 'freeze_date.txt'
    if not freeze_file.exists():
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

def fetch_full_market_data(start_date="2015-01-01"):
    """
    Fetch the market + macro dataset using the primary institutional pipeline.
    Reuses fetch_live_market_data() to guarantee 100% parity and eliminate duplicate scraping.
    """
    return fetch_live_market_data(start_date=start_date)


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
# 4. REGIME PREDICTION WITH FROZEN MODEL
# ══════════════════════════════════════════════════════════════════════

def predict_regimes_frozen(model, scaler, label_map, feat):
    """
    Use the FROZEN model to predict regimes on new data.
    Uses online forward decoding (causal, no lookahead).
    """
    X = feat[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    # Reuses causal online forward decoding with posterior probabilities
    states, posteriors = online_forward_decode(model, X_scaled, return_posteriors=True)

    # Map states to regime names and apply causal anti-whipsaw filter
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


# ══════════════════════════════════════════════════════════════════════
# 5. PAPER TRADING ENGINE
# ══════════════════════════════════════════════════════════════════════

def _execute_buy(cash, target_weights, available_etfs, prices_today):
    """Allocate cash across available ETFs per target weights using whole integer units."""
    if not target_weights:
        return {}, {}, cash, 0.0
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
                orders.append([etf, units, price, buy_value, buy_charges, total_cost])

    total_order_cost = sum(o[5] for o in orders)
    if total_order_cost > cash and orders:
        while total_order_cost > cash and orders:
            orders.sort(key=lambda o: o[3], reverse=True)
            o = orders[0]
            o[1] -= 1  # units
            if o[1] > 0:
                o[3] = o[1] * o[2]  # buy_value
                o[4] = compute_buy_charges(o[3])  # buy_charges
                o[5] = o[3] + o[4]  # total_cost
            else:
                orders.pop(0)
            total_order_cost = sum(o[5] for o in orders)

    new_holdings = {etf: 0 for etf in available_etfs}
    new_cost_basis = {etf: 0.0 for etf in available_etfs}
    total_buy_charges = 0.0
    for etf, units, price, buy_value, buy_charges, total_cost in orders:
        new_holdings[etf] = units
        new_cost_basis[etf] = total_cost / units
        cash -= total_cost
        total_buy_charges += buy_charges

    return new_holdings, new_cost_basis, cash, total_buy_charges


def _execute_sell(holdings, cost_basis, available_etfs, prices_today):
    """Liquidate all current ETF holdings to cash at current prices."""
    n_scrips_sold = 0
    sell_turnover = 0.0
    realized_gain = 0.0
    for etf in available_etfs:
        if holdings.get(etf, 0) > 0 and etf in prices_today:
            sell_value = holdings[etf] * prices_today[etf]
            realized_gain += sell_value - (cost_basis.get(etf, 0.0) * holdings[etf])
            sell_turnover += sell_value
            n_scrips_sold += 1
            holdings[etf] = 0
            cost_basis[etf] = 0.0

    sell_charges = 0.0
    cash_proceeds = 0.0
    if sell_turnover > 0:
        sell_charges = compute_sell_charges(sell_turnover, n_scrips=n_scrips_sold)
        cash_proceeds = sell_turnover - sell_charges

    return cash_proceeds, sell_charges, realized_gain


def run_paper_trading(df, feat, regimes, regime_posteriors, etf_prices,
                      sector_mix, freeze_date, initial_capital=INITIAL_CAPITAL,
                      execution_mode=EXECUTION_MODE):
    """
    Simulate a paper trading portfolio from freeze_date onwards.

    Supports both 'T+0' (same-day execution) and 'T+1' (next-day execution)
    timing for stop exits, re-entry, and regime rebalancing.

    Returns a DataFrame with daily portfolio state:
      Date, Regime, NAV, Cash, Equity_Value, Drawdown, Daily_Return,
      per-ETF positions (units & value), rebalance flags
    """
    execution_mode = str(execution_mode).upper()
    if execution_mode not in ('T+0', 'T+1'):
        execution_mode = 'T+0'

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

    print(f"\n[Paper Trading] Simulating from {common_dates[0].strftime('%d %b %Y')} to {common_dates[-1].strftime('%d %b %Y')} ({len(common_dates)} sessions | Execution: {execution_mode})")

    etf_names = list(ETF_TICKERS.keys())
    available_etfs = [e for e in etf_names if e in etf_prices.columns and etf_prices[e].notna().any()]

    # Initialize portfolio
    portfolio_nav = initial_capital
    cash = initial_capital
    holdings = {etf: 0 for etf in available_etfs}          # whole units only (int)
    cost_basis = {etf: 0.0 for etf in available_etfs}      # weighted avg buy price
    realized_gains = 0.0                                    # cumulative realized P&L
    total_charges_paid = 0.0                                # cumulative txn costs
    max_portfolio_nav = initial_capital                     # All-time peak NAV of portfolio (never reset downwards)
    stop_peak_nav = initial_capital                         # Cycle peak NAV for trailing stop-loss check

    current_regime = None
    hold_days_since_rebalance = 0

    # Capital protection stop tracking
    in_cash = False
    days_in_cash = 0
    exit_next_day = False
    reenter_next_day = False
    pending_rebalance = None

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

        # ── 1. T+1 Execution Check: previous day pending actions ──
        if execution_mode == 'T+1':
            if exit_next_day:
                cash_proceeds, sell_chg, gain = _execute_sell(holdings, cost_basis, available_etfs, prices_today)
                cash += cash_proceeds
                day_charges += sell_chg
                total_charges_paid += sell_chg
                realized_gains += gain
                in_cash = True
                exit_next_day = False
                days_in_cash = 0
                current_regime = f"CASH (STOP {int(STOP_LOSS_DD*100)}%)"
                rebalanced = True

            elif reenter_next_day:
                target_weights = sector_mix.get(day_regime, {})
                if target_weights:
                    new_h, new_cb, cash, buy_chg = _execute_buy(cash, target_weights, available_etfs, prices_today)
                    holdings.update(new_h)
                    cost_basis.update(new_cb)
                    day_charges += buy_chg
                    total_charges_paid += buy_chg
                    in_cash = False
                    reenter_next_day = False
                    current_regime = day_regime
                    hold_days_since_rebalance = 0
                    rebalanced = True
                    stop_peak_nav = cash + sum(holdings[e] * prices_today.get(e, 0) for e in available_etfs)

            elif pending_rebalance is not None:
                target_regime = pending_rebalance
                pending_rebalance = None
                target_weights = sector_mix.get(target_regime, {})
                if target_weights:
                    cash_proceeds, sell_chg, gain = _execute_sell(holdings, cost_basis, available_etfs, prices_today)
                    cash += cash_proceeds
                    day_charges += sell_chg
                    total_charges_paid += sell_chg
                    realized_gains += gain

                    new_h, new_cb, cash, buy_chg = _execute_buy(cash, target_weights, available_etfs, prices_today)
                    holdings.update(new_h)
                    cost_basis.update(new_cb)
                    day_charges += buy_chg
                    total_charges_paid += buy_chg
                    current_regime = target_regime
                    hold_days_since_rebalance = 0
                    rebalanced = True

        # ── 2. Handle Cash Yield and Re-entry Condition while in Cash ──
        if in_cash:
            repo_rate = float(df.loc[date, 'RepoRate']) if ('RepoRate' in df.columns and not pd.isna(df.loc[date, 'RepoRate'])) else 6.50
            daily_repo = repo_rate / 100.0 / 252.0
            cash += cash * daily_repo
            days_in_cash += 1

            nifty_price = float(df.loc[date, 'NIFTY']) if date in df.index else 0.0
            ma_val = float(ma_filter.loc[date]) if date in ma_filter.index else 0.0
            if nifty_price > ma_val and days_in_cash >= STOP_COOLDOWN_DAYS:
                if execution_mode == 'T+0':
                    # ── T+0 Execution: Immediate re-entry into sector mix on date itself ──
                    target_weights = sector_mix.get(day_regime, {})
                    if target_weights:
                        new_h, new_cb, cash, buy_chg = _execute_buy(cash, target_weights, available_etfs, prices_today)
                        holdings.update(new_h)
                        cost_basis.update(new_cb)
                        day_charges += buy_chg
                        total_charges_paid += buy_chg
                        in_cash = False
                        days_in_cash = 0
                        reenter_next_day = False
                        current_regime = day_regime
                        hold_days_since_rebalance = 0
                        rebalanced = True
                        stop_peak_nav = cash + sum(holdings[e] * prices_today.get(e, 0) for e in available_etfs)
                else:
                    reenter_next_day = True

        # ── 3. Regular Regime-Driven Rebalance (only when active / not in cash) ──
        elif current_regime is None:
            # Initial allocation on freeze date
            target_weights = sector_mix.get(day_regime, {})
            if target_weights:
                new_h, new_cb, cash, buy_chg = _execute_buy(cash, target_weights, available_etfs, prices_today)
                holdings.update(new_h)
                cost_basis.update(new_cb)
                day_charges += buy_chg
                total_charges_paid += buy_chg
                current_regime = day_regime
                hold_days_since_rebalance = 0
                rebalanced = True
                stop_peak_nav = cash + sum(holdings[e] * prices_today.get(e, 0) for e in available_etfs)

        elif day_regime != current_regime and hold_days_since_rebalance >= MIN_HOLD_DAYS:
            if execution_mode == 'T+0':
                # ── T+0 Execution: Immediate rebalance on date itself ──
                target_weights = sector_mix.get(day_regime, {})
                if target_weights:
                    cash_proceeds, sell_chg, gain = _execute_sell(holdings, cost_basis, available_etfs, prices_today)
                    cash += cash_proceeds
                    day_charges += sell_chg
                    total_charges_paid += sell_chg
                    realized_gains += gain

                    new_h, new_cb, cash, buy_chg = _execute_buy(cash, target_weights, available_etfs, prices_today)
                    holdings.update(new_h)
                    cost_basis.update(new_cb)
                    day_charges += buy_chg
                    total_charges_paid += buy_chg
                    current_regime = day_regime
                    hold_days_since_rebalance = 0
                    rebalanced = True
            else:
                # ── T+1 Execution: Queue rebalance for next trading day ──
                pending_rebalance = day_regime
                hold_days_since_rebalance += 1
        else:
            hold_days_since_rebalance += 1

        # ── 4. Mark to Market & Drawdown Evaluation ──
        equity_value = sum(holdings[etf] * prices_today.get(etf, 0) for etf in available_etfs)
        portfolio_nav = cash + equity_value

        # True portfolio drawdown from all-time peak NAV of the portfolio (never reset downwards)
        max_portfolio_nav = max(max_portfolio_nav, portfolio_nav)
        drawdown = (portfolio_nav - max_portfolio_nav) / max_portfolio_nav if max_portfolio_nav > 0 else 0.0

        # Compute unrealized gain for tax tracking
        unrealized_gains = 0.0
        for etf in available_etfs:
            if holdings[etf] > 0 and etf in prices_today:
                unrealized_gains += (prices_today[etf] - cost_basis[etf]) * holdings[etf]

        if not in_cash:
            # Trailing stop-loss checks drawdown from cycle peak
            stop_peak_nav = max(stop_peak_nav, portfolio_nav)
            stop_dd = (portfolio_nav - stop_peak_nav) / stop_peak_nav if stop_peak_nav > 0 else 0.0
            if stop_dd <= STOP_LOSS_DD:
                if execution_mode == 'T+0':
                    # ── T+0 Execution: execute stoploss immediately on date itself ──
                    cash_proceeds, sell_chg, gain = _execute_sell(holdings, cost_basis, available_etfs, prices_today)
                    cash += cash_proceeds
                    day_charges += sell_chg
                    total_charges_paid += sell_chg
                    realized_gains += gain
                    in_cash = True
                    exit_next_day = False
                    current_regime = f"CASH (STOP {int(STOP_LOSS_DD*100)}%)"
                    rebalanced = True
                    days_in_cash = 0
                    equity_value = 0.0
                    portfolio_nav = cash
                    unrealized_gains = 0.0
                    # Recompute drawdown from all-time max NAV
                    drawdown = (portfolio_nav - max_portfolio_nav) / max_portfolio_nav if max_portfolio_nav > 0 else 0.0
                else:
                    exit_next_day = True

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
            'Market_Regime': day_regime,
            'Rebalanced': rebalanced,
            'NAV': portfolio_nav,
            'Peak_NAV': max_portfolio_nav,
            'Cash': cash,
            'Equity': equity_value,
            'Drawdown': drawdown,
            'In_Cash': in_cash,
            'Execution_Mode': execution_mode,
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
        history.loc[history.index[0], 'Daily_Return'] = (history['NAV'].iloc[0] / initial_capital) - 1
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

    apply_dark_figure_style()

    exec_mode = history['Execution_Mode'].iloc[-1] if 'Execution_Mode' in history.columns else EXECUTION_MODE
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f'INDIA HMM REGIME DETECTOR  ·  LIVE PAPER TRADING PORTFOLIO ({exec_mode} EXECUTION — FROZEN MODEL)',
                 fontsize=13, fontweight='bold', color='#f9fafb', y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
                           height_ratios=[3, 2, 2, 1.5])

    dates = history.index

    # ── Panel 1: Portfolio NAV vs NIFTY Benchmark (top, full width) ──
    ax1 = fig.add_subplot(gs[0, :])

    # Regime shading (strictly as per current market regime, never defaulted to Sideways on stop hit)
    for i in range(len(dates)):
        reg = history['Market_Regime'].iloc[i] if 'Market_Regime' in history.columns else history['Regime'].iloc[i]
        if 'CASH' in str(reg):
            probs = {r: history[r].iloc[i] for r in ('Bull', 'Bear', 'HighVol', 'Sideways') if r in history.columns}
            reg = max(probs, key=probs.get) if probs else 'HighVol'
        color = REGIME_COLORS.get(reg, '#f59e0b')
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
        f"MODEL FROZEN: {freeze_date.strftime('%d %b %Y')}  |  EXECUTION: [{exec_mode}]  |  ACTIVE REGIME: [{curr_regime.upper()}]\n\n"
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



# ══════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Paper Trading Portfolio (Frozen Model)")
    parser.add_argument('--execution', choices=['T+0', 'T+1'],
                        default=os.getenv('PORTFOLIO_EXECUTION_MODE', EXECUTION_MODE),
                        help="Trade execution timing: 'T+0' (same-day) or 'T+1' (next-day). Default: T+0")
    args, _ = parser.parse_known_args()
    exec_mode = args.execution

    print("=" * 70)
    print(f"  INDIA HMM REGIME DETECTOR  —  LIVE PAPER TRADING (EXECUTION: {exec_mode})")
    print("  No retraining. All new data = pure out-of-sample testing.")
    print("=" * 70)

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
    print(f"\n[6] Running paper trading simulation (Execution Mode: {exec_mode})...")
    history = run_paper_trading(
        df, feat, regimes, regime_posteriors, etf_prices,
        sector_mix, freeze_date, initial_capital=INITIAL_CAPITAL,
        execution_mode=exec_mode
    )

    if len(history) > 0:
        # Save portfolio history
        history.to_csv(os.path.join(DATA_DIR, 'paper_portfolio_history.csv'))
        print(f"✓ Portfolio history saved to {DATA_DIR} ({len(history)} rows)")

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
        print(f"  Execution   : {exec_mode} ({'immediate same-day' if exec_mode == 'T+0' else 'next-day'})")
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
    plot_paper_portfolio(history, sector_mix, freeze_date, str(FIGURES_DIR))

    print("\n✓ Paper portfolio update complete.")


if __name__ == '__main__':
    main()