-- ══════════════════════════════════════════════════════════════════════════
-- India Market Regime Detector — PostgreSQL + TimescaleDB Schema
-- Supports TimescaleDB Hypertables with automatic graceful fallback to
-- standard PostgreSQL partitioned / indexed tables if TimescaleDB extension
-- is not present.
-- ══════════════════════════════════════════════════════════════════════════

-- Enable TimescaleDB extension and UUID generation
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'TimescaleDB extension not available, continuing with standard PostgreSQL tables.';
END $$;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ══════════════════════════════════════════════════════════════════════════
-- 1. RAW MARKET BARS (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS market_bars (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(16) NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT DEFAULT 0,
    returns     DOUBLE PRECISION, -- Log returns: ln(Close_t / Close_t-1)
    source      VARCHAR(16) DEFAULT 'YAHOO',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol)
);

DO $$
BEGIN
    PERFORM create_hypertable('market_bars', 'time', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for market_bars.';
END $$;

CREATE INDEX IF NOT EXISTS idx_market_bars_symbol_time ON market_bars (symbol, time DESC);

-- ══════════════════════════════════════════════════════════════════════════
-- 2. MACROECONOMIC INDICATORS (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS macro_indicators (
    time                TIMESTAMPTZ PRIMARY KEY,
    cpi_yoy             DOUBLE PRECISION, -- MoSPI Combined CPI YoY %
    cpi_mom             DOUBLE PRECISION, -- MoSPI CPI MoM %
    wpi_yoy             DOUBLE PRECISION, -- DPIIT WPI YoY %
    wpi_mom             DOUBLE PRECISION, -- DPIIT WPI MoM %
    iip_yoy             DOUBLE PRECISION, -- MoSPI IIP YoY %
    repo_rate           DOUBLE PRECISION, -- RBI Official Policy Repo Rate %
    gsec_10y            DOUBLE PRECISION, -- GoI 10Y Benchmark Sovereign Yield %
    gsec_2y             DOUBLE PRECISION, -- GoI 2Y Sovereign Yield %
    yield_curve         DOUBLE PRECISION, -- Spread: 10Y - 2Y
    real_policy_rate    DOUBLE PRECISION, -- RepoRate - CPI
    pub_lag_offset_days INTEGER DEFAULT 30, -- Zero-lookahead publication offset
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

DO $$
BEGIN
    PERFORM create_hypertable('macro_indicators', 'time', chunk_time_interval => INTERVAL '2 years', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for macro_indicators.';
END $$;

-- ══════════════════════════════════════════════════════════════════════════
-- 3. QUANTITATIVE FEATURE STORE (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS feature_store (
    time                TIMESTAMPTZ PRIMARY KEY,
    ret_5d_pct          DOUBLE PRECISION NOT NULL, -- 5-day rolling return %
    vol_60d             DOUBLE PRECISION NOT NULL, -- 60-day annualized realized volatility
    price_vs_ma200      DOUBLE PRECISION NOT NULL, -- NIFTY / 200-day SMA ratio
    drawdown            DOUBLE PRECISION NOT NULL, -- Rolling peak-to-trough drawdown
    drawdown_diff20d    DOUBLE PRECISION NOT NULL, -- 20-day drawdown recovery delta
    vix_vs_ma20         DOUBLE PRECISION NOT NULL, -- India VIX / 20-day SMA ratio
    yield_curve         DOUBLE PRECISION NOT NULL, -- 10Y - 2Y sovereign spread
    cpi_yoy             DOUBLE PRECISION NOT NULL, -- CPI YoY inflation
    iip_yoy             DOUBLE PRECISION NOT NULL, -- IIP industrial output growth
    wpi_mom_vs_ma3m     DOUBLE PRECISION NOT NULL, -- WPI MoM surprise minus 3M MA
    real_rate_mom       DOUBLE PRECISION NOT NULL, -- 21-day real repo rate momentum
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

DO $$
BEGIN
    PERFORM create_hypertable('feature_store', 'time', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for feature_store.';
END $$;

-- ══════════════════════════════════════════════════════════════════════════
-- 4. REGIME CLASSIFICATION & POSTERIORS (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS regime_history (
    time            TIMESTAMPTZ PRIMARY KEY,
    regime          VARCHAR(16) NOT NULL, -- 'Bull', 'Bear', 'HighVol', 'Sideways'
    p_bull          DOUBLE PRECISION NOT NULL,
    p_bear          DOUBLE PRECISION NOT NULL,
    p_highvol       DOUBLE PRECISION NOT NULL,
    p_sideways      DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    is_transition   BOOLEAN DEFAULT FALSE,
    model_version   VARCHAR(32) DEFAULT 'HMM_v2.1',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

DO $$
BEGIN
    PERFORM create_hypertable('regime_history', 'time', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for regime_history.';
END $$;

CREATE INDEX IF NOT EXISTS idx_regime_history_regime ON regime_history (regime, time DESC);

-- ══════════════════════════════════════════════════════════════════════════
-- 5. ORDERS (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS orders (
    time            TIMESTAMPTZ NOT NULL,
    order_id        UUID NOT NULL,
    symbol          VARCHAR(16) NOT NULL,
    side            VARCHAR(4) NOT NULL, -- 'BUY' or 'SELL'
    quantity        INTEGER NOT NULL,
    target_weight   DOUBLE PRECISION,
    order_type      VARCHAR(16) NOT NULL, -- 'REBALANCE', 'STOP_LOSS', 'REENTRY'
    regime          VARCHAR(16) NOT NULL,
    status          VARCHAR(16) NOT NULL, -- 'SUBMITTED', 'FILLED', 'REJECTED'
    PRIMARY KEY (time, order_id)
);

DO $$
BEGIN
    PERFORM create_hypertable('orders', 'time', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for orders.';
END $$;

CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders (order_id);

-- ══════════════════════════════════════════════════════════════════════════
-- 6. EXECUTION FILLS & STATUTORY FRICTIONS (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS fills (
    time                TIMESTAMPTZ NOT NULL,
    fill_id             UUID NOT NULL,
    order_id            UUID NOT NULL,
    symbol              VARCHAR(16) NOT NULL,
    quantity            INTEGER NOT NULL,
    fill_price          DOUBLE PRECISION NOT NULL,
    gross_value         DOUBLE PRECISION NOT NULL,
    stt_tax             DOUBLE PRECISION NOT NULL,
    exchange_turnover   DOUBLE PRECISION NOT NULL,
    gst_tax             DOUBLE PRECISION NOT NULL,
    sebi_charges        DOUBLE PRECISION NOT NULL,
    stamp_duty          DOUBLE PRECISION NOT NULL,
    slippage_cost       DOUBLE PRECISION NOT NULL,
    total_friction      DOUBLE PRECISION NOT NULL,
    net_value           DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, fill_id)
);

DO $$
BEGIN
    PERFORM create_hypertable('fills', 'time', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for fills.';
END $$;

CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills (symbol, time DESC);

-- ══════════════════════════════════════════════════════════════════════════
-- 7. DAILY POSITIONS (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS positions (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(16) NOT NULL,
    units           INTEGER NOT NULL,
    average_cost    DOUBLE PRECISION NOT NULL,
    market_price    DOUBLE PRECISION NOT NULL,
    invested_value  DOUBLE PRECISION NOT NULL,
    market_value    DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION NOT NULL,
    weight_pct      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol)
);

DO $$
BEGIN
    PERFORM create_hypertable('positions', 'time', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for positions.';
END $$;

-- ══════════════════════════════════════════════════════════════════════════
-- 8. PORTFOLIO ACCOUNTING LEDGER (HYPERTABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS portfolio_ledger (
    time                TIMESTAMPTZ PRIMARY KEY,
    nav                 DOUBLE PRECISION NOT NULL,
    cash_balance        DOUBLE PRECISION NOT NULL,
    equity_value        DOUBLE PRECISION NOT NULL,
    peak_nav            DOUBLE PRECISION NOT NULL,
    drawdown_pct        DOUBLE PRECISION NOT NULL,
    daily_pnl           DOUBLE PRECISION NOT NULL,
    daily_return_pct    DOUBLE PRECISION NOT NULL,
    in_cash             BOOLEAN NOT NULL,
    stop_triggered      BOOLEAN DEFAULT FALSE,
    days_in_cash        INTEGER DEFAULT 0,
    active_regime       VARCHAR(16) NOT NULL,
    cumulative_friction DOUBLE PRECISION DEFAULT 0.0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

DO $$
BEGIN
    PERFORM create_hypertable('portfolio_ledger', 'time', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping create_hypertable for portfolio_ledger.';
END $$;

-- ══════════════════════════════════════════════════════════════════════════
-- 9. EOD RECONCILIATION AUDIT REPORTS (STANDARD RELATIONAL TABLE)
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS reconciliation_reports (
    report_date             DATE PRIMARY KEY,
    generated_at            TIMESTAMPTZ DEFAULT NOW(),
    status                  VARCHAR(16) NOT NULL, -- 'RECONCILED' or 'BREAK_DETECTED'
    position_breaks_count   INTEGER NOT NULL,
    cash_discrepancy_inr    DOUBLE PRECISION NOT NULL,
    nav_discrepancy_inr     DOUBLE PRECISION NOT NULL,
    stop_loss_integrity     VARCHAR(16) NOT NULL, -- 'PASS' or 'VIOLATION'
    audit_summary           JSONB NOT NULL,
    auditor_signature       VARCHAR(64) DEFAULT 'AUTO_RECON_ENGINE'
);
