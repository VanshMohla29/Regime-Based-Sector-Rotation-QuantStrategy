import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    BigInteger,
    Boolean,
    DateTime,
    Date,
    JSON,
    PrimaryKeyConstraint,
)
from .connection import Base


class MarketBar(Base):
    __tablename__ = "market_bars"

    time = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(16), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, default=0)
    returns = Column(Float, nullable=True)
    source = Column(String(16), default="YAHOO")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        PrimaryKeyConstraint("time", "symbol"),
    )


class SectorReturn(Base):
    """
    Sanitized daily log returns for the 8 NSE sector ETFs.
    Bad-tick filtering and bluechip/index proxy backfill (pre-listing history)
    are applied at ingestion, so training reads this table directly with no
    network access.
    """
    __tablename__ = "sector_returns"

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    bankbees = Column(Float, nullable=False)
    itbees = Column(Float, nullable=False)
    pharmabees = Column(Float, nullable=False)
    autobees = Column(Float, nullable=False)
    metalietf = Column(Float, nullable=False)
    morealty = Column(Float, nullable=False)
    cpseetf = Column(Float, nullable=False)
    infrabees = Column(Float, nullable=False)
    source = Column(String(16), default="YAHOO")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class MacroIndicator(Base):
    __tablename__ = "macro_indicators"

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    cpi_yoy = Column(Float, nullable=True)
    cpi_mom = Column(Float, nullable=True)
    wpi_yoy = Column(Float, nullable=True)
    wpi_mom = Column(Float, nullable=True)
    iip_yoy = Column(Float, nullable=True)
    repo_rate = Column(Float, nullable=True)
    gsec_10y = Column(Float, nullable=True)
    gsec_2y = Column(Float, nullable=True)
    yield_curve = Column(Float, nullable=True)
    real_policy_rate = Column(Float, nullable=True)
    pub_lag_offset_days = Column(Integer, default=30)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class FeatureStore(Base):
    __tablename__ = "feature_store"

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    ret_5d_pct = Column(Float, nullable=False)
    vol_60d = Column(Float, nullable=False)
    price_vs_ma200 = Column(Float, nullable=False)
    drawdown = Column(Float, nullable=False)
    drawdown_diff20d = Column(Float, nullable=False)
    vix_vs_ma20 = Column(Float, nullable=False)
    yield_curve = Column(Float, nullable=False)
    cpi_yoy = Column(Float, nullable=False)
    iip_yoy = Column(Float, nullable=False)
    wpi_mom_vs_ma3m = Column(Float, nullable=False)
    real_rate_mom = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class RegimeHistory(Base):
    __tablename__ = "regime_history"

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    regime = Column(String(16), nullable=False)
    p_bull = Column(Float, nullable=False)
    p_bear = Column(Float, nullable=False)
    p_highvol = Column(Float, nullable=False)
    p_sideways = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    is_transition = Column(Boolean, default=False)
    model_version = Column(String(32), default="HMM_v2.1")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Order(Base):
    __tablename__ = "orders"

    time = Column(DateTime(timezone=True), nullable=False)
    order_id = Column(String(36), default=lambda: str(uuid.uuid4()), nullable=False)
    symbol = Column(String(16), nullable=False)
    side = Column(String(4), nullable=False)  # BUY or SELL
    quantity = Column(Integer, nullable=False)
    target_weight = Column(Float, nullable=True)
    order_type = Column(String(16), nullable=False)  # REBALANCE, STOP_LOSS, REENTRY
    regime = Column(String(16), nullable=False)
    status = Column(String(16), default="FILLED", nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("time", "order_id"),
    )


class Fill(Base):
    __tablename__ = "fills"

    time = Column(DateTime(timezone=True), nullable=False)
    fill_id = Column(String(36), default=lambda: str(uuid.uuid4()), nullable=False)
    order_id = Column(String(36), nullable=False)
    symbol = Column(String(16), nullable=False)
    quantity = Column(Integer, nullable=False)
    fill_price = Column(Float, nullable=False)
    gross_value = Column(Float, nullable=False)
    stt_tax = Column(Float, default=0.0, nullable=False)
    exchange_turnover = Column(Float, default=0.0, nullable=False)
    gst_tax = Column(Float, default=0.0, nullable=False)
    sebi_charges = Column(Float, default=0.0, nullable=False)
    stamp_duty = Column(Float, default=0.0, nullable=False)
    slippage_cost = Column(Float, default=0.0, nullable=False)
    total_friction = Column(Float, default=0.0, nullable=False)
    net_value = Column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("time", "fill_id"),
    )


class Position(Base):
    __tablename__ = "positions"

    time = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(16), nullable=False)
    units = Column(Integer, nullable=False)
    average_cost = Column(Float, nullable=False)
    market_price = Column(Float, nullable=False)
    invested_value = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    weight_pct = Column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("time", "symbol"),
    )


class PortfolioLedger(Base):
    __tablename__ = "portfolio_ledger"

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    nav = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    equity_value = Column(Float, nullable=False)
    peak_nav = Column(Float, nullable=False)
    drawdown_pct = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    daily_return_pct = Column(Float, nullable=False)
    in_cash = Column(Boolean, nullable=False)
    stop_triggered = Column(Boolean, default=False)
    days_in_cash = Column(Integer, default=0)
    active_regime = Column(String(16), nullable=False)
    cumulative_friction = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ReconciliationReport(Base):
    __tablename__ = "reconciliation_reports"

    report_date = Column(Date, primary_key=True, nullable=False)
    generated_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    status = Column(String(16), nullable=False)  # RECONCILED or BREAK_DETECTED
    position_breaks_count = Column(Integer, default=0, nullable=False)
    cash_discrepancy_inr = Column(Float, default=0.0, nullable=False)
    nav_discrepancy_inr = Column(Float, default=0.0, nullable=False)
    stop_loss_integrity = Column(String(16), default="PASS", nullable=False)  # PASS or VIOLATION
    audit_summary = Column(JSON, nullable=False)
    auditor_signature = Column(String(64), default="AUTO_RECON_ENGINE")
