import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    Base,
    MarketBar,
    MacroIndicator,
    FeatureStore,
    RegimeHistory,
    PortfolioLedger,
    Position,
)


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_market_bar_crud(in_memory_db):
    ts = datetime(2026, 10, 15, 15, 30)
    bar = MarketBar(
        time=ts,
        symbol="^NSEI",
        open=25000.0,
        high=25200.0,
        low=24950.0,
        close=25150.0,
        volume=1000000,
        returns=0.006,
        source="TEST",
    )
    in_memory_db.add(bar)
    in_memory_db.commit()

    retrieved = in_memory_db.query(MarketBar).filter_by(symbol="^NSEI").first()
    assert retrieved is not None
    assert retrieved.close == 25150.0
    assert retrieved.returns == 0.006


def test_macro_indicator_crud(in_memory_db):
    ts = datetime(2026, 10, 15)
    macro = MacroIndicator(
        time=ts,
        cpi_yoy=5.54,
        iip_yoy=8.00,
        wpi_yoy=11.39,
        repo_rate=5.50,
        gsec_10y=7.06,
        yield_curve=1.78,
    )
    in_memory_db.add(macro)
    in_memory_db.commit()

    retrieved = in_memory_db.query(MacroIndicator).first()
    assert retrieved is not None
    assert retrieved.cpi_yoy == 5.54
    assert retrieved.repo_rate == 5.50


def test_feature_store_crud(in_memory_db):
    ts = datetime(2026, 10, 15, 15, 30)
    feat = FeatureStore(
        time=ts,
        ret_5d_pct=1.5,
        vol_60d=14.2,
        price_vs_ma200=1.04,
        drawdown=-0.02,
        drawdown_diff20d=0.01,
        vix_vs_ma20=0.95,
        yield_curve=1.45,
        cpi_yoy=4.82,
        iip_yoy=5.20,
        wpi_mom_vs_ma3m=0.35,
        real_rate_mom=0.15,
    )
    in_memory_db.add(feat)
    in_memory_db.commit()

    retrieved = in_memory_db.query(FeatureStore).first()
    assert retrieved is not None
    assert retrieved.vol_60d == 14.2
    assert retrieved.wpi_mom_vs_ma3m == 0.35


def test_regime_history_crud(in_memory_db):
    ts = datetime(2026, 10, 15, 15, 30)
    reg = RegimeHistory(
        time=ts,
        regime="Bull",
        p_bull=0.92,
        p_bear=0.01,
        p_highvol=0.02,
        p_sideways=0.05,
        confidence=0.92,
        is_transition=False,
    )
    in_memory_db.add(reg)
    in_memory_db.commit()

    retrieved = in_memory_db.query(RegimeHistory).first()
    assert retrieved is not None
    assert retrieved.regime == "Bull"
    assert retrieved.confidence == 0.92


def test_portfolio_ledger_and_position(in_memory_db):
    ts = datetime(2026, 10, 15, 15, 30)
    ledger = PortfolioLedger(
        time=ts,
        nav=1050000.0,
        cash_balance=50000.0,
        equity_value=1000000.0,
        peak_nav=1060000.0,
        drawdown_pct=-0.0094,
        daily_pnl=5000.0,
        daily_return_pct=0.0048,
        in_cash=False,
        stop_triggered=False,
        active_regime="Bull",
    )
    pos = Position(
        time=ts,
        symbol="BANKBEES",
        units=1000,
        average_cost=580.0,
        market_price=600.0,
        invested_value=580000.0,
        market_value=600000.0,
        unrealized_pnl=20000.0,
        weight_pct=600000.0 / 1050000.0,
    )
    in_memory_db.add_all([ledger, pos])
    in_memory_db.commit()

    retrieved_ledger = in_memory_db.query(PortfolioLedger).first()
    retrieved_pos = in_memory_db.query(Position).filter_by(symbol="BANKBEES").first()

    assert retrieved_ledger.nav == 1050000.0
    assert retrieved_pos.units == 1000
    assert retrieved_pos.unrealized_pnl == 20000.0
