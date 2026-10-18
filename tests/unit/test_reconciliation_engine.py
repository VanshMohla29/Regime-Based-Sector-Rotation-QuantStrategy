import pytest
from datetime import datetime, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    Base,
    Order,
    Fill,
    Position,
    PortfolioLedger,
)
from reconciliation.reconciler import OrderManagementSystem, DailyReconciliationEngine


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_oms_friction_calculation_buy_vs_sell():
    gross = 100_000.0
    buy_friction = OrderManagementSystem.compute_trade_friction("BUY", gross)
    sell_friction = OrderManagementSystem.compute_trade_friction("SELL", gross)

    # Buy should have stamp duty (0.015%), no STT
    assert buy_friction["stamp"] == pytest.approx(15.0)
    assert buy_friction["stt"] == 0.0

    # Sell should have STT (0.10%), no stamp duty
    assert sell_friction["stt"] == pytest.approx(100.0)
    assert sell_friction["stamp"] == 0.0

    # Both have turnover, gst, sebi, slippage
    assert buy_friction["slippage"] == pytest.approx(50.0)
    assert sell_friction["slippage"] == pytest.approx(50.0)
    assert buy_friction["total"] > 0
    assert sell_friction["total"] > buy_friction["total"]  # STT (100) > Stamp (15)


def test_oms_execute_order(test_db):
    ts = datetime(2026, 10, 15, 10, 0)
    fill = OrderManagementSystem.execute_order(
        db=test_db,
        timestamp=ts,
        symbol="ITBEES",
        side="BUY",
        quantity=500,
        fill_price=40.0,
        order_type="REBALANCE",
        regime="Bull",
        target_weight=0.20,
    )
    test_db.commit()

    assert fill is not None
    assert isinstance(fill, Fill)
    assert fill.symbol == "ITBEES"
    assert fill.gross_value == 20_000.0
    assert fill.net_value > 20_000.0  # Buy adds friction

    order = test_db.query(Order).filter_by(order_id=fill.order_id).first()
    assert order is not None
    assert order.side == "BUY"
    assert order.quantity == 500


def test_reconciler_clean_pass(test_db):
    target_dt = date(2026, 10, 15)
    ts = datetime(2026, 10, 15, 15, 30)

    # Setup matching ledger and positions
    ledger = PortfolioLedger(
        time=ts,
        nav=1_000_000.0,
        cash_balance=200_000.0,
        equity_value=800_000.0,
        peak_nav=1_000_000.0,
        drawdown_pct=0.0,
        daily_pnl=0.0,
        daily_return_pct=0.0,
        in_cash=False,
        active_regime="Bull",
    )
    pos = Position(
        time=ts,
        symbol="BANKBEES",
        units=1000,
        average_cost=800.0,
        market_price=800.0,
        invested_value=800_000.0,
        market_value=800_000.0,
        unrealized_pnl=0.0,
        weight_pct=0.80,
    )
    test_db.add_all([ledger, pos])
    test_db.commit()

    report = DailyReconciliationEngine.reconcile(test_db, target_dt)
    assert report.status == "RECONCILED"
    assert report.position_breaks_count == 0
    assert report.nav_discrepancy_inr == 0.0
    assert report.stop_loss_integrity == "PASS"


def test_reconciler_detects_position_break(test_db):
    target_dt = date(2026, 10, 15)
    ts = datetime(2026, 10, 15, 15, 30)

    # 1. Execute buy of 1000 units
    OrderManagementSystem.execute_order(
        db=test_db,
        timestamp=ts,
        symbol="PHARMABEES",
        side="BUY",
        quantity=1000,
        fill_price=30.0,
        order_type="REBALANCE",
        regime="Bear",
    )

    # 2. Record corrupted position holding 1200 units instead of 1000
    pos = Position(
        time=ts,
        symbol="PHARMABEES",
        units=1200,  # 200 unit phantom break!
        average_cost=30.0,
        market_price=30.0,
        invested_value=36_000.0,
        market_value=36_000.0,
        unrealized_pnl=0.0,
        weight_pct=0.036,
    )
    ledger = PortfolioLedger(
        time=ts,
        nav=1_000_000.0,
        cash_balance=964_000.0,
        equity_value=36_000.0,
        peak_nav=1_000_000.0,
        drawdown_pct=0.0,
        daily_pnl=0.0,
        daily_return_pct=0.0,
        in_cash=False,
        active_regime="Bear",
    )
    test_db.add_all([pos, ledger])
    test_db.commit()

    report = DailyReconciliationEngine.reconcile(test_db, target_dt)
    assert report.status == "BREAK_DETECTED"
    assert report.position_breaks_count == 1
    assert "PHARMABEES" in report.audit_summary["break_details"]
    assert report.audit_summary["break_details"]["PHARMABEES"]["discrepancy"] == 200


def test_reconciler_detects_stop_loss_violation(test_db):
    target_dt = date(2026, 10, 15)
    ts = datetime(2026, 10, 15, 15, 30)

    # Portfolio suffered -7% drawdown but failed to de-risk into cash
    ledger = PortfolioLedger(
        time=ts,
        nav=930_000.0,
        cash_balance=130_000.0,
        equity_value=800_000.0,  # Still holding equity!
        peak_nav=1_000_000.0,
        drawdown_pct=-0.07,     # -7% breaches -6% stop threshold
        daily_pnl=-70_000.0,
        daily_return_pct=-0.07,
        in_cash=False,          # Should be True!
        active_regime="Bear",
    )
    pos = Position(
        time=ts,
        symbol="BANKBEES",
        units=1000,
        average_cost=800.0,
        market_price=800.0,
        invested_value=800_000.0,
        market_value=800_000.0,
        unrealized_pnl=0.0,
        weight_pct=0.86,
    )
    test_db.add_all([ledger, pos])
    test_db.commit()

    report = DailyReconciliationEngine.reconcile(test_db, target_dt)
    assert report.status == "BREAK_DETECTED"
    assert report.stop_loss_integrity == "VIOLATION"


def test_reconciler_passes_valid_reentry_after_stop_loss(test_db):
    ts_cash = datetime(2026, 10, 14, 15, 30)
    ts_reenter = datetime(2026, 10, 15, 15, 30)
    target_dt = date(2026, 10, 15)

    # 1. Day 1: Portfolio stopped out in 100% cash
    ledger_cash = PortfolioLedger(
        time=ts_cash,
        nav=940_000.0,
        cash_balance=940_000.0,
        equity_value=0.0,
        peak_nav=1_000_000.0,
        drawdown_pct=-0.06,
        daily_pnl=0.0,
        daily_return_pct=0.0,
        in_cash=True,
        stop_triggered=True,
        active_regime="CASH (STOP -6%)",
    )
    test_db.add(ledger_cash)
    test_db.commit()

    # 2. Day 2: Execute valid re-entry buy order via OMS
    fill = OrderManagementSystem.execute_order(
        db=test_db,
        timestamp=ts_reenter,
        symbol="BANKBEES",
        side="BUY",
        quantity=1000,
        fill_price=900.0,
        order_type="REENTRY",
        regime="Bear",
        target_weight=0.95,
    )
    remaining_cash = 940_000.0 - fill.net_value

    ledger_reentry = PortfolioLedger(
        time=ts_reenter,
        nav=remaining_cash + 900_000.0,
        cash_balance=remaining_cash,
        equity_value=900_000.0,
        peak_nav=1_000_000.0,  # All-time peak
        drawdown_pct=-0.055,   # All-time drawdown from 1M
        daily_pnl=0.0,
        daily_return_pct=0.0,
        in_cash=False,
        stop_triggered=False,
        active_regime="Bear",
    )
    pos = Position(
        time=ts_reenter,
        symbol="BANKBEES",
        units=1000,
        average_cost=900.0,
        market_price=900.0,
        invested_value=900_000.0,
        market_value=900_000.0,
        unrealized_pnl=0.0,
        weight_pct=900_000.0 / (remaining_cash + 900_000.0),
    )
    test_db.add_all([ledger_reentry, pos])
    test_db.commit()

    report = DailyReconciliationEngine.reconcile(test_db, target_dt)
    assert report.status == "RECONCILED"
    assert report.stop_loss_integrity == "PASS"
    assert report.position_breaks_count == 0
    assert report.nav_discrepancy_inr == 0.0
    assert report.cash_discrepancy_inr == 0.0
