import uuid
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.models import (
    Order,
    Fill,
    Position,
    PortfolioLedger,
    ReconciliationReport,
)


class OrderManagementSystem:
    """
    Institutional Order Management & Execution System (OMS).
    Handles atomic execution of target allocations, stop-losses, and cash re-entries,
    computing the exact statutory Indian charges and recording full audit fills.
    """

    # Statutory Indian regulatory charges (Cash Delivery Equity)
    RATE_STT_SELL = 0.0010       # 0.10% STT on sell side
    RATE_TURNOVER = 0.0000345    # 0.00345% NSE exchange turnover
    RATE_GST = 0.18              # 18% GST on exchange turnover + brokerage
    RATE_SEBI = 0.000001         # ₹10 per crore (0.0001%)
    RATE_STAMP_BUY = 0.00015     # 0.015% stamp duty on buy side
    RATE_SLIPPAGE = 0.0005       # 0.05% market impact / slippage

    @classmethod
    def compute_trade_friction(cls, side: str, gross_value: float) -> Dict[str, float]:
        """Calculates itemized statutory taxes and exchange frictions for a trade."""
        stt = gross_value * cls.RATE_STT_SELL if side.upper() == "SELL" else 0.0
        turnover = gross_value * cls.RATE_TURNOVER
        gst = turnover * cls.RATE_GST
        sebi = gross_value * cls.RATE_SEBI
        stamp = gross_value * cls.RATE_STAMP_BUY if side.upper() == "BUY" else 0.0
        slippage = gross_value * cls.RATE_SLIPPAGE

        total = stt + turnover + gst + sebi + stamp + slippage
        return {
            "stt": stt,
            "turnover": turnover,
            "gst": gst,
            "sebi": sebi,
            "stamp": stamp,
            "slippage": slippage,
            "total": total,
        }

    @classmethod
    def execute_order(
        cls,
        db: Session,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        order_type: str,
        regime: str,
        target_weight: Optional[float] = None,
    ) -> Fill:
        """
        Atomically creates an Order and matching Fill, computes frictions,
        and saves them to the database.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        order_id = str(uuid.uuid4())
        fill_id = str(uuid.uuid4())
        gross_val = quantity * fill_price

        friction = cls.compute_trade_friction(side, gross_val)
        net_val = (gross_val - friction["total"]) if side.upper() == "SELL" else (gross_val + friction["total"])

        order = Order(
            time=timestamp,
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            target_weight=target_weight,
            order_type=order_type,
            regime=regime,
            status="FILLED",
        )

        fill = Fill(
            time=timestamp,
            fill_id=fill_id,
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            fill_price=fill_price,
            gross_value=gross_val,
            stt_tax=friction["stt"],
            exchange_turnover=friction["turnover"],
            gst_tax=friction["gst"],
            sebi_charges=friction["sebi"],
            stamp_duty=friction["stamp"],
            slippage_cost=friction["slippage"],
            total_friction=friction["total"],
            net_value=net_val,
        )

        db.add(order)
        db.add(fill)
        return fill


class DailyReconciliationEngine:
    """
    Automated End-Of-Day (EOD) Reconciliation & Audit Engine.
    Executes 4 rigorous institutional checks across positions, cash, NAV, and stops.
    """

    TOLERANCE_CASH = 0.05  # ₹0.05 tolerance for floating-point rounding
    TOLERANCE_NAV = 0.10   # ₹0.10 tolerance

    @classmethod
    def reconcile(cls, db: Session, target_date: date) -> ReconciliationReport:
        """
        Audits the portfolio on `target_date` and stores the official
        ReconciliationReport in the database.
        """
        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date, datetime.max.time())

        # 1. Fetch ledger snapshot for target_date
        ledger = (
            db.query(PortfolioLedger)
            .filter(PortfolioLedger.time >= start_dt, PortfolioLedger.time <= end_dt)
            .order_by(PortfolioLedger.time.desc())
            .first()
        )

        if not ledger:
            # If no ledger entry for target date, check most recent available
            ledger = db.query(PortfolioLedger).order_by(PortfolioLedger.time.desc()).first()
            if not ledger:
                # Return empty/default report if database was just initialized
                return ReconciliationReport(
                    report_date=target_date,
                    status="RECONCILED",
                    position_breaks_count=0,
                    cash_discrepancy_inr=0.0,
                    nav_discrepancy_inr=0.0,
                    stop_loss_integrity="PASS",
                    audit_summary={
                        "date": target_date.isoformat(),
                        "status": "RECONCILED",
                        "nav": 1000000.0,
                        "cash_balance": 1000000.0,
                        "equity_value": 0.0,
                        "computed_equity": 0.0,
                        "nav_discrepancy_inr": 0.0,
                        "cash_discrepancy_inr": 0.0,
                        "position_breaks": 0,
                        "break_details": {},
                        "stop_loss_integrity": "PASS",
                        "in_cash": True,
                        "drawdown_pct": 0.0,
                        "active_regime": "None",
                        "positions_count": 0,
                    },
                    auditor_signature="AUTO_RECON_ENGINE_v1",
                )

        # 2. Fetch all recorded active positions on this date
        positions = (
            db.query(Position)
            .filter(Position.time >= start_dt, Position.time <= end_dt)
            .all()
        )

        # 3. Check 1: NAV Identity: NAV = Cash + sum(Market Values)
        computed_equity = sum(p.market_value for p in positions)
        expected_nav = ledger.cash_balance + computed_equity
        nav_diff = abs(ledger.nav - expected_nav)

        # 4. Check 2: Cumulative Fills vs Recorded Positions (Position Breaks)
        position_breaks = 0
        break_details = {}

        # Get all distinct symbols traded up to target date
        symbols = [p.symbol for p in positions]
        for p in positions:
            # Calculate cumulative buys minus sells up to this date
            buys = (
                db.query(func.coalesce(func.sum(Fill.quantity), 0))
                .filter(Fill.symbol == p.symbol, Fill.time <= end_dt)
                .join(Order, Fill.order_id == Order.order_id)
                .filter(Order.side == "BUY")
                .scalar()
            )
            sells = (
                db.query(func.coalesce(func.sum(Fill.quantity), 0))
                .filter(Fill.symbol == p.symbol, Fill.time <= end_dt)
                .join(Order, Fill.order_id == Order.order_id)
                .filter(Order.side == "SELL")
                .scalar()
            )
            cleared_units = buys - sells

            # If fills exist for this symbol, compare against cleared units
            if (buys > 0 or sells > 0) and p.units != cleared_units:
                position_breaks += 1
                break_details[p.symbol] = {
                    "recorded_units": p.units,
                    "cleared_units": cleared_units,
                    "discrepancy": p.units - cleared_units,
                }

        # 5. Check 3: Cash Conservation
        # Fetch prior day's ledger to verify flow of funds
        prev_ledger = (
            db.query(PortfolioLedger)
            .filter(PortfolioLedger.time < start_dt)
            .order_by(PortfolioLedger.time.desc())
            .first()
        )

        cash_discrepancy = 0.0
        if prev_ledger:
            # Today's net cash flow from fills
            today_fills = (
                db.query(Fill)
                .filter(Fill.time >= start_dt, Fill.time <= end_dt)
                .join(Order, Fill.order_id == Order.order_id)
                .all()
            )
            net_trade_cash = 0.0
            for f in today_fills:
                # Sell adds gross - friction; Buy subtracts gross + friction
                fill_order = db.query(Order).filter(Order.order_id == f.order_id).first()
                if fill_order and fill_order.side == "SELL":
                    net_trade_cash += f.net_value
                else:
                    net_trade_cash -= f.net_value

            # Daily repo interest earned on cash if in cash
            repo_daily = (0.055 / 252.0) if ledger.in_cash else 0.0
            interest_earned = prev_ledger.cash_balance * repo_daily

            theoretical_cash = prev_ledger.cash_balance + net_trade_cash + interest_earned
            cash_discrepancy = abs(ledger.cash_balance - theoretical_cash)

        # 6. Check 4: Stop-Loss Integrity
        stop_integrity = "PASS"
        if ledger.drawdown_pct <= -0.06:
            # Must be in 100% cash with 0 equity exposure
            total_shares = sum(p.units for p in positions)
            if not ledger.in_cash or total_shares > 0 or ledger.equity_value > 0.01:
                stop_integrity = "VIOLATION"

        # Overall Reconciliation Status
        is_reconciled = (
            position_breaks == 0
            and nav_diff <= cls.TOLERANCE_NAV
            and cash_discrepancy <= cls.TOLERANCE_CASH
            and stop_integrity == "PASS"
        )
        status = "RECONCILED" if is_reconciled else "BREAK_DETECTED"

        audit_summary = {
            "date": target_date.isoformat(),
            "status": status,
            "nav": round(ledger.nav, 2),
            "cash_balance": round(ledger.cash_balance, 2),
            "equity_value": round(ledger.equity_value, 2),
            "computed_equity": round(computed_equity, 2),
            "nav_discrepancy_inr": round(nav_diff, 4),
            "cash_discrepancy_inr": round(cash_discrepancy, 4),
            "position_breaks": position_breaks,
            "break_details": break_details,
            "stop_loss_integrity": stop_integrity,
            "in_cash": ledger.in_cash,
            "drawdown_pct": round(ledger.drawdown_pct * 100, 2),
            "active_regime": ledger.active_regime,
            "positions_count": len(positions),
        }

        # Save or update report in DB
        report = (
            db.query(ReconciliationReport)
            .filter(ReconciliationReport.report_date == target_date)
            .first()
        )
        if not report:
            report = ReconciliationReport(
                report_date=target_date,
                status=status,
                position_breaks_count=position_breaks,
                cash_discrepancy_inr=round(cash_discrepancy, 4),
                nav_discrepancy_inr=round(nav_diff, 4),
                stop_loss_integrity=stop_integrity,
                audit_summary=audit_summary,
                auditor_signature="AUTO_RECON_ENGINE_v1",
            )
            db.add(report)
        else:
            report.status = status
            report.position_breaks_count = position_breaks
            report.cash_discrepancy_inr = round(cash_discrepancy, 4)
            report.nav_discrepancy_inr = round(nav_diff, 4)
            report.stop_loss_integrity = stop_integrity
            report.audit_summary = audit_summary
            report.generated_at = datetime.utcnow()

        db.flush()
        return report
