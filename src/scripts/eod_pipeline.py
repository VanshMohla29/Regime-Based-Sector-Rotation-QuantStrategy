"""
Automated End-of-Day (EOD) Market & Portfolio Execution Pipeline.
==================================================================
Runs at 18:00 IST (NSE market close settlement):
  1. Synchronizes latest official macro releases and daily market bars into TimescaleDB.
  2. Updates paper portfolio forward simulation and ETF sector allocations.
  3. Records accounting ledger and active positions in TimescaleDB.
  4. Executes daily trade and cash reconciliation audit.
  5. Dispatches executive status briefing via Telegram (if configured).
"""

import os
import sys
import json
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

# Ensure src and repo root are on sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'data')) else SRC_DIR
DATA_DIR = os.path.join(REPO_ROOT, "data") if os.path.exists(os.path.join(REPO_ROOT, "data")) else REPO_ROOT

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from database.connection import get_db, init_db
from database.models import PortfolioLedger, Position
from database.ingest_service import auto_sync_macro_from_sources, ingest_daily_update
from reconciliation.reconciler import DailyReconciliationEngine
import paper_portfolio


def sync_latest_portfolio_to_db():
    """
    Syncs paper_portfolio_history.csv into portfolio_ledger, positions, orders, and fills.
    Guarantees full relational integrity between daily ledgers, open positions,
    and executed fills so daily reconciliation never breaks.
    """
    csv_path = os.path.join(DATA_DIR, "paper_portfolio_history.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(REPO_ROOT, "paper_portfolio_history.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    sector_names = ['BANKBEES', 'ITBEES', 'PHARMABEES', 'AUTOBEES', 'METALIETF', 'MOREALTY', 'CPSEETF', 'INFRABEES']

    from reconciliation.reconciler import OrderManagementSystem
    from database.models import Order, Fill

    with get_db() as db:
        prev_units = {sec: 0 for sec in sector_names}

        for i, row in df.iterrows():
            ts = row['Date'].to_pydatetime()
            nav = float(row['NAV'])

            existing_ledger = db.query(PortfolioLedger).filter(PortfolioLedger.time == ts).first()
            if not existing_ledger:
                ledger_entry = PortfolioLedger(
                    time=ts,
                    nav=nav,
                    cash_balance=float(row['Cash']),
                    equity_value=float(row['Equity']),
                    peak_nav=float(df['NAV'].iloc[:i+1].max()),
                    drawdown_pct=float(row['Drawdown']),
                    daily_pnl=float(row.get('Unrealized_Gains', 0.0)),
                    daily_return_pct=float(row.get('Daily_Return', 0.0)),
                    in_cash=bool(row.get('In_Cash', False)),
                    stop_triggered="STOP" in str(row['Regime']),
                    days_in_cash=0,
                    active_regime=str(row['Regime']),
                    cumulative_friction=float(row.get('Total_Charges', 0.0)),
                )
                db.add(ledger_entry)
            else:
                existing_ledger.nav = nav
                existing_ledger.cash_balance = float(row['Cash'])
                existing_ledger.equity_value = float(row['Equity'])
                existing_ledger.peak_nav = float(df['NAV'].iloc[:i+1].max())
                existing_ledger.drawdown_pct = float(row['Drawdown'])
                existing_ledger.daily_pnl = float(row.get('Unrealized_Gains', 0.0))
                existing_ledger.daily_return_pct = float(row.get('Daily_Return', 0.0))
                existing_ledger.in_cash = bool(row.get('In_Cash', False))
                existing_ledger.stop_triggered = "STOP" in str(row['Regime'])
                existing_ledger.active_regime = str(row['Regime'])
                existing_ledger.cumulative_friction = float(row.get('Total_Charges', 0.0))

            # Clean existing positions, orders & fills for this session date before inserting current records
            db.query(Position).filter(Position.time == ts).delete()
            db.query(Fill).filter(Fill.time == ts).delete()
            db.query(Order).filter(Order.time == ts).delete()

            # Record Orders, Fills, and Positions
            for sec in sector_names:
                curr_u = int(row.get(f"{sec}_units", 0))
                p_u = prev_units[sec]
                px = float(row.get(f"{sec}_price", 0.0))

                if curr_u > p_u and px > 0:
                    OrderManagementSystem.execute_order(
                        db=db, timestamp=ts, symbol=sec, side="BUY",
                        quantity=curr_u - p_u, fill_price=px,
                        order_type="REENTRY" if "STOP" not in str(row['Regime']) else "REBALANCE",
                        regime=str(row['Regime']), target_weight=None
                    )
                elif curr_u < p_u and px > 0:
                    OrderManagementSystem.execute_order(
                        db=db, timestamp=ts, symbol=sec, side="SELL",
                        quantity=p_u - curr_u, fill_price=px,
                        order_type="STOP_LOSS" if "STOP" in str(row['Regime']) else "REBALANCE",
                        regime=str(row['Regime']), target_weight=None
                    )

                if curr_u > 0:
                    mkt_val = float(row.get(f"{sec}_value", 0.0))
                    avg_cost = float(row.get(f"{sec}_cost_basis", px))
                    unrealized = float(row.get(f"{sec}_pnl", 0.0))
                    weight = (mkt_val / nav) if nav > 0 else 0.0

                    db.add(
                        Position(
                            time=ts,
                            symbol=sec,
                            units=curr_u,
                            average_cost=avg_cost,
                            market_price=px,
                            invested_value=curr_u * avg_cost,
                            market_value=mkt_val,
                            unrealized_pnl=unrealized,
                            weight_pct=weight,
                        )
                    )
                prev_units[sec] = curr_u

        db.commit()
    latest_ts = df.iloc[-1]['Date'].to_pydatetime()
    return latest_ts


def dispatch_telegram_eod_summary(report_summary: dict, nav: float, regime: str, ret_pct: float):
    """Sends executive EOD briefing to Telegram if bot credentials are provided."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    status_emoji = "✅" if report_summary.get("status") == "RECONCILED" else "⚠️"
    msg = (
        f"📊 <b>EOD Quant Strategy Briefing</b> ({report_summary.get('date')})\n\n"
        f"• <b>Active Regime</b>: {regime}\n"
        f"• <b>Portfolio NAV</b>: ₹{nav:,.2f}\n"
        f"• <b>Daily Return</b>: {ret_pct:+.2f}%\n"
        f"• <b>Reconciliation</b>: {status_emoji} {report_summary.get('status')}\n"
        f"• <b>Breaks</b>: {report_summary.get('position_breaks', 0)} | Discrepancy: ₹{report_summary.get('cash_discrepancy', 0.0):.2f}\n"
        f"• <b>Stop-Loss Integrity</b>: {report_summary.get('stop_loss_integrity')}\n"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10):
            print("✓ Telegram EOD briefing dispatched successfully.")
    except Exception as e:
        print(f"  [warn] Could not send Telegram briefing: {e}")


def run_eod_pipeline():
    """Executes the full end-of-day quant rotation and reconciliation sequence."""
    start_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    print("\n" + "=" * 75)
    print(f"  STARTING EOD QUANT PIPELINE  ·  {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 75)

    # 1. Initialize DB tables if first run
    init_db()

    # 2. Automated macro & market data ingestion
    print("\n[Step 1/4] Automated Macro Sync & Daily Market Bar Ingestion...")
    try:
        auto_sync_macro_from_sources()
        ingest_res = ingest_daily_update(is_eod=True)
        print(f"✓ Data Ingestion Status: {ingest_res.get('status')}")
    except Exception as e:
        print(f"  [warn] Ingestion warning: {e}")

    # 3. Paper Portfolio Forward Simulation
    print("\n[Step 2/4] Paper Portfolio Simulation & ETF Allocation Update...")
    try:
        paper_portfolio.main()
        latest_ts = sync_latest_portfolio_to_db()
        print(f"✓ Portfolio simulation complete and synced to database (Timestamp: {latest_ts}).")
    except Exception as e:
        print(f"  [error] Portfolio simulation error: {e}")

    # 4. Daily Reconciliation Audit
    print("\n[Step 3/4] Running Daily Trade & Ledger Reconciliation Audit...")
    audit_summary = {"status": "SKIPPED", "date": start_time.strftime("%Y-%m-%d")}
    nav_val = 10_00_000.0
    active_regime = "Unknown"
    ret_pct = 0.0

    try:
        with get_db() as db:
            latest_ledger = db.query(PortfolioLedger).order_by(PortfolioLedger.time.desc()).first()
            if latest_ledger:
                target_date = latest_ledger.time.date()
                report = DailyReconciliationEngine.reconcile(db, target_date)
                audit_summary = report.audit_summary
                nav_val = latest_ledger.nav
                active_regime = latest_ledger.active_regime
                ret_pct = latest_ledger.daily_return_pct * 100.0
                print(f"✓ Reconciliation Audit Result: {report.status}")
                print(f"  Cash discrepancy : ₹{report.cash_discrepancy_inr:,.2f}")
                print(f"  Position breaks  : {report.position_breaks_count}")
                print(f"  Stop-loss check  : {report.stop_loss_integrity}")
    except Exception as e:
        print(f"  [error] Reconciliation error: {e}")

    # 5. Telegram Dispatch
    print("\n[Step 4/4] Dispatching Executive Briefing...")
    dispatch_telegram_eod_summary(audit_summary, nav_val, active_regime, ret_pct)

    elapsed = (datetime.now(ZoneInfo("Asia/Kolkata")) - start_time).total_seconds()
    print("=" * 75)
    print(f"  ✓ EOD PIPELINE FINISHED IN {elapsed:.2f}s")
    print("=" * 75 + "\n")
    return {"status": "SUCCESS", "elapsed_seconds": elapsed, "audit": audit_summary}


if __name__ == "__main__":
    run_eod_pipeline()
