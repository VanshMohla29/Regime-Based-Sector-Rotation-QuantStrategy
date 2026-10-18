"""
Database Seeding & Historical Migration Script.
Populates PostgreSQL + TimescaleDB (or SQLite fallback) from:
- regime_history_v2.csv (11-year daily macro, market returns, regime posteriors)
- paper_portfolio_history.csv (Forward live ledger snapshots & ETF positions)
- scratch/snapshot.pkl (Engineered feature store records)
"""

import os
import sys
import pickle
import pandas as pd
from datetime import datetime

# Set path to project root
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'data')) else SRC_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data") if os.path.exists(os.path.join(ROOT_DIR, "data")) else ROOT_DIR

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database.connection import init_db, get_db
from database.models import (
    MarketBar,
    MacroIndicator,
    FeatureStore,
    RegimeHistory,
    PortfolioLedger,
    Position,
)
from reconciliation.reconciler import DailyReconciliationEngine


def seed_database():
    print("=" * 70)
    print("  INITIALIZING TIMESCALEDB / RELATIONAL DATABASE SEEDING")
    print("=" * 70)

    # 1. Initialize DB tables / hypertables
    init_db()
    print("✓ Initialized database tables and TimescaleDB hypertables.")

    # 2. Seed from regime_history_v2.csv
    reg_csv = os.path.join(DATA_DIR, "regime_history_v2.csv")
    if not os.path.exists(reg_csv):
        reg_csv = os.path.join(ROOT_DIR, "regime_history_v2.csv")
    if os.path.exists(reg_csv):
        df_reg = pd.read_csv(reg_csv)
        df_reg['Date'] = pd.to_datetime(df_reg['Date'])
        print(f"✓ Found regime_history_v2.csv with {len(df_reg)} records.")

        with get_db() as db:
            existing_count = db.query(RegimeHistory).count()
            if existing_count < len(df_reg):
                db.query(RegimeHistory).delete()
                db.query(MarketBar).filter(MarketBar.symbol == "^NSEI").delete()
                db.query(MacroIndicator).delete()
                print(f"  Seeding {len(df_reg)} regime_history, market_bars (NIFTY), and macro_indicators sessions...")
                reg_objects = []
                bar_objects = []
                macro_objects = []

                for _, row in df_reg.iterrows():
                    ts = row['Date'].to_pydatetime()

                    # Regime History
                    reg_objects.append(
                        RegimeHistory(
                            time=ts,
                            regime=str(row['Regime']),
                            p_bull=float(row.get('Bull', 0.0)),
                            p_bear=float(row.get('Bear', 0.0)),
                            p_highvol=float(row.get('HighVol', 0.0)),
                            p_sideways=float(row.get('Sideways', 0.0)),
                            confidence=float(max(row.get('Bull', 0.0), row.get('Bear', 0.0), row.get('HighVol', 0.0), row.get('Sideways', 0.0))),
                            is_transition=False,
                            model_version="HMM_v2.1",
                        )
                    )

                    # NIFTY Market Bar
                    bar_objects.append(
                        MarketBar(
                            time=ts,
                            symbol="^NSEI",
                            open=float(row['NIFTY']),
                            high=float(row['NIFTY']),
                            low=float(row['NIFTY']),
                            close=float(row['NIFTY']),
                            returns=float(row['Returns']) if not pd.isna(row['Returns']) else 0.0,
                            source="NSE_HISTORICAL",
                        )
                    )

                    # Macro Indicators
                    macro_objects.append(
                        MacroIndicator(
                            time=ts,
                            cpi_yoy=float(row['CPI']) if not pd.isna(row.get('CPI')) else None,
                            cpi_mom=float(row.get('CPI_MoM')) if not pd.isna(row.get('CPI_MoM')) else None,
                            wpi_yoy=float(row['WPI']) if not pd.isna(row.get('WPI')) else None,
                            wpi_mom=float(row.get('WPI_MoM')) if not pd.isna(row.get('WPI_MoM')) else None,
                            iip_yoy=float(row['IIP']) if not pd.isna(row.get('IIP')) else None,
                            repo_rate=float(row['RepoRate']) if not pd.isna(row.get('RepoRate')) else None,
                            gsec_10y=float(row['GSecYield10']) if not pd.isna(row.get('GSecYield10')) else None,
                            yield_curve=float(row['YieldCurve']) if not pd.isna(row.get('YieldCurve')) else None,
                            pub_lag_offset_days=30,
                        )
                    )

                db.bulk_save_objects(reg_objects)
                db.bulk_save_objects(bar_objects)
                db.bulk_save_objects(macro_objects)
                print(f"  ✓ Seeded {len(reg_objects)} regime and macro sessions (2015-2026).")
            else:
                print(f"  ✓ regime_history already contains {existing_count} records. Skipping duplicate seed.")

    # 2b. Seed sector_returns (8 ETFs) + ^INDIAVIX bars: training reads these from the DB
    from database import market_store as ms
    sec_csv = os.path.join(DATA_DIR, "sector_returns_cache.csv")
    if not os.path.exists(sec_csv):
        sec_csv = os.path.join(ROOT_DIR, "sector_returns_cache.csv")
    if os.path.exists(sec_csv):
        df_sec_csv = pd.read_csv(sec_csv, index_col=0, parse_dates=True)[list(ms.SECTOR_COLUMNS)]
        n_sec = ms.upsert_sector_returns(df_sec_csv, source="CSV_SEED")
        print(f"  ✓ sector_returns: inserted {n_sec} new sessions ({len(df_sec_csv)} in cache CSV).")

    # Bootstrap / top-up ^NSEI + ^INDIAVIX bars (downloads only what is missing)
    import regime_detector_v2 as r2
    r2.sync_market_store()
    if ms.earliest_market_date(ms.VIX_SYMBOL) is None and os.path.exists(reg_csv) and 'VIX' in df_reg.columns:
        vix_series = df_reg.set_index('Date')['VIX']
        n_vix = ms.upsert_market_closes(vix_series, ms.VIX_SYMBOL, source="NSE_HISTORICAL")
        print(f"  ✓ ^INDIAVIX: network unavailable, seeded {n_vix} bars from regime_history_v2.csv.")
    n_vix_total = ms.load_market_closes()[ms.VIX_SYMBOL].notna().sum()
    print(f"  ✓ market_bars holds {n_vix_total} ^INDIAVIX bars.")

    # 3. Seed from paper_portfolio_history.csv
    paper_csv = os.path.join(DATA_DIR, "paper_portfolio_history.csv")
    if not os.path.exists(paper_csv):
        paper_csv = os.path.join(ROOT_DIR, "paper_portfolio_history.csv")
    if os.path.exists(paper_csv):
        df_paper = pd.read_csv(paper_csv)
        df_paper['Date'] = pd.to_datetime(df_paper['Date'])
        sector_names = ['BANKBEES', 'ITBEES', 'PHARMABEES', 'AUTOBEES', 'METALIETF', 'MOREALTY', 'CPSEETF', 'INFRABEES']

        with get_db() as db:
            existing_ledger = db.query(PortfolioLedger).count()
            if existing_ledger == 0:
                print(f"  Seeding portfolio_ledger and daily positions from {len(df_paper)} paper trading sessions...")
                ledger_objects = []
                pos_objects = []

                for _, row in df_paper.iterrows():
                    ts = row['Date'].to_pydatetime()
                    nav = float(row['NAV'])

                    ledger_objects.append(
                        PortfolioLedger(
                            time=ts,
                            nav=nav,
                            cash_balance=float(row['Cash']),
                            equity_value=float(row['Equity']),
                            peak_nav=nav,  # approximate initial peak
                            drawdown_pct=float(row['Drawdown']),
                            daily_pnl=float(row.get('Unrealized_Gains', 0.0)),
                            daily_return_pct=float(row.get('Daily_Return', 0.0)),
                            in_cash=bool(row.get('In_Cash', False)),
                            stop_triggered=False,
                            days_in_cash=0,
                            active_regime=str(row['Regime']),
                            cumulative_friction=float(row.get('Total_Charges', 0.0)),
                        )
                    )

                    # Extract positions per ETF
                    for sec in sector_names:
                        units_col = f"{sec}_units"
                        price_col = f"{sec}_price"
                        val_col = f"{sec}_value"
                        cost_col = f"{sec}_cost_basis"
                        pnl_col = f"{sec}_pnl"

                        units = int(row.get(units_col, 0))
                        if units > 0:
                            mkt_price = float(row.get(price_col, 0.0))
                            mkt_val = float(row.get(val_col, 0.0))
                            avg_cost = float(row.get(cost_col, mkt_price))
                            unrealized = float(row.get(pnl_col, 0.0))
                            weight = (mkt_val / nav) if nav > 0 else 0.0

                            pos_objects.append(
                                Position(
                                    time=ts,
                                    symbol=sec,
                                    units=units,
                                    average_cost=avg_cost,
                                    market_price=mkt_price,
                                    invested_value=units * avg_cost,
                                    market_value=mkt_val,
                                    unrealized_pnl=unrealized,
                                    weight_pct=weight,
                                )
                            )

                db.bulk_save_objects(ledger_objects)
                db.bulk_save_objects(pos_objects)
                print(f"  ✓ Seeded {len(ledger_objects)} portfolio ledger entries and {len(pos_objects)} position rows.")
            else:
                print(f"  ✓ portfolio_ledger already contains {existing_ledger} records.")

    # 4. Seed feature_store from cached snapshot or features if available
    candidate_snap_paths = [
        os.path.join(DATA_DIR, "snapshot.pkl"),
        os.path.join(ROOT_DIR, "snapshot.pkl"),
        os.getenv("SNAPSHOT_PATH", ""),
        "/Users/vansh/.gemini/antigravity-ide/brain/538f10e8-c790-4869-af56-8906eb5b76dc/scratch/snapshot.pkl",
    ]
    snap_path = next((p for p in candidate_snap_paths if p and os.path.exists(p)), None)
    if snap_path:
        with open(snap_path, "rb") as f:
            snap = pickle.load(f)
        df_snap = snap.get('df')
        if df_snap is not None and len(df_snap) > 100:
            import regime_detector_v2 as r2
            feat = r2.engineer_features(df_snap)
            X_features = r2.select_features_for_hmm(feat)

            with get_db() as db:
                existing_features = db.query(FeatureStore).count()
                if existing_features < len(X_features):
                    if existing_features > 0:
                        db.query(FeatureStore).delete()
                    print(f"  Seeding feature_store with {len(X_features)} pre-computed clean rows...")
                    feat_objects = []
                    for idx, row in X_features.iterrows():
                        ts = idx.to_pydatetime()
                        feat_objects.append(
                            FeatureStore(
                                time=ts,
                                ret_5d_pct=float(row['ret_5d_pct']),
                                vol_60d=float(row['vol_60d']),
                                price_vs_ma200=float(row['price_vs_ma200']),
                                drawdown=float(row['drawdown']),
                                drawdown_diff20d=float(row['drawdown_diff20d']),
                                vix_vs_ma20=float(row['vix_vs_ma20']),
                                yield_curve=float(row['yield_curve']),
                                cpi_yoy=float(row['cpi_yoy']),
                                iip_yoy=float(row['iip_yoy']),
                                wpi_mom_vs_ma3m=float(row['wpi_mom_vs_ma3m']),
                                real_rate_mom=float(row['real_rate_mom']),
                            )
                        )
                    db.bulk_save_objects(feat_objects)
                    print(f"  ✓ Seeded {len(feat_objects)} feature_store entries.")

    # 5. Run initial EOD reconciliation on latest available date
    with get_db() as db:
        latest_ledger = db.query(PortfolioLedger).order_by(PortfolioLedger.time.desc()).first()
        if latest_ledger:
            target_date = latest_ledger.time.date()
            print(f"\n  Running EOD Reconciliation for latest session ({target_date})...")
            report = DailyReconciliationEngine.reconcile(db, target_date)
            print(f"  ✓ Reconciliation status: {report.status}")
            print(f"    - Position Breaks: {report.position_breaks_count}")
            print(f"    - NAV Discrepancy: ₹{report.nav_discrepancy_inr:.4f}")
            print(f"    - Cash Discrepancy: ₹{report.cash_discrepancy_inr:.4f}")
            print(f"    - Stop Loss Integrity: {report.stop_loss_integrity}")

    print("\n" + "=" * 70)
    print("  SEEDING & INITIAL MIGRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    seed_database()
