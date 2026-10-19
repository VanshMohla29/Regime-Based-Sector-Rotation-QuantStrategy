"""
Incremental Daily Market & Macro Ingestion Service.
Fetches only delta updates (today's latest data), updates market bars (^NSEI, ^INDIAVIX),
sector-ETF returns, macro indicators, and writes directly to the TimescaleDB Feature Store.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database.connection import get_db
from database.models import MarketBar, MacroIndicator, FeatureStore
import regime_detector_v2 as r2


def get_latest_stored_date(symbol: str = "^NSEI") -> datetime:
    """Returns the most recent timestamp available in market_bars."""
    with get_db() as db:
        latest = (
            db.query(MarketBar.time)
            .filter(MarketBar.symbol == symbol)
            .order_by(MarketBar.time.desc())
            .first()
        )
        if latest:
            return latest[0]
        return datetime(2015, 1, 1)


def get_latest_feature_date() -> datetime:
    """Returns the most recent timestamp available in feature_store (falls back to market_bars)."""
    with get_db() as db:
        latest = db.query(FeatureStore.time).order_by(FeatureStore.time.desc()).first()
        if latest:
            return latest[0]
    return get_latest_stored_date("^NSEI")


def ingest_daily_update(force_reload: bool = False, is_eod: bool = False):
    """
    Checks the latest date in the database and ingests new trading days.
    If already up-to-date and not an EOD settlement run, completes in milliseconds.
    If is_eod=True, ensures final market close settlement data is ingested and updated.
    """
    # Append only the missing sessions of NIFTY / India VIX bars and the 8 sector-ETF
    # return series. No-op (no network) when the stored history is already current.
    market_new = r2.sync_market_store(force_full=force_reload, allow_current_day=True)
    sector_new = r2.sync_sector_store(force_full=force_reload, allow_current_day=True)

    latest_dt = get_latest_feature_date()
    if latest_dt.tzinfo is not None:
        latest_dt = latest_dt.replace(tzinfo=None)
    today = datetime.now()
    
    # If latest stored date is within the last 20 hours and not an EOD settlement run, return early
    if not force_reload and not is_eod and (today - latest_dt) < timedelta(hours=20):
        print(f"✓ Database is already up to date (Latest record: {latest_dt.strftime('%Y-%m-%d')}).")
        return {"status": "UP_TO_DATE", "latest_date": latest_dt.isoformat(),
                "market_bars_ingested": market_new, "sector_rows_ingested": sector_new}

    start_date = (latest_dt - timedelta(days=400)).strftime("%Y-%m-%d")
    print(f"Ingesting latest market & macro delta (fetch window: {start_date} to today)...")

    # Fetch recent market data
    df = r2.fetch_live_market_data(start_date=start_date, allow_current_day=True)
    if df is None or len(df) == 0:
        return {"status": "NO_NEW_DATA"}

    # Engineer features
    feat = r2.engineer_features(df)
    X_features = r2.select_features_for_hmm(feat)

    new_rows_count = 0
    with get_db() as db:
        for idx, row in X_features.iterrows():
            ts = idx.to_pydatetime()
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            if ts < latest_dt and not force_reload:
                continue
            if ts == latest_dt and not (force_reload or is_eod):
                continue

            # 1. Upsert into FeatureStore
            existing_f = db.query(FeatureStore).filter(FeatureStore.time == ts).first()
            if not existing_f:
                db.add(
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
            else:
                existing_f.ret_5d_pct = float(row['ret_5d_pct'])
                existing_f.vol_60d = float(row['vol_60d'])
                existing_f.price_vs_ma200 = float(row['price_vs_ma200'])
                existing_f.drawdown = float(row['drawdown'])
                existing_f.drawdown_diff20d = float(row['drawdown_diff20d'])
                existing_f.vix_vs_ma20 = float(row['vix_vs_ma20'])
                existing_f.yield_curve = float(row['yield_curve'])
                existing_f.cpi_yoy = float(row['cpi_yoy'])
                existing_f.iip_yoy = float(row['iip_yoy'])
                existing_f.wpi_mom_vs_ma3m = float(row['wpi_mom_vs_ma3m'])
                existing_f.real_rate_mom = float(row['real_rate_mom'])

            # 2. Upsert into MarketBar
            if idx in df.index:
                close_px = float(df.loc[idx, 'NIFTY'])
                ret_val = float(df.loc[idx, 'Returns']) if not pd.isna(df.loc[idx, 'Returns']) else 0.0
                existing_b = db.query(MarketBar).filter(MarketBar.time == ts, MarketBar.symbol == "^NSEI").first()
                if not existing_b:
                    db.add(
                        MarketBar(
                            time=ts,
                            symbol="^NSEI",
                            open=close_px,
                            high=close_px,
                            low=close_px,
                            close=close_px,
                            returns=ret_val,
                            source="YAHOO_LIVE",
                        )
                    )
                else:
                    existing_b.close = close_px
                    existing_b.high = max(existing_b.high or close_px, close_px)
                    existing_b.low = min(existing_b.low or close_px, close_px)
                    existing_b.returns = ret_val
                    existing_b.source = "YAHOO_LIVE"

            # 3. Upsert into MacroIndicator
            if idx in df.index:
                existing_m = db.query(MacroIndicator).filter(MacroIndicator.time == ts).first()
                if not existing_m:
                    db.add(
                        MacroIndicator(
                            time=ts,
                            cpi_yoy=float(df.loc[idx, 'CPI']) if not pd.isna(df.loc[idx, 'CPI']) else None,
                            cpi_mom=float(df.loc[idx, 'CPI_MoM']) if 'CPI_MoM' in df.columns and not pd.isna(df.loc[idx, 'CPI_MoM']) else None,
                            wpi_yoy=float(df.loc[idx, 'WPI']) if not pd.isna(df.loc[idx, 'WPI']) else None,
                            wpi_mom=float(df.loc[idx, 'WPI_MoM']) if 'WPI_MoM' in df.columns and not pd.isna(df.loc[idx, 'WPI_MoM']) else None,
                            iip_yoy=float(df.loc[idx, 'IIP']) if not pd.isna(df.loc[idx, 'IIP']) else None,
                            repo_rate=float(df.loc[idx, 'RepoRate']) if not pd.isna(df.loc[idx, 'RepoRate']) else None,
                            gsec_10y=float(df.loc[idx, 'GSecYield10']) if not pd.isna(df.loc[idx, 'GSecYield10']) else None,
                            gsec_2y=float(df.loc[idx, 'GSecYield2']) if not pd.isna(df.loc[idx, 'GSecYield2']) else None,
                            yield_curve=float(df.loc[idx, 'YieldCurve']) if not pd.isna(df.loc[idx, 'YieldCurve']) else None,
                        )
                    )
                else:
                    if not pd.isna(df.loc[idx, 'CPI']): existing_m.cpi_yoy = float(df.loc[idx, 'CPI'])
                    if 'CPI_MoM' in df.columns and not pd.isna(df.loc[idx, 'CPI_MoM']): existing_m.cpi_mom = float(df.loc[idx, 'CPI_MoM'])
                    if not pd.isna(df.loc[idx, 'WPI']): existing_m.wpi_yoy = float(df.loc[idx, 'WPI'])
                    if 'WPI_MoM' in df.columns and not pd.isna(df.loc[idx, 'WPI_MoM']): existing_m.wpi_mom = float(df.loc[idx, 'WPI_MoM'])
                    if not pd.isna(df.loc[idx, 'IIP']): existing_m.iip_yoy = float(df.loc[idx, 'IIP'])
                    if not pd.isna(df.loc[idx, 'RepoRate']): existing_m.repo_rate = float(df.loc[idx, 'RepoRate'])
                    if not pd.isna(df.loc[idx, 'GSecYield10']): existing_m.gsec_10y = float(df.loc[idx, 'GSecYield10'])
                    if not pd.isna(df.loc[idx, 'GSecYield2']): existing_m.gsec_2y = float(df.loc[idx, 'GSecYield2'])
                    if not pd.isna(df.loc[idx, 'YieldCurve']): existing_m.yield_curve = float(df.loc[idx, 'YieldCurve'])

            new_rows_count += 1

    print(f"✓ Ingestion complete. Ingested {new_rows_count} new trading days into TimescaleDB.")
    return {"status": "SUCCESS", "new_rows": new_rows_count,
            "market_bars_ingested": market_new, "sector_rows_ingested": sector_new}


def ingest_macro_release(
    effective_date: str,
    cpi_yoy: Optional[float] = None,
    cpi_mom: Optional[float] = None,
    iip_yoy: Optional[float] = None,
    wpi_yoy: Optional[float] = None,
    wpi_mom: Optional[float] = None,
    repo_rate: Optional[float] = None,
    gsec_10y: Optional[float] = None,
    gsec_2y: Optional[float] = None,
    forward_fill: bool = True,
):
    """
    Ingests official monthly macroeconomic releases (MoSPI, RBI, DPIIT) into TimescaleDB.
    Updates the MacroIndicator hypertable from effective_date forward.
    """
    dt = pd.to_datetime(effective_date).to_pydatetime()
    yield_curve = (gsec_10y - gsec_2y) if (gsec_10y is not None and gsec_2y is not None) else None

    with get_db() as db:
        record = db.query(MacroIndicator).filter(MacroIndicator.time == dt).first()
        if not record:
            record = MacroIndicator(time=dt)
            db.add(record)

        if cpi_yoy is not None: record.cpi_yoy = cpi_yoy
        if cpi_mom is not None: record.cpi_mom = cpi_mom
        if iip_yoy is not None: record.iip_yoy = iip_yoy
        if wpi_yoy is not None: record.wpi_yoy = wpi_yoy
        if wpi_mom is not None: record.wpi_mom = wpi_mom
        if repo_rate is not None: record.repo_rate = repo_rate
        if gsec_10y is not None: record.gsec_10y = gsec_10y
        if gsec_2y is not None: record.gsec_2y = gsec_2y
        if yield_curve is not None: record.yield_curve = yield_curve

        # Forward-fill subsequent records if requested
        if forward_fill:
            subsequent = (
                db.query(MacroIndicator)
                .filter(MacroIndicator.time > dt)
                .order_by(MacroIndicator.time.asc())
                .all()
            )
            for sub in subsequent:
                if cpi_yoy is not None: sub.cpi_yoy = cpi_yoy
                if cpi_mom is not None: sub.cpi_mom = cpi_mom
                if iip_yoy is not None: sub.iip_yoy = iip_yoy
                if wpi_yoy is not None: sub.wpi_yoy = wpi_yoy
                if wpi_mom is not None: sub.wpi_mom = wpi_mom
                if repo_rate is not None: sub.repo_rate = repo_rate
                if yield_curve is not None and sub.yield_curve is None: sub.yield_curve = yield_curve

    print(f"✓ Ingested macro release for {effective_date} into TimescaleDB.")
    return {"status": "SUCCESS", "effective_date": effective_date}


def auto_sync_macro_from_sources():
    """
    Automatically polls official macroeconomic data endpoints (FRED official mirrors
    for RBI Repo Rate, Sovereign Yields, IIP, and CPI) to ingest any newly published releases.
    Runs asynchronously during daily ETL so live trading/inference remains 100% instantaneous.
    """
    series_ids = {
        'cpi': 'INDCPIALLMINMEI',
        'iip': 'INDPRMNTO01GYSAM',
        'repo': 'IRSTCI01INM156N',
        'gsec10': 'INDIRLTLT01STM',
        'gsec2': 'INDIR3TIB01STM',
    }
    dfs = {}
    for name, sid in series_ids.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            d = pd.read_csv(url, index_col=0, parse_dates=True)
            d[name] = pd.to_numeric(d.iloc[:, 0], errors='coerce')
            dfs[name] = d[name].dropna()
        except Exception:
            continue

    if not dfs:
        return {"status": "SKIPPED", "reason": "Endpoints unreachable"}

    # Compute CPI YoY % change
    if 'cpi' in dfs and len(dfs['cpi']) > 12:
        cpi_yoy = (dfs['cpi'].pct_change(12) * 100).dropna()
        cpi_mom = (dfs['cpi'].pct_change(1) * 100).dropna()
    else:
        cpi_yoy = pd.Series(dtype=float)
        cpi_mom = pd.Series(dtype=float)

    updates_count = 0
    with get_db() as db:
        latest_macro = db.query(MacroIndicator).order_by(MacroIndicator.time.desc()).first()
        latest_macro_time = datetime(2015, 1, 1)
        if latest_macro and latest_macro.time:
            latest_macro_time = latest_macro.time.replace(tzinfo=None) if latest_macro.time.tzinfo else latest_macro.time

        all_dates = sorted(set().union(*(s.index for s in dfs.values())))
        for dt in all_dates:
            ts = dt.to_pydatetime()
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            record = db.query(MacroIndicator).filter(MacroIndicator.time == ts).first()
            if not record and ts > latest_macro_time:
                record = MacroIndicator(time=ts)
                db.add(record)
                updates_count += 1

            if record:
                if ts in cpi_yoy.index: record.cpi_yoy = float(cpi_yoy.loc[ts])
                if ts in cpi_mom.index: record.cpi_mom = float(cpi_mom.loc[ts])
                if 'iip' in dfs and ts in dfs['iip'].index: record.iip_yoy = float(dfs['iip'].loc[ts])
                if 'repo' in dfs and ts in dfs['repo'].index: record.repo_rate = float(dfs['repo'].loc[ts])
                if 'gsec10' in dfs and ts in dfs['gsec10'].index: record.gsec_10y = float(dfs['gsec10'].loc[ts])
                if 'gsec2' in dfs and ts in dfs['gsec2'].index: record.gsec_2y = float(dfs['gsec2'].loc[ts])
                if record.gsec_10y is not None and record.gsec_2y is not None:
                    record.yield_curve = record.gsec_10y - record.gsec_2y

    print(f"✓ Macro auto-sync complete. Ingested/updated {updates_count} macro release dates.")
    return {"status": "SUCCESS", "updated_dates": updates_count}


if __name__ == "__main__":
    auto_sync_macro_from_sources()
    ingest_daily_update()
