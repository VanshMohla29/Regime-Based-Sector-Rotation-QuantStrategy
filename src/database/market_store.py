"""
Market & Sector Time-Series Store.

Thin persistence layer (SQLite locally / PostgreSQL + TimescaleDB in Docker) for:
  - NIFTY 50 (^NSEI) and India VIX (^INDIAVIX) daily bars  -> ``market_bars``
  - Sanitized sector-ETF daily log returns (8 ETFs)         -> ``sector_returns``

Training reads the full history from here in milliseconds. The network is only
touched for the *delta* of trading sessions that are newer than the latest
stored row (see ``regime_detector_v2.sync_market_store`` / ``sync_sector_store``).
This module is intentionally free of any ``regime_detector_v2`` import so it can
be used from both the pipeline and the ingestion service without cycles.
"""

import os
import sys
from datetime import datetime, time as dtime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from database.connection import get_db  # noqa: E402
from database.models import MarketBar, SectorReturn  # noqa: E402

NIFTY_SYMBOL = "^NSEI"
VIX_SYMBOL = "^INDIAVIX"
MARKET_SYMBOLS = (NIFTY_SYMBOL, VIX_SYMBOL)

# Public (DataFrame) column name -> ORM attribute
SECTOR_COLUMNS = {
    "BANKBEES": "bankbees",
    "ITBEES": "itbees",
    "PHARMABEES": "pharmabees",
    "AUTOBEES": "autobees",
    "METALIETF": "metalietf",
    "MOREALTY": "morealty",
    "CPSEETF": "cpseetf",
    "INFRABEES": "infrabees",
}

# NSE cash market closes 15:30 IST; allow a buffer before treating the day as final.
_MARKET_FINAL_TIME = dtime(16, 0)
_IST_OFFSET = timedelta(hours=5, minutes=30)


# ── Helpers ──────────────────────────────────────────────────────────

def to_naive_index(idx) -> pd.DatetimeIndex:
    """Normalize any datetime index to tz-naive midnight timestamps."""
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _naive_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def expected_last_session(now: Optional[datetime] = None) -> pd.Timestamp:
    """
    Most recent weekday whose NSE session should be fully settled (IST).
    Exchange holidays are not modelled: on a holiday the delta fetch simply
    returns no new rows, which is cheap and harmless.
    """
    if now is None:
        now = datetime.utcnow() + _IST_OFFSET
    day = pd.Timestamp(now.date())
    if now.time() < _MARKET_FINAL_TIME:
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


# ── Market bars (^NSEI, ^INDIAVIX) ───────────────────────────────────

def latest_market_date(symbol: str = NIFTY_SYMBOL) -> Optional[pd.Timestamp]:
    with get_db() as db:
        row = (
            db.query(MarketBar.time)
            .filter(MarketBar.symbol == symbol)
            .order_by(MarketBar.time.desc())
            .first()
        )
    return _naive_ts(row[0]) if row else None


def earliest_market_date(symbol: str = NIFTY_SYMBOL) -> Optional[pd.Timestamp]:
    with get_db() as db:
        row = (
            db.query(MarketBar.time)
            .filter(MarketBar.symbol == symbol)
            .order_by(MarketBar.time.asc())
            .first()
        )
    return _naive_ts(row[0]) if row else None


def load_market_closes(start: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a tz-naive DataFrame indexed by date with columns
    ['^NSEI', '^INDIAVIX'] (closing levels) from the database.
    """
    with get_db() as db:
        q = db.query(MarketBar.time, MarketBar.symbol, MarketBar.close).filter(
            MarketBar.symbol.in_(MARKET_SYMBOLS)
        )
        if start is not None:
            q = q.filter(MarketBar.time >= pd.Timestamp(start).to_pydatetime())
        rows = q.order_by(MarketBar.time.asc()).all()
    if not rows:
        return pd.DataFrame(columns=list(MARKET_SYMBOLS))
    df = pd.DataFrame(rows, columns=["time", "symbol", "close"])
    df["time"] = to_naive_index(df["time"])
    wide = df.pivot(index="time", columns="symbol", values="close").sort_index()
    for sym in MARKET_SYMBOLS:
        if sym not in wide.columns:
            wide[sym] = float("nan")
    return wide[list(MARKET_SYMBOLS)]


def upsert_market_closes(closes: pd.Series, symbol: str, source: str = "YAHOO") -> int:
    """
    Inserts closing levels for ``symbol`` that are not already stored.
    Existing rows are never overwritten (history is immutable once settled).
    Returns the number of rows inserted.
    """
    closes = closes.dropna()
    if closes.empty:
        return 0
    closes = closes.copy()
    closes.index = to_naive_index(closes.index)
    closes = closes[~closes.index.duplicated(keep="last")]

    with get_db() as db:
        lo, hi = closes.index.min().to_pydatetime(), closes.index.max().to_pydatetime()
        existing = {
            _naive_ts(r[0])
            for r in db.query(MarketBar.time)
            .filter(MarketBar.symbol == symbol, MarketBar.time >= lo, MarketBar.time <= hi)
            .all()
        }
        prev_close = None
        objs = []
        for ts, px in closes.items():
            px = float(px)
            ret = None
            if prev_close is not None and prev_close > 0 and px > 0:
                ret = float(np.log(px / prev_close))
            prev_close = px
            if ts in existing:
                continue
            objs.append(
                MarketBar(
                    time=ts.to_pydatetime(),
                    symbol=symbol,
                    open=px,
                    high=px,
                    low=px,
                    close=px,
                    volume=0,
                    returns=ret,
                    source=source,
                )
            )
        if objs:
            db.bulk_save_objects(objs)
    return len(objs)


# ── Sector returns ───────────────────────────────────────────────────

def latest_sector_date() -> Optional[pd.Timestamp]:
    with get_db() as db:
        row = db.query(SectorReturn.time).order_by(SectorReturn.time.desc()).first()
    return _naive_ts(row[0]) if row else None


def load_sector_returns(start: Optional[str] = None) -> pd.DataFrame:
    """Returns sanitized sector log returns (columns = ETF names) from the database."""
    attrs = [getattr(SectorReturn, a) for a in SECTOR_COLUMNS.values()]
    with get_db() as db:
        q = db.query(SectorReturn.time, *attrs)
        if start is not None:
            q = q.filter(SectorReturn.time >= pd.Timestamp(start).to_pydatetime())
        rows = q.order_by(SectorReturn.time.asc()).all()
    if not rows:
        return pd.DataFrame(columns=list(SECTOR_COLUMNS))
    df = pd.DataFrame(rows, columns=["time"] + list(SECTOR_COLUMNS))
    df["time"] = to_naive_index(df["time"])
    return df.set_index("time").sort_index()


def upsert_sector_returns(df: pd.DataFrame, source: str = "YAHOO") -> int:
    """Inserts sector-return rows whose date is not yet stored. Returns rows inserted."""
    df = df.dropna()
    if df.empty:
        return 0
    df = df.copy()
    df.index = to_naive_index(df.index)
    df = df[~df.index.duplicated(keep="last")]

    with get_db() as db:
        lo, hi = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
        existing = {
            _naive_ts(r[0])
            for r in db.query(SectorReturn.time)
            .filter(SectorReturn.time >= lo, SectorReturn.time <= hi)
            .all()
        }
        objs = []
        for ts, row in df.iterrows():
            if ts in existing:
                continue
            kwargs = {attr: float(row[col]) for col, attr in SECTOR_COLUMNS.items()}
            objs.append(SectorReturn(time=ts.to_pydatetime(), source=source, **kwargs))
        if objs:
            db.bulk_save_objects(objs)
    return len(objs)
