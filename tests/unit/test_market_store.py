"""Unit tests for the DB-backed market / sector store (offline, temp SQLite)."""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import market_store as ms
from database.models import Base


@pytest.fixture
def store(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'store.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    @contextmanager
    def fake_get_db():
        db = Session()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    monkeypatch.setattr(ms, "get_db", fake_get_db)
    return ms


def _closes(start, n, base=100.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(base * np.exp(np.cumsum(np.full(n, 0.001))), index=idx)


def test_market_roundtrip_both_symbols(store):
    nifty, vix = _closes("2024-01-01", 30, 20000), _closes("2024-01-01", 30, 15)
    assert store.upsert_market_closes(nifty, store.NIFTY_SYMBOL) == 30
    assert store.upsert_market_closes(vix, store.VIX_SYMBOL) == 30

    wide = store.load_market_closes()
    assert list(wide.columns) == ["^NSEI", "^INDIAVIX"]
    assert len(wide) == 30
    np.testing.assert_allclose(wide["^NSEI"].values, nifty.values)
    np.testing.assert_allclose(wide["^INDIAVIX"].values, vix.values)


def test_market_upsert_is_idempotent_and_appends_only_delta(store):
    nifty = _closes("2024-01-01", 30, 20000)
    store.upsert_market_closes(nifty.iloc[:20], store.NIFTY_SYMBOL)
    assert store.latest_market_date() == nifty.index[19]

    # Overlapping window (as in the daily delta fetch) must only insert new sessions
    inserted = store.upsert_market_closes(nifty.iloc[15:], store.NIFTY_SYMBOL)
    assert inserted == 10
    assert store.upsert_market_closes(nifty, store.NIFTY_SYMBOL) == 0
    assert len(store.load_market_closes()) == 30
    assert store.earliest_market_date() == nifty.index[0]


def test_market_load_respects_start(store):
    nifty = _closes("2024-01-01", 30, 20000)
    store.upsert_market_closes(nifty, store.NIFTY_SYMBOL)
    out = store.load_market_closes(start=str(nifty.index[10].date()))
    assert out.index[0] == nifty.index[10]


def test_tz_aware_input_is_normalised(store):
    nifty = _closes("2024-01-01", 5, 20000)
    nifty.index = nifty.index.tz_localize("Asia/Kolkata")
    assert store.upsert_market_closes(nifty, store.NIFTY_SYMBOL) == 5
    assert store.load_market_closes().index.tz is None


def test_sector_roundtrip_and_delta(store):
    idx = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (40, 8)), index=idx, columns=list(store.SECTOR_COLUMNS))

    assert store.latest_sector_date() is None
    assert store.upsert_sector_returns(df.iloc[:30]) == 30
    assert store.upsert_sector_returns(df) == 10   # only the delta
    assert store.upsert_sector_returns(df) == 0    # idempotent

    out = store.load_sector_returns()
    assert list(out.columns) == list(store.SECTOR_COLUMNS)
    assert store.latest_sector_date() == idx[-1]
    np.testing.assert_allclose(out.values, df.values)


def test_expected_last_session_skips_weekends_and_open_market():
    from datetime import datetime
    # Sunday -> Friday
    assert ms.expected_last_session(datetime(2026, 10, 18, 14, 30)) == pd.Timestamp("2026-10-16")
    # Monday morning (before settlement) -> previous Friday
    assert ms.expected_last_session(datetime(2026, 10, 19, 10, 0)) == pd.Timestamp("2026-10-16")
    # Monday evening -> Monday
    assert ms.expected_last_session(datetime(2026, 10, 19, 17, 0)) == pd.Timestamp("2026-10-19")
