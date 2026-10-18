from datetime import datetime
from zoneinfo import ZoneInfo
from scripts.eod_scheduler import get_next_run_time
from scripts.eod_pipeline import sync_latest_portfolio_to_db, run_eod_pipeline


def test_eod_scheduler_next_run_time():
    next_run = get_next_run_time("18:00", "Asia/Kolkata")
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

    assert next_run > now_ist
    assert next_run.hour == 18
    assert next_run.minute == 0
    # Next run must be Monday (0) through Friday (4), never weekend
    assert next_run.weekday() in (0, 1, 2, 3, 4)


def test_sync_latest_portfolio_to_db():
    ts = sync_latest_portfolio_to_db()
    # If paper_portfolio_history.csv exists, ts should be a valid datetime
    if ts is not None:
        assert isinstance(ts, datetime)


def test_run_eod_pipeline():
    res = run_eod_pipeline()
    assert res['status'] == 'SUCCESS'
    assert 'elapsed_seconds' in res
    assert 'audit' in res
