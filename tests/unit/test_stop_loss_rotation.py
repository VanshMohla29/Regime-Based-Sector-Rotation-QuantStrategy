"""
Tests for the -6% drawdown stop / cooldown / SMA re-entry logic in
compute_sector_rotation_returns() (the backtest used for every headline number).

Scenarios are tiny hand-built price paths where the correct answer can be
worked out on paper, so a regression in the risk rules fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

import regime_detector_v2 as r2
from tests.helpers import make_sector_frame

N = 120
CRASH_DAY = 30
TC_HALF = 0.0034 / 2.0                     # one-way cost the function applies
DAILY_CASH = 0.065 / 252.0                 # default cash yield when no RepoRate column
STOP_DD = -0.06


@pytest.fixture
def idx():
    return pd.bdate_range("2022-01-03", periods=N)


def _run(idx, sector_returns, nifty, *, stop_dd=STOP_DD, cooldown=3, regimes=None, mix=None):
    sec = make_sector_frame(idx, sector_returns) if not isinstance(sector_returns, pd.DataFrame) else sector_returns
    result = pd.DataFrame({'Regime': regimes if regimes is not None else 'Bull', 'NIFTY': nifty}, index=idx)
    mix = mix or {'Bull': {'BANKBEES': 1.0}}
    ret, _ = r2.compute_sector_rotation_returns(result, mix, df_sec=sec, stop_dd=stop_dd, cooldown_days=cooldown)
    return ret


def _crash_path(crash):
    rets = np.full(N, 0.0005)
    rets[CRASH_DAY] = crash
    return rets


RISING = np.exp(0.001 * np.arange(N)) * 100       # always above its own 20-SMA
FALLING = np.exp(-0.001 * np.arange(N)) * 100     # always below its own 20-SMA


def test_no_stop_when_loss_stays_inside_threshold(idx):
    rets = _crash_path(-0.03)                      # -3% day vs a -6% stop
    out = _run(idx, rets, RISING)

    np.testing.assert_allclose(out.values, rets, atol=1e-12)


def test_stop_fires_and_caps_the_loss_at_the_stop_level(idx):
    out = _run(idx, _crash_path(-0.07), RISING)

    expected = np.log((1 + STOP_DD) * (1 - TC_HALF))     # exit at stop price, less one-way cost
    assert out.iloc[CRASH_DAY] == pytest.approx(expected, abs=1e-9)
    assert out.iloc[CRASH_DAY] > -0.07, "T+0 stop must realise a smaller loss than the raw crash"


def test_capital_is_parked_in_cash_during_cooldown(idx):
    out = _run(idx, _crash_path(-0.07), RISING, cooldown=3)

    # days 31-33 earn the cash yield only
    np.testing.assert_allclose(out.iloc[CRASH_DAY + 1: CRASH_DAY + 4].values, DAILY_CASH, atol=1e-12)


def test_reentry_waits_exactly_cooldown_days_then_pays_one_way_cost(idx):
    out = _run(idx, _crash_path(-0.07), RISING, cooldown=3)

    reentry = CRASH_DAY + 4
    assert out.iloc[reentry] == pytest.approx(0.0005 + np.log(1 - TC_HALF), abs=1e-9)
    assert out.iloc[reentry + 1] == pytest.approx(0.0005, abs=1e-12)     # fully invested again, no more cost


@pytest.mark.parametrize("cooldown", [1, 3, 6, 10])
def test_longer_cooldown_delays_reentry_by_the_same_amount(idx, cooldown):
    out = _run(idx, _crash_path(-0.07), RISING, cooldown=cooldown)

    reentry = CRASH_DAY + cooldown + 1
    # first invested day after the stop is the one carrying the re-entry cost
    invested = np.isclose(out.values[CRASH_DAY + 1:], 0.0005 + np.log(1 - TC_HALF), atol=1e-9)
    assert CRASH_DAY + 1 + int(np.argmax(invested)) == reentry


def test_no_reentry_while_nifty_is_below_its_moving_average(idx):
    out = _run(idx, _crash_path(-0.07), FALLING, cooldown=3)

    np.testing.assert_allclose(out.iloc[CRASH_DAY + 1:].values, DAILY_CASH, atol=1e-12)


def test_stop_disabled_with_deep_threshold(idx):
    rets = _crash_path(-0.07)
    out = _run(idx, rets, RISING, stop_dd=-0.99)

    np.testing.assert_allclose(out.values, rets, atol=1e-12)


def test_stop_level_trails_the_running_peak(idx):
    """Gains first, then a pull-back: the stop is measured from the new peak, not from the start."""
    rets = np.zeros(N)
    rets[:20] = 0.01                                # run up ~+22%
    rets[20] = -0.065                               # -6.5% from the peak -> must trigger
    out = _run(idx, rets, RISING)

    assert out.iloc[20] == pytest.approx(np.log((1 + STOP_DD) * (1 - TC_HALF)), abs=1e-9)


def test_signal_is_applied_with_one_day_lag_no_lookahead(idx):
    """Regime decided at close of t-1 sets the weights earned on day t (T+1 execution)."""
    switch = 60
    regimes = np.where(np.arange(N) < switch, 'Bull', 'Bear')
    sec = pd.DataFrame(0.0, index=idx, columns=make_sector_frame(idx, np.zeros(N)).columns)
    sec['BANKBEES'] = 0.001
    sec['ITBEES'] = -0.001
    mix = {'Bull': {'BANKBEES': 1.0}, 'Bear': {'ITBEES': 1.0}}

    out = _run(idx, sec, RISING, regimes=regimes, mix=mix, stop_dd=-0.99)

    assert out.iloc[switch] == pytest.approx(0.001)        # first 'Bear' row still earns Bull weights
    assert out.iloc[switch + 1] == pytest.approx(-0.001)   # Bear weights from the next day
