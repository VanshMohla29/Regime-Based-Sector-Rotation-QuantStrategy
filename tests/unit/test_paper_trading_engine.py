"""
Scenario tests for run_paper_trading(): T+0 vs T+1 execution, the -6% stop,
cooldown + same-day re-entry, and drawdown measured from the all-time peak NAV.

These lock in behaviour that was fixed by hand earlier (e.g. drawdown showing
0% after re-entry) so it cannot regress unnoticed.
"""

import numpy as np
import pandas as pd
import pytest

import paper_portfolio as pp

N = 80
CRASH = 15            # row where price drops 8%
INITIAL = 1_000_000


@pytest.fixture
def idx():
    return pd.bdate_range("2024-01-01", periods=N)


def _run(idx, *, mode="T+0", prices=None, regimes=None, nifty=None):
    if prices is None:
        prices = np.full(N, 100.0)
        prices[CRASH:] = 92.0
    if nifty is None:
        nifty = 20000 * np.exp(0.001 * np.arange(N))          # always above its 20-SMA
    df = pd.DataFrame({'NIFTY': nifty, 'VIX': 15.0, 'RepoRate': 6.5}, index=idx)
    feat = pd.DataFrame(index=idx)
    regimes = list(regimes) if regimes is not None else ['Bull'] * N
    post = pd.DataFrame({'Bull': 1.0, 'Bear': 0.0, 'HighVol': 0.0, 'Sideways': 0.0}, index=idx)
    etf = pd.DataFrame({'BANKBEES': prices}, index=idx)
    freeze = idx[0] - pd.Timedelta(days=1)
    return pp.run_paper_trading(df, feat, regimes, post, etf, {'Bull': {'BANKBEES': 1.0},
                                                               'Bear': {'BANKBEES': 1.0}},
                                freeze, initial_capital=INITIAL, execution_mode=mode)


# ── accounting invariants ────────────────────────────────────────────────

@pytest.fixture
def hist(idx):
    return _run(idx)


def test_nav_equals_cash_plus_equity_every_day(hist):
    np.testing.assert_allclose(hist['NAV'], hist['Cash'] + hist['Equity'])


def test_cash_never_negative(hist):
    assert (hist['Cash'] >= -1e-6).all()


def test_first_day_nav_is_initial_capital_minus_charges(hist):
    assert hist['NAV'].iloc[0] == pytest.approx(INITIAL - hist['Total_Charges'].iloc[0])


def test_cumulative_charges_are_non_decreasing_and_match_daily(hist):
    assert hist['Total_Charges'].is_monotonic_increasing
    assert hist['Total_Charges'].iloc[-1] == pytest.approx(hist['Day_Charges'].sum())


# ── drawdown = distance from the all-time peak NAV ───────────────────────

def test_drawdown_is_measured_from_all_time_peak_nav(hist):
    peak = np.maximum(INITIAL, hist['NAV'].cummax())

    np.testing.assert_allclose(hist['Peak_NAV'], peak)
    np.testing.assert_allclose(hist['Drawdown'], hist['NAV'] / peak - 1)
    assert (hist['Drawdown'] <= 1e-12).all()


def test_peak_nav_never_decreases(hist):
    assert hist['Peak_NAV'].is_monotonic_increasing


def test_drawdown_is_not_reset_to_zero_after_reentry(idx):
    """Regression: after the stop and re-entry the portfolio is still ~8% below its peak."""
    h = _run(idx)
    stop_row = int(np.argmax(h['In_Cash'].values))
    back_in = stop_row + int(np.argmax(~h['In_Cash'].values[stop_row:]))

    assert h['Drawdown'].iloc[back_in] < -0.05
    assert h['Drawdown'].iloc[back_in + 1] < -0.05


# ── T+0 stop & re-entry ──────────────────────────────────────────────────

def test_t0_stop_sells_on_the_crash_day_itself(idx):
    h = _run(idx, mode="T+0")

    assert h['In_Cash'].iloc[CRASH]
    assert h['Equity'].iloc[CRASH] == 0.0
    assert h['Rebalanced'].iloc[CRASH]
    assert h['Regime'].iloc[CRASH].startswith("CASH (STOP")
    assert not h['In_Cash'].iloc[CRASH - 1]


def test_t1_stop_is_deferred_one_trading_day(idx):
    h = _run(idx, mode="T+1")

    assert not h['In_Cash'].iloc[CRASH]
    assert h['In_Cash'].iloc[CRASH + 1]


def test_t0_reentry_happens_same_day_cooldown_is_met(idx):
    h = _run(idx, mode="T+0")
    stop_row = int(np.argmax(h['In_Cash'].values))

    # days_in_cash starts at 0 on the stop day -> cooldown of 3 is met 3 sessions later
    assert h['In_Cash'].iloc[stop_row + 1]
    assert h['In_Cash'].iloc[stop_row + 2]
    back = stop_row + pp.STOP_COOLDOWN_DAYS
    assert not h['In_Cash'].iloc[back]
    assert h['Equity'].iloc[back] > 0, "T+0 re-entry buys on the signal date, not the next day"
    assert h['Rebalanced'].iloc[back]


def test_no_reentry_while_nifty_below_moving_average(idx):
    falling = 20000 * np.exp(-0.002 * np.arange(N))
    h = _run(idx, nifty=falling)

    stop_row = int(np.argmax(h['In_Cash'].values))
    assert h['In_Cash'].iloc[stop_row:].all()


def test_cash_earns_repo_yield_while_parked(idx):
    h = _run(idx)
    stop_row = int(np.argmax(h['In_Cash'].values))
    c0, c1 = h['Cash'].iloc[stop_row], h['Cash'].iloc[stop_row + 1]

    assert c1 == pytest.approx(c0 * (1 + 6.5 / 100 / 252))


def test_stop_does_not_trigger_in_a_calm_market(idx):
    h = _run(idx, prices=np.full(N, 100.0))

    assert not h['In_Cash'].any()


# ── regime bookkeeping (feeds the Fig-9 background shading) ──────────────

def test_market_regime_column_tracks_model_regime_even_while_in_cash(idx):
    """Panel-1 background must follow the *model* regime, never the 'CASH (STOP)' label."""
    regimes = ['Bull'] * 20 + ['Bear'] * 30 + ['Bull'] * (N - 50)
    h = _run(idx, regimes=regimes)

    assert h['In_Cash'].any()
    assert list(h['Market_Regime']) == regimes
    assert set(h.loc[h['In_Cash'], 'Regime']) <= {h['Regime'][h['In_Cash']].iloc[0]}


def test_every_row_has_a_regime_label(hist):
    assert hist['Regime'].notna().all()
    assert hist['Market_Regime'].isin(['Bull', 'Bear', 'HighVol', 'Sideways']).all()


# ── outputs ──────────────────────────────────────────────────────────────

def test_first_daily_return_reflects_day_one_cost(hist):
    """Day-1 return is NAV0/initial - 1 (negative: you paid to enter)."""
    assert hist['Daily_Return'].iloc[0] == pytest.approx(hist['NAV'].iloc[0] / INITIAL - 1)
    assert hist['Daily_Return'].iloc[0] < 0


def test_cumulative_return_matches_nav(hist):
    np.testing.assert_allclose(hist['Cumulative_Return'], (hist['NAV'] / INITIAL - 1) * 100)


def test_unknown_execution_mode_falls_back_to_t0(idx):
    assert (_run(idx, mode="banana")['Execution_Mode'] == 'T+0').all()


def test_no_overlap_between_market_and_etf_dates_returns_empty_frame(idx):
    df = pd.DataFrame({'NIFTY': 1.0, 'VIX': 1.0, 'RepoRate': 6.5}, index=idx)
    far = pd.bdate_range("2030-01-01", periods=N)
    etf = pd.DataFrame({'BANKBEES': 100.0}, index=far)
    post = pd.DataFrame({'Bull': 1.0, 'Bear': 0.0, 'HighVol': 0.0, 'Sideways': 0.0}, index=idx)

    out = pp.run_paper_trading(df, pd.DataFrame(index=idx), ['Bull'] * N, post, etf,
                               {'Bull': {'BANKBEES': 1.0}}, idx[0] - pd.Timedelta(days=1))

    assert out.empty
