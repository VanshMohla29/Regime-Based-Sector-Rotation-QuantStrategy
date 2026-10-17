"""Unit tests for compute_performance_metrics() — the numbers quoted in the README."""

import numpy as np
import pytest

import regime_detector_v2 as r2

ANN = 252


def test_constant_gain_has_no_drawdown_and_perfect_win_rate():
    returns = np.full(250, 0.001)
    capital = 1_000_000 * np.exp(np.cumsum(returns))

    m = r2.compute_performance_metrics(returns, capital)

    assert m['ann_return'] == pytest.approx(np.exp(0.001 * ANN) - 1)
    assert m['max_dd'] == 0.0
    assert m['win_rate'] == 1.0
    assert m['ann_vol'] == pytest.approx(0.0, abs=1e-12)
    # no losing days -> profit factor and sortino are undefined
    assert np.isnan(m['profit_factor'])
    assert np.isnan(m['sortino'])


def test_max_drawdown_matches_hand_computation():
    # peak 110 -> trough 99  =>  (99-110)/110 = -10%
    capital = np.array([100.0, 110.0, 99.0, 120.0])
    returns = np.diff(np.log(capital), prepend=np.log(100.0))

    m = r2.compute_performance_metrics(returns, capital)

    assert m['max_dd'] == pytest.approx(-0.10)


def test_max_drawdown_is_never_positive():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 500)
    capital = 1e6 * np.exp(np.cumsum(returns))

    assert r2.compute_performance_metrics(returns, capital)['max_dd'] <= 0.0


def test_profit_factor_is_gross_gains_over_gross_losses():
    returns = np.array([0.02, -0.01, 0.03, -0.01, 0.01])  # gains .06, losses .02
    capital = 1e6 * np.exp(np.cumsum(returns))

    m = r2.compute_performance_metrics(returns, capital)

    assert m['profit_factor'] == pytest.approx(3.0, rel=1e-4)
    assert m['win_rate'] == pytest.approx(3 / 5)


def test_sharpe_is_positive_when_returns_beat_risk_free_and_negative_otherwise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 500)
    up = 0.002 + noise      # ~50% annualised, far above 6% risk-free
    down = -0.002 + noise
    cap = lambda r: 1e6 * np.exp(np.cumsum(r))  # noqa: E731

    assert r2.compute_performance_metrics(up, cap(up))['sharpe'] > 0
    assert r2.compute_performance_metrics(down, cap(down))['sharpe'] < 0


def test_cvar_is_no_better_than_var():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0003, 0.012, 1000)
    capital = 1e6 * np.exp(np.cumsum(returns))

    m = r2.compute_performance_metrics(returns, capital)

    assert m['cvar_95'] <= m['var_95'] < 0


def test_risk_free_rate_lowers_sharpe():
    rng = np.random.default_rng(3)
    returns = 0.001 + rng.normal(0, 0.01, 500)
    capital = 1e6 * np.exp(np.cumsum(returns))

    low = r2.compute_performance_metrics(returns, capital, risk_free=0.0)['sharpe']
    high = r2.compute_performance_metrics(returns, capital, risk_free=0.10)['sharpe']

    assert high < low
