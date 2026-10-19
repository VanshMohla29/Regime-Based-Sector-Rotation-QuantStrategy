"""
Tests for the causal anti-whipsaw filter.

Two copies of smooth_regimes_causal() exist (regime_detector_v2 and
paper_portfolio — the latter says it "mirrors exactly").  A parity test makes
sure they can never silently drift apart.
"""

import numpy as np
import pytest

import paper_portfolio as pp
import regime_detector_v2 as r2
from tests.helpers import REGIMES, runs

IMPLEMENTATIONS = [
    pytest.param(r2.smooth_regimes_causal, id="regime_detector_v2"),
    pytest.param(pp.smooth_regimes_causal, id="paper_portfolio"),
]


def _random_regimes(n, seed, p_switch=0.35):
    rng = np.random.default_rng(seed)
    seq = [REGIMES[rng.integers(4)]]
    for _ in range(n - 1):
        seq.append(REGIMES[rng.integers(4)] if rng.random() < p_switch else seq[-1])
    return seq


@pytest.mark.parametrize("smooth", IMPLEMENTATIONS)
def test_empty_input(smooth):
    assert smooth([]) == []


@pytest.mark.parametrize("smooth", IMPLEMENTATIONS)
def test_single_element(smooth):
    assert smooth(['Bull']) == ['Bull']


@pytest.mark.parametrize("smooth", IMPLEMENTATIONS)
def test_length_is_preserved(smooth):
    seq = _random_regimes(300, seed=1)
    assert len(smooth(seq, min_hold=7)) == len(seq)


@pytest.mark.parametrize("smooth", IMPLEMENTATIONS)
def test_min_hold_of_one_is_identity(smooth):
    seq = _random_regimes(200, seed=2)
    assert smooth(seq, min_hold=1) == seq


@pytest.mark.parametrize("smooth", IMPLEMENTATIONS)
def test_switch_is_blocked_until_min_hold_days_elapsed(smooth):
    seq = ['Bull'] * 3 + ['Bear'] * 12
    out = smooth(seq, min_hold=7)

    # Bull held for the full 7-day lockout (idx 0-6), Bear admitted from idx 7
    assert out[:7] == ['Bull'] * 7
    assert out[7:] == ['Bear'] * 8


@pytest.mark.parametrize("smooth", IMPLEMENTATIONS)
@pytest.mark.parametrize("min_hold", [2, 5, 7, 10])
@pytest.mark.parametrize("seed", range(5))
def test_every_completed_regime_run_lasts_at_least_min_hold(smooth, min_hold, seed):
    out = smooth(_random_regimes(400, seed), min_hold=min_hold)
    completed = runs(out)[:-1]          # last run may still be in progress

    assert all(length >= min_hold for _, length in completed)


@pytest.mark.parametrize("smooth", IMPLEMENTATIONS)
@pytest.mark.parametrize("seed", range(5))
def test_is_causal_output_at_t_ignores_future_inputs(smooth, seed):
    """The no-lookahead guarantee: smoothing a prefix == prefix of smoothing the whole."""
    seq = _random_regimes(150, seed)
    full = smooth(seq, min_hold=7)

    for t in range(len(seq)):
        assert smooth(seq[:t + 1], min_hold=7)[t] == full[t]


@pytest.mark.parametrize("seed", range(10))
def test_both_implementations_agree(seed):
    seq = _random_regimes(500, seed)
    assert r2.smooth_regimes_causal(seq) == pp.smooth_regimes_causal(seq)


def test_default_min_hold_matches_module_constant():
    assert r2.MIN_HOLD_DAYS == pp.MIN_HOLD_DAYS
