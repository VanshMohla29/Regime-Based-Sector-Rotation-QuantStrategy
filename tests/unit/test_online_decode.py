"""Tests for online_forward_decode() — the causal replacement for batch Viterbi in OOS evaluation."""

import numpy as np
import pytest

import regime_detector_v2 as r2


def _two_cluster_observations(seed=0):
    rng = np.random.default_rng(seed)
    truth = np.array([0] * 60 + [1] * 60 + [0] * 40)
    x = np.where(truth == 0, -3.0, 3.0) + rng.normal(0, 0.5, truth.size)
    return x.reshape(-1, 1), truth


def test_recovers_well_separated_states(two_state_hmm):
    X, truth = _two_cluster_observations()

    states = r2.online_forward_decode(two_state_hmm, X)

    assert (states == truth).mean() > 0.95


def test_output_shape_and_dtype(two_state_hmm):
    X, _ = _two_cluster_observations()

    states = r2.online_forward_decode(two_state_hmm, X)

    assert states.shape == (len(X),)
    assert np.issubdtype(states.dtype, np.integer)
    assert set(np.unique(states)) <= {0, 1}


@pytest.mark.parametrize("seed", range(5))
def test_decode_is_causal_prefix_consistency(two_state_hmm, seed):
    """
    Decoding only X[:k] must give exactly the first k states of decoding all of X.
    If a backward pass (Viterbi/smoothing) leaked in, early states would change
    once later observations were appended.
    """
    X, _ = _two_cluster_observations(seed)
    full = r2.online_forward_decode(two_state_hmm, X)

    for k in (1, 5, 30, 61, 100, len(X) - 1):
        np.testing.assert_array_equal(
            r2.online_forward_decode(two_state_hmm, X[:k]), full[:k]
        )


def test_future_shock_does_not_rewrite_the_past(two_state_hmm):
    X, _ = _two_cluster_observations()
    baseline = r2.online_forward_decode(two_state_hmm, X)

    shocked = X.copy()
    shocked[100:] = 3.0 * np.sign(shocked[100:] + 1e-9) * 50     # wild future data
    states = r2.online_forward_decode(two_state_hmm, shocked)

    np.testing.assert_array_equal(states[:100], baseline[:100])


def test_viterbi_can_differ_but_online_never_uses_future(two_state_hmm):
    """Documents *why* the online decoder exists: batch decode is allowed to look ahead."""
    X, _ = _two_cluster_observations()
    online = r2.online_forward_decode(two_state_hmm, X)
    _, viterbi = two_state_hmm.decode(X, algorithm='viterbi')

    # On clean data they should broadly agree; the guarantee we care about is causality above.
    assert (online == viterbi).mean() > 0.9
