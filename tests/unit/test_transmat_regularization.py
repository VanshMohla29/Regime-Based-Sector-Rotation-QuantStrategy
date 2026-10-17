"""Unit tests for regularize_transmat() — the Dirichlet fix for absorbing HMM states."""

import numpy as np
import pytest

import regime_detector_v2 as r2


def _random_stochastic(k, seed):
    rng = np.random.default_rng(seed)
    m = rng.random((k, k)) + 0.01
    return m / m.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("k", [2, 3, 4, 6])
@pytest.mark.parametrize("seed", range(5))
def test_output_is_row_stochastic_and_non_negative(k, seed):
    reg = r2.regularize_transmat(_random_stochastic(k, seed))

    assert reg.shape == (k, k)
    assert np.all(reg >= 0)
    np.testing.assert_allclose(reg.sum(axis=1), 1.0, atol=1e-12)


def test_absorbing_identity_matrix_is_made_ergodic():
    """A fully absorbing chain (P(i->i)=1) must come out communicating."""
    reg = r2.regularize_transmat(np.eye(4), max_diag=0.97, min_offdiag=0.005)

    assert np.all(np.diag(reg) <= 0.97 + 1e-9)
    off_diag = reg[~np.eye(4, dtype=bool)]
    assert np.all(off_diag >= 0.005 - 1e-12)


def test_diagonal_never_exceeds_max_diag():
    reg = r2.regularize_transmat(
        np.array([[0.9999, 0.0001, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.9990, 0.0010],
                  [0.0005, 0.0, 0.0, 0.9995]]),
        max_diag=0.98,
    )
    assert np.all(np.diag(reg) <= 0.98 + 1e-9)


def test_every_pair_of_states_is_reachable_after_regularisation():
    reg = r2.regularize_transmat(np.eye(4))
    assert np.all(reg > 0), "strictly positive matrix => irreducible & aperiodic chain"


def test_moderate_matrix_is_only_lightly_perturbed():
    original = np.array([[0.90, 0.05, 0.03, 0.02],
                         [0.04, 0.90, 0.03, 0.03],
                         [0.05, 0.05, 0.85, 0.05],
                         [0.02, 0.03, 0.05, 0.90]])
    reg = r2.regularize_transmat(original, alpha=1.0, max_diag=0.98)

    assert np.max(np.abs(reg - original)) < 0.01


def test_input_matrix_is_not_mutated():
    original = np.eye(3)
    snapshot = original.copy()

    r2.regularize_transmat(original)

    np.testing.assert_array_equal(original, snapshot)


def test_higher_alpha_smooths_more():
    base = np.array([[0.99, 0.01], [0.01, 0.99]])
    weak = r2.regularize_transmat(base, alpha=0.1, max_diag=1.0, min_offdiag=0.0)
    strong = r2.regularize_transmat(base, alpha=50.0, max_diag=1.0, min_offdiag=0.0)

    # stronger prior pulls the diagonal further toward uniform (0.5)
    assert strong[0, 0] < weak[0, 0]
