"""
Contract tests for the *committed* frozen artifacts (hmm_model.pkl, hmm_scaler.pkl,
label_map.pkl, learned_sector_mix.json, regime_history_v2.csv).

The API and the live paper portfolio both depend on these files; if a retrain
changes their shape/semantics this suite fails before anything is deployed.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

import paper_portfolio as pp
from tests.helpers import REGIMES, SECTORS

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _path(name):
    return os.path.join(ROOT, name)


@pytest.fixture(scope="module")
def model():
    with open(_path('hmm_model.pkl'), 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def scaler():
    with open(_path('hmm_scaler.pkl'), 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def label_map():
    with open(_path('label_map.pkl'), 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def sector_mix():
    with open(_path('learned_sector_mix.json')) as f:
        return json.load(f)


def test_model_and_scaler_agree_on_the_eleven_feature_contract(model, scaler):
    n = len(pp.FEATURE_COLS)
    assert n == 11
    assert model.means_.shape == (4, n)
    assert scaler.n_features_in_ == n


def test_transition_matrix_is_stochastic_and_has_no_absorbing_state(model):
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(np.diag(model.transmat_) < 0.99)


def test_label_map_covers_every_state_with_a_unique_known_regime(model, label_map):
    assert sorted(label_map) == list(range(model.n_components))
    assert sorted(v['name'] for v in label_map.values()) == sorted(REGIMES)
    assert all(v['color'].startswith('#') for v in label_map.values())


def test_frozen_model_scores_a_finite_posterior_for_a_neutral_observation(model, scaler):
    x = scaler.transform(np.zeros((1, len(pp.FEATURE_COLS))))
    post = model.predict_proba(x)[0]

    assert np.isfinite(post).all()
    assert post.sum() == pytest.approx(1.0)


def test_sector_mix_covers_all_regimes_and_sectors(sector_mix):
    assert sorted(sector_mix) == sorted(REGIMES)
    for regime, weights in sector_mix.items():
        assert sorted(weights) == sorted(SECTORS), regime


def test_sector_mix_weights_are_a_valid_capped_long_only_portfolio(sector_mix):
    for regime, weights in sector_mix.items():
        w = np.array(list(weights.values()))
        assert (w >= 0).all(), regime
        assert w.sum() == pytest.approx(1.0, abs=0.01), regime       # rounded to 3 dp
        assert (w <= 0.40 + 1e-9).all(), f"{regime}: per-sector cap of 40% breached"


def test_regime_history_csv_is_well_formed():
    df = pd.read_csv(_path('regime_history_v2.csv'), index_col=0, parse_dates=True)

    assert df.index.is_monotonic_increasing
    assert set(df['Regime'].unique()) <= set(REGIMES)
    np.testing.assert_allclose(df[REGIMES].sum(axis=1), 1.0, atol=1e-3)
    assert (df['NIFTY'] > 0).all()
