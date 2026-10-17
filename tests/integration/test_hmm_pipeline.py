"""
Integration tests for the HMM pipeline: train -> label -> walk-forward.

The headline test is the *future-invariance* check for walk-forward validation:
if the results of fold 0 change when data from after fold 0's test window is
removed, then future information leaked into that fold.
"""

import numpy as np
import pandas as pd
import pytest

import regime_detector_v2 as r2
from tests.helpers import REGIMES, make_sector_frame

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TRAIN_YEARS = 1
TEST_MONTHS = 6


@pytest.fixture(scope="module")
def market(synthetic_market_df):
    df = synthetic_market_df.copy()
    df['TrueRegime'] = 'Sideways'
    feat = r2.engineer_features(df)
    df_sec = make_sector_frame(df.index)
    return df, feat, df_sec


def _walk_forward(feat, df, df_sec):
    return r2.walk_forward_validation(feat, df, df_sec=df_sec, train_years=TRAIN_YEARS,
                                      test_months=TEST_MONTHS, n_states=4)


@pytest.fixture(scope="module")
def wf_full(market):
    df, feat, df_sec = market
    return _walk_forward(feat, df, df_sec)


# ── train_hmm / label_regimes ────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained(market):
    from sklearn.preprocessing import QuantileTransformer
    df, feat, _ = market
    X = QuantileTransformer(output_distribution='normal', n_quantiles=500,
                            random_state=42).fit_transform(r2.select_features_for_hmm(feat).values)
    model = r2.train_hmm(X, n_states=4, n_iter=100, n_init=2)
    return model, X


def test_trained_model_has_valid_regularised_parameters(trained):
    model, _ = trained

    assert model.n_components == 4
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(np.diag(model.transmat_) <= 0.97 + 1e-9), "no absorbing states after regularisation"
    assert np.all(model.transmat_ > 0)
    assert np.isclose(model.startprob_.sum(), 1.0)


def test_label_regimes_assigns_four_distinct_regimes_and_valid_posteriors(market, trained):
    df, feat, _ = market
    model, X = trained

    result, label_map, _ = r2.label_regimes(model, X, feat, df)

    assert sorted(v['name'] for v in label_map.values()) == sorted(REGIMES)
    assert set(result['Regime']) <= set(REGIMES)
    np.testing.assert_allclose(result[REGIMES].sum(axis=1), 1.0, atol=1e-6)
    assert result.index.equals(feat.index)


# ── walk-forward validation ──────────────────────────────────────────────

def test_walk_forward_produces_expanding_chronological_folds(wf_full):
    folds, daily = wf_full

    assert len(folds) >= 2
    assert folds['fold_start'].is_monotonic_increasing
    assert folds['n_train'].is_monotonic_increasing and folds['n_train'].nunique() == len(folds)


def test_oos_series_is_strictly_chronological_and_starts_after_first_training_window(market, wf_full):
    _, feat, _ = market
    _, daily = wf_full
    first_test_start = feat.index[0] + pd.DateOffset(years=TRAIN_YEARS)

    assert daily.index.is_monotonic_increasing
    assert not daily.index.has_duplicates
    assert daily.index.min() >= first_test_start, "training-period days must never be scored"
    assert np.isfinite(daily.values).all()


def test_fold_zero_is_unchanged_when_the_future_is_removed(market, wf_full):
    """
    No-lookahead proof: re-run with every row after fold 0's test window deleted.
    Fold 0 must be bit-for-bit identical.
    """
    df, feat, df_sec = market
    folds_full, daily_full = wf_full

    fold_end = feat.index[0] + pd.DateOffset(years=TRAIN_YEARS) + pd.DateOffset(months=TEST_MONTHS)
    cutoff = fold_end + pd.Timedelta(days=3)           # keep a few days so fold 0 still ends at fold_end
    feat_t, df_t, sec_t = feat[feat.index <= cutoff], df[df.index <= cutoff], df_sec[df_sec.index <= cutoff]

    folds_cut, daily_cut = _walk_forward(feat_t, df_t, sec_t)

    assert len(folds_cut) == 1
    for col in ('n_train', 'n_test', 'ann_ret_pct', 'sharpe', 'profit_factor', 'n_switches'):
        assert folds_cut.iloc[0][col] == pytest.approx(folds_full.iloc[0][col], rel=1e-9, abs=1e-12), col

    overlap = daily_cut.index
    pd.testing.assert_series_equal(daily_cut, daily_full.loc[overlap])


def test_corrupting_future_features_does_not_change_fold_zero(market, wf_full):
    """Same idea, different attack: poison (not delete) the data after fold 0."""
    df, feat, df_sec = market
    folds_full, _ = wf_full
    fold_end = feat.index[0] + pd.DateOffset(years=TRAIN_YEARS) + pd.DateOffset(months=TEST_MONTHS)

    poisoned = feat.copy()
    future = poisoned.index >= fold_end
    rng = np.random.default_rng(0)
    poisoned.loc[future, :] = poisoned.loc[future, :].values * 40 + rng.normal(0, 5, (future.sum(), poisoned.shape[1]))

    folds_poison, _ = _walk_forward(poisoned, df, df_sec)

    for col in ('ann_ret_pct', 'sharpe', 'n_switches'):
        assert folds_poison.iloc[0][col] == pytest.approx(folds_full.iloc[0][col], rel=1e-9, abs=1e-12), col
