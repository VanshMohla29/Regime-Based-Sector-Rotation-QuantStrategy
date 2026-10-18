"""Tests for feature engineering + the HMM feature contract shared with the API / paper portfolio."""

import pandas as pd
import pytest

import paper_portfolio as pp
import regime_detector_v2 as r2


@pytest.fixture(scope="module")
def feat(synthetic_market_df):
    return r2.engineer_features(synthetic_market_df)


def test_no_nans_survive_engineering(feat):
    assert not feat.isna().any().any()
    assert len(feat) > 400


def test_drawdown_is_bounded_between_minus_one_and_zero(feat):
    assert (feat['drawdown'] <= 0).all()
    assert (feat['drawdown'] > -1).all()


def test_drawdown_matches_independent_computation(synthetic_market_df, feat):
    nifty = synthetic_market_df['NIFTY']
    expected = (nifty / nifty.cummax() - 1).loc[feat.index]

    pd.testing.assert_series_equal(feat['drawdown'], expected, check_names=False)


def test_rsi_stays_in_0_100(feat):
    assert feat['rsi14'].between(0, 100).all()


def test_volatilities_are_non_negative(feat):
    for col in ('vol_1d', 'vol_5d', 'vol_20d', 'vol_60d'):
        assert (feat[col] >= 0).all(), col


def test_five_day_return_pct_matches_price_ratio(synthetic_market_df, feat):
    nifty = synthetic_market_df['NIFTY']
    expected = ((nifty / nifty.shift(5) - 1) * 100).loc[feat.index]

    pd.testing.assert_series_equal(feat['ret_5d_pct'], expected, check_names=False)


def test_features_are_causal_truncating_future_does_not_change_past(synthetic_market_df, feat):
    """
    A feature at date t may only depend on data up to t.  Re-engineering on a
    truncated history must reproduce the same rows (beyond the warm-up window).
    """
    cut = 600
    truncated = r2.engineer_features(synthetic_market_df.iloc[:cut])
    common = truncated.index.intersection(feat.index)

    assert len(common) > 100
    pd.testing.assert_frame_equal(truncated.loc[common], feat.loc[common])


# ── The training / serving contract ──────────────────────────────────────

EXPECTED_HMM_FEATURES = [
    'ret_5d_pct', 'vol_60d', 'price_vs_ma200', 'drawdown', 'drawdown_diff20d',
    'vix_vs_ma20', 'yield_curve', 'cpi_yoy', 'iip_yoy', 'wpi_mom_vs_ma3m', 'real_rate_mom',
]


def test_hmm_feature_set_is_the_documented_eleven(feat):
    assert list(r2.select_features_for_hmm(feat).columns) == EXPECTED_HMM_FEATURES


def test_paper_portfolio_feature_cols_match_training_features(feat):
    """Train/serve skew guard: a reordered column silently feeds the wrong data to the HMM."""
    assert list(r2.select_features_for_hmm(feat).columns) == pp.FEATURE_COLS
