"""
Shared pytest fixtures.

Every fixture here is fully synthetic and deterministic (fixed seeds) so the
whole suite runs offline in seconds — no Yahoo Finance / MoSPI / RBI calls.
"""

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM


@pytest.fixture(scope="session")
def business_days():
    """~3.5 years of weekday dates."""
    return pd.bdate_range("2021-01-01", periods=900)


@pytest.fixture(scope="session")
def synthetic_market_df(business_days):
    """
    Minimal replica of the frame produced by fetch_live_market_data():
    NIFTY, Returns (log), VIX and the macro columns engineer_features() reads.
    """
    rng = np.random.default_rng(42)
    n = len(business_days)
    log_ret = rng.normal(0.0004, 0.01, n)
    nifty = 15000 * np.exp(np.cumsum(log_ret))
    vix = np.clip(15 + np.cumsum(rng.normal(0, 0.4, n)) * 0.2 + rng.normal(0, 1, n), 9, 60)

    df = pd.DataFrame({
        'NIFTY': nifty,
        'Returns': log_ret,
        'VIX': vix,
        'YieldCurve': 0.5 + np.cumsum(rng.normal(0, 0.01, n)),
        'GSecYield10': 7.0 + np.cumsum(rng.normal(0, 0.01, n)),
        'RepoRate': np.full(n, 6.5),
        'CPI': 5.0 + np.cumsum(rng.normal(0, 0.02, n)),
        'IIP': 3.0 + np.cumsum(rng.normal(0, 0.03, n)),
        'WPI': 2.0 + np.cumsum(rng.normal(0, 0.02, n)),
    }, index=business_days)
    df['RealRate'] = df['RepoRate'] - df['CPI']
    return df


@pytest.fixture
def two_state_hmm():
    """
    Hand-parameterised 2-state, 1-feature Gaussian HMM (no EM training, so it is
    deterministic and instant).  State 0 ~ N(-3, 1), state 1 ~ N(+3, 1).
    """
    model = GaussianHMM(n_components=2, covariance_type='diag')
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.95, 0.05],
                                [0.05, 0.95]])
    model.means_ = np.array([[-3.0], [3.0]])
    model.covars_ = np.array([[1.0], [1.0]])
    return model
