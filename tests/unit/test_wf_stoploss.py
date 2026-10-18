"""Unit tests for the stop-loss overlay used by walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

import regime_detector_v2 as r2

REPO = 0.065 / 252


def _run(returns, above_ma, stop_dd=-0.06, cooldown=3):
    state = r2.new_stoploss_state()
    eff = [
        r2.stoploss_overlay_step(state, r, REPO, a, stop_dd=stop_dd, cooldown_days=cooldown)
        for r, a in zip(returns, above_ma)
    ]
    return np.array(eff), state


def test_stop_triggers_at_threshold_and_caps_loss():
    # -4% then -4%: cumulative -7.8% would breach -6% -> exit capped near -6% (+ half friction)
    eff, state = _run([-0.04, -0.04], [False, False])
    assert state['stops'] == 1 and state['in_cash']
    cum = np.exp(eff.sum()) - 1
    assert -0.065 < cum < -0.058


def test_no_stop_when_disabled():
    eff, state = _run([-0.04, -0.04, -0.04], [False] * 3, stop_dd=None)
    assert state['stops'] == 0
    assert np.isclose(eff.sum(), -0.12)


def test_cash_earns_repo_yield_while_stopped_out():
    rets = [-0.07] + [0.02] * 5
    eff, state = _run(rets, [False] * 6)
    # after the stop, market gains are NOT captured; only repo accrues
    assert state['in_cash']
    assert np.allclose(eff[1:], REPO)
    assert state['days_cash'] == 6


def test_reentry_requires_ma_and_cooldown():
    rets = [-0.07] + [0.01] * 8
    above = [False, True, True, True, True, True, True, True, True]
    eff, state = _run(rets, above, cooldown=3)
    # cooldown of 3 days in cash -> re-enter once above MA, then market returns flow again
    assert not state['in_cash']
    assert state['stops'] == 1
    assert eff[-1] == pytest.approx(0.01)

    # never above the MA -> never re-enters
    _, state_b = _run(rets, [False] * 9, cooldown=3)
    assert state_b['in_cash']


def test_peak_resets_on_reentry():
    # crash -> stop, recover above MA -> re-enter; a fresh -6% needed to stop again
    rets = [-0.07] + [0.0] * 4 + [-0.03, -0.02]
    above = [False, True, True, True, True, True, True]
    _, state = _run(rets, above, cooldown=3)
    assert state['stops'] == 1  # -5% from re-entry peak does not retrigger


def test_walk_forward_signature_exposes_stop_params():
    import inspect
    sig = inspect.signature(r2.walk_forward_validation)
    assert sig.parameters['stop_dd'].default == r2.STOP_LOSS_DD == -0.06
    assert sig.parameters['cooldown_days'].default == r2.STOP_COOLDOWN_DAYS


def test_model_selection_removed():
    assert not hasattr(r2, 'bic_aic_model_selection')
    assert not hasattr(r2, 'plot_model_selection')


def test_walk_forward_runs_with_and_without_stoploss(synthetic_market_df):
    from tests.helpers import make_sector_frame
    df = synthetic_market_df
    feat = r2.engineer_features(df)
    sec = make_sector_frame(df.index)
    kw = dict(train_years=1, test_months=6, n_states=4)
    f_on, s_on = r2.walk_forward_validation(feat, df, df_sec=sec, **kw)
    f_off, s_off = r2.walk_forward_validation(feat, df, df_sec=sec, stop_dd=None, **kw)
    assert {'stop_triggers', 'cash_days'} <= set(f_on.columns)
    assert len(s_on) == len(s_off) > 0
    assert f_off['stop_triggers'].sum() == 0
    # The overlay must actually engage on this seeded crash-prone sample and change the path
    assert f_on['stop_triggers'].sum() > 0
    assert f_on['cash_days'].sum() >= f_on['stop_triggers'].sum()
    assert not np.allclose(s_on.values, s_off.values)
    assert isinstance(s_on, pd.Series)
