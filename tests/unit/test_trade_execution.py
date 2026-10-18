"""Tests for NSE charge arithmetic and the buy/sell order-execution helpers in paper_portfolio.py."""

import numpy as np
import pytest

import paper_portfolio as pp


# ── Charges ──────────────────────────────────────────────────────────────

def test_buy_charges_hand_computed_for_one_lakh():
    # STT 100 + exch 2.97 + SEBI 0.10 + GST 18%×3.07 + stamp 15 + slippage 50
    assert pp.compute_buy_charges(100_000) == pytest.approx(168.6226, abs=1e-4)


def test_sell_charges_hand_computed_with_two_scrips():
    # STT 100 + exch 2.97 + SEBI 0.10 + GST 0.5526 + DP 2×15.93 + slippage 50
    assert pp.compute_sell_charges(100_000, n_scrips=2) == pytest.approx(185.4826, abs=1e-4)


def test_buy_charges_scale_linearly_with_turnover():
    assert pp.compute_buy_charges(2_000_000) == pytest.approx(2 * pp.compute_buy_charges(1_000_000))


def test_zero_turnover_costs():
    assert pp.compute_buy_charges(0) == 0.0
    # only the flat DP fee remains on a sell
    assert pp.compute_sell_charges(0, n_scrips=3) == pytest.approx(3 * pp.DP_CHARGE_PER_SCRIP)


def test_stamp_duty_is_buy_side_only_and_dp_is_sell_side_only():
    turnover = 1_000_000
    buy = pp.compute_buy_charges(turnover)
    sell = pp.compute_sell_charges(turnover, n_scrips=1)

    assert buy - sell == pytest.approx(turnover * pp.TC_STAMP - pp.DP_CHARGE_PER_SCRIP)


def test_dp_charge_is_per_scrip():
    one = pp.compute_sell_charges(500_000, n_scrips=1)
    four = pp.compute_sell_charges(500_000, n_scrips=4)

    assert four - one == pytest.approx(3 * pp.DP_CHARGE_PER_SCRIP)


# ── _execute_buy ─────────────────────────────────────────────────────────

ETFS = ['BANKBEES', 'ITBEES', 'PHARMABEES']


def test_buy_with_no_weights_is_a_noop():
    assert pp._execute_buy(100_000.0, {}, ETFS, {'BANKBEES': 100.0}) == ({}, {}, 100_000.0, 0.0)


def test_buy_cash_conservation_and_whole_units():
    prices = {'BANKBEES': 523.17, 'ITBEES': 41.9, 'PHARMABEES': 17.33}
    weights = {'BANKBEES': 0.5, 'ITBEES': 0.3, 'PHARMABEES': 0.2}
    cash = 1_000_000.0

    holdings, cost_basis, new_cash, charges = pp._execute_buy(cash, weights, ETFS, prices)

    spent = sum(holdings[e] * prices[e] for e in ETFS)
    assert all(float(u).is_integer() for u in holdings.values())
    assert new_cash == pytest.approx(cash - spent - charges)
    assert 0 <= new_cash < cash


@pytest.mark.parametrize("seed", range(10))
def test_buy_never_overdraws_cash(seed):
    rng = np.random.default_rng(seed)
    prices = {e: float(rng.uniform(5, 800)) for e in ETFS}
    w = rng.random(3)
    weights = dict(zip(ETFS, w / w.sum(), strict=True))
    cash = float(rng.uniform(10_000, 5_000_000))

    _, _, new_cash, charges = pp._execute_buy(cash, weights, ETFS, prices)

    assert new_cash >= -1e-6
    assert charges >= 0


def test_buy_skips_etf_when_allocation_cannot_afford_one_unit():
    holdings, _, cash, _ = pp._execute_buy(1_000.0, {'BANKBEES': 1.0}, ETFS, {'BANKBEES': 5_000.0})

    assert holdings['BANKBEES'] == 0
    assert cash == pytest.approx(1_000.0)


def test_cost_basis_includes_buy_charges():
    prices = {'BANKBEES': 100.0}
    holdings, cost_basis, _, charges = pp._execute_buy(100_000.0, {'BANKBEES': 1.0}, ['BANKBEES'], prices)

    units = holdings['BANKBEES']
    assert cost_basis['BANKBEES'] == pytest.approx((units * 100.0 + charges) / units)
    assert cost_basis['BANKBEES'] > 100.0


# ── _execute_sell ────────────────────────────────────────────────────────

def test_sell_liquidates_everything_and_zeroes_state_in_place():
    holdings = {'BANKBEES': 100, 'ITBEES': 50, 'PHARMABEES': 0}
    cost_basis = {'BANKBEES': 90.0, 'ITBEES': 40.0, 'PHARMABEES': 0.0}
    prices = {'BANKBEES': 100.0, 'ITBEES': 45.0, 'PHARMABEES': 10.0}

    proceeds, charges, gain = pp._execute_sell(holdings, cost_basis, ETFS, prices)

    turnover = 100 * 100.0 + 50 * 45.0
    assert holdings == {'BANKBEES': 0, 'ITBEES': 0, 'PHARMABEES': 0}
    assert all(v == 0.0 for v in cost_basis.values())
    assert gain == pytest.approx(100 * 10.0 + 50 * 5.0)
    assert charges == pytest.approx(pp.compute_sell_charges(turnover, n_scrips=2))
    assert proceeds == pytest.approx(turnover - charges)


def test_sell_with_nothing_held_returns_zeros():
    assert pp._execute_sell({'BANKBEES': 0}, {'BANKBEES': 0.0}, ['BANKBEES'], {'BANKBEES': 100.0}) == (0.0, 0.0, 0.0)


def test_round_trip_at_unchanged_price_loses_exactly_the_buy_charges():
    price = {'BANKBEES': 250.0}
    cash0 = 500_000.0
    holdings, cost_basis, cash1, buy_chg = pp._execute_buy(cash0, {'BANKBEES': 1.0}, ['BANKBEES'], price)

    proceeds, sell_chg, gain = pp._execute_sell(holdings, cost_basis, ['BANKBEES'], price)

    assert gain == pytest.approx(-buy_chg)                       # basis included the buy costs
    assert cash1 + proceeds == pytest.approx(cash0 - buy_chg - sell_chg)
    assert cash1 + proceeds < cash0
