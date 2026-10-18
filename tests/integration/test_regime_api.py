"""
HTTP-level tests for the FastAPI service using an in-memory model (no files, no network).

TestClient is used *without* a `with` block on purpose: that skips the startup
hook, so the module-level artifacts below are what the endpoints see.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import paper_portfolio as pp
import regime_api
from tests.helpers import REGIMES

pytestmark = pytest.mark.integration

N_FEAT = 11
LABELS = {0: {'name': 'Bull', 'color': '#10b981'}, 1: {'name': 'Bear', 'color': '#ef4444'},
          2: {'name': 'HighVol', 'color': '#f59e0b'}, 3: {'name': 'Sideways', 'color': '#3b82f6'}}
MIX = {r: {'BANKBEES': 1.0} for r in REGIMES}

PAYLOAD = dict(ret_5d_pct=1.0, vol_60d=0.15, price_vs_ma200=0.05, drawdown=-0.02,
               drawdown_diff20d=0.01, vix_vs_ma20=-0.1, yield_curve=0.4, cpi_yoy=5.0,
               iip_yoy=3.0, wpi_yoy=2.0, real_rate=0.5)


class SpyScaler:
    """Wraps a real scaler and records the vector the API hands to the model."""

    def __init__(self, inner):
        self.inner, self.seen = inner, None

    def transform(self, X):
        self.seen = np.array(X)
        return self.inner.transform(X)


@pytest.fixture
def client(monkeypatch):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(400, N_FEAT))
    scaler = SpyScaler(StandardScaler().fit(data))

    model = GaussianHMM(n_components=4, covariance_type='diag', n_iter=5, random_state=0).fit(
        scaler.inner.transform(data))

    monkeypatch.setattr(regime_api, '_model', model)
    monkeypatch.setattr(regime_api, '_scaler', scaler)
    monkeypatch.setattr(regime_api, '_label_map', LABELS)
    monkeypatch.setattr(regime_api, '_learned_sector_mix', MIX)
    c = TestClient(regime_api.app)
    c.spy = scaler
    return c


def test_health_reports_loaded_model(client):
    body = client.get('/health').json()

    assert body == {'status': 'online', 'model_loaded': True, 'n_states': 4}


def test_health_when_no_model_loaded(monkeypatch):
    monkeypatch.setattr(regime_api, '_model', None)

    body = TestClient(regime_api.app).get('/health').json()

    assert body['model_loaded'] is False and body['n_states'] is None


def test_predict_returns_a_coherent_regime_payload(client):
    r = client.post('/regime/predict', json=PAYLOAD)

    assert r.status_code == 200
    body = r.json()
    assert body['regime'] in REGIMES
    assert sum(body['posteriors'].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= body['market_exposure'] <= 1.0
    assert body['recommended_sector_mix'] == MIX[body['regime']]


def test_predicted_regime_is_the_argmax_posterior(client):
    body = client.post('/regime/predict', json=PAYLOAD).json()

    assert body['regime'] == max(body['posteriors'], key=body['posteriors'].get)


def test_market_exposure_is_posterior_weighted_regime_exposure(client):
    body = client.post('/regime/predict', json=PAYLOAD).json()

    expected = sum(p * regime_api.REGIME_EXPOSURE[r] for r, p in body['posteriors'].items())
    assert body['market_exposure'] == pytest.approx(expected, abs=1e-3)


def test_feature_vector_order_matches_the_training_feature_order(client):
    """
    Train/serve skew guard. Each field gets a unique value; position i of the
    vector sent to the scaler must be the field that plays the role of
    pp.FEATURE_COLS[i].  (The API names two fields differently from training:
    wpi_yoy <-> wpi_mom_vs_ma3m and real_rate <-> real_rate_mom.)
    """
    api_field_for = {'wpi_mom_vs_ma3m': 'wpi_yoy', 'real_rate_mom': 'real_rate'}
    payload = {f: float(i + 1) for i, f in enumerate(PAYLOAD)}

    assert client.post('/regime/predict', json=payload).status_code == 200

    expected = [payload[api_field_for.get(col, col)] for col in pp.FEATURE_COLS]
    assert client.spy.seen.shape == (1, N_FEAT)
    assert client.spy.seen[0].tolist() == expected


def test_predict_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(regime_api, '_model', None)
    monkeypatch.setattr(regime_api, '_scaler', None)

    assert TestClient(regime_api.app).post('/regime/predict', json=PAYLOAD).status_code == 503


@pytest.mark.parametrize("missing", list(PAYLOAD))
def test_predict_rejects_payload_missing_any_field(client, missing):
    bad = {k: v for k, v in PAYLOAD.items() if k != missing}

    assert client.post('/regime/predict', json=bad).status_code == 422


def test_predict_rejects_non_numeric_values(client):
    assert client.post('/regime/predict', json={**PAYLOAD, 'drawdown': 'lots'}).status_code == 422


@pytest.mark.parametrize("regime,exposure", [('Bull', 1.0), ('Bear', 0.0), ('HighVol', 0.2), ('Sideways', 0.6)])
def test_strategy_endpoint_exposures(client, regime, exposure):
    body = client.get('/regime/strategy', params={'regime': regime}).json()

    assert body['equity_exposure'] == exposure
    assert body['cash_exposure'] == pytest.approx(1.0 - exposure)
    assert body['sector_mix'] == MIX[regime]


def test_strategy_endpoint_rejects_unknown_regime(client):
    assert client.get('/regime/strategy', params={'regime': 'Euphoria'}).status_code == 400


def test_current_regime_reads_latest_history_row(client):
    r = client.get('/regime/current')

    assert r.status_code == 200
    body = r.json()
    assert body['regime'] in REGIMES
    assert sum(body['posteriors'].values()) == pytest.approx(1.0, abs=1e-2)
    assert 0.0 <= body['market_exposure'] <= 1.0


def test_openapi_schema_is_served(client):
    paths = client.get('/openapi.json').json()['paths']

    assert {'/health', '/regime/predict', '/regime/strategy', '/regime/current', '/portfolio/reconcile', '/portfolio/ledger', '/macro/ingest', '/market/ingest', '/macro/sync'} <= set(paths)


def test_portfolio_reconcile_endpoint(client):
    r = client.get('/portfolio/reconcile')
    assert r.status_code == 200
    body = r.json()
    assert 'status' in body
    assert body['status'] in ('RECONCILED', 'BREAK_DETECTED')
    assert 'position_breaks' in body
    assert 'stop_loss_integrity' in body


def test_portfolio_ledger_endpoint(client):
    r = client.get('/portfolio/ledger', params={'limit': 5})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    if len(body) > 0:
        assert 'nav' in body[0]
        assert 'cash' in body[0]
        assert 'regime' in body[0]


def test_macro_ingest_endpoint(client):
    payload = {
        "effective_date": "2026-10-12",
        "cpi_yoy": 5.4,
        "cpi_mom": 0.2,
        "iip_yoy": 4.5,
        "wpi_yoy": 2.1,
        "repo_rate": 6.5,
        "gsec_10y": 7.05,
        "gsec_2y": 5.25,
        "forward_fill": True
    }
    r = client.post('/macro/ingest', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'SUCCESS'
    assert body['effective_date'] == "2026-10-12"


def test_macro_sync_endpoint(client):
    r = client.post('/macro/sync')
    assert r.status_code == 200
    body = r.json()
    assert 'status' in body

