"""
India HMM Regime Detector — FastAPI Service
=============================================
Serves the trained HMM model as a REST API.
The React dashboard (or any client) can POST market features
and receive a real-time regime prediction.

Install:
    pip install fastapi uvicorn hmmlearn scikit-learn numpy pandas

Run:
    uvicorn regime_api:app --host 0.0.0.0 --port 8000
"""

import os
import json
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database.connection import get_db
from database.ingest_service import (
    auto_sync_macro_from_sources,
    ingest_daily_update,
    ingest_macro_release,
)
from database.models import PortfolioLedger
from reconciliation.reconciler import DailyReconciliationEngine

app = FastAPI(
    title="India HMM Regime Detector API",
    description="Probabilistic market regime detection for NIFTY 50 (Gaussian HMM)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
if not os.path.exists(os.path.join(ARTIFACT_DIR, "hmm_model.pkl")):
    ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = PROJECT_ROOT

MODEL_PATH   = os.path.join(ARTIFACT_DIR, "hmm_model.pkl")
SCALER_PATH  = os.path.join(ARTIFACT_DIR, "hmm_scaler.pkl")
LABEL_PATH   = os.path.join(ARTIFACT_DIR, "label_map.pkl")
MIX_PATH     = os.path.join(ARTIFACT_DIR, "learned_sector_mix.json")

_model      = None
_scaler     = None
_label_map  = None
_learned_sector_mix = {}

def load_artifacts():
    global _model, _scaler, _label_map, _learned_sector_mix
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, 'rb') as f:
            _label_map = pickle.load(f)
    if os.path.exists(MIX_PATH):
        with open(MIX_PATH, 'r') as f:
            _learned_sector_mix = json.load(f)

@app.on_event("startup")
def startup():
    load_artifacts()

class MarketFeatures(BaseModel):
    ret_5d_pct:        float = Field(..., description="5-day rolling return percentage (%)")
    vol_60d:           float = Field(..., description="60-day quarterly annualized realized volatility")
    price_vs_ma200:    float = Field(..., description="Price / 200-day SMA - 1 (secular trend ratio)")
    drawdown:          float = Field(..., description="Drawdown from all-time peak (negative)")
    drawdown_diff20d:  float = Field(..., description="Drawdown minus drawdown 20 trading days ago (recovery velocity)")
    vix_vs_ma20:       float = Field(..., description="India VIX / 20-day SMA - 1 (relative volatility shock)")
    yield_curve:       float = Field(..., description="10Y - 2Y sovereign yield curve spread (%)")
    cpi_yoy:           float = Field(..., description="Consumer Price Index YoY inflation (%)")
    iip_yoy:           float = Field(..., description="Index of Industrial Production YoY growth (%)")
    wpi_yoy:           float = Field(..., description="Wholesale Price Index YoY inflation (%)")
    real_rate:         float = Field(..., description="Real policy rate momentum: 1-year change in Repo Rate - CPI (%)")

class MacroIngestRequest(BaseModel):
    effective_date: str = Field(..., description="Publication / effective date in YYYY-MM-DD")
    cpi_yoy: Optional[float] = Field(None, description="All-India CPI YoY %")
    cpi_mom: Optional[float] = Field(None, description="CPI MoM %")
    iip_yoy: Optional[float] = Field(None, description="IIP YoY %")
    wpi_yoy: Optional[float] = Field(None, description="WPI YoY %")
    wpi_mom: Optional[float] = Field(None, description="WPI MoM %")
    repo_rate: Optional[float] = Field(None, description="RBI Policy Repo Rate %")
    gsec_10y: Optional[float] = Field(None, description="10Y G-Sec Yield %")
    gsec_2y: Optional[float] = Field(None, description="2Y G-Sec Yield %")
    forward_fill: bool = Field(True, description="Forward-fill subsequent trading days")

REGIME_EXPOSURE = {
    "Bull":     1.0,
    "Bear":     0.0,
    "HighVol":  0.2,
    "Sideways": 0.6,
}

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": _model is not None,
        "n_states": _model.n_components if _model else None,
    }

@app.post("/regime/predict")
def predict_regime(features: MarketFeatures):
    if _model is None or _scaler is None:
        raise HTTPException(503, "HMM model not loaded. Run train pipeline first.")

    raw = np.array([[
        features.ret_5d_pct,
        features.vol_60d,
        features.price_vs_ma200,
        features.drawdown,
        features.drawdown_diff20d,
        features.vix_vs_ma20,
        features.yield_curve,
        features.cpi_yoy,
        features.iip_yoy,
        features.wpi_yoy,
        features.real_rate,
    ]])

    scaled = _scaler.transform(raw)
    state = int(_model.predict(scaled)[0])
    posteriors = _model.predict_proba(scaled)[0]

    s_name = _label_map.get(state, {}).get("name", "Sideways") if _label_map else f"State_{state}"
    color  = _label_map.get(state, {}).get("color", "#3b82f6") if _label_map else "#3b82f6"

    probs = {}
    if _label_map:
        for s_idx, prob in enumerate(posteriors):
            r = _label_map.get(s_idx, {}).get("name", f"State_{s_idx}")
            probs[r] = round(float(prob), 4)

    exp = sum(probs.get(r, 0.0) * REGIME_EXPOSURE.get(r, 0.5) for r in probs)
    return {
        "regime": s_name,
        "color": color,
        "market_exposure": round(exp, 4),
        "posteriors": probs,
        "recommended_sector_mix": _learned_sector_mix.get(s_name, {})
    }

@app.get("/regime/strategy")
def regime_strategy(regime: str):
    if regime not in REGIME_EXPOSURE:
        raise HTTPException(400, f"Unknown regime. Choose from: {list(REGIME_EXPOSURE.keys())}")
    eq = REGIME_EXPOSURE[regime]
    return {
        "regime": regime,
        "equity_exposure": eq,
        "cash_exposure": round(1.0 - eq, 2),
        "sector_mix": _learned_sector_mix.get(regime, {})
    }

@app.get("/regime/current")
def current_regime():
    csv_path = os.path.join(DATA_DIR, "regime_history_v2.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(ARTIFACT_DIR, "regime_history_v2.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "regime_history_v2.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(503, "Regime history data not available.")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    latest = df.iloc[-1]
    curr_reg = latest["Regime"]
    probs = {
        "Bull": round(float(latest.get("Bull", 0.0)), 4),
        "Bear": round(float(latest.get("Bear", 0.0)), 4),
        "HighVol": round(float(latest.get("HighVol", 0.0)), 4),
        "Sideways": round(float(latest.get("Sideways", 0.0)), 4),
    }
    exp = sum(probs.get(r, 0.0) * REGIME_EXPOSURE.get(r, 0.5) for r in probs)
    return {
        "date": str(latest.name)[:10],
        "regime": curr_reg,
        "nifty": float(latest.get("NIFTY", 0)),
        "vix": float(latest.get("VIX", 0)),
        "cpi": float(latest.get("CPI", 0)),
        "wpi": float(latest.get("WPI", 0)),
        "yield_curve": float(latest.get("YieldCurve", 0)),
        "posteriors": probs,
        "market_exposure": round(exp, 4),
        "recommended_sector_mix": _learned_sector_mix.get(curr_reg, {}),
    }


@app.get("/portfolio/reconcile")
def get_reconciliation_report(target_date: Optional[str] = None):
    """
    Audits the portfolio on target_date (defaults to latest trading day)
    and returns position breaks, cash conservation, and stop-loss integrity.
    """
    try:
        with get_db() as db:
            if target_date:
                dt = datetime.strptime(target_date, "%Y-%m-%d").date()
            else:
                latest = db.query(PortfolioLedger).order_by(PortfolioLedger.time.desc()).first()
                dt = latest.time.date() if latest else datetime.now().date()

            report = DailyReconciliationEngine.reconcile(db, dt)
            return report.audit_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconciliation error: {str(e)}")


@app.get("/portfolio/ledger")
def get_portfolio_ledger(limit: int = 30):
    """Returns recent portfolio accounting ledger records from the database."""
    try:
        with get_db() as db:
            records = (
                db.query(PortfolioLedger)
                .order_by(PortfolioLedger.time.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "date": r.time.strftime("%Y-%m-%d"),
                    "nav": round(r.nav, 2),
                    "cash": round(r.cash_balance, 2),
                    "equity": round(r.equity_value, 2),
                    "drawdown_pct": round(r.drawdown_pct * 100, 2),
                    "daily_return_pct": round(r.daily_return_pct * 100, 2),
                    "in_cash": r.in_cash,
                    "regime": r.active_regime,
                }
                for r in records
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ledger error: {str(e)}")


@app.post("/macro/ingest")
def post_macro_release(req: MacroIngestRequest):
    """
    Ingests official monthly macroeconomic releases (MoSPI CPI/IIP, DPIIT WPI, RBI Repo Rate)
    into the TimescaleDB MacroIndicator hypertable.
    """
    try:
        res = ingest_macro_release(
            effective_date=req.effective_date,
            cpi_yoy=req.cpi_yoy,
            cpi_mom=req.cpi_mom,
            iip_yoy=req.iip_yoy,
            wpi_yoy=req.wpi_yoy,
            wpi_mom=req.wpi_mom,
            repo_rate=req.repo_rate,
            gsec_10y=req.gsec_10y,
            gsec_2y=req.gsec_2y,
            forward_fill=req.forward_fill,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Macro ingestion error: {str(e)}")


@app.post("/market/ingest")
def post_daily_market_update(force_reload: bool = False):
    """
    Triggers incremental daily ingestion of market bars and macro features into TimescaleDB.
    """
    try:
        res = ingest_daily_update(force_reload=force_reload)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market ingestion error: {str(e)}")


@app.post("/macro/sync")
def post_macro_auto_sync():
    """
    Automatically polls official macroeconomic mirrors (FRED/RBI/MoSPI) and ingests
    any new releases into TimescaleDB without blocking live inference.
    """
    try:
        res = auto_sync_macro_from_sources()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Macro auto-sync error: {str(e)}")
