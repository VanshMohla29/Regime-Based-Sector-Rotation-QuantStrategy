"""
Database module for India Market Regime Detector.
Supports PostgreSQL + TimescaleDB with automatic fallback to SQLite.
"""

from .connection import get_db, get_engine, init_db, SessionLocal
from .models import (
    Base,
    MarketBar,
    MacroIndicator,
    FeatureStore,
    RegimeHistory,
    Order,
    Fill,
    Position,
    PortfolioLedger,
    ReconciliationReport,
)

__all__ = [
    "get_db",
    "get_engine",
    "init_db",
    "SessionLocal",
    "Base",
    "MarketBar",
    "MacroIndicator",
    "FeatureStore",
    "RegimeHistory",
    "Order",
    "Fill",
    "Position",
    "PortfolioLedger",
    "ReconciliationReport",
]
