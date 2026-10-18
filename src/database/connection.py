import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_DB_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(_SRC_DIR), 'data')) else _SRC_DIR
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
os.makedirs(_DATA_DIR, exist_ok=True)
DEFAULT_SQLITE_PATH = os.path.join(_DATA_DIR, "regime_database.db")
DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH}"

def get_database_url() -> str:
    """
    Returns DATABASE_URL from environment or defaults to SQLite.
    Normalizes 'postgres://' to 'postgresql://' for SQLAlchemy 2.x.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return DEFAULT_SQLITE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url

def create_configured_engine(url: str = None):
    target_url = url or get_database_url()
    is_sqlite = target_url.startswith("sqlite")
    
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    
    try:
        engine = create_engine(
            target_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            echo=False
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        # If postgres failed to connect, fall back to SQLite
        if not is_sqlite:
            print(f"  [warn] Database connection to {target_url} failed ({e}). Falling back to SQLite.")
            return create_engine(
                DEFAULT_SQLITE_URL,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True,
                echo=False
            )
        raise e

# Global engine and session factory
_engine = None
_SessionLocal = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_configured_engine()
        from . import models  # noqa: F401
        Base.metadata.create_all(bind=_engine)
    return _engine

def SessionLocal():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal()

@contextmanager
def get_db():
    """Context manager for transactional database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db(engine=None):
    """
    Initializes all database tables.
    If PostgreSQL + TimescaleDB is connected, executes hypertable conversions.
    """
    eng = engine or get_engine()
    from . import models  # noqa: F401
    
    Base.metadata.create_all(bind=eng)
    
    # Check if we are running on PostgreSQL to apply TimescaleDB features
    if eng.dialect.name == "postgresql":
        with eng.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                hypertables = [
                    ("market_bars", "time"),
                    ("sector_returns", "time"),
                    ("macro_indicators", "time"),
                    ("feature_store", "time"),
                    ("regime_history", "time"),
                    ("orders", "time"),
                    ("fills", "time"),
                    ("positions", "time"),
                    ("portfolio_ledger", "time"),
                ]
                for tbl, col in hypertables:
                    try:
                        conn.execute(text(f"SELECT create_hypertable('{tbl}', '{col}', if_not_exists => TRUE);"))
                    except Exception:
                        pass
                conn.commit()
            except Exception as e:
                print(f"  [info] TimescaleDB extension check: {e}")
