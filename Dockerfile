# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────
# India Market Regime Detector — multi-stage image
#
#   builder  : compiles/installs pinned dependencies into a virtualenv
#   test     : builder + dev deps + tests  -> `docker build --target test .`
#   runtime  : slim, non-root image that serves the FastAPI app (default)
# ─────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.13

# ── builder ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Compiler toolchain is only needed if a wheel is unavailable (e.g. hmmlearn).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt


# ── test (CI / `docker compose run --rm tests`) ──────────────────────
FROM builder AS test

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

WORKDIR /app
COPY . .
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-m", "pytest", "-q"]


# ── runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Application code, data, artifacts, and figures
COPY src/ ./src/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY figures/ ./figures/

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "regime_api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
