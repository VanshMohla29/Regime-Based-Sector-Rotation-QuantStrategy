
.PHONY: install test test-fast lint cov docker-build docker-test up down

install:
	pip install -r requirements-dev.txt

test:            ## full suite
	python -m pytest

test-fast:       ## skip HMM training / walk-forward tests
	python -m pytest -m "not slow"

lint:
	ruff check .

cov:
	python -m pytest --cov=src --cov-report=term-missing

docker-build:
	docker build --target runtime -t regime-detector:latest .

docker-test:     ## run the suite inside a clean container
	docker compose --profile test run --rm --build tests

up:
	docker compose up -d --build api cron

down:
	docker compose down

eod-now:         ## trigger immediate EOD pipeline run
	docker compose exec -T api python src/scripts/eod_pipeline.py

cron-logs:       ## stream EOD scheduler logs
	docker compose logs -f cron
