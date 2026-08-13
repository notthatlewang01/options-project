PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.venv:
	python3 -m venv .venv

.PHONY: install
install: .venv ## Create the venv and install the package with all extras
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[analysis,dev]"

.PHONY: test
test: ## Run the full offline suite (deterministic, no network)
	$(PYTEST) -m "not live"

.PHONY: test-live
test-live: ## Run the single opt-in test that hits the real CBOE endpoint
	$(PYTEST) -m live -v

.PHONY: cov
cov: ## Offline suite with a coverage report
	$(PYTEST) -m "not live" --cov=spxrnd --cov-report=term-missing

.PHONY: lint
lint: ## Lint and format-check
	$(RUFF) check src tests
	$(RUFF) format --check src tests

.PHONY: fmt
fmt: ## Auto-format
	$(RUFF) check --fix src tests
	$(RUFF) format src tests

.PHONY: verify-archive
verify-archive: ## Check every raw capture against the checksum manifest
	@cd data/raw && shasum -a 256 -c ../../manifests/raw_archive.sha256 \
		| grep -v ': OK$$' || echo "all captures verified"

.PHONY: fixtures
fixtures: ## Regenerate the test fixture corpus from data/raw
	$(PY) tools/build_fixtures.py

.PHONY: collect
collect: ## Take exactly one snapshot into data/raw
	$(PY) -m spxrnd.cli.main collect --dir data

.PHONY: backfill
backfill: ## Rebuild the curated layer from the immutable raw archive
	$(PY) -m spxrnd.cli.main backfill --dir data

.PHONY: clean
clean: ## Remove build and test caches (never touches data/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov src/*.egg-info
	find src tests -name __pycache__ -type d -exec rm -rf {} +
