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
	@$(PY) -c "from pathlib import Path; from spxrnd.store import archive; \
r = archive.verify(Path('data/raw'), Path('manifests/raw_archive.sha256')); \
print(f'ok {len(r.ok)}  corrupted {len(r.corrupted)}  missing {len(r.missing)}  unlisted {len(r.unlisted)}'); \
[print('  CORRUPT', n) for n in r.corrupted]; [print('  MISSING', n) for n in r.missing]; \
raise SystemExit(0 if r.healthy else 1)"

.PHONY: write-manifest
write-manifest: ## Regenerate the archive checksum manifest
	@$(PY) -c "from pathlib import Path; from spxrnd.store import archive; \
print(archive.write_manifest(Path('data/raw'), Path('manifests/raw_archive.sha256')), 'captures')"

.PHONY: compress-archive
compress-archive: ## Gzip any uncompressed captures (verify-then-replace)
	@$(PY) -c "from pathlib import Path; from spxrnd.store import archive; \
rs = archive.compress_archive(Path('data/raw')); \
print(sum(not r.skipped for r in rs), 'compressed,', sum(r.skipped for r in rs), 'already done')"

.PHONY: fixtures
fixtures: ## Regenerate the test fixture corpus from data/raw
	$(PY) tools/build_fixtures.py

.PHONY: collect
collect: ## Take exactly one snapshot into data/raw
	@$(PY) -m spxrnd.cli.main --dir data collect

.PHONY: backfill
backfill: ## Rebuild the curated layer from the immutable raw archive
	@$(PY) -c "from pathlib import Path; from spxrnd.store import catalog; \
r = catalog.backfill(Path('data/raw'), Path('data/curated')); \
print(f'{len(r.written)} captures, {r.total_rows:,} rows, {r.total_bytes/1e6:.1f} MB, {len(r.failed)} failed'); \
d = catalog.divergence(Path('data/raw'), Path('data/curated')); \
print('divergence:', 'aligned' if d.aligned else d)"

.PHONY: agent
agent: ## Regenerate the launchd job from tools/make_launchd_plist.py
	$(PY) tools/make_launchd_plist.py

.PHONY: install-agent
install-agent: agent ## Install and load the launchd collector
	cp deploy/com.spxrnd.collect.plist ~/Library/LaunchAgents/
	launchctl unload ~/Library/LaunchAgents/com.spxrnd.collect.plist 2>/dev/null || true
	launchctl load ~/Library/LaunchAgents/com.spxrnd.collect.plist
	@echo "loaded. check with: launchctl list | grep spxrnd"

.PHONY: uninstall-agent
uninstall-agent: ## Unload and remove the launchd collector (data untouched)
	launchctl unload ~/Library/LaunchAgents/com.spxrnd.collect.plist 2>/dev/null || true
	rm -f ~/Library/LaunchAgents/com.spxrnd.collect.plist
	@echo "removed. data/ is untouched."

.PHONY: status
status: ## Is collection working?
	@$(PY) -m spxrnd.cli.main --dir data status

.PHONY: estimate
estimate: ## Densities and moments for the newest capture
	@$(PY) -m spxrnd.cli.main --dir data estimate --term-structure

.PHONY: clean
clean: ## Remove build and test caches (never touches data/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov src/*.egg-info
	find src tests -name __pycache__ -type d -exec rm -rf {} +
