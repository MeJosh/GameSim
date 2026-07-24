# GameSim developer tasks.
#
# Quick start:
#   make install     # create .venv and install the package + dev tools
#   make test        # run the test suite
#   make check       # lint + type-check + test (run this before committing)
#
# Run `make` or `make help` to list all targets.
#
# Note: this is a POSIX Makefile (works on macOS, Linux, WSL, or Git Bash on
# Windows). On native Windows without `make`, see the README for the equivalent
# raw commands.

# venv layout differs by OS: Scripts/ on Windows, bin/ elsewhere.
ifeq ($(OS),Windows_NT)
    # Windows rarely provides a real `python3` executable; the Python launcher is
    # the most reliable default for Git Bash/MSYS make. Override if needed:
    #   make install PYTHON=python
    PYTHON ?= py -3
    VENV_BIN := .venv/Scripts
else
    # Which interpreter to bootstrap the venv from. Override if needed:
    #   make install PYTHON=python3.12
    PYTHON ?= python3
    VENV_BIN := .venv/bin
endif

VENV_PY := $(VENV_BIN)/python
PIP     := $(VENV_PY) -m pip

.DEFAULT_GOAL := help

## ---------------------------------------------------------------------------
## Environment
## ---------------------------------------------------------------------------

# The venv is created once; this file-target means `make` won't rebuild it every
# time. Delete .venv (or run `make clean-venv`) to recreate from scratch.
$(VENV_PY):
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip

.PHONY: venv
venv: $(VENV_PY) ## Create the virtual environment (.venv) if missing

.PHONY: install
install: $(VENV_PY) ## Create venv and install the package + dev tools
	$(PIP) install -e ".[dev]"

.PHONY: install-rl
install-rl: $(VENV_PY) ## Also install the DRL extras (Phase 2: torch, sb3, etc.)
	$(PIP) install -e ".[dev,rl]"

## ---------------------------------------------------------------------------
## Quality gates
## ---------------------------------------------------------------------------

.PHONY: test
test: ## Run the test suite
	$(VENV_PY) -m pytest

.PHONY: test-cov
test-cov: ## Run tests with a coverage report
	$(VENV_PY) -m pytest --cov=gamesim --cov-report=term-missing

.PHONY: test-slow
test-slow: ## Run the slow/opt-in tests (e.g. the training smoke test; needs install-rl)
	$(VENV_PY) -m pytest -m slow

.PHONY: lint
lint: ## Check style with ruff
	$(VENV_PY) -m ruff check src tests

.PHONY: format
format: ## Auto-format with ruff
	$(VENV_PY) -m ruff format src tests
	$(VENV_PY) -m ruff check --fix src tests

.PHONY: typecheck
typecheck: ## Static type-check with mypy (strict)
	$(VENV_PY) -m mypy

.PHONY: check
check: lint typecheck test ## Run all quality gates (do this before committing)

## ---------------------------------------------------------------------------
## Training (Phase 2b -- needs `make install-rl`; not run in the dev sandbox)
## ---------------------------------------------------------------------------

# Override on the command line, e.g.: make train TIMESTEPS=200000 SEED=1
TIMESTEPS ?= 100000
SEED ?= 0
GAMES ?= 100
OPPONENT ?= all
CHECKPOINT ?= checkpoints/connect_four_maskable_ppo.zip

.PHONY: train
train: ## Train a Connect Four MaskablePPO agent via self-play (checkpoints/)
	$(VENV_PY) -m gamesim.rl.train --timesteps $(TIMESTEPS) --seed $(SEED)

.PHONY: evaluate
evaluate: ## Evaluate a trained checkpoint vs random/minimax baselines
	$(VENV_PY) -m gamesim.rl.evaluate --checkpoint $(CHECKPOINT) --opponent $(OPPONENT) --games $(GAMES) --seed $(SEED)

## ---------------------------------------------------------------------------
## Housekeeping
## ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove caches and build artifacts (keeps the venv)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-venv
clean-venv: ## Delete the virtual environment
	rm -rf .venv

.PHONY: help
help: ## Show this help
	@echo "GameSim make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
