# GameSim developer tasks.
#
# Quick start:
#   make install     # create .venv and install the package + dev tools
#   make test        # run the test suite
#   make check       # lint + format check + type-check + test
#
# Run `make` or `make help` to list all targets.
#
# Note: this is a POSIX Makefile (works on macOS, Linux, WSL, or Git Bash on
# Windows). On native Windows without `make`, see the README for the equivalent
# raw commands.
#
# Extra CLI flags (e.g. --help) can't be tacked onto the end of a `make` command --
# `make incremental-training --help` never reaches the script: `--help`/`-h` is one
# of make's *own* options (see `make --help`), intercepted before target parsing
# even starts, and there's no Makefile-level way around that. Pass extra flags
# through the ARGS variable instead:
#   make incremental-training ARGS="--help"
#   make train ARGS="--help"
#   make test ARGS="-k test_incremental -v"

# venv layout differs by OS: Scripts/ on Windows, bin/ elsewhere.
ifeq ($(OS),Windows_NT)
    # Windows rarely provides a real `python3` executable; the Python launcher is
    # the most reliable default for Git Bash/MSYS make. Override if needed:
    #   make install PYTHON=python
    PYTHON ?= py -3
    VENV_BIN := .venv/Scripts
    # The venv's interpreter is python.exe on Windows; without the extension this
    # target's file-existence check never matches the real file, so `make install*`
    # would try to recreate an already-installed venv on every run (and fail with a
    # permission error once something -- an open shell, an editor, antivirus -- has
    # it open).
    EXE_SUFFIX := .exe
else
    # Which interpreter to bootstrap the venv from. Override if needed:
    #   make install PYTHON=python3.12
    PYTHON ?= python3
    VENV_BIN := .venv/bin
    EXE_SUFFIX :=
endif

VENV_PY := $(VENV_BIN)/python$(EXE_SUFFIX)
PIP     := $(VENV_PY) -m pip

# Extra flags appended to CLI-backed targets below, e.g. `make train ARGS="--help"`.
ARGS ?=

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
	$(VENV_PY) -m pre_commit install

.PHONY: install-rl
install-rl: $(VENV_PY) ## Also install the DRL extras (Phase 2: torch, sb3, etc.)
	$(PIP) install -e ".[dev,rl]"
	$(VENV_PY) -m pre_commit install

.PHONY: install-web
install-web: $(VENV_PY) ## Also install the local browser play UI
	$(PIP) install -e ".[dev,web]"
	$(VENV_PY) -m pre_commit install

## ---------------------------------------------------------------------------
## Quality gates
## ---------------------------------------------------------------------------

.PHONY: test
test: ## Run the test suite (extra pytest flags: ARGS="-k foo -v")
	$(VENV_PY) -m pytest $(ARGS)

.PHONY: test-cov
test-cov: ## Run tests with a coverage report
	$(VENV_PY) -m pytest --cov=gamesim --cov-report=term-missing $(ARGS)

.PHONY: test-slow
test-slow: ## Run the slow/opt-in tests (e.g. the training smoke test; needs install-rl)
	$(VENV_PY) -m pytest -m slow $(ARGS)

.PHONY: lint
lint: ## Check style with ruff
	$(VENV_PY) -m ruff check src tests $(ARGS)

.PHONY: format
format: ## Auto-format with ruff
	$(VENV_PY) -m ruff format src tests $(ARGS)
	$(VENV_PY) -m ruff check --fix src tests $(ARGS)

.PHONY: format-check
format-check: ## Verify formatting with ruff
	$(VENV_PY) -m ruff format --check src tests $(ARGS)

.PHONY: typecheck
typecheck: ## Static type-check with mypy (strict)
	$(VENV_PY) -m mypy $(ARGS)

.PHONY: check
check: lint format-check typecheck test ## Run all quality gates

.PHONY: hooks
hooks: ## Run every pre-commit hook across the repository
	$(VENV_PY) -m pre_commit run --all-files

## ---------------------------------------------------------------------------
## Training (Phase 2b -- needs `make install-rl`; not run in the dev sandbox)
## ---------------------------------------------------------------------------

# Override on the command line, e.g.: make train TIMESTEPS=200000 SEED=1
TIMESTEPS ?= 100000
SEED ?= 0
GAMES ?= 100
OPPONENT ?= all
CHECKPOINT ?= checkpoints/connect_four_maskable_ppo.zip
MATCH_LOG ?= logs/connect_four_trained_vs_random.zip
AGENT_A ?= trained:$(CHECKPOINT)
AGENT_B ?= random

.PHONY: train
train: ## Train a Connect Four MaskablePPO agent via self-play (checkpoints/; see options: ARGS="--help")
	$(VENV_PY) -m gamesim.rl.train --timesteps $(TIMESTEPS) --seed $(SEED) $(ARGS)

.PHONY: evaluate
evaluate: ## Evaluate a trained checkpoint vs random/minimax baselines (see options: ARGS="--help")
	$(VENV_PY) -m gamesim.rl.evaluate --checkpoint $(CHECKPOINT) --opponent $(OPPONENT) --games $(GAMES) --seed $(SEED) $(ARGS)

.PHONY: record-matches
record-matches: ## Record trained-vs-random games for replay in the browser UI (see options: ARGS="--help")
	$(VENV_PY) -m gamesim.rl.record_matches --agent-a $(AGENT_A) --agent-b $(AGENT_B) --games $(GAMES) --seed $(SEED) --output $(MATCH_LOG) $(ARGS)

.PHONY: serve
serve: ## Run the local Connect Four browser UI (needs install-web)
	$(VENV_PY) -m gamesim.web $(ARGS)

REPORT_LOG ?= logs/connect_four_trained_vs_random.zip
REPORT_OUT ?= reports/connect_four_match.html

.PHONY: report
report: ## Write a standalone, self-contained HTML match report from a recorded log (see options: ARGS="--help")
	$(VENV_PY) -m gamesim.viz.report --log $(REPORT_LOG) --output $(REPORT_OUT) $(ARGS)

EUCHRE_NUM_HANDS ?= 50
EUCHRE_MATCH_LOG ?= logs/euchre_demo_match.zip
EUCHRE_REPORT_OUT ?= reports/euchre_match.html

.PHONY: record-euchre-demo
record-euchre-demo: ## Record a demo Euchre match log (RandomAgent vs RandomAgent, no RL extras needed; see options: ARGS="--help")
	$(VENV_PY) scripts/record_euchre_demo_match.py --num-hands $(EUCHRE_NUM_HANDS) --seed $(SEED) --output $(EUCHRE_MATCH_LOG) $(ARGS)

.PHONY: report-euchre
report-euchre: ## Write a standalone, self-contained HTML Euchre match report from a recorded log (see options: ARGS="--help")
	$(VENV_PY) -m gamesim.viz.report_euchre --log $(EUCHRE_MATCH_LOG) --output $(EUCHRE_REPORT_OUT) $(ARGS)

RUN_DIR ?= runs/incremental-smoke-001

.PHONY: incremental-smoke
incremental-smoke: ## Run the bounded incremental-training smoke experiment (needs install-rl; see options: ARGS="--help")
	$(VENV_PY) scripts/run_incremental_smoke.py --run-dir $(RUN_DIR) $(ARGS)

NUM_STAGES ?= 6

.PHONY: incremental-training
incremental-training: ## Run a staged incremental training experiment (needs install-rl; override RUN_DIR/NUM_STAGES; see options: ARGS="--help")
	$(VENV_PY) scripts/run_incremental_training.py --run-dir $(RUN_DIR) --num-stages $(NUM_STAGES) $(ARGS)

PROGRESS_JSON ?= $(RUN_DIR)/progress.json
PROGRESS_REPORT_OUT ?= reports/incremental_progress.html

.PHONY: progress-report
progress-report: ## Write a standalone HTML training-progress report from a run's progress.json (torch-free; see options: ARGS="--help")
	$(VENV_PY) -m gamesim.viz.progress_report --progress $(PROGRESS_JSON) --output $(PROGRESS_REPORT_OUT) $(ARGS)

## ---------------------------------------------------------------------------
## Housekeeping
## ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove caches and build artifacts (keeps the venv)
	@$(PYTHON) -c "import contextlib, glob, os, shutil; targets = ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'build', 'dist'] + glob.glob('*.egg-info'); [shutil.rmtree(t, ignore_errors=True) for t in targets]; exec(\"with contextlib.suppress(OSError): os.remove('.coverage')\")"
	@$(PYTHON) -c "import itertools, pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in itertools.chain(pathlib.Path('src').rglob('__pycache__'), pathlib.Path('tests').rglob('__pycache__'))]"

.PHONY: clean-venv
clean-venv: ## Delete the virtual environment
	@$(PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"

# The self-documenting help below reads this very file's "target: ## description"
# comments -- previously via `grep`/`awk`, which broke `make help` when invoked from
# a plain Windows cmd.exe prompt (no Unix toolchain on PATH; only Git Bash ships
# one). $(PYTHON) is the one interpreter this Makefile already assumes exists
# everywhere (it bootstraps the venv itself), so every recipe here that used to shell
# out to Unix-only tools (grep/awk/rm/find) now goes through a $(PYTHON) -c one-liner
# instead -- works identically under cmd.exe, PowerShell, and any POSIX shell. Each
# recipe is kept to one physical line on purpose: Make's `\`-continuation relies on
# the shell to glue the pieces back together, and cmd.exe doesn't honor `\` for that
# (it uses `^`) -- a multi-line recipe here would silently break again under cmd.exe.
.PHONY: help
help: ## Show this help
	@echo GameSim make targets:
	@$(PYTHON) -c "import re; [print('  {:<22} {}'.format(*m.groups())) for m in (re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line) for line in open('Makefile')) if m]"
	@$(PYTHON) -c "print(); print('See full CLI options via ARGS, e.g.:'); print('  make incremental-training ARGS=\"--help\"')"
