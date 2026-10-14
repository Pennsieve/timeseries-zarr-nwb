.PHONY: help venv install test test-cov lint typecheck check pre-commit

VENV_DIR      ?= venv
PYTHON        ?= python3
LIBRARY       ?= ../timeseries-zarr-py

.DEFAULT: help

help:
	@echo "Make Help for timeseries-zarr-nwb"
	@echo ""
	@echo "make venv       - create virtual environment and install all dependencies"
	@echo "make install    - install dependencies into existing venv"
	@echo "make pre-commit - install pre-commit hooks"
	@echo "make test       - run tests"
	@echo "make test-cov   - run tests with coverage report"
	@echo "make lint       - run linter + formatter with auto-fix"
	@echo "make typecheck  - run mypy --strict"
	@echo "make check      - run lint check, typecheck, and tests"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install --config-settings editable_mode=strict -e $(LIBRARY) -e ".[test,dev]"
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV_DIR)/bin/activate"

install:
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install --config-settings editable_mode=strict -e $(LIBRARY) -e ".[test,dev]"

test:
	$(VENV_DIR)/bin/python -m pytest tests/ -v

test-cov:
	$(VENV_DIR)/bin/python -m pytest tests/ -v --cov=timeseries_zarr_nwb --cov-report=term-missing

lint:
	$(VENV_DIR)/bin/ruff check --fix timeseries_zarr_nwb/ tests/
	$(VENV_DIR)/bin/ruff format timeseries_zarr_nwb/ tests/

typecheck:
	$(VENV_DIR)/bin/mypy timeseries_zarr_nwb/

check:
	$(VENV_DIR)/bin/ruff check timeseries_zarr_nwb/ tests/
	$(VENV_DIR)/bin/ruff format --check timeseries_zarr_nwb/ tests/
	$(VENV_DIR)/bin/mypy timeseries_zarr_nwb/
	$(VENV_DIR)/bin/python -m pytest tests/

pre-commit:
	$(VENV_DIR)/bin/pre-commit install
