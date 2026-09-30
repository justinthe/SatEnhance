VENV ?= .venv
PY   ?= $(VENV)/bin/python

.PHONY: venv lint test smoke build prefetch-models

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e common -e acquire -e enhance pytest pytest-cov responses "moto[server]" ruff

lint:
	$(VENV)/bin/ruff check .

test:
	$(PY) -m pytest common acquire/tests enhance/tests tests/shell -q

smoke:
	bash tests/smoke/test_pipeline_smoke.sh

build:
	./scripts/build.sh --cpu-only

prefetch-models:
	./scripts/run_system2.sh prefetch
