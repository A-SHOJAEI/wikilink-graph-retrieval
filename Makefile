PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /bin/bash
.DEFAULT_GOAL := all

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Config used by pipeline targets. Override: `make all CONFIG=configs/full.yaml`
CONFIG ?= configs/smoke.yaml

export PYTHONPATH := src

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@bash scripts/bootstrap_venv.sh
	@$(PIP) --version >/dev/null

data: setup
	@$(PY) -m wikilink_graph_retrieval.pipeline.data --config $(CONFIG)

train: setup
	@$(PY) -m wikilink_graph_retrieval.pipeline.train --config $(CONFIG)

eval: setup
	@$(PY) -m wikilink_graph_retrieval.pipeline.eval --config $(CONFIG)

report: setup
	@$(PY) -m wikilink_graph_retrieval.pipeline.report --config $(CONFIG)

all: data train eval report

clean:
	@rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache artifacts/* runs/* data/processed/* data/index/*
