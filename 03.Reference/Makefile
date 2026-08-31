PYTHON ?= python

.PHONY: help test clean dataset

help:
	@echo "Targets:"
	@echo "  test     Run the test suite (pytest)"
	@echo "  dataset  Download raw data (if missing) and build every processed file"
	@echo "  clean    Remove Python caches"

test:
	$(PYTHON) -m pytest tests -q

dataset:
	$(PYTHON) -m tools.bootstrap_dataset

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
