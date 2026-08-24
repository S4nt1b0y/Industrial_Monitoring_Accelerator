PYTHON ?= python

.PHONY: help test clean

help:
	@echo "Alvos disponiveis:"
	@echo "  test   Roda a suite de testes (pytest)"
	@echo "  clean  Remove caches Python"

test:
	$(PYTHON) -m pytest tests -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
