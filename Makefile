.PHONY: setup test reference
setup:
	python -m pip install -e ".[dev]"
test:
	pytest
reference:
	python scripts/run_reference_check.py
