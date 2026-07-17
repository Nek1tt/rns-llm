.PHONY: install install-cuda test test-cuda smoke benchmark clean

install:
	RNS_LLM_BUILD_CUDA=0 python -m pip install -e ".[dev]" --no-build-isolation

install-cuda:
	RNS_LLM_BUILD_CUDA=1 python -m pip install -e ".[dev]" --no-build-isolation

test:
	pytest -q

test-cuda:
	pytest -q -m cuda

smoke:
	python scripts/smoke_reference.py
	python scripts/smoke_cuda.py

benchmark:
	python benchmarks/benchmark_cuda.py --compare-separate

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -name '*.so' -delete
