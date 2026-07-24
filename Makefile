.PHONY: install-cuda clean \
        v014-preflight v014-matrix-quick v014-matrix-paper \
        v014-attention-quick v014-attention-paper \
        v014-ppl-quick v014-ppl-paper v014-nsight v014-nsight-exhaustive v014-summary v014-collect

install-cuda:
	RNS_LLM_BUILD_CUDA=1 python -m pip install . \
	  --no-build-isolation --no-deps --force-reinstall

v014-preflight:
	python scripts/preflight_v014.py

v014-matrix-quick:
	python benchmarks/benchmark_unified_matrix_v014.py \
	  --shape 16x768x768 --shape 128x768x768 \
	  --full-bits 8 16 --hybrid-bits 8 16 \
	  --lut-policies none two all --protected 3 \
	  --warmup 4 --iterations 10 --repeats 1 --concurrency 4 \
	  --output-dir results/v0.14.2/matrix

v014-matrix-paper:
	python benchmarks/benchmark_unified_matrix_v014.py \
	  --shape 16x2560x2560 --shape 128x2560x2560 \
	  --shape 16x2560x10240 --shape 128x2560x10240 \
	  --full-bits 8 16 32 --hybrid-bits 8 16 32 \
	  --lut-policies none one two all --protected 3 \
	  --warmup 8 --iterations 30 --repeats 3 --concurrency 4 \
	  --output-dir results/v0.14.2/matrix

v014-attention-quick:
	python benchmarks/benchmark_attention_v014.py \
	  --model facebook/opt-125m --seq 32 --protected 3 \
	  --full-bits 8 16 --hybrid-bits 8 16 \
	  --lut-policies none two all \
	  --warmup 3 --iterations 8 --concurrency 4 \
	  --output-dir results/v0.14.2/attention

v014-attention-paper:
	python benchmarks/benchmark_attention_v014.py \
	  --model facebook/opt-2.7b --seq 128 --protected 3 \
	  --full-bits 8 16 32 --hybrid-bits 8 16 32 \
	  --lut-policies none one two all \
	  --warmup 8 --iterations 30 --concurrency 4 \
	  --output-dir results/v0.14.2/attention

v014-ppl-quick:
	python scripts/evaluate_ppl_unified_v014.py \
	  --model facebook/opt-125m --attention-blocks 1 \
	  --max-eval-tokens 4096 --calibration-tokens 512 \
	  --variants native_int8 full_rns_int8_v07 full_rns_int8 hybrid_fp16 hybrid_rns_q16 \
	  --lut-policies none two \
	  --output-dir results/v0.14.2/ppl

v014-ppl-paper:
	python scripts/evaluate_ppl_unified_v014.py \
	  --model facebook/opt-2.7b --attention-blocks 4 \
	  --max-eval-tokens 32768 --calibration-tokens 2048 \
	  --variants native_int8 full_rns_int8_v07 full_rns_int8 full_rns_int16 full_rns_int32 \
	             hybrid_fp16 hybrid_rns_q8 hybrid_rns_q16 hybrid_rns_q32 \
	  --lut-policies two \
	  --output-dir results/v0.14.2/ppl

v014-nsight:
	python scripts/run_minimal_nsight_v014.py \
	  --output-root reports/v0.14.2 --total-minutes 55 --ncu-mode essential

v014-nsight-exhaustive:
	NCU_MODE=full bash scripts/run_all_nsight_v014.sh reports/v0.14.2

v014-summary:
	python scripts/summarize_v014.py \
	  --root results/v0.14.2 --reports-root reports/v0.14.2 --output-dir results/v0.14.2/summary

v014-collect:
	bash scripts/collect_v014_bundle.sh . rns_llm_v0142_results.zip

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -name '*.so' -delete
