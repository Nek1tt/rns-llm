#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-facebook/opt-2.7b}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ppl_v012}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-32768}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
STRIDE="${STRIDE:-1024}"
TARGET_PATTERNS="${TARGET_PATTERNS:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2}"

python scripts/evaluate_ppl_v012.py \
  --model "$MODEL_ID" \
  --output-dir "$OUTPUT_DIR" \
  --variants fp16,native_int8,hybrid_fp16,hybrid_rns_q16 \
  --target-patterns "$TARGET_PATTERNS" \
  --fallback best_effort \
  --calibration-batches 8 \
  --calibration-batch-size 1 \
  --calibration-sequence-length 128 \
  --max-sample-rows 64 \
  --absolute-threshold 6.0 \
  --max-protected-ratio 0.03 \
  --min-error-reduction 0.20 \
  --context-length "$CONTEXT_LENGTH" \
  --stride "$STRIDE" \
  --max-eval-tokens "$MAX_EVAL_TOKENS" \
  --gate-threshold-percent 5.0
