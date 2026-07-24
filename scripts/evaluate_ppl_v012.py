#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from rns_llm.ppl_v012 import (
    DEFAULT_TARGET_PATTERNS,
    SUPPORTED_VARIANTS,
    CalibrationCollector,
    apply_simulated_variant,
    build_calibration_blocks,
    build_calibration_plan,
    cleanup_model,
    environment_snapshot,
    evaluate_sliding_window_ppl,
    finalize_summary,
    save_json,
    select_target_linears,
    tokenize_dataset,
    verify_ideal_rns_equivalence,
    write_paper_artifacts,
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_model(model_id: str, device: torch.device):
    from transformers import AutoModelForCausalLM

    common = dict(
        low_cpu_mem_usage=True,
        device_map={"": device.index if device.type == "cuda" else "cpu"},
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            **common,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            **common,
        )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-model PPL evaluation for RNS-LLM v0.12.0"
    )
    parser.add_argument("--model", default="facebook/opt-2.7b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default=",".join(SUPPORTED_VARIANTS),
        help="Comma-separated: fp16,native_int8,hybrid_fp16,hybrid_rns_q16",
    )
    parser.add_argument(
        "--target-patterns",
        default=",".join(DEFAULT_TARGET_PATTERNS),
        help="Comma-separated linear-module suffixes",
    )
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calibration-split", default="validation")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--calibration-sequence-length", type=int, default=128)
    parser.add_argument("--max-sample-rows", type=int, default=64)
    parser.add_argument("--absolute-threshold", type=float, default=6.0)
    parser.add_argument("--max-protected-ratio", type=float, default=0.03)
    parser.add_argument("--min-error-reduction", type=float, default=0.20)
    parser.add_argument("--output-sample", type=int, default=128)
    parser.add_argument(
        "--fallback",
        choices=("best_effort", "fp16", "native_int8"),
        default="best_effort",
        help="Policy for layers that fail the local protected-channel gate",
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=8192,
        help="0 evaluates the full split; 8192 is a faster preview",
    )
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--seed", type=int, default=10010)
    parser.add_argument("--rns-check-layers", type=int, default=4)
    parser.add_argument("--gate-threshold-percent", type=float, default=5.0)
    args = parser.parse_args()

    variants = parse_csv(args.variants)
    unknown = sorted(set(variants) - set(SUPPORTED_VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    patterns = parse_csv(args.target_patterns)
    if not patterns:
        raise SystemExit("At least one target pattern is required")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for the paper PPL experiment")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.plan_file or args.output_dir / "ppl_calibration_plan_v012.json"
    summary_path = args.output_dir / "ppl_summary_v012.json"
    paper_dir = args.output_dir / "paper"

    tokenizer = load_tokenizer(args.model)
    calibration_tokens = tokenize_dataset(
        tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.calibration_split,
    )
    eval_tokens = tokenize_dataset(
        tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.eval_split,
    )

    collector = None
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("model") != args.model:
            raise SystemExit(
                f"Plan model {plan.get('model')!r} does not match requested {args.model!r}"
            )
        print(f"Using existing calibration plan: {plan_path}")
    else:
        model = load_model(args.model, device)
        targets = select_target_linears(model, patterns)
        if not targets:
            raise SystemExit(f"No target linear modules matched: {patterns}")
        print(f"Calibrating {len(targets)} linear modules")
        collector = CalibrationCollector(
            targets,
            threshold=args.absolute_threshold,
            max_sample_rows=args.max_sample_rows,
        )
        total_blocks = args.calibration_batches * args.calibration_batch_size
        blocks = build_calibration_blocks(
            calibration_tokens,
            block_count=total_blocks,
            sequence_length=args.calibration_sequence_length,
        )
        split_batch = max(1, args.calibration_batches // 2)
        with torch.inference_mode():
            for batch_index in range(args.calibration_batches):
                collector.phase = "fit" if batch_index < split_batch else "heldout"
                start = batch_index * args.calibration_batch_size
                input_ids = blocks[start : start + args.calibration_batch_size].to(device)
                attention_mask = torch.ones_like(input_ids, device=device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                print(
                    f"calibration batch {batch_index + 1}/{args.calibration_batches} "
                    f"phase={collector.phase}"
                )
        collector.close()
        plan = build_calibration_plan(
            model,
            collector,
            model_id=args.model,
            target_patterns=patterns,
            max_protected_ratio=args.max_protected_ratio,
            min_error_reduction=args.min_error_reduction,
            output_sample=args.output_sample,
            dataset_name=f"{args.dataset}/{args.dataset_config}:{args.calibration_split}",
            calibration_config={
                "batches": args.calibration_batches,
                "batch_size": args.calibration_batch_size,
                "sequence_length": args.calibration_sequence_length,
                "absolute_threshold": args.absolute_threshold,
                "max_sample_rows_per_phase": args.max_sample_rows,
                "seed": args.seed,
            },
        )
        rns_check = verify_ideal_rns_equivalence(
            model,
            collector,
            plan,
            max_layers=args.rns_check_layers,
        )
        plan["ideal_rns_equivalence_check"] = rns_check
        save_json(plan, plan_path)
        print(f"Saved calibration plan to {plan_path}")
        cleanup_model(model)
        collector = None

    summary = {
        "version": "0.12.0",
        "model": args.model,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "calibration_split": args.calibration_split,
        "eval_split": args.eval_split,
        "target_patterns": patterns,
        "variants_requested": variants,
        "fallback_policy": args.fallback,
        "environment": environment_snapshot(),
        "plan_file": str(plan_path),
        "evaluation_config": {
            "context_length": args.context_length,
            "stride": args.stride,
            "max_eval_tokens": args.max_eval_tokens,
            "seed": args.seed,
        },
        "results": {},
    }
    save_json(summary, summary_path)

    for variant_index, variant in enumerate(variants, start=1):
        print(f"\n=== Variant {variant_index}/{len(variants)}: {variant} ===")
        model = load_model(args.model, device)
        patch_info = apply_simulated_variant(
            model,
            plan,
            variant=variant,
            fallback=args.fallback,
        )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            result = evaluate_sliding_window_ppl(
                model,
                eval_tokens,
                device=device,
                context_length=args.context_length,
                stride=args.stride,
                max_eval_tokens=args.max_eval_tokens,
            )
            result["wall_seconds_including_setup_after_load"] = time.perf_counter() - started
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            result["patch_info"] = patch_info
            summary["results"][variant] = result
            print(
                f"{variant}: PPL={result['ppl']:.6f}, "
                f"peak={result['peak_allocated_bytes'] / 2**30:.2f} GiB"
            )
        except Exception as exc:
            summary["results"][variant] = {
                "error": repr(exc),
                "patch_info": patch_info,
                "wall_seconds_until_error": time.perf_counter() - started,
            }
            print(f"{variant}: ERROR {exc!r}")
        finalize_summary(summary, gate_threshold_percent=args.gate_threshold_percent)
        save_json(summary, summary_path)
        write_paper_artifacts(summary, paper_dir)
        cleanup_model(model)
        gc.collect()

    finalize_summary(summary, gate_threshold_percent=args.gate_threshold_percent)
    save_json(summary, summary_path)
    write_paper_artifacts(summary, paper_dir)
    print("\nFinal PPL requirement:")
    print(json.dumps(summary.get("ppl_requirement", {}), indent=2))
    print(f"Results: {summary_path}")
    print(f"Paper-ready snippets: {paper_dir}")


if __name__ == "__main__":
    main()
