from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from rns_llm.prefill_v011 import PrefillLayerV011


@contextmanager
def nvtx(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def repeat_rows(samples: torch.Tensor, m: int, device: torch.device) -> torch.Tensor:
    repeats = (m + int(samples.size(0)) - 1) // int(samples.size(0))
    return samples.repeat((repeats, 1))[:m].to(device=device, dtype=torch.float32).contiguous()


def first_pack(audit_path: Path, layer_index: int) -> Path:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    paths = sorted(
        [
            Path(item["pack_file"])
            for item in audit["layer_decisions"]
            if item.get("passed") and item.get("pack_file")
        ],
        key=lambda path: path.name,
    )
    if layer_index < 0 or layer_index >= len(paths):
        raise IndexError(f"layer_index={layer_index}, available packs={len(paths)}")
    return paths[layer_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--method", choices=[
        "fp16", "native_int8", "preprocess", "main_int8", "rns_correction",
        "fp16_correction", "rns_fused_epilogue", "hybrid_rns_serial",
        "hybrid_rns_parallel", "hybrid_fp16_serial", "hybrid_fp16_parallel",
    ], required=True)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--logical-bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--workspace-mib", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    device = torch.device("cuda")
    pack_path = first_pack(args.audit, args.layer_index)
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    layer = PrefillLayerV011.from_pack(pack, device=device, optimized_rns_bits=(8, 16))
    runner = layer.runner(
        args.m,
        logical_bits=args.logical_bits,
        workspace_bytes=args.workspace_mib * 1024 * 1024,
    )
    x = repeat_rows(pack["activation_samples"], args.m, device)

    # Prepared buffers and dependencies for component profiles.
    runner.cast_fp16(x)
    runner.preprocess_native(x)
    runner.preprocess_hybrid(x)
    runner.main_int8_only()
    runner.rns_correction_only()
    torch.cuda.synchronize()

    def call() -> torch.Tensor:
        if args.method == "fp16":
            with nvtx("V011_FP16_GEMM"):
                if not args.prepared:
                    runner.cast_fp16(x)
                return runner.fp16_core()
        if args.method == "native_int8":
            if not args.prepared:
                with nvtx("V011_NATIVE_PREPROCESS"):
                    runner.preprocess_native(x)
            with nvtx("V011_NATIVE_INT8_MAIN"):
                return runner.native_core()
        if args.method == "preprocess":
            with nvtx("V011_FUSED_PREPROCESS"):
                runner.preprocess_hybrid(x)
                return runner.main_q
        if args.method == "main_int8":
            with nvtx("V011_MAIN_INT8"):
                return runner.main_int8_only()
        if args.method == "rns_correction":
            with nvtx("V011_RNS_CORRECTION"):
                return runner.rns_correction_only()
        if args.method == "fp16_correction":
            with nvtx("V011_FP16_CORRECTION"):
                return runner.fp16_correction_only()
        if args.method == "rns_fused_epilogue":
            with nvtx("V011_RNS_FUSED_EPILOGUE"):
                return runner.rns_fused_epilogue_only()
        if args.method == "hybrid_rns_serial":
            if not args.prepared:
                with nvtx("V011_FUSED_PREPROCESS"):
                    runner.preprocess_hybrid(x)
            with nvtx("V011_MAIN_INT8"):
                runner.main_int8_only()
            with nvtx("V011_RNS_FUSED_EPILOGUE"):
                return runner.rns_fused_epilogue_only()
        if args.method == "hybrid_fp16_serial":
            if not args.prepared:
                with nvtx("V011_FUSED_PREPROCESS"):
                    runner.preprocess_hybrid(x)
            with nvtx("V011_MAIN_INT8"):
                runner.main_int8_only()
            with nvtx("V011_FP16_FUSED_EPILOGUE"):
                return runner.fp16_fused_epilogue_only()
        if args.method == "hybrid_rns_parallel":
            if not args.prepared:
                with nvtx("V011_FUSED_PREPROCESS"):
                    runner.preprocess_hybrid(x)
            with nvtx("V011_HYBRID_RNS_PARALLEL"):
                return runner.hybrid_rns_parallel_core()
        if args.method == "hybrid_fp16_parallel":
            if not args.prepared:
                with nvtx("V011_FUSED_PREPROCESS"):
                    runner.preprocess_hybrid(x)
            with nvtx("V011_HYBRID_FP16_PARALLEL"):
                return runner.hybrid_fp16_parallel_core()
        raise AssertionError(args.method)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with nvtx(
        f"V011_PROFILE/method={args.method}/prepared={int(args.prepared)}/"
        f"M={args.m}/K={layer.k}/N={layer.n}/P={layer.p}"
    ):
        for _ in range(args.repeats):
            call()
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    print({
        "pack": str(pack_path),
        "layer": layer.layer_name,
        "method": args.method,
        "prepared": args.prepared,
        "m": args.m,
        "k": layer.k,
        "n": layer.n,
        "p": layer.p,
        "repeats": args.repeats,
    })


if __name__ == "__main__":
    main()
