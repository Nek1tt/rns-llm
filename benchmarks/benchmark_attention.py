from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch

from rns_llm.backends import CudaRNSBackend
from rns_llm.reference import choose_moduli_for_dot


def time_ms(fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return {
        "p50_ms": median(samples),
        "p95_ms": samples[min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1)],
    }


def run_shape(
    backend: CudaRNSBackend,
    *,
    name: str,
    m: int,
    k: int,
    n: int,
    source_bits: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    max_abs = (1 << (source_bits - 1)) - 1
    dtype = torch.int8 if source_bits == 8 else torch.int16
    moduli = choose_moduli_for_dot(k, max_abs, max_abs, strategy="dense_coprime")
    device = torch.device("cuda")
    a = torch.randint(-max_abs, max_abs + 1, (m, k), dtype=dtype, device=device)
    b = torch.randint(-max_abs, max_abs + 1, (k, n), dtype=dtype, device=device)
    prepared = backend.prepare_weight(b, moduli)
    if prepared.kernel != "cublas":
        return {"name": name, "shape": [m, k, n], "skipped": "not cuBLAS-compatible"}
    workspace = backend.create_workspace(
        device=device, channels=len(moduli), m=m, n=n
    )
    rns = time_ms(
        lambda: backend.matmul_prepared_fused(
            a, prepared, lut_channels=2, workspace=workspace
        ),
        warmup,
        iterations,
    )
    fp16 = time_ms(
        lambda: torch.mm(a.to(torch.float16), b.to(torch.float16)),
        warmup,
        iterations,
    )
    fp32 = time_ms(
        lambda: torch.mm(a.to(torch.float32), b.to(torch.float32)),
        warmup,
        iterations,
    )
    return {
        "name": name,
        "shape": {"m": m, "k": k, "n": n},
        "moduli": list(moduli),
        "channels": len(moduli),
        "rns_fused": rns,
        "torch_fp16": fp16,
        "torch_fp32": fp32,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 256])
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--source-bits", type=int, choices=[8, 12], default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    backend = CudaRNSBackend()
    head_dim = args.hidden // args.heads
    results = []
    for tokens in args.tokens:
        rows = args.batch * tokens
        shapes = [
            (f"q_projection_tokens_{tokens}", rows, args.hidden, args.hidden),
            (f"qkv_fused_projection_tokens_{tokens}", rows, args.hidden, 3 * args.hidden),
            (f"output_projection_tokens_{tokens}", rows, args.hidden, args.hidden),
            (f"mlp_up_tokens_{tokens}", rows, args.hidden, 4 * args.hidden),
            (f"mlp_down_tokens_{tokens}", rows, 4 * args.hidden, args.hidden),
        ]
        # One attention head.  A complete attention implementation should batch
        # batch*heads such GEMMs; this shape isolates the QK^T / AV arithmetic.
        if tokens % 4 == 0:
            shapes.extend([
                (f"qk_single_head_tokens_{tokens}", tokens, head_dim, tokens),
                (f"av_single_head_tokens_{tokens}", tokens, tokens, head_dim),
            ])
        for name, m, k, n in shapes:
            results.append(run_shape(
                backend,
                name=name,
                m=m,
                k=k,
                n=n,
                source_bits=args.source_bits,
                warmup=args.warmup,
                iterations=args.iterations,
            ))

    payload = {
        "model_geometry": {
            "batch": args.batch,
            "hidden": args.hidden,
            "heads": args.heads,
            "head_dim": head_dim,
        },
        "source_bits": args.source_bits,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
