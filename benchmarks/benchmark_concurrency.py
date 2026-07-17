from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from rns_llm.backends import CudaRNSBackend
from rns_llm.reference import choose_moduli_for_dot


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def measure(callable_, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        callable_()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        callable_()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def summarize(samples: list[float], concurrency: int) -> dict:
    p50 = median(samples)
    return {
        "batch_p50_ms": p50,
        "batch_p95_ms": percentile(samples, 0.95),
        "throughput_requests_per_second": concurrency / (p50 / 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--source-bits", type=int, choices=[8, 12], default=8)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    max_abs = (1 << (args.source_bits - 1)) - 1
    dtype = torch.int8 if args.source_bits == 8 else torch.int16
    moduli = choose_moduli_for_dot(
        args.k, max_abs, max_abs, strategy="dense_coprime"
    )
    backend = CudaRNSBackend()
    weight = torch.randint(
        -max_abs, max_abs + 1, (args.k, args.n), dtype=dtype, device=device
    )
    prepared = backend.prepare_weight(weight, moduli)

    results = {}
    for concurrency in args.concurrency:
        activations = [
            torch.randint(
                -max_abs,
                max_abs + 1,
                (args.m, args.k),
                dtype=dtype,
                device=device,
            )
            for _ in range(concurrency)
        ]
        stream_workspaces = [
            backend.create_workspace(
                device=device, channels=len(moduli), m=args.m, n=args.n
            )
            for _ in range(concurrency)
        ]
        streams = [torch.cuda.Stream() for _ in range(concurrency)]
        batch_workspace = backend.create_request_batch_workspace(
            device=device,
            dtype=dtype,
            rows_per_request=[args.m] * concurrency,
            k=args.k,
            channels=len(moduli),
            n=args.n,
        )

        def independent_streams():
            outputs = []
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    outputs.append(
                        backend.matmul_prepared_fused(
                            activations[i],
                            prepared,
                            lut_channels=2,
                            workspace=stream_workspaces[i],
                        )
                    )
            for stream in streams:
                stream.synchronize()
            return outputs

        def continuous_batch():
            outputs = backend.matmul_prepared_fused_requests(
                activations,
                prepared,
                lut_channels=2,
                workspace=batch_workspace,
            )
            torch.cuda.synchronize()
            return outputs

        # Full correctness check before timing.  The merged path must be exactly
        # equal to independent execution and to an int64 CPU oracle.
        independent = [x.clone() for x in independent_streams()]
        merged = [x.clone() for x in continuous_batch()]
        max_error = 0
        for activation, left, right in zip(activations, independent, merged):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            expected = (
                activation.cpu().numpy().astype(np.int64)
                @ weight.cpu().numpy().astype(np.int64)
            )
            error = int(np.max(np.abs(right.cpu().numpy() - expected)))
            max_error = max(max_error, error)
        if max_error != 0:
            raise AssertionError(f"continuous batching error: {max_error}")

        stream_samples = measure(independent_streams, args.warmup, args.iterations)
        batch_samples = measure(continuous_batch, args.warmup, args.iterations)
        stream_summary = summarize(stream_samples, concurrency)
        batch_summary = summarize(batch_samples, concurrency)
        results[str(concurrency)] = {
            "independent_streams": stream_summary,
            "continuous_batch": batch_summary,
            "continuous_batch_speedup": (
                stream_summary["batch_p50_ms"] / batch_summary["batch_p50_ms"]
            ),
            "correctness": {"passed": True, "max_absolute_error": max_error},
        }

    payload = {
        "shape_per_request": {"m": args.m, "k": args.k, "n": args.n},
        "source_bits": args.source_bits,
        "moduli": list(moduli),
        "queue_delay_included": False,
        "note": (
            "Continuous batching measures execution after requests are ready; "
            "a production service must add queueing-delay metrics."
        ),
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
