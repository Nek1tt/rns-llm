from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from rns_llm.architecture_v013 import (
    RNSArchitectureRunner,
    build_compact_lut,
    prepare_rns_weight,
    quant_max,
    select_plan,
)


def rel_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(output.float() - reference.float())
            / torch.clamp(torch.linalg.vector_norm(reference.float()), min=1e-30)
        ).item()
    )


def quantize_scalar(value: float, scale: float, qmax: int) -> int:
    # Python round and CUDA nearbyint both use round-to-nearest-even.
    quantized = round(value / scale)
    return max(-qmax, min(qmax, int(quantized)))


def exact_sample_check(
    *,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    output: torch.Tensor,
    bits: int,
    samples: int = 8,
) -> dict[str, object]:
    rng = random.Random(130013 + bits)
    m, k = map(int, a_cpu.shape)
    n = int(b_cpu.shape[1])
    qmax = quant_max(bits)
    a_scales = activation_scales.detach().cpu()
    b_scales = weight_scales.detach().cpu()
    output_cpu = output.detach().cpu()
    candidates = {(0, 0), (m - 1, n - 1)}
    while len(candidates) < samples:
        candidates.add((rng.randrange(m), rng.randrange(n)))

    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    records: list[dict[str, object]] = []
    for row, col in sorted(candidates):
        sa = float(a_scales[row].item())
        sb = float(b_scales[col].item())
        integer_dot = 0
        for idx in range(k):
            qa = quantize_scalar(float(a_cpu[row, idx].item()), sa, qmax)
            qb = quantize_scalar(float(b_cpu[idx, col].item()), sb, qmax)
            integer_dot += qa * qb
        expected = float(integer_dot * sa * sb)
        actual = float(output_cpu[row, col].item())
        absolute_error = abs(actual - expected)
        relative_error = absolute_error / max(abs(expected), 1e-30)
        # The reconstruction is exact; only conversion of the exact integer to
        # double and then FP32 storage may round the dequantized scalar.
        tolerance = 2.5e-5 * max(1.0, abs(expected))
        if not math.isfinite(actual) or absolute_error > tolerance:
            raise RuntimeError(
                f"q{bits} exact sample mismatch at ({row},{col}): "
                f"actual={actual}, expected={expected}, error={absolute_error}, "
                f"tolerance={tolerance}"
            )
        maximum_absolute_error = max(maximum_absolute_error, absolute_error)
        maximum_relative_error = max(maximum_relative_error, relative_error)
        records.append(
            {
                "row": row,
                "col": col,
                "exact_integer_dot": str(integer_dot),
                "expected_dequantized": expected,
                "gpu_output": actual,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "tolerance": tolerance,
            }
        )
    return {
        "samples": records,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    torch.manual_seed(130013)
    device = torch.device("cuda")
    m, k, n = 3, 256, 128
    a = torch.randn((m, k), device=device, dtype=torch.float32)
    b = torch.randn((k, n), device=device, dtype=torch.float32) * 0.025
    # Include deterministic high-magnitude and zero rows/columns in addition
    # to random values so scale and signed reconstruction corner cases run.
    a[0].copy_(torch.linspace(-3.0, 3.0, k, device=device))
    a[-1].zero_()
    b[:, 0].copy_(torch.linspace(0.25, -0.25, k, device=device))
    b[:, -1].zero_()
    reference = a @ b
    a_cpu = a.detach().cpu()
    b_cpu = b.detach().cpu()
    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "shape": [m, k, n],
        "formats": {},
    }
    for bits in (8, 16, 32):
        plan = select_plan(k, bits, "large_primes")
        prepared = prepare_rns_weight(b, plan)
        outputs: dict[str, torch.Tensor] = {}
        runner_without_lut: RNSArchitectureRunner | None = None
        for label, lut_count in (
            ("none", 0),
            ("one", min(1, plan.channels)),
            ("two", min(2, plan.channels)),
            ("all", plan.channels),
        ):
            lut = build_compact_lut(plan.moduli, lut_count, device=device)
            runner = RNSArchitectureRunner(
                prepared,
                m=m,
                lut_channels=lut_count,
                compact_lut=lut,
            )
            output = runner.e2e(a).clone()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all():
                raise RuntimeError(f"q{bits} {label} LUT produced non-finite values")
            outputs[label] = output
            if label == "none":
                runner_without_lut = runner
        assert runner_without_lut is not None
        max_lut_difference = max(
            float((outputs["none"] - outputs[label]).abs().max().item())
            for label in ("one", "two", "all")
        )
        if max_lut_difference != 0.0:
            raise RuntimeError(
                f"q{bits} LUT and arithmetic paths disagree: {max_lut_difference}"
            )
        exactness = exact_sample_check(
            a_cpu=a_cpu,
            b_cpu=b_cpu,
            activation_scales=runner_without_lut.activation_scales,
            weight_scales=prepared.scales_n,
            output=outputs["none"],
            bits=bits,
        )
        report["formats"][f"q{bits}"] = {
            "channels": plan.channels,
            "moduli": list(plan.moduli),
            "required_bits": plan.required_bits,
            "product_bits": plan.product_bits,
            "relative_l2_to_fp32": rel_l2(outputs["none"], reference),
            "maximum_lut_path_difference": max_lut_difference,
            "exact_sample_check": exactness,
        }
    output_path = Path("results/v0.13/preflight_architecture_v013.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("v0.13 architecture preflight passed")


if __name__ == "__main__":
    main()
