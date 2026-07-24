from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from rns_llm.architecture_v013 import select_plan
from rns_llm.hybrid_v010 import choose_moduli
from rns_llm.unified_v014 import (
    FullRNSLinearV014,
    HybridLinearV014,
    NativeInt8LinearV014,
)


def relative_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(output.float() - reference.float())
            / torch.clamp(torch.linalg.vector_norm(reference.float()), min=1e-30)
        ).item()
    )


def resolve_luts(channels: int) -> list[int]:
    return sorted({0, min(1, channels), min(2, channels), channels})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--protected", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/v0.14.2/preflight_v014.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    if args.k % 4 or args.n % 4:
        raise SystemExit("K and N must be multiples of four")

    device = torch.device("cuda")
    torch.manual_seed(1400)
    layer = nn.Linear(
        args.k, args.n, bias=True, device=device, dtype=torch.float16
    ).eval()
    with torch.no_grad():
        layer.weight.normal_(0, 1 / math.sqrt(args.k))
        layer.bias.normal_(0, 0.01)
    x = torch.randn(args.m, args.k, device=device, dtype=torch.float16)
    reference = layer(x)

    rows: list[dict[str, object]] = []
    equality: list[dict[str, object]] = []

    native = NativeInt8LinearV014(layer).eval()
    native_output = native(x)
    rows.append({
        "architecture": "native_int8",
        "relative_l2": relative_l2(native_output, reference),
        "finite": bool(torch.isfinite(native_output).all()),
        "memory": native.memory_report(),
    })

    for bits in (8, 16, 32):
        channels = select_plan(args.k, bits).channels
        outputs: dict[int, torch.Tensor] = {}
        for lut in resolve_luts(channels):
            module = FullRNSLinearV014(
                layer,
                logical_bits=bits,
                lut_channels=lut,
                q8_backend="v013",
            ).eval()
            output = module(x).detach().clone()
            torch.cuda.synchronize()
            outputs[lut] = output
            rows.append({
                "architecture": "full_rns",
                "bits": bits,
                "channels": channels,
                "lut_channels": lut,
                "relative_l2": relative_l2(output, reference),
                "finite": bool(torch.isfinite(output).all()),
                "memory": module.memory_report(),
            })
        maximum = max(
            float((output - outputs[0]).abs().max().item())
            for output in outputs.values()
        )
        equality.append({
            "architecture": "full_rns",
            "bits": bits,
            "max_lut_difference": maximum,
        })

    # Legacy v0.7 q8 path: its optimized epilogue supports 0/1/2 active LUTs.
    v07_outputs: dict[int, torch.Tensor] = {}
    for lut in (0, 1, 2):
        module = FullRNSLinearV014(
            layer,
            logical_bits=8,
            lut_channels=lut,
            q8_backend="v07",
        ).eval()
        output = module(x).detach().clone()
        torch.cuda.synchronize()
        v07_outputs[lut] = output
        rows.append({
            "architecture": "full_rns_v07",
            "bits": 8,
            "lut_channels": lut,
            "relative_l2": relative_l2(output, reference),
            "finite": bool(torch.isfinite(output).all()),
            "memory": module.memory_report(),
        })
    equality.append({
        "architecture": "full_rns_v07",
        "bits": 8,
        "max_lut_difference": max(
            float((output - v07_outputs[0]).abs().max().item())
            for output in v07_outputs.values()
        ),
    })

    p_padded = ((args.protected + 3) // 4) * 4
    for bits in (8, 16, 32):
        channels = len(choose_moduli(bits, p_padded))
        execution_outputs: dict[str, dict[int, torch.Tensor]] = {}
        for execution in ("serial", "parallel"):
            outputs: dict[int, torch.Tensor] = {}
            for lut in resolve_luts(channels):
                module = HybridLinearV014(
                    layer,
                    protected_channels=args.protected,
                    correction="rns",
                    correction_bits=bits,
                    lut_channels=lut,
                    execution=execution,
                ).eval()
                output = module(x).detach().clone()
                torch.cuda.synchronize()
                outputs[lut] = output
                rows.append({
                    "architecture": "hybrid_rns",
                    "bits": bits,
                    "channels": channels,
                    "lut_channels": lut,
                    "execution": execution,
                    "relative_l2": relative_l2(output, reference),
                    "finite": bool(torch.isfinite(output).all()),
                    "memory": module.memory_report(),
                })
            execution_outputs[execution] = outputs
            equality.append({
                "architecture": "hybrid_rns",
                "bits": bits,
                "execution": execution,
                "max_lut_difference": max(
                    float((output - outputs[0]).abs().max().item())
                    for output in outputs.values()
                ),
            })
        equality.append({
            "architecture": "hybrid_rns",
            "bits": bits,
            "comparison": "serial_vs_parallel",
            "max_execution_difference": max(
                float((execution_outputs["serial"][lut] - execution_outputs["parallel"][lut]).abs().max().item())
                for lut in execution_outputs["serial"]
            ),
        })

    fp16_execution_outputs: dict[str, torch.Tensor] = {}
    for execution in ("serial", "parallel"):
        module = HybridLinearV014(
            layer,
            protected_channels=args.protected,
            correction="fp16",
            correction_bits=16,
            lut_channels=0,
            execution=execution,
        ).eval()
        output = module(x).detach().clone()
        torch.cuda.synchronize()
        fp16_execution_outputs[execution] = output
        rows.append({
            "architecture": "hybrid_fp16",
            "execution": execution,
            "relative_l2": relative_l2(output, reference),
            "finite": bool(torch.isfinite(output).all()),
            "memory": module.memory_report(),
        })
    equality.append({
        "architecture": "hybrid_fp16",
        "comparison": "serial_vs_parallel",
        "max_execution_difference": float(
            (fp16_execution_outputs["serial"] - fp16_execution_outputs["parallel"]).abs().max().item()
        ),
    })

    failures = []
    if not all(bool(row["finite"]) for row in rows):
        failures.append("non-finite output")
    for item in equality:
        is_execution = "max_execution_difference" in item
        difference = float(
            item.get("max_execution_difference", item.get("max_lut_difference", 0.0))
        )
        tolerance = 1e-5 if is_execution else 0.0
        if difference > tolerance:
            failures.append(
                f"{item.get('architecture')} q{item.get('bits', 'n/a')} "
                f"{item.get('comparison', 'LUT')} mismatch: {difference} "
                f"(tolerance={tolerance})"
            )

    payload = {
        "version": "0.14.2",
        "gpu": torch.cuda.get_device_name(0),
        "shape": [args.m, args.k, args.n],
        "rows": rows,
        "lut_equivalence": equality,
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
