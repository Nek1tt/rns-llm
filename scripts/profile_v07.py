from __future__ import annotations

import argparse
import json
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from rns_llm.backends import CudaRNSBackend
from rns_llm.layers import FastRNSLinearV07, RNSLinear
from rns_llm.v07_backend import V07FastPath


@contextmanager
def nvtx(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def build_workload(args):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    backend = CudaRNSBackend()

    source = nn.Linear(
        args.k,
        args.n,
        bias=True,
        dtype=torch.float16,
        device=device,
    ).eval()
    inputs = torch.randn(
        args.m,
        args.k,
        dtype=torch.float16,
        device=device,
    )

    if args.backend == "fp16":
        return lambda: F.linear(inputs, source.weight, source.bias), {
            "channels": 0,
            "weight_bytes": tensor_bytes(source.weight),
        }

    v07 = FastRNSLinearV07.from_linear(
        source,
        backend=backend,
        mode="rns",
        quant_bits=8,
        fused=True,
        lut_channels=2,
        moduli_strategy="dense_coprime",
        use_v07_epilogue=True,
        fuse_quantize_encode=True,
    ).eval()
    v07.prepare_weight()
    assert v07._prepared_weight is not None
    assert v07._v07_fast_path is not None

    if args.backend == "rns_v07":
        if args.stage == "e2e":
            fn = lambda: v07(inputs)
        else:
            activation_scale = v07._activation_scale(inputs)
            workspace = v07._v07_workspace(args.m, len(v07.moduli))
            activation_residues = v07._v07_fast_path.quantize_encode_fp16(
                inputs,
                v07._prepared_weight.moduli,
                scales=activation_scale,
                quant_max=v07.quant_max,
                output=workspace.activation_residues,
            )
            torch.cuda.synchronize()

            def fn():
                return v07._v07_fast_path.rns_encoded_dequant_fp16(
                    activation_residues,
                    v07._prepared_weight,
                    activation_scale=activation_scale,
                    weight_scale=v07._weight_scale,
                    bias=None if v07.bias is None else v07._bias_float_cache,
                    lut_channels=v07.lut_channels,
                    workspace=workspace,
                )

        return fn, {
            "channels": len(v07.moduli),
            "moduli": list(v07.moduli),
            "weight_bytes": tensor_bytes(v07._prepared_weight.residues),
            "stage": args.stage,
        }

    if args.backend == "rns_v06":
        if args.stage != "e2e":
            raise ValueError("rns_v06 only supports stage=e2e")
        v06 = RNSLinear.from_linear(
            source,
            backend=backend,
            mode="rns",
            quant_bits=8,
            fused=True,
            lut_channels=2,
            moduli_strategy="dense_coprime",
        ).eval()
        v06.prepare_weight()
        return lambda: v06(inputs), {
            "channels": len(v06.moduli),
            "moduli": list(v06.moduli),
        }

    if args.backend == "native_int8":
        fast = V07FastPath(backend)
        weight_scale = v07._weight_scale
        weight_q = v07._quantize(
            v07.weight.float(),
            weight_scale.unsqueeze(1),
        ).transpose(0, 1).contiguous()
        bias_float = source.bias.detach().float().contiguous()
        workspace = fast.create_native_workspace(
            device=device,
            m=args.m,
            k=args.k,
            n=args.n,
        )
        activation_scale = v07._activation_scale(inputs)

        if args.stage == "e2e":
            def fn():
                scale = v07._activation_scale(inputs)
                return fast.native_fp16_input_dequant_fp16(
                    inputs,
                    weight_q,
                    activation_scale=scale,
                    weight_scale=weight_scale,
                    bias=bias_float,
                    quant_max=v07.quant_max,
                    workspace=workspace,
                )
        else:
            activation_q = fast.quantize_fp16(
                inputs,
                scales=activation_scale,
                quant_max=v07.quant_max,
                output=workspace.activation_quantized,
            )
            torch.cuda.synchronize()

            def fn():
                return fast.native_int8_dequant_fp16(
                    activation_q,
                    weight_q,
                    activation_scale=activation_scale,
                    weight_scale=weight_scale,
                    bias=bias_float,
                    workspace=workspace,
                )

        return fn, {
            "channels": 1,
            "weight_bytes": tensor_bytes(weight_q),
            "stage": args.stage,
        }

    raise ValueError(f"unknown backend: {args.backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["fp16", "native_int8", "rns_v06", "rns_v07"],
        default="rns_v07",
    )
    parser.add_argument(
        "--stage",
        choices=["e2e", "prepared"],
        default="e2e",
        help="prepared excludes activation quantize/encode from the timed range",
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7071)
    parser.add_argument("--disable-profiler-api", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.backend == "fp16":
        args.stage = "e2e"

    # Build modules and caches under no_grad, not inference_mode: Parameter
    # version counters are inspected by weight-cache preparation.
    with torch.no_grad():
        fn, metadata = build_workload(args)
        for _ in range(args.warmup):
            output = fn()
        torch.cuda.synchronize()

        backend_label = {
            "fp16": "FP16",
            "native_int8": "NATIVE_INT8",
            "rns_v06": "RNS_V06",
            "rns_v07": "RNS_V07",
        }[args.backend]
        inner_label = (
            f"V07_{backend_label}_{args.stage.upper()}_"
            f"M{args.m}_K{args.k}_N{args.n}"
        )

        if not args.disable_profiler_api:
            torch.cuda.profiler.start()
        try:
            with nvtx("RNS_V07_PROFILE"):
                with nvtx(inner_label):
                    for _ in range(args.repeats):
                        output = fn()
                torch.cuda.synchronize()
        finally:
            if not args.disable_profiler_api:
                torch.cuda.profiler.stop()

        checksum = float(output.float().sum().item())

    print(
        json.dumps(
            {
                "version": "0.7.1",
                "backend": args.backend,
                "stage": args.stage,
                "shape": [args.m, args.k, args.n],
                "warmup": args.warmup,
                "repeats": args.repeats,
                "outer_nvtx": "RNS_V07_PROFILE",
                "inner_nvtx": inner_label,
                "metadata": metadata,
                "checksum_for_dead_code_prevention_only": checksum,
                "gpu": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
