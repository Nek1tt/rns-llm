from __future__ import annotations

import argparse
import copy
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
    install_full_rns_opt_attention,
    install_hybrid_opt_attention,
    install_native_int8_opt_attention,
)


def parse_shape(text: str):
    values = tuple(int(v) for v in text.lower().split("x"))
    if len(values) != 3: raise argparse.ArgumentTypeError("shape must be MxKxN")
    return values


def lut_count(label: str, channels: int) -> int:
    return {"none": 0, "one": min(1, channels), "two": min(2, channels), "all": channels}[label]


def build_matrix(args, device):
    m, k, n = args.shape
    layer = nn.Linear(k, n, bias=True, device=device, dtype=torch.float16).eval()
    with torch.no_grad():
        layer.weight.normal_(0, 1 / math.sqrt(k)); layer.bias.zero_()
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    if args.architecture == "fp16":
        module = layer
    elif args.architecture == "native_int8":
        module = NativeInt8LinearV014(layer).eval()
    elif args.architecture.startswith("full_rns_int"):
        is_v07 = args.architecture == "full_rns_int8_v07"
        bits = 8 if is_v07 else int(args.architecture.removeprefix("full_rns_int"))
        channels = select_plan(k, bits, args.moduli_policy).channels
        lut = lut_count(args.lut, channels)
        module = FullRNSLinearV014(
            layer, logical_bits=bits, lut_channels=lut,
            q8_backend="v07" if is_v07 or (bits == 8 and args.q8_backend == "v07") else "v013",
            moduli_policy=args.moduli_policy,
        ).eval()
    elif args.architecture.startswith("hybrid_fp16") or args.architecture.startswith("hybrid_rns_q"):
        parallel = args.architecture.endswith("_parallel")
        base_arch = args.architecture.removesuffix("_parallel")
        corr = "fp16" if base_arch == "hybrid_fp16" else "rns"
        bits = 16 if corr == "fp16" else int(base_arch.removeprefix("hybrid_rns_q"))
        channels = len(choose_moduli(bits, ((args.protected + 3)//4)*4)) if corr == "rns" else 0
        module = HybridLinearV014(
            layer, protected_channels=args.protected, correction_bits=bits,
            correction=corr, lut_channels=lut_count(args.lut, channels) if corr == "rns" else 0,
            execution="parallel" if parallel else args.hybrid_execution,
        ).eval()
    else:
        raise ValueError(args.architecture)
    return module, (lambda: module(x))


def causal_mask(batch, seq, device):
    mask = torch.zeros(seq, seq, device=device, dtype=torch.float16)
    mask = mask.masked_fill(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), torch.finfo(torch.float16).min)
    return mask.view(1, 1, seq, seq).expand(batch, 1, seq, seq).contiguous()


def build_attention(args, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    attention = copy.deepcopy(model.model.decoder.layers[0].self_attn).eval()
    hidden = int(model.config.hidden_size)
    del model
    x = torch.randn(args.batch, args.seq, hidden, device=device, dtype=torch.float16)
    mask = causal_mask(args.batch, args.seq, device)
    if args.architecture == "fp16":
        pass
    elif args.architecture.startswith("full_rns_int"):
        is_v07 = args.architecture == "full_rns_int8_v07"
        bits = 8 if is_v07 else int(args.architecture.removeprefix("full_rns_int"))
        channels = select_plan(hidden, bits, args.moduli_policy).channels
        install_full_rns_opt_attention(
            attention, logical_bits=bits, lut_channels=lut_count(args.lut, channels),
            q8_backend="v07" if is_v07 or (bits == 8 and args.q8_backend == "v07") else "v013",
            moduli_policy=args.moduli_policy,
        )
    elif args.architecture == "native_int8":
        install_native_int8_opt_attention(attention, include_out_proj=True)
    elif args.architecture.startswith("hybrid_fp16") or args.architecture.startswith("hybrid_rns_q"):
        parallel = args.architecture.endswith("_parallel")
        base_arch = args.architecture.removesuffix("_parallel")
        corr = "fp16" if base_arch == "hybrid_fp16" else "rns"
        bits = 16 if corr == "fp16" else int(base_arch.removeprefix("hybrid_rns_q"))
        channels = len(choose_moduli(bits, ((args.protected + 3)//4)*4)) if corr == "rns" else 0
        install_hybrid_opt_attention(
            attention, protected_channels=args.protected, correction_bits=bits,
            correction=corr, lut_channels=lut_count(args.lut, channels) if corr == "rns" else 0,
            execution="parallel" if parallel else args.hybrid_execution,
        )
    else:
        raise ValueError(args.architecture)
    return attention, (lambda: attention(x, attention_mask=mask, output_attentions=False)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["matrix", "attention"], default="matrix")
    ap.add_argument("--architecture", choices=[
        "fp16", "native_int8", "full_rns_int8", "full_rns_int16", "full_rns_int32",
        "full_rns_int8_v07", "hybrid_fp16", "hybrid_fp16_parallel",
        "hybrid_rns_q8", "hybrid_rns_q16", "hybrid_rns_q32",
        "hybrid_rns_q8_parallel", "hybrid_rns_q16_parallel", "hybrid_rns_q32_parallel",
    ], required=True)
    ap.add_argument("--lut", choices=["none", "one", "two", "all"], default="two")
    ap.add_argument("--shape", type=parse_shape, default=(128, 2560, 2560))
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--protected", type=int, default=3)
    ap.add_argument("--q8-backend", choices=["v07", "v013"], default="v013")
    ap.add_argument("--moduli-policy", choices=["dense_coprime", "large_primes", "school_small"], default="dense_coprime")
    ap.add_argument("--hybrid-execution", choices=["serial", "parallel"], default="serial")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--metadata", type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    device = torch.device("cuda")
    module, fn = build_matrix(args, device) if args.scope == "matrix" else build_attention(args, device)
    with torch.no_grad():
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        output = None
        start_status = torch.cuda.cudart().cudaProfilerStart()
        if start_status not in (None, 0):
            raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")
        torch.cuda.nvtx.range_push("PROFILE")
        try:
            for _ in range(args.iterations):
                output = fn()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
            stop_status = torch.cuda.cudart().cudaProfilerStop()
            if stop_status not in (None, 0):
                raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")
    payload = {
        "version": "0.14.2",
        "scope": args.scope,
        "architecture": args.architecture,
        "lut": args.lut,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model if args.scope == "attention" else None,
        "shape": list(args.shape) if args.scope == "matrix" else None,
        "batch": args.batch if args.scope == "attention" else None,
        "seq": args.seq if args.scope == "attention" else None,
        "protected": args.protected,
        "q8_backend": args.q8_backend,
        "moduli_policy": args.moduli_policy,
        "hybrid_execution": args.hybrid_execution,
        "checksum": float(output.float().sum().item()) if torch.is_tensor(output) else None,
    }
    print(json.dumps(payload, indent=2))
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
