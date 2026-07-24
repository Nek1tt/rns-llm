from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch

from rns_llm.hybrid_v010 import HybridCudaOps, pad_inner_dimension


@contextmanager
def nvtx(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def rows(samples: torch.Tensor, m: int, device: torch.device) -> torch.Tensor:
    x = samples.float()
    if x.shape[0] < m:
        x = x.repeat(((m + x.shape[0] - 1) // x.shape[0], 1))
    return x[:m].to(device).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=[
        "fp16", "native_int8", "full_rns_q8", "full_rns_q16", "full_rns_q32",
        "hybrid_int8_plus_fp16", "hybrid_int8_plus_rns_q8",
        "hybrid_int8_plus_rns_q16", "hybrid_int8_plus_rns_q32",
    ])
    parser.add_argument("--stage", choices=["e2e", "prepared"], default="e2e")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--disable-profiler-api", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    pack = torch.load(args.pack, map_location="cpu", weights_only=False)
    weight = pack["weight"].float().to(device).contiguous()
    bias = pack.get("bias")
    bias = torch.zeros(weight.shape[0], device=device) if bias is None else bias.float().to(device).contiguous()
    x = rows(pack["activation_samples"], args.m, device)
    protected = pack["protected_indices"].long().to(device)
    mask = torch.ones(weight.shape[1], dtype=torch.bool, device=device)
    mask[protected] = False
    safe = torch.nonzero(mask, as_tuple=False).flatten()
    ops = HybridCudaOps(device)

    method = args.method
    if method == "fp16":
        w16, b16, x16 = weight.half(), bias.half(), x.half()
        e2e = lambda: (x.half() @ w16.t() + b16).float()
        prepared = lambda: (x16 @ w16.t() + b16).float()
    elif method == "native_int8":
        xp, wp, _ = pad_inner_dimension(x, weight)
        pw = ops.prepare_native_weight(wp)
        pa = ops.prepare_native_activation(xp)
        out = torch.empty((args.m, weight.shape[0]), device=device)
        acc = torch.empty_like(out, dtype=torch.int32)
        e2e = lambda: ops.native_e2e(xp, pw, bias)
        prepared = lambda: ops.native_core(pa, pw, bias, output=out, accumulators=acc)
    elif method.startswith("full_rns_q"):
        bits = int(method.rsplit("q", 1)[1])
        xp, wp, _ = pad_inner_dimension(x, weight)
        pw = ops.prepare_rns_weight(wp, bits)
        pa = ops.prepare_rns_activation(xp, pw)
        out = torch.empty((args.m, weight.shape[0]), device=device)
        acc = torch.empty((len(pw.moduli), args.m, weight.shape[0]), device=device, dtype=torch.int32)
        e2e = lambda: ops.rns_e2e(xp, pw, bias)
        prepared = lambda: ops.rns_core(pa, pw, bias, output=out, accumulators=acc)
    else:
        xs = x.index_select(1, safe)
        ws = weight.index_select(1, safe)
        xp = x.index_select(1, protected)
        wp = weight.index_select(1, protected)
        xs_pad, ws_pad, safe_pad = pad_inner_dimension(xs, ws)
        xp_pad, wp_pad, prot_pad = pad_inner_dimension(xp, wp)
        main_w = ops.prepare_native_weight(ws_pad)
        main_a = ops.prepare_native_activation(xs_pad)
        main_out = torch.empty((args.m, weight.shape[0]), device=device)
        main_acc = torch.empty_like(main_out, dtype=torch.int32)

        if method == "hybrid_int8_plus_fp16":
            wp16, xp16 = wp.half(), xp.half()

            def e2e():
                with nvtx("HYBRID_GATHER"):
                    current_safe = x.index_select(1, safe)
                    current_prot = x.index_select(1, protected)
                    if safe_pad:
                        current_safe = torch.nn.functional.pad(current_safe, (0, safe_pad))
                with nvtx("HYBRID_NATIVE_INT8_MAIN"):
                    left = ops.native_e2e(current_safe.contiguous(), main_w, None)
                with nvtx("HYBRID_FP16_PROTECTED"):
                    right = (current_prot.half() @ wp16.t()).float()
                with nvtx("HYBRID_MERGE"):
                    return left + right + bias

            def prepared():
                with nvtx("HYBRID_NATIVE_INT8_MAIN"):
                    left = ops.native_core(main_a, main_w, None, output=main_out, accumulators=main_acc)
                with nvtx("HYBRID_FP16_PROTECTED"):
                    right = (xp16 @ wp16.t()).float()
                with nvtx("HYBRID_MERGE"):
                    return left + right + bias
        else:
            bits = int(method.rsplit("q", 1)[1])
            prot_w = ops.prepare_rns_weight(wp_pad, bits)
            prot_a = ops.prepare_rns_activation(xp_pad, prot_w)
            prot_out = torch.empty((args.m, weight.shape[0]), device=device)
            prot_acc = torch.empty((len(prot_w.moduli), args.m, weight.shape[0]), device=device, dtype=torch.int32)

            def e2e():
                with nvtx("HYBRID_GATHER"):
                    current_safe = x.index_select(1, safe)
                    current_prot = x.index_select(1, protected)
                    if safe_pad:
                        current_safe = torch.nn.functional.pad(current_safe, (0, safe_pad))
                    if prot_pad:
                        current_prot = torch.nn.functional.pad(current_prot, (0, prot_pad))
                with nvtx("HYBRID_NATIVE_INT8_MAIN"):
                    left = ops.native_e2e(current_safe.contiguous(), main_w, None)
                with nvtx(f"HYBRID_RNS_Q{bits}_PROTECTED"):
                    right = ops.rns_e2e(current_prot.contiguous(), prot_w, None)
                with nvtx("HYBRID_MERGE"):
                    return left + right + bias

            def prepared():
                with nvtx("HYBRID_NATIVE_INT8_MAIN"):
                    left = ops.native_core(main_a, main_w, None, output=main_out, accumulators=main_acc)
                with nvtx(f"HYBRID_RNS_Q{bits}_PROTECTED"):
                    right = ops.rns_core(prot_a, prot_w, None, output=prot_out, accumulators=prot_acc)
                with nvtx("HYBRID_MERGE"):
                    return left + right + bias

    target = e2e if args.stage == "e2e" else prepared
    for _ in range(args.warmup):
        target()
    torch.cuda.synchronize()

    label = f"HYBRID_PROFILE_{method.upper()}_{args.stage.upper()}_M{args.m}"
    if not args.disable_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()
    with nvtx(label):
        for _ in range(args.repeats):
            target()
    torch.cuda.synchronize()
    if not args.disable_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()
    print({"method": method, "stage": args.stage, "m": args.m, "repeats": args.repeats})


if __name__ == "__main__":
    main()
