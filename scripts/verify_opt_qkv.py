from __future__ import annotations

import argparse

import torch
from torch import nn

from rns_llm.backends import CudaRNSBackend
from rns_llm.layers import RNSLinear, RNSQKVProjection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--quant-bits", type=int, choices=[8, 12], default=8)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit('Install: pip install -e ".[transformer]"') from exc

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).cuda().eval()
    attention = model.model.decoder.layers[0].self_attn
    originals = (attention.q_proj, attention.k_proj, attention.v_proj)
    if any(not isinstance(x, nn.Linear) for x in originals):
        raise SystemExit("Expected OPT-style q/k/v nn.Linear modules")

    backend = CudaRNSBackend()
    separate = tuple(
        RNSLinear.from_linear(
            layer,
            backend=backend,
            mode="rns",
            quant_bits=args.quant_bits,
            fused=True,
            lut_channels=2,
        ).eval()
        for layer in originals
    )
    fused = RNSQKVProjection.from_linears(
        *originals,
        backend=backend,
        mode="rns",
        quant_bits=args.quant_bits,
        fused=True,
        lut_channels=2,
    ).eval()

    torch.manual_seed(2026)
    x = torch.randn(
        1,
        args.tokens,
        attention.embed_dim,
        device="cuda",
        dtype=torch.float16,
    )
    expected_rns = tuple(layer(x) for layer in separate)
    actual_rns = fused(x)
    fp_outputs = tuple(layer(x) for layer in originals)

    for name, actual, expected, fp in zip(("q", "k", "v"), actual_rns, expected_rns, fp_outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        diff = (actual.float() - fp.float()).abs()
        print(
            f"{name}: fused_equals_separate_rns=True "
            f"max_rns_fusion_error=0 "
            f"max_vs_fp={diff.max().item():.6g} "
            f"mean_vs_fp={diff.mean().item():.6g}"
        )
    print("PASS: QKV fusion changes launch architecture but not RNS arithmetic")


if __name__ == "__main__":
    main()
