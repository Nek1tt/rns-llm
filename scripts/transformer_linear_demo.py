from __future__ import annotations

import argparse

import torch
from torch import nn

from rns_llm.backends import CudaRNSBackend
from rns_llm.layers import RNSLinear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--output", type=int, default=768)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("The optimized demo currently expects --device cuda")

    torch.manual_seed(42)
    baseline = nn.Linear(args.hidden, args.output, bias=True, device=device).eval()
    backend = CudaRNSBackend()
    rns_layer = RNSLinear.from_linear(
        baseline,
        backend=backend,
        mode="rns",
        kernel="auto",
    ).eval()

    inputs = torch.randn(args.batch, args.tokens, args.hidden, device=device)
    with torch.no_grad():
        expected = baseline(inputs)
        actual = rns_layer(inputs)

    error = (actual - expected).float()
    print("shape:", tuple(actual.shape))
    print("max_abs_error:", error.abs().max().item())
    print("mean_abs_error:", error.abs().mean().item())
    print("NOTE: error is from simple int8 quantization, not RNS arithmetic.")


if __name__ == "__main__":
    main()
