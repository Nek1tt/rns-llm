#!/usr/bin/env python3
from __future__ import annotations

import json

import torch
from torch import nn

from rns_llm.ppl_v012 import (
    SimulatedQuantLinear,
    _garner_signed,
    apply_simulated_variant,
    choose_moduli,
    evaluate_sliding_window_ppl,
    finalize_summary,
)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 12)
        self.fc1 = nn.Linear(16, 24)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_proj(x), self.fc1(x)


class DummyLM(nn.Module):
    def __init__(self, vocab_size: int = 11) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, use_cache: bool = False):
        logits = torch.zeros((*input_ids.shape, self.vocab_size), dtype=torch.float32)
        shifted_logits = logits[:, :-1, :].reshape(-1, self.vocab_size)
        shifted_labels = labels[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100)
        return type("Output", (), {"loss": loss})()


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = Block()

    def forward(self, x: torch.Tensor):
        return self.block(x)


def plan() -> dict:
    selected = {
        "protected_indices": [1, 7],
        "protected_k": 2,
        "protected_k_padded": 4,
    }
    return {
        "layer_plans": {
            "block.q_proj": {"passed_local_gate": True, "selected": selected},
            "block.fc1": {"passed_local_gate": False, "selected": selected},
        }
    }


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 3, 16)
    for variant in ("native_int8", "hybrid_fp16", "hybrid_rns_q16"):
        model = Tiny().eval()
        info = apply_simulated_variant(model, plan(), variant=variant, fallback="best_effort")
        q, f = model(x)
        assert q.shape == (2, 3, 12)
        assert f.shape == (2, 3, 24)
        assert torch.isfinite(q).all() and torch.isfinite(f).all()
        assert info["replaced_layers"] == 2

    base = nn.Linear(16, 8).eval()
    wrapper = SimulatedQuantLinear(
        base,
        variant="hybrid_rns_q16",
        protected_indices=[1, 3, 5],
    )
    assert wrapper(torch.randn(4, 16)).shape == (4, 8)

    moduli = choose_moduli(16, 4)
    for value in (-2_000_000_000, -1, 0, 1, 2_000_000_000):
        residues = [value % modulus for modulus in moduli]
        assert _garner_signed(residues, moduli) == value

    ppl = evaluate_sliding_window_ppl(
        DummyLM(),
        torch.arange(20, dtype=torch.long).remainder(11).reshape(1, -1),
        device=torch.device("cpu"),
        context_length=8,
        stride=4,
        max_eval_tokens=16,
    )
    assert abs(ppl["ppl"] - 11.0) < 1e-4

    summary = {
        "results": {
            "fp16": {"ppl": 10.0},
            "hybrid_rns_q16": {"ppl": 10.4},
        }
    }
    finalize_summary(summary)
    assert summary["ppl_requirement"]["status"] == "PASS"
    print(json.dumps({"status": "PASS", "tests": 5}, indent=2))


if __name__ == "__main__":
    main()
