from __future__ import annotations

import argparse

import torch
from torch import nn

from rns_llm.backends import CudaRNSBackend
from rns_llm.layers import RNSLinear, install_opt_qkv_fusion


def find_parent(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_selected_linears(
    model: nn.Module,
    *,
    backend: CudaRNSBackend,
    patterns: tuple[str, ...],
    max_layers: int,
    quant_bits: int,
    adaptive_channels: bool,
) -> list[str]:
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(pattern in name for pattern in patterns)
    ]
    replaced = []
    for name, layer in candidates[:max_layers]:
        parent, attribute = find_parent(model, name)
        replacement = RNSLinear.from_linear(
            layer,
            backend=backend,
            mode="rns",
            quant_bits=quant_bits,
            fused=True,
            lut_channels=2,
            moduli_strategy="dense_coprime",
            adaptive_channels=adaptive_channels,
        ).eval()
        setattr(parent, attribute, replacement)
        replaced.append(name)
    return replaced


def replace_opt_attention_blocks(
    model: nn.Module,
    *,
    backend: CudaRNSBackend,
    blocks: int,
    quant_bits: int,
    include_out_proj: bool,
    adaptive_channels: bool,
) -> tuple[list[str], list[nn.Module]]:
    candidates = []
    for name, module in model.named_modules():
        if all(isinstance(getattr(module, attr, None), nn.Linear) for attr in ("q_proj", "k_proj", "v_proj")):
            candidates.append((name, module))

    replaced: list[str] = []
    coordinators: list[nn.Module] = []
    for name, attention in candidates[:blocks]:
        coordinator = install_opt_qkv_fusion(
            attention,
            backend=backend,
            quant_bits=quant_bits,
            moduli_strategy="dense_coprime",
            lut_channels=2,
            adaptive_channels=adaptive_channels,
        )
        coordinators.append(coordinator)
        replaced.append(f"{name}.qkv_fused")
        if include_out_proj:
            if not isinstance(attention.out_proj, nn.Linear):
                raise TypeError(f"{name}.out_proj is not nn.Linear")
            attention.out_proj = RNSLinear.from_linear(
                attention.out_proj,
                backend=backend,
                mode="rns",
                quant_bits=quant_bits,
                fused=True,
                lut_channels=2,
                moduli_strategy="dense_coprime",
                adaptive_channels=adaptive_channels,
            ).eval()
            replaced.append(f"{name}.out_proj")
    return replaced, coordinators


@torch.no_grad()
def perplexity(model, input_ids: torch.Tensor, stride: int, max_length: int) -> float:
    model.eval()
    sequence_length = input_ids.size(1)
    losses = []
    previous_end = 0
    for begin in range(0, sequence_length, stride):
        end = min(begin + max_length, sequence_length)
        target_length = end - previous_end
        window = input_ids[:, begin:end].to(model.device)
        labels = window.clone()
        labels[:, :-target_length] = -100
        output = model(window, labels=labels)
        losses.append(output.loss.float() * target_length)
        previous_end = end
        if end == sequence_length:
            break
    return float(torch.exp(torch.stack(losses).sum() / previous_end).item())


def load_public_dataset(args, load_dataset):
    try:
        return load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    except Exception as first_error:
        if (
            args.dataset == "Salesforce/wikitext"
            and args.dataset_config == "wikitext-2-raw-v1"
            and args.dataset_split == "test"
        ):
            from huggingface_hub import hf_hub_download

            parquet_path = hf_hub_download(
                repo_id="Salesforce/wikitext",
                repo_type="dataset",
                filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
            )
            dataset = load_dataset(
                "parquet", data_files={"test": parquet_path}, split="test"
            )
            print("WARNING: used direct Parquet fallback:", repr(first_error))
            return dataset
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-samples", type=int, default=64)
    parser.add_argument("--quant-bits", type=int, choices=[8, 12], default=8)
    parser.add_argument(
        "--replacement-mode", choices=["linears", "opt-qkv"], default="opt-qkv"
    )
    parser.add_argument("--attention-blocks", type=int, default=1)
    parser.add_argument("--include-out-proj", action="store_true")
    parser.add_argument("--adaptive-channels", action="store_true")
    # Legacy arbitrary-Linear mode.
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument(
        "--patterns", nargs="+", default=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install: pip install -e ".[transformer]"') from exc

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_public_dataset(args, load_dataset)
    text = "\n\n".join(dataset["text"][: args.dataset_samples])
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).to(device).eval()
    baseline_ppl = perplexity(model, input_ids, args.stride, args.max_length)

    backend = CudaRNSBackend()
    coordinators: list[nn.Module] = []
    if args.replacement_mode == "opt-qkv":
        replaced, coordinators = replace_opt_attention_blocks(
            model,
            backend=backend,
            blocks=args.attention_blocks,
            quant_bits=args.quant_bits,
            include_out_proj=args.include_out_proj,
            adaptive_channels=args.adaptive_channels,
        )
    else:
        replaced = replace_selected_linears(
            model,
            backend=backend,
            patterns=tuple(args.patterns),
            max_layers=args.max_layers,
            quant_bits=args.quant_bits,
            adaptive_channels=args.adaptive_channels,
        )
    if not replaced:
        raise SystemExit("No matching modules found")

    backend.reset_stats()
    rns_ppl = perplexity(model, input_ids, args.stride, args.max_length)
    stats = backend.stats_snapshot()
    relative_increase = rns_ppl / baseline_ppl - 1.0
    fused_compute_count = sum(
        getattr(coordinator.projection, "compute_count", 0) for coordinator in coordinators
    )

    if stats.get("fused_gemm_calls", 0) <= 0:
        raise RuntimeError(
            "No fused RNS GEMM calls were observed; refusing to report a misleading PPL result"
        )
    if coordinators and fused_compute_count <= 0:
        raise RuntimeError("QKV coordinators were installed but never executed")

    print("replacement_mode=", args.replacement_mode)
    print("replaced_modules:")
    for name in replaced:
        print(" -", name)
    print("backend_stats=", stats)
    print("qkv_fused_compute_count=", fused_compute_count)
    print(f"baseline_ppl={baseline_ppl:.6f}")
    print(f"rns_ppl={rns_ppl:.6f}")
    print(f"relative_increase={relative_increase * 100:.3f}%")
    print(f"under_5_percent={relative_increase < 0.05}")


if __name__ == "__main__":
    main()
