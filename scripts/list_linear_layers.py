"""Example: python scripts/list_linear_layers.py --model distilgpt2"""
import argparse


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="distilgpt2"); args = p.parse_args()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit('Install: pip install -e ".[transformer]"') from exc
    from rns_llm.integration import list_linear_layers
    model = AutoModelForCausalLM.from_pretrained(args.model)
    layers = list_linear_layers(model)
    if not layers:
        print("No nn.Linear layers found. Inspect model-specific projection modules.")
    for name, layer in layers:
        print(f"{name}: in_features={layer.in_features}, out_features={layer.out_features}")

if __name__ == "__main__": main()
