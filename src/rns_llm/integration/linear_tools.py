"""Find and replace Transformer Linear layers. OWNER: Transformer integration."""
from torch import nn
from rns_llm.layers.rns_linear import RNSLinear


def list_linear_layers(model: nn.Module):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _get_parent_module(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_by_name(model: nn.Module, layer_name: str, *, mode="torch", backend=None, moduli=(3,5,7,11)) -> RNSLinear:
    parent, attribute = _get_parent_module(model, layer_name)
    current = getattr(parent, attribute)
    if not isinstance(current, nn.Linear):
        raise TypeError(f"{layer_name!r} is not nn.Linear")
    replacement = RNSLinear.from_linear(current, mode=mode, backend=backend, moduli=moduli)
    replacement.to(device=current.weight.device, dtype=current.weight.dtype)
    setattr(parent, attribute, replacement)
    return replacement
