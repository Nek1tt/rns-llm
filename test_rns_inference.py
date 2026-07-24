import torch

from model import GPT, GPTConfig
from rns_inference import install_rns_inference
from rns_llm.layers import RNSLinear


def test_software_rns_installation_preserves_checkpoint_parameters_and_shapes():
    torch.manual_seed(12)
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=16,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
    ).eval()
    embedding_weight = model.transformer.wte.weight

    installed = install_rns_inference(
        model,
        mode="software-rns",
        quant_bits=8,
        include_attention_matmul=True,
    )
    logits, loss = model(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert len(installed.replaced_linears) == 5
    assert installed.attention_blocks == 1
    assert isinstance(model.lm_head, RNSLinear)
    assert model.lm_head.weight is embedding_weight
    assert installed.tied_lm_head
    assert logits.shape == (1, 1, 16)
    assert loss is None
    assert torch.isfinite(logits).all()
    assert installed.metadata()["qk_compute_count"] == 1
    assert installed.metadata()["av_compute_count"] == 1
