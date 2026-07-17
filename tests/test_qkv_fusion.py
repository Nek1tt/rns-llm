import torch
from torch import nn

from rns_llm.layers import CachedRNSQKV, RNSQKVProjection


def make_linears(dtype=torch.float32):
    torch.manual_seed(123)
    return tuple(nn.Linear(16, 16, bias=True, dtype=dtype).eval() for _ in range(3))


def test_torch_qkv_fusion_matches_three_linears_numerically():
    q, k, v = make_linears()
    fused = RNSQKVProjection.from_linears(q, k, v, mode="torch").eval()
    x = torch.randn(2, 5, 16)
    expected = (q(x), k(x), v(x))
    actual = fused(x)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=2e-7)


def test_cached_qkv_computes_once_for_three_projection_calls():
    q, k, v = make_linears()
    projection = RNSQKVProjection.from_linears(q, k, v, mode="torch").eval()
    coordinator = CachedRNSQKV(projection).eval()
    q_proxy, k_proxy, v_proxy = coordinator.slices()
    x = torch.randn(2, 5, 16)

    actual = (q_proxy(x), k_proxy(x), v_proxy(x))
    expected = (q(x), k(x), v(x))
    assert projection.compute_count == 1
    assert coordinator.cache_misses == 1
    assert coordinator.cache_hits == 2
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=2e-7)

    # A second logical attention call must recompute, not reuse stale tensors.
    _ = q_proxy(x)
    _ = k_proxy(x)
    _ = v_proxy(x)
    assert projection.compute_count == 2


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = make_linears()

    def forward(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)


def test_install_api_preserves_attention_projection_outputs_in_torch_mode():
    # The generic installer uses RNS mode, so here we test the same proxy/cache
    # mechanism directly with a torch-mode coordinator.
    attention = DummyAttention().eval()
    x = torch.randn(2, 3, 16)
    expected = attention(x)
    projection = RNSQKVProjection.from_linears(
        attention.q_proj, attention.k_proj, attention.v_proj, mode="torch"
    ).eval()
    coordinator = CachedRNSQKV(projection).eval()
    attention.rns_qkv = coordinator
    attention.q_proj, attention.k_proj, attention.v_proj = coordinator.slices()
    actual = attention(x)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=2e-7)
    assert projection.compute_count == 1
