import torch

from attention import Head, MultiHeadAttention


def test_head_output_shape():
    head = Head(n_embd=16, head_size=4, block_size=8)
    assert head(torch.randn(2, 8, 16)).shape == (2, 8, 4)


def test_causal_mask_blocks_the_future():
    """Changing the LAST token must not change any earlier position's output."""
    torch.manual_seed(0)
    head = Head(n_embd=16, head_size=4, block_size=8)
    x = torch.randn(1, 8, 16)
    x_changed = x.clone()
    x_changed[:, -1] += 5.0
    assert torch.allclose(head(x)[:, :-1], head(x_changed)[:, :-1], atol=1e-6)
    assert not torch.allclose(head(x)[:, -1], head(x_changed)[:, -1])


def test_multihead_shape():
    mha = MultiHeadAttention(n_embd=16, n_heads=4, block_size=8)
    assert mha(torch.randn(2, 8, 16)).shape == (2, 8, 16)
