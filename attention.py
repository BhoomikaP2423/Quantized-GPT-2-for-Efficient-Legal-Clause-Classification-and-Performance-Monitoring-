"""Masked scaled dot-product self-attention, written by hand
(no nn.MultiheadAttention)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular mask: position t may only look at positions <= t
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)   # (B, T, head_size)
        # scaled dot-product: Q K^T / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        # causal mask: -inf before softmax so future tokens get weight exactly 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v                                         # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_heads * head_size, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))
