"""Transformer block and the full decoder-only GPT."""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 32    # context length
    n_embd: int = 64        # embedding size
    n_heads: int = 4        # attention heads
    n_layers: int = 3       # transformer blocks

    def to_dict(self):
        return asdict(self)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: residual + layer norm + attention + MLP."""

    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttention(cfg.n_embd, cfg.n_heads, cfg.block_size)
        self.ff = FeedForward(cfg.n_embd)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)                      # (B, T, vocab)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
