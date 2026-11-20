import math

import torch

from generate import MiniGPT
from model import GPT, GPTConfig


def small_cfg(vocab=65):
    return GPTConfig(vocab_size=vocab, block_size=16, n_embd=32, n_heads=4, n_layers=2)


def test_logits_shape_and_loss_exists():
    model = GPT(small_cfg())
    idx = torch.randint(0, 65, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, 65)
    assert loss.item() > 0


def test_initial_loss_is_near_ln_vocab():
    torch.manual_seed(0)
    model = GPT(small_cfg())
    idx = torch.randint(0, 65, (16, 16))
    _, loss = model(idx, torch.randint(0, 65, (16, 16)))
    assert abs(loss.item() - math.log(65)) < 0.6


def test_one_step_lowers_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    model = GPT(small_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 65, (8, 16)); y = torch.randint(0, 65, (8, 16))
    first = model(x, y)[1].item()
    for _ in range(20):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first


def test_minigpt_generates_requested_length(tmp_path):
    chars = list("abcdefghijklmnopqrstuvwxyz \n")
    cfg = GPTConfig(vocab_size=len(chars), block_size=16, n_embd=32, n_heads=4, n_layers=2)
    path = tmp_path / "w.pt"
    torch.save({"state_dict": GPT(cfg).state_dict(), "config": cfg.to_dict(), "chars": chars}, path)
    out = MiniGPT(str(path)).complete("ab", max_new_tokens=10, top_k=5)
    assert len(out) == 12 and out.startswith("ab")
