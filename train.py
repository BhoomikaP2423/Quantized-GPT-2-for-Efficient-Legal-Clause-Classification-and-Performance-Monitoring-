"""Train the mini-GPT with next-token cross-entropy and log the loss."""
import argparse
import csv
import math

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # shifted by one
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, iters=50):
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size)
        total += model(x, y)[1].item()
    model.train()
    return total / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out", default="gpt.pt")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    text = open(a.data, encoding="utf-8").read()
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(tok.vocab_size, a.block_size, a.n_embd, a.n_heads, a.n_layers)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    print(f"vocab={tok.vocab_size}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"expected initial loss ~ ln(vocab) = {math.log(tok.vocab_size):.3f}")

    log = []
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            tr = estimate_loss(model, train_data, a.block_size, a.batch_size)
            va = estimate_loss(model, val_data, a.block_size, a.batch_size)
            log.append((step, tr, va))
            print(f"step {step:5d} | train {tr:.4f} | val {va:.4f}")
        x, y = get_batch(train_data, a.block_size, a.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                "chars": tok.chars}, a.out)
    print(f"saved {a.out}")

    with open("loss_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "train_loss", "val_loss"]); w.writerows(log)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s, t, v = zip(*log)
        plt.plot(s, t, label="train"); plt.plot(s, v, label="val")
        plt.xlabel("step"); plt.ylabel("cross-entropy loss"); plt.legend()
        plt.title("Mini-GPT training loss"); plt.savefig("loss_curve.png", dpi=120)
        print("saved loss_curve.png")
    except ImportError:
        print("matplotlib not installed; see loss_log.csv")


if __name__ == "__main__":
    main()
