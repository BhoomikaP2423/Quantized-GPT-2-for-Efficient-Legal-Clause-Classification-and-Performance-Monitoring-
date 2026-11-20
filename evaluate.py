"""Evaluate a trained checkpoint: val loss, perplexity, a unigram baseline,
and sample text. Writes RESULTS.md and fills the results block in README.md."""
import math
import re

import torch

from generate import MiniGPT
from tokenizer import CharTokenizer
from train import estimate_loss


def main(data="data/input.txt", weights="gpt.pt", batches=200):
    torch.manual_seed(0)
    text = open(data, encoding="utf-8").read()
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    gpt = MiniGPT(weights)
    bs = gpt.cfg.block_size
    val_loss = estimate_loss(gpt.model, val_ids, bs, 16, iters=batches)
    params = sum(p.numel() for p in gpt.model.parameters())

    counts = torch.bincount(train_ids, minlength=tok.vocab_size).float() + 1
    p = counts / counts.sum()
    unigram = -torch.log(p[val_ids]).mean().item()
    uniform = math.log(tok.vocab_size)

    samples = []
    for temp in (0.5, 0.8, 1.0):
        samples.append((temp, gpt.complete("\n", 250, temperature=temp, top_k=20)))

    md = [
        f"Corpus: `{data}` ({len(text):,} characters, vocab {tok.vocab_size}); "
        f"model: {params:,} parameters, context {bs}, "
        f"{gpt.cfg.n_layers} layers x {gpt.cfg.n_heads} heads, n_embd {gpt.cfg.n_embd}.",
        "",
        "| Model | Validation loss (nats/char) | Perplexity |",
        "|---|---|---|",
        f"| Uniform guess (ln vocab) | {uniform:.3f} | {math.exp(uniform):.1f} |",
        f"| Unigram character frequencies | {unigram:.3f} | {math.exp(unigram):.1f} |",
        f"| **Mini-GPT (this repo)** | **{val_loss:.3f}** | **{math.exp(val_loss):.1f}** |",
        "",
        "Samples (top-k 20):",
    ]
    for temp, s in samples:
        md += ["", f"Temperature {temp}:", "```", s, "```"]
    block = "\n".join(md)

    open("RESULTS.md", "w", encoding="utf-8").write("# Results\n\n" + block + "\n")
    try:
        readme = open("README.md", encoding="utf-8").read()
        new = re.sub(r"<!--RESULTS-->.*?<!--/RESULTS-->",
                     lambda m: "<!--RESULTS-->\n" + block + "\n<!--/RESULTS-->",
                     readme, flags=re.S)
        open("README.md", "w", encoding="utf-8").write(new)
    except FileNotFoundError:
        pass
    print(block)


if __name__ == "__main__":
    main()
