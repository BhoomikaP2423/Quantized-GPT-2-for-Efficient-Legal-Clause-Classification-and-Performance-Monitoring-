"""Load trained weights and generate text (temperature / top-k / top-p)."""
import argparse

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


class MiniGPT:
    "Load a trained GPT and generate text from a prompt."

    def __init__(self, weights="gpt.pt"):
        ckpt = torch.load(weights, map_location="cpu")
        self.cfg = GPTConfig(**ckpt["config"])
        self.tok = CharTokenizer(ckpt["chars"])
        self.model = GPT(self.cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def complete(self, prompt="\n", max_new_tokens=300, temperature=0.8,
                 top_k=20, top_p=None):
        idx = torch.tensor([self.tok.encode(prompt)], dtype=torch.long)
        for _ in range(max_new_tokens):
            logits, _ = self.model(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k:                                   # keep the k best tokens
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k)[0][:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p:                                   # nucleus sampling
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return self.tok.decode(idx[0].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="gpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=None)
    a = p.parse_args()
    gpt = MiniGPT(a.weights)
    print(gpt.complete(a.prompt, a.tokens, a.temperature, a.top_k, a.top_p))
