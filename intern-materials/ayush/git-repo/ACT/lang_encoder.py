"""
lang_encoder.py
===============
A tiny (~4M-param) language encoder to give the ACT policy language conditioning
(stock ACT has none). Tokenizes the task instruction (BERT word-piece tokenizer,
tokenizer only — no BERT weights), embeds it with a fresh 128-d embedding table,
runs 2 small transformer layers, mean-pools, and projects to the ACT model dim.
The pooled vector is added to the ACT encoder as ONE extra token.

Budget: embedding 30522*128 ≈ 3.9M + 2 layers (~0.2M) + proj ≈ 4.0M.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _TOKENIZER


class TinyLanguageEncoder(nn.Module):
    """Instruction string(s) -> one (B, out_dim) conditioning vector. ~4M params."""

    def __init__(self, out_dim: int, d: int = 128, n_layers: int = 2, n_heads: int = 4,
                 max_len: int = 32, vocab_size: int = 30522):
        super().__init__()
        self.max_len = max_len
        self.tok_embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    @torch.no_grad()
    def _tokenize(self, texts: list[str], device):
        tok = _get_tokenizer()
        out = tok(texts, padding="max_length", truncation=True,
                  max_length=self.max_len, return_tensors="pt")
        return out["input_ids"].to(device), out["attention_mask"].to(device)

    def forward(self, texts: list[str], device) -> Tensor:
        ids, mask = self._tokenize(texts, device)            # (B, L)
        x = self.tok_embed(ids)                              # (B, L, d)
        pos = self.pos_embed(torch.arange(x.shape[1], device=device))[None]
        x = x + pos
        # transformer wants src_key_padding_mask True where padded
        x = self.encoder(x, src_key_padding_mask=(mask == 0))
        # masked mean-pool over real tokens
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)    # (B, d)
        return self.norm(self.proj(pooled))                  # (B, out_dim)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
