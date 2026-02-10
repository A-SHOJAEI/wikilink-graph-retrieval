from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TextEncoderConfig:
    vocab_size: int
    max_len: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    embed_dim: int
    pad_id: int = 0


class TransformerTextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.proj = nn.Linear(cfg.d_model, cfg.embed_dim)
        self.out_ln = nn.LayerNorm(cfg.embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L]
        attention_mask: [B, L] with 1 for real tokens, 0 for padding
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        # Transformer uses src_key_padding_mask where True indicates padding positions.
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls = x[:, 0, :]  # [B, d_model]
        z = self.out_ln(self.proj(cls))
        return z


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)

