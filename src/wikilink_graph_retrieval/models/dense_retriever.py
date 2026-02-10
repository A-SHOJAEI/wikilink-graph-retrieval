from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from wikilink_graph_retrieval.models.fusion import FusionConfig, GraphFusion
from wikilink_graph_retrieval.models.text_encoder import TextEncoderConfig, TransformerTextEncoder, l2_normalize


@dataclass(frozen=True)
class DenseRetrieverConfig:
    text: TextEncoderConfig
    l2_normalize: bool
    fusion: FusionConfig


class DenseRetriever(nn.Module):
    """
    Dual-encoder retriever:
      query embedding = f_q(context_text)
      doc embedding   = fuse(f_d(doc_text), node_embedding(doc_id))
    """

    def __init__(self, cfg: DenseRetrieverConfig):
        super().__init__()
        self.cfg = cfg
        self.query_encoder = TransformerTextEncoder(cfg.text)
        self.doc_encoder = TransformerTextEncoder(cfg.text)
        self.fusion = GraphFusion(cfg.fusion)

    def encode_queries(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.query_encoder(input_ids, attention_mask)
        return l2_normalize(z) if self.cfg.l2_normalize else z

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_graph: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zt = self.doc_encoder(input_ids, attention_mask)
        z = self.fusion(zt, doc_graph)
        return l2_normalize(z) if self.cfg.l2_normalize else z

    def similarity(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return q @ d.T


def inbatch_contrastive_loss(sim: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    sim: [B, B] similarity matrix where diagonal are positives.
    """
    logits = sim / float(temperature)
    target = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, target)

