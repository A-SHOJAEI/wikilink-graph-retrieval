from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RetrievalStats:
    num_docs: int
    num_queries: int
    doc_embed_seconds: float
    query_embed_seconds: float
    search_seconds: float
    docs_per_sec: float
    queries_per_sec: float


def batched_topk_dot(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    k: int,
    doc_chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact top-k dot-product retrieval with doc-side chunking to control memory.
    Returns (topk_scores, topk_indices) where indices are row indices into doc_emb.
    """
    q = query_emb
    d = doc_emb
    assert q.ndim == 2 and d.ndim == 2
    assert q.shape[1] == d.shape[1]

    top_scores = None
    top_indices = None

    n_docs = d.shape[0]
    for start in range(0, n_docs, doc_chunk_size):
        end = min(start + doc_chunk_size, n_docs)
        chunk = d[start:end]  # [C, dim]
        scores = q @ chunk.T  # [Q, C]
        chunk_scores, chunk_idx = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
        chunk_idx = chunk_idx + start

        if top_scores is None:
            top_scores = chunk_scores
            top_indices = chunk_idx
        else:
            merged_scores = torch.cat([top_scores, chunk_scores], dim=1)
            merged_idx = torch.cat([top_indices, chunk_idx], dim=1)
            new_scores, new_pos = torch.topk(merged_scores, k=min(k, merged_scores.shape[1]), dim=1)
            new_idx = torch.gather(merged_idx, 1, new_pos)
            top_scores, top_indices = new_scores, new_idx

    assert top_scores is not None and top_indices is not None
    return top_scores, top_indices


def softmax_confidence_top1(scores_topk: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Approximate calibrated confidence by softmax over retrieved candidate scores.
    """
    s = scores_topk.astype(np.float64) / float(temperature)
    s = s - np.max(s, axis=1, keepdims=True)
    ex = np.exp(s)
    p = ex / np.sum(ex, axis=1, keepdims=True)
    return p[:, 0]


def timeit(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start)

