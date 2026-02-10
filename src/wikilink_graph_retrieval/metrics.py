from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    mrr_at_100: float
    ndcg_at_100: float
    ece_top1: float | None = None


def _recall_at_k(ranks: np.ndarray, k: int) -> float:
    # rank is 1-based; rank==0 indicates "not found in topK" upstream (shouldn't happen here)
    return float(np.mean((ranks >= 1) & (ranks <= k)))


def _mrr_at_k(ranks: np.ndarray, k: int) -> float:
    rr = np.where((ranks >= 1) & (ranks <= k), 1.0 / ranks, 0.0)
    return float(np.mean(rr))


def _ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    # Single relevant doc: DCG = 1/log2(rank+1), IDCG = 1.
    dcg = np.where((ranks >= 1) & (ranks <= k), 1.0 / np.log2(ranks + 1.0), 0.0)
    return float(np.mean(dcg))


def compute_ranking_metrics(ranks: np.ndarray) -> RankingMetrics:
    """
    ranks: 1-based rank positions of the correct target among retrieved candidates.
    """
    ranks = ranks.astype(np.int64)
    return RankingMetrics(
        recall_at_1=_recall_at_k(ranks, 1),
        recall_at_10=_recall_at_k(ranks, 10),
        recall_at_100=_recall_at_k(ranks, 100),
        mrr_at_100=_mrr_at_k(ranks, 100),
        ndcg_at_100=_ndcg_at_k(ranks, 100),
    )


def expected_calibration_error(
    correct: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """
    ECE for top-1 predictions.

    correct: bool array indicating whether top-1 was correct.
    probs: predicted probability/confidence for top-1 (0..1).
    """
    correct = correct.astype(bool)
    probs = probs.astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(correct[m]))
        conf = float(np.mean(probs[m]))
        ece += (float(np.mean(m)) * abs(acc - conf))
    return float(ece)


def ranks_from_topk(doc_ids_topk: np.ndarray, true_doc_ids: np.ndarray) -> np.ndarray:
    """
    doc_ids_topk: [num_queries, K] retrieved doc ids in rank order.
    true_doc_ids: [num_queries] true doc ids.
    Returns 1-based ranks; if not found, rank=K+1.
    """
    n, k = doc_ids_topk.shape
    out = np.empty((n,), dtype=np.int64)
    for i in range(n):
        hits = np.where(doc_ids_topk[i] == true_doc_ids[i])[0]
        out[i] = int(hits[0] + 1) if hits.size else int(k + 1)
    return out

