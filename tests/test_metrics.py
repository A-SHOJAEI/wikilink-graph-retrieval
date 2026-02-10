import numpy as np

from wikilink_graph_retrieval.metrics import (
    compute_ranking_metrics,
    expected_calibration_error,
    ranks_from_topk,
)


def test_ranks_from_topk_found_and_not_found():
    topk = np.array([[5, 2, 9], [1, 2, 3]], dtype=np.int64)
    true = np.array([2, 7], dtype=np.int64)
    ranks = ranks_from_topk(topk, true)
    assert ranks.tolist() == [2, 4]  # not found => K+1


def test_compute_ranking_metrics_simple():
    ranks = np.array([1, 2, 101], dtype=np.int64)
    m = compute_ranking_metrics(ranks)
    assert m.recall_at_1 == 1 / 3
    assert m.recall_at_10 == 2 / 3
    assert m.recall_at_100 == 2 / 3
    assert m.mrr_at_100 == (1.0 + 0.5 + 0.0) / 3


def test_ece_zero_when_perfect_calibration():
    correct = np.array([1, 0, 1, 0], dtype=np.int64)
    probs = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    ece = expected_calibration_error(correct, probs, n_bins=4)
    assert ece == 0.0

