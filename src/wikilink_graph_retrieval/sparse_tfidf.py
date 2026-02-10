from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from wikilink_graph_retrieval.metrics import compute_ranking_metrics, ranks_from_topk
from wikilink_graph_retrieval.utils import ensure_dir, save_json


@dataclass(frozen=True)
class TfidfIndexPaths:
    vectorizer_pkl: Path
    doc_matrix_npz: Path
    doc_ids_npy: Path
    meta_json: Path


def tfidf_paths(index_dir: Path) -> TfidfIndexPaths:
    return TfidfIndexPaths(
        vectorizer_pkl=index_dir / "vectorizer.pkl",
        doc_matrix_npz=index_dir / "doc_tfidf.npz",
        doc_ids_npy=index_dir / "doc_ids.npy",
        meta_json=index_dir / "meta.json",
    )


def build_tfidf_index(corpus: pd.DataFrame, *, index_dir: Path, max_features: int) -> TfidfIndexPaths:
    ensure_dir(index_dir)
    p = tfidf_paths(index_dir)

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )
    X = vec.fit_transform(corpus["text"].astype(str).tolist())
    doc_ids = corpus["doc_id"].astype(np.int64).to_numpy()

    with p.vectorizer_pkl.open("wb") as f:
        pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    sp.save_npz(p.doc_matrix_npz, X)
    np.save(p.doc_ids_npy, doc_ids)
    save_json(
        p.meta_json,
        {
            "kind": "tfidf",
            "max_features": int(max_features),
            "num_docs": int(X.shape[0]),
            "num_features": int(X.shape[1]),
        },
    )
    return p


def load_tfidf_index(index_dir: Path) -> tuple[TfidfVectorizer, sp.csr_matrix, np.ndarray]:
    p = tfidf_paths(index_dir)
    with p.vectorizer_pkl.open("rb") as f:
        vec: TfidfVectorizer = pickle.load(f)
    X = sp.load_npz(p.doc_matrix_npz).tocsr()
    doc_ids = np.load(p.doc_ids_npy)
    return vec, X, doc_ids


def eval_tfidf(
    *,
    index_dir: Path,
    queries: pd.DataFrame,
    k: int = 100,
) -> dict:
    vec, X, doc_ids = load_tfidf_index(index_dir)
    Q = vec.transform(queries["query_text"].astype(str).tolist())
    S = (Q @ X.T).tocsr()  # [nq, nd] sparse

    nq = S.shape[0]
    k_eff = int(k)
    topk_doc_ids = np.full((nq, k_eff), -1, dtype=np.int64)
    for i in range(nq):
        row = S.getrow(i)
        if row.nnz == 0:
            continue
        idx = row.indices
        data = row.data
        if row.nnz <= k_eff:
            order = np.argsort(-data)
            sel = idx[order]
        else:
            part = np.argpartition(-data, kth=k_eff - 1)[:k_eff]
            order = np.argsort(-data[part])
            sel = idx[part][order]
        topk_doc_ids[i, : sel.shape[0]] = doc_ids[sel]

    true_doc_ids = queries["doc_id"].astype(np.int64).to_numpy()
    ranks = ranks_from_topk(topk_doc_ids, true_doc_ids)
    m = compute_ranking_metrics(ranks)
    return {
        "k": int(k),
        "metrics": {
            "recall@1": m.recall_at_1,
            "recall@10": m.recall_at_10,
            "recall@100": m.recall_at_100,
            "mrr@100": m.mrr_at_100,
            "ndcg@100": m.ndcg_at_100,
        },
        "debug": {
            "num_queries": int(len(queries)),
            "num_docs": int(X.shape[0]),
        },
    }
