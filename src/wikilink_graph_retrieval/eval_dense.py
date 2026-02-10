from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wikilink_graph_retrieval.metrics import (
    compute_ranking_metrics,
    expected_calibration_error,
    ranks_from_topk,
)
from wikilink_graph_retrieval.models.dense_retriever import DenseRetriever, DenseRetrieverConfig
from wikilink_graph_retrieval.models.fusion import FusionConfig
from wikilink_graph_retrieval.models.text_encoder import TextEncoderConfig
from wikilink_graph_retrieval.retrieval import batched_topk_dot, softmax_confidence_top1
from wikilink_graph_retrieval.tokenizer import HashTokenizer
from wikilink_graph_retrieval.utils import resolve_device


@torch.inference_mode()
def eval_dense_retriever(
    *,
    name: str,
    ckpt_dir: Path,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    device: str,
    vocab_size: int,
    max_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    embed_dim: int,
    l2_normalize: bool,
    use_graph: bool,
    fusion_method: str,
    graph_emb_path: Path | None,
    k: int = 100,
    doc_batch_size: int = 256,
    query_batch_size: int = 256,
    doc_chunk_size: int = 8192,
) -> dict:
    tdev = resolve_device(device)
    tokenizer = HashTokenizer(vocab_size=vocab_size, max_len=max_len)

    cfg = DenseRetrieverConfig(
        text=TextEncoderConfig(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            embed_dim=embed_dim,
        ),
        l2_normalize=l2_normalize,
        fusion=FusionConfig(method=fusion_method, embed_dim=embed_dim, graph_dim=embed_dim),
    )
    model = DenseRetriever(cfg).to(tdev)
    sd = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    graph_emb = None
    if use_graph:
        if graph_emb_path is None:
            raise ValueError("use_graph=True requires graph_emb_path")
        graph_emb = torch.load(graph_emb_path, map_location="cpu").to(tdev)

    doc_ids = corpus["doc_id"].astype(np.int64).to_numpy()
    doc_texts = corpus["text"].astype(str).tolist()

    # Embed docs
    t0 = time.perf_counter()
    doc_embs = []
    for i in range(0, len(doc_texts), doc_batch_size):
        batch_text = doc_texts[i : i + doc_batch_size]
        tb = tokenizer.batch_encode(batch_text)
        input_ids = torch.from_numpy(tb.input_ids).to(tdev)
        attn = torch.from_numpy(tb.attention_mask.astype(np.int64)).to(tdev)
        if graph_emb is None:
            z = model.encode_docs(input_ids, attn, None)
        else:
            # doc_id is aligned with row index in corpus for toy + our extractors.
            gids = torch.from_numpy(doc_ids[i : i + len(batch_text)]).to(tdev)
            z = model.encode_docs(input_ids, attn, graph_emb[gids])
        doc_embs.append(z.detach().cpu())
    doc_emb = torch.cat(doc_embs, dim=0)
    doc_embed_s = time.perf_counter() - t0

    # Embed queries
    q_texts = queries["query_text"].astype(str).tolist()
    t1 = time.perf_counter()
    q_embs = []
    for i in range(0, len(q_texts), query_batch_size):
        batch_text = q_texts[i : i + query_batch_size]
        tb = tokenizer.batch_encode(batch_text)
        input_ids = torch.from_numpy(tb.input_ids).to(tdev)
        attn = torch.from_numpy(tb.attention_mask.astype(np.int64)).to(tdev)
        z = model.encode_queries(input_ids, attn)
        q_embs.append(z.detach().cpu())
    q_emb = torch.cat(q_embs, dim=0)
    query_embed_s = time.perf_counter() - t1

    # Search (exact)
    t2 = time.perf_counter()
    scores, idx = batched_topk_dot(q_emb.to(tdev), doc_emb.to(tdev), k=k, doc_chunk_size=doc_chunk_size)
    scores = scores.detach().cpu().numpy()
    idx = idx.detach().cpu().numpy()
    search_s = time.perf_counter() - t2

    topk_doc_ids = doc_ids[idx]
    true_doc_ids = queries["doc_id"].astype(np.int64).to_numpy()
    ranks = ranks_from_topk(topk_doc_ids, true_doc_ids)
    m = compute_ranking_metrics(ranks)

    # Calibration: treat softmax over top-k as an approximate confidence.
    top1_correct = (ranks == 1)
    top1_conf = softmax_confidence_top1(scores_topk=scores, temperature=1.0)
    ece = expected_calibration_error(top1_correct.astype(np.int64), top1_conf, n_bins=15)

    return {
        "k": int(k),
        "metrics": {
            "recall@1": m.recall_at_1,
            "recall@10": m.recall_at_10,
            "recall@100": m.recall_at_100,
            "mrr@100": m.mrr_at_100,
            "ndcg@100": m.ndcg_at_100,
            "ece_top1": float(ece),
        },
        "throughput": {
            "num_docs": int(len(doc_ids)),
            "num_queries": int(len(q_texts)),
            "doc_embed_seconds": float(doc_embed_s),
            "query_embed_seconds": float(query_embed_s),
            "search_seconds": float(search_s),
            "docs_per_sec": float(len(doc_ids) / max(1e-9, doc_embed_s)),
            "queries_per_sec": float(len(q_texts) / max(1e-9, query_embed_s)),
        },
    }

