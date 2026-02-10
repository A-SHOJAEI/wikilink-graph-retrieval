from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\\sA-Za-z0-9_]")


@dataclass(frozen=True)
class TokenBatch:
    input_ids: np.ndarray  # int64 [B, L]
    attention_mask: np.ndarray  # bool [B, L] (True for tokens to attend to)


class HashTokenizer:
    """
    A deterministic tokenizer that maps tokens to ids via stable hashing.

    This avoids external model downloads and keeps `make all` runnable offline.
    For large-scale experiments, consider swapping to a subword tokenizer.
    """

    def __init__(self, vocab_size: int, max_len: int, pad_id: int = 0, cls_id: int = 1):
        if vocab_size < 1024:
            raise ValueError("vocab_size too small; collisions will dominate")
        if max_len < 4:
            raise ValueError("max_len too small")
        if pad_id == cls_id:
            raise ValueError("pad_id and cls_id must differ")
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.cls_id = int(cls_id)

        # Token ids in [2, vocab_size-1] are hash buckets.
        self._bucket_mod = self.vocab_size - 2

    def _hash_token(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, byteorder="little", signed=False)
        return 2 + (v % self._bucket_mod)

    def tokenize(self, text: str) -> list[int]:
        toks = _TOKEN_RE.findall(text.lower())
        ids = [self.cls_id]
        for t in toks:
            ids.append(self._hash_token(t))
            if len(ids) >= self.max_len:
                break
        return ids

    def batch_encode(self, texts: list[str]) -> TokenBatch:
        b = len(texts)
        L = self.max_len
        input_ids = np.full((b, L), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((b, L), dtype=bool)
        for i, t in enumerate(texts):
            ids = self.tokenize(t)
            n = min(len(ids), L)
            input_ids[i, :n] = ids[:n]
            attention_mask[i, :n] = True
        return TokenBatch(input_ids=input_ids, attention_mask=attention_mask)
