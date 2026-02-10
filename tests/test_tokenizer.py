from wikilink_graph_retrieval.tokenizer import HashTokenizer


def test_hash_tokenizer_deterministic():
    tok = HashTokenizer(vocab_size=8192, max_len=16)
    a = tok.batch_encode(["Hello world", "Hello world"])
    assert (a.input_ids[0] == a.input_ids[1]).all()


def test_hash_tokenizer_has_cls_and_padding():
    tok = HashTokenizer(vocab_size=8192, max_len=8)
    b = tok.batch_encode(["hi"])
    assert b.input_ids.shape == (1, 8)
    assert b.input_ids[0, 0] == tok.cls_id
    assert bool(b.attention_mask[0, 0]) is True
    assert b.attention_mask[0, -1] in (True, False)
