from __future__ import annotations

from pathlib import Path

from wikilink_graph_retrieval.data.download import download_with_md5
from wikilink_graph_retrieval.data.toy import build_toy_dataset
from wikilink_graph_retrieval.data.wikidump_extract import ExtractConfig, extract_wikidump
from wikilink_graph_retrieval.pipeline._common import load_project_config, parse_args
from wikilink_graph_retrieval.utils import save_json


def main() -> None:
    args = parse_args("Prepare dataset (toy or wikidump) into processed parquet files.")
    cfg = load_project_config(args.config)

    if cfg.data.source == "toy":
        t = cfg.data.toy
        assert t is not None
        build_toy_dataset(
            cfg.paths.processed_dir,
            seed=cfg.seed,
            num_docs=t.num_docs,
            num_edges=t.num_edges,
            train_links=t.train_links,
            val_links=t.val_links,
            test_links=t.test_links,
            context_window_tokens=t.context_window_tokens,
        )
    else:
        w = cfg.data.wikidump
        assert w is not None
        pages = download_with_md5(w.urls.pages_xml, cfg.paths.raw_dir)
        pagelinks = download_with_md5(w.urls.pagelinks_sql, cfg.paths.raw_dir)
        extract_wikidump(
            ExtractConfig(
                pages_xml_bz2=pages,
                pagelinks_sql_gz=pagelinks,
                out_dir=cfg.paths.processed_dir,
                top_targets=w.top_targets,
                train_links=w.train_links,
                val_links=w.val_links,
                test_links=w.test_links,
                context_window_tokens=w.context_window_tokens,
                split=w.split,
            )
        )

    save_json(cfg.paths.processed_dir / "config_snapshot.json", {"config": str(args.config)})


if __name__ == "__main__":
    main()

