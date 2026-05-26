"""ViDoRe V3 contamination prevention 5-layer scan (KARARLAR S41).

Pipeline:
- L1 doc-id exact match (V3 8 public corpus split doc_id union, set intersect=0 mandatory)
- L2 filename/title 5+ word substring match (normalize lowercase)
- L3 13-gram MinHash LSH (datasketch num_perm=128 threshold=0.85, ~30 dk CPU)
- L4 TF-IDF char-ngram cosine > 0.85 (boilerplate stop-list mask)
- L5 manual 2500 pair spot-check (50x50 random — output a JSON for human review)

Reject threshold: %0.5 flagged max; > %1 → threshold 0.85→0.75 + retry.

Refs:
- KARARLAR S41 (DR Round C P2 3-agent UNANIMOUS)
- ViDoRe V3 8 public subtasks: finance_en, finance_fr, hr, industrial, computer_science,
  pharmaceuticals, physics, energy (plus 2 private: telecom, nuclear — MTEB only)

Compute budget: ~30 dk + 6GB RAM. A100 INDEPENDENT (CPU job).

Usage:
    python scripts/data/v3_contamination_scan.py \\
        --train-corpus data/stage1_corpus/manifest.jsonl \\
        --output data/stage1_corpus/contamination_report.json \\
        --filter-output data/stage1_corpus/manifest_filtered.jsonl \\
        --threshold-jaccard 0.85
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

V3_PUBLIC_SUBTASKS = (
    "vidore/vidore-finance-en-test",
    "vidore/vidore-finance-fr-test",
    "vidore/vidore-hr-test",
    "vidore/vidore-industrial-test",
    "vidore/vidore-computer-science-test",
    "vidore/vidore-pharmaceuticals-test",
    "vidore/vidore-physics-test",
    "vidore/vidore-energy-test",
)

NGRAM_SIZE = 13          # S41 verbatim
LSH_NUM_PERM = 128       # S41 verbatim (datasketch default)
DEFAULT_JACCARD = 0.85   # S41 verbatim
DEFAULT_TFIDF = 0.85     # S41 verbatim
MIN_FILENAME_WORDS = 5   # S41 verbatim


def _load_v3_corpus_doc_ids() -> set[str]:
    """V3 8 public subtask corpus split'lerinden tüm doc_id union."""
    from datasets import load_dataset

    all_ids: set[str] = set()
    for subtask in V3_PUBLIC_SUBTASKS:
        try:
            ds = load_dataset(subtask, "corpus", split="test", streaming=False)
            ids = {str(row["_id"]) for row in ds if "_id" in row}
            all_ids.update(ids)
            logger.info("L1: loaded %d doc_ids from %s", len(ids), subtask)
        except Exception as exc:
            logger.warning("L1: %s corpus load failed (%s) — skipping", subtask, exc)
    return all_ids


def _load_v3_filenames_titles() -> set[str]:
    """V3 corpus filename + title fields (lowercased) — L2 substring match için."""
    from datasets import load_dataset

    items: set[str] = set()
    for subtask in V3_PUBLIC_SUBTASKS:
        try:
            ds = load_dataset(subtask, "corpus", split="test", streaming=False)
            for row in ds:
                if "title" in row and row["title"]:
                    items.add(row["title"].lower().strip())
                if "filename" in row and row["filename"]:
                    items.add(row["filename"].lower().strip())
        except Exception as exc:
            logger.warning("L2: %s metadata load failed (%s)", subtask, exc)
    return items


def _ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    """Tokenize lowercase text into n-grams."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _layer1_doc_id(train_records: list[dict], v3_ids: set[str]) -> dict[str, set[str]]:
    """L1: exact doc_id match across train ∩ V3."""
    hits: dict[str, set[str]] = defaultdict(set)
    for rec in train_records:
        train_id = str(rec.get("doc_id", ""))
        if train_id and train_id in v3_ids:
            hits[train_id].add(rec.get("source", "unknown"))
    return hits


def _layer2_filename(
    train_records: list[dict], v3_titles: set[str]
) -> dict[str, list[str]]:
    """L2: filename/title 5+ word substring match (normalize lowercase)."""
    hits: dict[str, list[str]] = defaultdict(list)
    for rec in train_records:
        for field in ("filename", "title"):
            value = (rec.get(field, "") or "").lower().strip()
            if not value:
                continue
            if len(value.split()) < MIN_FILENAME_WORDS:
                continue
            if value in v3_titles:
                hits[str(rec.get("doc_id", ""))].append(value)
                continue
            # Substring (V3 title in train title or vice versa)
            for v3_title in v3_titles:
                if len(v3_title.split()) < MIN_FILENAME_WORDS:
                    continue
                if v3_title in value or value in v3_title:
                    hits[str(rec.get("doc_id", ""))].append(v3_title)
                    break
    return hits


def _layer3_minhash_lsh(
    train_records: list[dict], v3_corpus_texts: list[tuple[str, str]], threshold: float
) -> dict[str, list[tuple[str, float]]]:
    """L3: 13-gram MinHash LSH (datasketch). Returns train_doc_id → [(v3_doc_id, jaccard), ...]."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.error("L3: datasketch not installed. Run: pip install datasketch")
        return {}

    lsh = MinHashLSH(threshold=threshold, num_perm=LSH_NUM_PERM)
    v3_minhashes: dict[str, MinHash] = {}
    for v3_id, v3_text in v3_corpus_texts:
        mh = MinHash(num_perm=LSH_NUM_PERM)
        for ng in _ngrams(v3_text):
            mh.update(ng.encode("utf-8"))
        v3_minhashes[v3_id] = mh
        lsh.insert(v3_id, mh)

    hits: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for rec in train_records:
        text = rec.get("text", "")
        if not text or len(text) < 100:
            continue
        train_id = str(rec.get("doc_id", ""))
        mh = MinHash(num_perm=LSH_NUM_PERM)
        for ng in _ngrams(text):
            mh.update(ng.encode("utf-8"))
        candidates = lsh.query(mh)
        for v3_id in candidates:
            jacc = mh.jaccard(v3_minhashes[v3_id])
            if jacc >= threshold:
                hits[train_id].append((v3_id, float(jacc)))
    return hits


def _layer4_tfidf(
    train_records: list[dict],
    v3_corpus_texts: list[tuple[str, str]],
    threshold: float,
    boilerplate_stoplist: set[str] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """L4: TF-IDF char-ngram cosine > threshold + boilerplate stop-list mask."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        logger.error("L4: scikit-learn not installed. Run: pip install scikit-learn")
        return {}

    stop = boilerplate_stoplist or set()

    def _clean(text: str) -> str:
        if not stop:
            return text
        words = text.split()
        return " ".join(w for w in words if w.lower() not in stop)

    train_texts = [_clean(rec.get("text", "")) for rec in train_records]
    train_ids = [str(rec.get("doc_id", "")) for rec in train_records]
    v3_ids = [v3_id for v3_id, _ in v3_corpus_texts]
    v3_texts = [_clean(t) for _, t in v3_corpus_texts]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
    all_texts = train_texts + v3_texts
    matrix = vec.fit_transform(all_texts)
    train_matrix = matrix[: len(train_texts)]
    v3_matrix = matrix[len(train_texts) :]

    hits: dict[str, list[tuple[str, float]]] = defaultdict(list)
    sim = cosine_similarity(train_matrix, v3_matrix)
    for i, row in enumerate(sim):
        for j, score in enumerate(row):
            if score >= threshold:
                hits[train_ids[i]].append((v3_ids[j], float(score)))
    return hits


def _layer5_spot_check_sample(
    train_records: list[dict], v3_corpus_texts: list[tuple[str, str]], n: int = 2500
) -> list[dict[str, str]]:
    """L5: 2500-pair random sample for manual review."""
    import random

    rng = random.Random(42)
    pairs: list[dict[str, str]] = []
    train_sample = rng.sample(train_records, min(50, len(train_records)))
    v3_sample = rng.sample(v3_corpus_texts, min(50, len(v3_corpus_texts)))
    for tr in train_sample:
        for v3_id, v3_text in v3_sample:
            pairs.append(
                {
                    "train_doc_id": str(tr.get("doc_id", "")),
                    "train_source": tr.get("source", ""),
                    "train_text_excerpt": (tr.get("text", "") or "")[:500],
                    "v3_doc_id": v3_id,
                    "v3_text_excerpt": v3_text[:500],
                }
            )
            if len(pairs) >= n:
                return pairs
    return pairs


def _load_train_corpus(manifest_path: Path) -> list[dict]:
    """Manifest JSONL: {doc_id, source, text, filename?, title?} per line."""
    records: list[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("manifest parse skip: %s", exc)
    return records


def _load_v3_texts() -> list[tuple[str, str]]:
    """V3 corpus (doc_id, text) tuples for L3+L4."""
    from datasets import load_dataset

    items: list[tuple[str, str]] = []
    for subtask in V3_PUBLIC_SUBTASKS:
        try:
            ds = load_dataset(subtask, "corpus", split="test", streaming=False)
            for row in ds:
                doc_id = str(row.get("_id", ""))
                text = row.get("text", "") or row.get("content", "")
                if doc_id and text:
                    items.append((doc_id, text))
        except Exception as exc:
            logger.warning("V3 text load %s: %s", subtask, exc)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 contamination 5-layer scan (S41)")
    parser.add_argument("--train-corpus", type=Path, required=True,
                        help="JSONL manifest: {doc_id, source, text, filename?, title?}")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON report (per-layer flagged counts + samples)")
    parser.add_argument("--filter-output", type=Path,
                        help="Optional JSONL with flagged records removed")
    parser.add_argument("--threshold-jaccard", type=float, default=DEFAULT_JACCARD)
    parser.add_argument("--threshold-tfidf", type=float, default=DEFAULT_TFIDF)
    parser.add_argument("--skip-l4-tfidf", action="store_true",
                        help="L4 TF-IDF compute-heavy; skip on large corpora")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Loading train corpus manifest: %s", args.train_corpus)
    train_records = _load_train_corpus(args.train_corpus)
    logger.info("Train corpus: %d records", len(train_records))

    logger.info("Loading V3 corpus (8 public subtasks)…")
    v3_ids = _load_v3_corpus_doc_ids()
    v3_titles = _load_v3_filenames_titles()
    v3_texts = _load_v3_texts()
    logger.info("V3 corpus: %d doc_ids, %d titles, %d full-text records",
                len(v3_ids), len(v3_titles), len(v3_texts))

    report: dict[str, Any] = {
        "config": {
            "train_corpus_size": len(train_records),
            "v3_corpus_size": len(v3_texts),
            "threshold_jaccard": args.threshold_jaccard,
            "threshold_tfidf": args.threshold_tfidf,
            "ngram_size": NGRAM_SIZE,
            "lsh_num_perm": LSH_NUM_PERM,
        },
        "layers": {},
    }

    logger.info("L1: doc-id exact match")
    l1 = _layer1_doc_id(train_records, v3_ids)
    report["layers"]["L1_doc_id"] = {
        "flagged_count": len(l1),
        "samples": {k: list(v) for k, v in list(l1.items())[:20]},
    }

    logger.info("L2: filename/title 5+ word substring")
    l2 = _layer2_filename(train_records, v3_titles)
    report["layers"]["L2_filename"] = {
        "flagged_count": len(l2),
        "samples": {k: v[:5] for k, v in list(l2.items())[:20]},
    }

    logger.info("L3: 13-gram MinHash LSH (this is the slow ~30min step)")
    l3 = _layer3_minhash_lsh(train_records, v3_texts, args.threshold_jaccard)
    report["layers"]["L3_minhash"] = {
        "flagged_count": len(l3),
        "samples": {k: v[:5] for k, v in list(l3.items())[:20]},
    }

    if not args.skip_l4_tfidf:
        logger.info("L4: TF-IDF char-ngram cosine")
        l4 = _layer4_tfidf(train_records, v3_texts, args.threshold_tfidf)
        report["layers"]["L4_tfidf"] = {
            "flagged_count": len(l4),
            "samples": {k: v[:5] for k, v in list(l4.items())[:20]},
        }
    else:
        l4 = {}
        report["layers"]["L4_tfidf"] = {"skipped": True}

    logger.info("L5: 2500-pair manual review sample")
    l5_pairs = _layer5_spot_check_sample(train_records, v3_texts)
    report["layers"]["L5_manual"] = {
        "sample_count": len(l5_pairs),
        "pairs": l5_pairs[:50],   # full set saved to side file
    }
    l5_sidecar = args.output.with_suffix(".l5_pairs.json")
    l5_sidecar.write_text(json.dumps(l5_pairs, ensure_ascii=False, indent=2))
    logger.info("L5: full 2500 pairs → %s", l5_sidecar)

    # Aggregate: doc_ids flagged by any layer
    flagged_ids: set[str] = set()
    flagged_ids.update(l1.keys())
    flagged_ids.update(l2.keys())
    flagged_ids.update(l3.keys())
    flagged_ids.update(l4.keys())
    report["summary"] = {
        "total_flagged_unique": len(flagged_ids),
        "flagged_percent": (
            100.0 * len(flagged_ids) / len(train_records) if train_records else 0.0
        ),
        "reject_threshold_pct": 0.5,
        "verdict": "PASS"
        if 100.0 * len(flagged_ids) / max(1, len(train_records)) < 0.5
        else "REVIEW",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("Report → %s", args.output)

    if args.filter_output:
        clean = [r for r in train_records if str(r.get("doc_id", "")) not in flagged_ids]
        args.filter_output.parent.mkdir(parents=True, exist_ok=True)
        with args.filter_output.open("w", encoding="utf-8") as fh:
            for rec in clean:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(
            "Filtered manifest → %s (kept %d/%d)",
            args.filter_output, len(clean), len(train_records),
        )

    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
