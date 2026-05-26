"""NV-Retriever TopK-PercPos hard-negative mining.

Verbatim verified (2026-05-18 cross-validation, Nemotron paper §3.3.2):
    "We set the threshold as 0.95, meaning we select the K most relevant
    negative samples whose similarity to the query is less than 95% of
    the query-positive similarity score."

Algorithm (NV-Retriever arXiv:2407.15831, §3.4.3):
    1. Warm teacher retriever encodes query + positive doc.
    2. Retrieve top-N candidates from corpus (exclude positive).
    3. Filter: keep candidate iff score(q, cand) < 0.95 * score(q, pos).
    4. Return top-K filtered (K=2 in paper, K=4 also viable).

KARARLAR S28: NV-Retriever positive-aware, threshold 0.95 default;
    threshold=0.90 for legal/finance subsets (mevzuat duplicate paragraphs +
    KAP-style template repetition cause score/pos > 0.95 noise).

Teacher selection ladder:
    - Stage-1 mining: Ops-Colqwen3-4B zero-shot (warm, Qwen3-VL base)
    - Stage-2 mining: ColTurk-VDR Stage-1 checkpoint (iterative, TR in-domain)

Compute: ~25 min mining iter on A100 (30K queries × 100K corpus pages),
    with FAISS-IVFPQ vector index.

Refs:
- arXiv:2407.15831 NV-Retriever (Moreira et al., 2024)
- arXiv:2602.03992 Nemotron ColEmbed V2 §3.3.2 (verbatim threshold 0.95)
- colpali-engine/data/corpus_qry_collator.py (mined neg integration point)
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG = logging.getLogger("mine_hard_negatives")

DEFAULT_THRESHOLD = 0.95
LEGAL_FINANCE_THRESHOLD = 0.90
DEFAULT_K_NEG = 4
DEFAULT_CANDIDATE_POOL = 100


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_text: str
    positive_page_id: str


@dataclass(frozen=True)
class CorpusEntry:
    page_id: str
    embedding: np.ndarray  # (D,) or (T, D) for late-interaction


@dataclass(frozen=True)
class MinedNegatives:
    query_id: str
    positive_page_id: str
    positive_score: float
    negative_page_ids: list[str]
    negative_scores: list[float]


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Late-interaction MaxSim: sum over query tokens of max over doc tokens.

    query_emb shape: (T_q, D); doc_emb shape: (T_d, D). Both L2-normalized.
    """
    if query_emb.ndim == 1 and doc_emb.ndim == 1:
        return float(np.dot(query_emb, doc_emb))
    sims = query_emb @ doc_emb.T
    return float(sims.max(axis=1).sum())


def topk_percpos_mine(
    query: QueryRecord,
    query_emb: np.ndarray,
    positive_emb: np.ndarray,
    candidate_ids: list[str],
    candidate_embs: list[np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
    k_neg: int = DEFAULT_K_NEG,
) -> MinedNegatives:
    """Core TopK-PercPos algorithm — Prompt C verbatim pseudocode cross-validated.

    Filters false-negatives whose query-similarity is too close to positive's.
    """
    pos_score = maxsim_score(query_emb, positive_emb)
    max_neg_score = threshold * pos_score

    scored: list[tuple[str, float]] = []
    for cand_id, cand_emb in zip(candidate_ids, candidate_embs):
        if cand_id == query.positive_page_id:
            continue
        score = maxsim_score(query_emb, cand_emb)
        if score < max_neg_score:
            scored.append((cand_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    head = scored[:k_neg]
    return MinedNegatives(
        query_id=query.query_id,
        positive_page_id=query.positive_page_id,
        positive_score=pos_score,
        negative_page_ids=[pid for pid, _ in head],
        negative_scores=[s for _, s in head],
    )


def _threshold_for_domain(domain_tag: str | None) -> float:
    if domain_tag and domain_tag.lower() in {"legal", "finance", "mevzuat", "kap"}:
        return LEGAL_FINANCE_THRESHOLD
    return DEFAULT_THRESHOLD


def mine_corpus(
    queries: Iterable[QueryRecord],
    query_embeddings: dict[str, np.ndarray],
    corpus_embeddings: dict[str, np.ndarray],
    retrieve_topn_fn: Any,  # Callable[[np.ndarray, int], list[tuple[str, float]]]
    *,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    k_neg: int = DEFAULT_K_NEG,
    domain_tags: dict[str, str] | None = None,
) -> list[MinedNegatives]:
    """Mine hard negatives for every query in the iterable.

    `retrieve_topn_fn(query_emb, n)` must return list[(page_id, score)] from
    a vector index (FAISS, ScaNN, Qdrant). Caller owns indexing.
    """
    results: list[MinedNegatives] = []
    for q in queries:
        q_emb = query_embeddings[q.query_id]
        pos_emb = corpus_embeddings[q.positive_page_id]
        candidates = retrieve_topn_fn(q_emb, candidate_pool)
        cand_ids = [c[0] for c in candidates]
        cand_embs = [corpus_embeddings[c[0]] for c in candidates]
        threshold = _threshold_for_domain((domain_tags or {}).get(q.query_id))
        mined = topk_percpos_mine(q, q_emb, pos_emb, cand_ids, cand_embs, threshold, k_neg)
        results.append(mined)
    return results


def write_jsonl(records: list[MinedNegatives], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(
                json.dumps(
                    {
                        "query_id": r.query_id,
                        "positive_page_id": r.positive_page_id,
                        "positive_score": r.positive_score,
                        "hard_negatives": [
                            {"page_id": pid, "score": s}
                            for pid, s in zip(r.negative_page_ids, r.negative_scores)
                        ],
                    }
                )
                + "\n"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NV-Retriever TopK-PercPos hard-neg mining")
    parser.add_argument("--queries", type=Path, required=True, help="JSONL: {query_id, query, positive_page_id}")
    parser.add_argument("--query-embeddings", type=Path, required=True, help=".npz with query_id -> embedding")
    parser.add_argument("--corpus-embeddings", type=Path, required=True, help=".npz with page_id -> embedding")
    parser.add_argument("--faiss-index", type=Path, required=True, help="FAISS index of corpus embeddings")
    parser.add_argument("--domain-tags", type=Path, default=None, help="JSON: query_id -> domain tag")
    parser.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--k-neg", type=int, default=DEFAULT_K_NEG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import faiss  # type: ignore[import-not-found]

    q_embs_npz = np.load(args.query_embeddings)
    c_embs_npz = np.load(args.corpus_embeddings)
    query_embeddings = {k: q_embs_npz[k] for k in q_embs_npz.files}
    corpus_embeddings = {k: c_embs_npz[k] for k in c_embs_npz.files}
    index = faiss.read_index(str(args.faiss_index))
    corpus_ids = list(corpus_embeddings.keys())

    def retrieve_topn(q_emb: np.ndarray, n: int) -> list[tuple[str, float]]:
        emb = q_emb.mean(axis=0) if q_emb.ndim > 1 else q_emb
        scores, idxs = index.search(emb.reshape(1, -1).astype("float32"), n)
        return [(corpus_ids[i], float(scores[0][rank])) for rank, i in enumerate(idxs[0]) if i >= 0]

    queries = [
        QueryRecord(query_id=row["query_id"], query_text=row["query"], positive_page_id=row["positive_page_id"])
        for row in (json.loads(l) for l in args.queries.read_text().splitlines() if l.strip())
    ]
    domain_tags = json.loads(args.domain_tags.read_text()) if args.domain_tags else None

    LOG.info("Mining %d queries against corpus of %d entries", len(queries), len(corpus_embeddings))
    mined = mine_corpus(
        queries,
        query_embeddings,
        corpus_embeddings,
        retrieve_topn,
        candidate_pool=args.candidate_pool,
        k_neg=args.k_neg,
        domain_tags=domain_tags,
    )
    write_jsonl(mined, args.output)
    LOG.info("Wrote %d mined negative records to %s", len(mined), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
