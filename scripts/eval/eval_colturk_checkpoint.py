"""Evaluate a TRAINED ColTurk-VDR LoRA checkpoint on ViDoRe V3 (NDCG@10).

This is the eval harness that was MISSING — baseline_zeroshot.py only does
zero-shot baselines via mteb.evaluate (the Day-1 F6/F7 fragile path). Here we
load our own v5-native ColQwen3 + LoRA adapter and score ViDoRe V3 with a
self-contained colpali-native MaxSim loop (no mteb wrapper dependency).

Why colpali-native (not mteb):
- Our adapter is trained on transformers 5.9 + peft 0.19 + colpali 0.3.16, so its
  keys match the v5 ColQwen3 arch — F6 (v4 key-prefix mismatch) does NOT recur.
- ColQwen3Processor yields mm_token_type_ids natively — F7 does NOT recur.
- A manual MaxSim loop avoids the mteb ViDoRe(v3) ModelMeta/VisionRetriever
  wrapper unknowns that surfaced in Day-1.

ViDoRe V3 schema (HF, verified 2026-05-29):
  corpus  (split test): corpus_id:int64, image:image, doc_id, markdown, page_number_in_doc
  queries (split test): query_id:int64, query:string, language, ...
  qrels   (split test): query_id:int64, corpus_id:int64, score:int64

SMOKE DISCIPLINE: run --max-subtasks 1 (+ optional --corpus-limit) first to
validate the pipeline in ~10-20 min before the full 8-subtask sweep.

Run (Colab, after rescuing the adapter to HF Hub):
    python scripts/eval/eval_colturk_checkpoint.py \
        --adapter Verm1lion/ColTurk-VDR-Stage1-sweep-lr5e5 \
        --max-subtasks 1 --corpus-limit 500 --output eval/results/colturk_lr5e5_smoke.json
Then full:
    python scripts/eval/eval_colturk_checkpoint.py \
        --adapter Verm1lion/ColTurk-VDR-Stage1-sweep-lr5e5 \
        --output eval/results/colturk_lr5e5_v3.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ViDoRe V3 8 public subtasks (S41 / F9 verified slugs)
V3_PUBLIC_SUBTASKS: tuple[str, ...] = (
    "vidore/vidore_v3_finance_en",
    "vidore/vidore_v3_finance_fr",
    "vidore/vidore_v3_hr",
    "vidore/vidore_v3_industrial",
    "vidore/vidore_v3_computer_science",
    "vidore/vidore_v3_pharmaceuticals",
    "vidore/vidore_v3_physics",
    "vidore/vidore_v3_energy",
)

DEFAULT_BASE = "Qwen/Qwen3-VL-4B-Instruct"


def _ndcg_at_k(ranked_corpus_ids: list[int], rel: dict[int, int], k: int = 10) -> float:
    """NDCG@k. ranked_corpus_ids = docs sorted by score desc. rel = {corpus_id: grade}."""
    dcg = 0.0
    for rank, cid in enumerate(ranked_corpus_ids[:k], start=1):
        grade = rel.get(cid, 0)
        if grade > 0:
            dcg += grade / math.log2(rank + 1)
    ideal_grades = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_grades, start=1) if g > 0)
    return (dcg / idcg) if idcg > 0 else 0.0


def _recall_at_k(ranked_corpus_ids: list[int], rel: dict[int, int], k: int = 10) -> float:
    relevant = {cid for cid, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    topk = set(ranked_corpus_ids[:k])
    return len(topk & relevant) / len(relevant)


def _load_model(base: str, adapter: str):
    """ColQwen3 base + LoRA adapter (PeftModel) + ColQwen3Processor.

    adapter = HF repo id OR local path. If the adapter dir/repo already contains
    a full merged model it still loads; PeftModel.from_pretrained handles the
    adapter-on-base case (our sweep output).
    """
    import torch
    from colpali_engine.models import ColQwen3, ColQwen3Processor

    logger.info("Loading base ColQwen3: %s", base)
    model = ColQwen3.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()

    logger.info("Attaching LoRA adapter: %s", adapter)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter).eval()
    # Processor: prefer adapter repo (may carry processor config), fall back to base
    try:
        processor = ColQwen3Processor.from_pretrained(adapter)
    except Exception:
        processor = ColQwen3Processor.from_pretrained(base)
    return model, processor


def _encode_images(model, processor, images: list, batch_size: int) -> list:
    """Return list of per-doc multi-vector embeddings (CPU tensors)."""
    import torch

    embs: list = []
    for i in range(0, len(images), batch_size):
        batch = processor.process_images(images[i : i + batch_size]).to(model.device)
        with torch.no_grad():
            out = model(**batch)  # (B, seq, dim)
        # split per-sample, drop padding via attention mask if present
        mask = batch.get("attention_mask")
        for b in range(out.size(0)):
            if mask is not None:
                valid = mask[b].bool()
                embs.append(out[b][valid].to(torch.float32).cpu())
            else:
                embs.append(out[b].to(torch.float32).cpu())
        logger.info("  encoded images %d/%d", min(i + batch_size, len(images)), len(images))
    return embs


def _encode_queries(model, processor, queries: list[str], batch_size: int) -> list:
    import torch

    embs: list = []
    for i in range(0, len(queries), batch_size):
        batch = processor.process_queries(queries[i : i + batch_size]).to(model.device)
        with torch.no_grad():
            out = model(**batch)
        mask = batch.get("attention_mask")
        for b in range(out.size(0)):
            if mask is not None:
                valid = mask[b].bool()
                embs.append(out[b][valid].to(torch.float32).cpu())
            else:
                embs.append(out[b].to(torch.float32).cpu())
    return embs


def _maxsim_scores(processor, query_embs: list, doc_embs: list):
    """[Nq, Nd] MaxSim scores. Prefer processor.score_multi_vector (optimized);
    fall back to a manual loop so an API-name change cannot break eval."""
    import torch

    score_fn = getattr(processor, "score_multi_vector", None)
    if callable(score_fn):
        try:
            return score_fn(query_embs, doc_embs)  # returns (Nq, Nd) tensor
        except Exception as exc:
            logger.warning("score_multi_vector failed (%s) — manual MaxSim fallback", exc)

    nq, nd = len(query_embs), len(doc_embs)
    scores = torch.zeros(nq, nd, dtype=torch.float32)
    for qi, q in enumerate(query_embs):          # q: (Tq, D)
        for di, d in enumerate(doc_embs):        # d: (Td, D)
            sim = q @ d.T                         # (Tq, Td)
            scores[qi, di] = sim.max(dim=1).values.sum().item()
    return scores


def _eval_subtask(model, processor, subtask: str, batch_size: int, corpus_limit: int | None) -> dict[str, Any]:
    from datasets import load_dataset

    logger.info("=== %s ===", subtask)
    corpus = load_dataset(subtask, "corpus", split="test")
    queries = load_dataset(subtask, "queries", split="test")
    qrels = load_dataset(subtask, "qrels", split="test")

    if corpus_limit:
        corpus = corpus.select(range(min(corpus_limit, len(corpus))))

    corpus_ids = [int(r) for r in corpus["corpus_id"]]
    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    images = corpus["image"]
    q_ids = [int(r) for r in queries["query_id"]]
    q_texts = list(queries["query"])

    # qrels → {query_id: {corpus_id: grade}}, only for corpus we kept
    rel: dict[int, dict[int, int]] = {}
    for qid, cid, score in zip(qrels["query_id"], qrels["corpus_id"], qrels["score"]):
        cid = int(cid)
        if cid not in cid_to_idx:
            continue
        rel.setdefault(int(qid), {})[cid] = int(score)
    # keep only queries that have at least one relevant doc within kept corpus
    keep = [i for i, qid in enumerate(q_ids) if rel.get(qid)]
    if not keep:
        logger.warning("  no queries with relevant docs in kept corpus — skipping")
        return {"subtask": subtask, "ndcg_at_10": None, "recall_at_10": None, "n_queries": 0}
    q_ids = [q_ids[i] for i in keep]
    q_texts = [q_texts[i] for i in keep]

    logger.info("  corpus=%d queries=%d", len(images), len(q_texts))
    t0 = time.time()
    doc_embs = _encode_images(model, processor, images, batch_size)
    query_embs = _encode_queries(model, processor, q_texts, batch_size)
    scores = _maxsim_scores(processor, query_embs, doc_embs)
    logger.info("  encoded+scored in %.1fs", time.time() - t0)

    ndcgs, recalls = [], []
    for qi, qid in enumerate(q_ids):
        order = scores[qi].argsort(descending=True).tolist()
        ranked_cids = [corpus_ids[j] for j in order]
        ndcgs.append(_ndcg_at_k(ranked_cids, rel[qid], 10))
        recalls.append(_recall_at_k(ranked_cids, rel[qid], 10))

    res = {
        "subtask": subtask,
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "recall_at_10": sum(recalls) / len(recalls),
        "n_queries": len(q_ids),
        "n_corpus": len(images),
    }
    logger.info("  NDCG@10=%.4f recall@10=%.4f (n=%d)", res["ndcg_at_10"], res["recall_at_10"], len(q_ids))
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Eval trained ColTurk-VDR adapter on ViDoRe V3")
    p.add_argument("--adapter", required=True, help="HF repo id or local path of trained LoRA adapter")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--subtasks", default="", help="comma list to override the 8 public subtasks")
    p.add_argument("--max-subtasks", type=int, default=0, help="limit subtask count (smoke); 0 = all")
    p.add_argument("--corpus-limit", type=int, default=0, help="cap corpus per subtask (smoke); 0 = full")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", default="eval/results/colturk_v3_eval.json")
    args = p.parse_args()

    os.environ.setdefault("USE_LOCAL_DATASET", "0")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (A100).")

    subtasks = (
        [s.strip() for s in args.subtasks.split(",") if s.strip()]
        if args.subtasks else list(V3_PUBLIC_SUBTASKS)
    )
    if args.max_subtasks:
        subtasks = subtasks[: args.max_subtasks]

    model, processor = _load_model(args.base, args.adapter)

    results, ndcgs = [], []
    for st in subtasks:
        try:
            r = _eval_subtask(model, processor, st, args.batch_size,
                              args.corpus_limit or None)
            results.append(r)
            if r["ndcg_at_10"] is not None:
                ndcgs.append(r["ndcg_at_10"])
        except Exception as exc:
            logger.error("  %s FAILED: %s", st, exc)
            results.append({"subtask": st, "error": str(exc)})

    summary = {
        "adapter": args.adapter,
        "base": args.base,
        "ndcg_at_10_mean": (sum(ndcgs) / len(ndcgs)) if ndcgs else None,
        "subtasks_evaluated": len(ndcgs),
        "per_subtask": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== ColTurk-VDR ViDoRe V3 eval ===")
    print(f"adapter: {args.adapter}")
    for r in results:
        if r.get("ndcg_at_10") is not None:
            print(f"  {r['subtask']}: NDCG@10={r['ndcg_at_10']:.4f} recall@10={r['recall_at_10']:.4f} (n={r['n_queries']})")
        else:
            print(f"  {r['subtask']}: {r.get('error', 'no relevant docs')}")
    if summary["ndcg_at_10_mean"] is not None:
        print(f"\nMEAN NDCG@10 ({summary['subtasks_evaluated']} subtask): {summary['ndcg_at_10_mean']:.4f}")
    print(f"Saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
