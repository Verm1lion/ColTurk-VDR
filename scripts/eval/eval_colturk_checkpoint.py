"""Evaluate a TRAINED ColTurk-VDR LoRA checkpoint on ViDoRe V3 (NDCG@5/@10).

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

S52 VALIDITY MODE (scientific defensibility — flags added 2026-06-02):
  --binarize           binarized relevance (grade>0 -> 1); leaderboard reports this too (S52-C8)
  --bootstrap 1000     percentile 95% CI over per-query NDCG (S52-C9)
  --no-adapter         eval RAW base (no LoRA; custom_text_proj random) -> ~floor NDCG.
                       Causal control: proves training moved NDCG floor->Y (S52-D10).
  --max-visual-tokens 768   cap doc visual tokens to match TRAINING (768); 0 = processor
                       default = leaderboard-style. Run both for the 768-vs-default A/B (S52-E).
  (full-corpus = omit --corpus-limit  ->  true ranking over the whole subtask corpus, S52-C7)

Run (Colab, after rescuing the adapter to HF Hub):
    python scripts/eval/eval_colturk_checkpoint.py \
        --adapter Verm1ion/ColTurk-VDR-Stage1 --subtasks vidore/vidore_v3_finance_en \
        --corpus-limit 500 --query-limit 100 --output eval/results/smoke.json
Full-corpus official number (leaderboard-comparable) + validity:
    python scripts/eval/eval_colturk_checkpoint.py \
        --adapter /content/outputs/colturk-vdr-stage1/checkpoint-500 \
        --bootstrap 1000 --output eval/results/colturk_stage1_full.json
Base control (floor):
    python scripts/eval/eval_colturk_checkpoint.py \
        --no-adapter --query-limit 100 --output eval/results/base_control.json

CURVE mode (find the best checkpoint along a run's trajectory) — eval every
checkpoint-N/ under a local training output_dir, emit NDCG@10-vs-step + the best:
    python scripts/eval/eval_colturk_checkpoint.py \
        --checkpoints-root /content/outputs/colturk-vdr-stage1 \
        --corpus-limit 500 --query-limit 100 --output eval/results/curve.json
This is how we pick the best model from a longer run (and which step to stop at)
without betting on a single arbitrary checkpoint.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
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


def _mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


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


def _bootstrap_ci(values: list[float], n_boot: int, seed: int = 42, alpha: float = 0.05) -> list[float]:
    """Percentile bootstrap [lo, hi] for the MEAN of per-query scores (S52-C9).

    Seeded (42) so the CI is reproducible. O(n_boot * n) — fine for n up to a few k.
    """
    if not values or n_boot <= 0:
        return [round(_mean(values), 4), round(_mean(values), 4)]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[min(n_boot - 1, int((alpha / 2) * n_boot))]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


def _load_model(base: str, adapter: str | None, max_visual_tokens: int | None = None):
    """ColQwen3 base (+ optional LoRA adapter) + ColQwen3Processor.

    adapter = HF repo id OR local path. If None -> NO-ADAPTER base control: raw base
    with a RANDOM-init custom_text_proj head -> ~floor NDCG (S52-D10 causal control).
    max_visual_tokens: cap doc visual tokens (768 = train-match; None = processor default).
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

    if adapter:
        logger.info("Attaching LoRA adapter: %s", adapter)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).eval()
    else:
        logger.warning(
            "NO-ADAPTER base control: raw base, custom_text_proj is RANDOM-init "
            "-> embeddings ~meaningless -> floor NDCG expected (this is the causal control)."
        )

    proc_kwargs: dict[str, Any] = {}
    if max_visual_tokens:
        proc_kwargs["max_num_visual_tokens"] = max_visual_tokens
        logger.info("Processor max_num_visual_tokens=%d (train-match)", max_visual_tokens)
    # Processor: prefer adapter repo (may carry processor config), fall back to base
    proc_src = adapter if adapter else base
    try:
        processor = ColQwen3Processor.from_pretrained(proc_src, **proc_kwargs)
    except Exception:
        processor = ColQwen3Processor.from_pretrained(base, **proc_kwargs)
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


def _encode_queries(model, processor, queries: list[str], batch_size: int, max_length: int = 128) -> list:
    import torch

    embs: list = []
    for i in range(0, len(queries), batch_size):
        # max_length: ColQwen3Processor default is 50, which silently truncates verbose
        # ViDoRe V3 queries and degrades NDCG. 128 covers them without truncation.
        batch = processor.process_queries(queries[i : i + batch_size], max_length=max_length).to(model.device)
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


def _eval_subtask(
    model, processor, subtask: str, batch_size: int,
    corpus_limit: int | None, query_limit: int | None = None,
    binarize: bool = False, bootstrap: int = 0,
) -> dict[str, Any]:
    from datasets import load_dataset

    logger.info("=== %s ===", subtask)
    corpus = load_dataset(subtask, "corpus", split="test")
    queries = load_dataset(subtask, "queries", split="test")
    qrels = load_dataset(subtask, "qrels", split="test")

    all_q_ids = [int(r) for r in queries["query_id"]]
    all_q_texts = list(queries["query"])

    # rel from the FULL qrels (before any corpus subsetting) → {query_id: {corpus_id: grade}}
    rel_full: dict[int, dict[int, int]] = {}
    for qid, cid, score in zip(qrels["query_id"], qrels["corpus_id"], qrels["score"]):
        rel_full.setdefault(int(qid), {})[int(cid)] = int(score)

    # Query subset: only queries that HAVE qrels; cap by query_limit (smoke).
    sel_idx = [i for i, qid in enumerate(all_q_ids) if rel_full.get(qid)]
    if query_limit:
        sel_idx = sel_idx[:query_limit]
    if not sel_idx:
        logger.warning("  no queries with qrels — skipping")
        return {"subtask": subtask, "ndcg_at_10": None, "recall_at_10": None, "n_queries": 0}
    sel_qids = [all_q_ids[i] for i in sel_idx]
    sel_qtexts = [all_q_texts[i] for i in sel_idx]

    # GOLD-AWARE corpus subsetting (fixes the "first-N rows miss every gold doc" trap):
    # keep ALL gold docs for the selected queries, then fill with other docs up to the
    # limit so every evaluated query is answerable among `corpus_limit` real distractors.
    # corpus_limit=None → full corpus (true ranking over everything, S52-C7).
    if corpus_limit:
        gold = set()
        for qid in sel_qids:
            gold.update(rel_full[qid].keys())
        keep_cids = set(gold)
        for cid in corpus["corpus_id"]:
            if len(keep_cids) >= corpus_limit:
                break
            keep_cids.add(int(cid))
        idx_keep = [i for i, cid in enumerate(corpus["corpus_id"]) if int(cid) in keep_cids]
        corpus = corpus.select(idx_keep)
        logger.info("  corpus subset: %d gold + filler = %d docs (limit=%d)", len(gold), len(corpus), corpus_limit)

    corpus_ids = [int(r) for r in corpus["corpus_id"]]
    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    images = corpus["image"]

    # rel restricted to kept corpus, for the selected queries only
    rel: dict[int, dict[int, int]] = {}
    for qid in sel_qids:
        kept = {cid: g for cid, g in rel_full[qid].items() if cid in cid_to_idx}
        if kept:
            rel[qid] = kept
    keep = [i for i, qid in enumerate(sel_qids) if qid in rel]
    if not keep:
        logger.warning("  no selected queries have gold in kept corpus — skipping")
        return {"subtask": subtask, "ndcg_at_10": None, "recall_at_10": None, "n_queries": 0}
    q_ids = [sel_qids[i] for i in keep]
    q_texts = [sel_qtexts[i] for i in keep]

    # Binarized relevance (S52-C8): collapse graded scores to 1 (relevant/not).
    if binarize:
        rel = {qid: {cid: 1 for cid in d} for qid, d in rel.items()}

    logger.info("  corpus=%d queries=%d binarize=%s", len(images), len(q_texts), binarize)
    t0 = time.time()
    doc_embs = _encode_images(model, processor, images, batch_size)
    query_embs = _encode_queries(model, processor, q_texts, batch_size)
    scores = _maxsim_scores(processor, query_embs, doc_embs)
    logger.info("  encoded+scored in %.1fs", time.time() - t0)

    ndcg10, ndcg5, rec10, rec5 = [], [], [], []
    for qi, qid in enumerate(q_ids):
        order = scores[qi].argsort(descending=True).tolist()
        ranked_cids = [corpus_ids[j] for j in order]
        ndcg10.append(_ndcg_at_k(ranked_cids, rel[qid], 10))
        ndcg5.append(_ndcg_at_k(ranked_cids, rel[qid], 5))
        rec10.append(_recall_at_k(ranked_cids, rel[qid], 10))
        rec5.append(_recall_at_k(ranked_cids, rel[qid], 5))

    res: dict[str, Any] = {
        "subtask": subtask,
        "ndcg_at_10": _mean(ndcg10),
        "ndcg_at_5": _mean(ndcg5),
        "recall_at_10": _mean(rec10),
        "recall_at_5": _mean(rec5),
        "n_queries": len(q_ids),
        "n_corpus": len(images),
        "binarized": binarize,
    }
    if bootstrap:
        res["ndcg_at_10_ci95"] = _bootstrap_ci(ndcg10, bootstrap)
        res["ndcg_at_5_ci95"] = _bootstrap_ci(ndcg5, bootstrap)
    logger.info(
        "  NDCG@10=%.4f NDCG@5=%.4f recall@10=%.4f (n=%d)%s",
        res["ndcg_at_10"], res["ndcg_at_5"], res["recall_at_10"], len(q_ids),
        f" CI95={res['ndcg_at_10_ci95']}" if bootstrap else "",
    )
    return res


def _step_of(path: Any) -> int:
    """Extract training step from a 'checkpoint-N' dir name; -1 if none."""
    m = re.search(r"checkpoint-(\d+)", str(path))
    return int(m.group(1)) if m else -1


def _eval_adapter(
    base: str, adapter: str | None, subtasks: list[str], batch_size: int,
    corpus_limit: int | None, query_limit: int | None = None,
    binarize: bool = False, bootstrap: int = 0, max_visual_tokens: int | None = None,
) -> dict[str, Any]:
    """Load base(+adapter), eval the subtasks, free GPU, return a summary dict.

    Reloads the base per call (cheap vs corpus encoding, ~30-60s) so the curve
    loop cannot accumulate adapters or leak VRAM across checkpoints.
    adapter=None -> NO-ADAPTER base control (S52-D10).
    """
    import gc

    import torch

    model, processor = _load_model(base, adapter, max_visual_tokens)
    results, ndcg10s, ndcg5s, recalls = [], [], [], []
    for st in subtasks:
        try:
            r = _eval_subtask(model, processor, st, batch_size, corpus_limit, query_limit, binarize, bootstrap)
            results.append(r)
            if r["ndcg_at_10"] is not None:
                ndcg10s.append(r["ndcg_at_10"])
                ndcg5s.append(r["ndcg_at_5"])
                recalls.append(r["recall_at_10"])
        except Exception as exc:
            logger.error("  %s FAILED: %s", st, exc)
            results.append({"subtask": st, "error": str(exc)})

    summary = {
        "adapter": adapter if adapter else "(no-adapter base control)",
        "base": base,
        "binarized": binarize,
        "max_visual_tokens": max_visual_tokens or "processor-default",
        "ndcg_at_10_mean": (_mean(ndcg10s)) if ndcg10s else None,
        "ndcg_at_5_mean": (_mean(ndcg5s)) if ndcg5s else None,
        "recall_at_10_mean": (_mean(recalls)) if recalls else None,
        "subtasks_evaluated": len(ndcg10s),
        "per_subtask": results,
    }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _print_single(summary: dict[str, Any], out: Path) -> None:
    print("\n=== ColTurk-VDR ViDoRe V3 eval ===")
    print(f"adapter: {summary['adapter']}")
    print(f"binarized={summary['binarized']}  max_visual_tokens={summary['max_visual_tokens']}")
    for r in summary["per_subtask"]:
        if r.get("ndcg_at_10") is not None:
            ci = f" CI95={r['ndcg_at_10_ci95']}" if "ndcg_at_10_ci95" in r else ""
            print(
                f"  {r['subtask']}: NDCG@10={r['ndcg_at_10']:.4f} NDCG@5={r['ndcg_at_5']:.4f} "
                f"recall@10={r['recall_at_10']:.4f} (n={r['n_queries']}){ci}"
            )
        else:
            print(f"  {r['subtask']}: {r.get('error', 'no relevant docs')}")
    if summary["ndcg_at_10_mean"] is not None:
        print(
            f"\nMEAN NDCG@10 ({summary['subtasks_evaluated']} subtask): {summary['ndcg_at_10_mean']:.4f}"
            f"  | NDCG@5: {summary['ndcg_at_5_mean']:.4f}"
        )
    print(f"Saved -> {out}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Eval trained ColTurk-VDR adapter(s) on ViDoRe V3")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="HF repo id or local path of ONE trained LoRA adapter")
    src.add_argument(
        "--checkpoints-root",
        help="Dir containing checkpoint-*/ subdirs (a training output_dir). CURVE mode: "
             "eval every checkpoint, emit NDCG@10-vs-step + pick the best. Each checkpoint is "
             "independently reusable thanks to F28 (modules_to_save: custom_text_proj).",
    )
    src.add_argument(
        "--no-adapter",
        action="store_true",
        help="BASE CONTROL: eval RAW base (no LoRA); custom_text_proj is random-init -> ~floor "
             "NDCG. Causal proof that training moved the metric (S52-D10).",
    )
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--subtasks", default="", help="comma list to override the 8 public subtasks")
    p.add_argument("--max-subtasks", type=int, default=0, help="limit subtask count (smoke); 0 = all")
    p.add_argument("--corpus-limit", type=int, default=0,
                   help="cap corpus per subtask (smoke): keeps all gold docs for the selected "
                        "queries + filler up to this many; 0 = full corpus (true ranking, S52-C7)")
    p.add_argument("--query-limit", type=int, default=0,
                   help="cap evaluated queries per subtask (smoke); 0 = all queries with qrels. "
                        "Set this WITH --corpus-limit for a meaningful smoke (else gold-union ~= full corpus)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--binarize", action="store_true",
                   help="binarized relevance (grade>0 -> 1); leaderboard reports this too (S52-C8)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="bootstrap resamples for a seeded 95%% CI over per-query NDCG (e.g. 1000); 0 = off (S52-C9)")
    p.add_argument("--max-visual-tokens", type=int, default=0,
                   help="cap doc visual tokens (768 = train-match; 0 = processor default = leaderboard-style). S52-E")
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

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus_limit = args.corpus_limit or None
    query_limit = args.query_limit or None
    max_vt = args.max_visual_tokens or None
    binarize = args.binarize
    bootstrap = args.bootstrap

    # --- CURVE mode: eval every checkpoint, pick the best ---
    if args.checkpoints_root:
        root = Path(args.checkpoints_root)
        ckpts = sorted((c for c in root.glob("checkpoint-*") if c.is_dir()), key=_step_of)
        if not ckpts and (root / "adapter_config.json").exists():
            ckpts = [root]  # root itself is a single adapter
        if not ckpts:
            raise SystemExit(f"No checkpoint-*/ dirs (or adapter) under {root}")

        curve = []
        for ck in ckpts:
            step = _step_of(ck)
            logger.info("### checkpoint step=%s (%s) ###", step, ck)
            s = _eval_adapter(args.base, str(ck), subtasks, args.batch_size, corpus_limit,
                              query_limit, binarize, bootstrap, max_vt)
            curve.append({
                "step": step,
                "checkpoint": str(ck),
                "ndcg_at_10_mean": s["ndcg_at_10_mean"],
                "ndcg_at_5_mean": s["ndcg_at_5_mean"],
                "recall_at_10_mean": s["recall_at_10_mean"],
                "subtasks_evaluated": s["subtasks_evaluated"],
                "per_subtask": s["per_subtask"],
            })

        scored = [c for c in curve if c["ndcg_at_10_mean"] is not None]
        best = max(scored, key=lambda c: c["ndcg_at_10_mean"]) if scored else None
        report = {
            "mode": "curve",
            "base": args.base,
            "checkpoints_root": str(root),
            "subtasks": subtasks,
            "corpus_limit": args.corpus_limit,
            "query_limit": args.query_limit,
            "binarized": binarize,
            "max_visual_tokens": max_vt or "processor-default",
            "best": {"step": best["step"], "ndcg_at_10_mean": best["ndcg_at_10_mean"]} if best else None,
            "curve": curve,
        }
        out.write_text(json.dumps(report, indent=2))

        print("\n=== ColTurk-VDR checkpoint CURVE (ViDoRe V3) ===")
        print(f"{'step':>8} | {'NDCG@10':>8} | {'NDCG@5':>8} | {'recall@10':>9}")
        print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")
        for c in curve:
            nd = f"{c['ndcg_at_10_mean']:.4f}" if c["ndcg_at_10_mean"] is not None else "  n/a "
            n5 = f"{c['ndcg_at_5_mean']:.4f}" if c["ndcg_at_5_mean"] is not None else "  n/a "
            rc = f"{c['recall_at_10_mean']:.4f}" if c["recall_at_10_mean"] is not None else "  n/a "
            print(f"{c['step']:>8} | {nd:>8} | {n5:>8} | {rc:>9}")
        if best:
            print(f"\nBEST: checkpoint-{best['step']}  NDCG@10={best['ndcg_at_10_mean']:.4f}")
        print(f"Saved -> {out}")
        return 0

    # --- SINGLE mode (--adapter, or --no-adapter base control) ---
    adapter = None if args.no_adapter else args.adapter
    summary = _eval_adapter(args.base, adapter, subtasks, args.batch_size, corpus_limit,
                            query_limit, binarize, bootstrap, max_vt)
    out.write_text(json.dumps(summary, indent=2))
    _print_single(summary, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
