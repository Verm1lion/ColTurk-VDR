"""Empirical contamination scan: manu/colpali (Stage-1 train) ↔ ViDoRe V3 (eval).

S52-B6. The source-based clearance (KARARLAR 2026-06-01: ~18mo gap, EN-vs-FR,
ILLUIN doc-level dedup) argued LOW contamination risk. This script closes it
EMPIRICALLY by perceptual-image-hash (pHash) overlap — the right tool because
manu/colpali is image-based (S41's 13-gram MinHash is for TEXT and does not apply).

Method:
  1. pHash (64-bit) a random sample of manu/colpali-corpus train images.
  2. pHash a per-subtask sample of each ViDoRe V3 corpus.
  3. For every V3 image, find the MIN Hamming distance to any train image
     (vectorised numpy popcount, chunked).
  4. Flag V3 images with min-distance <= threshold (default 6 = near-duplicate;
     0 = exact). Report the % of V3 corpus that has a train near-dup, per subtask
     and overall. Expectation: ~0%  ->  contamination empirically clean.

A non-trivial overlap (>~0.5%) would be a leakage flag worth investigating before
trusting the held-out NDCG numbers.

Deps: imagehash, Pillow, numpy, datasets  (pip install imagehash).

Run (Colab):
    python scripts/data/phash_contamination_scan.py \
        --train-sample 3000 --v3-per-subtask 1000 --threshold 6 \
        --output eval/results/phash_contamination.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

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

TRAIN_CORPUS = "manu/colpali-corpus"  # Stage-1 training corpus (USE_LOCAL_DATASET=0 -> HF)

# 16-bit popcount lookup -> light-memory 64-bit Hamming over numpy arrays.
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def _popcount64(a: np.ndarray) -> np.ndarray:
    """Population count of a uint64 array (any shape) -> int array (same shape)."""
    v = a.view(np.uint16).reshape(a.shape + (4,))
    return _POP16[v].sum(axis=-1)


def _phash_ints(images, sample: int, seed: int, label: str) -> np.ndarray:
    """pHash a (random) sample of PIL images -> uint64 array. Skips unreadable images."""
    import imagehash

    n = len(images)
    idx = list(range(n))
    if sample and sample < n:
        random.Random(seed).shuffle(idx)
        idx = idx[:sample]
    out: list[int] = []
    for k, i in enumerate(idx):
        try:
            img = images[i]
            if img.mode != "RGB":
                img = img.convert("RGB")
            h = imagehash.phash(img)  # 8x8 -> 64-bit
            out.append(int(str(h), 16))
        except Exception as exc:  # corrupt image -> skip, do not abort the scan
            logger.warning("  %s[%d] phash failed: %s", label, i, exc)
        if (k + 1) % 500 == 0:
            logger.info("  %s phashed %d/%d", label, k + 1, len(idx))
    return np.array(out, dtype=np.uint64)


def _min_hamming(v3: np.ndarray, train: np.ndarray, chunk: int = 256) -> np.ndarray:
    """For each v3 hash, min Hamming distance to any train hash. Chunked over v3."""
    mins = np.empty(len(v3), dtype=np.int16)
    for s in range(0, len(v3), chunk):
        block = v3[s : s + chunk]                     # (C,)
        xor = block[:, None] ^ train[None, :]         # (C, N) uint64
        dist = _popcount64(xor)                       # (C, N)
        mins[s : s + chunk] = dist.min(axis=1)
    return mins


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="pHash contamination scan: manu/colpali vs ViDoRe V3")
    p.add_argument("--train-sample", type=int, default=3000, help="manu/colpali images to hash (0=all)")
    p.add_argument("--v3-per-subtask", type=int, default=1000, help="V3 images per subtask to hash (0=all)")
    p.add_argument("--threshold", type=int, default=6, help="Hamming <= this = near-duplicate flag")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subtasks", default="", help="comma list to override the 8 public subtasks")
    p.add_argument("--output", default="eval/results/phash_contamination.json")
    args = p.parse_args()

    os.environ.setdefault("USE_LOCAL_DATASET", "0")
    from datasets import load_dataset

    subtasks = (
        [s.strip() for s in args.subtasks.split(",") if s.strip()]
        if args.subtasks else list(V3_PUBLIC_SUBTASKS)
    )

    logger.info("Loading train corpus %s", TRAIN_CORPUS)
    train_imgs = load_dataset(TRAIN_CORPUS, split="train")["image"]
    train_hashes = _phash_ints(train_imgs, args.train_sample, args.seed, "train")
    logger.info("train hashes: %d", len(train_hashes))
    if len(train_hashes) == 0:
        raise SystemExit("No train hashes — aborting.")

    per_subtask: list[dict[str, Any]] = []
    tot_v3 = tot_flag = tot_exact = 0
    for st in subtasks:
        logger.info("=== %s ===", st)
        try:
            v3_imgs = load_dataset(st, "corpus", split="test")["image"]
        except Exception as exc:
            logger.error("  load failed: %s", exc)
            per_subtask.append({"subtask": st, "error": str(exc)})
            continue
        v3_hashes = _phash_ints(v3_imgs, args.v3_per_subtask, args.seed, st.split("/")[-1])
        if len(v3_hashes) == 0:
            per_subtask.append({"subtask": st, "n_v3": 0})
            continue
        mins = _min_hamming(v3_hashes, train_hashes)
        n_v3 = len(v3_hashes)
        n_flag = int((mins <= args.threshold).sum())
        n_exact = int((mins == 0).sum())
        per_subtask.append({
            "subtask": st,
            "n_v3_hashed": n_v3,
            "near_dup_le_thresh": n_flag,
            "exact_dup": n_exact,
            "near_dup_percent": round(100.0 * n_flag / n_v3, 4),
            "min_hamming_min": int(mins.min()),
            "min_hamming_median": int(np.median(mins)),
        })
        logger.info("  near-dup(<=%d): %d/%d (%.3f%%)  exact: %d  min-dist=%d",
                    args.threshold, n_flag, n_v3, 100.0 * n_flag / n_v3, n_exact, int(mins.min()))
        tot_v3 += n_v3
        tot_flag += n_flag
        tot_exact += n_exact

    overall_pct = round(100.0 * tot_flag / tot_v3, 4) if tot_v3 else 0.0
    verdict = "CLEAN" if overall_pct <= 0.5 else "INVESTIGATE"
    report = {
        "scan": "phash_contamination",
        "train_corpus": TRAIN_CORPUS,
        "train_hashed": len(train_hashes),
        "threshold_hamming": args.threshold,
        "overall": {
            "v3_hashed": tot_v3,
            "near_dup": tot_flag,
            "exact_dup": tot_exact,
            "near_dup_percent": overall_pct,
            "verdict": verdict,
        },
        "per_subtask": per_subtask,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== pHash contamination scan (manu/colpali ↔ ViDoRe V3) ===")
    print(f"train hashed: {len(train_hashes)} | threshold (Hamming): {args.threshold}")
    print(f"{'subtask':>40} | {'n_v3':>6} | {'near-dup%':>9} | {'exact':>5}")
    for r in per_subtask:
        if "near_dup_percent" in r:
            print(f"{r['subtask']:>40} | {r['n_v3_hashed']:>6} | {r['near_dup_percent']:>8.3f}% | {r['exact_dup']:>5}")
        else:
            print(f"{r['subtask']:>40} | {r.get('error', 'n/a')}")
    print(f"\nOVERALL near-dup: {tot_flag}/{tot_v3} ({overall_pct:.3f}%)  exact: {tot_exact}  -> {verdict}")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
