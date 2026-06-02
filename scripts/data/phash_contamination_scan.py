"""Empirical contamination scan: manu/colpali (Stage-1 train) ↔ ViDoRe V3 (eval).

S52-B6. The source-based clearance (KARARLAR 2026-06-01: ~18mo gap, EN-vs-FR,
ILLUIN doc-level dedup) argued LOW contamination risk. This script closes it
EMPIRICALLY by perceptual-image-hash (pHash) overlap — the right tool because
manu/colpali is image-based (S41's 13-gram MinHash is for TEXT and does not apply).

IMPORTANT — pHash over-flags DOCUMENTS. A 64-bit pHash is an 8x8 DCT low-frequency
signature; text-heavy pages (white bg, columns, tables, margins) collapse to similar
coarse signatures regardless of CONTENT. So Hamming<=6 catches "same page geometry",
NOT "same document". For documents the TRUE-duplicate bar is exact (0) or <=2, and any
flag must be VISUALLY confirmed. This script therefore:
  1. reports counts at multiple tiers (exact / <=2 / <=4 / <=6),
  2. with --dump-pairs, saves a contact-sheet of the lowest-Hamming flagged
     (v3 | nearest-train) pairs for eyeball confirmation (layout-collision vs true dup).

Method:
  1. pHash (64-bit) a random sample of manu/colpali-corpus train images.
  2. pHash a per-subtask sample of each ViDoRe V3 corpus.
  3. For every V3 image, MIN Hamming to any train image (+ which train image).
  4. Tier counts + (optional) contact-sheet of the closest pairs.

Deps: imagehash, Pillow, numpy, datasets  (pip install imagehash).

Run (Colab):
    python scripts/data/phash_contamination_scan.py \
        --train-sample 3000 --v3-per-subtask 1000 \
        --dump-pairs 24 --pairs-dir eval/results/phash_pairs \
        --output eval/results/phash_contamination.json
Then display eval/results/phash_pairs/contact_sheet.png inline and eyeball it.
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


def _to_rgb(img):
    return img if img.mode == "RGB" else img.convert("RGB")


def _phash_sample(images, sample: int, seed: int, label: str):
    """pHash a (random) sample of PIL images.

    Returns (hashes: uint64[M], orig_idxs: list[int]) — orig_idxs lets us re-fetch
    the actual image later (for the contact-sheet). Skips unreadable images.
    """
    import imagehash

    n = len(images)
    idx = list(range(n))
    if sample and sample < n:
        random.Random(seed).shuffle(idx)
        idx = idx[:sample]
    hashes: list[int] = []
    kept_idx: list[int] = []
    for k, i in enumerate(idx):
        try:
            h = imagehash.phash(_to_rgb(images[i]))  # 8x8 -> 64-bit
            hashes.append(int(str(h), 16))
            kept_idx.append(i)
        except Exception as exc:  # corrupt image -> skip, do not abort the scan
            logger.warning("  %s[%d] phash failed: %s", label, i, exc)
        if (k + 1) % 500 == 0:
            logger.info("  %s phashed %d/%d", label, k + 1, len(idx))
    return np.array(hashes, dtype=np.uint64), kept_idx


def _min_hamming(v3: np.ndarray, train: np.ndarray, chunk: int = 256):
    """For each v3 hash: (min Hamming to any train hash, argmin train index). Chunked."""
    mins = np.empty(len(v3), dtype=np.int16)
    args = np.empty(len(v3), dtype=np.int64)
    for s in range(0, len(v3), chunk):
        block = v3[s : s + chunk]                     # (C,)
        xor = block[:, None] ^ train[None, :]         # (C, N) uint64
        dist = _popcount64(xor)                       # (C, N)
        mins[s : s + chunk] = dist.min(axis=1)
        args[s : s + chunk] = dist.argmin(axis=1)
    return mins, args


def _build_contact_sheet(pairs: list[dict], out_path: Path, cols: int = 4, thumb: int = 160) -> None:
    """pairs: list of {v3_img, train_img, label}. Grid of (v3 | train) cells for eyeballing."""
    from PIL import Image, ImageDraw

    if not pairs:
        return
    pad, lab_h = 6, 16
    cell_w = thumb * 2 + pad * 3
    cell_h = thumb + lab_h + pad * 2
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, pr in enumerate(pairs):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        draw.text((x0 + pad, y0 + 2), pr["label"], fill="black")
        for j, key in enumerate(("v3_img", "train_img")):
            im = _to_rgb(pr[key]).copy()
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (x0 + pad + j * (thumb + pad), y0 + lab_h + pad))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    logger.info("contact sheet -> %s (%d pairs)", out_path, len(pairs))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="pHash contamination scan: manu/colpali vs ViDoRe V3")
    p.add_argument("--train-sample", type=int, default=3000, help="manu/colpali images to hash (0=all)")
    p.add_argument("--v3-per-subtask", type=int, default=1000, help="V3 images per subtask to hash (0=all)")
    p.add_argument("--dump-pairs", type=int, default=0,
                   help="save a contact-sheet of the N lowest-Hamming flagged (v3|train) pairs for visual check")
    p.add_argument("--pairs-dir", default="eval/results/phash_pairs", help="dir for the contact sheet + manifest")
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
    train_hashes, train_idxs = _phash_sample(train_imgs, args.train_sample, args.seed, "train")
    logger.info("train hashes: %d", len(train_hashes))
    if len(train_hashes) == 0:
        raise SystemExit("No train hashes — aborting.")

    per_subtask: list[dict[str, Any]] = []
    tot = {"v3": 0, "exact": 0, "le2": 0, "le4": 0, "le6": 0}
    flagged_pairs: list[dict] = []  # for contact sheet (global, sorted by hamming asc)

    for st in subtasks:
        logger.info("=== %s ===", st)
        try:
            v3_col = load_dataset(st, "corpus", split="test")["image"]
        except Exception as exc:
            logger.error("  load failed: %s", exc)
            per_subtask.append({"subtask": st, "error": str(exc)})
            continue
        v3_hashes, v3_idxs = _phash_sample(v3_col, args.v3_per_subtask, args.seed, st.split("/")[-1])
        if len(v3_hashes) == 0:
            per_subtask.append({"subtask": st, "n_v3": 0})
            continue
        mins, args_tr = _min_hamming(v3_hashes, train_hashes)
        n_v3 = len(v3_hashes)
        c_exact = int((mins == 0).sum())
        c_le2 = int((mins <= 2).sum())
        c_le4 = int((mins <= 4).sum())
        c_le6 = int((mins <= 6).sum())
        per_subtask.append({
            "subtask": st, "n_v3_hashed": n_v3,
            "exact": c_exact, "le2": c_le2, "le4": c_le4, "le6": c_le6,
            "le6_percent": round(100.0 * c_le6 / n_v3, 4),
            "le2_percent": round(100.0 * c_le2 / n_v3, 4),
            "min_hamming_min": int(mins.min()), "min_hamming_median": int(np.median(mins)),
        })
        logger.info("  exact=%d le2=%d le4=%d le6=%d (n=%d) min=%d",
                    c_exact, c_le2, c_le4, c_le6, n_v3, int(mins.min()))
        tot["v3"] += n_v3
        tot["exact"] += c_exact; tot["le2"] += c_le2; tot["le4"] += c_le4; tot["le6"] += c_le6

        if args.dump_pairs:
            order = np.argsort(mins)  # closest first
            for vi in order[: args.dump_pairs]:
                if mins[vi] > 6:
                    break
                flagged_pairs.append({
                    "_h": int(mins[vi]), "subtask": st,
                    "v3_orig_idx": v3_idxs[int(vi)], "train_orig_idx": train_idxs[int(args_tr[vi])],
                    "v3_col": v3_col, "train_col": train_imgs,
                    "label": f"{st.split('/')[-1][:14]} h={int(mins[vi])}",
                })

    le2_pct = round(100.0 * tot["le2"] / tot["v3"], 4) if tot["v3"] else 0.0
    le6_pct = round(100.0 * tot["le6"] / tot["v3"], 4) if tot["v3"] else 0.0
    # Recalibrated for DOCUMENTS: true-dup bar is exact/<=2; <=6 is layout-collision-prone.
    if tot["exact"] > 0 or le2_pct > 0.3:
        verdict = "INVESTIGATE-TRUEDUP"
    elif le6_pct > 0.5:
        verdict = "LAYOUT-FP-LIKELY (visual confirm via --dump-pairs)"
    else:
        verdict = "CLEAN"

    report = {
        "scan": "phash_contamination",
        "train_corpus": TRAIN_CORPUS,
        "train_hashed": len(train_hashes),
        "note": "pHash over-flags documents on LAYOUT; true-dup bar is exact/<=2, <=6 needs visual confirm",
        "overall": {
            "v3_hashed": tot["v3"], "exact": tot["exact"],
            "le2": tot["le2"], "le2_percent": le2_pct,
            "le4": tot["le4"], "le6": tot["le6"], "le6_percent": le6_pct,
            "verdict": verdict,
        },
        "per_subtask": per_subtask,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    # Contact sheet (global closest pairs across subtasks)
    if args.dump_pairs and flagged_pairs:
        flagged_pairs.sort(key=lambda d: d["_h"])
        top = flagged_pairs[: args.dump_pairs]
        pairs_render = [{
            "v3_img": pr["v3_col"][pr["v3_orig_idx"]],
            "train_img": pr["train_col"][pr["train_orig_idx"]],
            "label": pr["label"],
        } for pr in top]
        sheet_path = Path(args.pairs_dir) / "contact_sheet.png"
        _build_contact_sheet(pairs_render, sheet_path)
        manifest = [{k: pr[k] for k in ("subtask", "_h", "v3_orig_idx", "train_orig_idx")} for pr in top]
        (Path(args.pairs_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== pHash contamination scan (manu/colpali ↔ ViDoRe V3) ===")
    print(f"train hashed: {len(train_hashes)}  (pHash over-flags docs on LAYOUT; true-dup = exact/<=2)")
    print(f"{'subtask':>40} | {'n_v3':>6} | {'exact':>5} | {'<=2':>4} | {'<=4':>4} | {'<=6':>4}")
    for r in per_subtask:
        if "le6" in r:
            print(f"{r['subtask']:>40} | {r['n_v3_hashed']:>6} | {r['exact']:>5} | {r['le2']:>4} | {r['le4']:>4} | {r['le6']:>4}")
        else:
            print(f"{r['subtask']:>40} | {r.get('error', 'n/a')}")
    print(f"\nOVERALL  exact={tot['exact']}  <=2={tot['le2']} ({le2_pct}%)  <=6={tot['le6']} ({le6_pct}%)  -> {verdict}")
    if args.dump_pairs and flagged_pairs:
        print(f"contact sheet -> {Path(args.pairs_dir) / 'contact_sheet.png'}  (display + eyeball it)")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
