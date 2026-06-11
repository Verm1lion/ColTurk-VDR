"""Reformat a ColTurk-VDR eval JSON -> MTEB v2 `results` schema (submission fallback).

WHY this exists (S45 / Day-1 lesson): the canonical submission produces the results
JSON by running `mteb.evaluate(wrapper, ViDoRe(v3))`. But the mteb ViDoRe/vision path
was FRAGILE on our transformers-v5 stack (F6/F7). Our own colpali-native harness
(`eval_colturk_checkpoint.py`) already computes the EXACT numbers MTEB wants — this
script reshapes that output into MTEB's `results` JSON so we can submit self-reported
results even if `mteb.evaluate` can't run end-to-end. MTEB explicitly accepts
self-reported results accompanied by the eval code (which we publish).

INPUT  = a ColTurk eval JSON (the `--output` of eval_colturk_checkpoint.py), shape:
  { "ndcg_at_10_mean", "ndcg_at_5_mean", "recall_at_10_mean", "per_subtask": [
      {"subtask": "vidore/vidore_v3_finance_en", "ndcg_at_10", "ndcg_at_5",
       "recall_at_10", "recall_at_5", "n_queries", "n_corpus",
       "ndcg_at_10_ci95":[lo,hi], ...}, ... ] }

OUTPUT = MTEB `results` files (one per subtask) under
  <out>/<MODEL_SLUG>/<REVISION>/<TaskName>.json

⚠️ VERIFY-IN-COLAB (where `mteb` is installed) before opening the PR — three things
   this script CANNOT confirm offline (mteb not installed locally):
   1) the exact MTEB v2 TASK NAME per V3 subtask (TASK_NAME_MAP below) — get them from
      `mteb.get_benchmark("ViDoRe(v3)")` -> [t.metadata.name for t in tasks].
   2) the `dataset_revision` per task (`t.metadata.dataset["revision"]`).
   3) `mteb_version` string (`mteb.__version__`).
   The metric KEYS (ndcg_at_10 etc.) and the {scores:{split:[{hf_subset,...}]}} shape
   follow the documented MTEB ScoresDict; main_score for retrieval = ndcg_at_10.
"""
from __future__ import annotations

import argparse
import json
import os

# V3 public subtask slug (ours) -> MTEB v2 task name. PLACEHOLDER names — replace from
# `mteb.get_benchmark("ViDoRe(v3)")` in Colab. ViDoRe V3 languages: EN docs + EN/FR queries
# for *_en subtasks; FR docs+queries for finance_fr/physics/energy(FR). Set per task in Colab.
TASK_NAME_MAP = {
    "vidore/vidore_v3_finance_en":       "VidoreV3FinanceEn",        # VERIFY
    "vidore/vidore_v3_finance_fr":       "VidoreV3FinanceFr",        # VERIFY
    "vidore/vidore_v3_hr":               "VidoreV3Hr",               # VERIFY
    "vidore/vidore_v3_industrial":       "VidoreV3Industrial",       # VERIFY
    "vidore/vidore_v3_computer_science": "VidoreV3ComputerScience",  # VERIFY
    "vidore/vidore_v3_pharmaceuticals":  "VidoreV3Pharmaceuticals",  # VERIFY
    "vidore/vidore_v3_physics":          "VidoreV3Physics",          # VERIFY
    "vidore/vidore_v3_energy":           "VidoreV3Energy",           # VERIFY
}

# eng-Latn for *_en (+ fra-Latn queries on _en per V3); fra-Latn for the FR subtasks.
LANGS = {
    "vidore/vidore_v3_finance_en":       ["eng-Latn", "fra-Latn"],
    "vidore/vidore_v3_finance_fr":       ["fra-Latn"],
    "vidore/vidore_v3_hr":               ["eng-Latn", "fra-Latn"],
    "vidore/vidore_v3_industrial":       ["eng-Latn", "fra-Latn"],
    "vidore/vidore_v3_computer_science": ["eng-Latn", "fra-Latn"],
    "vidore/vidore_v3_pharmaceuticals":  ["eng-Latn", "fra-Latn"],
    "vidore/vidore_v3_physics":          ["fra-Latn"],
    "vidore/vidore_v3_energy":           ["fra-Latn"],
}


def _subtask_scores(st: dict) -> dict:
    """One per_subtask entry -> one MTEB ScoresDict row (retrieval main_score = ndcg@10)."""
    slug = st["subtask"]
    row = {
        "hf_subset": "default",
        "languages": LANGS.get(slug, ["eng-Latn"]),
        "main_score": st["ndcg_at_10"],          # MTEB retrieval main_score = NDCG@10
        "ndcg_at_10": st["ndcg_at_10"],
        "ndcg_at_5": st.get("ndcg_at_5"),
        "recall_at_10": st.get("recall_at_10"),
        "recall_at_5": st.get("recall_at_5"),
        "num_queries": st.get("n_queries"),
        "num_docs": st.get("n_corpus"),
    }
    if "ndcg_at_10_ci95" in st:
        row["ndcg_at_10_ci95"] = st["ndcg_at_10_ci95"]
    return {k: v for k, v in row.items() if v is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description="ColTurk eval JSON -> MTEB v2 results files (fallback).")
    ap.add_argument("--input", required=True, help="ColTurk eval JSON (per_subtask format)")
    ap.add_argument("--out", default="eval/mteb_results", help="output root dir")
    ap.add_argument("--model-slug", default="Verm1ion__ColTurk-VDR-Stage1",
                    help="MTEB results model dir name (org__name)")
    ap.add_argument("--revision", default="main", help="model revision/commit (results subdir)")
    ap.add_argument("--mteb-version", default="VERIFY", help="mteb.__version__ (fill in Colab)")
    ap.add_argument("--dataset-revision", default="VERIFY", help="per-task dataset revision (fill in Colab)")
    args = ap.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    per = data.get("per_subtask", [])
    if not per:
        print("ERROR: input has no per_subtask[] — wrong file?")
        return 2

    out_dir = os.path.join(args.out, args.model_slug, args.revision)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    unknown = []
    for st in per:
        slug = st["subtask"]
        task_name = TASK_NAME_MAP.get(slug)
        if not task_name:
            unknown.append(slug)
            continue
        doc = {
            "dataset_revision": args.dataset_revision,
            "task_name": task_name,
            "mteb_version": args.mteb_version,
            "scores": {"test": [_subtask_scores(st)]},
            "evaluation_time": None,
            "kg_co2_emissions": None,
        }
        path = os.path.join(out_dir, f"{task_name}.json")
        json.dump(doc, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        n += 1
        print(f"  wrote {path}  (main_score={st['ndcg_at_10']:.4f})")

    print(f"\nOK {n}/{len(per)} MTEB results files -> {out_dir}")
    print(f"mean NDCG@10 = {data.get('ndcg_at_10_mean')}")
    if unknown:
        print(f"⚠️ unmapped subtasks (fill TASK_NAME_MAP): {unknown}")
    if args.mteb_version == "VERIFY" or args.dataset_revision == "VERIFY":
        print("⚠️ VERIFY: set --mteb-version, --dataset-revision, and TASK_NAME_MAP from live mteb in Colab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
