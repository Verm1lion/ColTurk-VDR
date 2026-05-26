"""Day-1 critical: 3-model zero-shot baseline on ViDoRe V3 8 public subtasks.

OPEN_ITEMS.md A9 + KARARLAR S19. Run on Colab Pro+ A100 80GB ($0 marginal).

Pipeline:
1. Load 3 baselines via colpali-engine (M1 smoke + M2 anchor + M3 SoTA-candidate)
2. Load ViDoRe V3 8 public subtasks via `mteb.get_benchmark("ViDoRe(v3)")`
3. Zero-shot eval per model: NDCG@10, recall@10, mAP
4. Log to W&B project `colturk-vdr`, run group `baseline`, 3 distinct runs
5. Save results JSON: eval/results/baseline_zeroshot_{model_safe}.json
6. Print decision table for KARARLAR Log append (Mert manuel commits)

Refs:
- agent1_prompt1.md / agent2_prompt1.md (MTEB submission protocol)
- agent1_prompt2.md / agent2_prompt2.md (ColQwen3 = Qwen3VLModel verified)
- Nemotron ColEmbed V2 paper arXiv:2602.03992 (top-1 NDCG@10 63.42)

Run:
    KAGGLE_KEY=... HF_TOKEN=... WANDB_API_KEY=... \
        python scripts/eval/baseline_zeroshot.py
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results"
WANDB_PROJECT = "colturk-vdr"
WANDB_ENTITY = "verm1lion-colturk-vdr"
WANDB_GROUP = "baseline"


@dataclass(frozen=True)
class BaselineSpec:
    model_id: str
    model_class: str
    processor_class: str
    expected_signal: str
    run_name: str


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        model_id="Metric-AI/ColQwen2.5-3b-multilingual-v1.0",
        model_class="ColQwen2_5",
        processor_class="ColQwen2_5_Processor",
        expected_signal="low — training data IT/ES/EN/FR/DE only (no Turkish); V2 overfitting flag (Mace et al.)",
        run_name="m1-metric-ai-colqwen25-3b-multi-zeroshot",
    ),
    BaselineSpec(
        model_id="vidore/colqwen2.5-v0.2",
        model_class="ColQwen2_5",
        processor_class="ColQwen2_5_Processor",
        expected_signal="mid — V3 known anchor: 0.592 NDCG@10 reported (Nemotron paper Table 2)",
        run_name="m2-vidore-colqwen25-v02-zeroshot",
    ),
    BaselineSpec(
        model_id="OpenSearch-AI/Ops-Colqwen3-4B",
        model_class="ColQwen3",
        processor_class="ColQwen3Processor",
        expected_signal="high — Qwen3-VL-4B base, 128-dim, ViDoRe V1-V3 4B class claimed SoTA; Stage-1 warm-start candidate",
        run_name="m3-ops-colqwen3-4b-zeroshot",
    ),
)


def _load_baseline(spec: BaselineSpec) -> tuple[Any, Any]:
    """Import + instantiate model + processor by class name.

    ColQwen3 is verified Qwen3-VL native (class ColQwen3(Qwen3VLModel)) per
    raw fetch of colpali_engine/models/qwen3/colqwen3/modeling_colqwen3.py
    on 2026-05-18 — no custom adapter required (KARARLAR S17).
    """
    import colpali_engine.models as cpm

    model_cls = getattr(cpm, spec.model_class)
    proc_cls = getattr(cpm, spec.processor_class)

    model = model_cls.from_pretrained(
        spec.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    processor = proc_cls.from_pretrained(spec.model_id)
    return model, processor


def _evaluate(spec: BaselineSpec, model: Any) -> dict[str, Any]:
    """Run mteb.evaluate over ViDoRe V3 8 public subtasks.

    MTEB Python API path (not vidore-benchmark CLI — issue #122 V3 BEIR
    loading is broken). If MTEB ModelMeta does not exist for spec.model_id,
    a VisionRetriever wrapper around (model, processor) is required —
    Day-1 execution will surface this; see agent2_prompt1.md Q3a for the
    wrapper sketch.
    """
    import mteb

    tasks = mteb.get_benchmark("ViDoRe(v3)")
    output_folder = RESULTS_DIR / spec.model_id.replace("/", "__")
    output_folder.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results = mteb.evaluate(
        model,
        tasks=tasks,
        output_folder=str(output_folder),
    )
    elapsed = time.time() - started

    aggregated = {
        "model_id": spec.model_id,
        "elapsed_seconds": elapsed,
        "per_subtask": {},
        "ndcg_at_10_mean": None,
        "recall_at_10_mean": None,
        "map_mean": None,
    }

    ndcg_values: list[float] = []
    recall_values: list[float] = []
    map_values: list[float] = []

    for task_result in results:
        task_name = task_result.task_name
        score_row = task_result.scores["test"][0]
        ndcg = score_row.get("ndcg_at_10")
        recall = score_row.get("recall_at_10")
        mean_ap = score_row.get("map_at_10") or score_row.get("map_at_100")

        aggregated["per_subtask"][task_name] = {
            "ndcg_at_10": ndcg,
            "recall_at_10": recall,
            "map": mean_ap,
        }
        if ndcg is not None:
            ndcg_values.append(ndcg)
        if recall is not None:
            recall_values.append(recall)
        if mean_ap is not None:
            map_values.append(mean_ap)

    if ndcg_values:
        aggregated["ndcg_at_10_mean"] = sum(ndcg_values) / len(ndcg_values)
    if recall_values:
        aggregated["recall_at_10_mean"] = sum(recall_values) / len(recall_values)
    if map_values:
        aggregated["map_mean"] = sum(map_values) / len(map_values)

    summary_path = output_folder / "summary.json"
    summary_path.write_text(json.dumps(aggregated, indent=2))
    return aggregated


def _wandb_log(spec: BaselineSpec, summary: dict[str, Any]) -> None:
    """Best-effort W&B log; silent fallback if wandb unavailable / offline."""
    try:
        import wandb
    except ImportError:
        print(f"[wandb] skipped for {spec.run_name} (wandb not installed)")
        return

    if not os.environ.get("WANDB_API_KEY"):
        print(f"[wandb] skipped for {spec.run_name} (no WANDB_API_KEY)")
        return

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=WANDB_GROUP,
        name=spec.run_name,
        config={
            "model_id": spec.model_id,
            "model_class": spec.model_class,
            "expected_signal": spec.expected_signal,
            "benchmark": "ViDoRe(v3)",
            "subtask_count": 8,
            "evaluation_mode": "zero-shot",
        },
        reinit=True,
    )
    run.summary["ndcg_at_10_mean"] = summary["ndcg_at_10_mean"]
    run.summary["recall_at_10_mean"] = summary["recall_at_10_mean"]
    run.summary["map_mean"] = summary["map_mean"]
    run.summary["elapsed_seconds"] = summary["elapsed_seconds"]
    for task_name, scores in summary["per_subtask"].items():
        run.log({f"{task_name}/ndcg_at_10": scores["ndcg_at_10"]})
    run.finish()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (A100 80GB Colab Pro+ assumed).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    decision_rows: list[tuple[str, str, float | None]] = []

    for spec in BASELINES:
        print(f"\n=== {spec.run_name} ===")
        print(f"    Model: {spec.model_id}")
        print(f"    Expected: {spec.expected_signal}")

        model, processor = _load_baseline(spec)
        try:
            summary = _evaluate(spec, model)
        finally:
            del model, processor
            torch.cuda.empty_cache()

        _wandb_log(spec, summary)
        decision_rows.append((spec.run_name, spec.model_id, summary["ndcg_at_10_mean"]))

    print("\n=== Day-1 baseline summary (paste to KARARLAR Log) ===")
    for run_name, model_id, ndcg in decision_rows:
        ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "FAILED"
        print(f"  {run_name}: NDCG@10 mean = {ndcg_str}  ({model_id})")
    print("\nNext: pick Stage-1 base — if M3 leads, warm-start from Ops-Colqwen3-4B; else Qwen/Qwen3-VL-4B-Instruct from scratch.")


if __name__ == "__main__":
    main()
