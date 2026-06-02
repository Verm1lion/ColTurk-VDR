# Reproducibility — ColTurk-VDR Stage-1 (S52-F)

Anyone with the base model + the released adapter + this repo's eval script can
reproduce the reported ViDoRe V3 numbers exactly. Training and eval are seeded
(42) and the environment is pinned. Determinism was confirmed empirically: a
re-run reproduced the validation loss curve step-for-step, and the eval subset
selection is fully deterministic (first-N, no RNG).

## Environment (pinned — Colab A100 80GB, verified 2026-06-01)

| Package | Version |
|---|---|
| python | 3.12 |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| transformers | 5.9.0 |
| peft | 0.19.1 |
| colpali-engine | 0.3.16 |
| accelerate | 1.13.0 |
| torchao | 0.17.0 |
| mteb | 2.12.30 |
| configue | 6.0.1 |
| datasketch | 1.10.0 |
| imagehash | (pHash scan only) |
| GPU | NVIDIA A100-SXM4-80GB |

> transformers v5 is mandatory (colpali-engine 0.3.16 requires it). v4-era ColPali
> LoRA adapters are silent-broken under v5 (KARARLAR S45/F6) — our adapter is
> v5-native, trained on exactly this stack.

## Released artifacts

- **Base** (frozen): `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0).
- **Adapter + checkpoints**: `Verm1ion/ColTurk-VDR-Stage1` (HF Hub) — `checkpoint-N/`
  folders with `adapter_config.json` + `adapter_model.safetensors`
  (F28: `modules_to_save: custom_text_proj` → each checkpoint is independently
  reusable; the ColBERT projection head is saved, not just the LoRA deltas).
- **Config**: `configs/qwen3/train_colturk_stage1.yaml` (git).
- **Eval script**: `scripts/eval/eval_colturk_checkpoint.py` (git).
- **Results JSON**: `eval/results/*.json` (git committed) + mirrored to HF
  (`eval_results/` in the adapter repo, disconnect-proof).

## Training recipe (Stage-1, locked)

Base raw `Qwen/Qwen3-VL-4B-Instruct` · LoRA r=32 α=32 dropout=0.1
(`target_modules` = language_model attn+mlp; `modules_to_save: [custom_text_proj]`)
· LR 5e-5 (S47 sweep winner) · linear decay · warmup_steps 10 · adamw_torch ·
bf16 · effective batch 32 (per_device 2 × grad_accum 16) · `num_negs=2`
(Nemotron K=2 hard-neg) · `max_num_visual_tokens=768` · gradient_checkpointing ·
**seed 42** · corpus `manu/colpali` (EN+FR, ~108K queries after gold_in_top_100
filter ≈ 3385 steps = 1 epoch ceiling; curve-driven plateau stop).

```bash
python scripts/training/launch_stage1_sweep.py \
    configs/qwen3/train_colturk_stage1.yaml \
    --push-to-hub Verm1ion/ColTurk-VDR-Stage1 \
    [--resume-from-checkpoint /content/outputs/colturk-vdr-stage1/checkpoint-N]
```

## Eval — reproduce the numbers

All eval is a self-contained colpali-native MaxSim loop (no mteb wrapper; avoids
the Day-1 F6/F7 fragility). ViDoRe V3 8 public subtasks, split=test.

```bash
# Full-corpus official number (leaderboard-comparable) + 95% CI + NDCG@5/@10
python scripts/eval/eval_colturk_checkpoint.py \
    --adapter Verm1ion/ColTurk-VDR-Stage1  --bootstrap 1000 \
    --output eval/results/colturk_stage1_full.json

# 768-vs-default visual-token A/B (S52-E): add --max-visual-tokens 768 for train-match
# Binarized relevance (leaderboard also reports this, S52-C8): add --binarize
# Base control / causal floor (S52-D10):
python scripts/eval/eval_colturk_checkpoint.py --no-adapter \
    --query-limit 100 --output eval/results/base_control.json

# Curve (pick best checkpoint along a run):
python scripts/eval/eval_colturk_checkpoint.py \
    --checkpoints-root /content/outputs/colturk-vdr-stage1 \
    --corpus-limit 500 --query-limit 100 --output eval/results/curve.json
```

## Validity protocol (S52) — how the numbers are defended

- **Held-out**: ViDoRe V3 is never in training (manu/colpali EN+FR only).
- **Lockstep**: training loss↓ must track held-out NDCG↑ across checkpoints
  (overfit = loss↓ while NDCG flat/↓ on the *full* corpus).
- **Leakage tripwires**: no subtask anomalously ~1.0; cross-lingual EN≥FR
  (EN-only training ⇒ FR subtasks score lower — confirmed: finance_en > finance_fr).
- **Metric**: full-corpus (not smoke) NDCG@5 + NDCG@10, graded + binarized, with
  seeded bootstrap 95% CI.
- **Causal control**: `--no-adapter` floor ≪ trained NDCG.
- **Contamination**: `scripts/data/phash_contamination_scan.py` (pHash overlap %,
  expectation ~0; image-based so S41 text-MinHash does not apply).
- **768-vs-default**: visual-token cap A/B to confirm the speed lever didn't
  distort retrieval.

> Smoke eval (`--corpus-limit 500`) is deliberately easy (few distractors) → its
> NDCG is inflated and NOT leaderboard-comparable; use it only for trend + tripwires.
> Defensible numbers come from the full-corpus eval (omit `--corpus-limit`).
