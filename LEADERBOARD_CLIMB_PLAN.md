# ColTurk-VDR — ViDoRe V3 Leaderboard Climb Plan (Orta tier, locked 2026-06-05)

## Context
Stage-1 training (Qwen3-VL-4B + LoRA, colpali-engine 0.3.16) is converging. Measured full-corpus all-query NDCG@10: ckpt-500 = 0.5441, ckpt-1000 = 0.5584 (8/8 subtasks up, +0.0143), ckpt-1500 saved (eval pending). Training-more is now **marginal** (diminishing returns). A 7-method cross-validated deep-research round (Workflow `wtdstrjb7` + 2 follow-up agents) determined the highest-leverage methods BEYOND training-more, their realistic (overlap-aware) score contributions, our projected rank, and the Turkish go/no-go. This doc is the locked execution plan.

**Goal (user):** land high on the **ViDoRe V3 single-stage RETRIEVER** leaderboard (no reranker). Turkish only if it doesn't hurt rank. Submission deferred until score+rank are estimated.

**Effort tier (user choice): ORTA — high-confidence levers.** Skip the risky/expensive ones (bidirectional retrain, heavy data-scaling). Target ~50-80 GPU-hours (~10-15 Colab A100 sessions), projected ~0.58-0.595 NDCG@10 → rank ~6-8.

## Current standing (V3 retriever leaderboard, Feb 2026 — Nemotron paper Table 2)
| # | Model | NDCG@10 |
|---|---|---|
| 1 | nemotron-colembed-8b | 63.42 |
| 2 | tomoro-colqwen3-8b | 61.59 |
| 3 | nemotron-colembed-4b | 61.54 |
| 4 | Ops-Colqwen3-4b | 61.17 |
| 5 | tomoro-colqwen3-4b | 60.20 |
| 6 | nemotron-colembed-3b | 59.79 |
| 7 | jina-embeddings-v4 (7B) | 57.52 |
| 8 | colnomic-7b | 57.33 |
| 9 | llama-colembed-3b-v1 | 57.26 |
| **~10-11** | **OURS (ckpt-1000)** | **55.84** |

## Honest ceiling (adversarial verification corrected earlier over-optimism)
- **Dominant gap = DATA SCALE:** our ~108K queries vs Nemotron's 12.5M+ (116×) → alone −1.5 to −2.5 pts. This is the single biggest reason for the 5.7-pt gap to Nemotron-4B (same Qwen3-VL-4B base).
- **Realistic ceiling ~0.595-0.61 → rank ~5-6.** ColEmbed-3B (59.79) is reachable; **Nemotron-4B 61.54 and top-3 are NOT reachable** on single-A100 + 108K data.
- Old plan's "63-64 top-1" was naive headline-summing — **retracted**. Realistic ranks: Orta tier → 6-8; full recipe → 5-6.

## Locked decisions
- **TURKISH: NO-GO for V3.** V3 has zero Turkish (EN+FR docs; EN/FR/ES/DE/IT/PT queries). Even related multilingual (ES/DE/IT) scores −0.3% vs EN-only on V3. Turkish → dilutes EN/FR via negative transfer, zero V3 upside. (Turkish stays a *separate* ViDoRe-TR/portfolio deliverable, not for rank.) [arxiv 2010.03017, vdr-2b-multilingual blog]
- **ES/DE/IT multilingual: SKIP for V3** (neutral-to-negative). Keep EN+FR focus (manu/colpali already EN+FR).
- **Submission: DEFERRED** until full recipe done + rank estimated.
- **Effort: ORTA tier** (below).

## Verified levers (overlap-aware realistic gains)
| Lever | Realistic gain | Effort | Confidence | Orta tier? |
|---|---|---|---|---|
| ~~Hard-neg / num_negs 2→4~~ — **FAILED & DROPPED 2026-06-07** | predicted +0.8-1.5 → **measured −0.0158 @500 (8/8 subtasks lower); K=2 optimal** | medium | high → **REFUTED** | ❌ |
| Test-time: eval max_visual_tokens A/B + score-norm | +0.05-0.3 | low | medium | ✅ |
| Diverse-run merge (2 runs, diff seed/LR) | +0.4-1.0 | high | medium | ✅ |
| Finish Stage-1 (eval 1500, lock best) | (baseline) | low | high | ✅ |
| Bidirectional (done right) | +0 to +2.5 (uncertain) | medium | low | ❌ SKIP (Tam tier only) |
| EN/FR data scale-up | closes dominant gap | high | medium | ❌ SKIP (Tam tier) |
| Projection head / distillation | +0.1-0.8 | medium | low | ⏸ LATER |
| Turkish / ES-DE-IT multilingual | −0.3 to 0 | — | high | ❌ NEVER (V3) |

**Key nuances:** (1) our 6 checkpoints are SAME-run (low diversity) → soup of those only +0.1-0.4; the +0.4-1.0 merge gain requires **diverse runs** (different seed/LR). (2) merge_checkpoints.py assumes full models — needs LoRA-adapter-averaging fix. (3) mine_hard_negatives.py output is NOT wired into training — needs an integration layer.

## Execution sequence (Orta tier, eval-gated)

### Phase A — Finish Stage-1, lock best single checkpoint — ✅ DONE (2026-06-06)
- Full-corpus all-q curve (NDCG@10): **500 = 0.5441 → 1000 = 0.5584 (PEAK) → 1500 = 0.5518 (regressed)**.
- 1000→1500: 6/8 subtasks down (finance_en −0.021, cs −0.013 both CI-significant), net −0.0066; NDCG@5 + recall also down.
- **S52 lockstep tripwire fired:** 1000→1500 loss↓ (floor ~0.09) but held-out NDCG↓ = mild overfitting onset. The 108K manu/colpali data is exhausted at ~0.558.
- **LOCKED: ckpt-1000 = Stage-1 v0.1 (NDCG@10 0.5584), the best single checkpoint.** Training-more is over (1500/1750/2000 wasted). All recipe phases build on ckpt-1000. Confirms the data-scale bottleneck (Orta levers add ~+0.03; bigger gains would need the Tam-tier data work).

### Phase B — Test-time max_visual_tokens — ❌ TESTED & FAILED (2026-06-11)
- 768 train-match FULL-corpus all-q eval on ckpt-1000: **0.5419 < 0.5584 default (−0.0165, outside CI).** The "train-eval match" hypothesis was WRONG: more visual tokens at inference = more document detail; capping to 768 loses information (worst hit: finance_fr 0.401 — dense FR docs need resolution). The model generalizes across resolutions (NaFlex), so processor-default (uncapped max) is already optimal → the token-count axis is CLOSED (no 1024+ attempts).
- **Official eval setting: processor-default.** Detail → KARARLAR S55, JSON: `eval_results/ColTurk-VDR-Stage1_checkpoint-1000_vis768_allq.json`.

### Phase C — num_negs 2→4 — ❌ FAILED & DROPPED (2026-06-07)
- Fresh num_negs=4 run (per_device=2 per F29: in-batch `topk(2)` needs batch≥2), full-corpus all-q eval: **ckpt-500 = 0.5283, ckpt-625 = 0.5357** vs num_negs=2 **ckpt-500 = 0.5441** → **−0.0158 @500, lower on all 8/8 subtasks**, 500→625 slope parallel (not catching up), best-case linear extrapolation ≈ tie at peak (~0.558).
- **Cause:** ColPali `negative_passages[2:4]` (3rd/4th mined negs) are noisier than top-2 → dilute the contrastive signal. Nemotron K=2 (= num_negs=2) was already optimal. **Valuable negative result: K=2 confirmed; num_negs↑ lever refuted.**
- **Decision:** Stage-1 base stays num_negs=2 ckpt-1000 (0.5584). num_negs=4 ckpts (125–875) archived on HF, unused. C2 full FAISS mining NOT pursued (num_negs↑ already shown to hurt). Detail → STAGE1_VALIDITY_REPORT.md "Phase C" section + KARARLAR S51.

### Phase D — Diverse-run merge — ❌ TESTED & FAILED (2026-06-11)
- **D1 done:** run B (seed 1234, num_negs=2, proven recipe) trained to step 1786 (peak ~1000 overshoot; loss plateau ~0.13; all ckpts on HF `Verm1ion/ColTurk-VDR-Stage1-seed1234`).
- **D2 done:** full-model soup A(ckpt-1000) + B(ckpt-1000) 0.5/0.5 via `soup_lora_adapters.py`, full-corpus all-q eval = **0.5521 < A=0.5584 (−0.0063)**. The blend curve is LINEAR (0.5521 ≈ midpoint; inferred B@1000 ≈ 0.546) → **zero soup synergy.**
- **Root cause:** changing `seed` 42→1234 changed the LoRA **gaussian init** along with data order → A and B live in different basins; classical soups (Wortsman/Nemotron) average runs branched from the SAME init. Lesson: LoRA soup diversity must vary data-order/LR while HOLDING the init fixed (colpali config's single `seed` can't separate them without a code change).
- **Decision:** soup lever empirically eliminated (like num_negs; neutral, not harmful). **FINAL model = A ckpt-1000 (0.5584).** Weight sweep skipped (no-synergy signal). Detail → KARARLAR S54.

### Phase E — Submission prep (deferred, no GPU)
- Build MTEB **ModelMeta** (`late_interaction=True`, vision) + VisionRetriever wrapper (entirely TODO). Estimate final rank from leaderboard table. Then submit to `embeddings-benchmark/mteb` + `results`.

### Phase F — Finalize
- Update STAGE1_VALIDITY_REPORT (resolve 250-vs-500 lockstep with the 250/500/1000/1500 curve), KARARLAR S51 (Stage-1 result + honest ceiling) + S52 (validity protocol run) + S53 (research synthesis + Orta-tier lock + Turkish NO-GO).

## Score & rank projection — FINAL (2026-06-11; num_negs AND soup both measured & failed)
Both headline levers were empirically tested on this exact stack and eliminated: num_negs 2→4 (−0.0158) and diverse soup (−0.0063, zero synergy). Deep-research verdict (wf wqp47b3wl, 21 agents, web-verified): safe top-10 (≥0.585) P≈12-20%; the de-risked path tops out ~0.563-0.572; user chose optimal/de-risked (close-stretch rank ~10-11 acceptable).
- **FINAL model: ckpt-1000 @ processor-default = 0.5584 → rank ~10-11** (top-10 wall = 57.26-57.52 cluster at ranks 7-9).
- 768 train-match was the last upside attempt: **0.5419, FAILED (−0.0165)** → official setting stays processor-default. All three measured levers (num_negs −0.0158, soup −0.0063, 768 −0.0165) predicted positive, measured negative — 0.5584 is this stack's real ceiling, scientifically established.
- Skipped by decision: weight sweep (no-synergy), bidirectional gamble, hard-neg re-mine, data scale-up (high risk / low ROI for close-stretch goal).
- **→ All GPU work DONE. Remaining: Phase E submission only** (merged-model publish + MTEB ModelMeta/results PRs).

## Explicitly SKIP (with reason)
- **Bidirectional retrain** — biggest potential lever (+2.5) but most uncertain (LoRA+bidi untested; LLM2Vec shows pure-bidi-without-MNTP HURTS; Nemotron's +4.2% not isolated). Tam-tier only, gated on a 250-step pilot.
- **Heavy EN/FR data scale-up** — addresses the dominant gap but high effort (dataset sourcing + render); Tam tier.
- **Turkish / ES-DE-IT** — net-negative on V3.
- **Matryoshka, MUVERA/fixed-dim** — unvalidated on visual late-interaction / known to hurt NDCG.

## Engineering gaps to build (local, no GPU — can prep while GPU phases run)
1. **Mined-negs integration layer** (Phase C2): consume `mine_hard_negatives.py` JSONL as training negatives.
2. **LoRA-adapter merge fix** (Phase D2): `merge_checkpoints.py` currently assumes full models; add adapter-averaging incl. `custom_text_proj`.
3. **MTEB ModelMeta + VisionRetriever wrapper** (Phase E): none exists yet.
4. *(optional)* per-subtask score normalization in `eval_colturk_checkpoint.py`.

## Verification
- Every phase gated by full-corpus all-q eval (`eval_colturk_checkpoint.py`, bootstrap CI), result JSON pushed to HF (disconnect-proof).
- A method is kept only if its eval delta exceeds noise (~> CI half-width, ~0.005-0.01). If a lever doesn't move the needle, drop it and move on (don't sink GPU).
- Final: merged model all-q full-corpus number → map to leaderboard table → decide submission.

## Critical files
- `kaggle_drafts/colturk-vdr-stage1-train.ipynb` — Cell 8 eval (ready); add test-time A/B + diverse-run cells.
- `configs/qwen3/train_colturk_stage1.yaml` — num_negs 2→4 (Phase C1); 2nd-run variant (seed/LR) for Phase D.
- `scripts/training/mine_hard_negatives.py` + new integration layer (Phase C2).
- `scripts/training/merge_checkpoints.py` — LoRA-adapter merge fix (Phase D2).
- `scripts/eval/eval_colturk_checkpoint.py` — `--max-visual-tokens` (exists); optional score-norm.
- New: MTEB ModelMeta + wrapper (Phase E).
- `cv_project_bootstrap/KARARLAR.md` — S51/S52/S53 (Phase F).
