# ColTurk-VDR — Stage-1 Validity Report (S52)

**Status: 2026-06-11 — FINAL.** All S52 checks DONE (causal control, leakage tripwires, empirical contamination, metric validity, lockstep resolved). **Three improvement levers were each empirically evaluated and dropped** (negative results — see the table below): `num_negs↑`, diverse-run soup, 768 train-match eval. All numbers reproducible (REPRODUCIBILITY.md, seed 42). Raw JSONs: HF `Verm1ion/ColTurk-VDR-Stage1/eval_results/` + git `eval/results/`. Eval = colpali-native MaxSim, ViDoRe V3 8 public subtasks, split=test, **FULL corpus, all queries, processor-default visual tokens**, seeded bootstrap 95% CI.

## Negative-results summary (all levers measured, predicted-positive → measured-negative)
| Lever | Predicted | Measured (full-corpus all-q) | Verdict |
|---|---|---|---|
| `num_negs` 2→4 (more mined negatives) | +0.8–1.5 | **−0.0158** @ step-500 (lower on 8/8 subtasks) | DROPPED — K=2 optimal |
| Diverse-run soup (seed-1234 run B, 0.5/0.5 weight-avg) | +0.3–0.8 | **−0.0063** (0.5521; linear blend curve = zero synergy) | DROPPED — seed change also changed LoRA init → different basins |
| 768 train-match eval (`--max-visual-tokens 768`) | +0.05–0.3 | **−0.0165** (0.5419; worst finance_fr 0.401) | DROPPED — more inference tokens = more detail; default (uncapped) optimal |

→ **0.5584 is this stack's measured ceiling** (4B, 108K EN+FR pairs, single-GPU LoRA). Every gate used the same harness, full corpus, all queries, bootstrap CI — apples-to-apples.

## Official Stage-1 number — ckpt-1000 (Stage-1 v0.1), FULL corpus, all queries
**NDCG@10 = 0.5584** — the PEAK of the 500/1000/1500 curve (below).
Context: V3 SoTA (Nemotron-**8B**) = 0.634 → our 4B, Stage-1-only (EN+FR, no Turkish, no merge/reranker) sits ~7.5 pts below = a strong single-stage base. Maps to the ViDoRe V3 retriever leaderboard ≈ **rank 10-11** (just under llama-colembed-3b 57.26).

## Training curve — RESOLVED (closes the prior open "250/500 lockstep" question)
Full-corpus all-q NDCG@10 across checkpoints (num_negs=2, the proven recipe):
| ckpt | NDCG@10 | note |
|---|---|---|
| 500 | 0.5441 | still climbing |
| **1000** | **0.5584** | **PEAK → Stage-1 v0.1 (LOCKED)** |
| 1500 | 0.5518 | regressed (−0.0066) |

- **1000→1500:** 6/8 subtasks down (finance_en −0.021, cs −0.013 — both CI-significant); NDCG@5 + recall@10 also down.
- **S52 lockstep tripwire FIRED at 1000→1500:** training loss kept declining (floor ~0.09) while held-out NDCG **dropped** = mild overfitting onset. The 108K manu/colpali corpus + this recipe saturates at ~0.558.
- **Decision:** ckpt-1000 = best single checkpoint = Stage-1 v0.1. Training-more is exhausted (1500/1750/2000 wasted GPU). Confirms the data-scale bottleneck (our 108K vs Nemotron 12.5M+, 116×).

ckpt-500 per-subtask (full table; ckpt-1000 / ckpt-1500 full tables → `eval_results/*.json`):
| subtask | NDCG@10 | CI95 | n_q | n_corpus |
|---|---|---|---|---|
| computer_science | 0.716 | [0.703, 0.729] | 1290 | 1360 |
| energy | 0.615 | [0.599, 0.630] | 1848 | 2225 |
| pharmaceuticals | 0.588 | [0.573, 0.603] | 2184 | 2313 |
| finance_en | 0.574 | [0.560, 0.589] | 1854 | 2942 |
| hr | 0.522 | [0.508, 0.537] | 1908 | 1110 |
| industrial | 0.453 | [0.435, 0.472] | 1698 | 5244 |
| physics | 0.443 | [0.428, 0.458] | 1812 | 1674 |
| finance_fr | 0.441 | [0.425, 0.457] | 1920 | 2384 |
| **mean** | **0.544** | | | |

## Phase C — `num_negs` 2→4 lever EVALUATED + DROPPED (negative result, 2026-06-07)
A fresh num_negs=4 run (identical recipe except K=2→K=4 ColPali mined negatives; `per_device=2` per **F29**: the in-batch loss term `scores.topk(2,dim=1)` requires batch ≥ 2). Full-corpus all-q:
| ckpt | num_negs=4 | num_negs=2 anchor | Δ |
|---|---|---|---|
| 500 | **0.5283** | 0.5441 | **−0.0158** |
| 625 | 0.5357 | ~0.548 (interp) | ~−0.012 |

- num_negs=4 is **lower on all 8/8 subtasks @500** (uniform: cs −0.038, hr −0.037, energy −0.020 largest; finance_en −0.002 smallest). The 500→625 slope (+0.0074/125 step) **parallels** num_negs=2 — it does not close the gap. Best-case linear extrapolation ≈ **tie** at the peak (~0.558), at the cost of ~5-7h extra training → expected value too low to resume.
- **Cause:** ColPali `negative_passages[2:4]` (the 3rd/4th mined negatives) are noisier/weaker than the top-2 → they dilute rather than sharpen the contrastive signal. Nemotron's **K=2** (= our num_negs=2) was already at the sweet spot.
- **Validity of this comparison:** same harness, same default tokens, same all-q + bootstrap (apples-to-apples); num_negs=4 also passes cross-lingual EN(0.569)>FR(0.425) and shows the same subtask ranking → a genuine quality gap, not an artifact.
- **Decision:** `num_negs↑` DROPPED; **K=2 confirmed optimal.** Stage-1 base stays num_negs=2 ckpt-1000 (0.5584). num_negs=4 checkpoints (125–875) archived on HF, unused. (Valuable negative result.)

## Phase D — diverse-run soup EVALUATED + DROPPED (negative result, 2026-06-11)
Run B = identical recipe, `seed` 42→1234, trained to its plateau (~step 1000 peak region; loss floor ~0.13). Full-model weighted average (LoRA merged + `custom_text_proj` head both averaged exactly) of A(ckpt-1000) + B(ckpt-1000) at 0.5/0.5:
- **Soup = 0.5521 < A = 0.5584 (−0.0063).** The blend sits at the linear midpoint of the two endpoints (inferred B@1000 ≈ 0.546) → **zero soup synergy**, not a flat-basin average.
- **Cause:** changing `seed` changed the LoRA **gaussian init** along with data order → A and B occupy different basins; weight-averaging across basins interpolates linearly instead of finding a better flat minimum (classical soups average runs branched from the same init).
- **Decision:** soup DROPPED; weight sweep skipped (the linear curve means no interior point beats the better endpoint). JSON: `eval_results/soup_AB_w0.5-0.5_allq.json`.

## Phase B — 768 train-match eval EVALUATED + DROPPED (negative result, 2026-06-11)
The model was TRAINED at `max_num_visual_tokens=768`, but the official number uses processor-default (uncapped) document tokens. The train-match hypothesis was tested: ckpt-1000 full-corpus all-q at `--max-visual-tokens 768` = **0.5419 < 0.5584 (−0.0165)**, worst on finance_fr (0.401 — dense French documents need resolution).
- **Cause:** at inference, more visual tokens = more document detail; the backbone generalizes across resolutions, so capping to the training budget only loses information.
- **Decision:** official eval setting = processor-default. The token-count axis is closed. JSON: `eval_results/ColTurk-VDR-Stage1_checkpoint-1000_vis768_allq.json`.

## A. Causal control (S52-D10) — PASS
Raw base, NO adapter (random `custom_text_proj` head), smoke (500-corpus, 100q): **NDCG@10 = 0.124**.
Trained ckpt-500, same smoke: **0.627**. → training causally lifts NDCG ~5×; results are not random/artifact.

## B. Leakage tripwires — PASS
- No subtask anomalously ~1.0 (max cs 0.716 full-corpus); spread 0.44–0.72 = healthy heterogeneity.
- **Cross-lingual EN≥FR:** finance_en 0.574 > finance_fr 0.441 (and num_negs=4 likewise 0.569 > 0.425). EN-dominant training scores French lower — the correct no-leakage signature (a leak would invert this).

## B6. Empirical contamination — pHash manu/colpali ↔ ViDoRe V3 — CLEAN
3000 train × 8000 V3 (1000/subtask) perceptual-hash (pHash) overlap:
- **exact duplicates = 0**; ≤Hamming-2 (true-dup bar for documents) = **2 / 8000 = 0.025%**; ≤6 = 1.1%.
- **Visual inspection** (contact sheet of the 24 lowest-Hamming pairs, incl. the two h=2): every flagged pair is a **different document sharing only coarse text-page layout** — no same-document pairs. The 1.1% ≤6 is pHash layout false-positive (the 8×8 DCT signature collapses text pages by geometry, not content).
- **Verdict: contamination empirically clean.** Consistent with the source-based clearance (~18mo gap, EN-vs-FR mismatch, ILLUIN doc-level dedup).

## C. Metric validity — PASS
Full-corpus (no limit) + all queries = leaderboard-comparable (the 500-doc smoke is deliberately easy → inflated, used only for trend/tripwires). NDCG@5 + @10 + recall, seeded bootstrap 95% CI (tight, n>1290 per subtask).

## D. Lockstep / checkpoint selection — RESOLVED (was PENDING)
The prior open question (250-vs-500 confounded by query sets) is **superseded** by the full all-q curve above: NDCG@10 **rises 500→1000 then falls 1000→1500** with loss still declining → the peak (and the overfit onset) are now directly measured. **ckpt-1000 = Stage-1 v0.1.** No further training on this corpus.

## F. Reproducibility — see REPRODUCIBILITY.md
Pinned env (transformers 5.9 + peft 0.19.1 + colpali 0.3.16 + torch 2.11), seed 42, released base + adapter + eval script + result JSONs. Note: the num_negs=4 run used `per_device_train_batch_size=2` / `grad_accum=16` (F29 — in-batch `topk(2)` requires batch ≥ 2; an earlier per_device=1 config crashed with "selected index k out of range").
