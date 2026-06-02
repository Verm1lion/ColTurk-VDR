# ColTurk-VDR — Stage-1 Validity Report (S52)

**Status: 2026-06-02 — contamination + causal + metric checks DONE; one check pending (clean 250-vs-500).**
All numbers reproducible (REPRODUCIBILITY.md, seed 42). Raw JSONs: HF `Verm1ion/ColTurk-VDR-Stage1/eval_results/` + git `eval/results/`. Eval = colpali-native MaxSim, ViDoRe V3 8 public subtasks, split=test.

## Official Stage-1 number — ckpt-500, FULL corpus, all queries
**NDCG@10 = 0.5441 · NDCG@5 = 0.5146 · recall@10 = 0.599.**
Context: V3 SoTA (Nemotron-**8B**) = 0.634 → 4B + Stage-1-only (EN+FR, no Turkish, no merge/reranker) sits ~9 pts below, a strong base before the bigger levers.

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

## A. Causal control (S52-D10) — PASS
Raw base, NO adapter (random `custom_text_proj` head), smoke (500-corpus, 100q): **NDCG@10 = 0.124**.
Trained ckpt-500, same smoke: **0.627**. → training causally lifts NDCG ~5×. Results are not random/artifact.

## B. Leakage tripwires — PASS
- No subtask anomalously ~1.0 (max cs 0.716 full-corpus).
- **Cross-lingual EN≥FR:** finance_en 0.574 > finance_fr 0.441. EN-dominant training scores French lower — correct no-leakage signature (a leak would invert this).

## B6. Empirical contamination — pHash manu/colpali ↔ ViDoRe V3 — CLEAN
3000 train × 8000 V3 (1000/subtask) perceptual-hash (pHash) overlap:
- **exact duplicates = 0**; ≤Hamming-2 (true-dup bar for documents) = **2 / 8000 = 0.025%**; ≤6 = 1.1%.
- **Visual inspection** (contact sheet of the 24 lowest-Hamming pairs, incl. the two h=2): every flagged pair is a **different document sharing only coarse text-page layout** — no same-document pairs. The 1.1% ≤6 is pHash layout false-positive (expected: 8×8 DCT signature collapses text pages by geometry, not content).
- **Verdict: contamination empirically clean.** Consistent with the source-based clearance (~18mo gap, EN-vs-FR, ILLUIN doc-level dedup).

## C. Metric validity — PASS
Full-corpus (no limit) + all queries = leaderboard-comparable (the 500-doc smoke is deliberately easy → inflated, used only for trend/tripwires). NDCG@5 + @10, seeded bootstrap 95% CI (tight, n>1300).

## D. Lockstep / 250-vs-500 — PENDING
- Matched smoke (both 100q, 500-corpus): ckpt-250 **0.6267** ≈ ckpt-500 **0.6273** (flat).
- Full corpus: ckpt-250 @100q **0.492** vs ckpt-500 @all-q **0.544** — **CONFOUNDED** (different query sets).
- **TODO:** run ckpt-250 at all-q (full corpus, matched to ckpt-500) → clean 250-vs-500 delta → decides **still-improving (resume training)** vs **plateau (lock ckpt-500 = Stage-1 v0.1, go to merge/reranker/Stage-2)**. We stopped at step 500 of ~3385 (≈15% of 1 epoch) with loss still declining, so the question is open.

## F. Reproducibility — see REPRODUCIBILITY.md
Pinned env (transformers 5.9 + peft 0.19 + colpali 0.3.16 + torch 2.11), seed 42, released base + adapter + eval script + result JSONs.
