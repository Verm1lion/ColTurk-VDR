# ColTurk-VDR

> **v1.0 · 2026-06** — ColBERT-style late-interaction **visual document retriever** built on **Qwen3-VL-4B-Instruct** (Apache-2.0) via `colpali_engine.models.ColQwen3`. Evaluated on the full ViDoRe V3 public benchmark: **NDCG@10 = 0.5584** (8 subtasks, full corpus, all queries, bootstrap 95% CI).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Base Model: Apache-2.0](https://img.shields.io/badge/Base-Apache--2.0-blue.svg)](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ColTurk--VDR--Qwen3VL--4B-orange.svg)](https://huggingface.co/Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0)
[![Submission: MTEB v2](https://img.shields.io/badge/submission-MTEB%20v2-blue.svg)](https://github.com/embeddings-benchmark/mteb)

---

## What is this?

ColTurk-VDR is a multi-vector late-interaction (ColBERT/MaxSim) **visual document retriever**: it embeds document **page images** and text queries into per-token 128-dim vectors and ranks pages by MaxSim — no OCR, layout-aware, multilingual-capable. It is trained with LoRA (r=32) on a **single A100 80GB** from the raw `Qwen/Qwen3-VL-4B-Instruct` base using the [colpali-engine](https://github.com/illuin-tech/colpali) training stack (transformers v5 native).

The long-term goal is Turkish enterprise documents (e-invoices, KYC, legal, financial); v1.0 is the Stage-1 EN+FR foundation model, evaluated and submitted on ViDoRe V3.

## Results — ViDoRe V3 (8 public subtasks, full corpus, all queries)

**Official: NDCG@10 = 0.5584 · NDCG@5 = 0.5287 · recall@10 = 0.6110** (checkpoint-1000, processor-default visual tokens, seeded bootstrap 95% CI; raw JSONs in [`eval/results/`](eval/results/)).

| Subtask | NDCG@10 | 95% CI |
|---|---|---|
| computer_science | 0.7306 | [0.718, 0.743] |
| energy | 0.6238 | [0.608, 0.638] |
| pharmaceuticals | 0.6156 | [0.602, 0.629] |
| finance_en | 0.5851 | [0.571, 0.601] |
| hr | 0.5463 | [0.532, 0.560] |
| industrial | 0.4624 | [0.445, 0.482] |
| physics | 0.4564 | [0.443, 0.471] |
| finance_fr | 0.4467 | [0.430, 0.463] |

Training: 1000 steps (effective batch 32, LR 5e-5 linear, ~0.3 epoch of the 108K-pair [manu/colpali](https://huggingface.co/datasets/manu/colpali-queries) EN+FR set, `num_negs=2`, bf16, gradient checkpointing). The checkpoint curve peaks at step 1000 (500 → 0.5441, **1000 → 0.5584**, 1500 → 0.5518 = overfit onset); checkpoint selection is eval-gated, not loss-gated.

### Measured negative results (transparency)

Every additional lever was evaluated on the full benchmark with the same harness and **dropped on evidence**:

| Lever | Predicted | Measured | Verdict |
|---|---|---|---|
| `num_negs` 2→4 (more mined negatives) | positive | **−0.016** (worse on 8/8 subtasks) | K=2 optimal |
| Diverse-run weight averaging (2 runs, different seed) | positive | **−0.006** (linear blend, zero synergy) | seed change also changes LoRA init → different basins |
| Train-match visual-token cap (768) at eval | positive | **−0.017** | more inference tokens = more detail; uncapped optimal |

Full analysis: [STAGE1_VALIDITY_REPORT.md](STAGE1_VALIDITY_REPORT.md) (causal control, leakage tripwires, pHash contamination scan, bootstrap CIs, reproducibility).

## Usage

```python
import torch
from colpali_engine.models import ColQwen3, ColQwen3Processor

model = ColQwen3.from_pretrained(
    "Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0",   # merged full model — loads directly
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
).eval()
processor = ColQwen3Processor.from_pretrained("Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0")

# documents = list of PIL page images; queries = list of strings
doc_batch = processor.process_images(documents).to(model.device)
qry_batch = processor.process_queries(queries).to(model.device)
with torch.no_grad():
    doc_emb = model(**doc_batch)
    qry_emb = model(**qry_batch)
scores = processor.score_multi_vector(qry_emb, doc_emb)   # (n_queries, n_docs) MaxSim
```

Requirements: `colpali-engine>=0.3.16`, `transformers>=5.0`, `torch>=2.5`. The published repo is a **merged full model** (LoRA baked in) — no PEFT loading step, no adapter key-prefix issues across transformers versions.

## Reproduce the evaluation

```bash
python scripts/eval/eval_colturk_checkpoint.py \
    --adapter Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0 \
    --bootstrap 1000 --output eval/results/repro.json
```

The harness downloads the 8 ViDoRe V3 public subtasks (`vidore/vidore_v3_*`, split `test`), encodes the **full corpus** and **all queries**, scores with MaxSim, and reports NDCG@5/@10 + recall with a seeded bootstrap CI. Environment pins: [REPRODUCIBILITY.md](REPRODUCIBILITY.md) (seed 42; transformers 5.9 / peft 0.19 / colpali-engine 0.3.16 / torch 2.11).

## Repository map

```
configs/qwen3/          training configs (Stage-1 + ablation variants, all eval-gated)
scripts/training/       launcher (resume + HF checkpoint push), weight-averaging, attention experiments
scripts/eval/           ViDoRe V3 eval harness (full-corpus MaxSim + bootstrap CI), MTEB results builder
scripts/data/           corpus tooling, pHash contamination scan, synthetic-data QC
src/inference/          shared encode/MaxSim utilities
src/models/             MTEB integration wrapper
eval/results/           raw result JSONs (official numbers)
```

## Roadmap

- [x] Stage-1 EN+FR foundation (v1.0, this release) + ViDoRe V3 submission
- [ ] Stage-2 Turkish continual fine-tune (synthetic TR corpus pipeline is built; KVKK-compliant)
- [ ] ViDoRe-TR: public Turkish visual-retrieval split (BEIR format)
- [ ] Serving stack (FastAPI + Qdrant multi-vector + Docker Compose)

## License

- **Code:** MIT — [LICENSE](LICENSE)
- **Model weights:** inherit Apache-2.0 from the base model; dataset/license matrix in [LICENSE-NOTICE.md](LICENSE-NOTICE.md)
- **KVKK:** no real PII in any published artifact; Turkish data work uses synthetic or public-domain sources only.

## Author

[**Mert Karatay**](https://github.com/Verm1lion) — AI & Network Security Engineer, İstanbul · [HuggingFace: Verm1ion](https://huggingface.co/Verm1ion) · merttkaratayy@gmail.com

## Citation

```bibtex
@misc{karatay2026colturkvdr,
  author = {Karatay, Mert},
  title  = {ColTurk-VDR: A Late-Interaction Visual Document Retriever on Qwen3-VL-4B},
  year   = {2026},
  url    = {https://github.com/Verm1lion/ColTurk-VDR}
}
```

## Acknowledgments

- [ColPali / colpali-engine](https://github.com/illuin-tech/colpali) — late-interaction visual retrieval framework (Faysse et al., ICLR 2025)
- [ViDoRe V3](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3) — benchmark and evaluation datasets
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Apache-2.0 base vision-language model
- [NVIDIA Nemotron ColEmbed v2](https://arxiv.org/abs/2602.03992) — training-recipe reference (hard negatives, K=2)
