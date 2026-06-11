---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
language:
- en
- fr
tags:
- vidore
- colpali
- colqwen3
- late-interaction
- visual-document-retrieval
- multimodal
- retrieval
datasets:
- manu/colpali-queries
- manu/colpali-corpus
pipeline_tag: visual-document-retrieval
library_name: colpali
---

<p align="center">
  <img src="https://huggingface.co/Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0/resolve/main/assets/banner.png" alt="banner" width="100%" />
</p>

<p align="center">
  <a href="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3"><img src="https://img.shields.io/badge/ViDoRe%20V3-NDCG%4010%200.5584-4dd0e1?style=flat-square" /></a>
  <a href="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"><img src="https://img.shields.io/badge/Base-Qwen3--VL--4B-6f42c1?style=flat-square" /></a>
  <a href="https://huggingface.co/Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square" /></a>
  <a href="https://github.com/Verm1lion/ColTurk-VDR"><img src="https://img.shields.io/badge/GitHub-ColTurk--VDR-181717?style=flat-square&logo=github" /></a>
</p>

# ColTurk-VDR-Qwen3VL-4B v1.0

ColBERT-style **late-interaction visual document retriever** built on [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) with the [colpali-engine](https://github.com/illuin-tech/colpali) `ColQwen3` architecture (transformers v5 native). Pages are embedded as multi-vector 128-dim patch/token embeddings; queries and documents are scored with MaxSim.

This repository contains the **merged full model** (LoRA weights baked into the base) — it loads directly with `ColQwen3.from_pretrained`, with no PEFT step and no adapter key-prefix fragility across transformers versions. The original LoRA adapter is preserved under [`adapter/`](./tree/main/adapter) for reproducibility.

- **Developed by:** [Mert Karatay](https://github.com/Verm1lion) (merttkaratayy@gmail.com)
- **Model type:** multi-vector late-interaction visual retriever (ColBERT/MaxSim)
- **Languages:** English + French (training data); query side inherits Qwen3-VL multilinguality
- **License:** Apache-2.0 (inherited from the base model; training code MIT)
- **Repository / eval code:** https://github.com/Verm1lion/ColTurk-VDR

## Results — ViDoRe V3 (8 public subtasks)

Evaluated on the **full corpus with all queries** per subtask (no sampling), MaxSim scoring, processor-default visual tokens, seeded bootstrap 95% CI. Raw JSONs: [`eval/results/`](https://github.com/Verm1lion/ColTurk-VDR/tree/main/eval/results).

**Mean NDCG@10 = 0.5584 · NDCG@5 = 0.5287 · recall@10 = 0.6110**

| Subtask | NDCG@10 | 95% CI | n_queries | n_corpus |
|---|---|---|---|---|
| Vidore3ComputerScienceRetrieval | 0.7306 | [0.718, 0.743] | 1290 | 1360 |
| Vidore3EnergyRetrieval | 0.6238 | [0.608, 0.638] | 1848 | 2225 |
| Vidore3PharmaceuticalsRetrieval | 0.6156 | [0.602, 0.629] | 2184 | 2313 |
| Vidore3FinanceEnRetrieval | 0.5851 | [0.571, 0.601] | 1854 | 2942 |
| Vidore3HrRetrieval | 0.5463 | [0.532, 0.560] | 1908 | 1110 |
| Vidore3IndustrialRetrieval | 0.4624 | [0.445, 0.482] | 1698 | 5244 |
| Vidore3PhysicsRetrieval | 0.4564 | [0.443, 0.471] | 1812 | 1674 |
| Vidore3FinanceFrRetrieval | 0.4467 | [0.430, 0.463] | 1920 | 2384 |

## Usage

```python
import torch
from colpali_engine.models import ColQwen3, ColQwen3Processor

model_id = "Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0"
model = ColQwen3.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda:0",
    attn_implementation="sdpa",
).eval()
processor = ColQwen3Processor.from_pretrained(model_id)

# documents: list[PIL.Image] of page images; queries: list[str]
doc_batch = processor.process_images(documents).to(model.device)
qry_batch = processor.process_queries(queries).to(model.device)
with torch.no_grad():
    doc_emb = model(**doc_batch)
    qry_emb = model(**qry_batch)
scores = processor.score_multi_vector(qry_emb, doc_emb)   # (n_queries, n_docs)
```

Requirements: `colpali-engine>=0.3.16`, `transformers>=5.0`, `torch>=2.5`.

## Training

| | |
|---|---|
| Base | Qwen/Qwen3-VL-4B-Instruct (raw, no warm start) |
| Method | LoRA r=32, α=32, dropout 0.1 on language-model proj layers; `custom_text_proj` head fully trained |
| Data | [manu/colpali](https://huggingface.co/datasets/manu/colpali-queries) EN+FR, 108K query–page pairs, 2 mined hard negatives per query (K=2) |
| Loss | ColBERT pairwise negative CE (in-batch + explicit negatives) |
| Schedule | LR 5e-5, linear decay, warmup 10, effective batch 32, bf16, gradient checkpointing, `max_num_visual_tokens=768` (training) |
| Hardware | single A100 80GB |
| Selection | eval-gated checkpoint curve on the full benchmark: step 500 → 0.5441, **step 1000 → 0.5584 (peak, released)**, step 1500 → 0.5518 (overfit onset) |

### Measured negative results (transparency)

Each candidate improvement was evaluated on the full benchmark and dropped on evidence: more negatives (K=4: −0.016, worse on 8/8 subtasks), two-run weight averaging (−0.006, zero synergy across LoRA inits), train-matched visual-token cap at eval (−0.017; uncapped inference is better). Full validity report (causal control, leakage tripwires, pHash contamination scan, bootstrap CIs): [STAGE1_VALIDITY_REPORT.md](https://github.com/Verm1lion/ColTurk-VDR/blob/main/STAGE1_VALIDITY_REPORT.md).

## Evaluation protocol & reproduction

```bash
git clone https://github.com/Verm1lion/ColTurk-VDR
cd ColTurk-VDR
python scripts/eval/eval_colturk_checkpoint.py \
    --adapter Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0 \
    --bootstrap 1000 --output eval/results/repro.json
```

Environment pins and seeds: [REPRODUCIBILITY.md](https://github.com/Verm1lion/ColTurk-VDR/blob/main/REPRODUCIBILITY.md). Training data ↔ benchmark contamination was checked empirically (perceptual-hash scan over train images × V3 corpora: 0 exact duplicates, 0.025% at the document true-duplicate bar, visually inspected) — details in the validity report.

## Limitations

- Trained on 108K EN+FR pairs (single-GPU budget) — well below the multi-million-pair data scale of the top ViDoRe V3 entries; scores reflect that gap honestly.
- English and French document domains only in v1.0; Turkish document support is the next planned stage.
- Retrieval-only model: no reranking, no generation.

## Citation

```bibtex
@misc{karatay2026colturkvdr,
  author = {Karatay, Mert},
  title  = {ColTurk-VDR: A Late-Interaction Visual Document Retriever on Qwen3-VL-4B},
  year   = {2026},
  url    = {https://github.com/Verm1lion/ColTurk-VDR}
}
```
