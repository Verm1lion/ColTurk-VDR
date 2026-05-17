# `src/` — ColTurk-VDR source tree

| Package | Purpose | Day |
|---|---|---|
| `models/` | Model wrappers (Qwen3-VL backbone, ColBERT projection head, late-interaction MaxSim) | 5-6 |
| `data/` | Dataset loaders, augmentation, KVKK PII utils, BEIR format converters | 2-4 |
| `training/` | Stage 1/2 training loops, hard negative mining, LoRA setup, model merging | 5-11 |
| `inference/` | Retrieval pipeline, query encoding, MaxSim scoring, top-k batching | 7-12 |
| `rag/` | Reader integration (Qwen3-VL-8B), prompt templates, response synthesis | 12 |

See `docs/architecture.md` (post-Day-0) for the detailed module diagram.
