# Licensing Notice — Datasets, Models, and Components

> **Owned code:** MIT (see `LICENSE`).
>
> **Aggregate notice:** ColTurk-VDR depends on and/or is evaluated against multiple third-party assets. This file enumerates each, the license, and how it is used. **Anyone using this repo for commercial purposes MUST independently verify each upstream license.**

**Last updated:** 2026-05-18 (Day-0, pre-execution; final licenses verified before v1.0).

---

## 1. Base Models

| Model | HF / Source | License | Use in this repo | Verification status |
|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Apache-2.0 (verify before v1.0) | Primary base model — LoRA fine-tune target | ⏳ Pending (OPEN_ITEMS A2) |
| ColQwen2.5-3B-multilingual-v1.0 | [Metric-AI/ColQwen2.5-3b-multilingual-v1.0](https://huggingface.co/Metric-AI/ColQwen2.5-3b-multilingual-v1.0) | Apache-2.0 (verify) | Baseline reference for zero-shot eval | ⏳ Pending |
| ColPali v1.3 | [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) | MIT | Architecture reference | ✅ MIT |
| Qwen3-VL-8B-Instruct (reader, optional) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Apache-2.0 (verify) | Optional reranker / RAG reader | ⏳ Pending |
| DINOv2 ViT-S/14 (optional ablation) | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | Apache-2.0 | Optional vision encoder ablation (v1.1) | ✅ Apache-2.0 |

**Note on Qwen / Tongyi licensing:** The `Qwen3-VL` family is announced under Apache-2.0 but Tongyi Qianwen models have, in the past, used custom licenses for the 30B-A3B MoE variants. Confirm exact license text in the HF repo before redistribution.

---

## 2. Training & Evaluation Datasets

### 2.1 Primary (training)

| Dataset | Source | License | Use | Verification |
|---|---|---|---|---|
| DocVQA train | [rrc.cvc.uab.es](https://rrc.cvc.uab.es/?ch=17) | Research-only (CC-BY-NC variant suspected) | Fine-tune (ColPali recipe) | ⏳ Verify before commercial export |
| InfographicsVQA train | rrc.cvc.uab.es | Research-only | Fine-tune | ⏳ |
| ChartQA train | [github.com/vis-nlp/ChartQA](https://github.com/vis-nlp/ChartQA) | GPL-3.0 (verify) | Fine-tune | ⏳ |
| Vidore SynthDocBench family | HF `vidore/syntheticDocQA-*` | MIT (verify) | Fine-tune | ⏳ |
| WIT (Wikipedia-derived) | [github.com/google-research-datasets/wit](https://github.com/google-research-datasets/wit) | CC-BY-SA 3.0 | Multilingual aug | ✅ CC-BY-SA |

### 2.2 ViDoRe evaluation (no commercial port; portfolio + paper OK)

| Dataset | License | Use |
|---|---|---|
| ViDoRe V3 (8 public subtasks) | MIT (per ViDoRe team) | Primary leaderboard eval |
| ViDoRe V2 (multilingual) | MIT | Secondary multilingual eval |
| MTEB tasks (BEIR-derived) | Mixed (per task; mostly CC-BY 4.0) | MTEB Multilingual Retrieval (Turkish) eval |

### 2.3 Turkish slice (self-generated + public domain)

| Dataset | Source | License | Use | KVKK |
|---|---|---|---|---|
| **Verm1ion/turkish-doc-synth-Nk** (this repo) | Self-generated via Faker-tr + reportlab/WeasyPrint | MIT (intended) | Fine-tune + ViDoRe-TR eval | ✅ No real PII (Faker-tr invalidated TCKN/IBAN) |
| Resmi Gazete (T.C.) | [mevzuat.gov.tr](https://www.mevzuat.gov.tr/) | Public domain (kamu malı) | Fine-tune + eval slice | ✅ Government public |
| KAP faaliyet raporları | [kap.org.tr](https://www.kap.org.tr/) | Public (corporate disclosure) | Fine-tune + eval slice | ✅ Corporate disclosure (no individual PII) |
| YÖK Ulusal Tez Merkezi (open-access only) | [tez.yok.gov.tr](https://tez.yok.gov.tr/) | CC-BY / author-permitted (per thesis) | Fine-tune | ⚠️ Use only CC-BY/CC-BY-SA; skip closed-access |

### 2.4 EXPLICITLY EXCLUDED (not used in this repo)

| Dataset | License | Reason |
|---|---|---|
| MVTec AD 2 / LOCO | CC-BY-NC-SA 4.0 | Non-commercial; not relevant here but flagged for clarity |
| GenImage | CC-BY-NC-SA 4.0 | Non-commercial; not used |
| DF40 | Research-only | Not used |
| LongVU SFT subset | CC-BY-NC | Not used |
| Any real customer / KYC / banking document | — | KVKK / PII protection |

---

## 3. Code Dependencies (key libraries)

| Library | License | Pinned version target |
|---|---|---|
| PyTorch | BSD-3 | 2.5+ |
| transformers | Apache-2.0 | 4.50+ |
| colpali-engine | MIT | Latest stable (verify Qwen3-VL support) |
| vidore-benchmark | MIT | Latest |
| mteb | Apache-2.0 | Latest |
| qdrant-client | Apache-2.0 | 1.11+ |
| FastAPI | MIT | 0.115+ |
| vLLM | Apache-2.0 | Latest |

---

## 4. Commercial Use Statement

This repository's code is MIT-licensed and may be used commercially. However, **the trained model weights and evaluation results depend on third-party datasets and base models**, several of which have non-commercial or research-only restrictions (notably DocVQA-family and Tongyi license variations). Anyone wishing to deploy ColTurk-VDR commercially MUST:

1. Re-verify each upstream dataset/model license at the time of deployment.
2. Substitute non-commercial datasets with permissively-licensed alternatives (e.g., regenerate the Turkish slice from `Verm1ion/turkish-doc-synth-Nk` synthetic + public-domain sources only).
3. Train and host their own weights under a license consistent with their use.

---

## 5. KVKK / GDPR Compliance

- **No real personal data** is used or distributed via this repo. All Turkish content is either:
  - Synthetic (Faker-tr with invalidated TCKN/IBAN — checksum-failing identifiers), or
  - Public domain government records (mevzuat.gov.tr, KAP corporate disclosures, YÖK open-access theses), or
  - Public domain academic content (university theses with open-access licenses)
- See `docs/KVKK-compliance.md` (post-Day-0) for the full audit pipeline.

---

## 6. Reporting Issues

If you believe any asset is mis-licensed or there is a KVKK/GDPR concern, please open a GitHub issue with the `license` label.
