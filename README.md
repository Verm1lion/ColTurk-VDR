# ColTurk-VDR

> **🚧 WIP — v0.1-pre · Day-1 baseline ready · 2026-05-18**
>
> ColBERT-style late-interaction **visual document retriever** for Turkish enterprise documents (e-invoices, KYC, legal, financial). Built on **Qwen3-VL-4B-Instruct** (Apache-2.0) via `colpali_engine.models.ColQwen3` (Qwen3-VL native, verified 2026-05-18).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Base Model: Apache-2.0](https://img.shields.io/badge/Base-Apache--2.0-blue.svg)](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
[![Status: Day-1 baseline ready](https://img.shields.io/badge/status-Day--1%20baseline%20ready-yellowgreen.svg)]()
[![Submission: MTEB v2](https://img.shields.io/badge/submission-MTEB%20v2%20canonical-blue.svg)](https://github.com/embeddings-benchmark/mteb)

---

## What is this?

**EN:** ColTurk-VDR brings ColBERT-style multi-vector late-interaction retrieval to **Turkish enterprise documents**. Trained on a single A100 80GB ($0 marginal cost), it targets the [ViDoRe V3 MTEB leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) and ships with **ViDoRe-TR**, the first public Turkish split for visual document retrieval. MIT-licensed code, Apache-2.0 base model, KVKK-compliant synthetic data pipeline included.

**TR:** ColTurk-VDR, Türkçe kurumsal belgeler (e-fatura, KYC, hukuki, finansal) için ColBERT-tarzı **çok-vektörlü geç-etkileşim retrieval** modeli. Tek A100 80GB üzerinde ($0 marjinal maliyet) eğitilir, **ViDoRe V3 MTEB leaderboard**'ı birincil hedef, **ViDoRe-TR** Türkçe community split yan ürün olarak release edilir.

## Why this matters

Türkiye'deki e-fatura zorunluluğu (GİB 589 Tebliği, 1 Ocak 2026), KYC pipeline'ları (Akbank Lab, Garanti BBVA, Yapı Kredi, iyzico, Papara) ve KOBİ ERP entegrasyonları (Logo, Mikro, KoçSistem) — hepsi Türkçe doküman üzerinde **sayfa düzeyinde retrieval** ihtiyacı. Klasik OCR → text retrieval pipeline'ı Türkçe karakter (ğ, ş, ı, ç, ö, ü), tablo yapısı ve layout bilgisini kaybediyor. ColTurk-VDR doğrudan görsel patch embedding'leri üzerinde late-interaction yapar — OCR'siz, layout-aware, çok-dilli.

## Hedef Leaderboard'lar

| Hedef | URL | Mevcut SOTA | Hedefimiz |
|---|---|---|---|
| **ViDoRe V3 MTEB** (primary) | [HF Space](https://huggingface.co/spaces/vidore/vidore-leaderboard) | NVIDIA Nemotron ColEmbed-8B-v2 NDCG@10 63.42 | Top-15 (NDCG@10 52-58) |
| **MTEB Multilingual Retrieval (Turkish)** | [MTEB](https://github.com/embeddings-benchmark/mteb) | Açık | #1 açık-ağırlık Türkçe |
| **ViDoRe-TR community split** | community PR | Henüz yok | Yaratan + maintainer |

## Quick start (planned, post-v0.1)

```bash
docker compose up
# → http://localhost:8000 (FastAPI)
# → http://localhost:8501 (Streamlit dashboard)
# → http://localhost:6333 (Qdrant)
```

## Architecture

```
Image (page) → Qwen3-VL-4B Vision → Patch tokens → 128-dim projection
                                                          ↓
Query (text) → Qwen3-VL-4B Text → Query tokens → 128-dim projection
                                                          ↓
                                MaxSim(q, d) = Σᵢ maxⱼ (qᵢ · dⱼ)
                                                          ↓
                                                NDCG@10, recall@k, mAP
```

## Status & Roadmap

- [x] Day-0 setup (2026-05-18) — repo + scaffolding
- [ ] Day 1: ColQwen2.5-3B-multilingual zero-shot baseline on ViDoRe V3
- [ ] Day 2-4: Turkish corpus build (mevzuat.gov.tr + KAP + YÖK + synthetic e-fatura)
- [ ] Day 5-7: Stage-1 training (English-heavy ColPali recipe)
- [ ] Day 8: First ViDoRe V3 baseline submission
- [ ] Day 9-10: Stage-2 fine-tune (Turkish-heavy 5:1)
- [ ] Day 11: Reranker + model merging
- [ ] Day 12: Productization (FastAPI + Qdrant + Streamlit + Docker Compose)
- [ ] Day 13-14: Demo video + LinkedIn launch + ViDoRe-TR community PR

Detailed plan: see `docs/`.

## Lisans / License

- **Code:** MIT — see [LICENSE](LICENSE)
- **Datasets & base models:** see [LICENSE-NOTICE.md](LICENSE-NOTICE.md) for the full matrix
- **KVKK:** No real PII used. All Turkish data is synthetic (Faker-tr with invalidated TCKN/IBAN) or public domain (mevzuat.gov.tr, KAP, YÖK). See `docs/KVKK-compliance.md` (post-v0.1).

## Author

[**Mert Karatay**](https://github.com/Verm1lion) — AI & Network Security Engineer, İstanbul. Background: ConvNeXt-Tiny + ArcFace + Triplet metric learning (TÜBİTAK 2209-A signature verification, CEDAR 0.9844 AUC). [HuggingFace: Verm1ion](https://huggingface.co/Verm1ion).

## Citation

```bibtex
@misc{karatay2026colturkvdr,
  author = {Karatay, Mert},
  title  = {ColTurk-VDR: A Late-Interaction Visual Document Retriever for Turkish Enterprise Documents},
  year   = {2026},
  url    = {https://github.com/Verm1lion/ColTurk-VDR}
}
```

## Acknowledgments

- [ColPali](https://github.com/illuin-tech/colpali) — late-interaction visual retrieval framework (Faysse et al., ICLR 2025)
- [NVIDIA Nemotron ColEmbed v2](https://arxiv.org/html/2602.03992v1) — model merging + hard negative mining recipe
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Apache-2.0 base vision-language model
- [Trendyol Tech](https://medium.com/trendyol-tech) — ConvNeXt+ArcFace ecosystem inspiration (1B image vector search blog)
