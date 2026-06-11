# ViDoRe V3 / MTEB Submission Guide — ColTurk-VDR v1.0

Step-by-step submission of `Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0` (NDCG@10 0.5584, 8 public subtasks) to the MTEB ViDoRe V3 retriever leaderboard. Written so maintainers hit zero friction.

## 0. Prerequisites (done)
- [x] GitHub repo public + clean history: https://github.com/Verm1lion/ColTurk-VDR
- [x] Official eval JSONs committed under `eval/results/`
- [x] Model card: `docs/MODEL_CARD_v1.md`
- [ ] Merged model published → run notebook **Cell 14 (PUBLISH)** → `https://huggingface.co/Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0` (public)

## 1. ModelMeta PR → `embeddings-benchmark/mteb`
Fork: `Verm1lion/mteb` (exists). Edit **`mteb/models/model_implementations/colqwen_models.py`** — append (mirrors the existing ColQwen3-family entries in that file; `ColQwen3Wrapper` is already defined there):

```python
colturk_vdr_4b = ModelMeta(
    loader=ColQwen3Wrapper,
    name="Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0",
    model_type=["late-interaction"],
    languages=["eng-Latn", "fra-Latn"],
    revision="<HF MODEL COMMIT SHA — Cell 14 prints it>",
    release_date="2026-06-11",
    modalities=["image", "text"],
    n_parameters=4_505_515_136,
    n_embedding_parameters=None,
    memory_usage_mb=8593,            # bf16 full model
    max_tokens=262144,               # Qwen3-VL context (same as the other Qwen3-VL entries in this file)
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/Verm1lion/ColTurk-VDR",
    public_training_data="https://huggingface.co/datasets/manu/colpali-queries",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,          # match the plain colqwen3 entries (not the instruction-tuned ones)
    training_datasets=COLPALI_TRAINING_DATA,   # reuse the constant if defined in this file/colpali_models.py; else inline {"manu/colpali-queries": ["train"]}
    extra_requirements_groups=["colqwen3"],
)
```

Sanity check before the PR (Colab/local with mteb installed):
```python
import mteb; m = mteb.get_model_meta("Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0"); print(m.name, m.revision)
```

**PR text (paste):**
> Add ColTurk-VDR-Qwen3VL-4B-v1.0 — a ColBERT-style late-interaction visual document retriever (Qwen3-VL-4B-Instruct + LoRA, merged weights, colpali-engine ColQwen3 architecture). Apache-2.0, open weights, public training code and data. Evaluated on the 8 public ViDoRe V3 retrieval tasks (full corpus, all queries): mean NDCG@10 = 0.5584. Results PR: <link after step 2>. Eval code & raw JSONs: https://github.com/Verm1lion/ColTurk-VDR

## 2. Results PR → `embeddings-benchmark/results`
Fork: `Verm1lion/results` (exists). Notebook **Cell 14** generates the per-task files under
`eval/mteb_results/Verm1ion__ColTurk-VDR-Qwen3VL-4B-v1.0/<model-sha>/Vidore3*Retrieval.json`
with live `mteb_version` + per-task `dataset_revision` (taken from `mteb.get_benchmark`). Copy that folder into the results repo's layout (`results/Verm1ion__ColTurk-VDR-Qwen3VL-4B-v1.0/<model-sha>/`), follow the repo README if a registry file also needs the model added, open the PR.

**PR text (paste):**
> Self-reported ViDoRe V3 results for Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0 (ModelMeta PR: <link>). 8 public retrieval tasks, full corpus, all queries, MaxSim; mean NDCG@10 = 0.5584. Produced with the published eval harness (https://github.com/Verm1lion/ColTurk-VDR/blob/main/scripts/eval/eval_colturk_checkpoint.py); raw JSONs with bootstrap 95% CIs in the repo. Happy to re-run anything on request.

## 3. Private-split issue → `embeddings-benchmark/mteb`
> **Title:** ViDoRe V3 private-split evaluation request: ColTurk-VDR-Qwen3VL-4B-v1.0
> **Body:** Could the team queue `Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0` for the two held-out ViDoRe V3 tasks (Vidore3TelecomRetrieval, Vidore3NuclearRetrieval)? Public weights (Apache-2.0), loads with the existing `ColQwen3Wrapper`; ModelMeta PR #<id>, results PR #<id>. Thanks!

## 4. Post-merge checks
- Leaderboard row renders with the score (known matching bug: mteb issue **#3918** — if the row is missing/blank after merge, comment there with the model name).
- `mteb.get_model_meta(...)` resolves on main.

## 5. Closing
- **Revoke the HF + W&B + GitHub tokens** used during development (they appeared in working notes).
- Tag a GitHub release `v1.0.0`; announcement wording: “submitted + under review — public 8-subtask NDCG@10 0.5584” (no rank promises; private-split queue is maintainer-paced).
