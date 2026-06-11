"""ColTurk-VDR MTEB v2 submission wrapper + ModelMeta (Phase E).

Wraps our v5-native ColQwen3 + LoRA adapter (Stage-1 v0.1 = ckpt-1000, NDCG@10 0.5584)
as a late-interaction (multi-vector, MaxSim) VISION retriever for `mteb`. All encoding
goes through the SHARED, S52-validated core in `src.inference.encode`, so the number a
submission reproduces == the number we report (eval_colturk_checkpoint.py).

CANONICAL submission path (S21): register this ModelMeta in `embeddings-benchmark/mteb`,
run `mteb.evaluate(model, mteb.get_benchmark("ViDoRe(v3)"))` -> results JSON ->
`embeddings-benchmark/results` PR. If `mteb.evaluate` cannot run on our stack (Day-1 F6/F7
risk), fall back to `scripts/eval/build_mteb_results.py` (self-reported results from our
own harness — MTEB accepts these with published eval code).

⚠️ VERIFY-IN-COLAB (mteb not installed locally — 2026-06-07):
  1) The EXACT mteb v2 encoder protocol for ViDoRe/vision retrieval. mteb late-interaction
     vision models implement multi-vector encode + a max_sim similarity. Confirm the required
     method names against the installed package, e.g.:
        import mteb, inspect
        print(mteb.__version__)
        # an existing ViDoRe/ColPali model in mteb (e.g. ColQwen2 / ColPali) is the template:
        print(inspect.getsource(<that model's wrapper>))
     and align encode_queries/encode_corpus (names below) + similarity to it.
  2) ModelMeta required/renamed fields for this mteb version (some are version-specific).
  3) Then: `m = MODEL_META.load_model(); mteb.evaluate(m, mteb.get_benchmark("ViDoRe(v3)"))`.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BASE = "Qwen/Qwen3-VL-4B-Instruct"
ADAPTER = "Verm1ion/ColTurk-VDR-Stage1"     # ckpt-1000 = Stage-1 v0.1 (make this HF repo PUBLIC before submit)
REVISION = "main"                            # pin to the ckpt-1000 commit before submitting


class ColTurkVDRModel:
    """Late-interaction (multi-vector / MaxSim) vision retriever wrapper for mteb.

    Encoding delegates to src.inference.encode (the exact eval-harness path). Returns
    a LIST of variable-length (T_i, D) tensors per item — NOT a padded array — which is
    the multi-vector contract late-interaction tasks expect.
    """

    def __init__(self, base: str = BASE, adapter: str | None = ADAPTER,
                 device: str = "cuda:0", batch_size: int = 4, max_visual_tokens: int | None = None):
        from src.inference.encode import load_colturk

        self.model, self.processor = load_colturk(
            base=base, adapter=adapter, device=device, max_visual_tokens=max_visual_tokens
        )
        self.batch_size = batch_size

    # --- multi-vector encode (method NAMES per mteb v2 ViDoRe protocol — VERIFY in Colab) ---
    def encode_queries(self, queries, **kwargs):
        from src.inference.encode import encode_queries
        return encode_queries(self.model, self.processor, list(queries), batch_size=max(1, self.batch_size * 2))

    def encode_corpus(self, corpus, **kwargs):
        """corpus = list of images (PIL) OR dicts carrying an image field. mteb's ViDoRe
        corpus rows expose the page image; extract it, then encode."""
        from src.inference.encode import encode_images
        images = [self._as_image(c) for c in corpus]
        return encode_images(self.model, self.processor, images, batch_size=self.batch_size)

    # mteb may dispatch a single `encode(..., prompt_type=...)` instead of the two above —
    # provide it too so whichever the installed version calls, it routes correctly.
    def encode(self, inputs, prompt_type=None, **kwargs):
        is_query = (str(prompt_type).lower().endswith("query")) if prompt_type is not None else False
        return self.encode_queries(inputs, **kwargs) if is_query else self.encode_corpus(inputs, **kwargs)

    def similarity(self, query_embeddings, corpus_embeddings):
        """MaxSim (Nq, Nd) — late-interaction scoring."""
        from src.inference.encode import maxsim_scores
        return maxsim_scores(self.processor, list(query_embeddings), list(corpus_embeddings))

    @staticmethod
    def _as_image(item):
        if hasattr(item, "convert") or hasattr(item, "size"):     # already a PIL image
            return item
        if isinstance(item, dict):
            for k in ("image", "page_image", "img"):
                if item.get(k) is not None:
                    return item[k]
        return item


def _loader(**kwargs):
    return ColTurkVDRModel(**kwargs)


def build_model_meta():
    """Return an mteb.ModelMeta for ColTurk-VDR. Import is lazy so this module loads even
    where mteb isn't installed (e.g. local dev). Call in Colab to register/run."""
    from mteb import ModelMeta  # VERIFY: import path/fields may differ by mteb version

    return ModelMeta(
        name="Verm1ion/ColTurk-VDR-Stage1",
        revision=REVISION,
        release_date="2026-06-07",
        languages=["eng-Latn", "fra-Latn"],          # ViDoRe V3 docs EN+FR
        n_parameters=4_000_000_000,                    # Qwen3-VL-4B base (+ LoRA r=32 + custom_text_proj)
        max_tokens=None,
        embed_dim=128,                                 # late-interaction per-token dim (custom_text_proj)
        license="apache-2.0",                          # base Apache-2.0 + LoRA adapter
        open_weights=True,
        public_training_code="https://github.com/Verm1lion/ColTurk-VDR",
        public_training_data="manu/colpali (EN+FR)",
        framework=["colpali-engine", "PyTorch"],
        reference="https://huggingface.co/Verm1ion/ColTurk-VDR-Stage1",
        similarity_fn_name="max_sim",                  # late-interaction MaxSim — VERIFY enum value
        use_instructions=False,
        adapted_from="Qwen/Qwen3-VL-4B-Instruct",
        loader=_loader,                                # mteb calls this to instantiate the model
        modalities=["image", "text"],                  # VERIFY field name for vision in this version
    )


# Convenience for Colab: `from src.models.colturk_mteb import MODEL_META` then mteb.evaluate(...)
# (kept lazy — only build when mteb is importable)
MODEL_META = None
try:  # noqa: SIM105
    MODEL_META = build_model_meta()
except Exception as _exc:  # mteb not installed (local) — fine; build it in Colab
    logger.info("ModelMeta not built (mteb unavailable here): %s", _exc)
