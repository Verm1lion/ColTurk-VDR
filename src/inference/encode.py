"""Shared inference core for ColTurk-VDR — load + multi-vector encode + MaxSim.

Extracted (2026-06-07, Phase E) from the S52-validated eval harness
`scripts/eval/eval_colturk_checkpoint.py` so BOTH the eval harness and the MTEB
submission wrapper (`src/models/colturk_mteb.py`) call the exact same, proven code
path — no drift between the number we report and the number a submission reproduces.

Why colpali-native (not mteb's own loader): our adapter is trained on
transformers 5.9 + peft 0.19 + colpali 0.3.16, so its keys match the v5 ColQwen3
arch (F6 v4 key-prefix mismatch does NOT recur) and ColQwen3Processor yields
mm_token_type_ids natively (F7 does NOT recur). KARARLAR S17/S45.

Multi-vector contract (ColBERT/late-interaction):
  encode_* -> list of per-item (T_i, D) float32 CPU tensors (no padding, variable T_i).
  maxsim_scores -> (Nq, Nd) score matrix, MaxSim(q,d)=Σ_i max_j (q_i·d_j).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_colturk(
    base: str = "Qwen/Qwen3-VL-4B-Instruct",
    adapter: str | None = None,
    max_visual_tokens: int | None = None,
    device: str = "cuda:0",
    attn_implementation: str = "sdpa",
):
    """ColQwen3 base (+ optional LoRA adapter) + ColQwen3Processor -> (model, processor).

    adapter:
      - HF repo id or local LoRA dir (has adapter_config.json) -> base + PeftModel.
      - local FULL merged dir (Phase-D soup: config.json, NO adapter_config.json) -> direct.
      - None -> raw base, custom_text_proj RANDOM-init -> ~floor (S52-D10 causal control).
    max_visual_tokens: cap doc visual tokens (768 = train-match; None = processor default).
    """
    import os

    import torch
    from colpali_engine.models import ColQwen3, ColQwen3Processor

    is_merged_full = bool(
        adapter
        and os.path.isdir(adapter)
        and not os.path.exists(os.path.join(adapter, "adapter_config.json"))
        and os.path.exists(os.path.join(adapter, "config.json"))
    )

    if is_merged_full:
        logger.info("Loading MERGED FULL model (soup, no adapter_config.json): %s", adapter)
        model = ColQwen3.from_pretrained(
            adapter, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation=attn_implementation,
        ).eval()
    else:
        logger.info("Loading base ColQwen3: %s", base)
        model = ColQwen3.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation=attn_implementation,
        ).eval()
        if adapter:
            logger.info("Attaching LoRA adapter: %s", adapter)
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).eval()
        else:
            logger.warning(
                "NO-ADAPTER base control: custom_text_proj RANDOM-init -> floor NDCG expected."
            )

    proc_kwargs: dict[str, Any] = {}
    if max_visual_tokens:
        proc_kwargs["max_num_visual_tokens"] = max_visual_tokens
        logger.info("Processor max_num_visual_tokens=%d (train-match)", max_visual_tokens)
    proc_src = adapter if adapter else base
    try:
        processor = ColQwen3Processor.from_pretrained(proc_src, **proc_kwargs)
    except Exception:
        processor = ColQwen3Processor.from_pretrained(base, **proc_kwargs)
    return model, processor


def encode_images(model, processor, images: list, batch_size: int = 4) -> list:
    """list of PIL images -> list of per-doc (T_i, D) float32 CPU tensors (padding dropped)."""
    import torch

    embs: list = []
    for i in range(0, len(images), batch_size):
        batch = processor.process_images(images[i : i + batch_size]).to(model.device)
        with torch.no_grad():
            out = model(**batch)  # (B, seq, dim)
        mask = batch.get("attention_mask")
        for b in range(out.size(0)):
            if mask is not None:
                embs.append(out[b][mask[b].bool()].to(torch.float32).cpu())
            else:
                embs.append(out[b].to(torch.float32).cpu())
        logger.info("  encoded images %d/%d", min(i + batch_size, len(images)), len(images))
    return embs


def encode_queries(
    model, processor, queries: list[str], batch_size: int = 8, max_length: int = 128
) -> list:
    """list of query strings -> list of per-query (T_i, D) float32 CPU tensors.

    max_length=128 (not the ColQwen3Processor default 50, which silently truncates
    verbose ViDoRe V3 queries and degrades NDCG).
    """
    import torch

    embs: list = []
    for i in range(0, len(queries), batch_size):
        batch = processor.process_queries(
            queries[i : i + batch_size], max_length=max_length
        ).to(model.device)
        with torch.no_grad():
            out = model(**batch)
        mask = batch.get("attention_mask")
        for b in range(out.size(0)):
            if mask is not None:
                embs.append(out[b][mask[b].bool()].to(torch.float32).cpu())
            else:
                embs.append(out[b].to(torch.float32).cpu())
    return embs


def maxsim_scores(processor, query_embs: list, doc_embs: list):
    """[Nq, Nd] MaxSim. Prefer processor.score_multi_vector (optimized); manual fallback
    so an API-name change cannot break scoring."""
    import torch

    score_fn = getattr(processor, "score_multi_vector", None)
    if callable(score_fn):
        try:
            return score_fn(query_embs, doc_embs)  # (Nq, Nd)
        except Exception as exc:  # noqa: BLE001
            logger.warning("score_multi_vector failed (%s) — manual MaxSim fallback", exc)

    nq, nd = len(query_embs), len(doc_embs)
    scores = torch.zeros(nq, nd, dtype=torch.float32)
    for qi, q in enumerate(query_embs):        # q: (Tq, D)
        for di, d in enumerate(doc_embs):      # d: (Td, D)
            scores[qi, di] = (q @ d.T).max(dim=1).values.sum().item()
    return scores
