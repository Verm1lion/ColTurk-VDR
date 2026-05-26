"""Bidirectional attention patch (NV-Embed style) for ColQwen3 / Qwen3-VL.

NV-Embed paper (arXiv:2405.17428) Table 3/4 verified:
    Retrieval-only MTEB avg: bidirectional 59.36 vs causal 58.33 = +1.03 NDCG@10.
    Full MTEB 56-task: bidirectional 69.32 vs causal 68.47 = +0.85.

Nemotron ColEmbed V2 §3.3.1 verbatim:
    "we observed substantial performance improvement"
    Nemotron uses bidirectional attention; ColPali/ColQwen3 default is causal.

Implementation: ~10 LOC forward-hook style — set `is_causal = False` on every
LLM decoder layer's self-attention module, and ensure causal mask is not
applied during retrieval forward pass.

KARARLAR S29: bidirectional adapter P0 — beklenen +0.8 ila +1.5 NDCG@10 lift
on ColTurk-VDR Stage-2.

Compatibility note: Qwen3-VL was pre-trained with causal attention. Flipping
to bidirectional at fine-tune time alters the inductive bias; this is exactly
what NV-Embed and Nemotron paper report works for embedding tasks (not
generation). A/B ablation post-Stage-2 is mandatory before Stage-3 merge.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

LOG = logging.getLogger("patch_bidirectional")


def patch_to_bidirectional(model: nn.Module) -> int:
    """Switch every decoder layer's self-attention to bidirectional (in-place).

    Walks the model tree, finds attention modules with `is_causal` attribute,
    and flips them. Returns count of patched modules for verification.

    Compatible with Qwen3-VL family (Qwen3VLModel, Qwen3_5Model) and Llama-style
    decoder stacks. Visual tower attention (SigLIP) is untouched — only
    `language_model` / decoder layers are affected.
    """
    patched = 0
    target_root = _resolve_language_model_root(model)

    for module in target_root.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
            patched += 1

    if patched == 0:
        LOG.warning("No `is_causal` attributes found — model may be already non-causal or use different attribute name")
    else:
        LOG.info("Patched %d attention modules to bidirectional", patched)
    return patched


def _resolve_language_model_root(model: nn.Module) -> nn.Module:
    """Return the decoder root (handles ColQwen3, Qwen3VLModel, plain Qwen3Model)."""
    for attr in ("language_model", "model", "transformer"):
        if hasattr(model, attr):
            root = getattr(model, attr)
            if hasattr(root, "layers") or hasattr(root, "decoder"):
                return root
    return model


def assert_bidirectional(model: nn.Module) -> None:
    """Verify no causal attention remains in the language path."""
    target_root = _resolve_language_model_root(model)
    offenders = [
        f"{name} (is_causal={getattr(mod, 'is_causal')})"
        for name, mod in target_root.named_modules()
        if getattr(mod, "is_causal", False) is True
    ]
    if offenders:
        raise RuntimeError(f"Causal attention still present in {len(offenders)} modules: {offenders[:5]}")


def main() -> None:
    """Smoke test: load ColQwen3, patch, verify, run dummy forward."""
    import argparse

    parser = argparse.ArgumentParser(description="Bidirectional attention patch smoke test")
    parser.add_argument("--model-id", default="OpenSearch-AI/Ops-Colqwen3-4B")
    args = parser.parse_args()

    from colpali_engine.models import ColQwen3  # type: ignore[import-not-found]

    model = ColQwen3.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()

    n = patch_to_bidirectional(model)
    assert_bidirectional(model)
    LOG.info("Bidirectional patch succeeded — %d modules flipped, no causal attention remains", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
