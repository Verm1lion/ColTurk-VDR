"""Bidirectional attention patch for ColQwen3 / Qwen3-VL — CORRECTED 2026-06-07.

⚠️ THE OLD PATCH WAS A LIKELY SILENT NO-OP (research wqp47b3wl, web-verified):
  Qwen3VLTextModel.forward calls module-level `create_causal_mask()` and passes the
  resulting EXPLICIT ADDITIVE causal mask to every attention layer. Flipping only
  `module.is_causal=False` does NOT remove that explicit mask → under `sdpa` (our
  default) attention STAYS causal. (Under FA2 the runtime `is_causal` IS read, so the
  flip *does* take effect there — i.e. the old patch's behavior was attn-impl-dependent.)
  A full retrain on a no-op patch would waste ~50 GPU-h, so this version:
    1) ALSO overrides `create_causal_mask` -> bidirectional (the real fix), AND
    2) replaces the flag-only `assert_bidirectional` with an OUTPUT-LEVEL PROBE that
       perturbs a LATER token and checks an EARLIER token's embedding changes
       (causal: invariant -> probe FAILS; bidi: changes -> probe PASSES). The old
       assert would PASS on a no-op; this one cannot.

Evidence on the GAIN (adversarially discounted): NV-Embed 2405.17428 reports bidi vs
causal +1.03 retrieval-MTEB (text, full-FT). But Hydra (2603.28554, OUR exact stack)
shows NO clean bidi-vs-causal isolation, and LLM2Vec (2404.05961) shows cold bidi
WITHOUT an MNTP warmup can HURT decoder LLMs. So: treat bidi as a GATED swing lever,
budget <= +0.1 NDCG, and SHIP ONLY IF a short A/B beats the causal baseline. Whether
Qwen3-VL needs an MNTP warmup is unknown -> only add it if the cold-bidi pilot regresses.

Usage (Colab, BEFORE any full bidi run — de-risk first):
    from scripts.training.patch_bidirectional import patch_to_bidirectional, probe_bidirectional
    n, mask_ok = patch_to_bidirectional(model)
    bidi, delta = probe_bidirectional(model, processor)   # MUST be True; delta >> 1e-4
Run the smoke under BOTH attn_implementation="sdpa" AND "flash_attention_2".
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

LOG = logging.getLogger("patch_bidirectional")

# stash originals so the monkeypatch is idempotent / reversible
_ORIG_MASK_FNS: dict[str, Any] = {}


def _override_causal_mask() -> bool:
    """Override module-level `create_causal_mask` in Qwen3-VL modeling -> bidirectional.

    Returns it to a padding-aware NON-causal additive mask (real tokens attend to all
    real tokens, never to padding; no causal triangle). Returns None when no padding
    info is available (= full bidirectional attention; padding outputs are dropped
    downstream by the encoder's attention-mask gather, so this is safe).

    Returns True if at least one modeling module's `create_causal_mask` was patched.
    """
    import sys

    def _bidi_mask(*args, **kwargs):
        # transformers v5 create_causal_mask(config, input_embeds, attention_mask,
        # cache_position, past_key_values, position_ids=..., or_mask_function=...).
        am = kwargs.get("attention_mask")
        ie = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
        if am is None and len(args) >= 3:
            am = args[2]
        if ie is None and len(args) >= 2:
            ie = args[1]
        if am is None or ie is None or not torch.is_tensor(am) or am.dim() != 2:
            return None  # full bidirectional; padding dropped downstream
        dtype = ie.dtype if torch.is_tensor(ie) else torch.float32
        min_val = torch.finfo(dtype).min
        bsz, seq = am.shape
        key_pad = am == 0                                   # (B, S) True where padding
        add = torch.zeros((bsz, 1, seq, seq), dtype=dtype, device=am.device)
        add = add.masked_fill(key_pad[:, None, None, :], min_val)   # mask PAD keys only, no causal triangle
        return add

    patched = False
    for mod_name, module in list(sys.modules.items()):
        if module is None or "qwen3_vl" not in mod_name or "modeling" not in mod_name:
            continue
        if hasattr(module, "create_causal_mask"):
            if mod_name not in _ORIG_MASK_FNS:
                _ORIG_MASK_FNS[mod_name] = getattr(module, "create_causal_mask")
            setattr(module, "create_causal_mask", _bidi_mask)
            LOG.info("Overrode create_causal_mask in %s -> bidirectional", mod_name)
            patched = True
    if not patched:
        LOG.warning(
            "create_causal_mask NOT found in any loaded qwen3_vl modeling module — "
            "the mask is NOT overridden. Bidi will be a NO-OP under sdpa. Import the "
            "model FIRST, then patch; or verify the transformers Qwen3-VL module name."
        )
    return patched


def patch_to_bidirectional(model: nn.Module) -> tuple[int, bool]:
    """Switch the decoder to bidirectional (in-place). Returns (n_modules, mask_overridden).

    Does BOTH things required for a real effect:
      - flips `is_causal=False` on every language-path attention module (FA2 path), and
      - overrides `create_causal_mask` -> bidirectional (sdpa path; the old bug).
    The visual tower (SigLIP) is untouched. ALWAYS confirm with `probe_bidirectional`.
    """
    patched = 0
    target_root = _resolve_language_model_root(model)
    for module in target_root.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
            patched += 1

    mask_ok = _override_causal_mask()

    if patched == 0:
        LOG.warning("No `is_causal` attributes found — different attribute name or already non-causal")
    LOG.info("Flipped %d attention modules; create_causal_mask overridden=%s", patched, mask_ok)
    if not mask_ok:
        LOG.error("MASK NOT OVERRIDDEN -> bidi is a NO-OP under sdpa. Run probe_bidirectional to confirm before training!")
    return patched, mask_ok


def _resolve_language_model_root(model: nn.Module) -> nn.Module:
    """Return the decoder root (handles ColQwen3, Qwen3VLModel, plain Qwen3Model)."""
    for attr in ("language_model", "model", "transformer"):
        if hasattr(model, attr):
            root = getattr(model, attr)
            if hasattr(root, "layers") or hasattr(root, "decoder"):
                return root
    return model


@torch.no_grad()
def probe_bidirectional(model: nn.Module, processor: Any, atol: float = 1e-4) -> tuple[bool, float]:
    """OUTPUT-LEVEL bidirectionality probe (catches the silent no-op the flag-check misses).

    Perturb the LAST real token's id and check whether an EARLY token's output embedding
    changes. Causal attention => early token cannot see the later one => output invariant
    (delta ~ 0 => probe FAILS, still causal/no-op). Bidirectional => early token attends to
    the later one => output changes (delta >> atol => probe PASSES).

    Returns (is_bidirectional, max_abs_delta_at_token0).
    """
    model.eval()
    dev = next(model.parameters()).device
    batch = processor.process_queries(["alpha beta gamma delta epsilon zeta"]).to(dev)
    out1 = model(**batch)
    emb1 = out1[0] if torch.is_tensor(out1) else out1.last_hidden_state[0]   # (S, D)

    batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    ids = batch2["input_ids"]
    am = batch2.get("attention_mask")
    last = (int(am[0].sum().item()) - 1) if (am is not None and torch.is_tensor(am)) else ids.shape[1] - 1
    last = max(1, last)  # never the very first token
    try:
        vocab = model.config.get_text_config().vocab_size
    except Exception:
        vocab = int(getattr(getattr(model, "config", None), "vocab_size", 100000) or 100000)
    orig = int(ids[0, last].item())
    new_id = vocab // 2 if orig != vocab // 2 else vocab // 3   # clearly-different, in-range token
    batch2["input_ids"][0, last] = new_id
    out2 = model(**batch2)
    emb2 = out2[0] if torch.is_tensor(out2) else out2.last_hidden_state[0]

    delta = (emb1[0].float() - emb2[0].float()).abs().max().item()   # token-0 sensitivity to token `last`
    is_bidi = delta > atol
    LOG.info("Bidi probe: token-0 max|Δ| when token-%d changed = %.3e -> %s",
             last, delta, "BIDIRECTIONAL ✓" if is_bidi else "STILL CAUSAL ✗ (no-op!)")
    return is_bidi, delta


def assert_bidirectional(model: nn.Module, processor: Any | None = None) -> None:
    """Hard gate. Prefer the OUTPUT probe (needs processor); fall back to the flag check
    only if no processor is given (and warn that the flag check cannot catch a no-op)."""
    if processor is not None:
        is_bidi, delta = probe_bidirectional(model, processor)
        if not is_bidi:
            raise RuntimeError(
                f"Bidi probe FAILED (token-0 Δ={delta:.3e} <= atol): attention is still CAUSAL "
                f"(silent no-op). The create_causal_mask override did not take effect under this "
                f"attn_implementation. Do NOT train — fix the mask override first."
            )
        return
    LOG.warning("assert_bidirectional WITHOUT processor only checks the is_causal FLAG — "
                "it CANNOT detect the sdpa silent no-op. Pass a processor for the real probe.")
    target_root = _resolve_language_model_root(model)
    offenders = [n for n, m in target_root.named_modules() if getattr(m, "is_causal", False) is True]
    if offenders:
        raise RuntimeError(f"Causal flag still set in {len(offenders)} modules: {offenders[:5]}")


def main() -> None:
    """Smoke test: load ColQwen3, patch, OUTPUT-PROBE, under the given attn impl."""
    import argparse

    parser = argparse.ArgumentParser(description="Bidirectional attention patch smoke test (+ output probe)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                        help="VERIFY under BOTH sdpa and flash_attention_2 — the no-op risk differs")
    args = parser.parse_args()

    from colpali_engine.models import ColQwen3, ColQwen3Processor  # type: ignore[import-not-found]

    model = ColQwen3.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation=args.attn,
    ).eval()
    processor = ColQwen3Processor.from_pretrained(args.model_id)

    n, mask_ok = patch_to_bidirectional(model)
    is_bidi, delta = probe_bidirectional(model, processor)
    LOG.info("attn=%s | flipped=%d | mask_overridden=%s | PROBE bidi=%s (Δ=%.3e)",
             args.attn, n, mask_ok, is_bidi, delta)
    if not is_bidi:
        raise SystemExit("PROBE FAILED — bidi is a NO-OP under this attn impl. Fix before training.")
    LOG.info("Bidirectional patch VERIFIED at the output level — safe to proceed to the gated A/B.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
