#!/usr/bin/env python
"""Model-soup for ColTurk-VDR LoRA checkpoints (Phase D) — FULL-MODEL averaging.

Why full-model averaging (not adapter-level): our checkpoints are PEFT LoRA adapters with
a ``modules_to_save=["custom_text_proj"]`` head (F28). Two pitfalls of adapter-level merge:
  1. peft's ``add_weighted_adapter`` IGNORES modules_to_save -> the custom_text_proj head
     (which DOMINATES retrieval: random vs trained head = 0.124 vs 0.63 NDCG, S52-D10) would
     NOT be averaged.
  2. Averaging lora_A and lora_B separately ("linear") is only an approximation
     (avg(B)@avg(A) != avg(B@A)).
So instead we ``merge_and_unload`` each adapter into a FULL ColQwen3 (LoRA baked into the
base weights + trained custom_text_proj restored as a normal module), then take the exact
weighted average of the full state_dicts. This is the true Wortsman/Nemotron weight-average
(ColEmbed §3.3.6) and handles every tensor uniformly and exactly.

Output = a FULL merged model dir (config.json + model safetensors + processor), NO
adapter_config.json. The eval harness loads it directly (it now branches: dir without
adapter_config.json -> ColQwen3.from_pretrained(dir) instead of PeftModel) — see
scripts/eval/eval_colturk_checkpoint.py::_load_model.

NOTE: gains are real only with DIVERSE runs (different seed/LR/data). Averaging same-run
checkpoints (our 250..1500) yields little (verified +0.1-0.4). Use on best checkpoints of 2+
diverse runs (Phase D). Memory: loads one 4B model at a time on CPU + a fp32 accumulator
(~16GB) -> needs a high-RAM box (Colab high-RAM ok).

Usage:
  python scripts/training/soup_lora_adapters.py \
      --base Qwen/Qwen3-VL-4B-Instruct \
      --adapters Verm1ion/ColTurk-VDR-Stage1:checkpoint-1500 Verm1ion/ColTurk-VDR-runB:checkpoint-1500 \
      --weights 0.5 0.5 \
      --output /content/ckpts/soup_AB \
      --push-to-hub Verm1ion/ColTurk-VDR-soup

  # eval the soup exactly like any checkpoint (harness auto-detects full-model dir):
  python scripts/eval/eval_colturk_checkpoint.py --adapter /content/ckpts/soup_AB --bootstrap 1000 \
      --output eval/results/soup_AB.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_adapter(spec: str, workdir: str) -> str:
    """Local dir, or 'hf_repo:subfolder' (downloaded). Must contain adapter_config.json."""
    if os.path.isdir(spec) and os.path.exists(os.path.join(spec, "adapter_config.json")):
        return spec
    if ":" in spec and not os.path.isdir(spec):
        repo, sub = spec.split(":", 1)
        from huggingface_hub import snapshot_download

        local_root = os.path.join(workdir, repo.replace("/", "__"))
        logger.info("Downloading %s (%s) from HF -> %s", repo, sub, local_root)
        snapshot_download(repo, repo_type="model", allow_patterns=f"{sub}/*", local_dir=local_root)
        path = os.path.join(local_root, sub)
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            raise FileNotFoundError(f"adapter_config.json not found under {path}")
        return path
    raise FileNotFoundError(f"Cannot resolve adapter spec (need local adapter dir or 'repo:subfolder'): {spec}")


def soup_full(base: str, adapter_dirs: list[str], weights: list[float], out_dir: str, device: str = "cpu") -> None:
    import torch
    from colpali_engine.models import ColQwen3, ColQwen3Processor
    from peft import PeftModel

    s = sum(weights)
    weights = [w / s for w in weights]
    logger.info("Full-model soup of %d checkpoints, normalized weights %s", len(adapter_dirs), [round(w, 4) for w in weights])

    acc: dict[str, "torch.Tensor"] = {}
    last_merged = None
    for i, (w, d) in enumerate(zip(weights, adapter_dirs)):
        logger.info("[%d/%d] load base+adapter -> merge_and_unload: %s (w=%.4f)", i + 1, len(adapter_dirs), d, w)
        base_model = ColQwen3.from_pretrained(base, torch_dtype=torch.bfloat16).to(device)
        pm = PeftModel.from_pretrained(base_model, d)
        merged = pm.merge_and_unload()  # ColQwen3 with LoRA baked + trained custom_text_proj
        sd = merged.state_dict()
        for k, v in sd.items():
            t = v.detach().to(torch.float32) * w
            acc[k] = t if k not in acc else acc[k] + t
        last_merged = merged
        del base_model, pm, sd
        if i < len(adapter_dirs) - 1:
            del merged
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    if last_merged is None:
        raise RuntimeError("No adapters were merged — empty adapter_dirs?")

    # cast accumulator to bf16 and load into the last merged model (same arch/keys)
    logger.info("Casting averaged state_dict to bf16 and loading into model")
    for k in list(acc.keys()):
        acc[k] = acc[k].to(torch.bfloat16)
    missing, unexpected = last_merged.load_state_dict(acc, strict=False)
    if missing:
        logger.warning("load_state_dict missing keys (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        logger.warning("load_state_dict unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])

    os.makedirs(out_dir, exist_ok=True)
    logger.info("Saving merged FULL model -> %s", out_dir)
    last_merged.save_pretrained(out_dir)
    # processor so the eval harness can load proc from the merged dir
    try:
        ColQwen3Processor.from_pretrained(adapter_dirs[0]).save_pretrained(out_dir)
    except Exception:
        ColQwen3Processor.from_pretrained(base).save_pretrained(out_dir)
    with open(os.path.join(out_dir, "soup_manifest.json"), "w") as f:
        json.dump({"base": base, "adapters": adapter_dirs, "weights": weights, "method": "full-model weighted average"}, f, indent=2)
    logger.info("Soup complete. Eval with: --adapter %s (harness auto-detects full-model dir).", out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-model weighted-average soup of ColTurk-VDR LoRA checkpoints.")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct", help="base model id (LoRA was trained on this)")
    ap.add_argument("--adapters", nargs="+", required=True, help="local adapter dirs or 'hf_repo:subfolder' (>=2)")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="per-checkpoint weights (default uniform)")
    ap.add_argument("--output", required=True, help="output dir for the merged FULL model")
    ap.add_argument("--workdir", default="/content/ckpts/_soup_dl", help="scratch dir for HF downloads")
    ap.add_argument("--device", default="cpu", help="cpu (safe, ~24GB RAM) or cuda:0")
    ap.add_argument("--push-to-hub", default=None, help="optional HF model repo id to upload the soup")
    args = ap.parse_args()

    if len(args.adapters) < 2:
        ap.error("need >=2 adapters to soup")
    weights = args.weights or [1.0] * len(args.adapters)
    if len(weights) != len(args.adapters):
        ap.error("--weights count must match --adapters count")

    dirs = [_resolve_adapter(a, args.workdir) for a in args.adapters]
    soup_full(args.base, dirs, weights, args.output, device=args.device)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.push_to_hub, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=args.output, repo_id=args.push_to_hub, repo_type="model")
        logger.info("Pushed soup -> https://huggingface.co/%s", args.push_to_hub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
