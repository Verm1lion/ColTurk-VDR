"""Stage-3 model merging — Nemotron linear weighted-average recipe.

Nemotron ColEmbed V2 §3.3.6 verbatim verified (2026-05-18 cross-validation):
    "For Nemotron ColEmbed V2 models, we employed a simple weighted average
    of model weights for merging."

NOT SLERP, NOT DARE-TIES, NOT task-arithmetic. Just simple linear average.
Paper: 4B/8B models merged from 4 individual checkpoints each;
beklenen lift +0.8 ila +1.5 NDCG@10 (3B'de daha düşük, 8B'de en yüksek).

Diversity maximization (Nemotron §3.3.6): "variations in the training blend
and in some hyperparameters". Pratik 4-varyant matris:
    A: seed=42,   lr=5e-5, hard-neg=0.95, blend=full-TR
    B: seed=1337, lr=3e-5, hard-neg=0.90, blend=TR-heavy
    C: seed=2024, lr=7e-5, hard-neg=0.95, blend=TR+FR cross-lingual
    D: seed=7777, lr=5e-5, hard-neg=0.93, blend=TR+ColPali-EN-replay

Mergekit native: `mergekit-yaml configs/merge.yml` with `merge_method: linear`.
Fallback: state_dict averaging (5 lines) when mergekit Qwen3-VL edge cases hit
(issue #669/#672/#675, Mart 2026 — bazı vision tower weight handling bug'ları).

LoRA pre-merge: ColPali-engine LoRA-trained verir; mergekit full weights
bekler. Her checkpoint için merge_and_unload() çağrılır (zaten 4 ayrı full
model olarak diske yazılı olmalı).

KARARLAR S30: model merging P1, linear weighted average, normalize=true.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

LOG = logging.getLogger("merge_checkpoints")


def merge_with_mergekit(yaml_path: Path, output_path: Path) -> bool:
    """Run `mergekit-yaml` via subprocess. Returns True if successful."""
    import subprocess

    cmd = [
        "mergekit-yaml",
        str(yaml_path),
        str(output_path),
        "--cuda",
        "--lazy-unpickle",
        "--allow-crimes",
    ]
    LOG.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        LOG.info("mergekit-yaml output:\n%s", result.stdout)
        return True
    except subprocess.CalledProcessError as exc:
        LOG.error("mergekit-yaml failed (returncode=%s):\n%s", exc.returncode, exc.stderr)
        return False
    except FileNotFoundError:
        LOG.error("mergekit-yaml not installed. Install with: pip install mergekit")
        return False


def merge_with_state_dict(
    checkpoint_paths: list[Path],
    weights: list[float],
    output_path: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Fallback: manual state_dict weighted average.

    Used when mergekit hits Qwen3-VL-specific bugs (issues open as of Mart 2026).
    Loads checkpoints sequentially to keep RAM bounded.
    """
    from transformers import AutoModel  # type: ignore[import-not-found]

    if len(checkpoint_paths) != len(weights):
        raise ValueError("checkpoint_paths and weights must have equal length")
    if not abs(sum(weights) - 1.0) < 1e-6:
        LOG.warning("Weights sum to %.4f, normalizing to 1.0", sum(weights))
        total = sum(weights)
        weights = [w / total for w in weights]

    LOG.info("Loading first checkpoint: %s (weight=%.4f)", checkpoint_paths[0], weights[0])
    base_model = AutoModel.from_pretrained(
        str(checkpoint_paths[0]), torch_dtype=dtype, trust_remote_code=True
    )
    merged_sd = {k: v.float() * weights[0] for k, v in base_model.state_dict().items()}
    del base_model

    for ckpt, w in zip(checkpoint_paths[1:], weights[1:]):
        LOG.info("Adding checkpoint: %s (weight=%.4f)", ckpt, w)
        m = AutoModel.from_pretrained(str(ckpt), torch_dtype=dtype, trust_remote_code=True)
        for k, v in m.state_dict().items():
            if k in merged_sd:
                merged_sd[k] = merged_sd[k] + v.float() * w
            else:
                LOG.warning("Key %s missing from base; skipping", k)
        del m

    LOG.info("Loading base model again for save (with merged weights)...")
    final = AutoModel.from_pretrained(
        str(checkpoint_paths[0]), torch_dtype=dtype, trust_remote_code=True
    )
    final.load_state_dict({k: v.to(dtype) for k, v in merged_sd.items()})
    output_path.mkdir(parents=True, exist_ok=True)
    final.save_pretrained(str(output_path))
    LOG.info("Merged model saved to %s", output_path)


def merge(
    checkpoints: list[Path],
    weights: list[float],
    output_path: Path,
    config_yaml: Path | None = None,
) -> None:
    """Try mergekit first; fall back to state_dict averaging on failure."""
    if config_yaml is not None and config_yaml.exists():
        if merge_with_mergekit(config_yaml, output_path):
            return
        LOG.warning("mergekit-yaml failed; falling back to manual state_dict averaging")
    merge_with_state_dict(checkpoints, weights, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 Nemotron linear weighted-average merge")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="Per-checkpoint weights (default: uniform 1/N)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mergekit-config", type=Path, default=None,
        help="Optional mergekit YAML to try first (linear merge_method)",
    )
    args = parser.parse_args()

    if args.weights is None:
        n = len(args.checkpoints)
        args.weights = [1.0 / n] * n

    merge(args.checkpoints, args.weights, args.output, args.mergekit_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
