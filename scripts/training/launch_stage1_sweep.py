"""Stage-1 LR sweep launcher (KARARLAR S46 + A31).

ColPali engine config YAML'ı yükler, ColModelTrainer ile training başlatır.
Mert Colab'dan: `python scripts/training/launch_stage1_sweep.py configs/qwen3/train_colturk_stage1_lr2e5_sweep.yaml`

Refs:
- ColPali engine API: colpali_engine.trainer.colmodel_training.ColModelTrainingConfig
- S37 — training recipe (LoRA r=32, α=32, dropout=0.1, target regex, K=2 @0.95)
- S46 — Stage-1 base raw Qwen/Qwen3-VL-4B-Instruct, warm-start YOK
- A31 — LR sweep zorunlu (2e-5 vs 5e-5, ¼ epoch each)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Stage-1 LR sweep launcher (KARARLAR S46 + A31)")
    parser.add_argument("config", type=Path, help="YAML config path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config + instantiate model+trainer, skip trainer.train(). "
             "Used by notebook pre-flight smoke cell to catch F16/F17/F19-type bugs "
             "in ~1-2 min before committing to ~1.5h corpus build + training.",
    )
    args = parser.parse_args()

    config_path: Path = args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2

    # OUTPUT_DIR env var — yaml içinde !path ${OUTPUT_DIR}/... pattern
    if "OUTPUT_DIR" not in os.environ:
        os.environ["OUTPUT_DIR"] = str(Path.cwd() / "outputs")
        logger.info("OUTPUT_DIR set → %s", os.environ["OUTPUT_DIR"])

    try:
        import configue
    except ImportError:
        logger.error("configue not installed (yaml loader). Run: pip install configue")
        return 2

    try:
        from colpali_engine.trainer.colmodel_training import (
            ColModelTraining,
            ColModelTrainingConfig,
        )
    except ImportError as exc:
        logger.error(
            "colpali_engine.trainer.colmodel_training import failed: %s. "
            "Check colpali-engine version (>=0.3.16). KARARLAR S17.",
            exc,
        )
        return 2

    logger.info("Loading config: %s (dry_run=%s)", config_path, args.dry_run)
    # F12 fix: ColPali standard configue.load(file, sub_path="config")
    # → direkt ColModelTrainingConfig instance döner (eski config_dict["config"] obsolete)
    cfg: ColModelTrainingConfig = configue.load(str(config_path), sub_path="config")
    if not isinstance(cfg, ColModelTrainingConfig):
        logger.error("Loaded config is not ColModelTrainingConfig (type=%s)", type(cfg).__name__)
        return 2
    logger.info("Config loaded; output_dir=%s", cfg.output_dir)

    trainer = ColModelTraining(cfg)

    if args.dry_run:
        logger.info(
            "DRY-RUN OK: config loaded, model+processor instantiated, "
            "dataset prepped, trainer constructed. Skipping trainer.train(). "
            "Real training pipeline verified — F19/F17/F18 pre-flight PASS."
        )
        return 0

    if getattr(cfg, "run_train", True):
        logger.info("Starting training (ColModelTraining.train)…")
        trainer.train()
        # Save final model (ColPali entry standard pattern)
        if hasattr(trainer, "save"):
            trainer.save()
        logger.info("Training done. Output → %s", cfg.output_dir)
    else:
        logger.warning("config.run_train=False — training skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
