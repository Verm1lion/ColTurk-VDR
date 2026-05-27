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
             "Catches F16/F17/F19/F20/F22-type bugs (config + model + trainer construct).",
    )
    parser.add_argument(
        "--smoke-step",
        action="store_true",
        help="Like --dry-run but ALSO runs 1 training step (~30-60sn A100). "
             "Catches F23+ (dataloader + sampler + collator + forward + compute_loss + "
             "backward + optim) — issues that trigger only at trainer.train() time.",
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

    # F20 fix: colpali_engine.utils.dataset_transformation module top'ta:
    #   USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"
    # Default "1" (TRUE) → load_train_set_ir base_path = "./data_dir/" (yok).
    # "0" override → base_path = "manu/" → HF manu/colpali-corpus + manu/colpali-queries
    # (her ikisi public, parquet, WebFetch ile mevcut doğrulandı).
    # colpali_engine import'tan ÖNCE set edilmeli (module-load time read).
    os.environ.setdefault("USE_LOCAL_DATASET", "0")
    logger.info("USE_LOCAL_DATASET=%s (HF manu/* path)", os.environ["USE_LOCAL_DATASET"])

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

    logger.info(
        "Loading config: %s (dry_run=%s smoke_step=%s)",
        config_path, args.dry_run, args.smoke_step,
    )
    # F12 fix: ColPali standard configue.load(file, sub_path="config")
    # → direkt ColModelTrainingConfig instance döner (eski config_dict["config"] obsolete)
    cfg: ColModelTrainingConfig = configue.load(str(config_path), sub_path="config")
    if not isinstance(cfg, ColModelTrainingConfig):
        logger.error("Loaded config is not ColModelTrainingConfig (type=%s)", type(cfg).__name__)
        return 2
    logger.info("Config loaded; output_dir=%s", cfg.output_dir)

    # F23 fix: ColPali ContrastiveTrainer._get_train_sampler(self) — v4-style, no train_dataset arg.
    # transformers v5 Trainer._get_dataloader (line 987) calls sampler_fn(dataset) — passes dataset arg.
    # TypeError: "_get_train_sampler() takes 1 positional argument but 2 were given".
    # Monkey-patch in module dict BEFORE ColModelTraining instantiates ContrastiveTrainer (in .train()).
    from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer

    def _patched_get_train_sampler(self, train_dataset=None):
        """v5-compat signature. Routes:
        - train_dataset_list None (single dataset, our case): delegate to v5 Trainer base
          which accepts and USES the train_dataset arg internally.
        - train_dataset_list set (multi-dataset ColPali concat): use SingleDatasetBatchSampler
          (ignores train_dataset arg, uses self.train_dataset_list set in __init__).
        """
        if self.train_dataset_list is None:
            from transformers import Trainer
            return Trainer._get_train_sampler(self, train_dataset)
        import torch
        from colpali_engine.data.sampler import SingleDatasetBatchSampler
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.train_dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )

    ContrastiveTrainer._get_train_sampler = _patched_get_train_sampler
    logger.info(
        "F23 patch applied: ContrastiveTrainer._get_train_sampler now accepts train_dataset arg"
    )

    # --smoke-step: override tr_args to do 1 step + no save + no wandb (catches F24+ pre-flight)
    if args.smoke_step:
        logger.info("SMOKE-STEP: overriding tr_args (max_steps=1, save=no, report_to=[])")
        cfg.tr_args.max_steps = 1
        cfg.tr_args.save_strategy = "no"
        cfg.tr_args.report_to = []
        cfg.tr_args.logging_steps = 1

    trainer = ColModelTraining(cfg)

    if args.dry_run:
        logger.info(
            "DRY-RUN OK: config loaded, model+processor instantiated, "
            "dataset prepped, trainer constructed. Skipping trainer.train()."
        )
        return 0

    if args.smoke_step:
        logger.info("SMOKE-STEP: invoking trainer.train() for 1 step…")
        trainer.train()
        logger.info(
            "SMOKE-STEP OK: 1 training step completed without error. "
            "Full pipeline (dataloader + sampler + collator + forward + compute_loss + "
            "backward + optim) verified."
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
