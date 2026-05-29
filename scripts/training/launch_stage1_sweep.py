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
    parser.add_argument(
        "--push-to-hub",
        default="",
        help="HF repo id to push the trained adapter to after training "
             "(e.g. Verm1ion/ColTurk-VDR-Stage1-val-lr5e5). S49 persistence fix — "
             "Colab /content is ephemeral; without this the adapter is lost on disconnect. "
             "Uses HF_TOKEN env. Private repo. Skipped for --dry-run/--smoke-step.",
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

    # W&B routing: HF WandbCallback defaults project="huggingface" + key-owner default entity +
    # no grouping (verified transformers v5.9.0 integration_utils WandbCallback.setup). Set these so
    # the two sweep runs land at wandb.ai/verm1lion-colturk-vdr/colturk-vdr grouped under stage1-lr-sweep.
    # Read by wandb SDK at init (on_train_begin); setdefault preserves any user override. Observability
    # only — S47 verdict reads local trainer_state.json, NOT wandb.
    os.environ.setdefault("WANDB_PROJECT", "colturk-vdr")
    os.environ.setdefault("WANDB_ENTITY", "verm1lion-colturk-vdr")
    os.environ.setdefault("WANDB_RUN_GROUP", "stage1-lr-sweep")

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

    # F24 fix: ColPali get_train_dataloader sets self.query_prefix/pos_prefix/neg_prefix
    # ONLY in the multi-dataset (train_dataset_list set) branch. For a single dataset
    # (load_train_set_ir returns one ColPaliEngineDataset → train_dataset_list=None),
    # get_train_dataloader early-returns via super().get_train_dataloader() WITHOUT
    # setting those prefixes, then compute_loss crashes:
    #   AttributeError: 'ContrastiveTrainer' object has no attribute 'query_prefix'
    # Patch: set the 3 prefixes from the collator unconditionally before delegating.
    # Collator class attrs (VisualRetrieverCollator): query_prefix="query_",
    # pos_doc_prefix="doc_", neg_doc_prefix="neg_doc_" (WebFetch verified).
    _orig_get_train_dataloader = ContrastiveTrainer.get_train_dataloader

    def _patched_get_train_dataloader(self):
        self.query_prefix = self.data_collator.query_prefix
        self.pos_prefix = self.data_collator.pos_doc_prefix
        self.neg_prefix = self.data_collator.neg_doc_prefix
        return _orig_get_train_dataloader(self)

    ContrastiveTrainer.get_train_dataloader = _patched_get_train_dataloader
    logger.info(
        "F24 patch applied: get_train_dataloader sets query/pos/neg prefixes unconditionally"
    )

    # F25 fix: W&B graceful fallback. yaml has report_to="wandb"; if WANDB_API_KEY is
    # missing/invalid (<40 chars — legacy keys are 40 hex, new keys "wandb_v1_..." are 80+),
    # wandb.init() in the on_train_begin callback HARD-CRASHES the whole training.
    # The sweep verdict (S47) reads trainer_state.json (local file), NOT wandb — so wandb is
    # purely optional live-monitoring. Disable it gracefully if key invalid → training proceeds,
    # loss still logged to trainer_state.json.
    _wandb_key = os.environ.get("WANDB_API_KEY", "")
    _report_to = cfg.tr_args.report_to
    _report_list = _report_to if isinstance(_report_to, list) else ([_report_to] if _report_to else [])
    if "wandb" in _report_list and len(_wandb_key) < 40:
        logger.warning(
            "WANDB_API_KEY invalid (len=%d, need 40+) — disabling wandb (report_to=[]). "
            "Training continues; loss still saved to trainer_state.json for S47 verdict. "
            "To enable wandb: paste a valid 40+ char key into Colab Secrets (wandb.ai/authorize).",
            len(_wandb_key),
        )
        cfg.tr_args.report_to = []
    elif "wandb" in _report_list:
        logger.info("WANDB_API_KEY valid (len=%d) — wandb logging enabled", len(_wandb_key))

    # --smoke-step: override tr_args to do 1 step + TEST save path + no wandb (catches F24+/save pre-flight)
    if args.smoke_step:
        logger.info(
            "SMOKE-STEP: overriding tr_args (max_steps=1, save_strategy=steps + save_steps=1 "
            "to TEST checkpoint/PEFT-adapter save path, report_to=[])"
        )
        cfg.tr_args.max_steps = 1
        cfg.tr_args.save_strategy = "steps"
        cfg.tr_args.save_steps = 1          # F26 pre-flight: validate save path (real run saves @ step 500)
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
        logger.info("SMOKE-STEP: invoking trainer.train() for 1 step (+ save @ step 1)…")
        trainer.train()
        logger.info(
            "SMOKE-STEP OK: 1 training step + checkpoint save completed without error. "
            "Full pipeline (dataloader + sampler + collator + forward + compute_loss + "
            "backward + optim + PEFT-adapter save) verified."
        )
        return 0

    if getattr(cfg, "run_train", True):
        logger.info("Starting training (ColModelTraining.train)…")
        trainer.train()
        # Save final model (ColPali entry standard pattern)
        if hasattr(trainer, "save"):
            trainer.save()
        logger.info("Training done. Output → %s", cfg.output_dir)

        # S49 persistence: push adapter off-box before the ephemeral Colab disk dies.
        if args.push_to_hub:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ.get("HF_TOKEN"))
                api.create_repo(args.push_to_hub, repo_type="model", private=True, exist_ok=True)
                api.upload_folder(
                    folder_path=str(cfg.output_dir),
                    repo_id=args.push_to_hub,
                    repo_type="model",
                )
                logger.info("Pushed adapter → https://huggingface.co/%s", args.push_to_hub)
            except Exception as exc:
                logger.error(
                    "push-to-hub FAILED (%s) — adapter still on local disk at %s. "
                    "Rescue manually before the session dies.", exc, cfg.output_dir
                )
    else:
        logger.warning("config.run_train=False — training skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
