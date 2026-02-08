"""
Testing entry point. Loads a checkpoint and evaluates on all test sets.

Usage:
    python test.py checkpoint=checkpoints/last.ckpt
    python test.py checkpoint=checkpoints/last.ckpt wandb.project=my-project
"""

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from neural_cdcl.dataset import create_test_dataloader
from neural_cdcl.lightning_module import CDCLLightningModule

logger = logging.getLogger(__name__)


def resolve_path(path: str, orig_cwd: str) -> str:
    """Resolve a potentially relative path against the original working directory."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(orig_cwd) / p
    return str(p)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is None:
        raise ValueError("Must specify checkpoint path: python test.py checkpoint=path/to/ckpt")

    checkpoint_path = resolve_path(checkpoint_path, orig_cwd)
    tokenizer_path = resolve_path(cfg.data.tokenizer_path, orig_cwd)

    logger.info("Loading checkpoint from %s", checkpoint_path)

    # Resolve paths in config
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg_resolved["data"]["tokenizer_path"] = tokenizer_path
    for name, path in cfg.data.test_sets.items():
        cfg_resolved["data"]["test_sets"][name] = resolve_path(path, orig_cwd)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    # Load model from checkpoint
    model = CDCLLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg_resolved)

    # Create test dataloaders
    test_dataloaders = []
    for name in cfg.data.test_sets.keys():
        dl = create_test_dataloader(
            json_path=cfg_resolved.data.test_sets[name],
            tokenizer_path=tokenizer_path,
            max_seq_len=cfg.data.val_max_seq_len,
            batch_size=cfg.training.get("val_batch_size", cfg.training.batch_size),
            num_workers=cfg.training.num_workers,
        )
        test_dataloaders.append(dl)

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"test-{cfg.wandb.name}" if cfg.wandb.name else "test",
        tags=list(cfg.wandb.tags) + ["test"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Create trainer and test
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision=cfg.training.precision,
        logger=wandb_logger,
    )

    trainer.validate(model, test_dataloaders)

    logger.info("Testing complete.")


if __name__ == "__main__":
    main()
