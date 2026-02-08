"""
Training entry point for the CDCL transformer model.

Usage:
    python train.py
    python train.py model.d_model=256 training.batch_size=16
    python train.py --config-name=default
"""

import logging
import os
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from neural_cdcl.dataset import (
    CDCLTrainDataset,
    collate_fn,
    create_test_dataloader,
    preprocess_training_data,
)
from neural_cdcl.lightning_module import CDCLLightningModule
from neural_cdcl.tokenizer import build_tokenizer, get_vocab_size, load_tokenizer

logger = logging.getLogger(__name__)


class EpochTimerCallback(pl.Callback):
    """Logs epoch duration to stdout."""

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._epoch_start
        print(
            f"Epoch {trainer.current_epoch} completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)"
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._val_start
        logger.info("Validation completed in %.1fs", elapsed)


def resolve_path(path: str, orig_cwd: str) -> str:
    """Resolve a potentially relative path against the original working directory."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(orig_cwd) / p
    return str(p)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Ensure logging shows on console (Hydra can redirect it)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)

    # Hydra changes cwd; resolve all paths relative to the original cwd
    orig_cwd = hydra.utils.get_original_cwd()

    print("=" * 60)
    print("NeuralCDCL2 Training")
    print("=" * 60)
    print(f"  Model: d_model={cfg.model.d_model}, layers={cfg.model.n_layers}, "
          f"heads={cfg.model.n_heads}, d_ff={cfg.model.d_ff}")
    print(f"  Train seq len: {cfg.data.train_max_seq_len}, RoPE max: {cfg.model.max_seq_len}")
    print(f"  Training: bs={cfg.training.batch_size}, "
          f"accum={cfg.training.gradient_accumulation_steps}, "
          f"eff_bs={cfg.training.batch_size * cfg.training.gradient_accumulation_steps}")
    print(f"  LR: {cfg.training.lr}, warmup: {cfg.training.warmup_steps} steps")
    print(f"  Epochs: {cfg.training.max_epochs}, precision: {cfg.training.precision}")
    print("=" * 60)

    # Resolve paths
    train_path = resolve_path(cfg.data.train_path, orig_cwd)
    tokenizer_path = resolve_path(cfg.data.tokenizer_path, orig_cwd)
    preprocessed_dir = resolve_path(cfg.data.preprocessed_dir, orig_cwd)
    checkpoint_dir = resolve_path(cfg.checkpointing.dirpath, orig_cwd)

    test_set_paths = {}
    for name, path in cfg.data.test_sets.items():
        test_set_paths[name] = resolve_path(path, orig_cwd)

    # Step 1: Build tokenizer if it doesn't exist
    if not Path(tokenizer_path).exists():
        logger.info("Building tokenizer from %s...", train_path)
        tokenizer = build_tokenizer(train_path, tokenizer_path)
        logger.info("Tokenizer built. Vocab size: %d", get_vocab_size(tokenizer))
    else:
        tokenizer = load_tokenizer(tokenizer_path)
        logger.info("Loaded existing tokenizer. Vocab size: %d", get_vocab_size(tokenizer))

    # Step 2: Preprocess training data if needed
    preprocess_training_data(
        train_path=train_path,
        tokenizer_path=tokenizer_path,
        output_dir=preprocessed_dir,
        max_seq_len=cfg.data.val_max_seq_len,
    )

    # Update config with resolved paths for the Lightning module
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg_resolved["data"]["tokenizer_path"] = tokenizer_path
    cfg_resolved["data"]["preprocessed_dir"] = preprocessed_dir
    for name, path in test_set_paths.items():
        cfg_resolved["data"]["test_sets"][name] = path
    cfg_resolved = OmegaConf.create(cfg_resolved)

    # Step 3: Create datasets and dataloaders
    # Training: filter to train_max_seq_len (shorter sequences for memory efficiency)
    train_max_seq_len = cfg.data.train_max_seq_len
    train_dataset = CDCLTrainDataset(preprocessed_dir, train_max_seq_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation: use full max_seq_len (no filtering) with smaller batch size
    val_batch_size = cfg.training.get("val_batch_size", cfg.training.batch_size)
    val_dataloaders = []
    max_val_samples = cfg.data.get("max_val_samples", 0)
    for name in cfg.data.test_sets.keys():
        dl = create_test_dataloader(
            json_path=test_set_paths[name],
            tokenizer_path=tokenizer_path,
            max_seq_len=cfg.data.val_max_seq_len,
            batch_size=val_batch_size,
            num_workers=cfg.training.num_workers,
            max_samples=max_val_samples,
        )
        val_dataloaders.append(dl)

    # Step 4: Create model
    model = CDCLLightningModule(cfg_resolved)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Batches/epoch: {len(train_dataloader)}")
    print("=" * 60)

    # Step 5: Setup WandB logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Step 6: Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.checkpointing.monitor,
        mode=cfg.checkpointing.mode,
        save_top_k=cfg.checkpointing.save_top_k,
        save_last=cfg.checkpointing.save_last,
        filename="{epoch}-{step}-{iid/combined/token_accuracy:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    epoch_timer = EpochTimerCallback()

    # Step 7: Create trainer and fit
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        val_check_interval=cfg.training.val_check_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, epoch_timer],
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloader, val_dataloaders)


if __name__ == "__main__":
    main()
