"""
PyTorch Lightning module for CDCL transformer training.
"""

import logging
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig

from .dataset import (
    CDCLTrainDataset,
    CDCLTestDataset,
    collate_fn,
    create_test_dataloader,
)
from .masking import compute_loss_mask
from .metrics import exact_match, per_command_metrics, token_accuracy
from .model import CDCLTransformer
from .tokenizer import EOS_TOKEN, PAD_ID, PAD_TOKEN, decode, decode_tokens, get_vocab_size, load_tokenizer

logger = logging.getLogger(__name__)


class CDCLLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Load tokenizer and set vocab size
        self.tokenizer = load_tokenizer(cfg.data.tokenizer_path)
        vocab_size = get_vocab_size(self.tokenizer)

        self.model = CDCLTransformer(
            vocab_size=vocab_size,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            d_ff=cfg.model.d_ff,
            max_seq_len=cfg.model.max_seq_len,
        )

        # Store test set names for validation logging
        self.test_set_names = list(cfg.data.test_sets.keys())

        # Accumulate validation outputs per dataloader
        self._val_outputs: Dict[int, List] = defaultdict(list)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, loss_mask, attention_mask = batch

        # Autoregressive: predict next token
        logits = self(input_ids[:, :-1], attention_mask[:, :-1])
        targets = input_ids[:, 1:]
        target_mask = loss_mask[:, 1:]  # Shift mask to align with targets

        # Masked cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        loss = loss.reshape(targets.shape)
        mask_float = target_mask.float()
        loss = (loss * mask_float).sum() / mask_float.sum().clamp(min=1)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, loss_mask, attention_mask = batch

        logits = self(input_ids[:, :-1], attention_mask[:, :-1])
        targets = input_ids[:, 1:]
        target_mask = loss_mask[:, 1:]

        # Loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        mask_float = target_mask.float()
        masked_loss = (loss * mask_float).sum() / mask_float.sum().clamp(min=1)

        # Token accuracy
        preds = logits.argmax(dim=-1)
        tok_correct, tok_total = token_accuracy(preds, targets, target_mask)

        # Exact match
        em_correct, em_total = exact_match(preds, targets, target_mask)

        output = {
            "loss": masked_loss,
            "tok_correct": tok_correct,
            "tok_total": tok_total,
            "em_correct": em_correct,
            "em_total": em_total,
            "preds": preds.cpu(),
            "targets": targets.cpu(),
            "target_mask": target_mask.cpu(),
            "input_ids": input_ids.cpu(),
        }

        self._val_outputs[dataloader_idx].append(output)
        return output

    def on_validation_epoch_end(self):
        wandb_metrics = {}

        for dl_idx, outputs in self._val_outputs.items():
            if dl_idx >= len(self.test_set_names):
                continue

            name = self.test_set_names[dl_idx]
            # Parse "iid_combined" -> dist="iid", subset="combined"
            dist, subset = name.split("_", 1)
            prefix = f"{dist}/{subset}"

            # Aggregate metrics
            total_loss = sum(o["loss"].item() * o["em_total"] for o in outputs)
            total_samples = sum(o["em_total"] for o in outputs)
            total_tok_correct = sum(o["tok_correct"] for o in outputs)
            total_tok_total = sum(o["tok_total"] for o in outputs)
            total_em_correct = sum(o["em_correct"] for o in outputs)

            avg_loss = total_loss / max(total_samples, 1)
            tok_acc = total_tok_correct / max(total_tok_total, 1)
            em_acc = total_em_correct / max(total_samples, 1)

            # self.log for PL callbacks (e.g. ModelCheckpoint)
            self.log(f"{prefix}/loss", avg_loss, prog_bar=False, sync_dist=True)
            self.log(f"{prefix}/token_accuracy", tok_acc, prog_bar=(name == "iid_combined"), sync_dist=True)
            self.log(f"{prefix}/exact_match", em_acc, prog_bar=False, sync_dist=True)

            # Collect for direct wandb logging (PL's self.log may not flush to
            # WandbLogger from on_validation_epoch_end with multiple dataloaders)
            wandb_metrics[f"{prefix}/loss"] = avg_loss
            wandb_metrics[f"{prefix}/token_accuracy"] = tok_acc
            wandb_metrics[f"{prefix}/exact_match"] = em_acc

            # Per-command metrics and tables for combined sets
            if name in ("iid_combined", "ood_combined"):
                self._log_detailed_metrics(outputs, prefix, dist, wandb_metrics)

        # Explicitly log all scalar metrics to wandb
        if wandb_metrics and self.logger and hasattr(self.logger, "experiment"):
            wandb_metrics["trainer/global_step"] = self.global_step
            self.logger.experiment.log(wandb_metrics)

        self._val_outputs.clear()

    def _log_detailed_metrics(self, outputs, prefix, dist, wandb_metrics):
        """Log per-command metrics and prediction tables for combined test sets."""
        # Decode predictions and ground truths for per-command analysis
        all_pred_texts = []
        all_gt_texts = []
        all_masks = []
        all_em = []

        for o in outputs:
            preds = o["preds"]
            targets = o["targets"]
            target_mask = o["target_mask"]
            input_ids = o["input_ids"]

            for i in range(preds.size(0)):
                seq_len = (input_ids[i] != PAD_ID).sum().item()

                # Reconstruct full sequence with predictions replacing non-masked positions
                full_pred = input_ids[i, :seq_len].clone()
                for j in range(preds.size(1)):
                    pos = j + 1  # Position in full sequence
                    if pos < seq_len and j < target_mask.size(1) and target_mask[i, j] == 1:
                        full_pred[pos] = preds[i, j]

                # Use decode_tokens for accurate positional comparison
                pred_token_list = decode_tokens(self.tokenizer, full_pred.tolist())
                gt_token_list = decode_tokens(self.tokenizer, input_ids[i, :seq_len].tolist())

                # Filter out EOS for display text and mask computation
                gt_tokens_no_eos = [t for t in gt_token_list if t not in (PAD_TOKEN, EOS_TOKEN)]
                pred_tokens_no_eos = [t for t in pred_token_list if t not in (PAD_TOKEN, EOS_TOKEN)]

                gt_text = " ".join(gt_tokens_no_eos)
                pred_text = " ".join(pred_tokens_no_eos)
                raw_mask = compute_loss_mask(gt_tokens_no_eos)

                all_pred_texts.append(pred_text)
                all_gt_texts.append(gt_text)
                all_masks.append(raw_mask)

                # Check exact match on non-EOS tokens
                is_match = True
                if len(pred_tokens_no_eos) != len(gt_tokens_no_eos):
                    is_match = False
                else:
                    for p, g, m in zip(pred_tokens_no_eos, gt_tokens_no_eos, raw_mask):
                        if m == 1 and p != g:
                            is_match = False
                            break
                all_em.append(is_match)

        # Per-command metrics (separate WandB section: iid_cmd/ or ood_cmd/)
        cmd_metrics = per_command_metrics(all_pred_texts, all_gt_texts, all_masks)
        for cmd, (correct, total) in cmd_metrics.items():
            if total > 0:
                acc = correct / total
                self.log(f"{dist}_cmd/{cmd}", acc, sync_dist=True)
                wandb_metrics[f"{dist}_cmd/{cmd}"] = acc

        # WandB table (100 samples) - added to wandb_metrics for single log call
        if self.logger and hasattr(self.logger, "experiment"):
            sample_size = min(self.cfg.wandb.table_sample_size, len(all_pred_texts))
            indices = random.sample(range(len(all_pred_texts)), sample_size)

            table = wandb.Table(columns=["PREDICTED", "GROUND_TRUTH", "EXACT_MATCH"])
            for idx in indices:
                table.add_data(
                    all_pred_texts[idx],
                    all_gt_texts[idx],
                    all_em[idx],
                )
            wandb_metrics[f"{prefix}/examples"] = table

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
            betas=(0.9, 0.95),
        )

        # Linear warmup + cosine annealing (single LambdaLR)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.training.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            # Cosine decay with 10% floor (don't decay all the way to 0)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
