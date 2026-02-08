"""
Manual evaluation script. Takes a checkpoint and one or more test set paths.

Usage:
    python evaluate.py --checkpoint checkpoints/last.ckpt --test_sets output/iid_test.json
    python evaluate.py --checkpoint checkpoints/last.ckpt --test_sets output/iid_test.json output/ood_test.json
    python evaluate.py --checkpoint checkpoints/last.ckpt --test_sets output/iid_test.json --log_wandb
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from neural_cdcl.dataset import CDCLTestDataset, collate_fn
from neural_cdcl.lightning_module import CDCLLightningModule
from neural_cdcl.masking import compute_loss_mask
from neural_cdcl.metrics import exact_match, per_command_metrics, token_accuracy
from neural_cdcl.tokenizer import PAD_ID, decode, load_tokenizer

logger = logging.getLogger(__name__)


def evaluate_test_set(model, tokenizer, test_path, max_seq_len, tokenizer_path, device, batch_size=32):
    """Evaluate model on a single test set."""
    dataset = CDCLTestDataset(test_path, tokenizer_path, max_seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    total_tok_correct = 0
    total_tok_total = 0
    total_em_correct = 0
    total_em_total = 0
    all_pred_texts = []
    all_gt_texts = []
    all_masks = []
    all_em = []

    model.eval()
    with torch.no_grad():
        for input_ids, loss_mask, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            loss_mask = loss_mask.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids[:, :-1], attention_mask[:, :-1])
            targets = input_ids[:, 1:]
            target_mask = loss_mask[:, 1:]

            preds = logits.argmax(dim=-1)

            tc, tt = token_accuracy(preds, targets, target_mask)
            total_tok_correct += tc
            total_tok_total += tt

            ec, et = exact_match(preds, targets, target_mask)
            total_em_correct += ec
            total_em_total += et

            # Decode for per-command metrics
            for i in range(input_ids.size(0)):
                seq_len = (input_ids[i] != PAD_ID).sum().item()

                full_pred = input_ids[i, :seq_len].clone()
                for j in range(preds.size(1)):
                    pos = j + 1
                    if pos < seq_len and j < target_mask.size(1) and target_mask[i, j] == 1:
                        full_pred[pos] = preds[i, j]

                pred_text = decode(tokenizer, full_pred.tolist())
                gt_text = decode(tokenizer, input_ids[i, :seq_len].tolist())

                gt_tokens = gt_text.split()
                raw_mask = compute_loss_mask(gt_tokens)

                all_pred_texts.append(pred_text)
                all_gt_texts.append(gt_text)
                all_masks.append(raw_mask)

                pred_tokens = pred_text.split()
                is_match = pred_tokens == gt_tokens or (
                    len(pred_tokens) == len(gt_tokens)
                    and all(
                        p == g or m == 0
                        for p, g, m in zip(pred_tokens, gt_tokens, raw_mask)
                    )
                )
                all_em.append(is_match)

    tok_acc = total_tok_correct / max(total_tok_total, 1)
    em_acc = total_em_correct / max(total_em_total, 1)
    cmd_metrics = per_command_metrics(all_pred_texts, all_gt_texts, all_masks)

    return {
        "token_accuracy": tok_acc,
        "exact_match": em_acc,
        "per_command": {cmd: c / t for cmd, (c, t) in cmd_metrics.items() if t > 0},
        "pred_texts": all_pred_texts,
        "gt_texts": all_gt_texts,
        "em_list": all_em,
        "n_samples": total_em_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CDCL model on test sets")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_sets", type=str, nargs="+", required=True, help="Path(s) to test set JSON files")
    parser.add_argument("--tokenizer_path", type=str, default="models/tokenizer.json")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_wandb", action="store_true", help="Log results to WandB")
    parser.add_argument("--wandb_project", type=str, default="neural-cdcl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = OmegaConf.create(ckpt["hyper_parameters"])
    model = CDCLLightningModule(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer_path)

    wandb_run = None
    if args.log_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, job_type="evaluate")

    for test_path in args.test_sets:
        test_name = Path(test_path).stem
        logger.info("Evaluating on %s...", test_name)

        results = evaluate_test_set(
            model.model,
            tokenizer,
            test_path,
            args.max_seq_len,
            args.tokenizer_path,
            device,
            args.batch_size,
        )

        print(f"\n{'='*60}")
        print(f"Results for {test_name} ({results['n_samples']} samples)")
        print(f"{'='*60}")
        print(f"  Token Accuracy: {results['token_accuracy']:.4f}")
        print(f"  Exact Match:    {results['exact_match']:.4f}")
        print(f"\n  Per-command accuracy:")
        for cmd, acc in sorted(results["per_command"].items()):
            print(f"    {cmd:30s}: {acc:.4f}")

        if wandb_run:
            import wandb
            wandb_run.log({
                f"eval/{test_name}/token_accuracy": results["token_accuracy"],
                f"eval/{test_name}/exact_match": results["exact_match"],
            })
            for cmd, acc in results["per_command"].items():
                wandb_run.log({f"eval/{test_name}/cmd/{cmd}": acc})

            # Table with 100 samples
            sample_size = min(100, len(results["pred_texts"]))
            indices = random.sample(range(len(results["pred_texts"])), sample_size)
            table = wandb.Table(columns=["PREDICTED", "GROUND_TRUTH", "EXACT_MATCH"])
            for idx in indices:
                table.add_data(
                    results["pred_texts"][idx],
                    results["gt_texts"][idx],
                    results["em_list"][idx],
                )
            wandb_run.log({f"eval/{test_name}/examples": table})

    if wandb_run:
        wandb_run.finish()

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
