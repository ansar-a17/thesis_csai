"""
train timesformer on ucf101 with early action sampling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import TimesformerForVideoClassification, TimesformerConfig
from tqdm import tqdm
import argparse
import json
import os
import random
import sys
import time

from sampling import UCF101EarlyActionDataset

MODEL_ID = "facebook/timesformer-base-finetuned-k400"

# wrapper to return plain logits from huggingface output
class TimeSformerWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        # x is b t c h w for timesformer pixel values
        return self.model(pixel_values=x).logits


def load_timesformer_model(num_labels):
    config = TimesformerConfig.from_pretrained(MODEL_ID)
    config.num_frames = 8
    config.num_labels = num_labels

    # prioritize safetensors to avoid torch load restrictions
    load_attempts = [
        {"revision": "main", "use_safetensors": True},
        {"revision": "refs/pr/5", "use_safetensors": True},
    ]

    for attempt in load_attempts:
        try:
            revision = attempt["revision"]
            print(f"Trying TimeSformer weights from revision: {revision} (safetensors)")
            return TimesformerForVideoClassification.from_pretrained(
                MODEL_ID,
                config=config,
                revision=revision,
                use_safetensors=attempt["use_safetensors"],
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            print(f"Failed loading revision '{revision}': {exc}")

    raise RuntimeError(
        "Could not load TimeSformer safetensors weights. "
        "This script loads safetensors only; check your network/cache and model revision access."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TimeSformer on UCF101 and evaluate Top-1/Top-5 and per-class metrics."
    )
    parser.add_argument("--video_root", type=str, default="UCF101")
    parser.add_argument("--train_split", type=str, default="splits/trainlist01.txt")
    parser.add_argument("--test_split", type=str, default="splits/testlist01.txt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Stop training when validation loss does not improve for this many epochs. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss decrease required to count as an improvement.",
    )
    parser.add_argument(
        "--eval_fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5],
        help="Temporal fractions used for final evaluation.",
    )
    parser.add_argument(
        "--checkpoint_fraction",
        type=float,
        default=0.5,
        help="Temporal fraction used to select the best model checkpoint.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_name", type=str, default="best_timesformer_model.pth")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_div(num, den):
    return (num / den) if den else 0.0


def compute_topk_correct(outputs, labels, k):
    k = min(k, outputs.size(1))
    topk_idx = outputs.topk(k, dim=1).indices
    return topk_idx.eq(labels.unsqueeze(1)).any(dim=1).sum().item()


def build_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_dataset(video_root, split_path, clip_len, fraction, transform):
    return UCF101EarlyActionDataset(
        video_root,
        split_path,
        clip_len=clip_len,
        fraction=fraction,
        transform=transform,
        model_type="transformer",
    )


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


# training loop
def train_epoch(model, loader, criterion, optimizer, device, show_progress=True):
    model.train()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", disable=not show_progress)
    for clips, labels in pbar:
        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        top1_correct += outputs.argmax(1).eq(labels).sum().item()
        top5_correct += compute_topk_correct(outputs, labels, k=5)
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "top1": f"{100.0 * safe_div(top1_correct, total):.2f}%",
            "top5": f"{100.0 * safe_div(top5_correct, total):.2f}%",
        })

    return {
        "loss": running_loss / max(1, len(loader)),
        "top1": 100.0 * safe_div(top1_correct, total),
        "top5": 100.0 * safe_div(top5_correct, total),
    }


# validation loop
def evaluate(model, loader, criterion, device, num_classes, show_progress=True):
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluation", disable=not show_progress)
        for clips, labels in pbar:
            clips, labels = clips.to(device), labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = outputs.argmax(1)
            top1_correct += predicted.eq(labels).sum().item()
            top5_correct += compute_topk_correct(outputs, labels, k=5)
            total += labels.size(0)

            y_true = labels.detach().cpu().tolist()
            y_pred = predicted.detach().cpu().tolist()
            for t, p in zip(y_true, y_pred):
                if p == t:
                    tp[t] += 1
                else:
                    fp[p] += 1
                    fn[t] += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "top1": f"{100.0 * safe_div(top1_correct, total):.2f}%",
                "top5": f"{100.0 * safe_div(top5_correct, total):.2f}%",
            })

    per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for c in range(num_classes):
        precision = safe_div(tp[c], tp[c] + fp[c])
        recall = safe_div(tp[c], tp[c] + fn[c])
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = tp[c] + fn[c]

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    macro_precision = safe_div(macro_precision, num_classes)
    macro_recall = safe_div(macro_recall, num_classes)
    macro_f1 = safe_div(macro_f1, num_classes)

    return {
        "loss": running_loss / max(1, len(loader)),
        "top1": 100.0 * safe_div(top1_correct, total),
        "top5": 100.0 * safe_div(top5_correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_tqdm = (not args.disable_tqdm) and sys.stdout.isatty()
    pin_memory = device.type == "cuda"

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

    print(f"Using device: {device}")
    print(f"SLURM job id: {slurm_job_id}")
    print(f"Train fraction: {args.train_fraction}")
    print(f"Eval fractions: {args.eval_fractions}")
    print(f"Checkpoint selection fraction: {args.checkpoint_fraction}")

    train_transform = build_transforms(train=True)
    test_transform = build_transforms(train=False)

    print("Loading datasets...")
    # split files list video paths relative to video_root
    train_dataset = make_dataset(
        args.video_root,
        args.train_split,
        args.clip_len,
        args.train_fraction,
        train_transform,
    )
    val_dataset = make_dataset(
        args.video_root,
        args.test_split,
        args.clip_len,
        args.checkpoint_fraction,
        test_transform,
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    num_classes = len(train_dataset.class_to_idx)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    print("Loading pre-trained TimeSformer (Kinetics-400)...")
    base_model = load_timesformer_model(num_classes)
    model = TimeSformerWrapper(base_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\nStarting training...\n")
    best_top1 = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            show_progress=use_tqdm,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=num_classes,
            show_progress=use_tqdm,
        )
        scheduler.step()

        epoch_time = time.time() - start_time

        print(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Top-1: {train_metrics['top1']:.2f}%, "
            f"Train Top-5: {train_metrics['top5']:.2f}%"
        )
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Top-1: {val_metrics['top1']:.2f}%, "
            f"Val Top-5: {val_metrics['top5']:.2f}%, "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )
        print(f"Epoch Time: {epoch_time / 60:.2f} min\n")
        epochs_trained = epoch

        # early stopping based on validation loss
        if val_metrics["loss"] < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            print(
                f"Validation loss improved to {best_val_loss:.4f}. "
                "Resetting early-stopping counter."
            )
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    f"Validation loss did not improve for {epochs_without_improvement}/"
                    f"{args.early_stopping_patience} epochs."
                )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_top1": best_top1,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved best model to: {checkpoint_path} (Top-1: {best_top1:.2f}%)\n")

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping triggered: validation loss has not improved for "
                f"{args.early_stopping_patience} consecutive epochs."
            )
            break

    if not os.path.exists(checkpoint_path):
        # fallback checkpoint for final evaluation
        torch.save(
            {
                "epoch": epochs_trained,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_top1": best_top1,
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"No best checkpoint found during training; saved current model to: {checkpoint_path}")

    # evaluate best checkpoint at multiple temporal fractions
    print("Training complete. Loading best checkpoint for final temporal evaluation...")
    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    final_results = {
        "slurm_job_id": slurm_job_id,
        "checkpoint_path": checkpoint_path,
        "best_val_top1": best_ckpt.get("best_val_top1", best_top1),
        "epochs_completed": epochs_trained,
        "max_epochs": args.epochs,
        "early_stopping": {
            "enabled": args.early_stopping_patience > 0,
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "best_val_loss": best_val_loss,
            "stopped_early": epochs_trained < args.epochs,
        },
        "temporal_conditions": {},
    }

    for fraction in args.eval_fractions:
        print(f"\nEvaluating temporal fraction={fraction}")
        eval_dataset = make_dataset(
            args.video_root,
            args.test_split,
            args.clip_len,
            fraction,
            test_transform,
        )
        eval_loader = make_loader(
            eval_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        metrics = evaluate(
            model,
            eval_loader,
            criterion,
            device,
            num_classes=num_classes,
            show_progress=use_tqdm,
        )

        per_class_named = {}
        for idx_str, cls_metrics in metrics["per_class"].items():
            cls_name = idx_to_class[int(idx_str)]
            per_class_named[cls_name] = cls_metrics

        final_results["temporal_conditions"][str(fraction)] = {
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "per_class": per_class_named,
        }

        print(
            f"Fraction {fraction}: "
            f"Top-1={metrics['top1']:.2f}% | "
            f"Top-5={metrics['top5']:.2f}% | "
            f"Macro P/R/F1={metrics['macro_precision']:.4f}/"
            f"{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}"
        )

    # write final summary json to outputs
    metrics_path = os.path.join(args.output_dir, f"metrics_summary_{slurm_job_id}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nFinal metrics written to: {metrics_path}")
    print(f"Best model checkpoint: {checkpoint_path}")
