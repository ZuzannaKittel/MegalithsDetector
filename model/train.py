from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_SEED = 42
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "best_megalith_model.pth"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset_prep.dataset import MegalithDataset  # noqa: E402


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_adapted_resnet(pretrained: bool = True) -> nn.Module:
    """Load ResNet-18 and adapt its first convolution to one-channel LRM."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def calculate_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, dict]:
    """Run one training or validation epoch and collect binary metrics."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    tp = fp = tn = fn = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        targets = labels.to(device).unsqueeze(1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
        target_int = targets.to(torch.int64)

        tp += int(((predictions == 1) & (target_int == 1)).sum().item())
        fp += int(((predictions == 1) & (target_int == 0)).sum().item())
        tn += int(((predictions == 0) & (target_int == 0)).sum().item())
        fn += int(((predictions == 0) & (target_int == 1)).sum().item())

    if total_samples == 0:
        raise ValueError("A DataLoader produced no samples.")

    return total_loss / total_samples, calculate_metrics(tp, fp, tn, fn)


def train_model() -> None:
    set_seed()
    device = select_device()
    print(f"Training on device: {device.type.upper()}")

    split_dir = PROJECT_DIR / "data" / "dataset" / "splits"
    train_dir = split_dir / "train"
    validation_dir = split_dir / "validation"

    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise FileNotFoundError(
            "Spatial train/validation splits were not found. "
            "Run: python dataset_prep/map_labels.py"
        )

    train_dataset = MegalithDataset(train_dir, is_train=True)
    val_dataset = MegalithDataset(validation_dir, is_train=False)

    train_counts = {0: train_dataset.labels.count(0), 1: train_dataset.labels.count(1)}
    val_counts = {0: val_dataset.labels.count(0), 1: val_dataset.labels.count(1)}

    if not all(train_counts.values()):
        raise ValueError(f"Training split must contain both classes; found {train_counts}.")
    if not all(val_counts.values()):
        raise ValueError(f"Validation split must contain both classes; found {val_counts}.")

    print(
        "Training samples: "
        f"{len(train_dataset)} ({train_counts[1]} positive, {train_counts[0]} negative)"
    )
    print(
        "Validation samples: "
        f"{len(val_dataset)} ({val_counts[1]} positive, {val_counts[0]} negative)"
    )
    print("Validation uses real labelled patches only; no synthetic samples or augmentation.")
    print("Classification metrics use a 0.50 sigmoid-score threshold.")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ImageNet weights are downloaded on first use if they are not cached locally.
    model = get_adapted_resnet(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{EPOCHS} [{elapsed:.1f}s] | "
            f"train loss {train_loss:.4f}, F1 {train_metrics['f1']:.3f}, "
            f"precision {train_metrics['precision']:.3f}, "
            f"recall {train_metrics['recall']:.3f} | "
            f"val loss {val_loss:.4f}, F1 {val_metrics['f1']:.3f}, "
            f"precision {val_metrics['precision']:.3f}, "
            f"recall {val_metrics['recall']:.3f}, "
            f"accuracy {val_metrics['accuracy']:.3f}"
        )
        print(
            "  Validation confusion matrix: "
            f"TP={val_metrics['tp']} FP={val_metrics['fp']} "
            f"TN={val_metrics['tn']} FN={val_metrics['fn']}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Keep a plain state_dict so model/predict.py can load it directly.
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  Saved new best checkpoint: {CHECKPOINT_PATH.name}")

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(
        "Caution: validation contains only 3 positive patches from one held-out DEM. "
        "Treat its metrics as an initial pipeline check, not a reliable estimate "
        "of performance on new geographic areas."
    )


if __name__ == "__main__":
    train_model()
