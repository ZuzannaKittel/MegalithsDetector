from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


PATCH_SIZE = 512
REAL_POSITIVE_OVERSAMPLE = 5


class MegalithDataset(Dataset):
    """Load labelled LRM arrays, oversampling real positives only in training."""

    def __init__(self, data_dir: Path, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        positive_dir = self.data_dir / "positive"
        negative_dir = self.data_dir / "negative"

        if not positive_dir.is_dir() or not negative_dir.is_dir():
            raise FileNotFoundError(
                f"Expected positive/ and negative/ directories inside {self.data_dir}"
            )

        all_positive = sorted(positive_dir.glob("*.npy"))
        self.neg_files = sorted(negative_dir.glob("*.npy"))

        real_positive = [
            path for path in all_positive
            if "synth" not in path.name.lower()
        ]
        synthetic_positive = [
            path for path in all_positive
            if "synth" in path.name.lower()
        ]

        if self.is_train:
            self.pos_files = (real_positive * REAL_POSITIVE_OVERSAMPLE) + synthetic_positive
        else:
            # Never use synthetic examples or augmentation for validation.
            self.pos_files = real_positive

        self.all_files = self.pos_files + self.neg_files
        self.labels = ([1] * len(self.pos_files)) + ([0] * len(self.neg_files))

        print(
            f"Dataset loaded from {self.data_dir}: "
            f"{len(self.pos_files)} positives "
            f"({len(real_positive)} real, {len(synthetic_positive)} synthetic; "
            f"real oversampling x{REAL_POSITIVE_OVERSAMPLE if self.is_train else 1}), "
            f"{len(self.neg_files)} negatives."
        )

        if not self.all_files:
            raise ValueError(f"No .npy samples found in {self.data_dir}")

        if not self.is_train and synthetic_positive:
            print(
                f"Ignored {len(synthetic_positive)} synthetic positives in validation."
            )

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int):
        npy_path = self.all_files[index]
        lrm_array = np.load(npy_path).astype(np.float32, copy=False)

        if lrm_array.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Unexpected patch shape {lrm_array.shape} in {npy_path}; "
                f"expected {(PATCH_SIZE, PATCH_SIZE)}."
            )

        if not np.isfinite(lrm_array).all():
            raise ValueError(f"Non-finite values found in dataset patch: {npy_path}")

        mean = float(lrm_array.mean())
        std = float(lrm_array.std())
        if std > 1e-5:
            lrm_array = (lrm_array - mean) / std
        else:
            lrm_array = lrm_array - mean

        tensor = torch.from_numpy(np.ascontiguousarray(lrm_array)).unsqueeze(0)

        if self.is_train:
            # Random right-angle rotations and flips preserve the square patch shape.
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(2,))
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(1,))
            tensor = torch.rot90(tensor, k=random.randrange(4), dims=(1, 2))

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return tensor, label
