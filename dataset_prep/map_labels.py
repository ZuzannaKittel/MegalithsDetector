import shutil
from pathlib import Path

# Paths based on your new structure
project_dir = Path(__file__).resolve().parent.parent
dataset_dir = project_dir / "data" / "dataset"

pos_png_dir = dataset_dir / "positive"
npy_dir = dataset_dir / "arrays_npy"

# Final folders for PyTorch training
train_pos = dataset_dir / "train" / "positive"
train_neg = dataset_dir / "train" / "negative"
train_pos.mkdir(parents=True, exist_ok=True)
train_neg.mkdir(parents=True, exist_ok=True)

# Get the filenames of the 12 barrows your dad found
pos_stems = {p.stem for p in pos_png_dir.glob("*.png")}
print(f"Found {len(pos_stems)} positive tags from your dad.")

pos_count = 0
neg_count = 0

# Sort the raw math arrays based on his tags
for npy_file in npy_dir.glob("*.npy"):
    if npy_file.stem in pos_stems:
        shutil.copy(npy_file, train_pos / npy_file.name)
        pos_count += 1
    else:
        shutil.copy(npy_file, train_neg / npy_file.name)
        neg_count += 1

print(f"Success! Copied {pos_count} real positives and {neg_count} negatives to the train folder.")