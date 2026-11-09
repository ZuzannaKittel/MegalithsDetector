
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "dataset"

POSITIVE_PNG_DIR = DATASET_DIR / "positive"
NEGATIVE_NPY_DIR = DATASET_DIR / "negative"
NPY_DIR = DATASET_DIR / "arrays_npy"

SPLIT_DIR = DATASET_DIR / "splits"

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42


# --------------------------------------------------
# Identify the original DEM tile
# --------------------------------------------------

def get_source_group(stem: str) -> str:
    """
    Extract the source DEM filename from a patch filename.

    Expected format:
        DEM_NAME_patch_512_0000_0256

    Patches from the same original DEM must remain
    together in the same split.
    """

    match = re.match(
        r"^(.*)_patch_512_\d+_\d+$",
        stem
    )

    if not match:
        raise ValueError(
            f"Cannot identify source DEM for patch: {stem}"
        )

    return match.group(1)


# --------------------------------------------------
# Collect explicitly labelled samples
# --------------------------------------------------

def collect_samples():
    npy_files = {
        path.stem: path
        for path in NPY_DIR.glob("*.npy")
    }

    positive_stems = {
        path.stem
        for path in POSITIVE_PNG_DIR.glob("*.png")
    }

    # Negative labels are already stored as .npy files.
    negative_stems = {
        path.stem
        for path in NEGATIVE_NPY_DIR.glob("*.npy")
    }

    overlap = positive_stems & negative_stems

    if overlap:
        raise ValueError(
            "Some samples are labelled both positive and negative: "
            f"{list(overlap)[:5]}"
        )

    labelled_stems = positive_stems | negative_stems

    missing_arrays = labelled_stems - set(npy_files)

    if missing_arrays:
        raise FileNotFoundError(
            "Missing matching .npy arrays: "
            f"{list(missing_arrays)[:5]}"
        )

    samples_by_group = defaultdict(list)

    for stem in sorted(positive_stems):
        path = npy_files[stem]
        group = get_source_group(stem)
        samples_by_group[group].append((path, 1))

    for stem in sorted(negative_stems):
        path = npy_files[stem]
        group = get_source_group(stem)
        samples_by_group[group].append((path, 0))

    return samples_by_group


# --------------------------------------------------
# Split by original DEM tile
# --------------------------------------------------

def split_samples(samples_by_group):
    groups = sorted(samples_by_group)

    if len(groups) < 2:
        raise ValueError(
            "At least two independent DEM source groups "
            "are needed for a spatially separated split."
        )

    validation_group_count = max(
        1,
        round(len(groups) * VALIDATION_FRACTION)
    )

    if validation_group_count >= len(groups):
        raise ValueError(
            "Validation split would consume all DEM groups. "
            "Reduce VALIDATION_FRACTION."
        )

    rng = random.Random(RANDOM_SEED)

    for _ in range(1000):
        shuffled_groups = groups.copy()
        rng.shuffle(shuffled_groups)

        validation_groups = set(
            shuffled_groups[:validation_group_count]
        )

        train_samples = [
            sample
            for group in groups
            if group not in validation_groups
            for sample in samples_by_group[group]
        ]

        validation_samples = [
            sample
            for group in validation_groups
            for sample in samples_by_group[group]
        ]

        train_labels = {
            label for _, label in train_samples
        }

        validation_labels = {
            label for _, label in validation_samples
        }

        # Both splits must contain positive and negative examples.
        if train_labels == {0, 1} and validation_labels == {0, 1}:
            return train_samples, validation_samples

    raise ValueError(
        "Could not create a split containing both classes "
        "in training and validation. Check how positives "
        "and negatives are distributed across DEM tiles."
    )


# --------------------------------------------------
# Write split directories
# --------------------------------------------------

def save_split(name, samples):
    for label_name in ("positive", "negative"):
        directory = SPLIT_DIR / name / label_name
        directory.mkdir(parents=True, exist_ok=True)

    counts = {0: 0, 1: 0}

    for source_path, label in samples:
        label_name = "positive" if label == 1 else "negative"

        destination = (
            SPLIT_DIR
            / name
            / label_name
            / source_path.name
        )

        shutil.copy2(source_path, destination)
        counts[label] += 1

    print(
        f"{name}: "
        f"{counts[1]} positive, "
        f"{counts[0]} negative"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    samples_by_group = collect_samples()

    print(f"Found {len(samples_by_group)} DEM source groups.")

    print("\nSamples by DEM:")
    for group in sorted(samples_by_group):
        samples = samples_by_group[group]
        positives = sum(label == 1 for _, label in samples)
        negatives = sum(label == 0 for _, label in samples)

        print(
            f"  {group}: "
            f"{positives} positive, "
            f"{negatives} negative"
        )

    train_samples, validation_samples = split_samples(
        samples_by_group
    )

    # Remove previous real-data splits.
    # Synthetic samples are generated separately afterwards.
    shutil.rmtree(SPLIT_DIR / "train", ignore_errors=True)
    shutil.rmtree(SPLIT_DIR / "validation", ignore_errors=True)

    save_split("train", train_samples)
    save_split("validation", validation_samples)

    print("\nDataset split complete.")


if __name__ == "__main__":
    main()
