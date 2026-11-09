import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dataset_prep.process_and_patch import load_asc_grid
from model.train import get_adapted_resnet


PATCH_SIZE = 512
STRIDE = 256
LRM_SIGMA = 20.0
MODEL_PATH = project_root / "best_megalith_model.pth"


def non_max_suppression(boxes, iou_threshold=0.3):
    """Keep high-scoring windows and remove heavily overlapping ones."""

    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda item: item[2], reverse=True)
    keep = []

    while boxes:
        current = boxes.pop(0)
        keep.append(current)

        cx, cy, _ = current
        remaining = []

        for box in boxes:
            bx, by, _ = box

            inter_x1 = max(cx, bx)
            inter_y1 = max(cy, by)
            inter_x2 = min(cx + PATCH_SIZE, bx + PATCH_SIZE)
            inter_y2 = min(cy + PATCH_SIZE, by + PATCH_SIZE)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h

            union = (2 * PATCH_SIZE * PATCH_SIZE) - intersection
            iou = intersection / union if union else 0

            if iou < iou_threshold:
                remaining.append(box)

        boxes = remaining

    return keep


def scan_map(
    asc_path: Path,
    threshold: float = 0.65,
    output_dir: Path | None = None,
    batch_size: int = 8,
):
    """
    Run the existing patch classifier over one DEM tile.

    Returns a dictionary containing the annotated image path,
    candidate detections, and basic scan metadata.
    """

    asc_path = Path(asc_path)

    if not asc_path.exists():
        raise FileNotFoundError(f"DEM file not found: {asc_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Scanning {asc_path.name} on {device.type.upper()}")

    # Load the trained model without downloading ImageNet weights.
    model = get_adapted_resnet(pretrained=False).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load DEM and spatial metadata.
    dem, header = load_asc_grid(asc_path)

    if not np.isfinite(dem).any():
        raise ValueError("The DEM contains no valid elevation values.")

    cellsize = header.get("cellsize", 1.0)

    if "xllcorner" in header:
        xll = header["xllcorner"]
    elif "xllcenter" in header:
        xll = header["xllcenter"] - cellsize / 2
    else:
        xll = 0.0

    if "yllcorner" in header:
        yll = header["yllcorner"]
    elif "yllcenter" in header:
        yll = header["yllcenter"] - cellsize / 2
    else:
        yll = 0.0

    # Calculate Local Relief Model.
    nan_mask = np.isnan(dem)
    mean_elevation = np.nanmean(dem)
    dem_filled = np.where(nan_mask, mean_elevation, dem)

    dem_trend = gaussian_filter(dem_filled, sigma=LRM_SIGMA)
    lrm = dem - dem_trend
    lrm[nan_mask] = np.nan

    height, width = lrm.shape

    # Prepare hillshade for visualisation.
    lrm_clean = np.nan_to_num(lrm, nan=0.0)
    lrm_smooth = gaussian_filter(lrm_clean, sigma=1.5)

    light_source = LightSource(azdeg=315.0, altdeg=20.0)
    shaded = light_source.hillshade(
        lrm_smooth * 3.0,
        vert_exag=1,
        dx=1,
        dy=1,
    )

    hillshade = Image.fromarray(
        (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
    ).convert("RGB")

    # Collect valid windows before running batched inference.
    windows = []
    skipped_windows = 0

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            original_patch = lrm[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            if np.isnan(original_patch).sum() > (
                PATCH_SIZE * PATCH_SIZE * 0.3
            ):
                skipped_windows += 1
                continue

            patch = lrm_clean[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Match the per-patch z-score normalisation used in training.
            patch_std = np.std(patch)

            if patch_std > 1e-5:
                patch_norm = (patch - np.mean(patch)) / patch_std
            else:
                patch_norm = patch - np.mean(patch)

            windows.append((x, y, patch_norm.astype(np.float32)))

    # Batched model inference.
    raw_detections = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]

            batch_array = np.stack([item[2] for item in batch])
            tensor = torch.from_numpy(batch_array).unsqueeze(1).to(device)

            logits = model(tensor).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

            for (x, y, _), score in zip(batch, scores):
                if float(score) >= threshold:
                    raw_detections.append((x, y, float(score)))

    detections = non_max_suppression(
        raw_detections,
        iou_threshold=0.3,
    )

    print(
        f"Windows evaluated: {len(windows)} | "
        f"Raw hits: {len(raw_detections)} | "
        f"After NMS: {len(detections)}"
    )

    # Prepare output directory.
    if output_dir is None:
        output_dir = project_root / "static" / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = hillshade.copy()
    draw = ImageDraw.Draw(annotated)

    structured_detections = []

    for index, (x, y, confidence) in enumerate(detections, start=1):
        center_x_pixel = x + PATCH_SIZE / 2
        center_y_pixel = y + PATCH_SIZE / 2

        # Convert pixel position to map coordinates (PL-1992 / EPSG:2180).
        geo_x = xll + center_x_pixel * cellsize
        geo_y = yll + (height - center_y_pixel) * cellsize

        draw.rectangle(
            [
                x,
                y,
                x + PATCH_SIZE,
                y + PATCH_SIZE,
            ],
            outline="red",
            width=5,
        )

        structured_detections.append({
            "id": index,
            "x": int(x),
            "y": int(y),
            "width": PATCH_SIZE,
            "height": PATCH_SIZE,
            "score": round(confidence, 4),
            "score_percent": round(confidence * 100, 1),
            "map_x": round(float(geo_x), 1),
            "map_y": round(float(geo_y), 1),
        })

    output_path = output_dir / f"{asc_path.stem}_annotated.png"
    annotated.save(output_path)

    return {
        "source_file": asc_path.name,
        "image_path": f"results/{output_path.name}",
        "detections": structured_detections,
        "threshold": threshold,
        "device": device.type,
        "windows_evaluated": len(windows),
        "windows_skipped": skipped_windows,
        "raw_hits": len(raw_detections),
        "detections_after_nms": len(detections),
        "dem_width": width,
        "dem_height": height,
        "cellsize": cellsize,
    }


def scan_all_raw_files(threshold: float = 0.65):
    raw_dir = project_root / "data" / "raw"
    raw_files = sorted(raw_dir.glob("*.asc"))

    if not raw_files:
        print("No .asc files found in data/raw/")
        return

    for asc_file in raw_files:
        result = scan_map(asc_file, threshold=threshold)
        print(
            f"{result['source_file']}: "
            f"{result['detections_after_nms']} candidates"
        )


if __name__ == "__main__":
    scan_all_raw_files(threshold=0.65)