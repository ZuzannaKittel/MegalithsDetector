import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset_prep.process_and_patch import load_asc_grid  # noqa: E402
from model.train import get_adapted_resnet, select_device  # noqa: E402


PATCH_SIZE = 512
STRIDE = 256
LRM_SIGMA = 20.0
DEFAULT_THRESHOLD = 0.50
MODEL_PATH = PROJECT_DIR / "best_megalith_model.pth"


def non_max_suppression(boxes, iou_threshold: float = 0.3):
    """Suppress highly overlapping equal-size windows, keeping higher scores."""
    if not boxes:
        return []
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")

    remaining = sorted(boxes, key=lambda item: item[2], reverse=True)
    kept = []

    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        cx, cy, _ = current
        filtered = []

        for bx, by, score in remaining:
            inter_w = max(0, min(cx + PATCH_SIZE, bx + PATCH_SIZE) - max(cx, bx))
            inter_h = max(0, min(cy + PATCH_SIZE, by + PATCH_SIZE) - max(cy, by))
            intersection = inter_w * inter_h
            union = (2 * PATCH_SIZE * PATCH_SIZE) - intersection
            iou = intersection / union if union else 0.0

            if iou < iou_threshold:
                filtered.append((bx, by, score))

        remaining = filtered

    return kept


def get_map_metadata(header: dict) -> tuple[float, float, float]:
    """Return cell size and lower-left corner coordinates from an ESRI grid header."""
    if "cellsize" not in header:
        raise ValueError("The DEM header is missing the required 'cellsize' value.")

    cellsize = float(header["cellsize"])
    if not np.isfinite(cellsize) or cellsize <= 0:
        raise ValueError(f"Invalid DEM cellsize: {cellsize}")

    if "xllcorner" in header:
        xll = float(header["xllcorner"])
    elif "xllcenter" in header:
        xll = float(header["xllcenter"]) - cellsize / 2
    else:
        raise ValueError("The DEM header needs xllcorner or xllcenter for map coordinates.")

    if "yllcorner" in header:
        yll = float(header["yllcorner"])
    elif "yllcenter" in header:
        yll = float(header["yllcenter"]) - cellsize / 2
    else:
        raise ValueError("The DEM header needs yllcorner or yllcenter for map coordinates.")

    if not np.isfinite([xll, yll]).all():
        raise ValueError("The DEM origin coordinates are not finite numbers.")

    return cellsize, xll, yll


def load_model(device: torch.device) -> torch.nn.Module:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. Run model/train.py first."
        )

    model = get_adapted_resnet(pretrained=False).to(device)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def scan_map(
    asc_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: Path | None = None,
    batch_size: int = 8,
):
    """Classify overlapping terrain windows and save an annotated hillshade map."""
    asc_path = Path(asc_path)

    if not asc_path.is_file():
        raise FileNotFoundError(f"DEM file not found: {asc_path}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    threshold = float(threshold)
    device = select_device()
    print(f"Scanning {asc_path.name} on {device.type.upper()}")
    model = load_model(device)

    dem, header = load_asc_grid(asc_path)
    if dem.ndim != 2 or dem.size == 0:
        raise ValueError("The DEM did not load as a non-empty 2D grid.")

    valid_mask = np.isfinite(dem)
    if not valid_mask.any():
        raise ValueError("The DEM contains no valid elevation values.")

    cellsize, xll, yll = get_map_metadata(header)
    height, width = dem.shape
    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError(
            f"DEM is {width}x{height} pixels; at least {PATCH_SIZE}x{PATCH_SIZE} "
            "pixels are required for one inference window."
        )

    # Local Relief Model. Invalid cells are temporarily filled for filtering,
    # then restored to NaN so windows with too many invalid cells can be skipped.
    mean_elevation = float(dem[valid_mask].mean())
    dem_filled = np.where(valid_mask, dem, mean_elevation).astype(np.float32)
    dem_trend = gaussian_filter(dem_filled, sigma=LRM_SIGMA)
    lrm = dem_filled - dem_trend
    lrm[~valid_mask] = np.nan

    lrm_clean = np.nan_to_num(lrm, nan=0.0, posinf=0.0, neginf=0.0)
    lrm_smooth = gaussian_filter(lrm_clean, sigma=1.5)

    light_source = LightSource(azdeg=315.0, altdeg=20.0)
    shaded = light_source.hillshade(
        lrm_smooth * 3.0,
        vert_exag=1,
        dx=cellsize,
        dy=cellsize,
    )
    hillshade = Image.fromarray(
        (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
    ).convert("RGB")

    windows = []
    skipped_windows = 0
    max_invalid_cells = int(PATCH_SIZE * PATCH_SIZE * 0.30)

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            original_patch = lrm[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if int((~np.isfinite(original_patch)).sum()) > max_invalid_cells:
                skipped_windows += 1
                continue

            patch = lrm_clean[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_mean = float(patch.mean())
            patch_std = float(patch.std())
            if patch_std > 1e-5:
                patch_norm = (patch - patch_mean) / patch_std
            else:
                patch_norm = patch - patch_mean

            windows.append((x, y, patch_norm.astype(np.float32, copy=False)))

    raw_detections = []
    all_window_scores = []

    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            batch_array = np.stack([item[2] for item in batch])
            tensor = torch.from_numpy(batch_array).unsqueeze(1).to(device)
            logits = model(tensor).reshape(-1)
            scores = torch.sigmoid(logits).cpu().numpy()

            for (x, y, _), score in zip(batch, scores):
                score = float(score)
                all_window_scores.append((x, y, score))

                if score >= threshold:
                    raw_detections.append((x, y, score))

    detections = non_max_suppression(raw_detections, iou_threshold=0.3)
    print(
        f"Windows evaluated: {len(windows)} | skipped: {skipped_windows} | "
        f"raw hits: {len(raw_detections)} | after NMS: {len(detections)}"
    )

    if output_dir is None:
        output_dir = PROJECT_DIR / "static" / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = hillshade.copy()
    draw = ImageDraw.Draw(annotated)
    structured_detections = []

    window_size_m = PATCH_SIZE * cellsize

    for index, (x, y, score) in enumerate(detections, start=1):
        center_x_pixel = x + PATCH_SIZE / 2
        center_y_pixel = y + PATCH_SIZE / 2

        # ESRI ASCII grids start at the top row; map Y increases northward.
        map_x = xll + center_x_pixel * cellsize
        map_y = yll + (height - center_y_pixel) * cellsize

        draw.rectangle(
            [x, y, x + PATCH_SIZE - 1, y + PATCH_SIZE - 1],
            outline=(220, 55, 45),
            width=5,
        )

        structured_detections.append({
            "id": index,
            "x": int(x),
            "y": int(y),
            "width": PATCH_SIZE,
            "height": PATCH_SIZE,
            "width_m": round(float(window_size_m), 2),
            "height_m": round(float(window_size_m), 2),
            "score": round(score, 5),
            "score_percent": round(score * 100, 1),
            "map_x": round(float(map_x), 2),
            "map_y": round(float(map_y), 2),
        })

    output_path = output_dir / f"{asc_path.stem}_annotated.png"
    annotated.save(output_path)

    top_window_scores = sorted(
        all_window_scores,
        key=lambda item: item[2],
        reverse=True,
    )[:10]

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
        "window_width_m": round(float(window_size_m), 2),
        "window_height_m": round(float(window_size_m), 2),
        "top_window_scores": [
            {
                "x": int(x),
                "y": int(y),
                "score": round(score, 5),
            }
            for x, y, score in top_window_scores
        ],
    }


def scan_all_raw_files(threshold: float = DEFAULT_THRESHOLD):
    raw_dir = PROJECT_DIR / "data" / "raw"
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
    scan_all_raw_files()
