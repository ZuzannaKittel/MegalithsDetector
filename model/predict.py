import sys
from pathlib import Path

# Register project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource

from dataset_prep.process_and_patch import load_asc_grid
from model.train import get_adapted_resnet

def non_max_suppression(boxes, iou_threshold=0.3):
    """Merges overlapping bounding boxes based on Intersection over Union (IoU)."""
    if len(boxes) == 0:
        return []

    # Sort boxes by confidence score (descending)
    boxes = sorted(boxes, key=lambda x: x[2], reverse=True)
    keep = []

    while len(boxes) > 0:
        current = boxes.pop(0)
        keep.append(current)

        cx1, cy1, conf1 = current
        cw, ch = 512, 512

        remaining = []
        for box in boxes:
            bx1, by1, conf2 = box
            
            # Compute Intersection
            inter_x1 = max(cx1, bx1)
            inter_y1 = max(cy1, by1)
            inter_x2 = min(cx1 + cw, bx1 + cw)
            inter_y2 = min(cy1 + ch, by1 + ch)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h

            # Compute IoU
            union = (cw * ch) + (cw * ch) - intersection
            iou = intersection / union if union > 0 else 0

            # Keep box only if it doesn't heavily overlap with the current accepted box
            if iou < iou_threshold:
                remaining.append(box)

        boxes = remaining

    return keep

def scan_map(asc_path: Path, threshold: float = 0.65):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Scanning: {asc_path.name} on {device.type.upper()}")
    print(f"==========================================")

    model_path = project_root / "best_megalith_model.pth"
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}")
        return

    model = get_adapted_resnet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Unpack the spatial metadata header alongside the DEM
    dem, header = load_asc_grid(asc_path)
    
    # Extract spatial parameters safely (handles corner & center header formats)
    cellsize = header.get('cellsize', 1.0)

    if 'xllcorner' in header:
        xll = header['xllcorner']
    elif 'xllcenter' in header:
        xll = header['xllcenter'] - (cellsize / 2.0)
    else:
        xll = 0.0
        print("Warning: Neither xllcorner nor xllcenter found in header.")

    if 'yllcorner' in header:
        yll = header['yllcorner']
    elif 'yllcenter' in header:
        yll = header['yllcenter'] - (cellsize / 2.0)
    else:
        yll = 0.0
        print("Warning: Neither yllcorner nor yllcenter found in header.")

    nan_mask = np.isnan(dem)
    dem_filled = np.where(nan_mask, np.nanmean(dem), dem)
    dem_trend = gaussian_filter(dem_filled, sigma=20.0)
    lrm = dem - dem_trend
    lrm[nan_mask] = np.nan

    h, w = lrm.shape
    patch_size = 512
    stride = 256

    lrm_clean = np.nan_to_num(lrm, nan=0.0)
    lrm_smooth = gaussian_filter(lrm_clean, sigma=1.5)
    
    ls = LightSource(azdeg=315.0, altdeg=20.0)
    shaded = ls.hillshade(lrm_smooth * 3.0, vert_exag=1, dx=1, dy=1)
    full_hillshade = Image.fromarray((np.clip(shaded, 0, 1) * 255).astype(np.uint8)).convert("RGB")

    raw_detections = []

    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = lrm_clean[y:y+patch_size, x:x+patch_size]

                # Skip border regions (>30% NaN)
                if np.isnan(lrm[y:y+patch_size, x:x+patch_size]).sum() > (patch_size * patch_size * 0.3):
                    continue

                # Z-Score Normalization matching training
                patch_std = np.std(patch)
                if patch_std > 1e-5:
                    patch_norm = (patch - np.mean(patch)) / patch_std
                else:
                    patch_norm = patch - np.mean(patch)

                tensor = torch.from_numpy(patch_norm).float().unsqueeze(0).unsqueeze(0).to(device)
                
                output = model(tensor)
                confidence = torch.sigmoid(output).item()

                if confidence >= threshold:
                    raw_detections.append((x, y, confidence))

    filtered_detections = non_max_suppression(raw_detections, iou_threshold=0.3)
    print(f"Raw window hits: {len(raw_detections)} -> Merged detections after NMS: {len(filtered_detections)}")

    draw = ImageDraw.Draw(full_hillshade)
    crop_dir = project_root / "data" / "predictions" / "crops" / asc_path.stem
    crop_dir.mkdir(parents=True, exist_ok=True)

    for i, (x, y, conf) in enumerate(filtered_detections):
        # Georeferencing Calculation (Center of 512x512 bounding box)
        center_x_pixel = x + (patch_size / 2)
        center_y_pixel = y + (patch_size / 2)
        
        # Convert pixel offset to PL-1992 (EPSG:2180) coordinates
        geo_x = xll + (center_x_pixel * cellsize)
        geo_y = yll + ((h - center_y_pixel) * cellsize)
        
        print(f"  [Detection #{i+1}] Conf: {conf*100:.1f}% | PL-1992 (X: {geo_x:.1f}, Y: {geo_y:.1f})")
        
        # Annotate full hillshade map & save individual crop tile
        draw.rectangle([x, y, x + patch_size, y + patch_size], outline="red", width=5)
        crop = full_hillshade.crop((x, y, x + patch_size, y + patch_size))
        crop.save(crop_dir / f"det_{i+1:02d}_conf_{int(conf*100)}pct_PL1992_X{int(geo_x)}_Y{int(geo_y)}.png")

    if len(filtered_detections) > 0:
        out_dir = project_root / "data" / "predictions"
        out_path = out_dir / f"predicted_{asc_path.stem}.png"
        full_hillshade.save(out_path)
        print(f"Saved annotated map: {out_path}")

def scan_all_raw_files(threshold: float = 0.65):
    raw_files = sorted(list((project_root / "data" / "raw").glob("*.asc")))
    if not raw_files:
        print("No .asc files found in data/raw/")
        return
        
    for asc_file in raw_files:
        scan_map(asc_file, threshold=threshold)

if __name__ == "__main__":
    scan_all_raw_files(threshold=0.65)