import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource

def load_asc_grid(asc_path: Path):
    """Parses an ESRI ASCII grid file directly into a NumPy array."""
    header = {}
    with open(asc_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(6):
            line = f.readline().strip().split()
            if len(line) == 2:
                header[line[0].lower()] = float(line[1].replace(',', '.'))
        data = np.loadtxt(f, dtype=np.float32)

    nodata_val = header.get('nodata_value', -9999.0)
    data[data == nodata_val] = np.nan
    return data

def calculate_patch_hillshade(dem: np.ndarray, azimuth=315.0, altitude=20.0, z_factor=3.0):
    """Computes a 3D hillshade using standard cartographic NW lighting."""
    # Apply Z-exaggeration to steepen the slopes artificially
    dem_exaggerated = dem * z_factor
    
    # Use Matplotlib's built-in LightSource for standard cartographic shadows (NW)
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    shaded = ls.hillshade(dem_exaggerated, vert_exag=1, dx=1, dy=1)
    
    return (shaded * 255).astype(np.uint8)

# Paths
project_dir = Path(__file__).resolve().parent.parent
raw_dir = project_dir / "data" / "raw"
dataset_dir = project_dir / "data" / "dataset"

img_dir = dataset_dir / "images_png"
npy_dir = dataset_dir / "arrays_npy"
img_dir.mkdir(parents=True, exist_ok=True)
npy_dir.mkdir(parents=True, exist_ok=True)

# Processing Parameters
patch_size = 512
stride = 256
sigma_pixels = 20.0  # For LRM trend surface smoothing

# Find all .asc files in the raw folder
asc_files = list(raw_dir.glob("*.asc"))
print(f"Found {len(asc_files)} .asc files to process.")

total_patches = 0

for asc_path in asc_files:
    file_stem = asc_path.stem  # Get filename without extension to use as a prefix
    print(f"\n--- Processing {file_stem} ---")
    
    # 1. Load Data
    dem = load_asc_grid(asc_path)
    
    # 2. Compute LRM in memory
    nan_mask = np.isnan(dem)
    dem_filled = np.where(nan_mask, np.nanmean(dem), dem)
    dem_trend = gaussian_filter(dem_filled, sigma=sigma_pixels)
    lrm = dem - dem_trend
    lrm[nan_mask] = np.nan
    
    # 3. Slice into Patches
    h, w = lrm.shape
    file_patch_count = 0
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = lrm[y:y+patch_size, x:x+patch_size]
            
            # Skip patches that are mostly empty borders
            if np.isnan(patch).sum() > (patch_size * patch_size * 0.1):
                continue
                
            # Prepend the file stem so patches from different files don't overwrite each other
            patch_id = f"{file_stem}_patch_512_{y:04d}_{x:04d}"
            patch_clean = np.nan_to_num(patch, nan=0.0)
            
            # Save pure math array for PyTorch
            np.save(npy_dir / f"{patch_id}.npy", patch_clean)
            
            # Process for Human Vision (Denoise + 315-degree NW Hillshade)
            patch_smooth = gaussian_filter(patch_clean, sigma=1.5)
            hillshade_img = calculate_patch_hillshade(patch_smooth)
            
            Image.fromarray(hillshade_img, mode='L').save(img_dir / f"{patch_id}.png")
            
            file_patch_count += 1
            total_patches += 1
            
    print(f"Extracted {file_patch_count} patches from {file_stem}.")

print(f"\nSuccess! Generated a total of {total_patches} patches across {len(asc_files)} files.")