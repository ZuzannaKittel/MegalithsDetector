import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource

def load_asc_grid(asc_path: Path):
    """Parses an ESRI ASCII grid file robustly, handling inhomogeneous lines and metadata."""
    header = {}
    data_list = []
    
    with open(asc_path, 'r', encoding='utf-8', errors='ignore') as f:
        # 1. Parse the 6-line header
        for _ in range(6):
            line = f.readline().strip().split()
            if len(line) == 2:
                header[line[0].lower()] = float(line[1].replace(',', '.'))
                
        # 2. Parse the grid data robustly (flattened reading)
        for line in f:
            # Split by whitespace and convert only valid numeric strings
            row_vals = [float(val) for val in line.strip().split() if val]
            data_list.extend(row_vals)
            
    # 3. Reshape the flat list back into a 2D matrix using header dimensions
    ncols = int(header.get('ncols', 0))
    nrows = int(header.get('nrows', 0))
    
    if ncols > 0 and nrows > 0 and len(data_list) == (ncols * nrows):
        data = np.array(data_list, dtype=np.float32).reshape((nrows, ncols))
    else:
        # Fallback if the dimensions don't match the header (rare)
        print(f"Warning: Data length ({len(data_list)}) doesn't match header {nrows}x{ncols}. Attempting forced reshape...")
        data = np.array(data_list, dtype=np.float32)
        # Attempt to reshape based on ncols
        data = data[:(len(data) // ncols) * ncols].reshape(-1, ncols)

    # 4. Handle NoData values
    nodata_val = header.get('nodata_value', -9999.0)
    data[data == nodata_val] = np.nan
    
    return data, header

def calculate_patch_hillshade(dem: np.ndarray, azimuth=315.0, altitude=20.0, z_factor=3.0):
    dem_exaggerated = dem * z_factor
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    shaded = ls.hillshade(dem_exaggerated, vert_exag=1, dx=1, dy=1)
    return (shaded * 255).astype(np.uint8)


# --- This block PREVENTS the script from running automatically when imported ---
if __name__ == "__main__":
    # Paths
    project_dir = Path(__file__).resolve().parent.parent
    raw_dir = project_dir / "data" / "raw"
    dataset_dir = project_dir / "data" / "dataset"

    img_dir = dataset_dir / "images_png"
    npy_dir = dataset_dir / "arrays_npy"
    missing_img_dir = dataset_dir / "images_png_missing" # Dedicated folder for the new PNGs

    img_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    missing_img_dir.mkdir(parents=True, exist_ok=True)

    # Find all previously generated patches (even if moved to 'positive' or 'negative' folders)
    existing_patches = {p.stem for p in dataset_dir.rglob("*.npy")}
    print(f"Found {len(existing_patches)} existing patches. Skipping these...")

    # Processing Parameters
    patch_size = 512
    stride = 256
    sigma_pixels = 20.0  

    asc_files = list(raw_dir.glob("*.asc"))
    print(f"Found {len(asc_files)} .asc files to process.")

    total_new_patches = 0

    for asc_path in asc_files:
        file_stem = asc_path.stem  
        print(f"\n--- Processing {file_stem} ---")
        
        # Unpack both variables here so it matches the updated function
        dem, _ = load_asc_grid(asc_path)
        
        nan_mask = np.isnan(dem)
        dem_filled = np.where(nan_mask, np.nanmean(dem), dem)
        dem_trend = gaussian_filter(dem_filled, sigma=sigma_pixels)
        lrm = dem - dem_trend
        lrm[nan_mask] = np.nan
        
        h, w = lrm.shape
        file_new_patch_count = 0
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_id = f"{file_stem}_patch_512_{y:04d}_{x:04d}"
                
                # 1. SKIP ALREADY GENERATED PATCHES
                if patch_id in existing_patches:
                    continue
                    
                patch = lrm[y:y+patch_size, x:x+patch_size]
                
                # 2. RELAXED EDGE FILTER (50% instead of 10%)
                if np.isnan(patch).sum() > (patch_size * patch_size * 0.5):
                    continue
                    
                patch_clean = np.nan_to_num(patch, nan=0.0)
                
                # Save math array to the main array folder
                np.save(npy_dir / f"{patch_id}.npy", patch_clean)
                
                # Save the PNG specifically to the 'missing' folder so it's easy to send
                patch_smooth = gaussian_filter(patch_clean, sigma=1.5)
                hillshade_img = calculate_patch_hillshade(patch_smooth)
                Image.fromarray(hillshade_img, mode='L').save(missing_img_dir / f"{patch_id}.png")
                
                file_new_patch_count += 1
                total_new_patches += 1
                
        print(f"Extracted {file_new_patch_count} NEW patches from {file_stem}.")

    print(f"\nSuccess! Generated {total_new_patches} previously missing patches.")