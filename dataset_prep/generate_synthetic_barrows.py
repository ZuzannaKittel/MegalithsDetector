import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

def generate_kuyavian_barrow_mask(
    patch_size: int = 512,
    length: float = 80.0,
    head_width: float = 10.0,
    tail_width: float = 4.0,
    peak_height: float = 1.2,
    ditch_depth: float = 0.35,
    rotation_deg: float = 45.0,
    blur_sigma: float = 2.5,
    offset_y: int = 0,
    offset_x: int = 0
) -> np.ndarray:
    """Procedurally generates a chaotic, weathered Kuyavian long barrow."""
    pad = int(max(length, head_width) * 1.5)
    canvas_w = int(length) + 2 * pad
    canvas_h = int(head_width * 3) + 2 * pad
    
    y_coords, x_coords = np.ogrid[:canvas_h, :canvas_w]
    x_rel = x_coords - pad
    y_rel = y_coords - (canvas_h // 2)
    
    heightmap = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    length_mask = (x_rel >= 0) & (x_rel <= length)
    progress = np.clip(x_rel / length, 0.0, 1.0)
    current_width = head_width + progress * (tail_width - head_width)
    half_width = current_width / 2.0
    
    height_taper = peak_height * (1.0 - 0.3 * progress)
    within_mound = length_mask & (np.abs(y_rel) <= half_width)
    
    y_norm = y_rel / np.maximum(half_width, 1e-5)
    mound_profile = height_taper * (1.0 - y_norm**2)
    heightmap[within_mound] = mound_profile[within_mound]
    
    ditch_width = 4.0
    ditch_inner = half_width
    ditch_outer = half_width + ditch_width
    within_ditch = length_mask & (np.abs(y_rel) > ditch_inner) & (np.abs(y_rel) <= ditch_outer)
    
    ditch_y_norm = (np.abs(y_rel) - ditch_inner) / ditch_width
    ditch_profile = -ditch_depth * np.sin(np.pi * ditch_y_norm)
    heightmap[within_ditch] = ditch_profile[within_ditch]
    
    # --- CHAOS INJECTION ---
    # 1. Structural Imperfections (Bumps and uneven settling)
    bumpy_noise = gaussian_filter(np.random.normal(0, 0.25, heightmap.shape), sigma=3.0)
    heightmap[within_mound] += bumpy_noise[within_mound]
    
    # 2. Anthropogenic Damage (Looting pits / Plough damage)
    craters = np.random.rand(*heightmap.shape) > 0.998  # 0.2% chance of a deep pit
    craters_smoothed = gaussian_filter(craters.astype(float), sigma=2.0) * -1.0
    heightmap += craters_smoothed
    
    # Final 5,000-year weathering erosion
    heightmap = gaussian_filter(heightmap, sigma=blur_sigma)
    
    # Rotate
    rotated_heightmap = rotate(heightmap, rotation_deg, reshape=False, order=1, mode='constant', cval=0.0)
    
    # Place on final canvas WITH RANDOM JITTER
    patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    rh, rw = rotated_heightmap.shape
    
    start_y = (patch_size - rh) // 2 + offset_y
    start_x = (patch_size - rw) // 2 + offset_x
    
    y1, y2 = max(0, start_y), min(patch_size, start_y + rh)
    x1, x2 = max(0, start_x), min(patch_size, start_x + rw)
    ry1, ry2 = max(0, -start_y), min(rh, patch_size - start_y)
    rx1, rx2 = max(0, -start_x), min(rw, patch_size - start_x)
    
    if y1 < y2 and x1 < x2:
        patch[y1:y2, x1:x2] = rotated_heightmap[ry1:ry2, rx1:rx2]
        
    return patch

def inject_synthetic_barrows(
    npy_dir: Path, 
    output_npy_dir: Path, 
    output_png_dir: Path, 
    num_samples: int = 50
):
    output_npy_dir.mkdir(parents=True, exist_ok=True)
    output_png_dir.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(npy_dir.glob("*.npy"))
    if not npy_files:
        print(f"Error: No .npy files found in {npy_dir}")
        return

    print(f"Generating {num_samples} chaotic synthetic tiles...")
    ls = LightSource(azdeg=315.0, altdeg=20.0)
    
    for i in range(num_samples):
        bg_path = np.random.choice(npy_files)
        bg_patch = np.load(bg_path)
        
        # Extreme randomization of parameters
        length = np.random.uniform(40.0, 140.0)
        head_width = np.random.uniform(7.0, 15.0)
        tail_width = np.random.uniform(2.0, 7.0)
        peak_height = np.random.uniform(0.6, 1.2) # Slightly lower, more eroded
        rotation = np.random.uniform(0.0, 360.0)
        
        # Random placement anywhere within the 512x512 tile
        max_offset = 200
        offset_y = np.random.randint(-max_offset, max_offset)
        offset_x = np.random.randint(-max_offset, max_offset)
        
        synth_barrow = generate_kuyavian_barrow_mask(
            patch_size=512,
            length=length,
            head_width=head_width,
            tail_width=tail_width,
            peak_height=peak_height,
            rotation_deg=rotation,
            blur_sigma=np.random.uniform(2.5, 4.0), # Heavier blurring
            offset_y=offset_y,
            offset_x=offset_x
        )
        
        combined_lrm = bg_patch + synth_barrow
        synth_id = f"synth_patch_512_{i:04d}"
        
        np.save(output_npy_dir / f"{synth_id}.npy", combined_lrm)
        
        patch_smooth = gaussian_filter(combined_lrm, sigma=1.5)
        shaded = ls.hillshade(patch_smooth * 3.0, vert_exag=1, dx=1, dy=1)
        shaded_img = (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
        
        plt.imsave(output_png_dir / f"{synth_id}.png", shaded_img, cmap='gray')

    print(f"Success! Saved {num_samples} samples.")

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent

    split_dir = project_dir / "data" / "dataset" / "splits"

    train_neg_dir = split_dir / "train" / "negative"
    train_pos_dir = split_dir / "train" / "positive"

    synth_png_dir = (
        split_dir / "synthetic_verification"
    )

    inject_synthetic_barrows(
        train_neg_dir,
        train_pos_dir,
        synth_png_dir,
        num_samples=150
    )