# Megaliths Detector: AI-Powered Archaeological Feature Detection on LiDAR Data

An end-to-end geospatial machine learning pipeline designed to automatically detect prehistoric Kuyavian long barrows (*grobowce kujawskie*, c. 4000–3000 BCE) in Poland using high-resolution LiDAR Digital Elevation Models (DEM) from the Polish Geoportal.

The project processes raw 2.5D ESRI ASCII elevation grids (`.asc`), isolates microtopographic features via Local Relief Modeling (LRM), injects procedurally generated synthetic barrows to combat class imbalance, trains an adapted 1-channel ResNet-18 model on Apple Silicon (MPS), and performs sliding-window inference with Non-Maximum Suppression (NMS).

---

## Key Features

* **Geospatial Preprocessing & LRM:** Subtraction of Gaussian-filtered trend surfaces ($\sigma = 20.0$) from raw DEMs to decouple shallow microtopographic mounds ($1\,\text{m}$ height) from macro-terrain slopes.
* **Human-in-the-Loop Visualization:** Generates high-contrast 3D hillshade PNG tiles ($512 \times 512\,\text{m}$) using cartographic North-West ($315^\circ$) lighting, $3\times$ vertical Z-exaggeration, and low sun angles ($20^\circ$) for expert domain tagging.
* **Chaotic Synthetic Barrow Generator:** Procedurally generates realistic, weathered $3\text{D}$ trapezoidal barrows (tapering width, broad head, narrow tail, excavation ditches, irregular looting pits, and erosion noise) blended onto background terrain to augment sparse ground-truth data.
* **Adapted Vision Architecture:** Surgically modifies pre-trained ResNet-18 to accept 1-channel spatial elevation matrices natively, preserving learned visual filters by averaging pre-trained RGB weights into the new input conv layer.
* **Apple Silicon Hardware Acceleration:** Native Metal Performance Shaders (`mps`) support for high-speed training and inference on Apple M-series chips.
* **Sliding-Window Inference & NMS:** Scans entire map sheets with a $50\%$ overlapping stride, normalizes inputs via Z-score standardization, applies Intersection over Union (IoU = 0.3) Non-Maximum Suppression to collapse duplicate window hits, and extracts individual cropped PNG tiles for every detection.

---

## Directory Structure

```text
megaliths-detector/
├── .gitignore                      # Prevents heavy .asc, .npy, and .png data from tracking
├── README.md                       # Documentation
├── requirements.txt                # Python environment dependencies
├── best_megalith_model.pth         # Saved weights for top-performing model checkpoint
│
├── data/                           # Local data directory (Ignored by Git)
│   ├── raw/                        # Input ESRI ASCII raster grids (.asc from Geoportal)
│   ├── processed/                  # Full-scale in-memory LRM outputs
│   ├── dataset/
│   │   ├── arrays_npy/             # Clean 512x512 float32 LRM numpy matrices
│   │   ├── images_png/             # 3D hillshade PNG tiles for expert visual inspection
│   │   └── train/
│   │       ├── positive/           # Verified real + synthetic barrow arrays
│   │       └── negative/           # Barren terrain background arrays
│   └── predictions/
│       ├── predicted_*.png         # Full map sheets with red detection bounding boxes
│       └── crops/                  # Individual cropped 512x512 PNG tiles per detection
│
├── dataset_prep/                   # Data processing and augmentation modules
│   ├── process_and_patch.py        # Converts .asc -> LRM, slices 512x512 tiles with 256m stride
│   ├── generate_synthetic.py      # Procedural barrow synthesis and blending script
│   ├── map_labels.py               # Links expert-sorted PNG tags back to .npy arrays
│   └── dataset.py                  # PyTorch Custom Dataset class with geometric augmentations
│
└── model/                          # Machine learning architecture & execution
    ├── train.py                    # PyTorch training loop, MPS device target, loss evaluation
    └── predict.py                  # Full-map sliding window inference engine with NMS
