# Megaliths Detector: AI-Powered Archaeological Feature Detection on LiDAR Data

An end-to-end geospatial machine learning pipeline designed to automatically detect prehistoric Kuyavian long barrows (*grobowce kujawskie*, c. 4000–3000 BCE) in Poland using high-resolution LiDAR Digital Elevation Models (DEM) from the Polish Geoportal.

The project processes raw 2.5D ESRI ASCII elevation grids (`.asc`), isolates microtopographic features via Local Relief Modeling (LRM), injects procedurally generated synthetic barrows to combat class imbalance, trains an adapted 1-channel ResNet-18 model on Apple Silicon (MPS), and performs sliding-window inference with Non-Maximum Suppression (NMS).

---

## Key Features

* **Geospatial Preprocessing & LRM:** Subtraction of Gaussian-filtered trend surfaces ($\sigma = 20.0$) from raw DEMs to decouple shallow microtopographic mounds (1 m height) from macro-terrain slopes.
* **Human-in-the-Loop Visualization:** Generates high-contrast 3D hillshade PNG tiles ($512 \times 512$ m) using cartographic North-West ($315^\circ$) lighting, $3\times$ vertical Z-exaggeration, and low sun angles ($20^\circ$) for expert domain tagging.
* **Chaotic Synthetic Barrow Generator:** Procedurally generates realistic, weathered 3D trapezoidal barrows (tapering width, broad head, narrow tail, excavation ditches, irregular looting pits, and erosion noise) blended onto background terrain to augment sparse ground-truth data.
* **Adapted Vision Architecture:** Surgically modifies pre-trained ResNet-18 to accept 1-channel spatial elevation matrices natively, preserving learned visual filters by averaging pre-trained RGB weights into the new input conv layer.
* **Apple Silicon Hardware Acceleration:** Native Metal Performance Shaders (`mps`) support for high-speed training and inference on Apple M-series chips.
* **Sliding-Window Inference & NMS:** Scans entire map sheets with a 50% overlapping stride, normalizes inputs via Z-score standardization, applies Intersection over Union (IoU = 0.3) Non-Maximum Suppression to collapse duplicate window hits, and extracts individual cropped PNG tiles for every detection.

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
│   ├── generate_synthetic.py       # Procedural barrow synthesis and blending script
│   ├── map_labels.py               # Links expert-sorted PNG tags back to .npy arrays
│   └── dataset.py                  # PyTorch Custom Dataset class with geometric augmentations
│
└── model/                          # Machine learning architecture & execution
    ├── train.py                    # PyTorch training loop, MPS device target, loss evaluation
    └── predict.py                  # Full-map sliding window inference engine with NMS
```

---

## Workflow & Pipeline

```text
[Raw .asc DEM] ──> [LRM Extraction] ──> [Sliding Window 512x512] ──> [3D Hillshade PNGs]
                                                                            │
                                                                  (Expert Tagging by Archeologist)
                                                                            │
                                                                            ▼
[Trained ResNet-18] <── [Z-Score Norm + Augmentation] <── [Real Tags + Synthetic Generation]
        │
        ▼
[Full Map Scan] ──> [Sliding Window Inference] ──> [NMS Box Merging] ──> [Annotated Maps & Crop Tiles]
```

### 1. Preprocessing & Tile Generation
Raw ESRI ASCII raster files placed in `data/raw/` are processed into Local Relief Models to strip away regional elevation trends:
$$\text{LRM} = \text{DEM} - \text{GaussianFilter}(\text{DEM}, \sigma = 20.0)$$
The resulting map is sliced into overlapping $512 \times 512$ m patches using a 256-meter stride (50% overlap).

### 2. Expert Human Labeling
Cartographically shaded PNGs are generated with $315^\circ$ light angles to prevent relief inversion. The expert sorts these PNGs into `positive` and `negative` folders.

### 3. Synthetic Augmentation & Data Normalization
`generate_synthetic.py` blends procedurally modeled trapezoidal long barrows into negative tiles, introducing random rotational angles, spatial offsets, structural weathering, and looting craters. Inputs are normalized on the fly using Z-score standardization:
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

### 4. Model Adaptation & Training
Training uses PyTorch on `torch.device("mps")`:
* **Backbone:** ResNet-18 (Conv1 adapted from 3 RGB channels to 1 channel).
* **Augmentations:** Random horizontal/vertical flips and $90^\circ, 180^\circ, 270^\circ$ spatial rotations.
* **Loss Function:** `BCEWithLogitsLoss()`
* **Optimizer:** `AdamW(lr=1e-4, weight_decay=1e-2)`

### 5. Inference & NMS Box Merging
`predict.py` loads `best_megalith_model.pth`, calculates LRM across target map sheets, and slides a $512 \times 512$ window across the terrain. Overlapping window triggers are merged using Non-Maximum Suppression (IoU = 0.3) based on confidence scores.

---

## Quick Start Guide

### Prerequisites
* Python 3.10+
* macOS with Apple Silicon (M-series GPU) or CUDA/CPU system

### Setup & Run
```bash
# 1. Clone repository
git clone [https://github.com/ZuzannaKittel/MegalithsDetector.git](https://github.com/ZuzannaKittel/MegalithsDetector.git)
cd MegalithsDetector

# 2. Activate virtual environment & install dependencies
source megaliths_venv/bin/activate
pip install -r requirements.txt

# 3. Process raw .asc files into patches
python dataset_prep/process_and_patch.py

# 4. Generate synthetic barrows to expand training set
python dataset_prep/generate_synthetic.py

# 5. Train the neural network on Apple Silicon
python model/train.py

# 6. Run full-map inference & generate detection crops
python model/predict.py
```

---

## Predictions & Review Outputs

After running `predict.py`, inspect `data/predictions/`:
1. **`predicted_[map_name].png`**: Complete visual map sheet featuring solid red $512 \times 512$ m bounding boxes around all merged detections.
2. **`data/predictions/crops/[map_name]/`**: Isolated, individual PNG crops named with confidence percentages (e.g., `detection_01_conf_97pct_x0_y1024.png`) ready for immediate expert verification.
