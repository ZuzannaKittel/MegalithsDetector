# MegalithsDetector

**An end-to-end geospatial machine learning application for screening LiDAR-derived elevation data for prehistoric archaeological structures.**

MegalithsDetector explores the detection of **Kuyavian long barrows** (*grobowce kujawskie*), prehistoric monumental tombs dating to approximately 4000–3000 BCE, in Poland.

The project combines terrain preprocessing, synthetic data generation, deep learning and an interactive Flask interface to identify terrain windows that may contain archaeological features.

<!-- Add an application screenshot here, e.g.:
![MegalithsDetector interface](docs/images/interface.png)
-->
<img width="1758" height="1109" alt="image" src="https://github.com/user-attachments/assets/fec1eba1-7795-447f-bc40-0d3ee453dc3d" />

## Features

- **Terrain preprocessing:** Converts ESRI ASCII elevation grids into Local Relief Models (LRM) to highlight local topographic features.
- **Patch generation:** Extracts overlapping 512 × 512 terrain patches for visual inspection and model training.
- **Synthetic data generation:** Procedurally creates weathered, trapezoidal long-barrow-like structures to supplement a limited set of positive examples.
- **Deep learning:** Uses an ImageNet-pretrained ResNet-18 adapted to accept single-channel elevation data.
- **Spatially separated evaluation:** Splits labelled samples by their original DEM source to reduce spatial leakage between training and validation.
- **Full-map inference:** Applies sliding-window classification and Non-Maximum Suppression (NMS) to identify potential structures.
- **Interactive review:** Provides a Flask web interface for selecting DEM files, adjusting the detection threshold, viewing annotated maps and reviewing individual candidates.

## Pipeline

```text
      ESRI ASCII DEM
             |
             v
    Local Relief Model
             |
             v
   512 x 512 terrain patches
             |
       +-----+------+
       |            |
       v            v
  Expert labels   Synthetic
  (real samples)  augmentation
       |            |
       +-----+------+
             |
             v
    Spatial train/validation
             split
             |
             v
    ResNet-18 training
             |
             v
      Trained model
             |
             v
     Full-map inference
             |
             v
   Non-Maximum Suppression
             |
             v
   Annotated map + candidate
        review interface
```

## Technical approach

**Terrain representation**

The pipeline computes a Local Relief Model by subtracting a Gaussian-filtered elevation trend from the original DEM:

\[
LRM = DEM - GaussianFilter(DEM, \sigma=20)
\]

The resulting terrain representation emphasizes local relief rather than broader elevation differences. Patches are extracted using a 512-pixel window and a 256-pixel stride.

**Synthetic augmentation**

A procedural generator creates synthetic long-barrow-like structures with varying dimensions, orientations, erosion, surface irregularities and excavation features. These are blended into background terrain patches and added to the training set.

**Model architecture**

The classifier is based on ResNet-18 with its first convolution adapted from three input channels to one. Pretrained RGB convolution weights are averaged to initialize the single-channel layer.

Training uses PyTorch, `BCEWithLogitsLoss` and the AdamW optimizer. The implementation supports Apple Silicon MPS, CUDA and CPU execution.

**Inference**

The trained model evaluates overlapping terrain windows across a complete DEM. Predictions above a user-defined sigmoid-score threshold are filtered using Non-Maximum Suppression (IoU = 0.3) to reduce overlapping candidate windows.

## Interactive application

The Flask application provides:

- DEM selection and configurable detection threshold.
- Annotated terrain visualisation.
- Scan statistics, including evaluated windows and detections before and after NMS.
- Candidate-level model scores and map coordinates.
- Accept/reject controls for human review.

## Getting started

### Requirements

- Python 3.10+
- PyTorch and Torchvision
- Flask
- NumPy, SciPy, Matplotlib and Pillow

### Installation

```bash
git clone https://github.com/ZuzannaKittel/MegalithsDetector.git
cd MegalithsDetector

python -m venv megaliths_venv
source megaliths_venv/bin/activate

pip install -r requirements.txt
```

### Prepare the dataset and train

Place source `.asc` DEM files in `data/raw/`.

Run the preprocessing and patch-generation script:

```bash
python dataset_prep/process_and_patch.py
```

After inspecting and manually labelling the generated terrain patches, prepare the spatially separated dataset:

```bash
python dataset_prep/map_labels.py
```

Generate synthetic training samples:

```bash
python dataset_prep/generate_synthetic_barrows.py
```

Train the model:

```bash
python model/train.py
```

Training saves the best checkpoint as `best_megalith_model.pth` in the project root.

### Run the application

With a trained checkpoint and source DEM files available:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Scanned maps are saved locally under `static/results/`.

## Limitations

This is an experimental archaeological screening tool, not an automated archaeological identification system.

- The available labelled dataset is small and covers only a limited number of DEM tiles.
- Validation currently uses a single held-out DEM and only three real positive patches. Its metrics are therefore not a reliable estimate of performance on unseen geographic areas.
- Synthetic samples supplement, but do not replace, real archaeological examples.
- Model scores are not calibrated probabilities.
- Detection boxes represent evaluated terrain windows, not precise archaeological boundaries.

All candidates require expert inspection and independent archaeological verification.

## Technologies

`Python` · `PyTorch` · `Torchvision` · `ResNet-18` · `NumPy` · `SciPy` · `Matplotlib` · `Flask` · `JavaScript` · `HTML` · `CSS`

---

**Author:**

[GitHub repository](https://github.com/ZuzannaKittel/MegalithsDetector)

Copyright (c) 2026 Zuzanna Kittel
