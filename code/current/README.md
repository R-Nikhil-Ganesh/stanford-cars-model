# Vehicle Make Classification - Codebase

A clean, modular structure categorizing models, pipelines, and evaluation frameworks for vehicle make classification across 32-make and 35-make datasets.

---

## Directory Overview

```text
code/current/
├── 32_makes/                         # 32-Make Model Pipeline & Artifacts
│   ├── train_efficientnet_b0.py      # Training script for 32-make model
│   ├── train_efficientnet_b0.ipynb   # Interactive training notebook
│   ├── output_analysis_32makes.py    # ONNX diagnostic evaluation adapter
│   ├── output_analysis_32makes.ipynb # Full diagnostic evaluation notebook
│   └── output_efficientnet_b0/       # Checkpoints (.onnx/.keras), metrics, plots
│
├── 35_makes/                         # 35-Make Merged Production Model Pipeline & Artifacts
│   ├── train_efficientnet_b0_merged.py    # Merged dataset training pipeline
│   ├── train_efficientnet_b0_merged.ipynb # Interactive training notebook
│   ├── run_train_merged.sh                # Headless training execution runner
│   ├── external_dataset_analysis.ipynb    # Dataset inspection & alignment
│   ├── output_analysis_35makes.py         # ONNX diagnostic evaluation adapter
│   ├── output_analysis_35makes.ipynb      # Full diagnostic evaluation notebook
│   └── output_efficientnet_b0_merged/     # Checkpoints (.onnx/.keras), metrics, plots
├── sample_inference/                 # Self-sufficient standalone inference testing suite
│   ├── models/                       # Copied best weights (32-make & 35-make ONNX)
│   ├── images/                       # Curated batch of 15 sample test images + manifest
│   ├── output/                       # Inference outputs, predictions, and annotated images
│   ├── infer.py                      # Standalone, zero-dependency inference CLI & Python API
│   ├── test_all_samples.sh           # One-click test runner
│   └── README.md                     # Sample inference documentation
│
├── output_analysis.py                # Universal 100% ONNX diagnostic & explainability engine
├── infer_merged.py                   # Standalone production ONNX inference engine
├── export_onnx.py                    # ONNX conversion and validation utility
└── README.md                         # This documentation
```

---

## 1. Production Model Inference (`infer_merged.py`)

A production-grade, ultra-low-latency inference engine powered by **ONNX Runtime** (executing in ~15 ms per image on CPU with zero GPU memory overhead):

```bash
# Single image inference
python infer_merged.py --image path/to/vehicle.jpg

# Batch inference on directory with CSV export
python infer_merged.py --image-dir path/to/images/ --save-csv predictions.csv

# Batch inference on CSV manifest with JSON export
python infer_merged.py --csv /path/to/manifest.csv --image-col image_path --save-json results.json

# Generate visual annotated banner
python infer_merged.py --image path/to/vehicle.jpg --save-annotated annotated_vehicle.jpg
```

**Python API:**
```python
from infer_merged import MergedVehicleClassifier

clf = MergedVehicleClassifier()
res = clf.predict_image("path/to/vehicle.jpg", top_k=5)
print(f"Top-1: {res['top1_make']} ({res['top1_confidence']:.2%})")
```

---

## 2. Universal ONNX Diagnostic & Explainability Engine (`output_analysis.py`)

Evaluates any trained ONNX model across all key dimensions:
- **Test set evaluation**: Overall accuracy, Macro F1, Weighted F1, Cross-Entropy Loss
- **Per-class metrics**: Classification report & color-coded F1 bar ranking
- **Confusion matrix**: Normalized heatmap & top confusion pairs
- **Hardest misclassifications**: High-confidence failure analysis
- **Analytical Grad-CAM**: Sub-millisecond class activation heatmaps ($L^c = \text{ReLU}\left( \sum_k w_k^c A^k \right)$)
- **Latent space visualization**: 2D t-SNE and PCA manifold projections
- **Calibration analysis**: Confidence histograms & accuracy vs. coverage rejection curves

```bash
# Evaluate the 35-make merged model (auto-detected)
python output_analysis.py --run-all

# Evaluate the 32-make model
python output_analysis.py \
  --model-path 32_makes/output_efficientnet_b0/models/efficientnet_b0_best.onnx \
  --output-dir 32_makes/output_efficientnet_b0 \
  --splits-dir /home/researchadmin/Econ/resized_640x640/splits_filtered \
  --run-all
```

---

## 3. Training Pipelines

```bash
# Launch 35-make merged model training
bash 35_makes/run_train_merged.sh

# Or invoke python script directly
python 35_makes/train_efficientnet_b0_merged.py \
  --splits-dir /home/researchadmin/Econ/external_datasets/merged_data \
  --output-dir 35_makes/output_efficientnet_b0_merged \
  --img-size 640 \
  --batch-size 8 \
  --epochs 15
```

---

## 4. Self-Sufficient Sample Inference Testing (`sample_inference/`)

A self-contained testing package containing model weights, 15 sample vehicle images, and an inference runner:

```bash
# Run one-click test on both 35-make and 32-make models
bash sample_inference/test_all_samples.sh

# Or run interactively
cd sample_inference
python infer.py --model 35
python infer.py --model 32
```
