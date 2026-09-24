#!/usr/bin/env python3
"""
EfficientNet-B0 (32 Vehicle Makes): Diagnostic & Explainability Analysis Engine
=============================================================================
This module contains the complete diagnostic evaluation, Grad-CAM interpretability,
latent space visualization (t-SNE/PCA), error inspection, and confidence calibration
logic for the 32-make vehicle classification model.

Can be imported in Jupyter notebooks or run directly from the command line:
    python output_analysis_32makes.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure writable cache directories before importing heavy scientific libraries
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import tensorflow as tf
from tensorflow.keras import models, layers, mixed_precision
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure display fallback
try:
    from IPython.display import display
except ImportError:
    display = print

# ==============================================================================
# 1. Global Constants & Default File Paths
# ==============================================================================
CLASS_NAMES = [
    "Acura", "Alfa Romeo", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler",
    "Cupra", "Dodge", "Ford", "Honda", "Hyundai", "Isuzu", "Jeep", "Kia", "Land Rover",
    "Lexus", "Lincoln", "MINI", "Mazda", "Mitsubishi", "Nissan", "Opel", "Peugeot",
    "Porsche", "Renault", "Skoda", "Suzuki", "Toyota", "Volkswagen", "Volvo"
]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

CURRENT_DIR = Path("/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current")
OUTPUT_DIR  = CURRENT_DIR / "output_efficientnet_b0"
MODELS_DIR  = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR   = OUTPUT_DIR / "plots"
PRED_DIR    = OUTPUT_DIR / "predictions"
SPLITS_DIR  = Path("/home/researchadmin/Econ/resized_640x640/splits_filtered")

DEFAULT_MODEL_PATH = MODELS_DIR / "efficientnet_b0_best.keras"


def setup_environment():
    """Configures hardware settings, plot styling, and mixed precision."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 120

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        mixed_precision.set_global_policy("mixed_float16")
        print(f"[Hardware] GPU active: {gpus[0].name} (mixed_float16 enabled)")
    else:
        print("[Hardware] Running on CPU")

    print(f"[Framework] TensorFlow version: {tf.__version__}")


# ==============================================================================
# 2. Model Loading & Topology Introspection
# ==============================================================================
def load_analysis_model(model_path: Optional[Path] = None) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Loads the trained Keras checkpoint and extracts architecture metadata.
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    assert target_path.exists(), f"Model checkpoint not found at {target_path}"

    print(f"Loading trained weights from: {target_path}...")
    model = models.load_model(str(target_path))

    total_params     = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    last_conv_layer  = model.get_layer("top_activation")
    embedding_layer  = model.get_layer("avg_pool")
    prediction_layer = model.get_layer("predictions")

    info = {
        "model_path": str(target_path),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "last_conv_layer": last_conv_layer.name,
        "last_conv_shape": last_conv_layer.output.shape,
        "embedding_layer": embedding_layer.name,
        "embedding_shape": embedding_layer.output.shape,
        "prediction_layer": prediction_layer.name,
        "prediction_shape": prediction_layer.output.shape,
    }

    print(f"[Architecture] Total Parameters:     {total_params:,}")
    print(f"[Architecture] Trainable Parameters: {trainable_params:,}")
    print(f"[Architecture] Input Shape:          {model.input_shape}")
    print(f"[Architecture] Output Shape:         {model.output_shape}")
    print(f"[Diagnostics] Final Conv Layer:     '{last_conv_layer.name}' (Shape: {last_conv_layer.output.shape})")
    print(f"[Diagnostics] Feature Embed Layer:  '{embedding_layer.name}' (Shape: {embedding_layer.output.shape})")
    print(f"[Diagnostics] Classification Head:  '{prediction_layer.name}' (Shape: {prediction_layer.output.shape})")

    return model, info


# ==============================================================================
# 3. Test Evaluation & Metrics Breakdown
# ==============================================================================
def load_test_evaluation(
    splits_dir: Optional[Path] = None,
    pred_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    model: Optional[tf.keras.Model] = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads the test set manifest, test predictions, and summary test metrics.
    """
    s_dir = Path(splits_dir) if splits_dir else SPLITS_DIR
    p_dir = Path(pred_dir) if pred_dir else PRED_DIR
    r_dir = Path(reports_dir) if reports_dir else REPORTS_DIR

    # 1. Load test manifest
    test_df_full = pd.read_csv(s_dir / "test.csv")
    test_df = test_df_full[test_df_full["make"].isin(CLASS_NAMES)].copy().reset_index(drop=True)
    test_df["target_idx"] = test_df["make"].map(CLASS_TO_IDX)
    print(f"[Test Set] Total images: {len(test_df):,} spanning {test_df['make'].nunique()} makes")

    # 2. Load cached predictions
    y_true_path = p_dir / "y_true.npy"
    y_pred_path = p_dir / "y_pred.npy"

    if y_true_path.exists() and y_pred_path.exists():
        print("[Predictions] Loading cached test predictions...")
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
    else:
        assert model is not None, "Model required to compute predictions if cache is missing."
        print("[Predictions] Running batched inference on test set (batch_size=64)...")
        def parse_img(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [640, 640])
            return img, label

        test_ds = tf.data.Dataset.from_tensor_slices((test_df["image_path"].values, test_df["target_idx"].values.astype("int32")))
        test_ds = test_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)

        raw_preds = model.predict(test_ds, verbose=1)
        y_pred = np.argmax(raw_preds, axis=-1)
        y_true = test_df["target_idx"].values

        p_dir.mkdir(parents=True, exist_ok=True)
        np.save(y_true_path, y_true)
        np.save(y_pred_path, y_pred)

    # 3. Load summary metrics
    metrics_path = r_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            test_metrics = json.load(f)
    else:
        test_metrics = {
            "compile_metrics": float(np.mean(y_pred == y_true[:len(y_pred)])),
            "loss": 0.0
        }

    print("\n" + "="*50)
    print(f"  Test Overall Accuracy:   {test_metrics.get('compile_metrics', 0)*100:.2f}%")
    if "top_5_accuracy" in test_metrics:
        print(f"  Test Top-5 Accuracy:     {test_metrics['top_5_accuracy']*100:.2f}%")
    print(f"  Test Macro F1:           {test_metrics.get('macro_f1', 0)*100:.2f}%")
    print(f"  Test Weighted F1:        {test_metrics.get('weighted_f1', 0)*100:.2f}%")
    print(f"  Test Cross-Entropy Loss: {test_metrics.get('loss', 0):.4f}")
    print("="*50)

    return test_df, y_true, y_pred, test_metrics


def load_classification_report_df(reports_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads test_classification_report.csv and formats summary tables.
    """
    r_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    report_df = pd.read_csv(r_dir / "test_classification_report.csv", index_col=0)
    class_report = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore").copy()
    for col in ["precision", "recall", "f1-score", "support"]:
        class_report[col] = pd.to_numeric(class_report[col])

    class_report_sorted = class_report.sort_values(by="f1-score", ascending=False)
    display_summary = pd.concat([class_report_sorted.head(8), class_report_sorted.tail(5)]).copy()

    formatted_summary = display_summary.copy()
    for col in ["precision", "recall", "f1-score"]:
        formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{float(x)*100:.2f}%")
    formatted_summary["support"] = formatted_summary["support"].apply(lambda x: f"{int(x):,}")

    print("Per-Class Classification Report (Top 8 & Bottom 5 by F1-Score):")
    display(formatted_summary)

    return class_report, formatted_summary


def plot_f1_ranking(class_report: pd.DataFrame, test_metrics: Dict[str, Any], save_path: Optional[Path] = None):
    """Plots horizontal bar chart of F1-scores color-coded by performance thresholds."""
    plt.figure(figsize=(14, 10))
    f1_sorted_asc = class_report.sort_values(by="f1-score", ascending=True)

    bar_colors = []
    for val in f1_sorted_asc["f1-score"]:
        if val >= 0.95:
            bar_colors.append("#10b981")  # Green
        elif val >= 0.90:
            bar_colors.append("#3b82f6")  # Blue
        elif val >= 0.80:
            bar_colors.append("#f59e0b")  # Amber
        else:
            bar_colors.append("#ef4444")  # Red

    bars = plt.barh(f1_sorted_asc.index, f1_sorted_asc["f1-score"], color=bar_colors, height=0.68)

    for bar, (_, row) in zip(bars, f1_sorted_asc.iterrows()):
        f1 = row["f1-score"]
        supp = int(row["support"])
        text_x = bar.get_width() - 0.04 if bar.get_width() > 0.4 else bar.get_width() + 0.01
        text_c = "white" if bar.get_width() > 0.4 else "#1f2937"
        ha = "right" if bar.get_width() > 0.4 else "left"
        plt.text(text_x, bar.get_y() + bar.get_height()/2, f"{f1*100:.1f}% (n={supp:,})", 
                 va="center", ha=ha, color=text_c, fontsize=8.5, fontweight="bold")

    plt.axvline(test_metrics.get("macro_f1", 0), color="#6366f1", linestyle="--", linewidth=1.5, 
                label=f"Macro Avg: {test_metrics.get('macro_f1', 0)*100:.1f}%")
    plt.axvline(test_metrics.get("weighted_f1", 0), color="#059669", linestyle="-.", linewidth=1.5, 
                label=f"Weighted Avg: {test_metrics.get('weighted_f1', 0)*100:.1f}%")

    plt.xlim(0, 1.05)
    plt.xlabel("F1-Score", fontsize=11, fontweight="bold")
    plt.title("Test F1-Score by Make (Sorted) | Green: >=95% | Blue: 90-95% | Amber: 80-90% | Red: <80%", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
        print(f"[Plot Saved] {save_path}")
    plt.show()


# ==============================================================================
# 4. Confusion Matrix & Error Pair Breakdown
# ==============================================================================
def plot_confusion_matrix_and_top_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 15,
    save_path: Optional[Path] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Computes and plots 32x32 normalized confusion matrix and top error pairs."""
    min_len = min(len(y_true), len(y_pred))
    cm_raw = confusion_matrix(y_true[:min_len], y_pred[:min_len], labels=range(NUM_CLASSES))
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_raw / row_sums

    fig, (ax_heat, ax_top) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"width_ratios": [1.3, 1]})

    # Heatmap
    im = ax_heat.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Recall Proportion", rotation=-90, va="bottom", fontsize=10)

    ax_heat.set_xticks(range(NUM_CLASSES))
    ax_heat.set_yticks(range(NUM_CLASSES))
    ax_heat.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=8)
    ax_heat.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax_heat.set_xlabel("Predicted Make", fontsize=11, fontweight="bold")
    ax_heat.set_ylabel("True Make", fontsize=11, fontweight="bold")
    ax_heat.set_title("Normalized Confusion Matrix (32 Makes)", fontsize=13, fontweight="bold", pad=10)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            if val >= 0.05:
                col = "white" if val > 0.5 else "#1f2937"
                ax_heat.text(j, i, f"{val*100:.0f}%", ha="center", va="center", color=col, fontsize=6.5,
                             fontweight="bold" if i == j else "normal")

    # Top Error Pairs
    error_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm_raw[i, j] > 0:
                error_pairs.append({
                    "true": CLASS_NAMES[i],
                    "pred": CLASS_NAMES[j],
                    "count": cm_raw[i, j],
                    "pct_of_true": cm_norm[i, j] * 100,
                    "label": f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}"
                })

    top_errors_df = pd.DataFrame(error_pairs).sort_values(by="count", ascending=True).tail(top_k)

    y_pos = np.arange(len(top_errors_df))
    bars_err = ax_top.barh(y_pos, top_errors_df["count"], color="#ef4444", alpha=0.85, height=0.65)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(top_errors_df["label"], fontsize=9, fontweight="medium")
    ax_top.set_xlabel("Number of Misclassified Images", fontsize=10, fontweight="bold")
    ax_top.set_title(f"Top {top_k} Most Frequent Confusion Pairs (True → Predicted)", fontsize=12, fontweight="bold", pad=10)

    for bar, (_, row) in zip(bars_err, top_errors_df.iterrows()):
        ax_top.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2, 
                    f"{int(row['count']):,} ({row['pct_of_true']:.1f}% of true)", 
                    va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1f2937")

    ax_top.set_xlim(0, top_errors_df["count"].max() * 1.35)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
        print(f"[Plot Saved] {save_path}")
    plt.show()

    return cm_norm, top_errors_df


# ==============================================================================
# 5. Sample Evaluation Cache & Hardest Error Inspection
# ==============================================================================
def get_or_compute_sample_cache(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    cache_path: Optional[Path] = None,
    samples_per_make: int = 10
) -> Dict[str, np.ndarray]:
    """
    Loads or precomputes balanced evaluation cache (embeddings, PCA, t-SNE, predictions).
    """
    c_path = Path(cache_path) if cache_path else PRED_DIR / "sample_evaluation_cache.npz"

    if c_path.exists():
        print(f"[Cache] Loading precomputed sample evaluation cache: {c_path}...")
        sample_cache = dict(np.load(c_path, allow_pickle=True))
    else:
        print(f"[Cache] Computing balanced sample evaluation ({samples_per_make} per make)...")
        sampled_rows = []
        for cls_idx in range(NUM_CLASSES):
            sub = test_df[test_df["target_idx"] == cls_idx]
            n_take = min(len(sub), samples_per_make)
            sampled_rows.append(sub.sample(n=n_take, random_state=42))
        sample_manifest = pd.concat(sampled_rows).reset_index(drop=True)
        sample_paths = sample_manifest["image_path"].values
        sample_true  = sample_manifest["target_idx"].values

        def parse_simg(p, l):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [640, 640])
            return img, l

        s_ds = tf.data.Dataset.from_tensor_slices((sample_paths, sample_true)).map(parse_simg, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
        s_preds = model.predict(s_ds, verbose=1)
        sample_pred  = np.argmax(s_preds, axis=-1)
        sample_confs = np.max(s_preds, axis=-1)

        feat_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
        embeddings = feat_extractor.predict(s_ds, verbose=1)

        pca_2d = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        try:
            tsne_2d = TSNE(n_components=2, perplexity=25, random_state=42, max_iter=1000, init="pca", learning_rate="auto").fit_transform(embeddings)
        except TypeError:
            tsne_2d = TSNE(n_components=2, perplexity=25, random_state=42, n_iter=1000, init="pca", learning_rate="auto").fit_transform(embeddings)

        sample_cache = {
            "image_paths": sample_paths,
            "true_labels": sample_true,
            "pred_labels": sample_pred,
            "pred_confs": sample_confs,
            "embeddings": embeddings,
            "pca_2d": pca_2d,
            "tsne_2d": tsne_2d
        }
        c_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(c_path, **sample_cache)
        print(f"[Cache Saved] {c_path}")

    return sample_cache


def plot_hardest_misclassifications(sample_cache: Dict[str, np.ndarray], n_show: int = 8, save_path: Optional[Path] = None):
    """Displays gallery of the highest-confidence false predictions in the sample set."""
    sample_paths = sample_cache["image_paths"]
    sample_true  = sample_cache["true_labels"]
    sample_pred  = sample_cache["pred_labels"]
    sample_confs = sample_cache["pred_confs"]

    error_mask = (sample_pred != sample_true)
    error_indices = np.where(error_mask)[0]
    sorted_error_indices = error_indices[np.argsort(-sample_confs[error_indices])]

    print(f"[Sample Set] Accuracy: {np.mean(~error_mask)*100:.2f}% | Total Errors: {len(error_indices)} / {len(sample_true)}")
    print(f"[Hardest Errors] Top {min(n_show, len(sorted_error_indices))} highest-confidence false predictions:")

    n_display = min(n_show, len(sorted_error_indices))
    if n_display > 0:
        plt.figure(figsize=(18, 9))
        for idx, e_idx in enumerate(sorted_error_indices[:n_display]):
            img_p = sample_paths[e_idx]
            t_name = IDX_TO_CLASS[sample_true[e_idx]]
            p_name = IDX_TO_CLASS[sample_pred[e_idx]]
            conf_val = sample_confs[e_idx]

            try:
                img = Image.open(img_p).convert("RGB")
            except Exception:
                continue

            ax = plt.subplot(2, 4, idx + 1)
            ax.imshow(img)
            ax.set_title(f"True: {t_name}\nPred: {p_name} ({conf_val*100:.1f}%)", fontsize=11, fontweight="bold", color="#dc2626")
            ax.axis("off")

        plt.suptitle("Highest-Confidence Misclassified Vehicles (True Label vs. Predicted Label)", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), bbox_inches="tight")
            print(f"[Plot Saved] {save_path}")
        plt.show()


# ==============================================================================
# 6. Grad-CAM Engine & Comparative Diagnostics
# ==============================================================================
class GradCAMAnalyzer:
    """Computes and visualizes Gradient-Weighted Class Activation Maps (Grad-CAM)."""

    def __init__(self, model: tf.keras.Model, last_conv_name: str = "top_activation", pred_layer_name: str = "predictions"):
        self.model = model
        self.grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_name).output, model.get_layer(pred_layer_name).output]
        )
        self.jet_colormap = matplotlib.colormaps["jet"] if hasattr(matplotlib, "colormaps") else cm.get_cmap("jet")
        self.resampling_filter = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        print("[Grad-CAM] Analyzer initialized.")

    def compute_gradcam(self, image_path: str, target_class_idx: Optional[int] = None, img_size: int = 640):
        """Computes normalized 2D Grad-CAM heatmap and blended overlay for a given image."""
        raw_img = Image.open(image_path).convert("RGB")
        img_resized = raw_img.resize((img_size, img_size), Image.BILINEAR if not hasattr(Image, "Resampling") else Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_batch)
            if target_class_idx is None:
                target_class_idx = tf.argmax(predictions[0])
            loss = predictions[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.numpy()
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize(raw_img.size, self.resampling_filter)
        heatmap_upsampled = np.array(heatmap_img, dtype=np.float32) / 255.0

        colored_heatmap = self.jet_colormap(heatmap_upsampled)[:, :, :3]
        orig_np = np.array(raw_img, dtype=np.float32) / 255.0
        overlay = 0.55 * orig_np + 0.45 * colored_heatmap
        overlay = np.clip(overlay, 0, 1)

        pred_class_idx = int(tf.argmax(predictions[0]))
        confidence = float(predictions[0][pred_class_idx])

        return raw_img, overlay, pred_class_idx, confidence, heatmap_upsampled

    def plot_highest_confidence_true(
        self,
        sample_cache: Dict[str, np.ndarray],
        target_makes: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ):
        """Visualizes Grad-CAM on highest-confidence TRUE predictions."""
        sample_paths = sample_cache["image_paths"]
        sample_true  = sample_cache["true_labels"]
        sample_pred  = sample_cache["pred_labels"]
        sample_confs = sample_cache["pred_confs"]

        correct_mask = (sample_pred == sample_true)
        makes = target_makes or ["Porsche", "Jeep", "BMW", "Audi", "Land Rover", "Ford"]
        selected_correct = []

        for make in makes:
            m_idx = CLASS_TO_IDX[make]
            c_indices = np.where((sample_true == m_idx) & correct_mask)[0]
            if len(c_indices) > 0:
                best_i = c_indices[np.argmax(sample_confs[c_indices])]
                selected_correct.append((make, sample_paths[best_i], sample_confs[best_i]))

        print(f"[Grad-CAM True] Visualizing {len(selected_correct)} highest-confidence correct predictions:")
        for make, p, conf in selected_correct:
            print(f"  - {make:12s} | Confidence: {conf*100:6.2f}% | Sample: {Path(p).parent.name}/{Path(p).name}")

        plt.figure(figsize=(18, 9))
        for i, (make, path, conf) in enumerate(selected_correct):
            raw_img, overlay, pred_idx, conf_calc, _ = self.compute_gradcam(path)

            ax1 = plt.subplot(2, len(selected_correct), i + 1)
            ax1.imshow(raw_img)
            ax1.set_title(f"{make}\n(Original)", fontsize=10, fontweight="bold")
            ax1.axis("off")

            ax2 = plt.subplot(2, len(selected_correct), len(selected_correct) + i + 1)
            ax2.imshow(overlay)
            ax2.set_title(f"Grad-CAM\n(Conf: {conf_calc*100:.1f}%)", fontsize=10, fontweight="bold", color="#16a34a")
            ax2.axis("off")

        plt.suptitle("Grad-CAM: Highest-Confidence TRUE Predictions (Attending to Iconic Brand Signatures)", 
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), bbox_inches="tight")
            print(f"[Plot Saved] {save_path}")
        plt.show()

    def plot_comparative_highest_confidence_false(
        self,
        sample_cache: Dict[str, np.ndarray],
        n_samples: int = 4,
        save_path: Optional[Path] = None
    ):
        """Visualizes 3-column comparative Grad-CAM on highest-confidence FALSE predictions (hardest errors)."""
        sample_paths = sample_cache["image_paths"]
        sample_true  = sample_cache["true_labels"]
        sample_pred  = sample_cache["pred_labels"]
        sample_confs = sample_cache["pred_confs"]

        false_mask = (sample_pred != sample_true)
        sorted_false_indices = np.where(false_mask)[0]
        sorted_false_indices = sorted_false_indices[np.argsort(-sample_confs[sorted_false_indices])]

        selected_false = []
        seen_makes = set()
        for idx in sorted_false_indices:
            t_m = IDX_TO_CLASS[sample_true[idx]]
            p_m = IDX_TO_CLASS[sample_pred[idx]]
            if t_m not in seen_makes:
                seen_makes.add(t_m)
                selected_false.append((t_m, p_m, sample_paths[idx], sample_confs[idx], sample_true[idx], sample_pred[idx]))
            if len(selected_false) >= n_samples:
                break

        print(f"[Grad-CAM False] Visualizing {len(selected_false)} highest-confidence FALSE predictions (Hardest Errors):")
        for t_m, p_m, p, conf, _, _ in selected_false:
            print(f"  - True: {t_m:12s} -> Pred: {p_m:12s} | False Confidence: {conf*100:.2f}% | Sample: {Path(p).parent.name}/{Path(p).name}")

        fig, axes = plt.subplots(len(selected_false), 3, figsize=(16, 4.8 * len(selected_false)))
        if len(selected_false) == 1:
            axes = np.expand_dims(axes, 0)

        for i, (true_m, pred_m, path, conf_val, t_idx, p_idx) in enumerate(selected_false):
            raw_img, overlay_pred, _, conf_p, _ = self.compute_gradcam(path, target_class_idx=p_idx)
            _, overlay_true, _, conf_t, _ = self.compute_gradcam(path, target_class_idx=t_idx)

            model_name = Path(path).parent.name

            # 1. Original
            axes[i, 0].imshow(raw_img)
            axes[i, 0].set_title(f"Original Vehicle ({model_name})\nTrue Make: {true_m}", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            # 2. Attention for PREDICTED Make (Misleading Features)
            axes[i, 1].imshow(overlay_pred)
            axes[i, 1].set_title(f"Attention for PREDICTED: {pred_m}\n(Misleading Cues, Conf: {conf_p*100:.1f}%)", 
                                 fontsize=11, fontweight="bold", color="#dc2626")
            axes[i, 1].axis("off")

            # 3. Attention for TRUE Make (Evidence for Ground Truth)
            axes[i, 2].imshow(overlay_true)
            axes[i, 2].set_title(f"Attention for TRUE: {true_m}\n(Evidence for Correct Make)", 
                                 fontsize=11, fontweight="bold", color="#2563eb")
            axes[i, 2].axis("off")

        plt.suptitle("Comparative Grad-CAM on Highest-Confidence FALSE Predictions\n(Dissecting Why the Model Was Strongly Fooled)", 
                     fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), bbox_inches="tight")
            print(f"[Plot Saved] {save_path}")
        plt.show()

    def list_models(self, test_df: pd.DataFrame, make: Optional[str] = None) -> List[str]:
        """Convenience method to list available models for a make."""
        return list_available_models(test_df, make)

    def plot_gradcam_by_make_model(
        self,
        test_df: pd.DataFrame,
        make: str,
        model_name: Optional[str] = None,
        n_samples: int = 3,
        target_class: Optional[str] = None,
        mode: str = "auto",
        random_state: Optional[int] = 42,
        save_path: Optional[Path] = None
    ):
        """
        Visualizes Grad-CAM for a specified vehicle make and model.

        Parameters:
        - test_df: DataFrame containing the test split (must contain 'make', 'model', 'image_path').
        - make: Name of the vehicle make (case-insensitive, e.g. 'Porsche', 'Jeep', 'BMW', 'Ford').
        - model_name: Optional model name (case-insensitive or substring, e.g. 'Cayenne', 'Wrangler', '911', 'Civic').
                      If None, samples across all models for the specified make.
        - n_samples: Number of vehicle sample images to display (default: 3).
        - target_class: Optional target make to force Grad-CAM attention for (e.g. 'Chrysler').
        - mode: Layout mode:
                'auto': 2 columns (Original, Grad-CAM) if all correct; 3 columns if any misclassified.
                'comparative': Always 3 columns (Original, Attention for Predicted Make, Attention for True Make).
                'single': Always 2 columns (Original, Winning Grad-CAM).
        - random_state: Random seed for sampling images (default: 42).
        - save_path: Optional path to save the generated figure PNG.
        """
        # 1. Match make (case-insensitive)
        make_matches = test_df[test_df["make"].str.lower() == make.strip().lower()]
        if len(make_matches) == 0:
            partial = [m for m in CLASS_NAMES if make.strip().lower() in m.lower() or m.lower() in make.strip().lower()]
            err_msg = f"Make '{make}' not recognized among the 32 classes.\nAvailable makes: {CLASS_NAMES}"
            if partial:
                err_msg += f"\nDid you mean: {partial}?"
            raise ValueError(err_msg)

        matched_make = make_matches["make"].iloc[0]

        # 2. Match model (case-insensitive / substring)
        if model_name is not None and str(model_name).strip() != "":
            clean_model = str(model_name).strip().lower()
            # Try exact match first
            model_matches = make_matches[make_matches["model"].astype(str).str.lower() == clean_model]
            if len(model_matches) == 0:
                # Try substring match
                model_matches = make_matches[make_matches["model"].astype(str).str.lower().str.contains(clean_model, regex=False)]
            
            if len(model_matches) == 0:
                avail_models = sorted(make_matches["model"].dropna().unique())
                raise ValueError(
                    f"Model '{model_name}' not found for make '{matched_make}'.\n"
                    f"Available models for {matched_make} ({len(avail_models)}): {avail_models}"
                )
            matched_model = model_matches["model"].iloc[0]
            subset = model_matches
        else:
            matched_model = "All Models"
            subset = make_matches

        # 3. Sample images
        total_avail = len(subset)
        n_take = min(n_samples, total_avail)
        if total_avail > n_take and random_state is not None:
            sampled_rows = subset.sample(n=n_take, random_state=random_state).reset_index(drop=True)
        else:
            sampled_rows = subset.head(n_take).reset_index(drop=True)

        print(f"[Grad-CAM Query] Make: '{matched_make}', Model: '{matched_model}'")
        print(f"                 Found {total_avail:,} test images. Visualizing {n_take} sample(s):")

        # 4. Evaluate each sample
        sample_results = []
        any_misclassified = False
        for i, (_, row) in enumerate(sampled_rows.iterrows()):
            img_path = row["image_path"]
            t_make   = row["make"]
            t_model  = row["model"]
            t_idx    = CLASS_TO_IDX[t_make]

            raw_img, overlay_pred, pred_idx, conf_pred, _ = self.compute_gradcam(img_path)
            p_make = IDX_TO_CLASS[pred_idx]
            is_corr = (pred_idx == t_idx)
            if not is_corr:
                any_misclassified = True

            if target_class is not None:
                t_tgt_idx = CLASS_TO_IDX[target_class]
                _, overlay_secondary, _, conf_sec, _ = self.compute_gradcam(img_path, target_class_idx=t_tgt_idx)
                sec_label = target_class
                sec_conf = conf_sec
            elif not is_corr or mode == "comparative":
                _, overlay_secondary, _, conf_sec, _ = self.compute_gradcam(img_path, target_class_idx=t_idx)
                sec_label = f"TRUE: {t_make}"
                sec_conf = conf_sec
            else:
                overlay_secondary = None
                sec_label = None
                sec_conf = None

            sample_results.append({
                "img_path": img_path,
                "raw_img": raw_img,
                "overlay_pred": overlay_pred,
                "overlay_secondary": overlay_secondary,
                "true_make": t_make,
                "true_model": t_model,
                "pred_make": p_make,
                "conf_pred": conf_pred,
                "sec_label": sec_label,
                "sec_conf": sec_conf,
                "is_correct": is_corr
            })
            status_str = "CORRECT" if is_corr else "MISCLASSIFIED"
            print(f"  - Sample {i+1}: True={t_make} {t_model} | Pred={p_make} (Conf: {conf_pred*100:.2f}%) [{status_str}] | File: {Path(img_path).name}")

        # 5. Decide layout: 2 columns vs 3 columns
        use_3_columns = (mode == "comparative") or (mode == "auto" and any_misclassified) or (target_class is not None)

        if use_3_columns:
            fig, axes = plt.subplots(n_take, 3, figsize=(16, 4.8 * n_take), squeeze=False)
            for i, res in enumerate(sample_results):
                # Col 1: Original
                axes[i, 0].imshow(res["raw_img"])
                axes[i, 0].set_title(f"Original Vehicle\n{res['true_make']} {res['true_model']}", fontsize=11, fontweight="bold")
                axes[i, 0].axis("off")

                # Col 2: Attention for PREDICTED
                pred_color = "#16a34a" if res["is_correct"] else "#dc2626"
                axes[i, 1].imshow(res["overlay_pred"])
                axes[i, 1].set_title(f"Attention for PREDICTED: {res['pred_make']}\n(Conf: {res['conf_pred']*100:.1f}%)", 
                                     fontsize=11, fontweight="bold", color=pred_color)
                axes[i, 1].axis("off")

                # Col 3: Attention for Secondary / Ground Truth
                if res["overlay_secondary"] is not None:
                    axes[i, 2].imshow(res["overlay_secondary"])
                    axes[i, 2].set_title(f"Attention for {res['sec_label']}\n(Evidence for Correct Class)", 
                                         fontsize=11, fontweight="bold", color="#2563eb")
                else:
                    axes[i, 2].imshow(res["overlay_pred"])
                    axes[i, 2].set_title(f"Grad-CAM (Winning Class)\n{res['pred_make']}", fontsize=11, fontweight="bold", color="#16a34a")
                axes[i, 2].axis("off")
        else:
            # Clean 2-column layout (Original vs Grad-CAM Overlay)
            fig, axes = plt.subplots(n_take, 2, figsize=(12, 4.8 * n_take), squeeze=False)
            for i, res in enumerate(sample_results):
                axes[i, 0].imshow(res["raw_img"])
                axes[i, 0].set_title(f"Original Vehicle\n{res['true_make']} {res['true_model']}", fontsize=11, fontweight="bold")
                axes[i, 0].axis("off")

                pred_color = "#16a34a" if res["is_correct"] else "#dc2626"
                axes[i, 1].imshow(res["overlay_pred"])
                axes[i, 1].set_title(f"Grad-CAM: {res['pred_make']}\n(Conf: {res['conf_pred']*100:.1f}%)", 
                                     fontsize=11, fontweight="bold", color=pred_color)
                axes[i, 1].axis("off")

        plt.suptitle(f"Grad-CAM Vehicle Explainability: {matched_make} {matched_model}", 
                     fontsize=14, fontweight="bold", y=0.99 if n_take > 1 else 1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), bbox_inches="tight")
            print(f"[Plot Saved] {save_path}")
        plt.show()


def list_available_models(test_df: pd.DataFrame, make: Optional[str] = None) -> List[str]:
    """Lists all vehicle models available in the test dataset for a given make."""
    if make is not None:
        sub = test_df[test_df["make"].str.lower() == make.strip().lower()]
        if len(sub) == 0:
            print(f"Make '{make}' not found. Available makes: {CLASS_NAMES}")
            return []
        matched_make = sub["make"].iloc[0]
        models = sorted(sub["model"].dropna().unique())
        print(f"Available models for {matched_make} ({len(models)} models, {len(sub):,} images):")
        for i in range(0, len(models), 5):
            print("  " + ", ".join(models[i:i+5]))
        return models
    else:
        make_counts = test_df.groupby("make")["model"].nunique()
        print("Unique models count per make in test set:")
        for m, c in make_counts.items():
            print(f"  {m:14s}: {c:3d} models")
        return list(make_counts.index)



# ==============================================================================
# 7. Latent Feature Space Projections (t-SNE & PCA)
# ==============================================================================
def plot_latent_space_tsne_pca(sample_cache: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """Plots side-by-side t-SNE and PCA 2D projections of 1,280-D latent embeddings."""
    sample_true = sample_cache["true_labels"]
    pca_2d      = sample_cache["pca_2d"]
    tsne_2d     = sample_cache["tsne_2d"]

    fig, (ax_tsne, ax_pca) = plt.subplots(1, 2, figsize=(22, 10))
    scatter_cmap = matplotlib.colormaps["tab20"] if hasattr(matplotlib, "colormaps") else cm.get_cmap("tab20")

    # 1. t-SNE
    for idx in range(NUM_CLASSES):
        mask = (sample_true == idx)
        if np.sum(mask) > 0:
            ax_tsne.scatter(
                tsne_2d[mask, 0], 
                tsne_2d[mask, 1], 
                s=45, 
                alpha=0.8, 
                label=CLASS_NAMES[idx],
                color=scatter_cmap(idx % 20),
                edgecolors="none"
            )

    ax_tsne.set_title("t-SNE 2D Projection of Latent Embeddings (1,280-D → 2D)\nBalanced Sample Across 32 Makes", 
                      fontsize=12, fontweight="bold", pad=10)
    ax_tsne.set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
    ax_tsne.set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
    ax_tsne.grid(True, linestyle="--", alpha=0.4)

    # 2. PCA
    for idx in range(NUM_CLASSES):
        mask = (sample_true == idx)
        if np.sum(mask) > 0:
            ax_pca.scatter(
                pca_2d[mask, 0], 
                pca_2d[mask, 1], 
                s=45, 
                alpha=0.8, 
                label=CLASS_NAMES[idx],
                color=scatter_cmap(idx % 20),
                edgecolors="none"
            )

    ax_pca.set_title("PCA 2D Projection of Latent Embeddings (1,280-D → 2D)\nGlobal Variance Decomposition", 
                     fontsize=12, fontweight="bold", pad=10)
    ax_pca.set_xlabel("Principal Component 1", fontsize=10, fontweight="bold")
    ax_pca.set_ylabel("Principal Component 2", fontsize=10, fontweight="bold")
    ax_pca.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax_tsne.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.02), loc="upper center", ncol=8, fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
        print(f"[Plot Saved] {save_path}")
    plt.show()


# ==============================================================================
# 8. Prediction Confidence & Calibration Analysis
# ==============================================================================
def plot_confidence_calibration(sample_cache: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """Plots confidence score distribution histograms and accuracy-coverage rejection curves."""
    sample_true  = sample_cache["true_labels"]
    sample_pred  = sample_cache["pred_labels"]
    sample_confs = sample_cache["pred_confs"]

    is_correct = (sample_pred == sample_true)
    correct_confs = sample_confs[is_correct]
    error_confs   = sample_confs[~is_correct]

    plt.figure(figsize=(15, 5))

    # Subplot 1: Confidence Distribution
    plt.subplot(1, 2, 1)
    plt.hist(correct_confs, bins=25, alpha=0.65, color="#16a34a", density=True, label=f"Correct (Mean: {np.mean(correct_confs)*100:.1f}%)")
    if len(error_confs) > 0:
        plt.hist(error_confs, bins=25, alpha=0.65, color="#dc2626", density=True, label=f"Incorrect (Mean: {np.mean(error_confs)*100:.1f}%)")
    plt.xlabel("Softmax Prediction Confidence", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.title("Confidence Distribution: Correct vs. Incorrect Predictions", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Accuracy vs. Coverage
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(0.3, 0.99, 50)
    retained_accs = []
    coverage_pcts = []

    for t in thresholds:
        mask = (sample_confs >= t)
        if np.sum(mask) > 0:
            retained_accs.append(np.mean(is_correct[mask]) * 100)
            coverage_pcts.append(np.mean(mask) * 100)
        else:
            retained_accs.append(np.nan)
            coverage_pcts.append(0)

    plt.plot(thresholds, retained_accs, "o-", color="#2563eb", linewidth=2, label="Accuracy on Retained Samples (%)")
    plt.plot(thresholds, coverage_pcts, "s--", color="#f59e0b", linewidth=1.5, label="Dataset Coverage (% Retained)")
    plt.xlabel("Confidence Rejection Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    plt.title("Accuracy vs. Coverage with Confidence Thresholding", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
        print(f"[Plot Saved] {save_path}")
    plt.show()

    print(f"[Calibration] Mean confidence on correct predictions:   {np.mean(correct_confs)*100:.1f}%")
    if len(error_confs) > 0:
        print(f"[Calibration] Mean confidence on incorrect predictions: {np.mean(error_confs)*100:.1f}%")
        print(f"[Calibration] Confidence gap (Correct - Error):        {(np.mean(correct_confs) - np.mean(error_confs))*100:.1f}%")


# ==============================================================================
# 9. Complete Standalone Execution Routine
# ==============================================================================
def run_full_analysis():
    """Executes the full diagnostic suite and saves plots to disk."""
    print("="*70)
    print("  EfficientNet-B0 (32 Makes): Running Complete Diagnostic Suite")
    print("="*70)

    setup_environment()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model Loading
    model, model_info = load_analysis_model()

    # 2. Test Evaluation
    test_df, y_true, y_pred, test_metrics = load_test_evaluation(model=model)

    # 3. Classification Report & F1 Chart
    class_report, _ = load_classification_report_df()
    plot_f1_ranking(class_report, test_metrics, save_path=PLOTS_DIR / "f1_score_ranking_32makes.png")

    # 4. Confusion Matrix & Top Errors
    cm_norm, top_errors_df = plot_confusion_matrix_and_top_errors(
        y_true, y_pred, top_k=15, save_path=PLOTS_DIR / "confusion_matrix_32makes.png"
    )

    # 5. Sample Evaluation & Hardest Errors
    sample_cache = get_or_compute_sample_cache(model, test_df)
    plot_hardest_misclassifications(sample_cache, n_show=8, save_path=PLOTS_DIR / "hardest_misclassifications.png")

    # 6. Grad-CAM Analysis
    grad_analyzer = GradCAMAnalyzer(model)
    grad_analyzer.plot_highest_confidence_true(
        sample_cache, save_path=PLOTS_DIR / "gradcam_highest_confidence_true.png"
    )
    grad_analyzer.plot_comparative_highest_confidence_false(
        sample_cache, n_samples=4, save_path=PLOTS_DIR / "gradcam_highest_confidence_false.png"
    )

    # 7. Latent Space Projections
    plot_latent_space_tsne_pca(sample_cache, save_path=PLOTS_DIR / "latent_space_tsne_pca.png")

    # 8. Confidence Calibration
    plot_confidence_calibration(sample_cache, save_path=PLOTS_DIR / "confidence_calibration.png")

    print("\n" + "="*70)
    print("  Complete Analysis Finished Successfully!")
    print(f"  All plots saved to: {PLOTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    run_full_analysis()

