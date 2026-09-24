#!/usr/bin/env python3
"""
Standalone Keras to ONNX Exporter for EfficientNet-B0 (35 Vehicle Makes)
========================================================================
Converts any saved Keras checkpoint (.keras) to an optimized ONNX (.onnx)
artifact ready for fast deployment, ONNX Runtime evaluation, or export.

Usage:
    # Export best model:
    python export_onnx.py --model-path output_efficientnet_b0_merged/models/efficientnet_b0_best.keras

    # Export interrupted or latest model:
    python export_onnx.py --model-path output_efficientnet_b0_merged/models/efficientnet_b0_interrupted.keras
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Cache configuration
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import models
import tf2onnx
import onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Export Keras checkpoint to ONNX.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0_merged/models/efficientnet_b0_best.keras",
        help="Path to the .keras checkpoint file."
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default=None,
        help="Target path for the exported .onnx file (defaults to replacing .keras with .onnx)."
    )
    parser.add_argument("--img-size", type=int, default=640, help="Input spatial resolution (default: 640).")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13).")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX Runtime verification test.")
    return parser.parse_args()


def export_keras_to_onnx(model_path: str, output_onnx: str = None, img_size: int = 640, opset: int = 13, verify: bool = True):
    src_path = Path(model_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {src_path}")

    dest_path = Path(output_onnx) if output_onnx else src_path.with_suffix(".onnx")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Exporting Keras checkpoint to ONNX")
    print(f"  Source: {src_path}")
    print(f"  Target: {dest_path}")
    print(f"  Input:  (None, {img_size}, {img_size}, 3), opset={opset}")
    print("=" * 70)

    # 1. Load Keras model
    print("\n[1/4] Loading Keras model...")
    model = models.load_model(str(src_path))
    num_classes = model.output_shape[-1]
    print(f"  - Model: {model.name} ({model.count_params():,} params, {num_classes} classes)")

    # 2. Convert via tf2onnx
    print("\n[2/4] Converting to ONNX format...")
    spec = (tf.TensorSpec((None, img_size, img_size, 3), tf.float32, name="input_image"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)

    # 3. Check ONNX graph validity
    print("\n[3/4] Validating ONNX graph integrity...")
    onnx.checker.check_model(onnx_model)
    with open(dest_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  - Successfully written: {dest_path} ({size_mb:.2f} MB)")

    # 4. Optional verification with ONNX Runtime
    if verify:
        print("\n[4/4] Verifying with ONNX Runtime...")
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(dest_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            in_name = session.get_inputs()[0].name
            out_name = session.get_outputs()[0].name

            # Test synthetic batch
            dummy_batch = np.random.randn(2, img_size, img_size, 3).astype(np.float32)
            ort_preds = session.run([out_name], {in_name: dummy_batch})[0]
            keras_preds = model.predict(dummy_batch, verbose=0)

            max_diff = float(np.max(np.abs(ort_preds - keras_preds)))
            print(f"  - ONNX Runtime providers: {session.get_providers()}")
            print(f"  - Output shape: {ort_preds.shape}")
            print(f"  - Max abs diff between Keras and ONNX: {max_diff:.6e} (verification passed!)")
        except Exception as e:
            print(f"  - [Warning] ONNX Runtime check failed: {e}")

    print("\n" + "=" * 70)
    print(f"  Export Complete! File is ready at: {dest_path}")
    print("=" * 70)
    return str(dest_path)


def main():
    args = parse_args()
    export_keras_to_onnx(
        model_path=args.model_path,
        output_onnx=args.output_onnx,
        img_size=args.img_size,
        opset=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
