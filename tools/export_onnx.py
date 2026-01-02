#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Export trained YOLOX model to ONNX format.

This script exports a trained YOLOX checkpoint to ONNX format,
which can then be optimized with OpenVINO.
"""

import argparse
import os
import torch
from pathlib import Path

# Add YOLOX to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "YOLOX"))

from yolox.exp import get_exp


def export_onnx(exp_file, ckpt_path, output_path, opset=11, simplify=True):
    """
    Export YOLOX model to ONNX format.
    
    Args:
        exp_file: Path to experiment file
        ckpt_path: Path to checkpoint file
        output_path: Path to save ONNX model
        opset: ONNX opset version
        simplify: Whether to simplify ONNX model
    """
    print(f"Loading experiment from: {exp_file}")
    exp = get_exp(exp_file, None)
    
    print(f"Loading checkpoint from: {ckpt_path}")
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    
    print(f"Exporting to ONNX (opset={opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=opset,
    )
    
    print(f"✓ ONNX model saved to: {output_path}")
    
    # Simplify ONNX model
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            model_simp, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(model_simp, output_path)
                print("✓ ONNX model simplified successfully")
            else:
                print("⚠ ONNX simplification check failed, using original model")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")
    
    # Print model info
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        print(f"\nModel info:")
        print(f"  Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"  Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOX model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-f", "--exp-file",
        type=str,
        required=True,
        help="Path to experiment file"
    )
    
    parser.add_argument(
        "-c", "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save ONNX model (default: same as checkpoint with .onnx extension)"
    )
    
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version"
    )
    
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Do not simplify ONNX model"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        args.output = str(Path(args.ckpt).with_suffix(".onnx"))
    
    # Check inputs
    if not os.path.exists(args.exp_file):
        print(f"Error: Experiment file not found: {args.exp_file}")
        return 1
    
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint file not found: {args.ckpt}")
        return 1
    
    # Export
    try:
        export_onnx(
            args.exp_file,
            args.ckpt,
            args.output,
            args.opset,
            not args.no_simplify
        )
        print("\n✅ Export complete!")
        return 0
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
