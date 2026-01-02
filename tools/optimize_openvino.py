#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Optimize ONNX model with OpenVINO.

This script converts an ONNX model to OpenVINO IR format
for optimized inference on Intel hardware.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path


def optimize_openvino(onnx_path, output_dir, fp16=True, benchmark=True):
    """
    Convert ONNX model to OpenVINO IR format.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save OpenVINO IR files
        fp16: Whether to use FP16 precision
        benchmark: Whether to run benchmark
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = Path(onnx_path).stem
    output_model = os.path.join(output_dir, model_name)
    
    print(f"Converting ONNX to OpenVINO IR...")
    print(f"  Input: {onnx_path}")
    print(f"  Output: {output_model}.xml")
    print(f"  Precision: {'FP16' if fp16 else 'FP32'}")
    
    # Build conversion command
    cmd = [
        "mo",
        "--input_model", onnx_path,
        "--output_dir", output_dir,
        "--model_name", model_name,
    ]
    
    if fp16:
        cmd.extend(["--data_type", "FP16"])
    
    # Run conversion
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Conversion successful!")
        
        # Print output files
        xml_file = f"{output_model}.xml"
        bin_file = f"{output_model}.bin"
        
        if os.path.exists(xml_file) and os.path.exists(bin_file):
            xml_size = os.path.getsize(xml_file) / 1024
            bin_size = os.path.getsize(bin_file) / (1024 * 1024)
            print(f"\nGenerated files:")
            print(f"  {xml_file} ({xml_size:.1f} KB)")
            print(f"  {bin_file} ({bin_size:.1f} MB)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed!")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ OpenVINO Model Optimizer not found!")
        print("\nPlease install OpenVINO:")
        print("  pip install openvino-dev")
        return False
    
    # Run benchmark
    if benchmark:
        print("\nRunning benchmark...")
        xml_file = f"{output_model}.xml"
        
        try:
            cmd = [
                "benchmark_app",
                "-m", xml_file,
                "-d", "CPU",
                "-niter", "100",
                "-nireq", "1",
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Parse benchmark results
            output = result.stdout
            for line in output.split('\n'):
                if 'Throughput:' in line or 'Latency:' in line:
                    print(f"  {line.strip()}")
                    
        except subprocess.CalledProcessError as e:
            print(f"⚠ Benchmark failed: {e}")
        except FileNotFoundError:
            print("⚠ benchmark_app not found, skipping benchmark")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ONNX model with OpenVINO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="models/openvino",
        help="Output directory for OpenVINO IR files"
    )
    
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision instead of FP16"
    )
    
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip benchmark"
    )
    
    args = parser.parse_args()
    
    # Check input
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        return 1
    
    # Optimize
    success = optimize_openvino(
        args.onnx,
        args.output,
        fp16=not args.fp32,
        benchmark=not args.no_benchmark
    )
    
    if success:
        print("\n✅ Optimization complete!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
