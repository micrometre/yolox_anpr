#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Visualize CCPD dataset annotations.

This script helps verify that the CCPD dataset is properly formatted
and annotations are correctly parsed.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path

# Add YOLOX to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "YOLOX"))

from yolox.data.datasets.ccpd_dataset import CCPDDataset
from yolox.data.datasets.ccpd_classes import decode_license_plate


def draw_annotation(img, anno, color=(0, 255, 0)):
    """
    Draw annotation on image.
    
    Args:
        img: Image array
        anno: Annotation dict from parse_filename
        color: BGR color tuple
    """
    if anno is None:
        return img
    
    # Draw bounding box
    bbox = anno['bbox']
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    # Draw four vertices
    vertices = anno['vertices']
    for i, vertex in enumerate(vertices):
        cv2.circle(img, tuple(vertex), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i), tuple(vertex), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
    
    # Draw polygon connecting vertices
    pts = np.array(vertices, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (255, 0, 0), 2)
    
    # Draw license plate text
    plate_text = anno['plate_text']
    text_pos = (bbox[0], bbox[1] - 10)
    cv2.putText(img, plate_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
               0.8, color, 2)
    
    return img


def visualize_dataset(data_dir, num_samples=10, output_dir=None):
    """
    Visualize random samples from CCPD dataset.
    
    Args:
        data_dir: Path to CCPD2019/CCPD2020 directory
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations (optional)
    """
    print(f"Loading CCPD dataset from: {data_dir}")
    
    # Auto-detect available subsets
    available_subsets = []
    for subset in ["ccpd_base", "ccpd_green", "ccpd_blur", "ccpd_challenge", 
                   "ccpd_db", "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"]:
        subset_path = os.path.join(data_dir, subset)
        if os.path.exists(subset_path):
            available_subsets.append(subset)
    
    if not available_subsets:
        print(f"Error: No CCPD subsets found in {data_dir}")
        print("Expected subsets: ccpd_base, ccpd_green, ccpd_blur, etc.")
        return
    
    print(f"Found subsets: {', '.join(available_subsets)}")
    
    # Create dataset without split to see all images
    dataset = CCPDDataset(
        data_dir=data_dir,
        image_sets=available_subsets,  # Use all available subsets
        img_size=(640, 640),
        split=None,  # Use all images
    )
    
    print(f"Dataset size: {len(dataset)} images")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving visualizations to: {output_dir}")
    
    # Randomly sample images
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        img_path = dataset.ids[idx]
        filename = os.path.basename(img_path)
        
        print(f"\n[{i+1}/{num_samples}] {filename}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Error: Failed to load image")
            continue
        
        # Parse annotation
        anno = dataset.parse_filename(filename)
        if anno is None:
            print(f"  Error: Failed to parse annotation")
            continue
        
        # Print annotation details
        print(f"  BBox: {anno['bbox']}")
        print(f"  Plate: {anno['plate_text']}")
        print(f"  Vertices: {anno['vertices']}")
        
        # Draw annotation
        vis_img = draw_annotation(img.copy(), anno)
        
        # Save or display
        if output_dir:
            output_path = os.path.join(output_dir, f"vis_{i:03d}_{filename}")
            cv2.imwrite(output_path, vis_img)
            print(f"  Saved to: {output_path}")
        else:
            # Display
            cv2.imshow(f"CCPD Sample {i+1}/{num_samples}", vis_img)
            print("  Press any key to continue, 'q' to quit...")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == ord('q'):
                break
    
    print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CCPD dataset annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/CCPD2019",
        help="Path to CCPD2019 directory"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (if not set, displays interactively)"
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory not found: {args.dataset}")
        print("\nPlease download CCPD dataset from:")
        print("  Google Drive: https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc")
        print("  BaiduYun (code: hm0u): https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw")
        print(f"\nExtract to: {args.dataset}")
        return 1
    
    visualize_dataset(args.dataset, args.num_samples, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
