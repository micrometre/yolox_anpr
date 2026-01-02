#!/usr/bin/env python3
"""
Quick test script for CCPD dataset with ccpd_green.
This works with the CCPD2020 dataset structure.
"""

import sys
import os
sys.path.insert(0, 'YOLOX')

from yolox.data.datasets.ccpd_dataset import CCPDDataset
import cv2
import numpy as np

# Your dataset path
data_dir = "dataset/CCPD2020"

print("Testing CCPD dataset loader...")
print(f"Dataset directory: {data_dir}")

# Create dataset with ccpd_green subset
try:
    dataset = CCPDDataset(
        data_dir=data_dir,
        image_sets=["ccpd_green"],  # Use ccpd_green instead of ccpd_base
        img_size=(640, 640),
        split=None,  # Use all images (train+val+test)
    )
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total images: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test first image
        print(f"\nTesting first image...")
        img_path = dataset.ids[0]
        filename = os.path.basename(img_path)
        print(f"  File: {filename}")
        
        # Parse annotation
        anno = dataset.parse_filename(filename)
        if anno:
            print(f"  BBox: {anno['bbox']}")
            print(f"  Plate: {anno['plate_text']}")
            print(f"  ✓ Annotation parsing works!")
        else:
            print(f"  ✗ Failed to parse annotation")
            
        # Try to load image
        img, target, img_info, img_id = dataset.pull_item(0)
        print(f"  Image shape: {img.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  ✓ Image loading works!")
        
        print(f"\n✅ Dataset is working correctly!")
        print(f"\nYou can now:")
        print(f"  1. Visualize: python tools/visualize_ccpd.py --dataset {data_dir} --num-samples 10")
        print(f"  2. Train: Update exp files to use image_sets=['ccpd_green']")
        
    else:
        print(f"\n⚠ No images found in dataset")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"\n💡 Troubleshooting:")
    print(f"  1. Check dataset path: {data_dir}")
    print(f"  2. Verify ccpd_green directory exists")
    print(f"  3. Ensure images are in .jpg format")
