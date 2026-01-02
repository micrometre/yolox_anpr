#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
CCPD Dataset loader for license plate detection.

Dataset structure:
    CCPD2019/
        ccpd_base/
        ccpd_blur/
        ccpd_challenge/
        ccpd_db/
        ccpd_fn/
        ccpd_rotate/
        ccpd_tilt/
        ccpd_weather/
        ccpd_green/
        splits/
            train.txt
            val.txt
            test.txt

Filename format:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    
Fields (separated by '-'):
    1. Area: License plate area ratio (025 = 0.25)
    2. Tilt: Horizontal_Vertical tilt degrees (95_113)
    3. BBox: Left-top & Right-bottom coords (154&383_386&473)
    4. Four vertices: Exact (x,y) of 4 corners, starting from right-bottom
    5. License plate number: 7 character indices (0_0_22_27_27_33_16)
    6. Brightness: 0-100
    7. Blurriness: 0-100
"""

import os
import os.path
import cv2
import numpy as np
from pathlib import Path

from .datasets_wrapper import CacheDataset, cache_read_img
from .ccpd_classes import PROVINCES, ALPHABETS, ALPHANUMERIC, decode_license_plate


class CCPDDataset(CacheDataset):
    """
    CCPD Detection Dataset for License Plate Detection.
    
    Args:
        data_dir (str): Path to CCPD2019 directory
        image_sets (list): List of subset names to use, e.g., ['ccpd_base', 'ccpd_blur']
        img_size (tuple): Input image size (height, width)
        preproc (callable): Preprocessing function
        split (str): 'train', 'val', or 'test'. If None, uses all images.
        cache (bool): Whether to cache images
        cache_type (str): 'ram' or 'disk'
    """
    
    def __init__(
        self,
        data_dir,
        image_sets=["ccpd_base"],
        img_size=(640, 640),
        preproc=None,
        split=None,
        cache=False,
        cache_type="ram",
    ):
        self.root = data_dir
        self.image_sets = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.split = split
        
        # CCPD has only one class: license plate
        self._classes = ["license_plate"]
        self.cats = [{"id": 0, "name": "license_plate"}]
        self.class_ids = [0]
        
        # Load image paths
        self.ids = []
        self._load_image_paths()
        
        self.num_imgs = len(self.ids)
        print(f"Loaded {self.num_imgs} images from CCPD dataset (split={split})")
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Prepare for caching
        path_filename = [str(Path(img_path).relative_to(self.root)) for img_path in self.ids]
        
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name=f"cache_ccpd_{split if split else 'all'}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
    
    def _load_image_paths(self):
        """Load image paths from specified subsets and split."""
        all_images = []
        
        # Collect all images from specified subsets
        for subset in self.image_sets:
            subset_dir = os.path.join(self.root, subset)
            if not os.path.exists(subset_dir):
                print(f"Warning: Subset directory {subset_dir} not found, skipping...")
                continue
            
            # Check if subset has train/val/test subdirectories
            has_splits = all(os.path.exists(os.path.join(subset_dir, split)) 
                           for split in ['train', 'val', 'test'])
            
            if has_splits and self.split is not None:
                # Load from specific split subdirectory
                split_dir = os.path.join(subset_dir, self.split)
                if os.path.exists(split_dir):
                    for img_file in os.listdir(split_dir):
                        if img_file.endswith('.jpg'):
                            img_path = os.path.join(split_dir, img_file)
                            all_images.append(img_path)
                    print(f"Loaded from {split_dir}")
                else:
                    print(f"Warning: Split directory {split_dir} not found")
            elif has_splits:
                # Load from all splits (train + val + test)
                for split in ['train', 'val', 'test']:
                    split_dir = os.path.join(subset_dir, split)
                    if os.path.exists(split_dir):
                        for img_file in os.listdir(split_dir):
                            if img_file.endswith('.jpg'):
                                img_path = os.path.join(split_dir, img_file)
                                all_images.append(img_path)
                print(f"Loaded from all splits in {subset_dir}")
            else:
                # Original behavior: load directly from subset directory
                for img_file in os.listdir(subset_dir):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(subset_dir, img_file)
                        all_images.append(img_path)
        
        # Filter by split file if specified (for datasets with split txt files)
        if self.split is not None:
            split_file = os.path.join(self.root, "splits", f"{self.split}.txt")
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    split_images = set(line.strip() for line in f)
                
                # Filter images that are in the split
                filtered_images = []
                for img_path in all_images:
                    img_name = os.path.basename(img_path)
                    if img_name in split_images:
                        filtered_images.append(img_path)
                
                self.ids = filtered_images
                print(f"Using split file: {split_file} ({len(filtered_images)} images)")
            else:
                # No split file, use all images collected
                self.ids = all_images
        else:
            self.ids = all_images
    
    def parse_filename(self, filename):
        """
        Parse CCPD filename to extract annotations.
        
        Args:
            filename: Image filename (without path)
            
        Returns:
            dict: Parsed annotations containing:
                - bbox: [x1, y1, x2, y2]
                - vertices: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - plate_number: List of 7 character indices
                - plate_text: Decoded license plate string
        """
        basename = os.path.splitext(filename)[0]
        parts = basename.split('-')
        
        if len(parts) < 7:
            # Invalid filename format
            return None
        
        try:
            # Parse bounding box (field 2: left-top & right-bottom)
            bbox_parts = parts[2].split('_')
            lt = bbox_parts[0].split('&')  # left-top
            rb = bbox_parts[1].split('&')  # right-bottom
            x1, y1 = int(lt[0]), int(lt[1])
            x2, y2 = int(rb[0]), int(rb[1])
            bbox = [x1, y1, x2, y2]
            
            # Parse four vertices (field 3)
            vertices_parts = parts[3].split('_')
            vertices = []
            for v in vertices_parts:
                coords = v.split('&')
                vertices.append([int(coords[0]), int(coords[1])])
            
            # Parse license plate number (field 4)
            plate_indices = [int(x) for x in parts[4].split('_')]
            plate_text = decode_license_plate(plate_indices)
            
            return {
                'bbox': bbox,
                'vertices': vertices,
                'plate_number': plate_indices,
                'plate_text': plate_text
            }
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse filename {filename}: {e}")
            return None
    
    def _load_annotations(self):
        """Load all annotations."""
        return [self.load_anno_from_ids(i) for i in range(self.num_imgs)]
    
    def load_anno_from_ids(self, index):
        """
        Load annotation for a specific image index.
        
        Returns:
            tuple: (res, img_info, resized_info)
                - res: np.array of shape [1, 5] containing [x1, y1, x2, y2, class_id]
                - img_info: (height, width) of original image
                - resized_info: (height, width) of resized image
        """
        img_path = self.ids[index]
        filename = os.path.basename(img_path)
        
        # Parse annotation from filename
        anno = self.parse_filename(filename)
        
        if anno is None:
            # Return empty annotation if parsing failed
            return (np.empty((0, 5)), (0, 0), (0, 0))
        
        # Get original image size
        img = cv2.imread(img_path)
        if img is None:
            return (np.empty((0, 5)), (0, 0), (0, 0))
        
        height, width = img.shape[:2]
        img_info = (height, width)
        
        # Create annotation array: [x1, y1, x2, y2, class_id]
        bbox = anno['bbox']
        res = np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 0]], dtype=np.float32)
        
        # Calculate resize ratio
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))
        
        return (res, img_info, resized_info)
    
    def load_anno(self, index):
        """Load annotation for mixup/mosaic."""
        return self.annotations[index][0]
    
    def load_resized_img(self, index):
        """Load and resize image."""
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img
    
    def load_image(self, index):
        """Load image from disk."""
        img_path = self.ids[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None, f"Failed to load image: {img_path}"
        return img
    
    @cache_read_img(use_cache=True)
    def read_img(self, index):
        """Read image with caching support."""
        return self.load_resized_img(index)
    
    def pull_item(self, index):
        """
        Returns the original image and target at an index for mixup.
        
        Returns:
            tuple: (img, target, img_info, index)
        """
        target, img_info, _ = self.annotations[index]
        img = self.read_img(index)
        return img, target, img_info, index
    
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        Get item with preprocessing.
        
        Returns:
            tuple: (img, target, img_info, img_id)
        """
        img, target, img_info, img_id = self.pull_item(index)
        
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        
        return img, target, img_info, img_id
    
    def __len__(self):
        return self.num_imgs
