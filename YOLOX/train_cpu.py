#!/usr/bin/env python3
"""
CPU Training wrapper for YOLOX.

This script forces YOLOX to train on CPU by setting CUDA_VISIBLE_DEVICES to empty.
"""

import os
import sys

# Force CPU training by hiding CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Now import and run the training script
sys.path.insert(0, os.path.dirname(__file__))

# Import the train module
from tools import train

# Run the training
if __name__ == "__main__":
    sys.exit(train.main() if hasattr(train, 'main') else 0)
