#!/bin/bash
# Quick 1-Epoch Test Training Script for YOLOX on CCPD Dataset
#
# This script runs just 1 epoch for testing purposes

# IMPORTANT: Set this BEFORE changing directory
export CUDA_VISIBLE_DEVICES=''

cd /home/ubuntu/repos/yolox/YOLOX

echo "========================================="
echo "YOLOX CCPD Training - 1 EPOCH TEST"
echo "========================================="
echo "Dataset: CCPD2020/ccpd_green"
echo "Model: YOLOX-Nano"
echo "Epochs: 1 (for testing)"
echo "Device: CPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Batch size: 2"
echo "========================================="
echo ""

# Run training with environment variable explicitly set (NO FP16 on CPU)
env CUDA_VISIBLE_DEVICES='' python tools/train.py \
    -f exps/ccpd/yolox_ccpd_nano_test.py \
    -d 1 \
    -b 2 \
    -o \
    "$@"

echo ""
echo "========================================="
echo "Training test complete!"
echo "Check results in: YOLOX_outputs/yolox_ccpd_nano_test/"
echo "========================================="
