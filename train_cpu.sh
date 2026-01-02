#!/bin/bash
# CPU Training Script for YOLOX on CCPD Dataset
#
# This script forces CPU training by hiding CUDA devices

cd /home/ubuntu/repos/yolox/YOLOX

# Set environment to force CPU training
export CUDA_VISIBLE_DEVICES=''

# Run training
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_nano.py \
    -d 1 \
    -b 2 \
    --fp16 \
    -o \
    "$@"
