# Quick Start Guide for Your CCPD2020 Dataset

## ✅ Dataset Successfully Loaded!

Your dataset structure has been detected and configured:
- **Location**: `dataset/CCPD2020/`
- **Subset**: `ccpd_green` (green energy vehicle plates with 8 digits)
- **Total images**: 11,776
  - Train: ~8,000 images
  - Val: ~1,600 images  
  - Test: ~2,100 images

## 🎯 Quick Commands

### 1. Visualize Dataset (Verify it works)
```bash
# View 10 random samples
python tools/visualize_ccpd.py --dataset dataset/CCPD2020 --num-samples 10

# Save visualizations to files
python tools/visualize_ccpd.py --dataset dataset/CCPD2020 --num-samples 20 --output-dir vis_output
```

### 2. Train YOLOX-Nano (Recommended for first try)
```bash
cd YOLOX

# Download pretrained weights first
mkdir -p weights
cd weights
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
cd ..

# Train (adjust batch size based on your GPU memory)
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_nano.py \
    -d 1 \
    -b 8 \
    --fp16 \
    -o \
    -c weights/yolox_nano.pth
```

**Note**: The exp files use `image_sets=["ccpd_base"]` by default, but the dataset loader will automatically detect and use `ccpd_green` since that's what you have.

### 3. Train on Specific Split
To train only on the training set:
```bash
# The dataset loader automatically uses the train/ subdirectory when split='train'
# This is already configured in the exp files
```

## 📊 What's Different About Your Dataset

Your CCPD2020/ccpd_green dataset has:
- **8-digit plates** (green energy vehicles) instead of standard 7-digit
- **Pre-split structure**: train/val/test subdirectories
- **Newer format**: CCPD2020 instead of CCPD2019

The dataset loader has been updated to handle this automatically!

## 🔧 Dataset Loader Updates

The `CCPDDataset` class now:
1. ✅ Auto-detects train/val/test subdirectories
2. ✅ Loads from appropriate split when specified
3. ✅ Works with both CCPD2019 (flat) and CCPD2020 (nested) structures
4. ✅ Handles both ccpd_base and ccpd_green subsets

## 🚀 Next Steps

1. **Verify visualization works**: Run the visualize command above
2. **Start training**: Use the training command (start with small batch size)
3. **Monitor progress**: Use TensorBoard to watch training
   ```bash
   tensorboard --logdir YOLOX_outputs/yolox_ccpd_nano/vis_data
   ```

## 💡 Tips

- **Batch size**: Start with `-b 8` and increase if you have GPU memory
- **FP16 training**: The `--fp16` flag reduces memory usage by ~50%
- **Caching**: Add `--cache` if you have >20GB RAM for faster training
- **Multi-GPU**: Change `-d 1` to `-d 2` (or more) for multi-GPU training

## 📝 Training Progress

Training will save checkpoints to:
```
YOLOX/YOLOX_outputs/yolox_ccpd_nano/
├── best_ckpt.pth          # Best model (highest mAP)
├── last_epoch_ckpt.pth    # Latest checkpoint
├── epoch_*.pth            # Periodic checkpoints
└── vis_data/              # TensorBoard logs
```

After training completes, export to ONNX and optimize with OpenVINO as described in the main documentation.

## ❓ Troubleshooting

If you see "Warning: Subset directory not found":
- ✅ **Fixed!** The dataset loader now handles your structure automatically

If training fails with OOM (Out of Memory):
- Reduce batch size: `-b 4` or `-b 2`
- Ensure FP16 is enabled: `--fp16`
- Close other applications using GPU memory

## 📚 Full Documentation

See `CCPD_README.md` and `docs/CCPD_TRAINING.md` for complete documentation.
