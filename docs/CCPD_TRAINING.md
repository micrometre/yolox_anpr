# CCPD License Plate Detection Training Guide

Complete guide for training YOLOX models on the CCPD dataset for license plate detection, with OpenVINO optimization for deployment.

## Table of Contents

1. [Dataset Setup](#dataset-setup)
2. [Training](#training)
3. [Model Export](#model-export)
4. [OpenVINO Optimization](#openvino-optimization)
5. [Inference](#inference)
6. [Troubleshooting](#troubleshooting)

## Dataset Setup

### Download CCPD Dataset

The CCPD (Chinese City Parking Dataset) contains 300k+ images with license plate annotations.

**Download from:**
- Google Drive: https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc
- BaiduYun Drive (code: hm0u): https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw

**Extract to:**
```bash
cd /home/ubuntu/repos/yolox
mkdir -p datasets/CCPD2019
cd datasets/CCPD2019
# Extract CCPD2019.tar.xz here
tar xf CCPD2019.tar.xz
```

### Dataset Structure

After extraction, your directory should look like:
```
datasets/CCPD2019/
├── ccpd_base/          # Base dataset (training)
├── ccpd_blur/          # Blurred plates (testing)
├── ccpd_challenge/     # Challenging cases (testing)
├── ccpd_db/            # Dark/bright (testing)
├── ccpd_fn/            # Far/near (testing)
├── ccpd_rotate/        # Rotated plates (testing)
├── ccpd_tilt/          # Tilted plates (testing)
├── ccpd_weather/       # Weather conditions (testing)
├── ccpd_green/         # Green energy plates (testing)
└── splits/             # Train/val/test splits (optional)
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Verify Dataset

Visualize some samples to verify the dataset is correctly loaded:

```bash
cd /home/ubuntu/repos/yolox
python tools/visualize_ccpd.py --dataset datasets/CCPD2019 --num-samples 10
```

This will display 10 random images with annotations. Press any key to continue, 'q' to quit.

To save visualizations instead:
```bash
python tools/visualize_ccpd.py --dataset datasets/CCPD2019 --num-samples 20 --output-dir ccpd_vis
```

## Training

### Option 1: Train from Scratch

Train YOLOX-Nano (for edge/mobile deployment):
```bash
cd /home/ubuntu/repos/yolox/YOLOX
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_nano.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -o
```

Train YOLOX-S (for server deployment):
```bash
cd /home/ubuntu/repos/yolox/YOLOX
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_s.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -o
```

### Option 2: Fine-tune from COCO Pretrained Weights (Recommended)

Download pretrained weights:
```bash
cd /home/ubuntu/repos/yolox/YOLOX
mkdir -p weights
cd weights

# For YOLOX-Nano
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth

# For YOLOX-S
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

Fine-tune YOLOX-Nano:
```bash
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_nano.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -o \
    -c weights/yolox_nano.pth
```

Fine-tune YOLOX-S:
```bash
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_s.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -o \
    -c weights/yolox_s.pth
```

### Training Parameters

- `-f`: Experiment file
- `-d`: Number of GPUs (use 1 for single GPU)
- `-b`: Batch size (adjust based on GPU memory)
- `--fp16`: Use mixed precision training (faster, less memory)
- `-o`: Occupy GPU memory first
- `-c`: Checkpoint to resume/fine-tune from

### Multi-GPU Training

For multi-GPU training:
```bash
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_s.py \
    -d 4 \
    -b 64 \
    --fp16 \
    -o
```

### Training with Caching

Speed up training by caching images in RAM:
```bash
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_s.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -o \
    --cache
```

> **Note:** Ensure you have enough RAM (CCPD base dataset is ~20GB)

### Monitor Training

Training logs and checkpoints are saved to:
```
YOLOX_outputs/yolox_ccpd_nano/  # or yolox_ccpd_s
├── train_log.txt
├── vis_data/
│   └── events.out.tfevents.*
└── epoch_*.pth
└── best_ckpt.pth
└── last_epoch_ckpt.pth
```

View with TensorBoard:
```bash
tensorboard --logdir YOLOX_outputs/yolox_ccpd_nano/vis_data
```

## Model Export

### Export to ONNX

After training, export the best checkpoint to ONNX:

```bash
cd /home/ubuntu/repos/yolox
python tools/export_onnx.py \
    -f YOLOX/exps/ccpd/yolox_ccpd_nano.py \
    -c YOLOX/YOLOX_outputs/yolox_ccpd_nano/best_ckpt.pth \
    -o models/yolox_ccpd_nano.onnx
```

For YOLOX-S:
```bash
python tools/export_onnx.py \
    -f YOLOX/exps/ccpd/yolox_ccpd_s.py \
    -c YOLOX/YOLOX_outputs/yolox_ccpd_s/best_ckpt.pth \
    -o models/yolox_ccpd_s.onnx
```

The script will:
- Export model to ONNX format
- Simplify the ONNX graph (requires `onnx-simplifier`)
- Print model information

## OpenVINO Optimization

### Convert ONNX to OpenVINO IR

Convert the ONNX model to OpenVINO IR format for optimized inference:

```bash
python tools/optimize_openvino.py \
    --onnx models/yolox_ccpd_nano.onnx \
    --output models/openvino
```

This will generate:
- `models/openvino/yolox_ccpd_nano.xml` (model structure)
- `models/openvino/yolox_ccpd_nano.bin` (model weights)

### FP16 vs FP32

By default, the script uses FP16 precision for faster inference. For FP32:
```bash
python tools/optimize_openvino.py \
    --onnx models/yolox_ccpd_nano.onnx \
    --output models/openvino \
    --fp32
```

### Benchmark

The script automatically runs a benchmark. To skip:
```bash
python tools/optimize_openvino.py \
    --onnx models/yolox_ccpd_nano.onnx \
    --output models/openvino \
    --no-benchmark
```

## Inference

### Image Inference

Detect license plates in an image:

```bash
python tools/demo_ccpd_openvino.py \
    --model models/openvino/yolox_ccpd_nano.xml \
    --image test_images/car.jpg \
    --output results/detected.jpg \
    --threshold 0.5
```

### Video Inference

For video processing, use the existing video demo script:

```bash
python tools/demo_video_openvino.py \
    test_videos/traffic.mp4 \
    --model models/openvino/yolox_ccpd_nano.xml \
    --output-dir results \
    --threshold 0.5 \
    --device CPU
```

### Device Selection

OpenVINO supports multiple devices:
- `CPU`: Intel CPU
- `GPU`: Intel integrated/discrete GPU
- `AUTO`: Automatic device selection
- `MULTI:CPU,GPU`: Multi-device execution

Example with GPU:
```bash
python tools/demo_ccpd_openvino.py \
    --model models/openvino/yolox_ccpd_nano.xml \
    --image test_images/car.jpg \
    --device GPU
```

## Troubleshooting

### Out of Memory During Training

Reduce batch size:
```bash
python tools/train.py -f exps/ccpd/yolox_ccpd_nano.py -d 1 -b 8 --fp16 -o
```

Or use gradient accumulation (not directly supported, requires code modification).

### Low Detection Accuracy

1. **Train longer**: Increase `max_epoch` in the exp file
2. **Use larger model**: Switch from Nano to S
3. **Adjust augmentation**: Modify augmentation parameters in exp file
4. **Use more data**: Include additional CCPD subsets in training

### ONNX Export Fails

Install required packages:
```bash
pip install onnx onnx-simplifier
```

### OpenVINO Conversion Fails

Install OpenVINO development tools:
```bash
pip install openvino-dev
```

### Slow Inference

1. **Use FP16**: Ensure you're using FP16 precision
2. **Use GPU**: If available, use `--device GPU`
3. **Batch processing**: Process multiple images at once
4. **Smaller model**: Use YOLOX-Nano instead of YOLOX-S

## Performance Benchmarks

Expected performance on Intel Core i7 (CPU):

| Model | Size | FP32 Latency | FP16 Latency | mAP@0.5 |
|-------|------|--------------|--------------|---------|
| YOLOX-Nano | 0.9 MB | ~15ms | ~10ms | ~85% |
| YOLOX-S | 9 MB | ~30ms | ~20ms | ~90% |

> **Note:** Actual performance depends on hardware and image size.

## Next Steps

1. **Integrate OCR**: Add license plate text recognition
2. **Deploy to edge**: Use OpenVINO on edge devices (Raspberry Pi, etc.)
3. **Optimize further**: Use INT8 quantization for even faster inference
4. **Multi-language**: Train on datasets with different license plate formats

## References

- [YOLOX Paper](https://arxiv.org/abs/2107.08430)
- [CCPD Dataset Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf)
- [OpenVINO Documentation](https://docs.openvino.ai/)
