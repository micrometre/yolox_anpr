# CCPD License Plate Detection Pipeline

Complete training and deployment pipeline for license plate detection using YOLOX and OpenVINO.

## Quick Start

### 1. Download Dataset

Download CCPD dataset (~20GB):
- [Google Drive](https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc)
- [BaiduYun (code: hm0u)](https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw)

Extract to `datasets/CCPD2019/`

### 2. Verify Dataset

```bash
python tools/visualize_ccpd.py --dataset datasets/CCPD2019 --num-samples 10
```

### 3. Train Model

Download pretrained weights:
```bash
cd YOLOX/weights
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
```

Train YOLOX-Nano:
```bash
cd YOLOX
python tools/train.py \
    -f exps/ccpd/yolox_ccpd_nano.py \
    -d 1 -b 16 --fp16 -o \
    -c weights/yolox_nano.pth
```

### 4. Export to ONNX

```bash
python tools/export_onnx.py \
    -f YOLOX/exps/ccpd/yolox_ccpd_nano.py \
    -c YOLOX/YOLOX_outputs/yolox_ccpd_nano/best_ckpt.pth \
    -o models/yolox_ccpd_nano.onnx
```

### 5. Optimize with OpenVINO

```bash
python tools/optimize_openvino.py \
    --onnx models/yolox_ccpd_nano.onnx \
    --output models/openvino
```

### 6. Run Inference

```bash
python tools/demo_ccpd_openvino.py \
    --model models/openvino/yolox_ccpd_nano.xml \
    --image test_images/car.jpg \
    --output results/detected.jpg
```

## Pipeline Overview

```
┌─────────────┐     ┌──────────┐     ┌─────────┐     ┌──────────────┐
│ CCPD Dataset│────▶│  YOLOX   │────▶│  ONNX   │────▶│  OpenVINO IR │
│   (300k+)   │     │ Training │     │ Export  │     │ Optimization │
└─────────────┘     └──────────┘     └─────────┘     └──────────────┘
                          │                                  │
                          ▼                                  ▼
                    ┌──────────┐                      ┌──────────┐
                    │PyTorch   │                      │Fast      │
                    │Inference │                      │Inference │
                    └──────────┘                      └──────────┘
```

## Components

### Dataset Loader
- `yolox/data/datasets/ccpd_dataset.py`: CCPD dataset loader
- `yolox/data/datasets/ccpd_classes.py`: Character mappings

### Training Configs
- `exps/ccpd/yolox_ccpd_nano.py`: YOLOX-Nano config (edge/mobile)
- `exps/ccpd/yolox_ccpd_s.py`: YOLOX-S config (server/cloud)

### Utilities
- `tools/visualize_ccpd.py`: Visualize dataset annotations
- `tools/export_onnx.py`: Export trained model to ONNX
- `tools/optimize_openvino.py`: Convert ONNX to OpenVINO IR
- `tools/demo_ccpd_openvino.py`: OpenVINO inference demo

### Documentation
- `docs/CCPD_TRAINING.md`: Complete training guide

## Model Comparison

| Model | Parameters | Size | Latency (CPU) | mAP@0.5 |
|-------|-----------|------|---------------|---------|
| YOLOX-Nano | 0.9M | 1.8 MB | ~10ms | ~85% |
| YOLOX-S | 9.0M | 18 MB | ~20ms | ~90% |

## Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install opencv-python numpy

# YOLOX dependencies
cd YOLOX
pip install -r requirements.txt
pip install -e .

# Export dependencies
pip install onnx onnx-simplifier

# OpenVINO
pip install openvino openvino-dev
```

## Training Tips

1. **Start with pretrained weights**: Fine-tuning from COCO weights converges faster
2. **Use FP16 training**: Faster and uses less GPU memory
3. **Cache dataset**: Use `--cache` flag if you have enough RAM
4. **Monitor with TensorBoard**: Track training progress
5. **Adjust batch size**: Based on your GPU memory

## Deployment Options

### Edge Devices
- Use YOLOX-Nano with OpenVINO FP16
- Target: Raspberry Pi, Intel NUC, edge servers

### Cloud/Server
- Use YOLOX-S with OpenVINO FP16
- Target: High-throughput applications

### Mobile
- Export to TFLite or ONNX Runtime Mobile
- Use YOLOX-Nano for best performance

## Performance Optimization

1. **FP16 Precision**: 2x faster inference with minimal accuracy loss
2. **Batch Processing**: Process multiple images simultaneously
3. **Multi-device**: Use `MULTI:CPU,GPU` for parallel execution
4. **INT8 Quantization**: Further optimization (requires calibration)

## License Plate Recognition (Future)

This pipeline focuses on **detection**. For full ANPR (Automatic Number Plate Recognition):

1. Detect plates (this pipeline)
2. Crop detected regions
3. Apply OCR (PaddleOCR, EasyOCR, or custom model)
4. Post-process text

## Support

For issues or questions:
- Check `docs/CCPD_TRAINING.md` for detailed guide
- Review YOLOX documentation
- Check OpenVINO documentation

## Citation

If you use this pipeline, please cite:

```bibtex
@inproceedings{xu2018towards,
  title={Towards end-to-end license plate detection and recognition: A large dataset and baseline},
  author={Xu, Zhenbo and Yang, Wei and Meng, Ajin and Lu, Nanxue and Huang, Huan and Ying, Changchun and Huang, Liusheng},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={255--271},
  year={2018}
}

@article{ge2021yolox,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
