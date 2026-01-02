#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
CCPD License Plate Detection with OpenVINO.

This script performs license plate detection using a trained YOLOX model
optimized with OpenVINO for fast inference on Intel hardware.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import time

# OpenVINO Runtime API
from openvino import Core


def preprocess(img, input_size=(640, 640)):
    """Preprocess image for YOLOX inference (letterbox + normalize)."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose((2, 0, 1))  # HWC -> CHW
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = np.expand_dims(padded_img, axis=0)  # Add batch dimension
    
    return padded_img, r


def nms(boxes, scores, nms_thresh):
    """Non-maximum suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def postprocess(outputs, conf_thresh, nms_thresh, ratio):
    """Postprocess YOLOX output."""
    predictions = outputs[0]  # [num_anchors, 6] for single class
    
    # Convert center format to corner format
    box_corner = np.zeros_like(predictions[:, :4])
    box_corner[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
    box_corner[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1
    box_corner[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
    box_corner[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2
    predictions[:, :4] = box_corner
    
    # Get objectness and class scores
    obj_conf = predictions[:, 4]
    class_conf = predictions[:, 5:]
    class_pred = np.argmax(class_conf, axis=1)
    class_score = class_conf[np.arange(len(class_pred)), class_pred]
    
    # Combined score
    scores = obj_conf * class_score
    
    # Filter by confidence
    mask = scores > conf_thresh
    boxes = predictions[mask, :4]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return None, None
    
    # Scale boxes back to original image size
    boxes = boxes / ratio
    
    # Apply NMS
    keep = nms(boxes, scores, nms_thresh)
    final_boxes = boxes[keep]
    final_scores = scores[keep]
    
    return np.array(final_boxes), np.array(final_scores)


def draw_detections(img, boxes, scores, threshold=0.5):
    """Draw bounding boxes on image."""
    if boxes is None:
        return img
    
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
            
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Plate: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img


def detect_image(model_path, image_path, output_path=None, threshold=0.5, device="CPU"):
    """
    Detect license plates in an image.
    
    Args:
        model_path: Path to OpenVINO IR model (.xml)
        image_path: Path to input image
        output_path: Path to save output image (optional)
        threshold: Detection confidence threshold
        device: OpenVINO device (CPU, GPU, AUTO)
    """
    # Initialize OpenVINO
    print(f"Initializing OpenVINO with device: {device}")
    core = Core()
    
    # Load model
    print(f"Loading model: {model_path}")
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    print(f"Input shape: {input_layer.shape}")
    print(f"Output shape: {output_layer.shape}")
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Preprocess
    input_tensor, ratio = preprocess(img, (640, 640))
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    results = compiled_model([input_tensor])
    outputs = results[output_layer]
    inference_time = (time.time() - start_time) * 1000
    
    print(f"Inference time: {inference_time:.2f} ms")
    
    # Postprocess
    boxes, scores = postprocess(outputs, threshold, 0.45, ratio)
    
    if boxes is not None:
        print(f"Detected {len(boxes)} license plate(s)")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            print(f"  Plate {i+1}: bbox={box.astype(int).tolist()}, confidence={score:.3f}")
    else:
        print("No license plates detected")
    
    # Draw detections
    result_img = draw_detections(img.copy(), boxes, scores, threshold)
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Saved result to: {output_path}")
    else:
        cv2.imshow("License Plate Detection", result_img)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="CCPD License Plate Detection with OpenVINO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to OpenVINO IR model (.xml)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output image (if not set, displays interactively)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device: CPU, GPU, AUTO"
    )
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Detect
    try:
        detect_image(args.model, args.image, args.output, args.threshold, args.device)
        print("\n✅ Detection complete!")
        return 0
    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
