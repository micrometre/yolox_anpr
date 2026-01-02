#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Modified YOLOX demo to save individual video frames with detections

import argparse
import os
import time
from loguru import logger

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Video Frame Saver")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", required=True, help="path to video file")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="cpu", type=str, help="device to run model")
    parser.add_argument("--conf", default=0.3, type=float, help="confidence threshold")
    parser.add_argument("--nms", default=0.3, type=float, help="nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--output", default="video_frames", help="output folder for frames")
    parser.add_argument("--skip-frames", type=int, default=1, help="save every N frames (1=all frames)")
    parser.add_argument("--max-frames", type=int, default=None, help="maximum frames to save")
    parser.add_argument("--save-all", action="store_true", help="save all frames, not just with detections")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true")
    return parser


class Predictor(object):
    def __init__(self, model, exp, cls_names=COCO_CLASSES, device="cpu", fp16=False, legacy=False):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, 0
        output = output.cpu()

        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, len(bboxes)


def video_frame_demo(predictor, args):
    cap = cv2.VideoCapture(args.path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {args.path}")
    logger.info(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Create output folder
    os.makedirs(args.output, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    detection_count = 0
    
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break
        
        frame_count += 1
        
        # Skip frames based on skip-frames parameter
        if frame_count % args.skip_frames != 0:
            continue
        
        # Stop if max frames reached
        if args.max_frames and saved_count >= args.max_frames:
            break
        
        # Run inference
        outputs, img_info = predictor.inference(frame)
        result_frame, num_detections = predictor.visual(outputs[0], img_info, predictor.confthre)
        
        # Save frame if it has detections or if save_all is True
        if args.save_all or num_detections > 0:
            frame_filename = f"frame_{frame_count:06d}_det{num_detections}.jpg"
            save_path = os.path.join(args.output, frame_filename)
            cv2.imwrite(save_path, result_frame)
            saved_count += 1
            
            if num_detections > 0:
                detection_count += 1
                logger.info(f"Frame {frame_count}: {num_detections} detections - saved as {frame_filename}")
        
        # Progress update every 100 frames
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames, saved {saved_count} frames")
    
    cap.release()
    
    logger.info("=" * 60)
    logger.info(f"Processing complete!")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Frames saved: {saved_count}")
    logger.info(f"Frames with detections: {detection_count}")
    logger.info(f"Output folder: {args.output}")
    logger.info("=" * 60)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    if args.ckpt is None:
        logger.error("Please provide checkpoint path with -c/--ckpt")
        return
    
    logger.info("Loading checkpoint from {}".format(args.ckpt))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("Checkpoint loaded successfully")

    if args.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)

    predictor = Predictor(model, exp, COCO_CLASSES, args.device, args.fp16, args.legacy)
    video_frame_demo(predictor, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
