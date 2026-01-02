#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .ccpd_dataset import CCPDDataset
from .ccpd_classes import PROVINCES, ALPHABETS, ALPHANUMERIC, decode_license_plate
from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import CacheDataset, ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection
