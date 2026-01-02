#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
CCPD character mappings for Chinese license plates.

A valid Chinese license plate consists of 7 characters:
- Province (1 character): Chinese character
- Alphabets (1 character): A-Z (excluding I and O)
- Alphanumeric (5 characters): A-Z and 0-9 (excluding I and O)
"""

# Chinese provinces
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"
]

# Alphabets (no I or O)
ALPHABETS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O'
]

# Alphanumeric (no I or O)
ALPHANUMERIC = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3',
    '4', '5', '6', '7', '8', '9', 'O'
]


def decode_license_plate(indices):
    """
    Decode license plate from character indices.
    
    Args:
        indices: List of 7 integers representing character indices
        
    Returns:
        str: Decoded license plate string
    """
    if len(indices) != 7:
        return ""
    
    plate = ""
    plate += PROVINCES[indices[0]]
    plate += ALPHABETS[indices[1]]
    for i in range(2, 7):
        plate += ALPHANUMERIC[indices[i]]
    
    return plate
