#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
job_dir = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
out_dir = ROOT / 'experiment_outputs' / 'fix_center_perspective'
out_dir.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(job_dir / 'original.png'))
rect_info = json.loads((job_dir / 'rectification.json').read_text(encoding='utf-8'))
poly = np.array(rect_info['source_polygon'], dtype=np.float32)

# Get the 4 corners of the body (table)
# source_body_corners from rectification
body_corners = np.array(rect_info['source_body_corners'], dtype=np.float32)
# Sort to get tl,tr,br,bl
# body_corners are in some order
print('body_corners:', body_corners.tolist())

# Try to order properly
def order_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # tl
    rect[2] = pts[np.argmax(s)] # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # tr
    rect[3] = pts[np.argmax(diff)] # bl
    return rect

src = order_points(body_corners)
print('src ordered:', src.tolist())

# Compute current orientation
# Top edge vector
top_vec = src[1] - src[0]
bottom_vec = src[2] - src[3]
print('top_vec:', top_vec, 'angle:', np.degrees(np.arctan2(top_vec[1], top_vec[0])))
print('bottom_vec:', bottom_vec, 'angle:', np.degrees(np.arctan2(bottom_vec[1], bottom_vec[0])))

# Widths
top_w = np.linalg.norm(src[1]-src[0])
bottom_w = np.linalg.norm(src[2]-src[3])
left_h = np.linalg.norm(src[3]-src[0])
right_h = np.linalg.norm(src[2]-src[1])
print(f'widths top={top_w:.1f} bottom={bottom_w:.1f}, heights left={left_h:.1f} right={right_h:.1f}')
