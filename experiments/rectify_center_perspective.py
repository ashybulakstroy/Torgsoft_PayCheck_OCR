#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
job_dir = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
out_dir = ROOT / 'experiment_outputs' / 'rectify_center_perspective'
out_dir.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(job_dir / 'original.png'))
rect_info = json.loads((job_dir / 'rectification.json').read_text(encoding='utf-8'))
body_corners = np.array(rect_info['source_body_corners'], dtype=np.float32)

def order_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

src = order_points(body_corners)
h,w = orig.shape[:2]

# 1) Rotate to make horizontal edges parallel to image edges
# compute average angle of top and bottom edges
top_vec = src[1] - src[0]
bottom_vec = src[2] - src[3]
angle = np.degrees(np.arctan2((top_vec[1]+bottom_vec[1])/2, (top_vec[0]+bottom_vec[0])/2))
# we want to rotate so angle becomes 0
# rotate around center of the polygon
center = np.mean(src, axis=0)
M_rot = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
# compute image size after rotation
cos = np.abs(M_rot[0,0]); sin = np.abs(M_rot[0,1])
nw = int((h*sin) + (w*cos))
nh = int((h*cos) + (w*sin))
M_rot[0,2] += (nw/2) - center[0]
M_rot[1,2] += (nh/2) - center[1]
rotated = cv2.warpAffine(orig, M_rot, (nw,nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
src_rot = cv2.transform(src.reshape(1,-1,2), M_rot).reshape(4,2)
cv2.imwrite(str(out_dir / '01_rotated.png'), rotated)
print('rotated:', nw,nh)
print('src_rot:', src_rot.tolist())

# 2) Now fix perspective to make rectangle - stretch top to match bottom width, center perspective
src_rot = order_points(src_rot)
top_w = np.linalg.norm(src_rot[1]-src_rot[0])
bottom_w = np.linalg.norm(src_rot[2]-src_rot[3])
avg_w = (top_w + bottom_w)/2
# use bottom width as target (wider) and stretch top
target_w = bottom_w
target_h = max(np.linalg.norm(src_rot[3]-src_rot[0]), np.linalg.norm(src_rot[2]-src_rot[1]))
# dst rectangle centered
margin = 20
dst = np.array([
    [margin, margin],
    [margin + target_w, margin],
    [margin + target_w, margin + target_h],
    [margin, margin + target_h]
], dtype=np.float32)
# but we need to keep vertical alignment centered
M_persp = cv2.getPerspectiveTransform(src_rot, dst)
rectified = cv2.warpPerspective(rotated, M_persp, (int(target_w + 2*margin), int(target_h + 2*margin)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(out_dir / '02_rectified.png'), rectified)
print('done:', rectified.shape)
