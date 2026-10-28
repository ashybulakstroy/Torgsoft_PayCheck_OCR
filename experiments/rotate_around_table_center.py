#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
job_dir = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
out_dir = ROOT / 'experiment_outputs' / 'rotate_around_table_center'
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
# center of the table
table_center = np.mean(src, axis=0)
print('table center:', table_center.tolist())

h,w = orig.shape[:2]
# angle of top edge
vec = src[1] - src[0]
angle = np.degrees(np.arctan2(vec[1], vec[0]))
print('angle:', angle)
rot_angle = -angle  # counterclockwise
print('rotate by:', rot_angle)

# rotate around table center
M = cv2.getRotationMatrix2D(tuple(table_center), rot_angle, 1.0)
rotated = cv2.warpAffine(orig, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(out_dir / 'rotated_full.png'), rotated)

# crop the table area after rotation
src_r = cv2.transform(src.reshape(1,-1,2), M).reshape(4,2)
# axis aligned bbox
x1,y1 = src_r.min(axis=0)
x2,y2 = src_r.max(axis=0)
margin = 10
x1 = max(0,int(x1)-margin); y1 = max(0,int(y1)-margin)
x2 = min(w,int(x2)+margin); y2 = min(h,int(y2)+margin)
crop = rotated[y1:y2, x1:x2]
cv2.imwrite(str(out_dir / 'crop_rotated.png'), crop)
# draw border
vis = crop.copy()
cv2.rectangle(vis,(0,0),(vis.shape[1]-1,vis.shape[0]-1),(0,0,255),2)
cv2.imwrite(str(out_dir / 'crop_rotated_border.png'), vis)
print('crop:', crop.shape)
