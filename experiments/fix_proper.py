#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
job_dir = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
out_dir = ROOT / 'experiment_outputs' / 'fix_proper'
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
center = (w/2.0, h/2.0)

# compute angle of top edge
vec = src[1] - src[0]
angle = np.degrees(np.arctan2(vec[1], vec[0]))
print('top edge angle:', angle)
# rotate counterclockwise to level? if angle positive (going down) we need to rotate counterclockwise by angle? 
# to make horizontal, rotate by -angle
rot_angle = -angle  # degrees
print('rotating by:', rot_angle)

M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
cos = abs(M[0,0]); sin = abs(M[0,1])
nw = int((h*sin) + (w*cos))
nh = int((h*cos) + (w*sin))
M[0,2] += (nw/2) - center[0]
M[1,2] += (nh/2) - center[1]
rotated = cv2.warpAffine(orig, M, (nw,nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
src_r = cv2.transform(src.reshape(1,-1,2), M).reshape(4,2)
src_r = order_points(src_r)
print('after rot:', src_r.tolist())
cv2.imwrite(str(out_dir / 'rotated.png'), rotated)

# now make perfect rectangle - stretch to equal top/bottom width
top_w = np.linalg.norm(src_r[1]-src_r[0])
bottom_w = np.linalg.norm(src_r[2]-src_r[3])
left_h = np.linalg.norm(src_r[3]-src_r[0])
right_h = np.linalg.norm(src_r[2]-src_r[1])
target_w = max(top_w, bottom_w)
target_h = max(left_h, right_h)
margin = 15
dst = np.array([
    [margin, margin],
    [margin + target_w, margin],
    [margin + target_w, margin + target_h],
    [margin, margin + target_h]
], dtype=np.float32)
Mp = cv2.getPerspectiveTransform(src_r, dst)
final = cv2.warpPerspective(rotated, Mp, (int(target_w+2*margin), int(target_h+2*margin)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(out_dir / 'final.png'), final)
# draw border
vis = final.copy()
cv2.rectangle(vis,(0,0),(vis.shape[1]-1,vis.shape[0]-1),(0,0,255),3)
cv2.imwrite(str(out_dir / 'final_border.png'), vis)
print('final:', final.shape)
