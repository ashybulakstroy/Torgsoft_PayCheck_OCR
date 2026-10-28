#!/usr/bin/env python
"""Полная сетка: оба верхних угла из layout.json (grid boundaries)."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'full_grid_corners'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']

src_poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)
margin = rect_info.get('margin', 16)
body_size = rect_info.get('body_size', [503, 840])

# Homography: source_body_corners -> rectified corners
dst_tl = np.array([margin, margin], np.float32)
dst_tr = np.array([margin + body_size[0], margin], np.float32)
dst_br = np.array([margin + body_size[0], margin + body_size[1]], np.float32)
dst_bl = np.array([margin, margin + body_size[1]], np.float32)
H = cv2.getPerspectiveTransform(src_poly, np.array([dst_tl, dst_tr, dst_br, dst_bl], np.float32))
H_inv = np.linalg.inv(H)

def map_back(x, y):
    p = np.array([x, y, 1.0], np.float32)
    q = H_inv @ p
    return (q[:2] / q[2]).astype(np.float32)

# ALL 4 corners from grid in rectified -> map back to original
grid_tl = map_back(layout['body_left'], layout['row_boundaries'][0])
grid_tr = map_back(layout['right'], layout['row_boundaries'][0])
grid_br = map_back(layout['right'], layout['row_boundaries'][-1])
grid_bl = map_back(layout['body_left'], layout['row_boundaries'][-1])

src = np.array([grid_tl, grid_tr, grid_br, grid_bl], dtype=np.float32)
print('Grid corners (all 4 from layout):')
for i, pt in enumerate(src):
    print(f'  {i}: [{pt[0]:.1f}, {pt[1]:.1f}]')

cv2.imwrite(str(OUT / 'original.png'), orig)

vis = orig.copy()
for i, pt in enumerate(src):
    cv2.circle(vis, tuple(np.int32(pt)), 10, (0,0,255), -1)
    cv2.putText(vis, f'{i}', tuple(np.int32(pt)+[5,5]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
cv2.polylines(vis, [np.int32(src)], True, (0,255,255), 2)
cv2.imwrite(str(OUT / 'corners.png'), vis)

# Rectify
width = int((np.linalg.norm(src[1]-src[0]) + np.linalg.norm(src[2]-src[3]))/2)
height = int((np.linalg.norm(src[3]-src[0]) + np.linalg.norm(src[2]-src[1]))/2)
margin2 = 20
dst = np.array([[margin2,margin2],[margin2+width,margin2],[margin2+width,margin2+height],[margin2,margin2+height]], np.float32)
M = cv2.getPerspectiveTransform(src, dst)
rect = cv2.warpPerspective(orig, M, (width+2*margin2, height+2*margin2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rectified_grid_all4.png'), rect)

curr = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'current_rectified.png'), curr)

print('Done:', OUT)