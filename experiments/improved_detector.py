#!/usr/bin/env python
"""Улучшенный детектор: находит верний верхний левый угол таблицы через сетку в rectified."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'improved_detector'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))

# Current polygon from detector
src_poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)

# Rectify using current polygon
margin = rect_info.get('margin', 16)
body_size = rect_info.get('body_size', [503, 840])
rect_w, rect_h = body_size[0] + 2*margin, body_size[1] + 2*margin
dst_tl = np.array([margin, margin], np.float32)
dst_tr = np.array([margin + body_size[0], margin], np.float32)
dst_br = np.array([margin + body_size[0], margin + body_size[1]], np.float32)
dst_bl = np.array([margin, margin + body_size[1]], np.float32)

H = cv2.getPerspectiveTransform(src_poly, np.array([dst_tl, dst_tr, dst_br, dst_bl], np.float32))
rectified = cv2.warpPerspective(orig, H, (rect_w, rect_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rectified.png'), rectified)

# Use layout.json row boundaries (already detected well)
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']
line_ys = layout['row_boundaries']
print('Row boundaries from layout:', line_ys)

# First row boundary = table top (in rectified coords, relative to body_top)
body_top = layout['body_top']  # 16
table_top_y = line_ys[0] - body_top  # relative to roi
print(f'Table top at y={table_top_y} (in roi)')

# Left edge x in rectified roi = body_left - margin
body_left = layout['body_left']  # 16
table_left_x = body_left - margin  # 0

# Map back to original
H_inv = np.linalg.inv(H)
def map_back(pt, in_roi=True):
    x, y = pt
    if in_roi:
        x += margin
        y += margin
    p = np.array([x, y, 1.0], np.float32)
    q = H_inv @ p
    return (q[:2] / q[2]).astype(np.float32)

# Top-left corner of table
true_tl = map_back([table_left_x, table_top_y])
# Top-right: right edge at same y
true_tr = map_back([layout['right'] - margin, table_top_y])
# Bottom corners from original polygon
true_br = src_poly[2]
true_bl = src_poly[3]

src = np.array([true_tl, true_tr, true_br, true_bl], dtype=np.float32)
print('Improved corners:')
for i, pt in enumerate(src):
    print(f'  {i}: [{pt[0]:.1f}, {pt[1]:.1f}]')

# Draw on original
vis = orig.copy()
for i, pt in enumerate(src):
    cv2.circle(vis, tuple(np.int32(pt)), 8, (0,0,255), -1)
    cv2.putText(vis, f'{i}', tuple(np.int32(pt)+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.polylines(vis, [np.int32(src)], True, (0,255,255), 2)
cv2.imwrite(str(OUT / 'corners.png'), vis)

# Rectify with improved corners
width = int((np.linalg.norm(src[1]-src[0]) + np.linalg.norm(src[2]-src[3]))/2)
height = int((np.linalg.norm(src[3]-src[0]) + np.linalg.norm(src[2]-src[1]))/2)
margin2 = 20
dst = np.array([[margin2,margin2],[margin2+width,margin2],[margin2+width,margin2+height],[margin2,margin2+height]], np.float32)
M = cv2.getPerspectiveTransform(src, dst)
rect = cv2.warpPerspective(orig, M, (width+2*margin2, height+2*margin2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rectified_improved.png'), rect)

curr = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'current_rectified.png'), curr)
print('Done:', OUT)