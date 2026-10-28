#!/usr/bin/env python
"""Нахождение ИСТИННОГО верхнего левого угла таблицы через первую строку сетки."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'true_top_left'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']

src_poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)
margin = rect_info.get('margin', 16)
body_size = rect_info.get('body_size', [503, 840])

# Rectification homography
dst_tl = np.array([margin, margin], np.float32)
dst_tr = np.array([margin + body_size[0], margin], np.float32)
dst_br = np.array([margin + body_size[0], margin + body_size[1]], np.float32)
dst_bl = np.array([margin, margin + body_size[1]], np.float32)
H = cv2.getPerspectiveTransform(src_poly, np.array([dst_tl, dst_tr, dst_br, dst_bl], np.float32))
H_inv = np.linalg.inv(H)

# Table top in rectified = first row boundary
body_top = layout['body_top']  # 16
body_left = layout['body_left']  # 16
table_top_y_rect = layout['row_boundaries'][0]  # 17
table_left_x_rect = body_left  # 16

# Map to original
def map_back(x, y):
    p = np.array([x, y, 1.0], np.float32)
    q = H_inv @ p
    return (q[:2] / q[2]).astype(np.float32)

true_tl = map_back(table_left_x_rect, table_top_y_rect)
true_tr = map_back(layout['right'], table_top_y_rect)
# Bottom corners from polygon (they match)
true_br = src_poly[2]
true_bl = src_poly[3]

src = np.array([true_tl, true_tr, true_br, true_bl], dtype=np.float32)
print('TRUE table corners (top from first grid line):')
for i, pt in enumerate(src):
    print(f'  {i}: [{pt[0]:.1f}, {pt[1]:.1f}]')

cv2.imwrite(str(OUT / 'original.png'), orig)

# Draw
vis = orig.copy()
for i, pt in enumerate(src):
    cv2.circle(vis, tuple(np.int32(pt)), 8, (0,0,255), -1)
    cv2.putText(vis, f'{i}', tuple(np.int32(pt)+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.polylines(vis, [np.int32(src)], True, (0,255,255), 2)
cv2.imwrite(str(OUT / 'corners.png'), vis)

# Also draw polygon for comparison
vis2 = orig.copy()
for i, pt in enumerate(src_poly):
    cv2.circle(vis2, tuple(np.int32(pt)), 6, (255,0,0), -1)
cv2.polylines(vis2, [np.int32(src_poly)], True, (255,0,0), 1)
cv2.imwrite(str(OUT / 'polygon_vs_true.png'), vis2)

# Rectify with TRUE corners
width = int((np.linalg.norm(src[1]-src[0]) + np.linalg.norm(src[2]-src[3]))/2)
height = int((np.linalg.norm(src[3]-src[0]) + np.linalg.norm(src[2]-src[1]))/2)
margin2 = 20
dst = np.array([[margin2,margin2],[margin2+width,margin2],[margin2+width,margin2+height],[margin2,margin2+height]], np.float32)
M = cv2.getPerspectiveTransform(src, dst)
rect = cv2.warpPerspective(orig, M, (width+2*margin2, height+2*margin2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rectified_true_table.png'), rect)

curr = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'current_rectified.png'), curr)

print('Done:', OUT)