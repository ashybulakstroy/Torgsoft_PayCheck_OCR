#!/usr/bin/env python
"""Выпрямление только тела таблицы (без шапки) простой гомографией."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'table_body_homography'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)

# Table body corners = polygon corners (только тело, без шапки)
# poly order: tl, tr, br, bl
poly = poly.astype(np.float32)
tl_poly, tr_poly, br_poly, bl_poly = poly

# Make top edge horizontal at y of tr (point 1)
y_top = tr_poly[1]
# Interpolate left edge (tl->bl) at y_top
def interp_x_at_y(p1, p2, y_target):
    x1, y1 = p1; x2, y2 = p2
    if abs(y2 - y1) < 1e-6:
        return x1
    t = (y_target - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

left_top_x = interp_x_at_y(tl_poly, bl_poly, y_top)

src = np.array([
    [left_top_x, y_top],
    [tr_poly[0], tr_poly[1]],
    [br_poly[0], br_poly[1]],
    [bl_poly[0], bl_poly[1]],
], dtype=np.float32)

print('Table body corners (top horizontal):')
for i, pt in enumerate(src):
    print(f'  {i}: {pt}')

cv2.imwrite(str(OUT / 'original.png'), orig)

# Draw corners (only table body)
vis = orig.copy()
for i, pt in enumerate(src):
    cv2.circle(vis, tuple(np.int32(pt)), 8, (0,0,255), -1)
    cv2.putText(vis, f'{i}', tuple(np.int32(pt)+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.polylines(vis, [np.int32(src)], True, (0,255,255), 2)  # yellow = table body
cv2.imwrite(str(OUT / 'corners.png'), vis)

# Target rectangle
width = int((np.linalg.norm(src[1]-src[0]) + np.linalg.norm(src[2]-src[3]))/2)
height = int((np.linalg.norm(src[3]-src[0]) + np.linalg.norm(src[2]-src[1]))/2)
margin = 20
dst = np.array([[margin,margin],[margin+width,margin],[margin+width,margin+height],[margin,margin+height]], np.float32)

M = cv2.getPerspectiveTransform(src, dst)
rect = cv2.warpPerspective(orig, M, (width+2*margin, height+2*margin), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rectified_table_body.png'), rect)

# Compare with current rectified
curr = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'current_rectified.png'), curr)

print('Done:', OUT)