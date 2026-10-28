#!/usr/bin/env python
"""Визуальный выбор правильной точки 0 — несколько кандидатов на фото."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'pick_point0'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']

src_poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)
margin = rect_info.get('margin', 16)
body_size = rect_info.get('body_size', [503, 840])

# Homography
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

# Candidates for true top-left
candidates = []

# 1. Polygon top-left (detector)
candidates.append(('polygon_tl', src_poly[0]))

# 2. First grid line (row_boundaries[0]=17)
candidates.append(('grid_row0', map_back(layout['body_left'], layout['row_boundaries'][0])))

# 3. Second grid line (row_boundaries[1]=59) - first data row?
candidates.append(('grid_row1', map_back(layout['body_left'], layout['row_boundaries'][1])))

# 4. body_top in rectified
candidates.append(('body_top', map_back(layout['body_left'], layout['body_top'])))

# 5. Intersection of left edge (src_poly[0]->src_poly[3]) at y of src_poly[1] (make top horizontal)
y_top = src_poly[1,1]
x_top_left = src_poly[0,0] + (src_poly[3,0] - src_poly[0,0]) * (y_top - src_poly[0,1]) / (src_poly[3,1] - src_poly[0,1])
candidates.append(('horizontal_top', np.array([x_top_left, y_top], dtype=np.float32)))

# 6. Third grid line (row_boundaries[2]=107.75)
candidates.append(('grid_row2', map_back(layout['body_left'], layout['row_boundaries'][2])))

print('Кандидаты для точки 0 (верхний левый угол таблицы):')
for name, pt in candidates:
    print(f'  {name}: [{pt[0]:.1f}, {pt[1]:.1f}]')

# Draw all on original
vis = orig.copy()
colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (255,0,255), (0,255,255)]
for idx, (name, pt) in enumerate(candidates):
    cv2.circle(vis, tuple(np.int32(pt)), 10, colors[idx % len(colors)], -1)
    cv2.putText(vis, name, tuple(np.int32(pt)+[5,5]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx % len(colors)], 2)

cv2.imwrite(str(OUT / 'candidates.png'), vis)

# Also draw each candidate separately with rectified result
for name, pt0 in candidates:
    # Build rectification with this top-left
    pt1 = map_back(layout['right'], layout['row_boundaries'][0]) if 'grid' in name else src_poly[1]
    pt2 = src_poly[2]
    pt3 = src_poly[3]
    
    src = np.array([pt0, pt1, pt2, pt3], dtype=np.float32)
    width = int((np.linalg.norm(src[1]-src[0]) + np.linalg.norm(src[2]-src[3]))/2)
    height = int((np.linalg.norm(src[3]-src[0]) + np.linalg.norm(src[2]-src[1]))/2)
    margin2 = 20
    dst = np.array([[margin2,margin2],[margin2+width,margin2],[margin2+width,margin2+height],[margin2,margin2+height]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    rect = cv2.warpPerspective(orig, M, (width+2*margin2, height+2*margin2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    cv2.imwrite(str(OUT / f'rectified_{name}.png'), rect)

print('Done:', OUT)