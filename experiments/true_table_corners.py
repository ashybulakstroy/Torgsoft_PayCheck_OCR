#!/usr/bin/env python
"""Находка истинных углов таблицы через обратную гомографию из rectified grid."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'true_table_corners'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']

# Source corners used for rectification (source_body_corners)
src_poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)
# Order: tl, tr, br, bl
src_tl, src_tr, src_br, src_bl = src_poly

# Rectified image size and margin
margin = rect_info.get('margin', 16)
body_size = rect_info.get('body_size', [503, 840])
rect_w, rect_h = body_size[0] + 2*margin, body_size[1] + 2*margin

# Dest corners in rectified image (with margin)
dst_tl = np.array([margin, margin], np.float32)
dst_tr = np.array([margin + body_size[0], margin], np.float32)
dst_br = np.array([margin + body_size[0], margin + body_size[1]], np.float32)
dst_bl = np.array([margin, margin + body_size[1]], np.float32)

# Table grid bounds in rectified coords (from layout)
body_left = layout['body_left']   # 16
body_top = layout['body_top']     # 16
body_right = layout['right']      # 519
body_bottom = layout['body_bottom'] # 856

# Table corners in rectified coords
tbl_tl_r = np.array([body_left, body_top], np.float32)
tbl_tr_r = np.array([body_right, body_top], np.float32)
tbl_br_r = np.array([body_right, body_bottom], np.float32)
tbl_bl_r = np.array([body_left, body_bottom], np.float32)

# Homography: source_body_corners -> rectified corners (with margin)
H = cv2.getPerspectiveTransform(src_poly, np.array([dst_tl, dst_tr, dst_br, dst_bl], np.float32))
H_inv = np.linalg.inv(H)

# Map table corners back to original
def map_back(pt):
    p = np.array([pt[0], pt[1], 1.0], np.float32)
    q = H_inv @ p
    return (q[:2] / q[2]).astype(np.float32)

tbl_tl = map_back(tbl_tl_r)
tbl_tr = map_back(tbl_tr_r)
tbl_br = map_back(tbl_br_r)
tbl_bl = map_back(tbl_bl_r)

src = np.array([tbl_tl, tbl_tr, tbl_br, tbl_bl], dtype=np.float32)

print('True table corners (from grid back-projection):')
for i, pt in enumerate(src):
    print(f'  {i}: [{pt[0]:.1f}, {pt[1]:.1f}]')

cv2.imwrite(str(OUT / 'original.png'), orig)

# Draw on original
vis = orig.copy()
for i, pt in enumerate(src):
    cv2.circle(vis, tuple(np.int32(pt)), 8, (0,0,255), -1)
    cv2.putText(vis, f'{i}', tuple(np.int32(pt)+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.polylines(vis, [np.int32(src)], True, (0,255,255), 2)
cv2.imwrite(str(OUT / 'corners.png'), vis)

# Rectify using these true corners
width = int((np.linalg.norm(src[1]-src[0]) + np.linalg.norm(src[2]-src[3]))/2)
height = int((np.linalg.norm(src[3]-src[0]) + np.linalg.norm(src[2]-src[1]))/2)
margin2 = 20
dst = np.array([[margin2,margin2],[margin2+width,margin2],[margin2+width,margin2+height],[margin2,margin2+height]], np.float32)
M = cv2.getPerspectiveTransform(src, dst)
rect = cv2.warpPerspective(orig, M, (width+2*margin2, height+2*margin2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rectified_true_table.png'), rect)

# Compare
curr = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'current_rectified.png'), curr)

print('Done:', OUT)