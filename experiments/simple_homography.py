#!/usr/bin/env python
"""Простая гомография 4 угла → прямоугольник для трапеции."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'simple_homography'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)
# source_body_corners уже в порядке tl,tr,br,bl
src = poly.copy()
cv2.imwrite(str(OUT / 'original.png'), orig)

# Draw 4 corners on original
vis = orig.copy()
for i, pt in enumerate(src):
    cv2.circle(vis, tuple(np.int32(pt)), 8, (0,0,255), -1)
    cv2.putText(vis, f'{i}', tuple(np.int32(pt)+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.polylines(vis, [np.int32(src)], True, (255,0,0), 3)
cv2.imwrite(str(OUT / 'corners.png'), vis)

# Target rectangle
width = int((np.linalg.norm(src[1]-src[0]) + np.linalg.norm(src[2]-src[3]))/2)
height = int((np.linalg.norm(src[3]-src[0]) + np.linalg.norm(src[2]-src[1]))/2)
margin = 20
dst = np.array([[margin,margin],[margin+width,margin],[margin+width,margin+height],[margin,margin+height]], np.float32)

M = cv2.getPerspectiveTransform(src, dst)
rect = cv2.warpPerspective(orig, M, (width+2*margin, height+2*margin), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rectified_simple.png'), rect)

# Compare with current rectified
curr = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'current_rectified.png'), curr)

print('Simple homography done:', OUT)
print('src:', src.tolist())
print('dst size:', width, height)