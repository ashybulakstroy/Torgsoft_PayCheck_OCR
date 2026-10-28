#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
job_dir = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
out_dir = ROOT / 'experiment_outputs' / 'fix_trapezoid_subtle'
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
table_center = np.mean(src, axis=0)
h,w = orig.shape[:2]
rot_angle = 2.5
M = cv2.getRotationMatrix2D(tuple(table_center), rot_angle, 1.0)
rotated = cv2.warpAffine(orig, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
src_r = cv2.transform(src.reshape(1,-1,2), M).reshape(4,2)
src_r = order_points(src_r)

# Very subtle - only scale top edge slightly
# Don't force perfect rectangle, just equalize
top_w = np.linalg.norm(src_r[1]-src_r[0])
bottom_w = np.linalg.norm(src_r[2]-src_r[3])
diff = bottom_w - top_w
# scale top by small factor
scale = 1.0 + (diff * 0.3 / top_w)  # only 30% correction
# build custom dst - keep bottom same, adjust top
margin = 10
target_h = max(np.linalg.norm(src_r[3]-src_r[0]), np.linalg.norm(src_r[2]-src_r[1]))
dst = src_r.copy()
dst[0] = src_r[0] * np.array([1,1]) + (src_r[1]-src_r[0])*(1-scale)*0.5  # adjust
dst[1] = src_r[1] - (src_r[1]-src_r[0])*(1-scale)*0.5
# better approach - keep verticals mostly vertical
center_x = (src_r[0,0]+src_r[1,0]+src_r[2,0]+src_r[3,0])/4
dst2 = np.array([
    [src_r[0,0] + (src_r[0,0]-center_x)*(scale-1)*0.3, src_r[0,1]],
    [src_r[1,0] + (src_r[1,0]-center_x)*(scale-1)*0.3, src_r[1,1]],
    [src_r[2,0], src_r[2,1]],
    [src_r[3,0], src_r[3,1]],
], dtype=np.float32)
Mp = cv2.getPerspectiveTransform(src_r, dst2)
# get bbox
corners = cv2.transform(src_r.reshape(1,-1,2), Mp).reshape(4,2)
x1,y1=corners.min(0); x2,y2=corners.max(0)
final = cv2.warpPerspective(rotated, Mp, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
crop = final[max(0,int(y1)-margin):min(h,int(y2)+margin), max(0,int(x1)-margin):min(w,int(x2)+margin)]
cv2.imwrite(str(out_dir / 'crop.png'), crop)
vis = crop.copy()
cv2.rectangle(vis,(0,0),(vis.shape[1]-1,vis.shape[0]-1),(0,0,255),2)
cv2.imwrite(str(out_dir / 'crop_border.png'), vis)
print(crop.shape)
