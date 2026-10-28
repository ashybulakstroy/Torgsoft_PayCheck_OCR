#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
job_dir = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
out_dir = ROOT / 'experiment_outputs' / 'rotate_fix'
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
center = np.array([w/2, h/2], dtype=np.float32)

# Try rotations 0,1,2,3,4,5 degrees counterclockwise
for deg in [0, 1, 2, 3, 4, 5]:
    M = cv2.getRotationMatrix2D(tuple(center), -deg, 1.0)  # negative = counterclockwise in some coords, but opencv: positive is counterclockwise? check
    # OpenCV: angle is in degrees, positive = counterclockwise
    M = cv2.getRotationMatrix2D(tuple(center), deg, 1.0) if False else cv2.getRotationMatrix2D(tuple(center), -deg, 1.0) # try
    # actually opencv rotates counterclockwise for positive angle
    M = cv2.getRotationMatrix2D(tuple(center), -deg, 1.0) # -deg means clockwise? no wait - standard is positive CCW
    M = cv2.getRotationMatrix2D(tuple(center), -deg, 1.0) if deg > 0 else cv2.getRotationMatrix2D(tuple(center), 0, 1.0)
    # better: try both
    pass

# Try -2, -3, -4 degrees (counterclockwise)
for deg in [-2, -3, -4, -2.5, -3.5]:
    M = cv2.getRotationMatrix2D(tuple(center), deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nw = int((h*sin) + (w*cos))
    nh = int((h*cos) + (w*sin))
    M[0,2] += (nw/2) - center[0]
    M[1,2] += (nh/2) - center[1]
    rotated = cv2.warpAffine(orig, M, (nw,nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    # crop by body
    src_rot = cv2.transform(src.reshape(1,-1,2), M).reshape(4,2)
    src_rot = order_points(src_rot)
    # perspective fix
    top_w = np.linalg.norm(src_rot[1]-src_rot[0])
    bottom_w = np.linalg.norm(src_rot[2]-src_rot[3])
    target_w = bottom_w
    target_h = max(np.linalg.norm(src_rot[3]-src_rot[0]), np.linalg.norm(src_rot[2]-src_rot[1]))
    margin = 20
    dst = np.array([[margin,margin],[margin+target_w,margin],[margin+target_w,margin+target_h],[margin,margin+target_h]], dtype=np.float32)
    Mp = cv2.getPerspectiveTransform(src_rot,dst)
    out_img = cv2.warpPerspective(rotated, Mp, (int(target_w+2*margin), int(target_h+2*margin)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    cv2.imwrite(str(out_dir / f'rot{deg}.png'), out_img)
    print(deg, out_img.shape)
