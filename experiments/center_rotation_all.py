#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np
from App.table_boundary import detect_table_boundary

ROOT = Path('.')
IN = ROOT / 'data' / 'in_data'
OUT = ROOT / 'experiment_outputs' / 'center_rotation_all'
OUT.mkdir(parents=True, exist_ok=True)

def order_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

files = sorted(IN.glob('*.jp*g')) + sorted(IN.glob('*.png'))
stats = {'files_tested':0,'ok':0,'failed':0}
for f in files:
    stats['files_tested'] += 1
    img = cv2.imread(str(f))
    if img is None:
        stats['failed'] += 1
        print(f'{stats["files_tested"]}. {f.name}: FAIL (read error)')
        continue
    safe = f.stem.replace(' ','_')[:40]
    cv2.imwrite(str(OUT / f'{safe}_before.png'), img)
    b = detect_table_boundary(img)
    if b.get('status') != 'detected':
        stats['failed'] += 1
        print(f'{stats["files_tested"]}. {f.name}: FAIL boundary={b.get("status")}')
        continue
    src = order_points(np.array(b['polygon'], dtype=np.float32))
    h,w = img.shape[:2]
    top_vec = src[1]-src[0]; bottom_vec = src[2]-src[3]
    angle = float(np.degrees(np.arctan2((top_vec[1]+bottom_vec[1])/2, (top_vec[0]+bottom_vec[0])/2)))
    center = np.mean(src, axis=0)
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nw = int((h*sin)+(w*cos)); nh = int((h*cos)+(w*sin))
    M[0,2] += (nw/2)-center[0]; M[1,2] += (nh/2)-center[1]
    rotated = cv2.warpAffine(img, M, (nw,nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    cv2.imwrite(str(OUT / f'{safe}_before.png'), img)
    cv2.imwrite(str(OUT / f'{safe}_rotated.png'), rotated)
    stats['ok'] += 1
    print(f'{stats["files_tested"]}. {f.name}: OK angle={angle:.2f}')

print(json.dumps(stats, indent=2))
print('out:', OUT)