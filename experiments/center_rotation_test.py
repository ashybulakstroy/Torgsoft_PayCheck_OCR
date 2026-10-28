#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import webbrowser
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = (ROOT / 'reports' / 'center_rotation' / 'index.html').resolve()
OUT.parent.mkdir(parents=True, exist_ok=True)
IMGDIR = OUT.parent

orig = cv2.imread(str(JOB / 'original.png'))
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
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
cv2.imwrite(str(IMGDIR / 'before.png'), orig)
cv2.imwrite(str(IMGDIR / 'current_rectified.png'), cv2.imread(str(JOB / 'rectified.png')))

# Rotate around table center by average edge angle
top_vec = src[1] - src[0]
bottom_vec = src[2] - src[3]
angle = float(np.degrees(np.arctan2((top_vec[1]+bottom_vec[1])/2, (top_vec[0]+bottom_vec[0])/2)))
center = np.mean(src, axis=0)
M_rot = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
cos = np.abs(M_rot[0,0]); sin = np.abs(M_rot[0,1])
nw = int((h*sin) + (w*cos))
nh = int((h*cos) + (w*sin))
M_rot[0,2] += (nw/2) - center[0]
M_rot[1,2] += (nh/2) - center[1]
rotated = cv2.warpAffine(orig, M_rot, (nw,nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
src_rot = order_points(cv2.transform(src.reshape(1,-1,2), M_rot).reshape(4,2))
cv2.imwrite(str(IMGDIR / 'rotated.png'), rotated)

html = f'''<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>Center rotation</title>
<style>body{{font-family:sans-serif;margin:20px}}.row{{display:flex;gap:20px;flex-wrap:wrap}}.col{{flex:1;min-width:300px}}img{{max-width:100%;border:1px solid #ccc}}</style></head><body>
<h1>Поворот вокруг центра таблицы (только поворот, без выпрямления)</h1>
<p>Угол поворота: {angle:.3f}°. Исходный чек: {JOB.name}</p>
<div class="row">
<div class="col"><h3>До: исходное фото</h3><img src="before.png"></div>
<div class="col"><h3>После поворота</h3><img src="rotated.png"></div>
</div>
</body></html>'''
OUT.write_text(html, encoding='utf-8')
print('report:', OUT)
webbrowser.open(OUT.as_uri())
print('angle:', angle)