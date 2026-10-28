#!/usr/bin/env python
"""Зум на проблемную ячейку для сравнения."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'cell_zoom'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))

from App.table_boundary import detect_table_boundary
from App.wave_dewarp import rectify

b = detect_table_boundary(orig)
poly = b['polygon']

output, mx, my, wave, diag = rectify(orig, poly)

layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']
row_boundaries = layout['row_boundaries']
v_levels = wave['vertical_levels']

col2_x_start = int(v_levels[1])
col2_x_end = int(v_levels[2])
row10_y = int(row_boundaries[10])
row11_y = int(row_boundaries[11])

# Extract cell from wave result
cell_wave = output[row10_y:row11_y, col2_x_start:col2_x_end]
cv2.imwrite(str(OUT / 'cell_wave.png'), cell_wave)

# Extract from current rectified
curr = cv2.imread(str(JOB / 'rectified.png'))
cell_curr = curr[row10_y:row11_y, col2_x_start:col2_x_end]
cv2.imwrite(str(OUT / 'cell_current.png'), cell_curr)

# Extract from simple homography
margin = 20
width = int((np.linalg.norm(poly[1]-poly[0]) + np.linalg.norm(poly[2]-poly[3]))/2)
height = int((np.linalg.norm(poly[3]-poly[0]) + np.linalg.norm(poly[2]-poly[1]))/2)
dst = np.array([[margin,margin],[margin+width,margin],[margin+width,margin+height],[margin,margin+height]], np.float32)
M = cv2.getPerspectiveTransform(poly.astype(np.float32), dst)
simple = cv2.warpPerspective(orig, M, (width+2*margin, height+2*margin), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cell_simple = simple[row10_y:row11_y, col2_x_start:col2_x_end]
cv2.imwrite(str(OUT / 'cell_simple.png'), cell_simple)

# Zoomed 4x
for name, img in [('wave', cell_wave), ('current', cell_curr), ('simple', cell_simple)]:
    zoom = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / f'cell_{name}_zoom4.png'), zoom)

# Also full rectified for context
cv2.imwrite(str(OUT / 'wave_full.png'), output)
cv2.imwrite(str(OUT / 'current_full.png'), curr)
cv2.imwrite(str(OUT / 'simple_full.png'), simple)

print('Done:', OUT)