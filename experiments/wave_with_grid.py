#!/usr/bin/env python
"""Wave Dewarp с наложенной сеткой ячеек."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'wave_with_grid'
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

# Draw grid on wave result
vis = output.copy()
for y in row_boundaries:
    cv2.line(vis, (0, int(y)), (vis.shape[1], int(y)), (0, 255, 0), 1)
for x in v_levels:
    cv2.line(vis, (int(x), 0), (int(x), vis.shape[0]), (255, 0, 0), 1)

# Highlight problematic cell (rows 10-11, col 2)
row10_y = int(row_boundaries[10])
row11_y = int(row_boundaries[11])
col2_x_start = int(v_levels[1])
col2_x_end = int(v_levels[2])
cv2.rectangle(vis, (col2_x_start, row10_y), (col2_x_end, row11_y), (0, 0, 255), 2)

cv2.imwrite(str(OUT / 'wave_with_grid.png'), vis)

# Also zoom on problematic cell
cell = output[row10_y:row11_y, col2_x_start:col2_x_end]
zoom = cv2.resize(cell, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(OUT / 'cell_zoom6.png'), zoom)

# Draw grid on zoom
zoom_vis = zoom.copy()
h, w = zoom.shape[:2]
# Map grid lines to zoom coordinates
for y in row_boundaries:
    cv2.line(zoom_vis, (0, int((y - row10_y) * 6)), (w, int((y - row10_y) * 6)), (0, 255, 0), 1)
for x in v_levels:
    cv2.line(zoom_vis, (int((x - col2_x_start) * 6), 0), (int((x - col2_x_start) * 6), h), (255, 0, 0), 1)
cv2.imwrite(str(OUT / 'cell_zoom6_grid.png'), zoom_vis)

# Full with grid
cv2.imwrite(str(OUT / 'wave_full_grid.png'), vis)

# Current for comparison
curr = cv2.imread(str(JOB / 'rectified.png'))
vis_curr = curr.copy()
for y in row_boundaries:
    cv2.line(vis_curr, (0, int(y)), (vis_curr.shape[1], int(y)), (0, 255, 0), 1)
for x in v_levels:
    cv2.line(vis_curr, (int(x), 0), (int(x), vis_curr.shape[0]), (255, 0, 0), 1)
cv2.rectangle(vis_curr, (col2_x_start, row10_y), (col2_x_end, row11_y), (0, 0, 255), 2)
cv2.imwrite(str(OUT / 'current_with_grid.png'), vis_curr)

print('Done:', OUT)