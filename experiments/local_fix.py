#!/usr/bin/env python
"""Локальная коррекция map_y и map_x в проблемной ячейке (rows 10-11, col 2)."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'local_fix'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))

from App.table_boundary import detect_table_boundary
from App.wave_dewarp import rectify

b = detect_table_boundary(orig)
poly = b['polygon']

output, mx, my, wave, diag = rectify(orig, poly)
print('Wave status:', wave.get('status'))

# Layout row boundaries
import json
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']
row_boundaries = layout['row_boundaries']
v_levels = wave['vertical_levels']

col2_x_start = v_levels[1]
col2_x_end = v_levels[2]
row10_y = row_boundaries[10]
row11_y = row_boundaries[11]

y_start = int(row10_y)
y_end = int(row11_y)
x_start = int(col2_x_start)
x_end = int(col2_x_end)

cell_my = my[y_start:y_end, x_start:x_end]
cell_mx = mx[y_start:y_end, x_start:x_end]

print(f'Cell my shape: {cell_my.shape}')
print(f'Cell mx shape: {cell_mx.shape}')

# Check mx diffs across x (horizontal smoothness) for middle row
mid_y = cell_mx.shape[0] // 2
row = cell_mx[mid_y, :]
diffs = np.diff(row)
print(f'mx diffs at mid-y: min={diffs.min():.3f}, max={diffs.max():.3f}, mean={diffs.mean():.3f}')

# Check my diffs down y for left/center/right x
for x_rel in [0, cell_my.shape[1]//2, cell_my.shape[1]-1]:
    col = cell_my[:, x_rel]
    diffs = np.diff(col)
    print(f'  my x={x_rel}: diffs min={diffs.min():.3f}, max={diffs.max():.3f}, mean={diffs.mean():.3f}')

# Simple manual smoothing of both maps in problematic region
my_smooth = my.copy()
mx_smooth = mx.copy()
for y in range(y_start+1, y_end-1):
    for x in range(x_start+1, x_end-1):
        my_smooth[y, x] = (my[y-1,x] + my[y+1,x] + my[y,x-1] + my[y,x+1]) * 0.25
        mx_smooth[y, x] = (mx[y-1,x] + mx[y+1,x] + mx[y,x-1] + mx[y,x+1]) * 0.25

fixed = cv2.remap(orig, mx_smooth.astype(np.float32), my_smooth.astype(np.float32), cv2.INTER_CUBIC,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'wave_local_fix.png'), fixed)
cv2.imwrite(str(OUT / 'wave_original.png'), output)

curr = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'current_rectified.png'), curr)

print('Done:', OUT)