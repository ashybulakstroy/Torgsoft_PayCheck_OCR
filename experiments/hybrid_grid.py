#!/usr/bin/env python
"""Гибрид: горизонтали из layout.json (Current), вертикали из Wave Dewarp."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'hybrid_grid'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))

from App.table_boundary import detect_table_boundary
from App.wave_dewarp import rectify, ordered_axis_map

b = detect_table_boundary(orig)
poly = b['polygon']

# Get Wave vertical levels
output_wave, mx_wave, my_wave, wave, _ = rectify(orig, poly)
v_levels = wave['vertical_levels']

# Get horizontal levels from layout (Current Coons)
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']
h_levels = np.array(layout['row_boundaries'], dtype=np.float32)  # 14 levels

print('Hybrid: horizontal from layout, vertical from Wave')
print('h_levels:', h_levels)
print('v_levels:', v_levels)

# Build hybrid maps using ordered_axis_map logic but with our levels
margin = 20
height, width = 840 + 2*margin, 503 + 2*margin  # from rectification.json body_size
# Actually need crop size from wave
h, w = output_wave.shape[:2]
print(f'Wave output size: {w}x{h}')

# We need to rebuild mx, my using hybrid levels
# ordered_axis_map takes (rules, length, along) where rules have .values curves
# Instead, let's directly construct the interpolation

# Horizontal map: for each y in [0..h), x = interp from vertical levels
# Vertical map: for each x in [0..w), y = interp from horizontal levels

# Destination coordinates
dst_y = np.arange(h, dtype=float)
dst_x = np.arange(w, dtype=float)

# Horizontal interpolation: my[y, x] = interp of v_levels at y
# v_levels are x-coordinates of vertical lines at each horizontal level
# Actually v_levels are x positions at regular y intervals in rectified space
# Wait, v_levels are median x positions of vertical rules across the image
# They don't vary with y in rectified space (they're straight)

# For hybrid:
# - Horizontal lines (my) should follow layout row_boundaries (straight horizontal)
# - Vertical lines (mx) should follow wave vertical levels (straight vertical)

# Simple approach: my = identity in y (straight horizontals), mx = identity in x (straight verticals)
# But we need to map from rectified back to original

# Better: use the original wave maps but replace horizontal interpolation with layout levels
# The wave maps are backward maps: mx[y_rect, x_rect] = x_orig, my[y_rect, x_rect] = y_orig

# Let's create new maps:
# For each rectified pixel (y, x):
#   my_new[y, x] = my_wave[y, x]  # keep vertical coordinate from wave
#   mx_new[y, x] = mx_wave[y, x]  # keep horizontal coordinate from wave
# But force horizontals to be straight by adjusting my? No...

# Actually the issue is: Wave my has curved horizontals (0.75px error), Current my has straight horizontals.
# Wave mx has straight verticals, Current mx has curved verticals.

# Hybrid: use Wave mx (good verticals), Current my (good horizontals)
# But they're in different coordinate systems (different rectifications).

# Let's try: rectify with simple homography, then use layout row_boundaries as horizontal grid,
# and wave vertical levels as vertical grid.

# Build simple homography first
poly_simple = poly.astype(np.float32)
simple_width = int((np.linalg.norm(poly_simple[1]-poly_simple[0]) + np.linalg.norm(poly_simple[2]-poly_simple[3]))/2)
simple_height = int((np.linalg.norm(poly_simple[3]-poly_simple[0]) + np.linalg.norm(poly_simple[2]-poly_simple[1]))/2)
margin = 20
dst = np.array([[margin,margin],[margin+simple_width,margin],[margin+simple_width,margin+simple_height],[margin,margin+simple_height]], np.float32)
M = cv2.getPerspectiveTransform(poly_simple, dst)
simple = cv2.warpPerspective(orig, M, (simple_width+2*margin, simple_height+2*margin), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

# On simple rectified, use layout row_boundaries (they're in Current rectified coords)
# Need to map row_boundaries from Current rectified to Simple rectified
# Both are rectifications of same polygon - they should be similar but not identical

# Let's just use the layout row_boundaries directly on the simple rectified (assuming same scale)
# and wave v_levels on simple rectified (need to map)

# For now, let's do a simpler hybrid: 
# 1. Take wave rectified output
# 2. Force horizontal lines to be perfectly straight at layout row_boundaries
# by adjusting my map

# In wave rectified space, we know the row_boundaries from layout
# We can create a corrected my where my[y, x] is interpolated from layout boundaries

# But wave rectified and Current rectified have different coordinate systems.
# Let's work in original space with backward maps.

# The backward maps mx, my map from rectified -> original
# Current rectified uses Coons patch with body_size, margin
# Wave rectified uses different output size

# Easiest: just visualize the hybrid by drawing both grids on wave output
# and see if manual hybrid looks good.

# Let's try: create a rectification where:
# - Horizontal grid lines = layout row_boundaries (perfectly straight)
# - Vertical grid lines = wave v_levels (perfectly straight)
# This is just a grid rectification!

# Create a grid in rectified space with straight lines at those positions
# Then map back to original using... we need the forward map.

# Actually, the simplest: 
# Take the simple homography rectified image
# Draw grid at layout row_boundaries (horizontal) and wave v_levels mapped to simple space
# If they align well, we can use simple + grid for OCR

# Let's map v_levels from wave rectified to simple rectified
# Both are rectifications of the same polygon
# Wave rectified size: output_wave.shape
# Simple rectified size: simple.shape

wave_h, wave_w = output_wave.shape[:2]
simple_h, simple_w = simple.shape[:2]

print(f'Wave: {wave_w}x{wave_h}, Simple: {simple_w}x{simple_h}')

# Map wave v_levels (x in wave rectified) to simple rectified
# They should correspond to same physical vertical lines
# Scale factor
scale_x = simple_w / wave_w
v_levels_simple = np.array(v_levels) * scale_x

# Layout row_boundaries are in Current rectified space
# Current rectified from rectification.json
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
curr_margin = rect_info.get('margin', 16)
curr_body_size = rect_info.get('body_size', [503, 840])
curr_w = curr_body_size[0] + 2*curr_margin
curr_h = curr_body_size[1] + 2*curr_margin

# Map row_boundaries from Current to Simple
scale_y = simple_h / curr_h
scale_x2 = simple_w / curr_w
row_boundaries_simple = np.array(layout['row_boundaries']) * scale_y

print(f'Scale factors: x={scale_x:.3f}, x2={scale_x2:.3f}, y={scale_y:.3f}')
print(f'v_levels_simple: {v_levels_simple}')
print(f'row_boundaries_simple: {row_boundaries_simple}')

# Draw on simple rectified
vis_simple = simple.copy()
for y in row_boundaries_simple:
    cv2.line(vis_simple, (0, int(y)), (simple_w, int(y)), (0, 255, 0), 1)
for x in v_levels_simple:
    cv2.line(vis_simple, (int(x), 0), (int(x), simple_h), (255, 0, 0), 1)
cv2.imwrite(str(OUT / 'simple_hybrid_grid.png'), vis_simple)

# Draw on wave rectified for comparison
vis_wave = output_wave.copy()
for y in h_levels:
    cv2.line(vis_wave, (0, int(y)), (wave_w, int(y)), (0, 255, 0), 1)
for x in v_levels:
    cv2.line(vis_wave, (int(x), 0), (int(x), wave_h), (255, 0, 0), 1)
cv2.imwrite(str(OUT / 'wave_original_grid.png'), vis_wave)

# Draw on current rectified
curr = cv2.imread(str(JOB / 'rectified.png'))
vis_curr = curr.copy()
for y in layout['row_boundaries']:
    cv2.line(vis_curr, (0, int(y)), (curr.shape[1], int(y)), (0, 255, 0), 1)
for x in v_levels:
    cv2.line(vis_curr, (int(x), 0), (int(x), curr.shape[0]), (255, 0, 0), 1)
cv2.imwrite(str(OUT / 'current_wave_vertical_grid.png'), vis_curr)

print('Done:', OUT)