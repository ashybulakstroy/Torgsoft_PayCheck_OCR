#!/usr/bin/env python
"""Гибридная ректификация: горизонтали из layout, вертикали из Wave — прямой билд map."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'hybrid_direct'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))

from App.table_boundary import detect_table_boundary
from App.wave_dewarp import rectify as wave_rectify

b = detect_table_boundary(orig)
poly = b['polygon']

# Get wave vertical levels
_, _, _, wave, _ = wave_rectify(orig, poly)
v_levels = wave['vertical_levels']  # in wave rectified space

# Layout horizontal levels (in Current rectified space with margin)
h_levels_rect = np.array(layout['row_boundaries'], dtype=np.float32)
margin = rect_info.get('margin', 16)

# We need both in the SAME rectified space.
# Current rectified space: body_size + 2*margin
body_size = rect_info.get('body_size', [503, 840])
curr_w = body_size[0] + 2*margin
curr_h = body_size[1] + 2*margin

# Wave rectified space: different size
# Get wave output size
_, _, _, wave2, _ = wave_rectify(orig, poly)
wave_h, wave_w = wave2['horizontal_levels'], wave2['vertical_levels']  # wait, these are levels not size

# Actually get wave output image size
_, _, _, wave_full, _ = wave_rectify(orig, poly)
# Need to run wave_rectify to get output image
_, mx_w, my_w, wave_full, _ = wave_rectify(orig, poly)
# wave_full is metadata, not image
# Let's just get the output image size from wave_rectify return
output_wave, mx_w, my_w, wave_meta, _ = wave_rectify(orig, poly)
wave_h, wave_w = output_wave.shape[:2]

print(f'Current rectified: {curr_w}x{curr_h}')
print(f'Wave rectified: {wave_w}x{wave_h}')

# Map v_levels from wave space to current space
scale_x = curr_w / wave_w
v_levels_curr = np.array(wave_meta['vertical_levels']) * scale_x
h_levels_curr = h_levels_rect  # already in current space

print(f'h_levels_curr: {h_levels_curr}')
print(f'v_levels_curr: {v_levels_curr}')

# Build hybrid rectification in Current space
# Create a regular grid in rectified space at these levels
# Then map back to original using... we need the backward map from Current rectified to original

# Current rectified backward map is in rectification.json
# But we don't have the map file loaded... 
# Let's load the rectification map
map_data = np.load(JOB / 'rectification_map.npz')
map_x_curr = map_data['map_x']
map_y_curr = map_data['map_y']

# These maps go from Current rectified -> original
# We can resample them at our hybrid grid points

# Hybrid grid in Current rectified space
dst_y = np.arange(curr_h, dtype=np.float32)
dst_x = np.arange(curr_w, dtype=np.float32)

# For each destination pixel, we want to know where it comes from in original
# Current maps give us this for regular grid
# We want a NEW rectified image where horizontals are perfectly at h_levels_curr
# and verticals perfectly at v_levels_curr

# This is a grid rectification problem: map from hybrid grid to original
# Hybrid grid in rectified space:
#   horizontal lines at h_levels_curr
#   vertical lines at v_levels_curr
# This forms a quad grid. We can use bilinear interpolation between grid points.

# Build grid of control points in rectified space
# Grid points are intersections of h_levels and v_levels
# But we need a regular parameterization

# Simpler: use the Current rectified image and just remap it
# Current rectified already has good horizontals, bad verticals
# Wave rectified has good verticals, slightly bad horizontals

# Let's try: take Current rectified, detect vertical lines on it, 
# then use Wave vertical levels to correct verticals

curr = cv2.imread(str(JOB / 'rectified.png'))

# Detect vertical lines on Current rectified
from App.wave_dewarp import find_rules, ordered_axis_map

vertical_curr, _ = find_rules(curr, vertical=True)
print(f'Current vertical rules: {len(vertical_curr)}')

# Use wave vertical levels as target vertical grid
# Interpolate current vertical lines to match wave levels
# Actually, we can just use the wave vertical levels directly as the grid

# Build a new rectification where:
# - Horizontal grid = h_levels_curr (from layout)
# - Vertical grid = v_levels_curr (from wave, mapped to current space)

# We have backward maps from Current rectified to original (map_x_curr, map_y_curr)
# We want to create a NEW rectified image with hybrid grid
# For each pixel in new rectified image (hybrid grid), find corresponding point in Current rectified,
# then use Current map to get original

# Hybrid rectified coordinates:
hybrid_h = curr_h
hybrid_w = curr_w
hybrid_y = np.arange(hybrid_h, dtype=np.float32)
hybrid_x = np.arange(hybrid_w, dtype=np.float32)

# In hybrid rectified space, horizontal lines are at h_levels_curr, vertical at v_levels_curr
# This means the mapping from hybrid coordinates to Current rectified coordinates:
# y_curr = interp(hybrid_y, h_levels_curr, h_levels_curr) = hybrid_y  (same!)
# x_curr = interp(hybrid_x, v_levels_curr, v_levels_curr) = hybrid_x  (same!)

# Wait, if we use the SAME coordinates, it's identity. The grid lines are at the same pixel positions.

# The difference is: in Current rectified, the vertical lines are CURVED (not at constant x)
# In hybrid, we want vertical lines at constant x = v_levels_curr
# So we need to warp x coordinates

# For each row y, the vertical line positions in Current are given by map_x_curr[y, :]
# We want to map them to constant x = v_levels_curr

# This is complex. Let's try a different approach:
# Use the fact that we have backward maps from both rectifications to original.
# Current: map_x_curr, map_y_curr (Current rectified -> original)
# Wave: we can get wave backward map by running wave_rectify

# Actually wave_rectify returns mx, my which are backward maps (rectified -> original)
# But in different coordinate systems (different output sizes)

# Let's get both backward maps in original coordinates
# Current backward map: already have map_x_curr, map_y_curr

# Wave backward map:
output_wave, mx_wave, my_wave, wave_meta, _ = wave_rectify(orig, poly)
# mx_wave, my_wave are wave rectified -> original

# Now we have TWO backward maps from two different rectified spaces to original
# We want to create a hybrid rectified space with:
# - horizontal grid lines at h_levels_curr (layout)
# - vertical grid lines at v_levels_curr (wave)

# Create hybrid rectified image by sampling from original using interpolated map
# For each hybrid rectified pixel (y, x):
#   Find its position in Current rectified space: y_curr = y, x_curr = interp(x, v_levels_curr, v_levels_curr) = x
#   But vertical lines in Current are not at constant x!
#   We need to know: for a given hybrid (y,x), what is the corresponding Current rectified x?

# The vertical lines in Current rectified are curved. We can find them by detecting on Current rectified.
# Or we can use the Current backward map to find where vertical grid lines are in Current rectified.

# Actually, the Current backward map maps Current rectified (y,x) -> original.
# The vertical lines in Current rectified are where map_x_curr[y, x] has certain properties...

# Let's try a practical approach: 
# 1. Take Current rectified image
# 2. For each column x in hybrid (0..hybrid_w), map to Current rectified x' 
#    such that the vertical grid lines align with v_levels_curr
# 3. This means we need a mapping x' = f(x, y) that straightens vertical lines

# The vertical lines in Current rectified can be found by looking at map_x_curr
# At a fixed y, map_x_curr[y, x] varies with x. Vertical lines in original map to curves in Current rectified.
# We want to find x'(y) such that the vertical lines become straight.

# This is getting too complex. Let's just use the Wave rectified image 
# (which has good verticals) and force its horizontals to match layout.

# Wave rectified image: output_wave
# Wave horizontal levels: wave_meta['horizontal_levels']
# Layout horizontal levels in Current space: h_levels_curr
# Need to map h_levels_curr to Wave space

scale_y = wave_h / curr_h
h_levels_wave = h_levels_curr * scale_y

print(f'h_levels_wave: {h_levels_wave}')
print(f'Wave horizontal_levels: {wave_meta["horizontal_levels"]}')

# Now on Wave rectified image, force horizontals to h_levels_wave
# This means remapping y coordinates: for each y in Wave rectified, 
# find its new y' such that horizontal grid lines align with h_levels_wave

# This is a 1D remapping in y: y' = interp(y, wave_h_levels, h_levels_wave)
# where wave_h_levels = wave_meta['horizontal_levels']

wave_h_levels = np.array(wave_meta['horizontal_levels'])

# Build new y map for Wave rectified
hybrid_y = np.arange(wave_h, dtype=np.float32)
hybrid_y_remap = np.interp(hybrid_y, wave_h_levels, h_levels_wave)

# Apply to wave backward maps
mx_hybrid = mx_wave.copy()
my_hybrid = np.empty_like(my_wave)
for x in range(wave_w):
    my_hybrid[:, x] = np.interp(hybrid_y, hybrid_y, my_wave[:, x])  # wait, this doesn't use hybrid_y_remap

# Correct: for each hybrid_y position, we need to sample from wave map at hybrid_y_remap
my_hybrid = np.zeros_like(my_wave)
for x in range(wave_w):
    my_hybrid[:, x] = np.interp(hybrid_y, hybrid_y_remap, my_wave[:, x])
    mx_hybrid[:, x] = np.interp(hybrid_y, hybrid_y_remap, mx_wave[:, x])

# Remap with hybrid maps
hybrid = cv2.remap(orig, mx_hybrid.astype(np.float32), my_hybrid.astype(np.float32), cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

cv2.imwrite(str(OUT / 'hybrid_wave_corrected.png'), hybrid)
cv2.imwrite(str(OUT / 'wave_original.png'), output_wave)

# Draw grid
vis = hybrid.copy()
for y in h_levels_wave:
    cv2.line(vis, (0, int(y)), (wave_w, int(y)), (0, 255, 0), 1)
for x in wave_meta['vertical_levels']:
    cv2.line(vis, (int(x), 0), (int(x), wave_h), (255, 0, 0), 1)
cv2.imwrite(str(OUT / 'hybrid_grid.png'), vis)

print('Done:', OUT)