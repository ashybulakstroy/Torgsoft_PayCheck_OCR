#!/usr/bin/env python
"""Правильное визуальное сравнение для e54310b7a330459d9968cc4a53633549."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e54310b7a330459d9968cc4a53633549'
OUT = ROOT / 'experiment_outputs' / 'regression_check_e543_v2'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
cv2.imwrite(str(OUT / '01_original.png'), orig)

# Baseline rectified (from web_outputs)
baseline_rect = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / '02_baseline_rectified.png'), baseline_rect)

# Current rectified
from App.table_boundary import detect_table_boundary
from App.receipt_rectification import rectify_table

b = detect_table_boundary(orig)
poly = b['polygon']
rectified, mx, my, metadata = rectify_table(orig, poly, trust_top=b["status"]=="detected")
cv2.imwrite(str(OUT / '03_current_rectified.png'), rectified)

# Difference (resize to same size)
h = min(baseline_rect.shape[0], rectified.shape[0])
w = min(baseline_rect.shape[1], rectified.shape[1])
base_crop = baseline_rect[:h, :w]
curr_crop = rectified[:h, :w]
diff = cv2.absdiff(base_crop, curr_crop)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, diff_thresh = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)
diff_color = cv2.applyColorMap(diff_thresh, cv2.COLORMAP_JET)
cv2.imwrite(str(OUT / '04_diff_heatmap.png'), diff_color)

# Side by side
side = np.full((h, w*2+20, 3), 255, np.uint8)
side[:h, :w] = base_crop
side[:h, w+20:w*2+20] = curr_crop
# Labels
cv2.putText(side, 'Baseline', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.putText(side, 'Current', (baseline_rect.shape[1]+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imwrite(str(OUT / '05_side_by_side.png'), side)

# Baseline OCR
baseline_ocr = json.loads((JOB / 'ocr_rectified.json').read_text(encoding='utf-8'))['lines']

# Draw baseline OCR boxes
vis_base = baseline_rect.copy()
for line in baseline_ocr:
    box = line['box']
    pts = np.array(box, np.int32).reshape(-1,1,2)
    cv2.polylines(vis_base, [pts], True, (0,255,0), 1)
    # Text label
    x, y = int(box[0][0]), int(box[0][1])
    cv2.putText(vis_base, line['text'][:20], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
cv2.imwrite(str(OUT / '06_baseline_ocr_boxes.png'), vis_base)

# Baseline grid
vis_base_grid = baseline_rect.copy()
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']
for y in layout['row_boundaries']:
    cv2.line(vis_base_grid, (0, int(y)), (baseline_rect.shape[1], int(y)), (0,255,0), 1)
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
v_levels = np.array(rect_info.get('vertical_levels', []))
for x in v_levels:
    cv2.line(vis_base_grid, (int(x), 0), (int(x), baseline_rect.shape[0]), (255,0,0), 1)
cv2.imwrite(str(OUT / '07_baseline_grid.png'), vis_base_grid)

# Current grid
vis_curr_grid = rectified.copy()
for y in metadata.get('horizontal_levels', []):
    cv2.line(vis_curr_grid, (0, int(y)), (rectified.shape[1], int(y)), (0,255,0), 1)
for x in metadata.get('vertical_levels', []):
    cv2.line(vis_curr_grid, (int(x), 0), (int(x), rectified.shape[0]), (255,0,0), 1)
cv2.imwrite(str(OUT / '08_current_grid.png'), vis_curr_grid)

# Receipt comparison
baseline_receipt = json.loads((JOB / 'receipt.json').read_text(encoding='utf-8'))
baseline_rows = baseline_receipt.get('rows', [])

print(f'Baseline OCR lines: {len(baseline_ocr)}')
print(f'Baseline receipt rows: {len(baseline_receipt.get("rows", []))}')
print(f'Current metadata: method={metadata.get("method")}, status={metadata.get("status")}')
print(f'Current horizontal_levels: {len(metadata.get("horizontal_levels", []))}')
print(f'Current vertical_levels: {len(metadata.get("vertical_levels", []))}')

# Save metadata
json.dump(metadata, open(OUT / 'current_metadata.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print('Done:', OUT)