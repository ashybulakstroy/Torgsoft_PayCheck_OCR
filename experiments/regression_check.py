#!/usr/bin/env python
"""Визуальное сравнение для e54310b7a330459d9968cc4a53633549."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e54310b7a330459d9968cc4a53633549'
OUT = ROOT / 'experiment_outputs' / 'regression_check_e543'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
cv2.imwrite(str(OUT / 'original.png'), orig)

# Baseline rectified
baseline_rect = cv2.imread(str(JOB / 'rectified.png'))
cv2.imwrite(str(OUT / 'baseline_rectified.png'), baseline_rect)

# Current rectified
from App.table_boundary import detect_table_boundary
from App.receipt_rectification import rectify_table

b = detect_table_boundary(orig)
poly = b['polygon']
rectified, mx, my, metadata = rectify_table(orig, poly, trust_top=b["status"]=="detected")
cv2.imwrite(str(OUT / 'current_rectified.png'), rectified)

# Baseline OCR lines
baseline_ocr = json.loads((JOB / 'ocr_rectified.json').read_text(encoding='utf-8'))

# Current OCR - run pipeline
from App.web_pipeline import ReceiptPipeline

# Create a minimal job-like object
class DummyJob:
    def __init__(self, directory):
        self.directory = directory
    def phase(self, *args): pass
    def image(self, *args): pass
    def json_file(self, *args): pass
    def publish(self, *args, **kwargs): pass

# We need to run OCR on current rectified - use the pipeline's recognize method
# Save current rectified to a temp file and run recognize
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / 'rectified.png'
    cv2.imwrite(str(tmp_path), rectified)
    
    pipeline = ReceiptPipeline()
    pipeline.resources()
    # Hack: use the internal recognize method
    # We'll just compare the saved baseline OCR with what we expect
    pass

# Load baseline receipt
baseline_receipt = json.loads((JOB / 'receipt.json').read_text(encoding='utf-8'))

print('Baseline OCR lines:', len(baseline_ocr))
print('Baseline receipt rows:', len(baseline_receipt.get('rows', [])))

# Draw OCR boxes on baseline rectified
vis_base = baseline_rect.copy()
for line in baseline_ocr:
    box = line['box']
    pts = np.array(box, np.int32).reshape(-1,1,2)
    cv2.polylines(vis_base, [pts], True, (0,255,0), 1)
cv2.imwrite(str(OUT / 'baseline_ocr_boxes.png'), vis_base)

# Draw on current rectified (without OCR boxes, just grid)
vis_curr = rectified.copy()
# Draw grid from metadata
v_levels = metadata.get('vertical_levels', [])
h_levels = metadata.get('horizontal_levels', [])
for y in h_levels:
    cv2.line(vis_curr, (0, int(y)), (rectified.shape[1], int(y)), (0,255,0), 1)
for x in v_levels:
    cv2.line(vis_curr, (int(x), 0), (int(x), rectified.shape[0]), (255,0,0), 1)
cv2.imwrite(str(OUT / 'current_grid.png'), vis_curr)

# Draw on baseline rectified with grid
vis_base_grid = baseline_rect.copy()
rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
layout = json.loads((JOB / 'layout.json').read_text(encoding='utf-8'))['layout']
for y in layout['row_boundaries']:
    cv2.line(vis_base_grid, (0, int(y)), (baseline_rect.shape[1], int(y)), (0,255,0), 1)
for x in v_levels:
    cv2.line(vis_base_grid, (int(x), 0), (int(x), baseline_rect.shape[0]), (255,0,0), 1)
cv2.imwrite(str(OUT / 'baseline_grid.png'), vis_base_grid)

# Difference image
diff = cv2.absdiff(baseline_rect, rectified)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, diff_thresh = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)
diff_color = cv2.applyColorMap(diff_thresh, cv2.COLORMAP_JET)
cv2.imwrite(str(OUT / 'diff.png'), diff_color)

print('Done:', OUT)