#!/usr/bin/env python
"""Создание визуальных отчётов для проваленных чеков."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
REPORT_DIR = ROOT / 'reports' / 'ocr_regression' / '20261028_042322_150542'
OUT = ROOT / 'experiment_outputs' / 'fail_visual_reports'
OUT.mkdir(parents=True, exist_ok=True)

# Representative failed checks
checks = [
    'e649d140e9b547ca9722dbca5a165e77',  # first check
    'ff18e2fe41ca46c5b9011620290b5711',  # 0→12 lines
    '7bbc6159cbc04759b0293ce8d5eece38',  # 0→12 lines
    '30c0654231b2415eb348cbc0e43e8a35',  # 13→14 lines
    'da485d76181d42e49a1af44ba5969953',  # -2 lines
]

for check_id in checks:
    job_dir = REPORT_DIR / check_id
    if not job_dir.exists():
        print(f'Skipping {check_id}: not found')
        continue
    
    orig = cv2.imread(str(job_dir / 'original.png'))
    baseline_rect = cv2.imread(str(job_dir / 'rectified.png'))
    
    # Run current pipeline to get candidate rectified
    from App.table_boundary import detect_table_boundary
    from App.receipt_rectification import rectify_table
    from App.web_pipeline import table_layout, grid_cells
    
    b = detect_table_boundary(orig)
    poly = b['polygon']
    rectified, mx, my, metadata = rectify_table(orig, poly, trust_top=b['status']=='detected')
    
    # Save candidate rectified
    cv2.imwrite(str(OUT / f'{check_id}_candidate_rectified.png'), rectified)
    cv2.imwrite(str(OUT / f'{check_id}_baseline_rectified.png'), baseline_rect)
    cv2.imwrite(str(OUT / f'{check_id}_original.png'), orig)
    
    # Draw grids
    vis_base = baseline_rect.copy()
    vis_cand = rectified.copy()
    
    # Baseline grid from layout.json
    layout = json.loads((job_dir / 'layout.json').read_text(encoding='utf-8'))['layout']
    for y in layout['row_boundaries']:
        cv2.line(vis_base, (0, int(y)), (baseline_rect.shape[1], int(y)), (0, 255, 0), 1)
    for x in layout.get('vertical_levels', []):
        cv2.line(vis_base, (int(x), 0), (int(x), baseline_rect.shape[0]), (255, 0, 0), 1)
    
    # Candidate grid from metadata
    h_levels = metadata.get('horizontal_levels', [])
    v_levels = metadata.get('vertical_levels', [])
    for y in h_levels:
        cv2.line(vis_cand, (0, int(y)), (rectified.shape[1], int(y)), (0, 255, 0), 1)
    for x in v_levels:
        cv2.line(vis_cand, (int(x), 0), (int(x), rectified.shape[0]), (255, 0, 0), 1)
    
    cv2.imwrite(str(OUT / f'{check_id}_baseline_grid.png'), vis_base)
    cv2.imwrite(str(OUT / f'{check_id}_candidate_grid.png'), vis_cand)
    
    # Side by side diff (resize to same)
    h = min(baseline_rect.shape[0], rectified.shape[0])
    w = min(baseline_rect.shape[1], rectified.shape[1])
    base_crop = baseline_rect[:h, :w]
    cand_crop = rectified[:h, :w]
    diff = cv2.absdiff(base_crop, cand_crop)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)
    diff_color = cv2.applyColorMap(diff_thresh, cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT / f'{check_id}_diff.png'), diff_color)
    
    # OCR comparison
    baseline_ocr = json.loads((job_dir / 'ocr_rectified.json').read_text(encoding='utf-8'))
    candidate_ocr = json.loads((job_dir / 'ocr_rectified.json').read_text(encoding='utf-8'))
    
    print(f'{check_id}: baseline lines={len(baseline_ocr.get("lines", []))}, candidate lines={len(candidate_ocr.get("lines", []))}')
    print(f'  baseline receipt rows={len(json.loads((job_dir/"receipt.json").read_text()).get("rows", []))}')
    print(f'  candidate method={metadata.get("method")}, h_levels={len(metadata.get("horizontal_levels", []))}, v_levels={len(metadata.get("vertical_levels", []))}')

print('Done:', OUT)