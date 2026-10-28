#!/usr/bin/env python
"""Детальное сравнение receipt.json для проваленных чеков."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
REPORT_DIR = ROOT / 'reports' / 'ocr_regression' / '20261028_042322_150542'

checks = [
    'e649d140e9b547ca9722dbca5a165e77',
    'ff18e2fe41ca46c5b9011620290b5711',
    '7bbc6159cbc04759b0293ce8d5eece38',
    '30c0654231b2415eb348cbc0e43e8a35',
    'da485d76181d42e49a1af44ba5969953',
]

for check_id in checks:
    job_dir = REPORT_DIR / check_id
    if not job_dir.exists():
        continue
    
    baseline_receipt = json.loads((job_dir / 'receipt.json').read_text(encoding='utf-8'))
    
    from App.table_boundary import detect_table_boundary
    from App.receipt_rectification import rectify_table
    from App.web_pipeline import table_layout, grid_cells
    from App.ocr_run import MODEL
    
    orig = cv2.imread(str(job_dir / 'original.png'))
    b = detect_table_boundary(orig)
    poly = b['polygon']
    rectified, mx, my, metadata = rectify_table(orig, poly, trust_top=b['status']=='detected')
    
    layout = table_layout(rectified, metadata)
    cells = grid_cells(layout)
    
    # OCR each cell - cells is list of dicts with 'box'
    rows_dict = {}
    for cell in cells:
        box = np.array(cell['box'], np.int32)
        x1, y1 = box[:,0].min(), box[:,1].min()
        x2, y2 = box[:,0].max(), box[:,1].max()
        cell_img = rectified[y1:y2, x1:x2]
        if cell_img.size == 0:
            continue
        results = MODEL(cell_img)
        text = ' '.join([r[1] for r in results]) if results else ''
        row_idx = cell.get('row', 0)
        role = cell.get('role', '')
        if row_idx not in rows_dict:
            rows_dict[row_idx] = {}
        rows_dict[row_idx][role] = text.strip()
    
    # Convert to rows list
    candidate_rows = []
    for row_idx in sorted(rows_dict.keys()):
        row_data = rows_dict[row_idx]
        # Order: number, name, quantity, price
        row_list = [
            row_data.get('number', ''),
            row_data.get('name', ''),
            row_data.get('quantity', ''),
            row_data.get('price', '')
        ]
        candidate_rows.append(row_list)
    
    baseline_receipt = json.loads((job_dir / 'receipt.json').read_text(encoding='utf-8'))
    base_rows = baseline_receipt.get('rows', [])
    cand_rows = candidate_rows
    
    print(f'\n=== {check_id} ===')
    print(f'Baseline rows: {len(base_rows)}')
    print(f'Candidate rows: {len(cand_rows)}')
    
    for i, (br, cr) in enumerate(zip(base_rows, cand_rows)):
        if br != cr:
            print(f'  Row {i} DIFF:')
            print(f'    base: {br}')
            print(f'    cand: {cr}')
    
    if len(base_rows) != len(cand_rows):
        print(f'  ROW COUNT DIFF: base={len(base_rows)}, cand={len(cand_rows)}')
    
    base_total = baseline_receipt.get('total')
    base_vat = baseline_receipt.get('vat')
    print(f'  Baseline total: {base_total}, vat: {base_vat}')