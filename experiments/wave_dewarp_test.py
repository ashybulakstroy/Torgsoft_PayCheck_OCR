#!/usr/bin/env python
"""Тест wave-dewarp (локальное выпрямление по сетке) на этом чеке."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'wave_dewarp_test'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))

# Use ordered polygon from improved detector
from App.table_boundary import detect_table_boundary
b = detect_table_boundary(orig)
poly = b['polygon']  # already ordered tl,tr,br,bl

# Try wave dewarp
from App.wave_dewarp import find_rules, rectify

try:
    output, mx, my, wave, _ = rectify(orig, poly)
    print('Wave status:', wave.get('status'))
    print('Straightness check:', wave.get('straightness_check'))
    cv2.imwrite(str(OUT / 'wave_rectified.png'), output)
    
    # Also simple homography for comparison
    margin = 20
    width = int((np.linalg.norm(poly[1]-poly[0]) + np.linalg.norm(poly[2]-poly[3]))/2)
    height = int((np.linalg.norm(poly[3]-poly[0]) + np.linalg.norm(poly[2]-poly[1]))/2)
    dst = np.array([[margin,margin],[margin+width,margin],[margin+width,margin+height],[margin,margin+height]], np.float32)
    M = cv2.getPerspectiveTransform(poly.astype(np.float32), dst)
    simple = cv2.warpPerspective(orig, M, (width+2*margin, height+2*margin), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    cv2.imwrite(str(OUT / 'simple_homography.png'), simple)
    
    # Current rectified
    curr = cv2.imread(str(JOB / 'rectified.png'))
    cv2.imwrite(str(OUT / 'current_rectified.png'), curr)
    
    print('Done:', OUT)
except Exception as e:
    print('Wave dewarp failed:', e)
    import traceback
    traceback.print_exc()