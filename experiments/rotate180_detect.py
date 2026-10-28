#!/usr/bin/env python
"""Поворот 180° и детекция углов."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path('.')
JOB = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'
OUT = ROOT / 'experiment_outputs' / 'rotate180_detect'
OUT.mkdir(parents=True, exist_ok=True)

orig = cv2.imread(str(JOB / 'original.png'))
h, w = orig.shape[:2]

# Rotate 180
M = cv2.getRotationMatrix2D((w/2, h/2), 180, 1.0)
rot180 = cv2.warpAffine(orig, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(str(OUT / 'rotated_180.png'), rot180)

# Detect on rotated
from App.table_boundary import detect_table_boundary
b = detect_table_boundary(rot180)
print('Rotated 180 boundary:', b.get('status'))
if b['polygon'] is not None:
    poly_rot = np.array(b['polygon'], dtype=np.float32)
    if len(poly_rot) != 4:
        # Take convex hull or approximate to 4 corners
        hull = cv2.convexHull(poly_rot.reshape(-1,1,2))
        poly_rot = hull.reshape(-1,2)[:4]  # take first 4
    print('Polygon (rotated 180):')
    for i, pt in enumerate(poly_rot):
        print(f'  {i}: [{pt[0]:.1f}, {pt[1]:.1f}]')
    
    # Map back to original coordinates
    M_inv = cv2.invertAffineTransform(M)
    poly_orig = cv2.transform(poly_rot.reshape(1,-1,2), M_inv).reshape(4,2)
    print('Polygon mapped back to original:')
    for i, pt in enumerate(poly_orig):
        print(f'  {i}: [{pt[0]:.1f}, {pt[1]:.1f}]')
    
    # Draw on original
    vis = orig.copy()
    for i, pt in enumerate(poly_orig):
        cv2.circle(vis, tuple(np.int32(pt)), 8, (0,0,255), -1)
        cv2.putText(vis, f'{i}', tuple(np.int32(pt)+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.polylines(vis, [np.int32(poly_orig)], True, (0,255,255), 2)
    cv2.imwrite(str(OUT / 'corners_from_rotated.png'), vis)
    
    # Also draw original detector polygon for comparison
    rect_info = json.loads((JOB / 'rectification.json').read_text(encoding='utf-8'))
    src_poly = np.array(rect_info['source_body_corners'], dtype=np.float32).reshape(4,2)
    vis2 = orig.copy()
    for i, pt in enumerate(src_poly):
        cv2.circle(vis2, tuple(np.int32(pt)), 6, (255,0,0), -1)
    cv2.polylines(vis2, [np.int32(src_poly)], True, (255,0,0), 1)
    for i, pt in enumerate(poly_orig):
        cv2.circle(vis2, tuple(np.int32(pt)), 8, (0,0,255), -1)
    cv2.polylines(vis2, [np.int32(poly_orig)], True, (0,255,255), 2)
    cv2.imwrite(str(OUT / 'comparison.png'), vis2)
    
    # Rectify with rotated-detected corners
    margin = 20
    width = int((np.linalg.norm(poly_orig[1]-poly_orig[0]) + np.linalg.norm(poly_orig[2]-poly_orig[3]))/2)
    height = int((np.linalg.norm(poly_orig[3]-poly_orig[0]) + np.linalg.norm(poly_orig[2]-poly_orig[1]))/2)
    dst = np.array([[margin,margin],[margin+width,margin],[margin+width,margin+height],[margin,margin+height]], np.float32)
    M_h = cv2.getPerspectiveTransform(poly_orig.astype(np.float32), dst)
    rect = cv2.warpPerspective(orig, M_h, (width+2*margin, height+2*margin), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    cv2.imwrite(str(OUT / 'rectified_from_rotated.png'), rect)
    
    curr = cv2.imread(str(JOB / 'rectified.png'))
    cv2.imwrite(str(OUT / 'current_rectified.png'), curr)

print('Done:', OUT)