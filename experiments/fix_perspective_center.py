#!/usr/bin/env python
from __future__ import annotations
import json, math
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('.')
job_dir = ROOT / 'web_outputs' / 'e649d140e9b547ca9722dbca5a165e77'

# load original image and polygon
orig = cv2.imread(str(job_dir / 'original.png'))
rect_info = json.loads((job_dir / 'rectification.json').read_text(encoding='utf-8'))
poly = np.array(rect_info['source_polygon'], dtype=np.float32)

print('poly:', poly.tolist())
h,w = orig.shape[:2]
print('orig:', w,h)
