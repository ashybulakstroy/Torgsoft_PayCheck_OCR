"""Регрессионная проверка Necessity Detector на тестовом фото.

Из корня проекта:
.\\.venv\\Scripts\\python.exe -m experiments.test_dewarp_necessity \
    "data\\to_do\\14e13295687b4112b32ce4b27a5c7189\\original.png"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
from experiments.table_geometry_detector import read_image, detect_geometry
from experiments.table_geometry_improver import rectify
from experiments.dewarp_necessity_detector import assess_dewarp_necessity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('photo', type=Path)
    args = parser.parse_args()
    image = read_image(args.photo)
    detection = detect_geometry(image)
    assert detection['mode'] != 'UNCERTAIN', detection
    original_result, perspective, report = rectify(image, detection, mode='auto')
    gate = report['dewarp_necessity']
    print('ORIGINAL:', gate['decision'], report['method'])
    # Контрольный чек: выпрямленная перспектива уже достаточно ровная.
    assert gate['decision'] == 'SKIP', gate
    assert report['method'] == 'homography', report
    assert np.array_equal(original_result, perspective)

    # Создаём искусственный физический изгиб нескольких печатных линий.
    height, width = perspective.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    wave = 9.0 * np.sin(np.pi * xx / (width - 1))
    curved = cv2.remap(perspective, xx, yy-wave.astype(np.float32),
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(255,255,255))
    bent = assess_dewarp_necessity(curved)
    print('SYNTHETIC CURVE:', bent['decision'], bent['reason'])
    assert bent['decision'] == 'DEWARP', bent
    print('PASS: 2 tests')

if __name__ == '__main__':
    main()
