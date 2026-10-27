"""Rotation-only experiment for a correctly detected but tilted receipt table."""
from __future__ import annotations

import cv2
import numpy as np


def rotation_angle(polygon):
    points = np.asarray(polygon, np.float32).reshape(4, 2)
    tl, tr, br, bl = points
    angles = [np.degrees(np.arctan2(*(tr - tl)[::-1])),
              np.degrees(np.arctan2(*(br - bl)[::-1]))]
    angle = float(np.median(angles))
    if not 2. <= abs(angle) <= 12.:
        raise ValueError(f"Наклон {angle:.2f}° вне безопасного диапазона")
    return angle


def rotate_and_crop(image, polygon, margin=16):
    """Rotate the whole photo around its centre, then crop the table region."""
    angle = rotation_angle(polygon)
    height, width = image.shape[:2]
    center = np.array([width / 2., height / 2.], np.float32)
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_width = int(np.ceil(height * sine + width * cosine))
    out_height = int(np.ceil(height * cosine + width * sine))
    matrix[:, 2] += np.array([out_width / 2., out_height / 2.]) - center
    rotated = cv2.warpAffine(image, matrix, (out_width, out_height), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    points = cv2.transform(np.asarray(polygon, np.float32)[None, :, :], matrix)[0]
    lo = np.maximum(0, np.floor(points.min(axis=0)).astype(int) - margin)
    hi = np.minimum([out_width, out_height], np.ceil(points.max(axis=0)).astype(int) + margin + 1)
    crop = rotated[lo[1]:hi[1], lo[0]:hi[0]].copy()
    return crop, dict(rotation_opencv_deg=angle, rotation_clockwise_deg=-angle,
                      rotated_polygon=(points - lo).tolist(), roi=[*lo.tolist(), *hi.tolist()])
