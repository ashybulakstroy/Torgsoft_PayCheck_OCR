
r"""Получить выпрямленную таблицу через существующие алгоритмы App.

Расположение в проекте:
    App/                      (существующий пакет)
    experiments/              (положить этот файл сюда)
    experiment_outputs/       (создастся автоматически)
    data/to_do/<id>/original.png
    .venv/Scripts/python.exe

Запуск из КОРНЯ проекта (PowerShell):
    .\.venv\Scripts\python.exe -m experiments.reproduce_rectified_table \
      "data\to_do\14e13295687b4112b32ce4b27a5c7189\original.png" \
      --output "experiment_outputs\14e13295687b4112b32ce4b27a5c7189\rectified_demo.png"

Требуются OpenCV и NumPy внутри существующего .venv.
Программа не изменяет исходный файл и не требует замены модулей App.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from App.table_boundary import detect_table_boundary
from App.receipt_rectification import rectify_table
from App.rotation_rectification import rotate_table_crop


def read_image(path: Path) -> np.ndarray:
    # imdecode/fromfile supports Unicode paths on Windows.
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise ValueError(f"Unable to encode result: {path}")
    encoded.tofile(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rectify printed receipt table with existing App modules"
    )
    parser.add_argument(
        "photo",
        type=Path,
        help="Input JPG/PNG file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rectified_demo.png")
    )
    args = parser.parse_args()

    image = read_image(args.photo)
    boundary = detect_table_boundary(image)
    polygon = boundary["polygon"]

    if polygon is None:
        print("Table not detected: manual review required")
        return 2

    # Match the route inside App/web_pipeline.py.
    if "partial_contour_replaced_by_full_grid_hypothesis" in boundary["warnings"]:
        rectified, map_x, map_y, metadata = rotate_table_crop(
            image, polygon
        )
    else:
        rectified, map_x, map_y, metadata = rectify_table(
            image,
            polygon,
            trust_top=boundary["status"] == "detected"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_image(args.output, rectified)

    # Optional overlay to inspect the automatically detected polygon.
    preview = image.copy()
    cv2.polylines(
        preview,
        [np.rint(polygon).astype(np.int32)],
        True,
        (0, 255, 0),
        3
    )

    overlay = args.output.with_name("detected_demo.png")
    write_image(overlay, preview)

    report = {
        "input": str(args.photo),
        "output": str(args.output),
        "overlay": str(overlay),
        "boundary_status": boundary["status"],
        "boundary_warnings": boundary["warnings"],
        "polygon": polygon.tolist(),
        "method": metadata.get("method"),
        "output_size": list(rectified.shape[1::-1]),
        "wave_status": metadata.get("wave_assessment", {}).get("status"),
        "wave_selected": metadata.get("wave_assessment", {}).get("selected"),
        "wave_reason": metadata.get("wave_assessment", {}).get("reason"),
    }

    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
