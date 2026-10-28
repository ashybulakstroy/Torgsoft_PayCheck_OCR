"""Детектор геометрии товарной таблицы Торгсофт.

Установка: experiments/table_geometry_detector.py (рядом с папкой App).
Из корня проекта:
    .\\.venv\\Scripts\\python.exe -m experiments.table_geometry_detector \
      "data\\to_do\\14e13295687b4112b32ce4b27a5c7189\\original.png" \
      --output "experiment_outputs\\14e13295687b4112b32ce4b27a5c7189\\detector.json"

Использует существующий App.table_boundary, но независимо определяет
четыре опорные точки сетки, даже если шеститочечный контур содержит
ступеньку нижней ячейки «Итого». Не модифицирует App или исходное фото.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from App.table_boundary import detect_table_boundary

VERSION = "table-geometry-detector-1.0"


def read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать фотографию: {path}")
    return image


def _line_intersection(a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    """Пересечение двух бесконечных прямых (абсциссы и ординаты в пикселях)."""
    u = b - a
    v = d - c
    denominator = float(u[0] * v[1] - u[1] * v[0])
    if abs(denominator) < 1e-6:
        return None
    w = c - a
    distance = float(w[0] * v[1] - w[1] * v[0]) / denominator
    return a + distance * u


def quadrilateral_from_polygon(polygon: np.ndarray) -> tuple[np.ndarray, dict]:
    """Получить TL, TR, BR, BL. Не подменять скошенную сетку convex hull.

    У чеков Торгсофт нижняя строка «Итого» занимает только правую часть.
    Поэтому 6-угольник имеет уступ: для ПОЛНОГО прямоугольника товарной
    таблицы BR восстанавливаем продолжением нижнего края последней строки.
    """
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4 or len(pts) > 14:
        raise ValueError(f"Неожиданное число вершин: {len(pts)}")
    if not np.isfinite(pts).all():
        raise ValueError("Контур содержит нечисловые координаты")
    h_range = float(np.ptp(pts[:, 1]))
    if h_range < 60:
        raise ValueError("Высота таблицы слишком мала")
    # При контуре из 6, 8, 10 ... точек на прямой верхней стороне могут
    # появиться дополнительные вершины. Берём крайние X из верхней полосы.
    top_band = pts[pts[:, 1] < pts[:, 1].min() + max(14., .11 * h_range)]
    bottom_band = pts[pts[:, 1] > pts[:, 1].min() + .68 * h_range]
    if len(top_band) < 2 or len(bottom_band) < 2:
        raise ValueError("Недостаточно точек верхней/нижней границы")
    tl = top_band[np.argmin(top_band[:, 0])]
    tr = top_band[np.argmax(top_band[:, 0])]
    bl = bottom_band[np.argmin(bottom_band[:, 0])]
    right = bottom_band[np.argmax(bottom_band[:, 0])]
    width = float(tr[0] - tl[0])
    if width <= 80:
        raise ValueError("Ширина верхней границы слишком мала")
    details: dict = {"source_polygon_points": int(len(pts)),
                     "bottom_right_strategy": "observed"}
    # Последняя строка «Итого» иногда существует только справа, из-за
    # чего низ внешнего контура имеет ступеньку. Опорой для продолжения
    # нижней границы служит её видимая горизонтальная часть.
    candidates = [p for p in bottom_band if
                  (bl[0] + 0.2 * width < p[0] < right[0] - 0.08 * width)
                  and abs(p[1] - bl[1]) < max(16., h_range * .055)]
    if len(pts) > 4 and candidates:
        supporting = min(candidates, key=lambda p: abs(p[1] - bl[1]))
        intersection = _line_intersection(bl, supporting, tr, right)
        if intersection is not None and (
            min(tr[0], right[0]) - 30 < intersection[0] < max(tr[0], right[0]) + 30
            and abs(intersection[1] - bl[1]) <= max(65., .1 * h_range)
        ):
            br = intersection
            details["bottom_right_strategy"] = "last_product_row_extrapolation"
            details["bottom_support_point"] = supporting.tolist()
        else:
            br = right
    else:
        br = right

    corners = np.asarray([tl, tr, br, bl], dtype=np.float32)
    return corners, details


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    return math.degrees(math.atan2(float(delta[1]), float(delta[0])))


def geometry_metrics(corners: np.ndarray, image_size: tuple[int, int]) -> dict:
    h, w = image_size
    tl, tr, br, bl = np.asarray(corners, np.float64)
    top_width = float(np.linalg.norm(tr - tl))
    bottom_width = float(np.linalg.norm(br - bl))
    left_height = float(np.linalg.norm(bl - tl))
    right_height = float(np.linalg.norm(br - tr))
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if min(top_width, bottom_width, left_height, right_height) < 45:
        raise ValueError("Слишком короткая сторона товарной таблицы")
    if area < 0.012 * w * h:
        raise ValueError("Обнаруженная таблица слишком мала")

    top_angle = _angle(tl, tr)
    bottom_angle = _angle(bl, br)
    left_lean = _angle(tl, bl) - 90.
    right_lean = _angle(tr, br) - 90.
    # Absolute widths computed from oblique edges; difference is relative.
    width_change_pct = abs(bottom_width - top_width) / max(top_width, 1) * 100
    angle_difference = abs(top_angle - bottom_angle)
    side_divergence = abs(left_lean - right_lean)
    # Convexity and aspect-ratio checks before applying any transform.
    convex = bool(cv2.isContourConvex(corners.astype(np.float32)))
    valid = bool(convex and (0.15 < max(top_width, bottom_width) /
                                   max(left_height, right_height) < 5.0))
    return {
        "valid": valid,
        "width_top_px": round(top_width, 2),
        "width_bottom_px": round(bottom_width, 2),
        "width_change_percent": round(width_change_pct, 2),
        "top_angle_degrees": round(top_angle, 2),
        "bottom_angle_degrees": round(bottom_angle, 2),
        "edge_angle_difference_degrees": round(angle_difference, 2),
        "left_lean_degrees": round(left_lean, 2),
        "right_lean_degrees": round(right_lean, 2),
        "side_divergence_degrees": round(side_divergence, 2),
        "table_area_percent": round(100 * area / (w * h), 2),
    }


def detect_geometry(image: np.ndarray) -> dict:
    """Определяет SKIP / HOMOGRAPHY / UNCERTAIN; тип DEWARP после warp.

    Пороговые значения предварительные; сравнивать с эталонными чеками.
    """
    boundary = detect_table_boundary(image)
    result: dict = {
        "version": VERSION,
        "mode": "UNCERTAIN",
        "status": boundary["status"],
        "warnings": list(boundary.get("warnings", [])),
        "polygon": None,
        "corners": None,
        "metrics": None,
        "reason": "",
    }
    polygon = boundary.get("polygon")
    if polygon is None:
        result["reason"] = "Границы товарной таблицы не найдены"
        return result
    result["polygon"] = np.asarray(polygon).tolist()
    if boundary["status"] != "detected":
        result["reason"] = "Детектор не уверен в границах таблицы"
        return result
    try:
        corners, info = quadrilateral_from_polygon(polygon)
        metrics = geometry_metrics(corners, image.shape[:2])
        if not metrics["valid"]:
            raise ValueError("Четырёхугольник не прошёл проверку выпуклости/пропорций")
        result["corners"] = corners.tolist()
        result["corners_info"] = info
        result["metrics"] = metrics
        needs_fix = (metrics["width_change_percent"] >= 2.5 or
                     abs(metrics["top_angle_degrees"]) >= 1.2 or
                     abs(metrics["bottom_angle_degrees"]) >= 1.2 or
                     max(abs(metrics["left_lean_degrees"]),
                         abs(metrics["right_lean_degrees"])) >= 1.5 or
                     metrics["edge_angle_difference_degrees"] >= 1.0)
        result["mode"] = "HOMOGRAPHY" if needs_fix else "SKIP"
        result["reason"] = ("Найдена перспективная деформация" if needs_fix
                            else "Границы достаточно прямые; коррекция не обязательна")
    except (ValueError, TypeError) as exc:
        result["reason"] = str(exc)
        result["warnings"].append("quadrilateral_reconstruction_failed")
    return result


def draw_detection(image: np.ndarray, result: dict) -> np.ndarray:
    preview = image.copy()
    if result["polygon"] is not None:
        poly = np.rint(result["polygon"]).astype(np.int32)
        cv2.polylines(preview, [poly], True, (0, 220, 255), 2)
    if result["corners"] is not None:
        pts = np.rint(result["corners"]).astype(np.int32)
        cv2.polylines(preview, [pts], True, (10, 220, 30), 3)
        for name, point in zip(("TL", "TR", "BR", "BL"), pts):
            cv2.circle(preview, tuple(point), 6, (0, 0, 255), -1)
            cv2.putText(preview, name, tuple(point + (5, -7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
    return preview


def main() -> int:
    parser = argparse.ArgumentParser(description="Детектор искажения товарной таблицы")
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment_outputs/detector.json"))
    args = parser.parse_args()
    image = read_image(args.photo)
    result = detect_geometry(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay = args.output.with_name("detected.png")
    success, encoded = cv2.imencode(".png", draw_detection(image, result))
    if success:
        encoded.tofile(str(overlay))
    print(json.dumps({"mode": result["mode"], "reason": result["reason"],
                      "report": str(args.output), "preview": str(overlay),
                      "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
    return 0 if result["mode"] != "UNCERTAIN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
