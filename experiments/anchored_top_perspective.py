"""Experiment: detect trapezoid table geometry and rectify it with its top edge anchored.

A rigid rotation cannot make non-parallel top/bottom table edges parallel.  This
experiment checks their angular divergence and, when it is real, uses one
projective transform from the detected quadrilateral to a rectangle.  The top
two corners are the first edge of that transform, so the output is visually
anchored to the printed header rather than to the photo centre.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

from App.receipt_rectification import rectify_table

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"R:\Temp\ocr_reports\anchored_top_perspective")


def edge_angle(left: np.ndarray, right: np.ndarray) -> float:
    delta = right - left
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def diagnose(corners: np.ndarray) -> dict:
    tl, tr, br, bl = corners
    top, bottom = edge_angle(tl, tr), edge_angle(bl, br)
    divergence = abs(float((bottom - top + 90.0) % 180.0 - 90.0))
    return {
        "top_angle_deg": top,
        "bottom_angle_deg": bottom,
        "edge_divergence_deg": divergence,
        "needs_perspective": divergence >= 1.5,
        "reason": "верх и низ таблицы непараллельны" if divergence >= 1.5 else "границы практически параллельны",
    }


def refine_top_right(corners: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, dict]:
    """Replace only an unsupported top-right corner by a nearby contour vertex.

    The body-corner extrapolator can accidentally borrow the x-coordinate of
    the lower right side.  A contour point on the printed top rule is stronger
    evidence.  Points 0, 2 and 3 are deliberately left untouched.
    """
    refined = corners.copy()
    tl, estimated_tr = corners[0], corners[1]
    height = max(1.0, float(np.linalg.norm(corners[3] - tl)))
    candidates = contour[(contour[:, 0] > (tl[0] + estimated_tr[0]) / 2)
                         & (contour[:, 1] <= min(tl[1], estimated_tr[1]) + max(32.0, height * .12))
                         & (contour[:, 0] < estimated_tr[0] - 8.0)]
    if not len(candidates):
        return refined, {"top_right_refined": False, "reason": "нет подтверждённой вершины на верхней линии"}
    chosen = min(candidates, key=lambda point: float(np.linalg.norm(point - estimated_tr)))
    refined[1] = chosen
    return refined, {"top_right_refined": True, "old_top_right": estimated_tr.tolist(),
                     "new_top_right": chosen.tolist(), "reason": "верхняя правая вершина взята из контура печатной сетки"}


def rectify(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, dict]:
    tl, tr, br, bl = corners.astype(np.float32)
    width = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    height = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    width, height = max(width, 16), max(height, 16)
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    output = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return output, {"output_size": [width, height], "matrix": matrix.tolist()}


def write_report(out: Path, job: str, source: np.ndarray, corners: np.ndarray, result: np.ndarray, diagnostic: dict, geometry: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    annotated = source.copy()
    cv2.polylines(annotated, [np.rint(corners).astype(np.int32)], True, (0, 190, 160), 3, cv2.LINE_AA)
    if diagnostic.get("top_right_refined"):
        old = tuple(np.rint(diagnostic["old_top_right"]).astype(int))
        cv2.drawMarker(annotated, old, (30, 30, 230), cv2.MARKER_TILTED_CROSS, 20, 3)
    cv2.imwrite(str(out / "source_with_polygon.png"), annotated)
    cv2.imwrite(str(out / "top_anchored_perspective.png"), result)
    (out / "diagnostic.json").write_text(json.dumps({"job": job, **diagnostic, **geometry}, ensure_ascii=False, indent=2), encoding="utf-8")
    content = f"""<!doctype html><meta charset='utf-8'><title>Перспектива таблицы</title>
<style>body{{font:16px Arial;margin:32px;color:#13243e}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}img{{max-width:100%;border:1px solid #ccd6e5}}code{{background:#eef3f9;padding:3px 6px}}</style>
<h1>Проверка: перспектива с закреплённой верхней гранью</h1>
<p>Чек: <code>{html.escape(job)}</code>. Верхний край: <b>{diagnostic['top_angle_deg']:.2f}°</b>; нижний: <b>{diagnostic['bottom_angle_deg']:.2f}°</b>; расхождение: <b>{diagnostic['edge_divergence_deg']:.2f}°</b>.</p>
<p>{html.escape(diagnostic['reason'])}. Поворот вокруг любой точки не изменяет это расхождение; справа показано проектное выпрямление трапеции в прямоугольник. Красный крест — прежняя ошибочная верхняя правая точка; зелёный контур — уточнённый.</p>
<div class='grid'><section><h2>До: полигон таблицы</h2><img src='source_with_polygon.png'></section><section><h2>После: верхняя грань закреплена</h2><img src='top_anchored_perspective.png'></section></div>"""
    path = out / "index.html"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="9a86bd9dbe724fff88acab317fe8a371")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    job_dir = ROOT / "web_outputs" / args.job
    metadata = json.loads((job_dir / "rectification.json").read_text(encoding="utf-8"))
    corners = np.asarray(metadata["source_body_corners"], np.float32)
    contour = np.asarray(metadata["source_polygon"], np.float32)
    source = cv2.imdecode(np.frombuffer((job_dir / "original.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("Не удалось открыть original.png")
    corners, refinement = refine_top_right(corners, contour)
    diagnostic = {**diagnose(corners), **refinement}
    _, geometry = rectify(source, corners)
    # Show the production-compatible map: unlike the simple homography above,
    # it keeps the stepped lower total cell.
    result, _, _, production_metadata = rectify_table(source, contour, trust_top=True)
    geometry = {**geometry, "production_method": production_metadata["method"],
                "production_top_right_refinement": production_metadata.get("top_right_refinement")}
    report = write_report(args.output, args.job, source, corners, result, diagnostic, geometry)
    print(json.dumps({**diagnostic, "report": str(report.resolve())}, ensure_ascii=False))
    if hasattr(__import__("os"), "startfile"):
        __import__("os").startfile(str(report.resolve()))


if __name__ == "__main__":
    main()
