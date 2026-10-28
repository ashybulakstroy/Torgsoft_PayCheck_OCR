"""Выпрямить табличную сетку чека; отдельный эксперимент, App не менять.

Файлы разместить в папке experiments/ рядом с App/ и data/.
Запускать из корня проекта через существующую .venv:

.\\.venv\\Scripts\\python.exe -m experiments.table_geometry_improver ^
 "data\\to_do\\14e13295687b4112b32ce4b27a5c7189\\original.png" ^
 --output "experiment_outputs\\14e13295687b4112b32ce4b27a5c7189\\rectified.png"

В PowerShell команда должна быть ОДНОЙ СТРОКОЙ (символ ^ для cmd.exe).
Выходы: rectified.png, perspective.png, detected.png, rectified.json.

Ступень 1: гомография по четырём углам (в том числе для 6-точечного
          контура с отдельным выступом ячейки «Итого»).
Ступень 2: НОВЫЙ Dewarping Necessity Detector на уже выпрямленном
          perspective.png. Если нет согласованной остаточной кривизны,
          сохраняется только Homography (без ненужного Dewarping).
Ступень 3: если изгиб доказан, строится карта по горизонтальным линиям
          и пяти вертикальным границам четырёх столбцов.
Ступень 4: строгий валидатор отклоняет ухудшение любой обнаруженной линии
          или сомнительное локальное преобразование.

Результат после комбинированного преобразования семплируется ИЗ ОРИГИНАЛА
за один проход через cv2.remap, без повторного масштабирования JPEG.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

from App.wave_dewarp import find_rules, ordered_axis_map, map_quality
from experiments.table_geometry_detector import (
    detect_geometry, draw_detection, read_image,
)
from experiments.dewarp_necessity_detector import assess_dewarp_necessity

VERSION = "table-geometry-improver-1.1"


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Не удалось закодировать PNG: {path}")
    encoded.tofile(str(path))


def select_vertical_grid(rules: list, expected: int, width: int) -> tuple[list, dict]:
    """Выбрать настоящие 5 границ 4 столбцов, игнорируя лишнюю 6-ю.

    Не меняем '==5' на '==6': перебираем только непротиворечивые группы
    с двумя внешними границами и достаточным покрытием всей высоты.
    """
    levels = np.asarray([np.median(rule.values) for rule in rules], dtype=float)
    info = {"found": int(len(rules)), "expected": int(expected), "selected_levels": []}
    if len(rules) < expected or len(rules) > 9:
        return [], info
    options = []
    for indexes in itertools.combinations(range(len(rules)), expected):
        subset = levels[list(indexes)]
        gaps = np.diff(subset)
        if min(gaps) < width * .045 or subset[-1] - subset[0] < width * .7:
            continue
        # Применяем только подтверждённые внешние линии.
        if subset[0] > width * .13 or subset[-1] < width * .83:
            continue
        coverage = sum(rules[i].coverage for i in indexes)
        border_cost = abs(subset[0] - width * .025) + abs(subset[-1] - width * .975)
        score = coverage - border_cost / width * 3
        options.append((score, indexes))
    if not options:
        return [], info
    indexes = max(options, key=lambda item: item[0])[1]
    selected = [rules[i] for i in indexes]
    info["selected_levels"] = [round(float(levels[i]), 2) for i in indexes]
    info["dropped_levels"] = [round(float(x), 2) for i, x in enumerate(levels) if i not in indexes]
    return selected, info


def _curvature_report(image: np.ndarray) -> dict:
    horizontal, _ = find_rules(image)
    vertical, _ = find_rules(image, vertical=True)
    def summarize(rules):
        values = [r.curvature for r in rules]
        return {"count": len(rules), "max_px": round(float(max(values, default=0)), 3),
                "median_px": round(float(np.median(values)) if values else 0., 3),
                "rules": [{"level_px": round(float(np.median(r.values)), 2),
                           "curvature_px": round(float(r.curvature), 3)} for r in rules]}
    return {"horizontal": summarize(horizontal), "vertical": summarize(vertical)}


def _validate_candidate(before: dict, after: dict, quality: dict,
                        displacement_px: float) -> tuple[bool, str]:
    if not quality.get("finite") or not quality.get("positive"):
        return False, "Карта искажает ориентацию пикселей"
    if quality["min_jacobian"] < .45 or quality["max_jacobian"] > 2.2:
        return False, "Чрезмерное локальное растяжение/сжатие"
    if displacement_px > 18:
        return False, "Слишком большое смещение пикселей"
    for axis in ("horizontal", "vertical"):
        old, new = before[axis], after[axis]
        if new["count"] < max(3, old["count"] - 2):
            return False, f"Потеря печатных линий: {axis}"
        if new["max_px"] > old["max_px"] + max(.35, old["max_px"] * .06):
            return False, f"Максимальная изогнутость отдельных линий выросла: {axis}"
        # Новое: сравниваем КАЖДУЮ линию с ближайшей строкой/колонкой,
        # а не только медиану/максимум всей таблицы.
        before_rules = old.get("rules", [])
        after_rules = new.get("rules", [])
        used = set()
        matched = 0
        for previous in before_rules:
            options = [(abs(previous["level_px"] - current["level_px"]), i, current)
                       for i, current in enumerate(after_rules) if i not in used]
            if not options:
                continue
            distance, index, current = min(options, key=lambda item: item[0])
            if distance > 7.0:
                continue
            used.add(index)
            matched += 1
            original_curve = previous["curvature_px"]
            corrected_curve = current["curvature_px"]
            # Допустим только незначительный шум измерения (<0.45px).
            if corrected_curve > original_curve + max(.45, original_curve * .08):
                return False, (f"Ухудшена отдельная линия {axis} "
                               f"на уровне {previous['level_px']}px: "
                               f"{original_curve:.2f} -> {corrected_curve:.2f}px")
        if len(before_rules) >= 4 and matched < .8 * len(before_rules):
            return False, f"Не удалось сопоставить линии до и после: {axis}"
    old_max = max(before["horizontal"]["max_px"], before["vertical"]["max_px"])
    new_max = max(after["horizontal"]["max_px"], after["vertical"]["max_px"])
    old_med = sum(before[a]["median_px"] for a in ("horizontal", "vertical"))
    new_med = sum(after[a]["median_px"] for a in ("horizontal", "vertical"))
    if new_max > old_max - .3 and new_med > old_med - .2:
        return False, "Нет подтверждённого улучшения геометрии"
    return True, "Сетка стала прямее и сохранила линии"


def rectify(image: np.ndarray, detection: dict, *, grid: bool = True,
            columns: int = 4, mode: str = "auto") -> tuple[np.ndarray, np.ndarray, dict]:
    """Возвращает (итог, гомография, отчёт). При UNCERTAIN — исключение.

    Для проверки создаём временный перспективный preview, но конечные
    пиксели grid берём напрямую из исходного фото с помощью обратной карты.
    """
    if mode not in {"auto", "homography", "grid"}:
        raise ValueError(f"Неподдерживаемый режим: {mode}")
    if detection["mode"] == "UNCERTAIN" or detection["corners"] is None:
        raise ValueError("Нет проверенного четырёхугольника, геометрию менять нельзя")
    corners = np.asarray(detection["corners"], dtype=np.float32)
    tl, tr, br, bl = corners
    top_width = float(np.linalg.norm(tr-tl))
    bottom_width = float(np.linalg.norm(br-bl))
    left_height = float(np.linalg.norm(bl-tl))
    right_height = float(np.linalg.norm(br-tr))
    # Немного увеличиваем изображение для чтения мелких цифр. Не больше 1.07x.
    width = int(round(max(top_width, bottom_width) * 1.07))
    height = int(round(max(left_height, right_height) * 1.04))
    width = max(160, min(width, 3200))
    height = max(160, min(height, 4800))
    margin = max(10, int(round(width*.027)))
    # Нижняя ступенька с итогом выступает под правой частью полной таблицы.
    footer_height = max(44, int(round(height*.05)))
    out_w = width + margin * 2
    out_h = height + margin + footer_height
    if out_w * out_h > 24_000_000:
        raise ValueError("Слишком большая выходная карта")
    dst = np.float32([[margin, margin], [margin + width, margin],
                      [margin + width, margin + height], [margin, margin + height]])
    H = cv2.getPerspectiveTransform(corners, dst)
    perspective = cv2.warpPerspective(
        image, H, (out_w, out_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    report = {"version": VERSION, "method": "homography", "mode": detection["mode"],
              "output_size": [out_w, out_h], "expected_columns": columns,
              "grid_assessment": {"status": "not_run"}}
    if not grid or mode == "homography":
        report["grid_assessment"] = {"status": "disabled", "reason": "Выбран режим только Homography"}
        report["dewarp_necessity"] = {"decision": "NOT_RUN", "needs_dewarp": False}
        return perspective, perspective, report
    # НОВОЕ: после перспективы отдельно проверить, НУЖЕН ли Dewarping.
    # В auto не запускаем сложную сеточную коррекцию на ровных таблицах.
    necessity = assess_dewarp_necessity(perspective)
    report["dewarp_necessity"] = necessity
    if mode == "auto" and not necessity["needs_dewarp"]:
        report["grid_assessment"] = {
            "status": "skipped_not_needed", "reason": necessity["reason"]
        }
        return perspective, perspective, report
    # mode='grid' — принудительный эксперимент, но валидатор НЕ отключаем.
    # Сначала выпрямляем горизонтальные линии (их число зависит от строк).
    try:
        h, w = perspective.shape[:2]
        if h < 80 or w < 80:
            raise ValueError("Недостаточно разрешения для grid dewarp")
        yy, xx = np.indices((h, w), dtype=np.float32)
        horizontal, _ = find_rules(perspective)
        if len(horizontal) < 4:
            raise ValueError("Мало горизонтальных линий")
        vy, ylevels = ordered_axis_map(horizontal, h, w)
        horizontal_preview = cv2.remap(
            perspective, xx, vy, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE)
        vertical, _ = find_rules(horizontal_preview, vertical=True)
        selected, selection = select_vertical_grid(vertical, columns+1, w)
        if len(selected) != columns+1:
            raise ValueError(f"Не подтверждены {columns+1} вертикальных границ")
        vx, xlevels = ordered_axis_map(selected, w, h)
        local_x = vx.T.copy()
        local_y = cv2.remap(
            vy, local_x, yy, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE)
        quality = map_quality(local_x, local_y)
        displacement = float(np.percentile(np.hypot(local_x-xx, local_y-yy), 95))
        # Составляем обратное отображение в исходный JPEG (один ресемплинг).
        destination = np.stack([local_x, local_y], axis=2).reshape(-1, 1, 2)
        source = cv2.perspectiveTransform(destination, np.linalg.inv(H)).reshape(h, w, 2)
        candidate = cv2.remap(
            image, source[:, :, 0].astype(np.float32),
            source[:, :, 1].astype(np.float32), cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        before = _curvature_report(perspective)
        after = _curvature_report(candidate)
        accept, reason = _validate_candidate(before, after, quality, displacement)
        report["grid_assessment"] = {
            "status": "accepted" if accept else "rejected",
            "reason": reason,
            "horizontal_count": len(horizontal),
            "vertical_selection": selection,
            "horizontal_levels": [round(float(x), 2) for x in ylevels],
            "vertical_levels": [round(float(x), 2) for x in xlevels],
            "map_quality": quality,
            "displacement_p95_px": round(displacement, 3),
            "before": before,
            "after": after,
        }
        if accept:
            report["method"] = "homography_plus_grid_dewarp"
            return candidate, perspective, report
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        report["grid_assessment"] = {"status": "not_available", "reason": str(error)}
    return perspective, perspective, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Исправление перспективы и сетки чека")
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment_outputs/rectified.png"))
    parser.add_argument("--mode", choices=("auto", "homography", "grid"), default="auto",
                        help="auto: Homography + детектор необходимости Dewarping; grid: принудительная попытка")
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    if args.columns < 2 or args.columns > 12:
        parser.error("--columns должен быть от 2 до 12")
    image = read_image(args.photo)
    detection = detect_geometry(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detected = args.output.with_name("detected.png")
    write_png(detected, draw_detection(image, detection))
    report = {"input": str(args.photo), "output": str(args.output),
              "detector": detection}
    if detection["mode"] == "UNCERTAIN":
        report["error"] = "Контур не подтверждён; исходное изображение не изменено"
        args.output.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(report["error"])
        return 2
    do_grid = args.mode in ("auto", "grid")
    result, perspective, correction = rectify(
        image, detection, grid=do_grid, columns=args.columns, mode=args.mode
    )
    report["improver"] = correction
    write_png(args.output, result)
    perspective_path = args.output.with_name("perspective.png")
    write_png(perspective_path, perspective)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(args.output), "preview": str(detected),
                      "perspective": str(perspective_path),
                      "report": str(report_path),
                      "method": correction["method"],
                      "grid": correction["grid_assessment"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
