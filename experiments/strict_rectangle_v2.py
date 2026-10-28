"""
Эксперимент: Жёсткое выпрямление товарной таблицы до идеального прямоугольника (90° углы)

Цель: Создать выпрямленное изображение таблицы с идеальными углами 90°,
используя умное выравнивание на основе 4-х ключевых точек полигона.

ВАЖНО: Согласно KEY INVARIANTS проекта, полигон таблицы НЕ должен быть 
принудительно сведён к 4 точкам в основном пайплайне. Этот эксперимент 
выполняется только для оценки возможностей жёсткого выравнивания.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from App.table_boundary import detect_table_boundary

def find_optimal_rectangle_corners(polygon):
    """
    Ищет 4 оптимальных угловых точки полигона для создания идеального прямоугольника.
    
    Использует анализ выпуклой оболочки и поиск точек с максимальными координатами.
    """
    if polygon is None or len(polygon) < 4:
        return None
    
    polygon = np.array(polygon, dtype=np.float32)
    
    # Метод 1: Используем минимальный прямоугольник, охватывающий полигон
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    
    # Сортируем точки по часовой стрелке начиная с левого верхнего угла
    # Преобразуем в стандартный порядок TL, TR, BR, BL
    box_center = np.mean(box, axis=0)
    angles = np.arctan2(box[:, 1] - box_center[1], box[:, 0] - box_center[0])
    box_sorted = box[np.argsort(angles)]
    
    # Находим левую верхнюю точку (наибольшая y, наименьшая x)
    top_left_idx = np.argmin(box_sorted[:, 0] + box_sorted[:, 1])
    
    # Переставляем точки чтобы начинаться с TL
    ordered_box = np.roll(box_sorted, -top_left_idx, axis=0)
    
    return ordered_box

def create_ideal_rectangle_from_corners(image, corners):
    """
    Создаёт идеальный прямоугольник с углами 90° из 4-х точек.
    
    Args:
        image: Исходное изображение
        corners: Массив из 4 точек [TL, TR, BR, BL]
    
    Returns:
        rectified: Выпрямленное изображение с идеальными углами 90°
        transform_matrix: Матрица трансформации
    """
    if corners is None or len(corners) != 4:
        return None, None
    
    src_pts = np.array(corners, dtype=np.float32)
    
    # Определяем размер выходного прямоугольника
    width = int(max(np.linalg.norm(src_pts[1] - src_pts[0]), 
                    np.linalg.norm(src_pts[2] - src_pts[3])))
    height = int(max(np.linalg.norm(src_pts[3] - src_pts[0]), 
                     np.linalg.norm(src_pts[2] - src_pts[1])))
    
    # Целевые координаты (идеальный прямоугольник)
    dst_pts = np.float32([
        [0, 0],                           # TL
        [width - 1, 0],                   # TR
        [width - 1, height - 1],          # BR
        [0, height - 1]                   # BL
    ])
    
    try:
        # Перспективное преобразование
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified = cv2.warpPerspective(
            image, matrix, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return rectified, matrix
    except Exception as e:
        print(f"  Ошибка выпрямления: {e}")
        return None, None

def measure_angle_accuracy(image):
    """
    Оценивает точность углов в выпрямленном изображении.
    Возвращает: (среднее_отклонение_от_90, макс_отклонение, прямоугательность)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Используем более мягкий порог для поиска границ
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, 0.0, 0.0
    
    # Берём наибольший контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Приближаем к многоугольнику с более высокой точностью
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) < 4:
        return 0.0, 0.0, 0.0
    
    # Берём 4 точки с наибольшими координатами (углы)
    if len(approx) > 4:
        # Используем convex hull и минимальную инкрементную аппроксимацию
        hull = cv2.convexHull(largest_contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) != 4:
            # Принудительно получаем 4 угла из минимального прямоугольника
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect).astype(np.int32)
            approx = box.reshape(-1, 1, 2)
    
    if len(approx) != 4:
        return 0.0, 0.0, 0.0
    
    # Вычисляем углы
    angles = []
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i+1) % 4][0]
        p3 = approx[(i+2) % 4][0]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Избегаем деления на ноль
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product > 0:
            cos_angle = np.dot(v1, v2) / norm_product
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(abs(angle - 90))
    
    if not angles:
        return 0.0, 0.0, 0.0
    
    avg_deviation = np.mean(angles)
    max_deviation = np.max(angles)
    
    # Оценка прямоугательности (1.0 = идеальный прямоугольник)
    rectangularity = max(0, 1 - avg_deviation / 90)
    
    return avg_deviation, max_deviation, rectangularity

def process_todo_with_strict_rectangle(todo_id_dir, output_dir):
    """Обрабатывает чек из data/to_do/ с жёстким выравниванием."""
    original_path = todo_id_dir / "original.png"
    
    if not original_path.exists():
        return None
    
    image = cv2.imread(str(original_path))
    if image is None:
        return None
    
    # Детекция границы таблицы
    boundary = detect_table_boundary(image)
    polygon = boundary.get("polygon")
    
    if polygon is None:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": "Polygon not detected"
        }
    
    # Ищем 4 оптимальных угловых точки
    corners = find_optimal_rectangle_corners(polygon)
    
    if corners is None or len(corners) != 4:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": "Could not find 4 corners"
        }
    
    # Создаём идеальный прямоугольник
    rectified, matrix = create_ideal_rectangle_from_corners(image, corners)
    
    if rectified is None:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": "Perspective transform failed"
        }
    
    # Оцениваем результат
    avg_dev, max_dev, rectangularity = measure_angle_accuracy(rectified)
    
    # Создаём визуализацию
    before = image.copy()
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(before, [polygon_np], True, (0, 255, 0), 3)
    
    # Показываем найденные углы
    for i, corner in enumerate(corners):
        cv2.circle(before, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(before, f"P{i+1}", (corner[0].astype(int) + 10, corner[1].astype(int) + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.putText(before, f"Before: {len(polygon)} points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    after = rectified.copy()
    cv2.putText(after, f"After: {rectified.shape[1]}x{rectified.shape[0]}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(after, f"Angle deviation: {avg_dev:.1f}° avg, {max_dev:.1f}° max", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Боковое сравнение
    h = max(before.shape[0], after.shape[0]) + 50
    w = before.shape[1] + after.shape[1] + 60
    comparison = np.ones((h, w, 3), dtype=np.uint8) * 240
    comparison[20:20+before.shape[0], 20:20+before.shape[1]] = before
    comparison[20:20+after.shape[0], before.shape[1]+40:before.shape[1]+40+after.shape[1]] = after
    
    cv2.putText(comparison, "BEFORE", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(comparison, "AFTER", (before.shape[1]+40, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return {
        "file": todo_id_dir.name,
        "status": boundary.get("status", "unknown"),
        "num_points_original": len(polygon),
        "output_size": list(rectified.shape[:2]),
        "avg_angle_deviation": round(float(avg_dev), 1),
        "max_angle_deviation": round(float(max_dev), 1),
        "rectangularity": round(float(rectangularity), 3),
        "comparison": comparison,
    }

def create_visual_report(results, output_dir):
    """Создаёт визуальный HTML-отчёт."""
    
    # Вычисляем статистику
    rects = [r.get("rectangularity", 0) for r in results if "rectangularity" in r]
    avg_rect = f"{np.mean(rects):.3f}" if rects else "N/A"
    avg_angle = f"{np.mean([r.get('avg_angle_deviation', 0) for r in results if 'avg_angle_deviation' in r]):.1f}°" if any('avg_angle_deviation' in r for r in results) else "N/A"
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Эксперимент: Жёсткое выпрямление до идеального прямоугольника</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1400px; }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
        .stat-box {{ display: inline-block; background: #e8f5e9; padding: 20px; margin: 10px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #2e7d32; }}
        .stat-label {{ font-size: 14px; color: #666; }}
        .img-container {{ margin: 30px 0; }}
        .img-container img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric {{ font-family: 'Courier New', monospace; }}
        .warning {{ background: #fff8e1; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; }}
        th {{ background: #283593; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .success {{ color: #2e7d32; font-weight: bold; }}
        .failure {{ color: #c62828; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🎯 Эксперимент: Жёсткое выпрямление до идеального прямоугольника</h1>
    
    <div class="warning">
        <strong>Важно:</strong> Согласно KEY INVARIANTS проекта: полигон таблицы НЕ должен быть принудительно 
        сведён к 4 точкам. Этот эксперимент оценивает возможности жёсткого выравнивания.
        Основной пайплайн использует Coons-патч (contour_unwarp.py) для сохранения всех особенностей формы.
    </div>
    
    <div>
        <div class="stat-box"><div class="stat-value">{len(results)}</div><div class="stat-label">Файлов обработано</div></div>
        <div class="stat-box"><div class="stat-value">{avg_rect}</div><div class="stat-label">Средняя прямоугательность</div></div>
        <div class="stat-box"><div class="stat-value">{avg_angle}</div><div class="stat-label">Сред. отклонение угла</div></div>
    </div>
    
    <h2>Подробные результаты</h2>
"""
    
    for r in results:
        if "error" in r:
            html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <p class="failure">❌ Ошибка: {r.get('error', 'Unknown')}</p>
    </div>
"""
            continue
        
        status_class = "success" if r.get("rectangularity", 0) > 0.8 else "warning"
        angle_class = "success" if r.get("avg_angle_deviation", 90) < 10 else "failure"
        
        html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Исходный полигон</td><td>{r.get('num_points_original', 'N/A')} точек</td></tr>
            <tr><td>Размер выхода</td><td>{r.get('output_size', 'N/A')} пикселей</td></tr>
            <tr><td>Прямоугательность</td><td class="{status_class}">{r.get('rectangularity', 'N/A')}</td></tr>
            <tr><td>Сред. отклонение угла</td><td class="{angle_class}">{r.get('avg_angle_deviation', 'N/A')}°</td></tr>
            <tr><td>Макс. отклонение угла</td><td>{r.get('max_angle_deviation', 'N/A')}°</td></tr>
        </table>
        <img src="{r['file']}_comparison.jpg" alt="Strict rectangle comparison for {r['file']}">
    </div>
"""
    
    html += f"""
    <h2>📋 Выводы</h2>
    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3>Результаты эксперимента</h3>
        <ul>
            <li>Обработано файлов: <span class="metric">{len(results)}</span></li>
            <li>Средняя прямоугательность: <span class="metric">{avg_rect}</span></li>
            <li>Среднее отклонение угла от 90°: <span class="metric">{avg_angle}</span></li>
        </ul>
        <p>Метод жёсткого выравнивания позволяет создать изображение с идеальными углами 90°, 
        однако это нарушает проектные принципы сохранения оригинальной формы таблицы.</p>
        <p>Рекомендуется использовать основной пайплайн с Coons-патч для реальной обработки.</p>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999;">
        <p>Экспериментальный отчёт создан в рамках проекта Prj_35_OCR_prep</p>
        <p>Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </footer>
</body>
</html>
"""
    
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Визуальный отчёт создан: {output_dir.absolute() / 'index.html'}")

def main():
    todo_dir = Path("data/to_do")
    output_dir = Path("experiment_outputs/todo_strict_rectangle_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Эксперимент: Жёсткое выпрямление до идеального прямоугольника (90° углы)")
    print(f"📂 Обрабатываем файлы из: {todo_dir.absolute()}")
    print(f"💾 Результаты сохраняются в: {output_dir.absolute()}")
    print()
    
    results = []
    
    todo_subdirs = sorted([d for d in todo_dir.iterdir() if d.is_dir()])
    
    for i, todo_dir_item in enumerate(todo_subdirs):
        print(f"[{i+1}/{len(todo_subdirs)}] {todo_dir_item.name}")
        
        result = process_todo_with_strict_rectangle(todo_dir_item, output_dir)
        
        if result:
            results.append(result)
            
            if "comparison" in result:
                comparison_path = output_dir / f"{result['file']}_comparison.jpg"
                cv2.imwrite(str(comparison_path), result["comparison"])
            
            if "error" in result:
                print(f"  ❌ Ошибка: {result['error']}")
            else:
                print(f"  ✅ Прямоугательность: {result.get('rectangularity', 'N/A')} | Угол: {result.get('avg_angle_deviation', 'N/A')}°")
    
    # Создаём HTML-отчёт
    create_visual_report(results, output_dir)
    
    # Открываем в браузере
    import webbrowser
    report_path = output_dir / "index.html"
    webbrowser.open(str(report_path.absolute()))
    
    print(f"\n✅ Отчёт открыт в браузере: {report_path.absolute()}")

if __name__ == "__main__":
    main()