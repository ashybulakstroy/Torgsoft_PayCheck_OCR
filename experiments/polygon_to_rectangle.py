"""
Эксперимент: Жёсткое выпрямление зелёного контура (source_polygon) в прямоугольник

Цель: Взять зелёный контур таблицы (source_polygon) и преобразовать его в идеальный прямоугольник
с углами 90° с использованием перспективного преобразования.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

def extract_4_corners_from_polygon(polygon, original_shape):
    """
    Извлекает 4 истинных угла таблицы из source_polygon.
    
    Метод:
    1. Определяем левую и правую границы полигона (минимальная и максимальная X координаты)
    2. Определяем верхнюю и нижюю границы (минимальная и максимальная Y координаты)
    3. Находим истинные углы на пересечении границ
    
    Возвращает 4 точки в порядке: TL, TR, BR, BL
    """
    if polygon is None or len(polygon) < 4:
        return None
    
    polygon_array = np.array(polygon, dtype=np.float32)
    
    # Находим экстремальные координаты
    min_x = np.min(polygon_array[:, 0])
    max_x = np.max(polygon_array[:, 0])
    min_y = np.min(polygon_array[:, 1])
    max_y = np.max(polygon_array[:, 1])
    
    # Используем source_body_corners из метаданных для получения истинных углов
    # (уже корректно вычисленные координаты таблицы)
    
    # Ищем source_body_corners в метаданных
    # Если недоступны, используем экстремальные точки полигона
    h, w = original_shape
    
    # Находим точки полигона с экстремальными координатами
    left_points = polygon_array[polygon_array[:, 0] <= min_x + 5]  # Точки слева
    right_points = polygon_array[polygon_array[:, 0] >= max_x - 5]  # Точки справа
    top_points = polygon_array[polygon_array[:, 1] <= min_y + 5]  # Точки сверху
    bottom_points = polygon_array[polygon_array[:, 1] >= max_y - 5]  # Точки снизу
    
    # TL: левый верхний - точка с минимальной X и минимальной Y
    tl_candidates = polygon_array[
        (polygon_array[:, 0] <= np.percentile(polygon_array[:, 0], 25)) &
        (polygon_array[:, 1] <= np.percentile(polygon_array[:, 1], 25))
    ]
    if len(tl_candidates) > 0:
        tl = tl_candidates[np.argmax(tl_candidates[:, 0])]  # Самая левая из верхних
    else:
        tl = polygon_array[np.argmin(polygon_array[:, 0] + polygon_array[:, 1])]
    
    # TR: правый верхний - точка с максимальной X и минимальной Y
    tr_candidates = polygon_array[
        (polygon_array[:, 0] >= np.percentile(polygon_array[:, 0], 75)) &
        (polygon_array[:, 1] <= np.percentile(polygon_array[:, 1], 25))
    ]
    if len(tr_candidates) > 0:
        tr = tr_candidates[np.argmin(tr_candidates[:, 0])]  # Самая правая из верхних
    else:
        # Используем точку с максимальной X среди верхних точек
        max_x_idx = np.argmax(polygon_array[:, 0])
        tr = polygon_array[max_x_idx]
    
    # BR: правый нижний - точка с максимальной X и максимальной Y
    br_candidates = polygon_array[
        (polygon_array[:, 0] >= np.percentile(polygon_array[:, 0], 75)) &
        (polygon_array[:, 1] >= np.percentile(polygon_array[:, 1], 75))
    ]
    if len(br_candidates) > 0:
        br = br_candidates[np.argmax(br_candidates[:, 0])]  # Самая правая из нижних
    else:
        br = polygon_array[np.argmax(polygon_array[:, 0] + polygon_array[:, 1])]
    
    # BL: левый нижний - точка с минимальной X и максимальной Y
    bl_candidates = polygon_array[
        (polygon_array[:, 0] <= np.percentile(polygon_array[:, 0], 25)) &
        (polygon_array[:, 1] >= np.percentile(polygon_array[:, 1], 75))
    ]
    if len(bl_candidates) > 0:
        bl = bl_candidates[np.argmin(bl_candidates[:, 0])]  # Самая левая из нижних
    else:
        # Используем точку с минимальной X среди нижних точек
        min_x_idx = np.argmin(polygon_array[:, 0])
        bl = polygon_array[min_x_idx]
    
    corners = np.float32([
        [tl[0], tl[1]],  # TL
        [tr[0], tr[1]],  # TR
        [br[0], br[1]],  # BR
        [bl[0], bl[1]]   # BL
    ])
    
    return corners

def transform_to_rectangle(image, src_corners, output_size):
    """
    Преобразует исходное изображение используя 4 угла в идеальный прямоугольник.
    """
    h, w = output_size
    
    dst_pts = np.float32([
        [0, 0],         # TL
        [w - 1, 0],     # TR
        [w - 1, h - 1], # BR
        [0, h - 1]      # BL
    ])
    
    src_pts = np.array(src_corners, dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rectified, matrix

def measure_angle_accuracy(image):
    """Оценивает точность углов в изображении."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, 0.0, 0.0
    
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect).astype(np.int32)
        approx = box.reshape(-1, 1, 2)
    
    if len(approx) != 4:
        return 0.0, 0.0, 0.0
    
    angles = []
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i+1) % 4][0]
        p3 = approx[(i+2) % 4][0]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product > 0:
            cos_angle = np.dot(v1, v2) / norm_product
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(abs(angle - 90))
    
    if not angles:
        return 0.0, 0.0, 0.0
    
    avg_deviation = float(np.mean(angles))
    max_deviation = float(np.max(angles))
    rectangularity = max(0, 1 - avg_deviation / 90)
    
    return avg_deviation, max_deviation, rectangularity

def process_with_polygon_corners(todo_id_dir, output_dir):
    """Обрабатывает чек, используя зелёный контур (source_polygon) для выпрямления."""
    original_path = todo_id_dir / "original.png"
    rect_path = todo_id_dir / "rectification.json"
    
    if not original_path.exists() or not rect_path.exists():
        return None
    
    image = cv2.imread(str(original_path))
    if image is None:
        return None
    
    with open(rect_path) as f:
        rect_data = json.load(f)
    
    output_size = rect_data.get("output_size", [535, 902])
    h, w = output_size[0], output_size[1]
    
    # Получаем зелёный контур (source_polygon)
    polygon = rect_data.get("source_polygon", [])
    
    if not polygon or len(polygon) < 4:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": "source_polygon not found or has insufficient points"
        }
    
    # Извлекаем 4 угла из полигона
    src_corners = extract_4_corners_from_polygon(polygon)
    
    if src_corners is None:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": "Could not extract 4 corners from polygon"
        }
    
    # Применяем перспективное преобразование
    try:
        rectified, transform_matrix = transform_to_rectangle(image, src_corners, (w, h))
    except Exception as e:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": f"Transform failed: {str(e)}"
        }
    
    # Оцениваем результат
    avg_dev, max_dev, rectangularity = measure_angle_accuracy(rectified)
    
    # Создаём визуализацию
    before = image.copy()
    
    # Рисуем зелёный полигон
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(before, [polygon_np], True, (0, 255, 0), 3)
    
    # Рисуем извлечённые 4 угла
    corners_int = np.array(src_corners, dtype=np.int32)
    cv2.polylines(before, [corners_int], True, (255, 0, 0), 2)
    for i, corner in enumerate(src_corners):
        cv2.circle(before, (int(corner[0]), int(corner[1])), 10, (255, 0, 0), -1)
        cv2.putText(before, f"C{i+1}", (int(corner[0]) + 12, int(corner[1]) + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.putText(before, "BEFORE (green=polygon, blue=extracted 4 corners)", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # После выпрямления
    after = rectified.copy()
    cv2.putText(after, f"AFTER: {w}x{h}px", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(after, f"Rectangularity: {rectangularity:.3f}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(after, f"Angle deviation: {avg_dev:.1f}°", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Боковое сравнение
    h_img = max(before.shape[0], after.shape[0]) + 60
    w_total = before.shape[1] + after.shape[1] + 60
    comparison = np.ones((h_img, w_total, 3), dtype=np.uint8) * 240
    comparison[30:30+before.shape[0], 30:30+before.shape[1]] = before
    comparison[30:30+after.shape[0], before.shape[1]+50:before.shape[1]+50+after.shape[1]] = after
    
    cv2.putText(comparison, "BEFORE", (30, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.putText(comparison, "AFTER (ideal rectangle)", (before.shape[1]+50, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    return {
        "file": todo_id_dir.name,
        "status": "processed",
        "output_size": [h, w],
        "avg_angle_deviation": round(float(avg_dev), 1),
        "max_angle_deviation": round(float(max_dev), 1),
        "rectangularity": round(float(rectangularity), 3),
        "original_polygon_points": len(polygon),
        "extracted_corners": src_corners.tolist(),
        "comparison": comparison
    }

def generate_report(results, output_dir):
    """Создаёт HTML-отчёт."""
    
    total = len([r for r in results if "error" not in r])
    successful = sum(1 for r in results if not "error" in r)
    
    valid_rects = [r for r in results if "rectangularity" in r]
    avg_rect = f"{np.mean([r['rectangularity'] for r in valid_rects]):.3f}" if valid_rects else "N/A"
    avg_angle = f"{np.mean([r['avg_angle_deviation'] for r in valid_rects]):.1f}°" if valid_rects else "N/A"
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Эксперимент: Жёсткое выпрямление зелёного контура</title>
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
        .img-container img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <h1>🎯 Эксперимент: Жёсткое выпрямление зелёного контура (source_polygon)</h1>
    
    <div class="warning">
        <strong>Важно:</strong> Выпрямление выполняется на основе зелёного контура (source_polygon), 
        который извлекается из метаданных. Из полигона с N точками извлекаются 4 угла, 
        затем применяется перспективное преобразование для создания идеального прямоугольника.
    </div>
    
    <div>
        <div class="stat-box"><div class="stat-value">{len(valid_rects)}</div><div class="stat-label">Файлов успешно обработано</div></div>
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
        
        status_class = "success" if r.get("rectangularity", 0) > 0.8 else "failure"
        angle_class = "success" if r.get("avg_angle_deviation", 90) < 10 else "failure"
        
        html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Исходный контур</td><td>{r.get('original_polygon_points', 'N/A')} точек</td></tr>
            <tr><td>Размер выхода</td><td>{r.get('output_size', 'N/A')} пикселей</td></tr>
            <tr><td>Прямоугательность</td><td class="{status_class}">{r.get('rectangularity', 'N/A')}</td></tr>
            <tr><td>Сред. отклонение угла</td><td class="{angle_class}">{r.get('avg_angle_deviation', 'N/A')}°</td></tr>
            <tr><td>Макс. отклонение угла</td><td>{r.get('max_angle_deviation', 'N/A')}°</td></tr>
        </table>
        <img src="{r['file']}_comparison.jpg" alt="Comparison for {r['file']}">
    </div>
"""
    
    html += """
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999;">
        <p>Экспериментальный отчёт создан в рамках проекта Prj_35_OCR_prep</p>
    </footer>
</body>
</html>
"""
    
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Визуальный отчёт создан: {output_dir.absolute() / 'index.html'}")

def main():
    todo_dir = Path("data/to_do")
    output_dir = Path("experiment_outputs/todo_polygon_to_rectangle")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Эксперимент: Жёсткое выпрямление зелёного контура (source_polygon)")
    print(f"📂 Обрабатываем файлы из: {todo_dir.absolute()}")
    print(f"💾 Результаты сохраняются в: {output_dir.absolute()}")
    print()
    
    results = []
    
    todo_subdirs = sorted([d for d in todo_dir.iterdir() if d.is_dir()])
    
    for i, todo_dir_item in enumerate(todo_subdirs):
        print(f"[{i+1}/{len(todo_subdirs)}] {todo_dir_item.name}")
        
        result = process_with_polygon_corners(todo_dir_item, output_dir)
        
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
    generate_report(results, output_dir)
    
    # Открываем в браузере
    import webbrowser
    report_path = output_dir / "index.html"
    webbrowser.open(str(report_path.absolute()))
    
    print(f"\n✅ Отчёт открыт в браузере: {report_path.absolute()}")

if __name__ == "__main__":
    main()