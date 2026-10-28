"""
Эксперимент: Жёсткое выпрямление товарной таблицы до идеального прямоугольника

Использует fitted_top, fitted_bottom, fitted_left, fitted_right из метаданных
для вычисления реальных углов таблицы на исходном изображении.

Алгоритм:
1. fitted_left и fitted_right описывают левую и правую границы (x = f(y))
2. fitted_top описывает верхнюю границу (y = f(x))
3. fitted_bottom описывает нижнюю границу (y = f(x))
4. Пересечение левой/правой границы с верхней/нижней даёт 4 угла:
   - TL = левая граница ∩ верхняя граница
   - TR = правая граница ∩ верхняя граница
   - BR = правая граница ∩ нижняя граница
   - BL = левая граница ∩ нижняя граница
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

def compute_table_corners(metadata, original_shape):
    """
    Вычисляет 4 угла таблицы на основе fitted_top, fitted_bottom, fitted_left, fitted_right.
    
    Координаты нормализованы в диапазоне [0, 1], где:
    - x: 0 = слева, 1 = справа (относительно ширины исходного изображения)
    - y: 0 = сверху, 1 = снизу (относительно высоты исходного изображения)
    
    fitted_left: x = f(y) или константа для вертикальной границы
    fitted_right: x = f(y) или константа
    fitted_top: y = f(x) или константа
    fitted_bottom: y = f(x) или константа
    """
    fitted_top = metadata.get("fitted_top", [])
    fitted_bottom = metadata.get("fitted_bottom", [])
    fitted_left = metadata.get("fitted_left", [])
    fitted_right = metadata.get("fitted_right", [])
    
    if not all([fitted_top, fitted_bottom, fitted_left, fitted_right]):
        return None
    
    h, w = original_shape
    
    # Извлекаем коэффициенты
    # fitted_top: y = a + b*x (или y = 0, если 1 коэффициент)
    if len(fitted_top) >= 2:
        a_top, b_top = fitted_top[0], fitted_top[1]
        y_top_func = lambda x: a_top + b_top * x
    elif len(fitted_top) == 1:
        y_top_func = lambda x: fitted_top[0]
    else:
        return None
    
    # fitted_bottom: y = a + b*x
    if len(fitted_bottom) >= 2:
        a_bottom, b_bottom = fitted_bottom[0], fitted_bottom[1]
        y_bottom_func = lambda x: a_bottom + b_bottom * x
    elif len(fitted_bottom) == 1:
        y_bottom_func = lambda x: fitted_bottom[0]
    else:
        return None
    
    # fitted_left: x = a + b*y (или x = const)
    if len(fitted_left) >= 2:
        a_left, b_left = fitted_left[0], fitted_left[1]
        x_left_func = lambda y: a_left + b_left * y
    elif len(fitted_left) == 1:
        x_left_func = lambda y: fitted_left[0]
    else:
        return None
    
    # fitted_right: x = a + b*y (или x = const)
    if len(fitted_right) >= 2:
        a_right, b_right = fitted_right[0], fitted_right[1]
        x_right_func = lambda y: a_right + b_right * y
    elif len(fitted_right) == 1:
        x_right_func = lambda y: fitted_right[0]
    else:
        return None
    
    # Находим углы путем пересечения границ
    
    # TL: x_left(y) и y_top(x) - итерационное решение
    # y = y_top(x), x = x_left(y)
    # Итерации:
    y = 0.0
    for _ in range(10):
        x = x_left_func(y)
        y = y_top_func(x)
    x_TL, y_TL = x, y
    
    # TR: x_right(y) и y_top(x)
    y = 0.0
    for _ in range(10):
        x = x_right_func(y)
        y = y_top_func(x)
    x_TR, y_TR = x, y
    
    # BR: x_right(y) и y_bottom(x)
    y = 1.0
    for _ in range(10):
        x = x_right_func(y)
        y = y_bottom_func(x)
    x_BR, y_BR = x, y
    
    # BL: x_left(y) и y_bottom(x)
    y = 1.0
    for _ in range(10):
        x = x_left_func(y)
        y = y_bottom_func(x)
    x_BL, y_BL = x, y
    
    # Переводим из нормализованных координат в пиксели
    corners_norm = np.float32([
        [x_TL, y_TL],  # TL
        [x_TR, y_TR],  # TR
        [x_BR, y_BR],  # BR
        [x_BL, y_BL]   # BL
    ])
    
    corners_pixel = corners_norm.copy()
    corners_pixel[:, 0] *= w
    corners_pixel[:, 1] *= h
    
    return corners_pixel

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

def process_with_computed_corners(todo_id_dir, output_dir):
    """Обрабатывает чек, вычисляя углы из fitted функций."""
    original_path = todo_id_dir / "original.png"
    rect_path = todo_id_dir / "rectification.json"
    
    if not original_path.exists() or not rect_path.exists():
        return None
    
    image = cv2.imread(str(original_path))
    if image is None:
        return None
    
    # Загружаем метаданные
    with open(rect_path) as f:
        rect_data = json.load(f)
    
    output_size = rect_data.get("output_size", [535, 902])
    h, w = output_size[0], output_size[1]
    
    # Вычисляем углы таблицы на исходном изображении
    original_shape = image.shape[:2]  # (height, width)
    corners = compute_table_corners(rect_data, original_shape)
    
    if corners is None:
        # Используем source_body_corners как fallback
        corners = np.array(rect_data.get("source_body_corners", []), dtype=np.float32)
        if len(corners) != 4:
            return {
                "file": todo_id_dir.name,
                "status": "error",
                "error": "Could not compute or find table corners"
            }
    
    # Перспективное преобразование
    src_pts = corners
    dst_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])
    
    try:
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified = cv2.warpPerspective(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
    except Exception as e:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": f"Perspective transform failed: {str(e)}"
        }
    
    # Оцениваем результат
    avg_dev, max_dev, rectangularity = measure_angle_accuracy(rectified)
    
    # Создаём визуализацию
    before = image.copy()
    
    # Рисуем исходный полигон
    polygon = np.array(rect_data.get("source_polygon", []), dtype=np.int32).reshape((-1, 1, 2))
    if len(polygon) > 0:
        cv2.polylines(before, [polygon], True, (0, 255, 0), 2)  # Зелёный полигон
    
    # Рисуем вычисленные углы таблицы
    corners_int = np.array(corners, dtype=np.int32)
    cv2.polylines(before, [corners_int], True, (0, 0, 255), 3)  # Красная линия для углов
    for i, corner in enumerate(corners):
        cv2.circle(before, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), -1)
        cv2.putText(before, f"P{i+1}", (int(corner[0]) + 12, int(corner[1]) + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(before, "BEFORE (green=poly, red=table corners)", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # После выпрямления
    after = rectified.copy()
    cv2.putText(after, f"AFTER: {w}x{h}px", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(after, f"Rectangularity: {rectangularity:.3f}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(after, f"Angle deviation: {avg_dev:.1f}° avg", (10, 80), 
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
        "table_corners": corners.tolist(),
        "comparison": comparison
    }

def generate_report(results, output_dir):
    """Создаёт HTML-отчёт."""
    
    rects = [r.get("rectangularity", 0) for r in results if "rectangularity" in r]
    avg_rect = f"{np.mean(rects):.3f}" if rects else "N/A"
    avg_angle = f"{np.mean([r.get('avg_angle_deviation', 0) for r in results if 'avg_angle_deviation' in r]):.1f}°" if any('avg_angle_deviation' in r for r in results) else "N/A"
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Эксперимент: Жёсткое выпрямление с вычисленными углами</title>
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
    <h1>🎯 Эксперимент: Жёсткое выпрямление с вычисленными углами таблицы</h1>
    
    <div class="warning">
        <strong>Важно:</strong> Углы таблицы вычисляются из fitted функций (fitted_top, fitted_bottom, 
        fitted_left, fitted_right) для точного определения реальных углов печатной таблицы на исходном изображении.
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
        
        status_class = "success" if r.get("rectangularity", 0) > 0.8 else "failure"
        angle_class = "success" if r.get("avg_angle_deviation", 90) < 10 else "failure"
        
        html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Размер выхода</td><td>{r.get('output_size', 'N/A')} пикселей</td></tr>
            <tr><td>Прямоугательность</td><td class="{status_class}">{r.get('rectangularity', 'N/A')}</td></tr>
            <tr><td>Сред. отклонение угла</td><td class="{angle_class}">{r.get('avg_angle_deviation', 'N/A')}°</td></tr>
            <tr><td>Макс. отклонение угла</td><td>{r.get('max_angle_deviation', 'N/A')}°</td></tr>
        </table>
        <img src="{r['file']}_comparison.jpg" alt="Comparison for {r['file']}">
    </div>
"""
    
    html += f"""
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
    output_dir = Path("experiment_outputs/todo_computed_corners")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Эксперимент: Жёсткое выпрямление с вычисленными углами")
    print(f"📂 Обрабатываем файлы из: {todo_dir.absolute()}")
    print(f"💾 Результаты сохраняются в: {output_dir.absolute()}")
    print()
    
    results = []
    
    todo_subdirs = sorted([d for d in todo_dir.iterdir() if d.is_dir()])
    
    for i, todo_dir_item in enumerate(todo_subdirs):
        print(f"[{i+1}/{len(todo_subdirs)}] {todo_dir_item.name}")
        
        result = process_with_computed_corners(todo_dir_item, output_dir)
        
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