"""
Эксперимент: Жёсткое выпрямление товарной таблицы до прямоугольника (90° углы) для файлов из data/to_do/

Цель: Создать выпрямленное изображение таблицы с идеальными прямыми углами 90°,
используя перспективное преобразование (homography) на основе четырёх углов полигона.

ВАЖНО: Согласно правилам проекта (KEY INVARIANTS), полигон таблицы НЕ должен быть 
принудительно сведён к 4 точкам. Этот эксперимент выполняется только для оценки 
возможности такого подхода.
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
from App.ocr_geometry import order_corners, perspective_warp

def create_strict_rectangular_crop(image, polygon, output_size=None):
    """
    Создаёт строго прямоугольный кроп таблицы с углами 90°.
    
    Использует перспективное преобразование на основе четырёх углов полигона.
    """
    if polygon is None or len(polygon) < 4:
        return None, None, "Not enough points for perspective transform"
    
    # Находим четыре угла полигона (минимальный прямоугольник)
    polygon_pts = np.array(polygon, dtype=np.float32)
    rect = cv2.minAreaRect(polygon_pts)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Используем 4 точки прямоугольника для перспективного преобразования
    src_pts = box.astype(np.float32)
    
    # Определяем размер выхода
    if output_size is None:
        width = int(rect[1][0])
        height = int(rect[1][1])
        output_size = (width, height)
    
    # Целевые координаты (идеальный прямоугольник)
    dst_pts = np.float32([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ])
    
    try:
        # Перспективное преобразование
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            image, matrix, output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return warped, matrix, None
    except Exception as e:
        return None, None, str(e)

def measure_rectangularity(image):
    """Оценивает, насколько изображение близко к идеальному прямоугольнику."""
    # Проверяем углы в 4-х точках изображения
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, "No contours found"
    
    # Берём наибольший контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Приближаем к многоугольнику
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) < 4:
        return 0.0, f"Only {len(approx)} corners found"
    
    # Проверяем углы
    angles = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i+1) % len(approx)][0]
        p3 = approx[(i+2) % len(approx)][0]
        
        # Векторы
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Угол
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle)
        angles.append(abs(angle_deg - 90))
    
    avg_deviation = np.mean(angles)
    max_deviation = np.max(angles)
    
    # Оценка качества (1.0 = идеальный прямоугольник)
    rectangularity_score = max(0, 1 - avg_deviation / 90)
    
    return rectangularity_score, f"Углы: {', '.join([f'{a:.1f}°' for a in angles])}, сред. отклонение: {avg_deviation:.2f}°"

def process_todo_image_strict(todo_id_dir):
    """Обрабатывает один чек из папки data/to_do/ с жёстким выпрямлением."""
    
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
    
    # Применяем строгое прямоугольное выпрямление
    rect_image, matrix, error = create_strict_rectangular_crop(image, polygon)
    
    if rect_image is None:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": f"Rectification failed: {error}"
        }
    
    # Оцениваем результат
    rectangularity_score, angle_info = measure_rectangularity(rect_image)
    
    # Создаём визуализацию
    stem = f"todo_{todo_id_dir.name}_strict"
    
    # Изображение "до"
    before = image.copy()
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(before, [polygon_np], True, (0, 255, 0), 3)
    
    # Показываем минимальный прямоугольник
    rect = cv2.minAreaRect(np.array(polygon, dtype=np.float32))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(before, [box], 0, (0, 255, 255), 2)
    
    cv2.putText(before, f"Before: {len(polygon)} points, rect={rectangularity_score_before:.3f}" if 'rectangularity_score_before' in locals() else f"Before: {len(polygon)} points", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(before, "BEFORE (original)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Изображение "после"
    after = rect_image.copy()
    cv2.putText(after, "AFTER (strict rectangle)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(after, angle_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Создаём боковое сравнение
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
        "rectangularity_after": round(float(rectangularity_score), 3),
        "angle_info": angle_info,
        "comparison": comparison,
        "output_size": rect_image.shape[:2]
    }

def main():
    todo_dir = Path("data/to_do")
    output_dir = Path("experiment_outputs/todo_strict_rectangle")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Эксперимент: Жёсткое выпрямление до прямоугольника (90° углы)")
    print(f"📂 Обрабатываем файлы из: {todo_dir.absolute()}")
    print(f"💾 Результаты сохраняются в: {output_dir.absolute()}")
    print()
    
    results = []
    
    # Обрабатываем подпапки
    todo_subdirs = sorted([d for d in todo_dir.iterdir() if d.is_dir()])
    
    for i, todo_dir_item in enumerate(todo_subdirs):
        print(f"[{i+1}/{len(todo_subdirs)}] {todo_dir_item.name}")
        
        result = process_todo_image_strict(todo_dir_item)
        
        if result:
            results.append(result)
            
            # Сохраняем визуализацию
            if "comparison" in result:
                comparison_path = output_dir / f"{result['file']}_comparison.jpg"
                cv2.imwrite(str(comparison_path), result["comparison"])
            
            print(f"  ✅ Прямоугольность после: {result.get('rectangularity_after', 'N/A')}")
            print(f"  📐 {result.get('angle_info', '')}")
        else:
            print(f"  ❌ Ошибка обработки")
    
    # Создаём HTML-отчёт
    generate_strict_report(results, output_dir)
    
    print(f"\n✅ Визуальный отчёт создан: {output_dir / 'index.html'}")
    
    # Открываем в браузере
    import webbrowser
    webbrowser.open(str((output_dir / "index.html").absolute()))

def generate_strict_report(results, output_dir):
    """Создаёт HTML-отчёт для строгого выпрямления."""
    
    # Вычисляем среднюю прямоугательность
    rects = [r.get("rectangularity_after", 0) for r in results if "rectangularity_after" in r]
    avg_rect = f"{np.mean(rects):.3f}" if rects else "N/A"
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Эксперимент: Жёсткое выпрямление до прямоугольника</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #1a237e; }}
        .stat-box {{ display: inline-block; background: #e8f5e9; padding: 15px; margin: 10px; border-radius: 5px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
        .img-container {{ margin: 20px 0; text-align: center; }}
        .img-container img {{ max-width: 100%; border: 1px solid #ddd; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th {{ background: #283593; color: white; padding: 8px; }}
        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .warning {{ background: #fff8e1; padding: 10px; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <h1>🎯 Эксперимент: Жёсткое выпрямление до строго прямоугольника</h1>
    
    <div class="warning">
        <strong>Важно:</strong> Этот эксперимент применяет перспективное преобразование для принудительного 
        создания прямоугольника с углами 90°. Согласно проектным требованиям, это только для оценки - 
        основной пайплайн сохраняет оригинальную форму полигона с помощью Coons-патча.
    </div>
    
    <div>
        <div class="stat-box"><div class="stat-value">{len(results)}</div><div>Файлов обработано</div></div>
        <div class="stat-box"><div class="stat-value">{avg_rect}</div><div>Средняя прямоугательность</div></div>
    </div>
    
    <h2>Результаты</h2>
"""
    
    for r in results:
        if "error" in r:
            html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <p style="color: red;">Ошибка: {r.get('error', 'Unknown')}</p>
    </div>
"""
            continue
            
        html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Исходная форма</td><td>{r.get('num_points_original', 'N/A')} точек</td></tr>
            <tr><td>Прямоугательность после</td><td class="stat-value">{r.get('rectangularity_after', 'N/A')}</td></tr>
            <tr><td>Углы</td><td>{r.get('angle_info', 'N/A')}</td></tr>
            <tr><td>Размер выхода</td><td>{r.get('output_size', 'N/A')}</td></tr>
        </table>
        <img src="{r['file']}_comparison.jpg" alt="Strict rectangle comparison for {r['file']}">
    </div>
"""
    
    html += """
    <footer>
        <p>Экспериментальный отчёт создан в рамках проекта Prj_35_OCR_prep</p>
        <p>Дата: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </footer>
</body>
</html>
"""
    
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    for r in results:
        if "error" in r:
            html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <p style="color: red;">Ошибка: {r.get('error', 'Unknown')}</p>
    </div>
"""
            continue
            
        html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Исходная форма</td><td>{r.get('num_points_original', 'N/A')} точек</td></tr>
            <tr><td>Прямоугательность после</td><td class="stat-value">{r.get('rectangularity_after', 'N/A')}</td></tr>
            <tr><td>Углы</td><td>{r.get('angle_info', 'N/A')}</td></tr>
            <tr><td>Размер выхода</td><td>{r.get('output_size', 'N/A')}</td></tr>
        </table>
        <img src="{r['file']}_comparison.jpg" alt="Strict rectangle comparison for {r['file']}">
    </div>
"""
    
    html += """
    <footer>
        <p>Экспериментальный отчёт создан в рамках проекта Prj_35_OCR_prep</p>
        <p>Дата: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </footer>
</body>
</html>
"""
    
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()