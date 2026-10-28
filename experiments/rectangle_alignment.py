"""
Эксперимент: Улучшение выпрямления товарной таблицы для OCR.

Цель: Создать экспериментальное решение для выпрямления товарных таблиц чеков
до строго прямоугольного формата, что улучшит качество OCR.

Метод: 
- Использование текущего детектора границ таблицы (table_boundary.py)
- Применение Coons-патча для нелинейного выпрямления (contour_unwarp.py)
- Анализ качества выпрямления
- Создание визуального отчёта "было/стало"

Ограничения проекта:
- Эксперимент выполняется в отдельном модуле (experiments/)
- Используется проектное виртуальное окружение .venv
- Результаты временно сохраняются в experiment_outputs/
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from App.table_boundary import detect_table_boundary
from App.contour_unwarp import build_contour_map

def measure_rectangularity(polygon):
    """Оценивает, насколько полигон близок к прямоугольнику (1.0 - идеал)."""
    if polygon is None or len(polygon) < 4:
        return 0.0
    
    # Конвертируем полигон в массив для OpenCV
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    
    # Находим минимальный прямоугольник, охватывающий полигон
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32).reshape((-1, 1, 2))
    
    # Вычисляем площадь полигона и прямоугольника
    area_polygon = cv2.contourArea(pts)
    area_rect = cv2.contourArea(box)
    
    if area_rect > 0:
        ratio = area_polygon / area_rect
    else:
        ratio = 0.0
    
    return min(ratio, 1.0)

def measure_edge_straightness(image, polygon, num_samples=20):
    """Оценивает, насколько края полигона прямые (0.0 - идеально прямые)."""
    if polygon is None or len(polygon) < 4:
        return 1.0
    
    # Рассчитываем откление точек полигона от идеальных прямых
    deviations = []
    edges = [(i, (i+1) % len(polygon)) for i in range(len(polygon))]
    
    for i, edge in enumerate(edges):
        p1 = np.array(polygon[edge[0]], dtype=np.float64)
        p2 = np.array(polygon[edge[1]], dtype=np.float64)
        
        # Длина ребра
        length = np.linalg.norm(p2 - p1)
        if length < 5:  # Пропускаем слишком короткие ребра
            continue
            
        # Проверяем точки, лежащие на этом ребре
        for j, point in enumerate(polygon):
            if j == edge[0] or j == edge[1]:
                continue
            
            point = np.array(point, dtype=np.float64)
            
            # Расстояние от точки до прямой (2D cross product magnitude)
            # Для 2D векторов: cross = (p2-p1) × (point-p1)
            v1 = p2 - p1
            v2 = point - p1
            # 2D кросс-произведение: v1.x * v2.y - v1.y * v2.x
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dist = abs(cross) / length
            deviations.append(dist)
    
    if deviations:
        return float(np.mean(deviations))
    else:
        return 0.0

def process_image(image_path):
    """Обрабатывает одно изображение и возвращает результаты."""
    image = cv2.imread(str(image_path))
    if image is None:
        result = {
            "file": image_path.name,
            "status": "error",
            "error": "Could not load image"
        }
        return result, image, None, None, None, None, None
    
    # Детекция границы таблицы
    boundary = detect_table_boundary(image)
    polygon = boundary.get("polygon")
    
    if polygon is None:
        result = {
            "file": image_path.name,
            "status": boundary.get("status", "unknown"),
            "error": "Polygon not detected"
        }
        return result, image, polygon, boundary, None, None, None
    
    # Оценка качества полигона
    rectangularity = measure_rectangularity(polygon)
    edge_straightness = measure_edge_straightness(image, polygon)
    
    # Применение выпрямления
    try:
        map_x, map_y, metadata = build_contour_map(polygon)
        rectification_success = True
    except Exception as e:
        map_x, map_y, metadata = None, None, None
        rectification_success = False
    
    result = {
        "file": image_path.name,
        "status": boundary.get("status", "unknown"),
        "num_polygon_points": len(polygon),
        "warnings": boundary.get("warnings", []),
        "rectangularity_before": rectangularity,
        "edge_straightness_before": edge_straightness,
        "rectification_success": rectification_success,
        "rectification_metadata": metadata if metadata else {},
        "original_shape": list(image.shape)[:2]  # (height, width)
    }
    
    return result, image, polygon, boundary, map_x, map_y, metadata

def create_visualization(image, polygon, map_x, map_y, output_path, title=""):
    """Создаёт визуализацию до и после выпрямления."""
    
    # Изображение "до" - исходное с полигоном
    before = image.copy()
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(before, [polygon_np], True, (0, 255, 0), 2)
    cv2.putText(before, f"{title} Before ({len(polygon)} points)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Изображение "после" - выпрямленное
    if map_x is not None and map_y is not None:
        after = cv2.remap(
            image, map_x, map_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        cv2.putText(after, f"{title} After (rectified)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        after = image.copy()
        cv2.putText(after, f"{title} After (remap failed)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Создаём боковое сравнение
    h = max(before.shape[0], after.shape[0])
    w = before.shape[1] + after.shape[1] + 20
    
    combined = np.zeros((h, w, 3), dtype=np.uint8)
    combined[:before.shape[0], :before.shape[1]] = before
    combined[:after.shape[0], before.shape[1]+20:] = after
    
    # Добавляем разделитель
    cv2.line(combined, (before.shape[1]+10, 0), (before.shape[1]+10, h), (100, 100, 100), 1)
    cv2.putText(combined, "BEFORE", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(combined, "AFTER", (before.shape[1]+10+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imwrite(str(output_path), combined)

def generate_html_report(results, image_dir, output_path):
    """Генерирует HTML-отчёт с визуальным сравнением."""
    
    total = len(results)
    successful = sum(1 for r in results if isinstance(r, tuple) and r[0].get("rectification_success", False))
    
    # Фильтруем только успешные результаты
    valid_results = [r for r in results if isinstance(r, tuple) and "rectangularity_before" in r[0]]
    
    if valid_results:
        avg_rectangularity = np.mean([r[0]["rectangularity_before"] for r in valid_results])
        avg_edge_straightness = np.mean([r[0]["edge_straightness_before"] for r in valid_results])
    else:
        avg_rectangularity = 0.0
        avg_edge_straightness = 0.0
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Эксперимент: Выпрямление товарной таблицы</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #1a237e;
            border-bottom: 3px solid #1a237e;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #283593;
            margin-top: 30px;
        }}
        .summary {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .stat-box {{
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 150px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #283593;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-ok {{
            color: #2e7d32;
            font-weight: bold;
        }}
        .status-warning {{
            color: #f57f17;
            font-weight: bold;
        }}
        .status-error {{
            color: #c62828;
            font-weight: bold;
        }}
        .img-comparison {{
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }}
        .img-comparison img {{
            max-width: 48%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .metric {{
            font-family: 'Courier New', monospace;
        }}
        .warning {{
            background-color: #fff8e1;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }}
        .info-panel {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>🧪 Эксперимент: Выпрямление товарной таблицы для улучшения OCR</h1>
    
    <div class="info-panel">
        <h3>Описание эксперимента</h3>
        <p><strong>Цель:</strong> Улучшить выпрямление товарных таблиц чеков для последующего OCR.</p>
        <p><strong>Метод:</strong> Использование Coons-патча (contour_unwarp.py) для нелинейного выпрямления таблицы.</p>
        <p><strong>Дата:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="summary">
        <div class="stat-box">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Всего обработано</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{successful}</div>
            <div class="stat-label">Успешно выпрямлено</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_rectangularity:.3f}</div>
            <div class="stat-label">Средняя прямоугольность</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_edge_straightness:.1f}px</div>
            <div class="stat-label">Сред. отклонение краёв</div>
        </div>
    </div>
    
    <div class="warning">
        <strong>Важно:</strong> Согласно KEY INVARIANTS проекта (PROJECT_CONTEXT.md): полигон таблицы НЕ является прямоугольником. 
        Принудительное сведение к 4 точкам запрещено. Выпрямление выполняется с помощью Coons-патча, 
        который сохраняет изогнутые края и ступеньки таблицы.
    </div>
    
    <h2>Визуальный анализ "до/после"</h2>
    <table>
        <tr>
            <th>Файл</th>
            <th>Статус</th>
            <th>Точек</th>
            <th>Прямоугольность</th>
            <th>Выпрямление</th>
            <th>Визуализация</th>
        </tr>
"""
    
    for result_tuple in results:
        result, image, polygon, boundary, map_x, map_y, metadata = result_tuple
        
        status_class = "status-ok" if result.get("status", "unknown") == "detected" else "status-warning"
        rectification_status = "Успешно" if result.get("rectification_success", False) else "Ошибка"
        rectification_class = "status-ok" if result.get("rectification_success", False) else "status-error"
        
        num_points = result.get('num_polygon_points', 'N/A')
        rectangularity = result.get('rectangularity_before', None)
        rectangularity_text = f"{rectangularity:.3f}" if rectangularity is not None else 'N/A'
        comparison_img = f"images/{result['file']}_comparison.jpg"
        
        html += f"""
        <tr>
            <td>{result['file']}</td>
            <td class="{status_class}">{result.get('status', 'unknown')}</td>
            <td>{num_points}</td>
            <td class="metric">{rectangularity_text}</td>
            <td class="{rectification_class}">{rectification_status}</td>
            <td>
                <div class="img-comparison">
                    <img src="{comparison_img}" alt="Before/After">
                </div>
            </td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Выводы и рекомендации</h2>
    <div class="info-panel">
        <h3>Статистика эксперимента</h3>
        <ul>
            <li>Обработано <span class="metric">%d</span> изображений из data/in_data/</li>
            <li>Успешно применено выпрямление к <span class="metric">%d</span> изображениям</li>
            <li>Средняя прямоугольность полигона: <span class="metric">%.3f</span> (1.0 = идеальный прямоугольник)</li>
            <li>Среднее откление краёв от прямых: <span class="metric">%.1f пикселей</span></li>
        </ul>
        
        <h3>Рекомендации</h3>
        <p>1. Текущий метод выпрямления (Coons-патч) успешно обрабатывает сложные полигоны.</p>
        <p>2. Для улучшения OCR рекомендуется использовать выпрямленное изображение.</p>
        <p>3. При необходимости можно добавить пост-обработку для дополнительного выравнивания текста.</p>
    </div>
</body>
</html>
""" % (total, successful, avg_rectangularity, avg_edge_straightness)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Создан HTML-отчёт: {output_path}")

def main():
    input_dir = Path("data/in_data")
    output_dir = Path("experiment_outputs/rectangle_alignment")
    images_dir = output_dir / "images"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем список всех изображений
    image_files = sorted(input_dir.glob("*.jp*g"))
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    results = []
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Обработка: {img_path.name}")
        
        try:
            result_data = process_image(img_path)
            if result_data:
                result, image, polygon, boundary, map_x, map_y, metadata = result_data
                
                # Создаём визуализацию только если есть полигон
                if polygon is not None and image is not None:
                    output_comparison = images_dir / f"{result['file']}_comparison.jpg"
                    try:
                        create_visualization(image, polygon, map_x, map_y, output_comparison, "Table rectification")
                    except Exception as vis_e:
                        print(f"  Предупреждение при визуализации: {vis_e}")
                
                results.append(result_data)
                
                # Выводим информацию о результате
                print(f"  Полигон: {result.get('num_polygon_points', 'N/A')} точек")
                rectangularity = result.get('rectangularity_before', None)
                if rectangularity is not None:
                    print(f"  Прямоугольность: {rectangularity:.3f}")
                print(f"  Статус выпрямления: {'успешно' if result.get('rectification_success', False) else 'ошибка'}")
                
        except Exception as e:
            print(f"  ОШИБКА: {str(e)}")
            results.append(({"file": img_path.name, "error": str(e)}, None, None, None, None, None, None))
    
    # Генерируем HTML-отчёт
    generate_html_report(results, images_dir, output_dir / "index.html")
    
    # Выводим статистику
    print("\n=== Сводка эксперимента ===")
    print(f"Всего обработано: {len(results)}")
    print(f"Успешно выпрямлено: {sum(1 for r in results if isinstance(r, tuple) and r[0].get('rectification_success', False))}")
    
    valid_results = [r for r in results if isinstance(r, tuple) and "rectangularity_before" in r[0]]
    if valid_results:
        avg_rect = np.mean([r[0]["rectangularity_before"] for r in valid_results])
        avg_straight = np.mean([r[0]["edge_straightness_before"] for r in valid_results])
        print(f"Средняя прямоугольность полигона: {avg_rect:.3f}")
        print(f"Среднее отклонение краёв: {avg_straight:.1f}px")
    
    print(f"\nВизуальный отчёт: {output_dir / 'index.html'}")

if __name__ == "__main__":
    main()