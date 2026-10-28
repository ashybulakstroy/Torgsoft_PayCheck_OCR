"""
Эксперимент: Улучшение выпрямления товарной таблицы для OCR.

Создаёт визуальный отчёт с изображениями "до/после" для всех 26 фотографий чеков
из data/in_data/, применяя текущий метод выпрямления Coons-патч.
"""

import os
import sys
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from App.table_boundary import detect_table_boundary
from App.contour_unwarp import build_contour_map

def measure_rectangularity(polygon):
    """Оценивает, насколько полигон близок к прямоугольнику (1.0 = идеал)."""
    if polygon is None or len(polygon) < 4:
        return 0.0
    
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32).reshape((-1, 1, 2))
    
    area_polygon = cv2.contourArea(pts)
    area_rect = cv2.contourArea(box)
    
    if area_rect > 0:
        ratio = area_polygon / area_rect
    else:
        ratio = 0.0
    
    return min(ratio, 1.0)

def measure_edge_straightness(polygon):
    """Оценивает среднее откление точек полигона от прямых линий ребёр."""
    if polygon is None or len(polygon) < 4:
        return 1.0
    
    deviations = []
    n = len(polygon)
    
    for i in range(n):
        p1 = np.array(polygon[i], dtype=np.float64)
        p2_idx = (i + 1) % n
        p2 = np.array(polygon[p2_idx], dtype=np.float64)
        
        length = np.linalg.norm(p2 - p1)
        if length < 5:
            continue
            
        for j in range(n):
            if j == i or j == p2_idx:
                continue
            
            point = np.array(polygon[j], dtype=np.float64)
            v1 = p2 - p1
            v2 = point - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dist = abs(cross) / length
            deviations.append(dist)
    
    if deviations:
        return float(np.mean(deviations))
    else:
        return 0.0

def process_single_image(image_path, output_dir):
    """Обрабатывает одно изображение: детекция, выпрямление, визуализация."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Детекция границы таблицы
    boundary = detect_table_boundary(image)
    polygon = boundary.get("polygon")
    
    if polygon is None:
        return {
            "file": image_path.name,
            "status": boundary.get("status", "unknown"),
            "error": "Polygon not detected",
            "original_shape": list(image.shape)[:2]
        }
    
    # Метрики полигона
    rectangularity = measure_rectangularity(polygon)
    edge_straightness = measure_edge_straightness(polygon)
    
    # Применение выпрямления
    rectification_success = False
    map_x, map_y, metadata = None, None, None
    error_msg = None
    
    try:
        map_x, map_y, metadata = build_contour_map(polygon)
        rectification_success = True
    except Exception as e:
        error_msg = str(e)
    
    # Создаём визуализацию
    stem = Path(image_path).stem.replace(" ", "_")
    
    # Изображение "до" - исходное с полигоном
    before = image.copy()
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(before, [polygon_np], True, (0, 255, 0), 3)
    
    status_text = f"Status: {boundary['status']} | Points: {len(polygon)} | Rect: {rectangularity:.3f}"
    cv2.putText(before, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(before, "BEFORE (original polygon)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    before_path = output_dir / f"{stem}_before.jpg"
    cv2.imwrite(str(before_path), before)
    
    # Изображение "после" - выпрямленное
    if rectification_success and map_x is not None and map_y is not None:
        after = cv2.remap(
            image, map_x, map_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        cv2.putText(after, "AFTER (Coons patch rectified)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        after = image.copy()
        error_text = f"Error: {error_msg[:30]}..." if error_msg else "Rectification failed"
        cv2.putText(after, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    after_path = output_dir / f"{stem}_after.jpg"
    cv2.imwrite(str(after_path), after)
    
    # Создаём боковое сравнение
    h = max(before.shape[0], after.shape[0]) + 20
    w = before.shape[1] + after.shape[1] + 40
    
    comparison = np.ones((h, w, 3), dtype=np.uint8) * 240
    
    # Размещаем "до" слева
    before_h, before_w = before.shape[:2]
    comparison[10:10+before_h, 10:10+before_w] = before
    
    # Размещаем "после" справа
    after_h, after_w = after.shape[:2]
    comparison[10:10+after_h, before_w+30:before_w+30+after_w] = after
    
    # Добавляем подписи
    cv2.putText(comparison, "BEFORE", (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(comparison, "AFTER", (before_w+30, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    comparison_path = output_dir / f"{stem}_comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    
    # Возвращаем результаты
    result = {
        "file": image_path.name,
        "status": boundary.get("status", "unknown"),
        "num_points": len(polygon),
        "rectangularity": round(rectangularity, 3),
        "edge_straightness": round(edge_straightness, 1),
        "rectification_success": rectification_success,
        "warnings": boundary.get("warnings", []),
        "comparison_file": comparison_path.name,
        "before_file": before_path.name,
        "after_file": after_path.name
    }
    
    if error_msg:
        result["error"] = error_msg
    
    return result

def generate_final_report(results, output_dir, report_path):
    """Создаёт финальный HTML-отчёт."""
    
    total = len(results)
    successful = sum(1 for r in results if r.get("rectification_success", False))
    detected = sum(1 for r in results if r.get("status") == "detected")
    needs_review = sum(1 for r in results if r.get("status") == "needs_review")
    
    valid_rects = [r for r in results if "rectangularity" in r]
    avg_rect = np.mean([r["rectangularity"] for r in valid_rects]) if valid_rects else 0
    avg_straight = np.mean([r["edge_straightness"] for r in valid_rects]) if valid_rects else 0
    
    # Создаём HTML
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Эксперимент: Выпрямление таблицы чека для OCR</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
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
        h3 {{
            color: #3949ab;
        }}
        .summary {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .stat-box {{
            background: white;
            padding: 20px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 160px;
        }}
        .stat-value {{
            font-size: 28px;
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
        .img-container {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }}
        .img-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .warning {{
            background-color: #fff8e1;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .info-panel {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            font-family: 'Courier New', monospace;
        }}
        .recommendation {{
            background-color: #e8f5e9;
            padding: 20px;
            border-left: 4px solid #4caf50;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .file-list {{
            columns: 2;
            gap: 20px;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .success-rate {{
            font-size: 32px;
            font-weight: bold;
            color: #2e7d32;
        }}
    </style>
</head>
<body>
    <h1>🧪 Эксперимент: Выпрявление товарной таблицы для улучшения OCR</h1>
    
    <div class="info-panel">
        <h3>Описание эксперимента</h3>
        <p><strong>Цель:</strong> Улучшить выпрямление товарных таблиц чеков для последующего распознавания OCR, 
        преобразуя изогнутые и искажённые таблицы в максимально прямоугольные формы.</p>
        <p><strong>Метод:</strong> Использование Coons-патча (contour_unwarp.py) для нелинейного выпрямления 
        таблицы на основе полигонального контура, обнаруженного детектором (table_boundary.py).</p>
        <p><strong>Дата эксперимента:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Файлы:</strong> 26 изображений из <code>data/in_data/</code></p>
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
            <div class="stat-value success-rate">{successful}/{total}</div>
            <div class="stat-label">Успех: {successful*100//total if total > 0 else 0}%</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_rect:.3f}</div>
            <div class="stat-label">Средняя прямоугольность</div>
        </div>
    </div>
    
    <div class="warning">
        <strong>Важно:</strong> Согласно KEY INVARIANTS проекта (PROJECT_CONTEXT.md): полигон таблицы НЕ является 
        прямоугольником. Принудительное сведение к 4 точкам запрещено. Выпрямление выполняется с помощью 
        Coons-патча, который сохраняет изогнутые края и ступеньки таблицы. Это позволяет сохранить всю 
        информацию о структуре таблицы для OCR.
    </div>
    
    <h2>📊 Статистика эксперимента</h2>
    <div class="info-panel">
        <ul>
            <li>Всего изображений обработано: <span class="metric">{total}</span></li>
            <li>Полигон найден для: <span class="metric">{len(valid_rects)}</span> / {total}</li>
            <li>Статус <code>detected</code>: <span class="metric">{detected}</span></li>
            <li>Статус <code>needs_review</code>: <span class="metric">{needs_review}</span></li>
            <li>Успешно выпрямлено: <span class="metric">{successful}</span> / {len(valid_rects)}</li>
            <li>Средняя прямоугольность полигона: <span class="metric">{avg_rect:.3f}</span> (1.0 = идеальный прямоугольник)</li>
            <li>Среднее отклонение краёв от прямых: <span class="metric">{avg_straight:.1f}</span> пикселей</li>
        </ul>
    </div>
    
    <h2>📸 Визуальный анализ "до/после"</h2>
"""
    
    for r in results:
        if "num_points" not in r:
            html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <p class="status-error">Ошибка: {r.get('error', 'Unknown error')}</p>
    </div>
"""
            continue
        
        status_class = "status-ok" if r['status'] == "detected" else "status-warning"
        rectification_class = "status-ok" if r.get('rectification_success', False) else "status-error"
        rectification_status = "✅ Успешно" if r.get('rectification_success', False) else "❌ Ошибка"
        warnings = ", ".join(r.get('warnings', [])) if r.get('warnings') else "—"
        
        html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <table>
            <tr>
                <th>Параметр</th>
                <th>Значение</th>
            </tr>
            <tr>
                <td>Статус детекции</td>
                <td class="{status_class}">{r['status']}</td>
            </tr>
            <tr>
                <td>Количество точек полигона</td>
                <td>{r['num_points']}</td>
            </tr>
            <tr>
                <td>Прямоугольность</td>
                <td class="metric">{r['rectangularity']:.3f}</td>
            </tr>
            <tr>
                <td>Отклонение краёв</td>
                <td class="metric">{r['edge_straightness']:.1f} px</td>
            </tr>
            <tr>
                <td>Статус выпрямления</td>
                <td class="{rectification_class}">{rectification_status}</td>
            </tr>
            <tr>
                <td>Предупреждения</td>
                <td>{warnings}</td>
            </tr>
        </table>
        <img src="{r['comparison_file']}" alt="Before/After comparison for {r['file']}">
    </div>
"""
    
    html += f"""
    <h2>✅ Выводы и рекомендации</h2>
    <div class="recommendation">
        <h3>Результаты эксперимента</h3>
        <ul>
            <li>Успешно выпрямлено <span class="metric">{successful}</span> из <span class="metric">{len(valid_rects)}</span> найденных таблиц.</li>
            <li>Средняя прямоугательность полигона: <span class="metric">{avg_rect:.3f}</span>.</li>
        </ul>
        
        <h3>Проблемные случаи</h3>
        <ul>
"""
    
    for r in results:
        if not r.get("rectification_success", False):
            html += f'            <li><code>{r["file"]}</code> — {r.get("error", "Ошибка выпрямления")}</li>\n'
    
    html += f"""
        </ul>
        
        <h3>Рекомендации по внедрению</h3>
        <p>1. Текущий метод выпрямления Coons-патч успешно применяется к {successful} из {len(valid_rects)} изображениям.</p>
        <p>2. Для улучшения OCR рекомендуется использовать выпрямленные изображения в пайплайне обработки.</p>
        <p>3. Для проблемных случаев рекомендуется ручная настройка или альтернативные методы выпрямления.</p>
    </div>
    
    <h2>📁 Список обработанных файлов</h2>
    <div class="file-list">
        <ul>
"""
    
    for r in results:
        status = r.get("status", "error")
        html += f"            <li>{r['file']} — {status}</li>\n"
    
    html += """
        </ul>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999;">
        <p>Экспериментальный отчёт создан в рамках проекта Prj_35_OCR_prep</p>
    </footer>
</body>
</html>
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Создан HTML-отчёт: {report_path}")

def main():
    input_dir = Path("data/in_data")
    output_dir = Path("experiment_outputs/rectangle_alignment")
    images_dir = output_dir / "images"
    
    # Очищаем и создаём директории
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем список изображений
    image_files = sorted(input_dir.glob("*.jp*g"))
    print(f"🔍 Найдено {len(image_files)} изображений для обработки")
    print()
    
    results = []
    
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}")
        
        result = process_single_image(img_path, images_dir)
        if result:
            results.append(result)
            status = "✅" if result.get("rectification_success", False) else "❌"
            rect_info = f"{result.get('rectangularity', 'N/A'):.3f}" if "rectangularity" in result else "N/A"
            print(f"  {status} Точек: {result.get('num_points', 'N/A')} | Прямоугальность: {rect_info}")
    
    # Копируем изображения в корневую директорию для HTML
    for r in results:
        if "comparison_file" in r:
            src = images_dir / r["comparison_file"]
            dst = output_dir / r["comparison_file"]
            if src.exists():
                shutil.copy2(src, dst)
    
    # Генерируем финальный отчёт
    report_path = output_dir / "index.html"
    generate_final_report(results, output_dir, report_path)
    
    # Выводим сводку
    print("\n" + "="*50)
    print("СВОДКА ЭКСПЕРИМЕНТА")
    print("="*50)
    print(f"Всего обработано: {len(results)}")
    print(f"Успешно выпрямлено: {sum(1 for r in results if r.get('rectification_success', False))}")
    print(f"Средняя прямоугательность: {np.mean([r.get('rectangularity', 0) for r in results]):.3f}")
    print(f"\nОтчёт: file://{report_path.absolute()}")

if __name__ == "__main__":
    main()