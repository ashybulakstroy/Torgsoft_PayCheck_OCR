"""
Эксперимент: Выпрямление товарной таблицы для файлов из data/to_do/

Цель: Улучшить выпрямление товарных таблиц чеков из папки data/to_do/ 
для последующего распознавания OCR.
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

def process_todo_image(todo_id_dir):
    """Обрабатывает один чек из папки data/to_do/."""
    
    # Ищем исходное изображение
    original_path = todo_id_dir / "original.png"
    upload_bin_path = todo_id_dir / "upload.bin"
    
    image = None
    image_source = None
    
    # Сначала пробуем original.png
    if original_path.exists():
        image = cv2.imread(str(original_path))
        if image is not None:
            image_source = "original.png"
    
    # Если не удалось загрузить, используем upload.bin (может быть RAW или JPEG)
    if image is None and upload_bin_path.exists():
        with open(upload_bin_path, 'rb') as f:
            data = f.read(20)  # Проверяем магические числа
            f.seek(0)
            
        # Проверяем формат изображения по магическим числам
        if data[:3] == b'\xff\xd8\xff':  # JPEG
            with open(upload_bin_path, 'rb') as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                image_source = "upload.bin (JPEG)"
        elif data[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
            with open(upload_bin_path, 'rb') as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                image_source = "upload.bin (PNG)"
    
    if image is None:
        return None, None
    
    # Детекция границы таблицы
    boundary = detect_table_boundary(image)
    polygon = boundary.get("polygon")
    
    if polygon is None:
        return {
            "file": todo_id_dir.name,
            "status": boundary.get("status", "unknown"),
            "error": "Polygon not detected",
            "image_source": image_source
        }, image
    
    # Метрики полигона
    rect = cv2.minAreaRect(np.array(polygon, dtype=np.float32).reshape((-1, 1, 2)))
    box = cv2.boxPoints(rect).astype(np.float32).reshape((-1, 1, 2))
    area_polygon = cv2.contourArea(np.array(polygon, dtype=np.float32).reshape((-1, 1, 2)))
    area_rect = cv2.contourArea(box)
    rectangularity = area_polygon / area_rect if area_rect > 0 else 0
    
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
    stem = f"todo_{todo_id_dir.name}"
    
    # Изображение "до" - исходное с полигоном
    before = image.copy()
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(before, [polygon_np], True, (0, 255, 0), 3)
    
    status_text = f"Status: {boundary['status']} | Points: {len(polygon)} | Rect: {rectangularity:.3f}"
    cv2.putText(before, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(before, f"Source: {image_source}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
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
        error_text = f"Error: {error_msg[:40]}..." if error_msg else "Rectification failed"
        cv2.putText(after, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Создаём боковое сравнение
    h = max(before.shape[0], after.shape[0]) + 50
    w = before.shape[1] + after.shape[1] + 60
    
    comparison = np.ones((h, w, 3), dtype=np.uint8) * 240
    
    before_h, before_w = before.shape[:2]
    comparison[20:20+before_h, 20:20+before_w] = before
    
    after_h, after_w = after.shape[:2]
    comparison[20:20+after_h, before_w+40:before_w+40+after_w] = after
    
    cv2.putText(comparison, "BEFORE", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(comparison, "AFTER", (before_w+40, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    result = {
        "file": todo_id_dir.name,
        "status": boundary.get("status", "unknown"),
        "num_points": len(polygon),
        "rectangularity": round(float(rectangularity), 3),
        "rectification_success": rectification_success,
        "warnings": boundary.get("warnings", []),
        "image_source": image_source,
        "comparison": comparison
    }
    
    if error_msg:
        result["error"] = error_msg
    
    return result, comparison

def main():
    todo_dir = Path("data/to_do")
    output_dir = Path("experiment_outputs/todo_rectangle_alignment")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Обработка файлов из data/to_do/")
    print(f"💾 Результаты сохраняются в: {output_dir.absolute()}")
    print()
    
    results = []
    
    # Обрабатываем каждую подпапку в data/to_do/
    todo_subdirs = sorted([d for d in todo_dir.iterdir() if d.is_dir()])
    
    for i, todo_dir_item in enumerate(todo_subdirs):
        print(f"[{i+1}/{len(todo_subdirs)}] Обработка: {todo_dir_item.name}")
        
        result, comparison_img = process_todo_image(todo_dir_item)
        
        if result:
            results.append(result)
            
            # Сохраняем визуализацию
            if comparison_img is not None:
                comparison_path = output_dir / f"{todo_dir_item.name}_comparison.jpg"
                cv2.imwrite(str(comparison_path), comparison_img)
            
            # Выводим результаты
            if "error" in result:
                print(f"  ❌ Ошибка: {result['error']}")
            else:
                print(f"  ✅ Успешно | Точек: {result['num_points']} | Прямоугательность: {result['rectangularity']:.3f}")
        else:
            print(f"  ❌ Не удалось загрузить изображение")
    
    # Создаём HTML-отчёт
    generate_todo_report(results, output_dir)
    
    # Открываем в браузере
    report_path = output_dir / "index.html"
    print(f"\n✅ Визуальный отчёт создан: {report_path.absolute()}")
    
    # Открываем в браузере
    import webbrowser
    webbrowser.open(str(report_path.absolute()))
    
    return results

def generate_todo_report(results, output_dir):
    """Создаёт HTML-отчёт для файлов из data/to_do/."""
    
    total = len(results)
    successful = sum(1 for r in results if r.get("rectification_success", False))
    
    valid_rects = [r for r in results if "rectangularity" in r]
    avg_rect = np.mean([r["rectangularity"] for r in valid_rects]) if valid_rects else 0
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Эксперимент: Выпрямление чеков из data/to_do/</title>
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
        .success-rate {{
            font-size: 32px;
            font-weight: bold;
            color: #2e7d32;
        }}
    </style>
</head>
<body>
    <h1>🧪 Эксперимент: Выпрямление чеков из data/to_do/</h1>
    
    <div class="info-panel">
        <h3>Описание эксперимента</h3>
        <p><strong>Цель:</strong> Улучшить выпрямление товарных таблиц чеков из папки <code>data/to_do/</code> для последующего распознавания OCR.</p>
        <p><strong>Метод:</strong> Использование Coons-патча (contour_unwarp.py) для нелинейного выпрямления таблицы.</p>
        <p><strong>Дата эксперимента:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Файлы:</strong> Файлы из <code>data/to_do/</code></p>
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
            <div class="stat-label">Средняя прямоугательность</div>
        </div>
    </div>
    
    <div class="warning">
        <strong>Важно:</strong> Согласно KEY INVARIANTS проекта (PROJECT_CONTEXT.md): полигон таблицы НЕ является 
        прямоугольником. Принудительное сведение к 4 точкам запрещено. Выпрямление выполняется с помощью 
        Coons-патча, который сохраняет изогнутые края и ступеньки таблицы.
    </div>
    
    <h2>📊 Статистика эксперимента</h2>
    <div class="info-panel">
        <ul>
            <li>Всего файлов обработано: <span class="metric">{total}</span></li>
            <li>Успешно выпрямлено: <span class="metric">{successful}</span></li>
            <li>Средняя прямоугательность: <span class="metric">{avg_rect:.3f}</span></li>
        </ul>
    </div>
    
    <h2>📸 Визуальный анализ "до/после"</h2>
"""
    
    for r in results:
        status_class = "status-ok" if r.get('status') == "detected" else "status-warning"
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
                <td class="{status_class}">{r.get('status', 'unknown')}</td>
            </tr>
            <tr>
                <td>Количество точек полигона</td>
                <td>{r.get('num_points', 'N/A')}</td>
            </tr>
            <tr>
                <td>Прямоугательность</td>
                <td class="metric">{r.get('rectangularity', 'N/A')}</td>
            </tr>
            <tr>
                <td>Статус выпрямления</td>
                <td class="{rectification_class}">{rectification_status}</td>
            </tr>
            <tr>
                <td>Источник изображения</td>
                <td>{r.get('image_source', 'N/A')}</td>
            </tr>
            <tr>
                <td>Предупреждения</td>
                <td>{warnings}</td>
            </tr>
        </table>
        <img src="{r['file']}_comparison.jpg" alt="Before/After for {r['file']}">
    </div>
"""
    
    if any(not r.get('rectification_success', False) for r in results):
        html += """
    <div class="warning">
        <strong>Проблемные случаи:</strong>
"""
        for r in results:
            if not r.get('rectification_success', False):
                html += f"        <p>- {r['file']}: {r.get('error', 'Неизвестная ошибка')}</p>\n"
        html += "    </div>\n"
    
    html += """
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999;">
        <p>Экспериментальный отчёт создан в рамках проекта Prj_35_OCR_prep</p>
    </footer>
</body>
</html>
"""
    
    report_path = output_dir / "index.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()