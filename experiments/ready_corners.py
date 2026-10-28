"""
Эксперимент: Жёсткое выпрямление товарной таблицы до идеального прямоугольника

Использует готовые координаты углов из source_body_corners:
- C1 (TL): source_body_corners[0] — левый верхний
- C2 (TR): source_body_corners[1] — правый верхний  
- C3 (BR): source_body_corners[2] — правый нижний
- C4 (BL): source_body_corners[3] — левый нижний

Все 4 угла взяты готовыми, только применяем perspective transform.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

def transform_to_perfect_rectangle(image, body_corners, output_size):
    """
    Применяет перспективное преобразование для создания идеального прямоугольника.
    
    Args:
        image: исходное изображение
        body_corners: 4 угла таблицы из source_body_corners [TL, TR, BR, BL]
        output_size: [height, width] выходного изображения
    
    Returns:
        rectified: выпрямленное изображение
        mask: маска для визуализации
    """
    h, w = output_size
    
    # Целевые координаты (идеальный прямоугольник)
    dst_pts = np.float32([
        [0, 0],         # TL
        [w - 1, 0],     # TR
        [w - 1, h - 1], # BR
        [0, h - 1]      # BL
    ])
    
    # Исходные координаты (углы таблицы из metadata)
    src_pts = np.array(body_corners, dtype=np.float32)
    
    # Перспективное преобразование
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rectified, matrix

def process_todo_with_body_corners(todo_id_dir, output_dir):
    """Обрабатывает чек с использованием готовых углов из source_body_corners."""
    original_path = todo_id_dir / "original.png"
    rect_path = todo_id_dir / "rectification.json"
    
    if not original_path.exists() or not rect_path.exists():
        return None
    
    image = cv2.imread(str(original_path))
    if image is None:
        return None
    
    with open(rect_path) as f:
        rect_data = json.load(f)
    
    # Получаем готовые углы таблицы
    body_corners = rect_data.get("source_body_corners", [])
    output_size = rect_data.get("output_size", [535, 902])
    
    if not body_corners or len(body_corners) != 4:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": "source_body_corners not found or invalid"
        }
    
    h, w = output_size[0], output_size[1]
    
    # Применяем жёсткое выпрямление
    try:
        rectified, matrix = transform_to_perfect_rectangle(image, body_corners, (w, h))
    except Exception as e:
        return {
            "file": todo_id_dir.name,
            "status": "error",
            "error": f"Transform failed: {str(e)}"
        }
    
    # Создаём визуализацию
    before = image.copy()
    
    # Рисуем source_polygon (зеленый контур)
    polygon = np.array(rect_data.get("source_polygon", []), dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(before, [polygon], True, (0, 255, 0), 2)
    
    # Рисуем углы таблицы (каждая точка своим цветом)
    corner_labels = ["C1 (TL)", "C2 (TR)", "C3 (BR)", "C4 (BL)"]
    corner_colors = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]  # Все углы синие (известные)
    
    for i, (corner, color, label) in enumerate(zip(body_corners, corner_colors, corner_labels)):
        cv2.circle(before, (int(corner[0]), int(corner[1])), 8, color, -1)
        cv2.putText(before, label, (int(corner[0]) + 12, int(corner[1]) + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Рисуем линии между углами
    body_corners_array = np.array(body_corners, dtype=np.int32)
    cv2.polylines(before, [body_corners_array], True, (255, 0, 0), 2)
    
    cv2.putText(before, "BEFORE (green=polygon, red=body corners)", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # После выпрямления
    after = rectified.copy()
    cv2.putText(after, f"AFTER: {w}x{h}px", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Добавляем прямоугольную рамку на выпрямленное изображение
    cv2.rectangle(after, (5, 5), (w-6, h-6), (0, 255, 255), 2)
    
    # Создаём боковое сравнение
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
        "body_corners": body_corners,
        "comparison": comparison
    }

def generate_report(results, output_dir):
    """Создаёт HTML-отчёт."""
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Эксперимент: Жёсткое выпрямление с готовыми углами</title>
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
    <h1>🎯 Эксперимент: Жёсткое выпрямление с готовыми углами из source_body_corners</h1>
    
    <div class="warning">
        <strong>Метод:</strong> Использование готовых координат углов таблицы из source_body_corners<br>
        C1 (TL), C2 (TR), C3 (BR), C4 (BL) — все 4 угла взяты из метаданных без вычислений
    </div>
    
    <div>
        <div class="stat-box"><div class="stat-value">{len([r for r in results if 'error' not in r])}</div><div class="stat-label">Файлов успешно обработано</div></div>
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
        
        corners = r.get('body_corners', [])
        corners_text = ", ".join([f"[{round(c[0],1)}, {round(c[1],1)}]" for c in corners])
        
        html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Размер выхода</td><td>{r.get('output_size', 'N/A')} пикселей</td></tr>
            <tr><td>Углы таблицы (body_corners)</td><td class="metric">{corners_text}</td></tr>
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
    output_dir = Path("experiment_outputs/todo_ready_corners")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Эксперимент: Жёсткое выпрямление с готовыми углами из source_body_corners")
    print(f"📂 Обрабатываем файлы из: {todo_dir.absolute()}")
    print(f"💾 Результаты сохраняются в: {output_dir.absolute()}")
    print()
    
    results = []
    
    todo_subdirs = sorted([d for d in todo_dir.iterdir() if d.is_dir()])
    
    for i, todo_dir_item in enumerate(todo_subdirs):
        print(f"[{i+1}/{len(todo_subdirs)}] {todo_dir_item.name}")
        
        result = process_todo_with_body_corners(todo_dir_item, output_dir)
        
        if result:
            results.append(result)
            
            if "comparison" in result:
                comparison_path = output_dir / f"{result['file']}_comparison.jpg"
                cv2.imwrite(str(comparison_path), result["comparison"])
            
            if "error" in result:
                print(f"  ❌ Ошибка: {result['error']}")
            else:
                corners = result.get('body_corners', [])
                print(f"  ✅ Углы: {len(corners)} точки")
                for i, c in enumerate(corners):
                    print(f"    C{i+1}: [{c[0]:.1f}, {c[1]:.1f}]")
    
    # Создаём HTML-отчёт
    generate_report(results, output_dir)
    
    # Открываем в браузере
    import webbrowser
    report_path = output_dir / "index.html"
    webbrowser.open(str(report_path.absolute()))
    
    print(f"\n✅ Отчёт открыт в браузере: {report_path.absolute()}")

if __name__ == "__main__":
    main()