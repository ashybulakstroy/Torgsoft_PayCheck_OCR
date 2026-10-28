"""
Эксперимент: Автоматическое выпрямление таблиц в data/in_data

Логика:
1. Автодетекция таблицы (findContours + minAreaRect)
2. Оценка необходимости выравнивания (TableAlignmentDetector)
3. Перспективное выпрямление (cv2.getPerspectiveTransform)
4. HTML-отчёт "до/после"
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

# Импорт детектора из соседнего эксперимента
sys.path.insert(0, str(Path(__file__).parent))
from auto_rectify_detector import TableAlignmentDetector, rectify_table_if_needed


def detect_table_corners(image):
    """Автоматически находит 4 угла таблицы на изображении."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Морфология для соединения линий таблицы
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтруем контуры по площади и форме
    filtered = []
    h, w = image.shape[:2]
    min_area = (w * h) * 0.1  # минимум 10% площади кадра
    max_area = w * h * 0.95
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        # Аппроксимация пolygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4:
            filtered.append((area, approx))
    
    if not filtered:
        return None
    
    # Выбираем самый большой подходящий контур
    filtered.sort(key=lambda x: x[0], reverse=True)
    best_approx = filtered[0][1]
    
    # Если больше 4 точек — упрощаем
    if len(best_approx) > 4:
        epsilon = 0.05 * cv2.arcLength(best_approx, True)
        best_approx = cv2.approxPolyDP(best_approx, epsilon, True)
    
    if len(best_approx) != 4:
        # fallback: minAreaRect
        x, y, w_rect, h_rect = cv2.boundingRect(best_approx)
        corners = np.array([[x, y], [x + w_rect, y], [x + w_rect, y + h_rect], [x, y + h_rect]], dtype=np.float32)
        return corners, None
    
    # Упорядочивание точек: TL, TR, BR, BL
    pts = best_approx.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    return corners, best_approx


def process_image(image_path, output_dir):
    """Обрабатывает одно изображение: детекция → выравнивание → визуализация."""
    image = cv2.imread(str(image_path))
    if image is None:
        return {"file": image_path.name, "status": "error", "error": "Cannot read image"}
    
    detection_result = detect_table_corners(image)
    if detection_result is None:
        return {"file": image_path.name, "status": "error", "error": "Table not detected"}
    
    corners, raw_contour = detection_result
    
    if corners is None:
        return {"file": image_path.name, "status": "error", "error": "Table corners not found"}
    
    # Оценка необходимости выравнивания
    detector = TableAlignmentDetector()
    detection = detector.detect(corners)
    needs_align = detection["needs_alignment"]
    
    # Выравнивание (если нужно)
    if needs_align:
        rectified, info = rectify_table_if_needed(image, corners)
    else:
        rectified = image.copy()
        info = {"aligned": False, "reason": "Already aligned", "detection": detection}
    
    # Визуализация "до"
    before = image.copy()
    cv2.polylines(before, [corners.reshape(-1, 1, 2).astype(int)], True, (0, 0, 255), 2)
    
    labels = ["C1 TL", "C2 TR", "C3 BR", "C4 BL"]
    for i, pt in enumerate(corners):
        pt_int = pt.astype(int)
        cv2.circle(before, tuple(pt_int), 8, (255, 0, 0), -1)
        cv2.putText(before, labels[i], (pt_int[0]+8, pt_int[1]-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.putText(before, f"Needs align: {needs_align}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Визуализация "после"
    after = rectified.copy()
    cv2.putText(after, "Rectified", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Боковое сравнение с выравниванием по высоте
    h_img = max(before.shape[0], after.shape[0])
    
    # Масштабируем изображения до одинаковой высоты
    if before.shape[0] != h_img:
        scale = h_img / before.shape[0]
        before = cv2.resize(before, (int(before.shape[1] * scale), h_img))
    if after.shape[0] != h_img:
        scale = h_img / after.shape[0]
        after = cv2.resize(after, (int(after.shape[1] * scale), h_img))
    
    gap = np.ones((h_img, 20, 3), dtype=np.uint8) * 200
    comparison = np.hstack([before, gap, after])
    
    # Сохранение
    output_dir.mkdir(parents=True, exist_ok=True)
    rectified_path = output_dir / f"{image_path.stem}_rectified.jpg"
    comparison_path = output_dir / f"{image_path.stem}_comparison.jpg"
    cv2.imwrite(str(rectified_path), rectified)
    cv2.imwrite(str(comparison_path), comparison)
    
    return {
        "file": image_path.name,
        "status": "processed",
        "needs_alignment": needs_align,
        "aligned": info.get("aligned", False),
        "metrics": detection["metrics"],
        "corners": corners.tolist(),
        "output_size": list(rectified.shape[:2]),
        "rectified_path": str(rectified_path),
        "comparison_path": str(comparison_path)
    }


def generate_report(results, output_dir):
    """Создаёт HTML-отчёт."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    errored = [r for r in results if r.get("status") == "error"]
    processed = [r for r in results if r.get("status") == "processed"]
    needed_align = [r for r in processed if r.get("needs_alignment")]
    already_aligned = [r for r in processed if not r.get("needs_alignment")]
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Auto Rectify Report — {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1600px; }}
        h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 10px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #e8f5e9; padding: 15px 25px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
        .img-container {{ margin: 25px 0; }}
        .img-container img {{ max-width: 100%; border: 1px solid #ddd; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 6px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #283593; color: white; }}
        .success {{ color: #2e7d32; }}
        .warning {{ color: #c62828; }}
    </style>
</head>
<body>
    <h1>Авто-выпрямление таблиц в data/in_data</h1>
    <p><strong>Дата запуска:</strong> {timestamp}</p>
    
    <div class="stats">
        <div class="stat-card"><div class="stat-value">{total}</div><div>Всего файлов</div></div>
        <div class="stat-card"><div class="stat-value">{len(processed)}</div><div>Обработано успешно</div></div>
        <div class="stat-card"><div class="stat-value">{len(needed_align)}</div><div>Требовало выравнивания</div></div>
        <div class="stat-card"><div class="stat-value">{len(errored)}</div><div>Ошибок</div></div>
    </div>
"""
    
    for r in results:
        if r.get("status") == "error":
            html += f"""
    <div class="img-container">
        <h3>{r['file']}</h3>
        <p class="warning">❌ Ошибка: {r.get('error', 'Unknown')}</p>
    </div>
"""
            continue
        
        status = "✅ Выровнено" if r.get("aligned") else "✅ Уже выровнено"
        html += f"""
    <div class="img-container">
        <h3>{r['file']} — <span class="{'success' if r.get('aligned') else ''}">{'требовало выравнивания' if r.get('needs_alignment') else 'не требовало выравнивания'}</span></h3>
        <table>
            <tr><th>Файл</th><td>{r['file']}</td></tr>
            <tr><th>Размер выхода</th><td>{r.get('output_size', 'N/A')} [H, W]</td></tr>
            <tr><th>Углы (C1/C2/C3/C4)</th><td>{[[round(c,1) for c in corner] for corner in r['corners']]}</td></tr>
            <tr><th>Метрики</th><td>{r['metrics']}</td></tr>
        </table>
        <img src="{Path(r.get('comparison_path', '')).name}" alt="Comparison">
    </div>
"""
    
    html += f"""
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999;">
        <p>Сгенерировано: {timestamp}</p>
    </footer>
</body>
</html>
"""
    
    report_path = output_dir / "index.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ HTML-отчёт создан: {report_path}")
    return report_path


def main():
    input_dir = Path("data/in_data")
    output_base = Path("experiment_outputs/in_data_rectify")
    output_base.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Авто-выпрямление таблиц в: {input_dir.absolute()}")
    print(f"💾 Результаты сохраняются в: {run_dir.absolute()}")
    print(f"📷 Обрабатываю изображения...\n")
    
    # Поддерживаемые форматы
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(set(image_files))
    
    if not image_files:
        print("⚠️ Нет изображений для обработки")
        return
    
    results = []
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}", end=" → ")
        result = process_image(img_path, run_dir)
        results.append(result)
        
        if result.get("status") == "error":
            print(f"❌ {result.get('error')}")
        elif result.get("aligned"):
            print(f"✅ {result['metrics']['avg_angle_deviation_deg']}° → выровнено")
        elif result.get("needs_alignment"):
            print(f"⚠️ {result['metrics']['avg_angle_deviation_deg']}° → выравнивание применено")
        else:
            print(f"✅ {result['metrics']['avg_angle_deviation_deg']}° → уже выровнено")
    
    # Генерируем отчёт
    report_path = generate_report(results, run_dir)
    
    # Открываем в браузере
    import webbrowser
    webbrowser.open(str(report_path.absolute()))
    
    print(f"\n✅ Готово! Отчёт открыт в браузере: {report_path}")


if __name__ == "__main__":
    main()