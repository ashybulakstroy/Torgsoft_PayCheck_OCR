"""
Автоматический детектор необходимости выравнивания таблицы.

Определяет, насколько исходная таблица отклонена от идеального прямоугольника:
1. Отклонение углов от 90°
2. Наклон сторон (не параллельны осям)
3. Эксцесс угла (отклонение точек от идеального прямоугольника)

Если отклонение превышает пороги — применяется выравнивание.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Пороги для определения необходимости выравнивания
ANGLE_DEVIATION_THRESHOLD = 2.0      # Градусы: среднее отклонение угла от 90°
SIDE_ANGLE_THRESHOLD = 3.0          # Градусы: максимальный наклон стороны
MAX_RECT_DEVIATION_THRESHOLD = 10.0  # Пикселей: макс. расстояние точки от идеального прямоугольника


def calculate_angle(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> float:
    """Вычисляет угол при вершине pt2 между точками pt1-pt2-pt3."""
    v1 = pt1 - pt2
    v2 = pt3 - pt2
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 90.0
    cos_angle = np.dot(v1, v2) / norm_product
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


def calculate_side_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """Вычисляет угол наклона стороны относительно горизонтали."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)  # Угол от 0 до 90


def get_ideal_rectangle_deviation(corners: np.ndarray, ideal_corners: np.ndarray) -> float:
    """Максимальное расстояние между реальными углами и идеальным прямоугольником."""
    deviations = []
    for real, ideal in zip(corners, ideal_corners):
        dev = np.linalg.norm(real - ideal)
        deviations.append(dev)
    return max(deviations)


def assess_table_alignment(corners: np.ndarray) -> Dict[str, Any]:
    """
    Оценивает выравнивание таблицы по 4-м углам.
    
    Args:
        corners: Массив из 4 точек [TL, TR, BR, BL] в порядке
        
    Returns:
        dict с метриками и флагом необходимости выравнивания
    """
    corners = np.array(corners, dtype=np.float32)
    
    # Вычисляем углы при каждой вершине
    angles = []
    for i in range(4):
        pt1 = corners[(i - 1) % 4]
        pt2 = corners[i]
        pt3 = corners[(i + 1) % 4]
        angle = calculate_angle(pt1, pt2, pt3)
        angles.append(angle)
    
    # Отклонения углов от 90°
    angle_deviations = [abs(a - 90.0) for a in angles]
    avg_angle_deviation = np.mean(angle_deviations)
    max_angle_deviation = max(angle_deviations)
    
    # Углы наклона сторон
    side_angles = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        angle = calculate_side_angle(p1, p2)
        side_angles.append(angle)
    
    max_side_deviation = max(abs(a - 0) for a in side_angles[:2])  # Верхняя и левая стороны
    avg_side_deviation = np.mean(abs(a - 0) for a in side_angles[:2])
    
    # Идеальный прямоугольник для сравнения
    tl = corners[0]
    tr = corners[1]
    br = corners[2]
    bl = corners[3]
    w = np.linalg.norm(tr - tl)
    h = np.linalg.norm(bl - tl)
    ideal_corners = np.array([
        [tl[0], tl[1]],
        [tl[0] + w, tl[1]],
        [tl[0] + w, tl[1] + h],
        [tl[0], tl[1] + h]
    ], dtype=np.float32)
    
    max_deviation = get_ideal_rectangle_deviation(corners, ideal_corners)
    
    # Вывод: нуждается ли в выравнивании?
    needs_alignment = (
        avg_angle_deviation > ANGLE_DEVIATION_THRESHOLD or
        max_side_deviation > SIDE_ANGLE_THRESHOLD or
        max_deviation > MAX_RECT_DEVIATION_THRESHOLD
    )
    
    return {
        "needs_alignment": needs_alignment,
        "metrics": {
            "avg_angle_deviation_deg": round(avg_angle_deviation, 2),
            "max_angle_deviation_deg": round(max_angle_deviation, 2),
            "max_side_deviation_deg": round(max_side_deviation, 2),
            "avg_side_deviation_deg": round(avg_side_deviation, 2),
            "max_rect_deviation_px": round(max_deviation, 1),
            "angles": [round(a, 1) for a in angles],
            "side_angles": [round(a, 1) for a in side_angles],
        }
    }


class TableAlignmentDetector:
    """Детектор необходимости выравнивания таблицы."""
    
    def __init__(self, 
                 angle_threshold: float = ANGLE_DEVIATION_THRESHOLD,
                 side_threshold: float = SIDE_ANGLE_THRESHOLD,
                 deviation_threshold: float = MAX_RECT_DEVIATION_THRESHOLD):
        self.angle_threshold = angle_threshold
        self.side_threshold = side_threshold
        self.deviation_threshold = deviation_threshold
    
    def detect(self, corners: np.ndarray) -> Dict[str, Any]:
        """Определяет необходимость выравнивания для набора углов."""
        corners = np.array(corners, dtype=np.float32)
        
        angles = []
        for i in range(4):
            pt1 = corners[(i - 1) % 4]
            pt2 = corners[i]
            pt3 = corners[(i + 1) % 4]
            angle = calculate_angle(pt1, pt2, pt3)
            angles.append(angle)
        
        angle_deviations = [abs(a - 90.0) for a in angles]
        avg_angle = np.mean(angle_deviations)
        max_angle = max(angle_deviations)
        
        side_angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            angle = calculate_side_angle(p1, p2)
            side_angles.append(angle)
        
        max_side = max(abs(a - 0) for a in side_angles[:2])
        
        tl = corners[0]
        tr = corners[1]
        br = corners[2]
        bl = corners[3]
        w = np.linalg.norm(tr - tl)
        h = np.linalg.norm(bl - tl)
        ideal_corners = np.array([
            [tl[0], tl[1]],
            [tl[0] + w, tl[1]],
            [tl[0] + w, tl[1] + h],
            [tl[0], tl[1] + h]
        ], dtype=np.float32)
        
        max_dev = get_ideal_rectangle_deviation(corners, ideal_corners)
        
        needs_alignment = (
            avg_angle > self.angle_threshold or
            max_side > self.side_threshold or
            max_dev > self.deviation_threshold
        )
        
        return {
            "needs_alignment": needs_alignment,
            "confidence": min(max_dev / self.deviation_threshold, 1.0) if max_dev > 0 else 0.0,
            "metrics": {
                "avg_angle_deviation_deg": round(avg_angle, 2),
                "max_angle_deviation_deg": round(max_angle, 2),
                "max_side_deviation_deg": round(max_side, 2),
                "max_rect_deviation_px": round(max_dev, 1)
            }
        }


def rectify_table_if_needed(image: np.ndarray, corners: np.ndarray, output_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Автоматически выравнивает таблицу если это необходимо.
    
    Args:
        image: исходное изображение
        corners: 4 угла таблицы [TL, TR, BR, BL]
        output_size: опциональный размер выхода (h, w)
        
    Returns:
        (выпрямленное_изображение, информация_о_выравнивании)
    """
    detector = TableAlignmentDetector()
    detection_result = detector.detect(corners)
    
    if not detection_result["needs_alignment"]:
        return image, {
            "aligned": False,
            "reason": "Table already well-aligned",
            "detection": detection_result
        }
    
    # Нужно выравнивание
    corners = np.array(corners, dtype=np.float32)
    
    # Определяем размер выхода
    if output_size is None:
        src_w = np.linalg.norm(corners[1] - corners[0])
        src_h = np.linalg.norm(corners[3] - corners[0])
        scale = 900 / src_w
        dst_w = int(src_w * scale)
        dst_h = int(src_h * scale)
    else:
        dst_h, dst_w = output_size
    
    dst_corners = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(corners, dst_corners)
    rectified = cv2.warpPerspective(
        image, matrix, (dst_w, dst_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rectified, {
        "aligned": True,
        "reason": "Table was misaligned - applied perspective transform",
        "detection": detection_result,
        "output_size": [dst_h, dst_w]
    }


# Тестовый запуск
if __name__ == "__main__":
    todo_dir = Path("data/to_do")
    
    for todo_item in sorted(todo_dir.iterdir()):
        if not todo_item.is_dir():
            continue
            
        print(f"\nОбрабатываю: {todo_item.name}")
        
        original_path = todo_item / "original.png"
        rect_path = todo_item / "rectification.json"
        
        if not original_path.exists() or not rect_path.exists():
            print("  Пропущен: отсутствуют файлы")
            continue
        
        img = cv2.imread(str(original_path))
        if img is None:
            print("  Ошибка загрузки изображения")
            continue
        
        with open(rect_path) as f:
            data = json.load(f)
        
        polygon = np.array(data["source_polygon"], dtype=np.float32)
        body_corners = np.array(data["source_body_corners"], dtype=np.float32)
        
        # Используем наши углы: C1=polygon[0], C2=polygon[5], C3=body_corners[2], C4=polygon[1]
        src_corners = np.array([
            polygon[0],
            polygon[5],
            body_corners[2],
            polygon[1]
        ], dtype=np.float32)
        
        print(f"  Углы таблицы:")
        for i, (label, c) in enumerate(zip(["C1 (TL)", "C2 (TR)", "C3 (BR)", "C4 (BL)"], src_corners)):
            print(f"    {label}: [{c[0]:.1f}, {c[1]:.1f}]")
        
        # Анализируем
        detector = TableAlignmentDetector()
        result = detector.detect(src_corners)
        
        print(f"  Анализ выравнивания:")
        print(f"    Необходимо выравнивание: {result['needs_alignment']}")
        print(f"    Уверенность: {result['confidence']:.2f}")
        print(f"    Метрики: {result['metrics']}")
        
        # Применяем выравнивание
        rectified, align_info = rectify_table_if_needed(img, src_corners)
        
        print(f"  Результат выравнивания:")
        print(f"    Выровнено: {align_info['aligned']}")
        print(f"    Причина: {align_info['reason']}")
        if align_info['aligned']:
            print(f"    Размер выхода: {align_info.get('output_size', 'N/A')}")
        
        # Сохраняем результат
        output_dir = Path("experiment_outputs/todo_auto_detect")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_img_path = output_dir / f"{todo_item.name}_rectified.jpg"
        cv2.imwrite(str(result_img_path), rectified)
        print(f"  Сохранено: {result_img_path}")