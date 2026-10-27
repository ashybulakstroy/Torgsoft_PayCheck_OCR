# PROJECT_CONTEXT.md — вся логика проекта в одном файле

Файл-контекст для передачи проекта другому ИИ или разработчику. Он описывает цель,
архитектуру, алгоритмы, контракты данных, команды и ограничения. **Источник истины —
код в `App/`**; при расхождении верен код, а этот файл нужно обновить.

Проект: `Prj_35_OCR_prep` — подготовка мобильных фотографий чеков/предзаказов
Торгсофт к OCR: поиск границы товарной таблицы, геометрическая коррекция,
нелинейное выпрямление, локальный OCR и сопоставление строк со справочником XLSX.

---

## 1. Главный принцип

Таблица на реальном фото — **не прямоугольник**. Её контур часто имеет форму «¬»:
изогнутый верх, ступенька/расширение ячейки итога снизу, разрывы рамки. Поэтому:

- детектор возвращает **полигон произвольной формы**, а не 4 угла;
- запрещено сводить полигон к `convex hull` или `approxPolyDP` до 4 точек «для красоты»;
- перспективный `warp` допустим только для настоящего выпуклого четырёхугольника;
- при сомнении возвращается `needs_review`, а исходные пиксели сохраняются;
- `status=detected` означает «найдена граница с поддержкой сетки», но **не** разрешение
  на warp и **не** доказанную точность OCR;
- исходник и выпрямленный вариант — разные сущности; нельзя подменять одно другим.

---

## 2. Окружение и запуск

Обязательно проектное окружение `.venv` (Windows, PowerShell):

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m App.contour_review
& .\.venv\Scripts\python.exe -m App --workers 1
& .\.venv\Scripts\python.exe -m App.contour_unwarp
& .\.venv\Scripts\python.exe -m App.ocr_run
& .\.venv\Scripts\python.exe -m App.ocr_catalog_eval
& .\.venv\Scripts\python.exe -m App.ocr_catalog_eval --reuse-matches
```

Не использовать глобальные `python`/`pip`/`pytest`. Зависимости ставить только через
`& .\.venv\Scripts\python.exe -m pip ...`.

Зависимости (`requirements-ocr-eval.txt`, плюс OpenCV/NumPy окружения):

- `rapidocr-onnxruntime==1.2.3` — OCR;
- `openpyxl==3.1.5` — чтение XLSX-справочников;
- `rapidfuzz==3.14.6` — быстрый поиск кандидатов;
- `opencv-python` (`cv2`), `numpy`.

OCR полностью локальный, изображения во внешние сервисы не отправляются.

---

## 3. Структура репозитория

```text
App/
  __init__.py              # пакет
  __main__.py              # python -m App -> batch_geometry_eval.main()
  table_boundary.py        # единый детектор границы таблицы (версия 2.0)
  ocr_geometry.py          # совместимость, order/warp/overlay, rectify_receipt
  batch_geometry_eval.py   # пакетная оценка геометрии, метрики, решение о warp
  contour_unwarp.py        # нелинейное выпрямление Coons-патчем, preview
  contour_review.py        # визуальная проверка детектора -> debug_notes/contour_fix
  geometry_review.py       # markdown-сводка по eval_outputs/summary.json
  ocr_eval.py              # legacy: OCR всего текста + поиск по каталогу
  ocr_run.py               # OCR выпрямленной таблицы и исходного кропа
  ocr_catalog_eval.py      # разбор строк таблицы + сопоставление со справочником
  tests/
    conftest.py            # добавляет ROOT в sys.path
    test_geometry.py
    test_contour_unwarp.py
data/
  in_data/                 # 22 исходных JPEG
  reference/torgsoft_ref/  # *.xlsx справочники (название, артикул, ...)
models/
  cyrillic_PP-OCRv5_rec_mobile.onnx   # кириллическая rec-модель OCR
  cyrillic_PP-OCRv3_rec_mobile.onnx
  cyrillic_v5_model.json, RapidOCR_LICENSE
eval_outputs/              # артефакты пайплайна; подпапка на каждое фото
debug_notes/               # результаты визуальной проверки контуров
debug_outputs/             # старые debug-файлы
docs/                      # отчёты baseline / contour_fix / unwarp / OCR
README.md, receipt_geometry_spec.md, photo_geometry_review.md
```

---

## 4. Поток данных (полный пайплайн)

```text
data/in_data/*.jpeg
        |
        v
[1] table_boundary.detect_table_boundary(image)
        -> polygon (N>=4), status (detected|needs_review), warnings, candidates
        |
        +--> contour_review.py        -> debug_notes/contour_fix/{results.json,index.html,after.jpg}
        |
        +--> batch_geometry_eval.py   -> eval_outputs/summary.json
        |                              eval_outputs/<stem>/{result.json,overlay.png}
        |                              (warped.png только если безопасно)
        |
        +--> contour_unwarp.build_contour_map(polygon)
        |        -> map_x/map_y (Coons patch), rectified.png
        |        -> eval_outputs/rectification_summary.json
        |        -> eval_outputs/<stem>/{rectification.json,rectification_map.npz,
        |                                rectified.png,warped.png}
        |
        +--> ocr_run.py
        |        -> OCR двух вариантов: original_crop и rectified
        |        -> eval_outputs/<stem>/ocr_{original_crop,rectified}.{json,txt}
        |        -> eval_outputs/ocr_run.json
        |
        +--> ocr_catalog_eval.py
                 -> разбор строк таблицы (layout + row boundaries)
                 -> нормализация + поиск в XLSX-справочнике
                 -> eval_outputs/ocr_catalog_{summary.json,matches.csv,
                    remaining.csv,matches.xlsx,report.html}
                    eval_outputs/<stem>/ocr_reference_matches.{json,csv}
```

`ocr_eval.py` — старый эксперимент по полному тексту документа (`ocr_comparison.json`),
не текущий построчный отчёт. Актуальные файлы имеют префикс `ocr_catalog_`.

---

## 5. `App/table_boundary.py` — детектор границы

`ALGORITHM_VERSION = "table-boundary-2.0"`

### 5.1. `BoundaryOptions` (frozen dataclass)

| Поле | Значение | Смысл |
| --- | --- | --- |
| `max_detection_side` | 1600 | длинная сторона рабочей копии |
| `adaptive_block` | 31 | блок адаптивного порога |
| `adaptive_c` | 3 | константа порога |
| `line_length` | 19 | длина направленного ядра |
| `angle_range` | 15 | ± градусов вокруг 0/90 |
| `angle_step` | 3 | шаг углов |
| `closing_sizes` | (3,5,7) | размеры соединения разрывов |
| `min_area_ratio` | .004 | мин. площадь/кадр |
| `min_box_fill` | .65 | мин. заполнение bbox |
| `min_segment_fraction` | .3 | мин. длина сегмента сетки |
| `min_grid_lines` | 3 | мин. число линий по каждой оси |
| `max_vertical_lines` | 6 | отсев плотных полос штрихкода |
| `polygon_tolerance_px` | 1.5 | допуск `approxPolyDP` |

### 5.2. Алгоритм

1. Уменьшить изображение так, чтобы длинная сторона была ≤ `max_detection_side`
   (масштаб `scale`, обратный пересчёт координат — по фактическим `source_w/width`,
   `source_h/height`).
2. `adaptiveThreshold(GAUSSIAN_C, BINARY_INV)`.
3. `_line_masks`: для баз 0° и 90° перебрать углы `base ± angle_range` с шагом
   `angle_step`; нарисовать линейное ядро `n×n`; перед открытием замкнуть разрывы
   малым ядром 7×7; объединить маски через `OR`.
4. `network = close(mask_h | mask_v, 5×5)`.
5. `HoughLinesP` по каждой маске -> сегменты (ось 0 = горизонтальные, ось 1 = вертикальные).
6. Кандидаты — контуры из трёх источников: `grid` (network), `adaptive` (порог),
   `edges` (Canny 20/80), для каждого `gap` из `closing_sizes`.
7. Фильтры контура: площадь ≥ `min_area_ratio`, `w ≥ 50`, `h ≥ 30`, не касается краёв
   кадра, `area/(w*h) ≥ min_box_fill`.
8. `_grid_support`: считает число различимых линий сетки по каждой оси внутри bbox
   (учитывает длину и направление сегментов, объединяет близкие позиции).
9. `ink`: сумма ненулевых пикселей масок внутри эрозированного контура.
10. `continuity = arcLength / (2*(w+h))` — штраф за рваную/присоединённую границу.
11. `score = ink / sqrt(area) * min(support_x,16) * min(support_y,5) / continuity^4`.
12. `_internal_vertical_span`: доля высоты, поддержанная двумя внутренними
    вертикальными разделителями (чтобы страница вокруг маленькой таблицы не выигрывала).
    `score *= min(1, column_span/.9)^4`.
13. Кандидат попадает в основной список, только если
    `min(support) >= min_grid_lines` и `support_y <= max_vertical_lines`.
14. Fallback для чистых сканов: если кандидатов нет, фон почти белый
    (`mean(gray>245) > .9`) и есть `edges` с `continuity < 1.3`, берётся самый большой
    outline-кандидат и добавляется warning `outline_without_grid`.
15. Если кандидатов нет: `polygon=None`, `status="needs_review"`,
    `warnings=["table_boundary_not_found"]`.
16. Сортировка по `score`. Если у лучшего `support_x <= 4`, ищется объемлющий кандидат
    (`area > 4×`, `support_x >= 6`, вложен, больший unweighted score) — предпочитается
    полная таблица, warning `partial_and_full_table_candidates`.
17. Уточнение полигона: заполнить контур, `MORPH_OPEN` эллипсом 5×5 (убрать текстовые
    отростки, **не** заполняя вогнутости), взять внешний контур,
    `approxPolyDP` с допуском `max(1.5, min(6, .008*min(w,h)))`.
18. Масштабировать точки в координаты оригинала.
19. Warnings:
    - `boundary_has_gaps_or_attached_text` при `continuity > 1.15`;
    - `ambiguous_boundary_candidates` если конкурент с score ≥ 85% лучшего имеет
      IoU < .8.
20. `status = "needs_review" if warnings else "detected"`.

### 5.3. Возвращаемый словарь

```python
{
  "polygon": np.ndarray (N,2) float32 | None,
  "status": "detected" | "needs_review",
  "warnings": list[str],
  "algorithm_version": "table-boundary-2.0",
  "selected": {score, area, bounds, support, source, gap, continuity, column_span},
  "candidates": [ ...до 10... ]
}
```

Баллы ранжируют кандидатов **внутри одного изображения**; это не вероятности.

---

## 6. `App/ocr_geometry.py` — совместимость и безопасный вход

### 6.1. Функции

- `clamp_corners_to_image(corners, image_shape)` — клип координат в кадр.
- `order_corners(corners)` -> `(4,2)` в порядке TL, TR, BR, BL.
  Берёт две верхние и две нижние точки (по `y`), затем сортирует по `x`.
  При <4 уникальных точках возвращает bbox-прямоугольник.
  Сортировка по углу вокруг центроида **не используется** (ломается на трапециях).
- `detect_table_polygon(image)` — вызывает `detect_table_boundary`; если полигон есть,
  для 4 точек возвращает `order_corners`, иначе полный полигон. Если таблицы нет,
  возвращает рамку кадра **только для обратной совместимости**.
- `detect_table_corners(image)` — legacy-имя, возвращает то же, что `detect_table_polygon`.
- `perspective_warp(image, corners, width=None, height=None)` — только для конечного
  выпуклого четырёхугольника; иначе возвращает копию входа. Размеры по длинам сторон;
  `getPerspectiveTransform` + `warpPerspective(INTER_LINEAR)`.
- `draw_debug_overlay(image, corners, title)` — полилиния, подписи углов, заголовок.
- `save_debug_outputs(image_path, out_dir)` — overlay + warped + список углов.

### 6.2. `rectify_receipt(image_bytes, *, mode="auto", manual_corners=None, options=None)`

Режимы: `auto`, `original`, `none`, `perspective`, `manual`.

Логика:

1. Декодировать байты в BGR (`cv2.imdecode`).
2. `manual`: проверить 4 конечные различные точки, порядок, выпуклость, площадь;
   boundary = `{"status":"detected","warnings":[],"algorithm_version":"manual"}`.
3. Иначе `detect_table_boundary(image, BoundaryOptions(**options.get("boundary", {})))`;
   4-точечный полигон упорядочить.
4. `quadrilateral = polygon is not None and polygon.shape == (4,2)`.
5. Для `auto`/`perspective`/`manual`: warp только если четырёхугольник и
   (`manual` или (`status=="detected"` и `should_generate_warp(...)`)).
   Иначе `status="needs_review"`, исходник сохраняется.
6. Warnings: `complex_boundary_requires_review` для не-4-точечного полигона;
   `unsafe_perspective_geometry` для небезопасного четырёхугольника.

Результат:

```python
{
  "image": np.ndarray,              # выпрямленное или исходное
  "original_image": np.ndarray,
  "status": "rectified" | "original_fallback" | "manual_rectified" | "needs_review",
  "method": "perspective" | "none",
  "source_polygon": list | None,
  "source_corners": list | None,    # только если реальный четырёхугольник
  "algorithm_version": str,
  "original_size": [h, w],
  "output_size": [h, w],
  "quality_metrics": {
      "manual_override": bool,
      "detected_four_corners": bool,
      "boundary_status": str
  },
  "warnings": list[str]
}
```

Важно: для сложного полигона `source_corners=None`, `method="none"`, исходник
не искажается «выдуманной» трапецией.

---

## 7. `App/batch_geometry_eval.py` — пакетная геометрия

`python -m App [--workers N]` -> `main()`.

- Вход: `data/in_data/*.{jpg,jpeg,png,webp}`; выход: `eval_outputs/`.
- `compute_rectangularity(corners)`:
  длины 4 сторон, `width_ratio = min/max`, среднее отклонение углов от 90°,
  `rectangularity_score = 1 - min(1, angle_dev/25*0.6 + (1-width_ratio)*0.4)`.
- `should_generate_warp(corners, image_shape)` — разрешает warp только если:
  конечные, выпуклые, все стороны >5 px, `width_ratio ≥ .15`,
  `rectangularity_score ≥ .75`, `angle_dev ≤ 10°`, точки не выходят за кадр ±5 px.
- `recommend_method(score)`: `≥.85 -> original_or_deskew`; `≥.70 -> deskew`;
  `≥.55 -> perspective`; иначе `needs_review`.
- `evaluate_image(path)`:
  - вызвать детектор;
  - если полигон не 4 точки или статус не `detected` -> `needs_review`,
    `safe_to_warp=False`, пустые метрики, `warped` = копия;
  - иначе метрики + `recommend_method`; если `should_generate_warp` ложно ->
    `needs_review`;
  - писать `overlay.png`, `result.json`; `warped.png` только если безопасно и ещё нет
    `rectification.json` (не перезаписывать явное выпрямление);
  - удалять устаревший `warped.png`, если он не относится к `rectification.json`.
- Итог: `eval_outputs/summary.json`; параллелизм через `multiprocessing.Pool`,
  `workers = min(args.workers, len(files), cpu_count)`.

`result.json` содержит: `file`, `size`, `corners`, `method`, `metrics`,
`safe_to_warp`, `boundary_status`, `warnings`, `algorithm_version`.

---

## 8. `App/contour_unwarp.py` — нелинейное выпрямление

`python -m App.contour_unwarp` читает `debug_notes/contour_fix/results.json`
(полигоны от `contour_review`) и пишет варианты в `eval_outputs/`.

### 8.1. `build_contour_map(polygon, margin=16, trust_top=True) -> (map_x, map_y, metadata)`

Идея: поверхность Coons по четырём кривым — верх, низ тела таблицы, левый и правый бок.

1. Нормализовать полигон в единичный bbox (`origin`, `extent`); требовать ≥4 точки,
   `min(extent) ≥ 10`.
2. Боковые кривые: только точки в полосах `x<.1` и `x>.9`. Точки группируются по `y`
   (порог .06); берётся внешняя огибающая группы; robust-полином 1-й степени,
   при ≥5 точках и разбросе `y>.7` пробуется квадратичный, если `|a2|<.1`.
3. `_fit`: итеративно перевзвешенный полином (8 итераций, Huber-подобные веса,
   MAD-оценка шума) — изолированные выступы не гнут край.
4. Верх: точки `y < .16` (или `.04` при `trust_top=False`); линейный фит
   `top_fit`; интерполяция по уникальным `x` (`top_curve`), экстраполяция за край по
   наклону. При `trust_top=False` используется только `top_fit` (прямая).
5. Низ тела: из рёбер полигона выбираются длинные нижние почти горизонтальные
   (`dx>.18`, `min(y)>.65`, наклон в пикселях <.35), исключается зона итога справа;
   берётся самое длинное, линейный фит `bottom_fit`.
6. Углы `tl,tr,br,bl` — пересечения боковых кривых с верхом/низом, итеративно (12 шагов).
   `physical = corners*extent+origin`.
7. `width`/`height` тела — средние длины сторон; `body_fraction` > .4.
8. `footer_height`: максимальный выступ полигона ниже `bottom_fit`, нормированный на
   высоту тела; > .25 высоты -> ошибка «Implausibly large total extension».
   Защита выходного размера `>10000`.
9. Сетка `u` (ширина+2*margin), `v` (высота+footer+2*margin); Coons-патч:
   билинейная интерполяция углов + поправки по top/bottom/left/right.
10. За пределами `v∈[0,1]` — линейное продолжение по производной на границе
    (не полиномиальная экстраполяция), чтобы не искажать итог/соседний текст.
11. Перевести в пиксели, проверить якобиан: конечный и `min(jacobian) > .02`, иначе
    «Nonlinear map folds or collapses pixels».

`metadata` включает: `method="contour_coons_patch"`, `algorithm_version="contour-unwarp-1.0"`,
`source_polygon`, `source_body_corners`, `body_size`, `footer_height`, `margin`,
`output_size`, `min/max_jacobian`, `fitted_top/bottom/left/right`, `top_knots`,
`trusted_top`, `normalization_origin/extent`.

### 8.2. Запись артефактов

Для каждого фото в `eval_outputs/<stem>/`:

- `rectified.png` — результат `cv2.remap(... INTER_CUBIC, BORDER_CONSTANT, white)`;
- `warped.png` — идентичная копия для совместимости;
- `rectification_map.npz` — `map_x`, `map_y`;
- `rectification.json` — метаданные + `file`, `output_file`, `status="review_preview"`,
  `boundary_status`, `warnings`, `original_size`.

Сводно: `eval_outputs/rectification_summary.json`, `rectified_contact.jpg`,
`rectified_review.html`. `_save_png` не перезаписывает идентичные байты (Windows может
держать файл открытым в просмотрщике).

`trust_top = (row["status"] == "detected")`: для спорных контуров верх аппроксимируется
прямой.

---

## 9. `App/ocr_run.py` — OCR до/после

1. Читает `eval_outputs/rectification_summary.json`.
2. Профиль OCR: `rapidocr_onnxruntime`, версия пакета, `recognition_model =
   cyrillic_PP-OCRv5_rec_mobile.onnx`, его SHA-256, `text_score=.3`,
   `rec_img_shape=[3,48,320]`, `use_angle_cls=True`.
3. `RapidOCR(rec_model_path=..., rec_img_shape=..., text_score=...)`.
4. Для каждого фото:
   - исходный кроп по bbox полигона с полями −16/+17 px -> `original_table_crop.png`;
   - два варианта: `original_crop` и `rectified`.
5. Кэш: если SHA-256 изображения, SHA модели и `text_score` совпали — OCR не повторяется.
6. Для выпрямленного варианта координаты блоков (`box`) через `map_x/map_y`
   (`cv2.remap`) переводятся обратно в оригинал -> `box_original`. Для кропа —
   прибавление `(x0,y0)`.
7. Пишет `ocr_<variant>.json` (текст, confidence, box, box_original, профиль,
   `elapsed_ms`, `phase_seconds`) и `.txt`; итог — `eval_outputs/ocr_run.json`.

Текущий `eval_outputs/ocr_comparison.json` создан старым `ocr_eval.py` (полный текст),
это не построчный отчёт.

---

## 10. `App/ocr_catalog_eval.py` — разбор строк и справочник

`python -m App.ocr_catalog_eval [--reuse-matches]`

### 10.1. Справочник

- `load_catalog(dir)`: все `*.xlsx`; в каждом листе первые 50 строк сканируются на
  заголовок (`название|наименование|товар|name`, `артикул|арт|код|sku`), иначе
  используется позиционная схема A/B. Служебные строки пропускаются; листы без
  данных — с предупреждением. Каждый `source` = `{workbook, sheet, row, article}`;
  в метаданных файла сохраняются `sha256`, `sheets` (`header_row`, `name_column`,
  `article_column`, `mode`) и `warnings`. Строки группируются по нормализованному
  имени; `catalog_conflicts(entries)` возвращает названия с разными непустыми
  артикулами.
- `normalize_name`: NFKC, casefold, `ё->е`, десятичная запятая -> точка,
  `x/х/×/*` между числами -> `x`, удаление не-`\w`, перевод визуально похожих
  кириллических букв `авсенкмортху` -> `abcehkmoptxy`.
- `numbers`: множество нормализованных чисел из текста.

### 10.2. `CatalogMatcher.match(raw_text)`

- `rapidfuzz.process.extract(..., scorer=fuzz.WRatio, limit=15)`;
- штраф за отсутствующие в справочнике числа: `score *= 1 - .15*|missing|/|query_numbers|`;
- `gap = best - second`;
- статусы:
  - `exact_name` — нормализованные строки равны;
  - `suggested` — score ≥ 90, gap ≥ 5, нет несовпавших чисел;
  - `review` — score ≥ 55;
  - `not_found` — иначе;
  - `unreadable` — запрос короче 3 символов или без букв.
- Кэш по нормализованному запросу.

### 10.3. Разбор таблицы

- `table_layout(rectified_image, metadata)`:
  - адаптивный порог 31/9;
  - вертикальные разделители по проекции после морфологии (длинные открытия);
  - `name_left`, `name_right`, `quantity_right`, `right`, `body_top/bottom`;
  - горизонтальные линии — по проекции в полосе названия + отдельная узкая полоса
    номеров строк; группировка позиций, добавление границ тела.
- `aligned_lines(payload, metadata, original=False)`:
  - для `original` блоки переводятся гомографией `source_body_corners -> тело` и
    центрируются в общей системе координат;
  - для `rectified` используются как есть.
- `choose_row_boundaries(lines, layout)`: базовые линии сетки + разрезание полос по
  нескольким ценовым блокам + возможная граница под шапкой. Режим
  `grid_with_price_splits` либо `grid`.
- `parse_rows`: для каждой полосы собирает текстовые блоки в колонки:
  `item_number`, `name`, `quantity`, `price`; классификация строки
  (`classify_name`): `header`, `tax` (НДС), `service` (доставка), `total` (итог),
  `product`, `unreadable`. Возвращает строки с `name_raw`, `quantity_raw`,
  `price_raw`, `ocr_confidence`, `raw_lines`.
- `classify_name` распознаёт заголовки (`атау`, `наименование`, `название`),
  `НДС/налог`, `доставка`, `итог/корыт/қорыт`.
- `stats`: число товарных строк, распределение статусов, среднее сходство.

### 10.4. Отчёты

Для каждого фото:

- `ocr_reference_matches.json` (до/после, layout, boundaries, segmentation, stats,
  `original_alignment="approximate_homography"`);
- `ocr_reference_matches.csv`.

Сводно:

- `eval_outputs/ocr_catalog_summary.json`;
- `eval_outputs/ocr_catalog_matches.csv` / `ocr_catalog_remaining.csv`;
- `eval_outputs/ocr_catalog_matches.xlsx` — листы «Сопоставление» и «Проверить»,
  подсветка: сходство ≥ 87 — зелёная, иначе жёлтая;
- `eval_outputs/ocr_catalog_report.html` — интерактивные фильтры и поиск.

`needs_review(row)`: товарная строка, у которой сходство <87, статус
`review/not_found/unreadable`, либо несколько артикулов.
`review_reason`: сходство <87, несовпавшие числа, близкие кандидаты (gap<5),
несколько артикулов.

Терминология: `exact_name` — совпадение нормализованного названия; `suggested` —
вероятный вариант; `review` — проверить; `not_found` — кандидат не найден.
Сходство — поисковая метрика, **не** точность OCR; confidence — оценка движка,
**не** калиброванная вероятность. В справочнике нет цен; количество — складской
остаток, не количество в чеке. Количество/цена сохраняются как прочитаны.

---

## 11. `App/ocr_eval.py` (legacy)

- `ocr_text(path)` — OCR всего изображения в одну строку.
- `text_metrics` — символы, слова, цифры, «ценовые» числа, токены, превью.
- `load_reference_catalog` — старый загрузчик первого столбца XLSX.
- `best_reference_match` — `SequenceMatcher` по всему тексту.
- `compare_ocr_for_summary` -> `eval_outputs/ocr_comparison.json`.
- `write_ocr_summary` -> markdown-сводка.

Используется тестом `test_reference_catalog_matches_known_product_names`.

---

## 12. `App/contour_review.py` и `geometry_review.py`

`contour_review.main()`:

- по всем изображениям `data/in_data`: детектор, `elapsed_ms`, полигон в список;
- `debug_notes/contour_fix/after_XX.png`, `after.jpg`, `results.json`, `index.html`;
- печатает `{photos, needs_review, report}`.

`geometry_review.py` читает `eval_outputs/summary.json` и пишет
`eval_outputs/geometry_review.md` с методами/баллами и ссылками на overlay/warped.

---

## 13. Тесты

`python -m pytest -q` (через `.venv`). Покрытие:

- `test_geometry.py`:
  - детекция прямоугольного синтетического документа;
  - `order_corners` при дублях, на трапеции, на реальной границе;
  - `should_generate_warp` отклоняет экстремальную геометрию;
  - реальное фото `09.34.47` даёт сложный полигон ≥5 точек и не warp-ится;
  - клип отрицательных координат;
  - fallback по текстовой области;
  - сохранение сложных рёбер и изогнутого верха;
  - загрузка справочника и поиск известного названия.
- `test_contour_unwarp.py`:
  - изогнутый верх -> плоский, полоса итога сохраняется (15..30 px);
  - вершины итога не наклоняют боковые края;
  - отсутствие верхнего левого угла -> устойчивая экстраполяция.

Известная особенность: тест справочника зависит от наличия `data/reference/torgsoft_ref/*.xlsx`.

---

## 14. Ключевые инварианты (не нарушать при изменениях)

1. Полигон таблицы — не прямоугольник; не применять hull/принудительные 4 точки.
2. `detected` ≠ разрешение на warp; решение принимает `should_generate_warp`.
3. Сложный или небезопасный контур -> `needs_review` + исходные пиксели.
4. Не подставлять выдуманные `source_corners` или трапецию из bbox.
5. Один итоговый `remap`/`warp` на исходном разрешении; поиск — на уменьшенной копии.
6. Сохранять исходный OCR-текст и координаты; не подменять текстом справочника.
7. Сохранять `algorithm_version` и метаданные преобразований.
8. Кэш OCR — по SHA изображения + SHA модели + порогу.
9. Не перезаписывать явные `rectification.json`/`warped.png` пакетным оценщиком.
10. Все запуски — через `.venv`.

---

## 15. Текущее состояние и ограничения

- Корпус: 22 фото. Полигон-кандидат найден для 22/22; `detected` — 19,
  `needs_review` — 3 (`2026-10-22 at 16.15.19`, `2026-10-23 at 13.57.55`,
  `2026-10-25 at 13.16.05`).
- Все 22 полигона имеют >4 точек, поэтому автоматический перспективный warp не
  выбирается; сохраняется исходник.
- P50/P95 детектора (последовательный прогон, 22 фото, без OCR): ~389 / ~825 ms.
- Справочник: 27 566 уникальных нормализованных названий в 3 XLSX-файлах на
  момент прогона; 12 названий с разными артикулами; 140 товарных строк;
  46 `exact_name`, 65 `suggested`, 29 `review`; 134 зелёных / 6 жёлтых.
- Среднее сходство до/после выпрямления: 92.71 -> 96.13 / 100. Это поисковая
  метрика, не CER/WER. Выпрямление помогает не всем кадрам; оригинал остаётся
  обязательным кандидатом.
- Привязка исходного OCR к строкам — приблизительная гомография; на сильно изогнутой
  бумаге возможны ошибки объединения строк.
- Не гарантируется качество на новых телефонах, сильных поворотах, исчезнувшей рамке
  и других форматах таблиц. Приёмка на 50 размеченных фото, CER/WER, интеграция с 1С
  и веб-форма в этой копии не выполнялись.
- `models/cyrillic_PP-OCRv5_rec_mobile.onnx` — кириллическая rec-модель; пакет
  движка остаётся `rapidocr-onnxruntime==1.2.3`.

---

## 16. Быстрая шпаргалка по данным

`table_boundary` результат: `polygon/status/warnings/algorithm_version/selected/candidates`.

`batch result.json`: `file/size/corners/method/metrics/safe_to_warp/boundary_status/
warnings/algorithm_version`.

`rectification.json`: `method/algorithm_version/source_polygon/source_body_corners/
body_size/footer_height/margin/output_size/min_jacobian/max_jacobian/fitted_* /file/
output_file/status/boundary_status/warnings/original_size`.

`ocr_<variant>.json`: `file/variant/image_sha256/profile/elapsed_ms/phase_seconds/
crop_origin/lines[{text,confidence,box,box_original}]`.

`ocr_reference_matches.json`: `file/segmentation/layout/row_boundaries/before/after/
original_alignment/before_stats/after_stats`.

`ocr_catalog_summary.json`: порог задаётся `SIMILARITY_PASS_THRESHOLD` в `.env`; `catalog_unique_names/workbooks/photos/
after_product_statuses/products/pass_similarity/similarity_passing/similarity_below_threshold/
remaining_review_count/remaining_rows/comparisons`.

---

## 17. Типовые задачи и куда смотреть

- «Неверно найдена граница» -> `table_boundary.py` (`BoundaryOptions`, скоринг,
  `_grid_support`, `_internal_vertical_span`), затем `contour_review.py` + `after.jpg`.
- «Не применяется/применяется лишний warp» -> `batch_geometry_eval.should_generate_warp`,
  `recommend_method`, `ocr_geometry.rectify_receipt`.
- «Кривое выпрямление» -> `contour_unwarp.build_contour_map` (фиты боков/верха/низа,
  footer, `trust_top`, якобиан).
- «Плохой OCR» -> `ocr_run.py` (модель, профиль, кроп, кэш, `box_original`).
- «Неверные строки/колонки» -> `ocr_catalog_eval.table_layout`,
  `choose_row_boundaries`, `parse_rows`, `classify_name`.
- «Неверное сопоставление» -> `normalize_name`, `numbers`, `CatalogMatcher.match`,
  `review_reason`.
- «Сломался отчёт» -> `ocr_catalog_eval.main`, `write_excel`, HTML/CSV/JSON.
- «Нужны метрики/регрессия» -> `App/tests/*`, `eval_outputs/summary.json`,
  `docs/*.md`.

При изменении алгоритма обновить `ALGORITHM_VERSION`/`algorithm_version`,
перегенерировать артефакты и прогнать тесты через `.venv`.
