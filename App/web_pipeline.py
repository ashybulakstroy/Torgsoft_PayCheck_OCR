"""Single-photo pipeline shared by the local web server and integration checks."""
from __future__ import annotations

import csv
import io
import json
import re
import threading
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR
from rapidfuzz import fuzz, process

from App.receipt_rectification import rectify_table
from App.rotation_rectification import rotate_table_crop
from App.ocr_catalog_eval import (ROOT, PASS_SIMILARITY, CatalogMatcher, aligned_lines,
                                  choose_row_boundaries, enforce_cell_boundaries, load_catalog,
                                  needs_review, parse_rows, review_reason, table_layout)
from App.ocr_run import MODEL
from App.table_boundary import detect_table_boundary
from App.receipt_editing import initialize_row, save_receipt, update_receipt
from App.manual_mappings import ManualMappings
from App.quantity_cells import recover_quantities
from App.numeric_cell_split import recover as recover_numeric_cells
from App.row_cell_split import repair as split_merged_product_rows

MAX_PIXELS = 25_000_000


def decode_photo(data):
    try:
        with Image.open(io.BytesIO(data)) as photo:
            if photo.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Поддерживаются фотографии JPG, PNG и WEBP.")
            if photo.width * photo.height > MAX_PIXELS:
                raise ValueError("Слишком большое фото. Максимальное разрешение — 25 мегапикселей.")
            rotated = photo.getexif().get(274, 1) != 1
            oriented = ImageOps.exif_transpose(photo)
            if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                oriented = Image.alpha_composite(background, rgba)
            image = cv2.cvtColor(np.asarray(oriented.convert("RGB")), cv2.COLOR_RGB2BGR)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Не удалось прочитать изображение. Выберите исправный JPG, PNG или WEBP.") from error
    if min(image.shape[:2]) < 80:
        raise ValueError("Изображение слишком маленькое для распознавания чека.")
    return image, rotated


def grid_cells(layout, rows=None):
    xs = [layout.get("body_left", 16), layout["name_left"], layout["name_right"], layout["quantity_right"], layout["right"]]
    bands = [(index+1, start, end, "cell") for index, (start, end) in enumerate(zip(layout["row_boundaries"], layout["row_boundaries"][1:]))]
    if rows is not None:
        bands = [(row["row"], *row["y_range"], row["kind"]) for row in rows]
    return [dict(id=f'{row_id}-{role}', row=row_id, role=role, kind=kind,
                 box=[[left, start], [right, start], [right, end], [left, end]])
            for row_id, start, end, kind in bands for role, left, right in
            zip(("number", "name", "quantity", "price"), xs, xs[1:])]


class _ObservedCall:
    def __init__(self, delegate, callback):
        self.delegate, self.callback = delegate, callback

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, *args, **kwargs):
        result = self.delegate(*args, **kwargs)
        self.callback(result)
        return result


class ReceiptPipeline:
    def __init__(self, knowledge=None, *, split_product_rows=True):
        self.split_product_rows = split_product_rows
        self.knowledge = knowledge if knowledge is not None else ManualMappings()
        self.engine = None
        self.matcher = None
        self.catalog_signature = None
        self.catalog_count = 0
        self.ocr_lock = threading.Lock()
        self.catalog_lock = threading.RLock()
        self.catalog_index = {}

    def ensure_catalog(self):
        directory = ROOT / "data/reference/torgsoft_ref"
        signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(directory.glob("*.xlsx")) if not p.name.startswith("~$"))
        with self.catalog_lock:
            if signature != self.catalog_signature:
                entries, workbooks = load_catalog(directory)
                self.matcher, self.catalog_count = CatalogMatcher(entries, knowledge=self.knowledge), len(entries)
                self.catalog_index = {entry["normalized"]:entry for entry in entries}
                self.catalog_signature = signature

    def catalog_entry(self, key):
        self.ensure_catalog()
        return self.catalog_index.get(key)

    def remember_selection(self, before, after, payload, job_id):
        if not payload.get("catalog_key") or not payload.get("confirm_name"):
            return None
        previous = next((row for row in before["rows"] if row["row"] == payload.get("row")), None)
        if not previous or previous["kind"] != "product":
            return None
        if not (payload.get("learn_mapping") or previous.get("needs_name_review") or
                previous.get("catalog_key") != payload["catalog_key"]):
            return None
        selected = next(row for row in after["rows"] if row["row"] == previous["row"])
        entry = self.catalog_entry(selected["catalog_key"])
        return self.knowledge.remember(previous["name_raw"], entry, selected["article"], job_id, previous["row"])

    def search_catalog(self, query):
        self.ensure_catalog()
        query = str(query).strip()[:200]
        if len(query) < 2:
            return []
        from App.ocr_catalog_eval import normalize_name
        direct = [entry for entry in self.matcher.entries if any(query.casefold() == source["article"].casefold() for source in entry["sources"])]
        ranked = [self.matcher.entries[index] for _, _, index in process.extract(normalize_name(query), self.matcher.names, scorer=fuzz.WRatio, limit=8)]
        found = {}
        for entry in direct + ranked:
            found.setdefault(entry["normalized"], dict(key=entry["normalized"], name=entry["name"],
                articles=sorted({source["article"] for source in entry["sources"] if source["article"]})))
        return list(found.values())[:8]

    def resources(self):
        self.ensure_catalog()
        if self.engine is None:
            self.engine = RapidOCR(rec_model_path=str(MODEL), rec_img_shape=[3, 48, 320], text_score=.3)

    def recognize(self, path, job, target, progress, on_partial=None):
        with self.ocr_lock:
            detector, recognizer = self.engine.text_detector, self.engine.text_recognizer

            detected = []
            def boxes_ready(result):
                boxes = result[0]
                detected[:] = list(self.engine.sorted_boxes(boxes)) if boxes is not None else []
                job.publish("boxes", target=target, boxes=boxes.tolist() if boxes is not None else [],
                            title="Обнаружены текстовые блоки", progress=progress)

            self.engine.text_detector = _ObservedCall(detector, boxes_ready)
            class StreamingRecognizer:
                def __getattr__(self, name):
                    return getattr(recognizer, name)

                def __call__(inner, images):
                    results, elapsed = [], 0.
                    size = max(1, recognizer.rec_batch_num)
                    for start in range(0, len(images), size):
                        batch, seconds = recognizer(images[start:start+size])
                        results.extend(batch)
                        elapsed += seconds
                        if on_partial and detected:
                            partial = [dict(text=str(text), confidence=float(confidence), box=np.asarray(box).tolist(),
                                            box_original=np.asarray(box).tolist())
                                       for box, (text, confidence) in zip(detected, results)
                                       if confidence >= self.engine.text_score]
                            on_partial(partial)
                            job.publish("message", title=f"Прочитано блоков: {len(results)} из {len(images)}", progress=progress+int(6*len(results)/len(images)))
                    return results, elapsed
            self.engine.text_recognizer = StreamingRecognizer()
            try:
                result, times = self.engine(str(path))
            finally:
                self.engine.text_detector, self.engine.text_recognizer = detector, recognizer
        return result or []

    def process(self, job):
        started = perf_counter()
        job.phase("input", 5, "Подготавливаем фотографию")
        image, rotated = decode_photo((job.directory / "upload.bin").read_bytes())
        job.image("original", image, "Исходное фото", "Ориентация EXIF учтена" if rotated else "Оригинал сохранён без изменений", role="original")
        job.phase("geometry", 13, "Ищем границы товарной таблицы")
        boundary = detect_table_boundary(image)
        polygon = boundary["polygon"]
        if polygon is None:
            raise ValueError("Не удалось найти рамку таблицы. Снимите чек целиком, ближе и при хорошем освещении.")
        job.publish("contour", target="original", polygon=polygon.tolist(), title="Граница таблицы найдена", progress=24)
        warnings = list(boundary["warnings"])
        header_bottom = max(0, int(polygon[:, 1].min())-4)
        if header_bottom >= 80:
            pad = int(max(15, np.ptp(polygon[:, 0])*.1))
            left = max(0, int(polygon[:, 0].min())-pad)
            right = min(image.shape[1], int(polygon[:, 0].max())+pad)
            job.image("header", image[:header_bottom, left:right], "Crop · шапка документа", "Номер заказа и дата, если они читаются")
        lo = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int)-16)
        hi = np.minimum([image.shape[1], image.shape[0]], np.ceil(polygon.max(axis=0)).astype(int)+17)
        crop = image[lo[1]:hi[1], lo[0]:hi[0]]
        job.image("crop", crop, "Crop · область таблицы", "Отдельное изображение, исходное фото сохранено")
        job.publish("region", target="original", box=[lo.tolist(), [int(hi[0]), int(lo[1])], hi.tolist(), [int(lo[0]), int(hi[1])]], progress=29)
        job.phase("geometry", 32, "Проверяем волны бумаги и выпрямляем таблицу")
        if "partial_contour_replaced_by_full_grid_hypothesis" in boundary["warnings"]:
            rectified, map_x, map_y, metadata = rotate_table_crop(image, polygon)
        else:
            rectified, map_x, map_y, metadata = rectify_table(
                image, polygon, trust_top=boundary["status"]=="detected")
        warnings.extend(metadata["warnings"])
        subtitle = {"wave_grid":"Выпрямлены волны по внутренней сетке; ячейка итога сохранена",
                    "original_crop_fallback":"Сохранена исходная геометрия: требуется проверка"}.get(metadata["method"],"Выпрямлены границы; нижняя ячейка итога сохранена")
        job.publish("message",title=subtitle,progress=37)
        job.image("rectified", rectified, "Rectify · рабочая таблица", subtitle)
        np.savez_compressed(job.directory / "rectification_map.npz", map_x=map_x, map_y=map_y)
        job.json_file("rectification.json", metadata)
        job.phase("layout", 40, "Находим строки и столбцы")
        layout = table_layout(rectified, metadata)
        layout.setdefault("body_left",metadata["margin"])
        job.publish("cells", target="rectified", cells=grid_cells(layout), title="Предварительная разметка ячеек", progress=44)
        job.phase("ocr", 47, "Готовим локальный OCR и справочники")
        self.resources()

        # Header OCR is separate from the table. It cannot contaminate product rows.
        header_lines = []
        header_bottom = max(0, int(polygon[:, 1].min())-4)
        if header_bottom >= 80:
            header_result = self.recognize(job.directory / "header.png", job, "header", 50)
            header_lines = [dict(text=str(text), confidence=float(confidence), box=box) for box, text, confidence in header_result]
            job.json_file("ocr_header.json", dict(lines=header_lines))
            header_text = "\n".join(line["text"] for line in header_lines)
            date = re.search(r"\b\d{2}[./]\d{2}[./]20\d{2}\b", header_text)
            number = re.search(r"(?:тапсырыс|заказ|№|\bno\b|\bne\b)[^\d\n]{0,15}(\d{6,})", header_text, re.IGNORECASE)
            job.publish("header", date=date.group(0) if date else "", order_number=number.group(1) if number else "")
        job.phase("ocr", 58, "Сканируем текст в ячейках")
        def draft_ready(partial):
            provisional = aligned_lines(dict(lines=partial), metadata)
            boundaries, _ = choose_row_boundaries(provisional, layout)
            draft = parse_rows(provisional, layout, boundaries)
            for row in draft:
                row.update(name=row["name_raw"], quantity=row["quantity_raw"], price=row["price_raw"],
                           stage="ocr", match=None, similarity=None, articles=[], needs_review=False,
                           ocr_name_confidence_pct=round(row["ocr_confidence"]*100, 2))
            job.publish("draft_rows", rows=draft, title="Чек собирается из OCR", progress=64)
        raw_result = self.recognize(job.directory / "rectified.png", job, "rectified", 64, on_partial=draft_ready)
        lines = []
        for box, text, confidence in raw_result:
            box = np.asarray(box, np.float32)
            mx = np.clip(box[:, 0], 0, map_x.shape[1]-1).reshape(1, -1)
            my = np.clip(box[:, 1], 0, map_x.shape[0]-1).reshape(1, -1)
            original_box = np.column_stack([cv2.remap(mapping, mx, my, cv2.INTER_LINEAR).reshape(-1) for mapping in (map_x, map_y)])
            lines.append(dict(text=str(text), confidence=float(confidence), box=box.tolist(), box_original=original_box.tolist()))
        if not lines:
            raise ValueError("Текст не распознан. Попробуйте более чёткую фотографию.")
        job.json_file("ocr_rectified.json", dict(lines=lines, variant="rectified"))
        (job.directory / "ocr_rectified.txt").write_text("\n".join(line["text"] for line in lines), encoding="utf-8")
        job.phase("layout", 74, "Объединяем строки товара и отделяем НДС")
        aligned = aligned_lines(dict(lines=lines), metadata)
        boundaries, segmentation = choose_row_boundaries(aligned, layout)
        layout["row_boundaries"] = enforce_cell_boundaries(aligned, layout, boundaries)
        rows = parse_rows(aligned, layout, layout["row_boundaries"])
        total_candidates = [line for line in aligned if line["center"][1] >= layout["body_bottom"]-5
                            and line["center"][0] > layout["name_right"] and re.search(r"\d[.,]\d{2}", line["text"])]
        total = max(total_candidates,key=lambda line:line["center"][1])["text"] if total_candidates else ""
        if self.split_product_rows:
            split_rows,row_split_check = split_merged_product_rows(dict(rows=rows,total_raw=total),layout,lines,rectified)
        else:
            split_rows,row_split_check = dict(rows=rows),[dict(applied=False,reason='disabled_for_independent_baseline')]
        rows = split_rows['rows']
        job.json_file('row_split.json',row_split_check)
        restored_count = sum(item['applied'] for item in row_split_check)
        if restored_count:
            job.publish('message',title=f'Склеенные товарные строки разделены: {restored_count}; печатная граница, числа и итог проверены',progress=76)
        with self.ocr_lock:
            recovered,quantity_check,_ = recover_quantities(dict(rows=rows,total_raw=total),rectified,layout,
                lambda crops:self.engine.text_recognizer(crops)[0])
        job.json_file("quantity_cells.json",quantity_check)
        if quantity_check["attempts"]:
            if quantity_check["applied"]:
                recovered_rows = {row["row"]:row for row in recovered["rows"] if row.get("quantity_source")=="cell_ocr"}
                for row in rows:
                    if row["row"] in recovered_rows:
                        found = recovered_rows[row["row"]]
                        row.update(quantity=found["quantity"],quantity_source="cell_ocr",quantity_ocr=found["quantity_ocr"])
                job.publish("message",title=f"Количество прочитано отдельно по ячейкам: {len(recovered_rows)}; сумма и НДС проверены",progress=77)
            else:
                job.publish("message",title="Повторное чтение количеств не подтверждено; строки оставлены на проверку",progress=77)
        split_receipt,numeric_split = recover_numeric_cells(dict(rows=rows,total_raw=total),layout,lines)
        rows = split_receipt['rows']
        job.json_file('numeric_split.json',numeric_split)
        split_count = sum(item['applied'] for item in numeric_split)
        if split_count:
            job.publish('message',title=f'Количество и цена разделены по исходному OCR: {split_count}; сумма и НДС проверены',progress=78)
        cells = grid_cells(layout, rows)
        job.json_file("layout.json", dict(layout=layout, cells=cells, segmentation=segmentation))
        job.publish("cells", target="rectified", cells=cells, title="Ячейки выделены, НДС — отдельная строка", progress=79)
        for row in rows:
            row.update(name=row["name_raw"], quantity=row.get("quantity",row["quantity_raw"]), price=row.get("price",row["price_raw"]), stage="ocr",
                       match=None, articles=[], similarity=None, needs_review=False,
                       ocr_name_confidence_pct=round(row["ocr_confidence"]*100, 2))
        job.publish("draft_rows", rows=rows, pass_similarity=PASS_SIMILARITY, title="OCR готов. Начинаем подбор названий", progress=80)
        job.phase("matching", 81, "Сопоставляем товары со справочником")
        for index, row in enumerate(rows):
            row["match"] = self.matcher.match(row["name_raw"]) if row["kind"] == "product" else dict(status="excluded_"+row["kind"], best=None, top_matches=[])
            row["ocr_name_confidence_pct"] = round(row["ocr_confidence"]*100, 2) if row["name_raw"].strip() else None
            best = row["match"]["best"] or {}
            row["articles"] = sorted({source["article"] for source in best.get("sources", []) if source["article"]})
            row["similarity"] = best.get("score")
            row["review_reason"] = review_reason(row)
            row["needs_review"] = needs_review(dict(kind=row["kind"], status=row["match"]["status"],
                                                    similarity=best.get("score", ""), articles="; ".join(row["articles"])))
            row["stage"] = "matched"
            initialize_row(row, PASS_SIMILARITY)
            job.publish("row", row=row, progress=81+int(15*(index+1)/max(1, len(rows))), title=f"Обработана строка {index+1} из {len(rows)}")

        header_text = "\n".join(line["text"] for line in header_lines)
        date = re.search(r"\b\d{2}[./]\d{2}[./]20\d{2}\b", header_text)
        number = re.search(r"(?:тапсырыс|заказ|№|\bno\b|\bne\b)[^\d\n]{0,15}(\d{6,})", header_text, re.IGNORECASE)
        products = [row for row in rows if row["kind"] == "product"]
        result = dict(schema_version="torgsoft-review-1.0", filename=job.filename, job_id=job.id,
                      order_number=number.group(1) if number else "", date=date.group(0) if date else "",
                      rows=rows, total_raw=total, product_count=len(products),
                      review_count=sum(row["needs_review"] for row in products), pass_similarity=PASS_SIMILARITY,
                      catalog_names=self.catalog_count, warnings=warnings, elapsed_ms=round((perf_counter()-started)*1000),
                      geometry_status=boundary["status"], rectification_method=metadata["method"],
                      raw_text="\n".join(line["text"] for line in lines),
                      downloads={key:f"/api/jobs/{job.id}/files/{name}" for key, name in
                                 (("json", "receipt.json"), ("csv", "receipt.csv"), ("text", "ocr_rectified.txt"))})
        update_receipt(result)
        save_receipt(job.directory, result)
        return result
