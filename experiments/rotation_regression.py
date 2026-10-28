"""Full-pipeline regression adapter for the reviewed rotation-only fallback."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from App.receipt_editing import parse_number
from experiments.guarded_printed_grid import matches_reviewed_reference, reviewed_disposition
from experiments.web_ocr_regression import current_pipeline, main

REFERENCE = Path(__file__).with_name("references") / "rotation_geometry_user_review.json"


def _number(value):
    parsed = parse_number(value)
    return None if parsed is None else format(Decimal(parsed), "f")


def _matches_review(case, result):
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    source = case["job"]["path"] / "upload.bin"
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference["upload_sha256"]:
        return False, "другой исходный файл"
    rectification = json.loads((case["output"] / "rectification.json").read_text(encoding="utf-8"))
    if rectification.get("method") != reference["rectification_method"]:
        return False, "не применён подтверждённый поворот"
    angle = float(rectification.get("rotation_clockwise_deg", 999))
    if abs(angle - reference["rotation_clockwise_deg"]) > reference["rotation_tolerance_deg"]:
        return False, "изменился подтверждённый угол поворота"
    receipt = result["receipt"]
    actual = [{"kind": row.get("kind"), "quantity": _number(row.get("quantity_raw")),
               "price": _number(row.get("price_raw"))}
              for row in receipt.get("rows", []) if row.get("kind") in {"product", "tax"}]
    if actual != reference["expected_numeric_rows"]:
        return False, "числовые строки отличаются от проверенного фото"
    calculation = receipt.get("calculation", {})
    if (receipt.get("order_number") != reference["expected_order_number"]
            or _number(receipt.get("total_raw")) != reference["expected_total"]
            or calculation.get("validation_status") != "match"):
        return False, "номер, итог или математическая проверка отличаются от эталона"
    return True, "поворот и числовая структура подтверждены пользователем и исходным фото"


def web_candidate(case):
    result = current_pipeline(case)
    verified, detail = _matches_review(case, result)
    disposition = reviewed_disposition(case, result["receipt"])
    if not verified and matches_reviewed_reference(case, result["receipt"]):
        verified = True
        detail = "результат совпал с ранее подтверждённым пользователем эталоном таблицы"
    if disposition == "tolerated_unchanged":
        detail = "небольшое изменение текста принято пользователем как отсутствие изменения"
    result.update(verified=verified, review_disposition=disposition, detail=detail)
    return result


if __name__ == "__main__":
    main("experiments.rotation_regression:web_candidate", "full_pipeline")
