import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from App.dewarp_necessity_detector import assess_dewarp_necessity
from App.ocr_activity import completed_text_rows, emit_status, preliminary_row_count, row_totals
from App.receipt_orientation_detector import orient_receipt
from App.receipt_rectification import rectify_table
from App.visual_sync import SCHEMA_VERSION, emit as emit_visual, raw_row


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiment_outputs" / "refactor_app_dependencies"


class EventJob:
    def __init__(self):
        self.events = []

    def publish(self, event_type, **payload):
        self.events.append({"type": event_type, **payload})


def synthetic_receipt():
    image = np.full((360, 240, 3), 255, np.uint8)
    polygon = np.array([[20, 30], [220, 30], [220, 330], [20, 330]], np.float32)
    for x in (20, 55, 150, 185, 220):
        cv2.line(image, (x, 30), (x, 330), (30, 30, 30), 2)
    for y in (30, 75, 145, 220, 285, 330):
        cv2.line(image, (20, y), (220, y), (30, 30, 30), 2)
    cv2.putText(image, "1", (34, 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
    return image, polygon


def test_app_imports_without_experiments_directory(tmp_path):
    isolated = tmp_path / "isolated"
    shutil.copytree(ROOT / "App", isolated / "App", ignore=shutil.ignore_patterns("__pycache__"))
    assert not (isolated / "experiments").exists()
    modules = [
        "App.web_pipeline", "App.web_server", "App.receipt_rectification",
        "App.ocr_activity", "App.visual_sync", "App.receipt_orientation_detector",
        "App.dewarp_necessity_detector",
    ]
    code = (
        "import sys; sys.path.insert(0, r'" + str(isolated) + "'); "
        + "; ".join(f"import {name}" for name in modules)
        + "; print('AUTONOMOUS_IMPORT_OK')"
    )
    result = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-I", "-c", code],
        cwd=isolated, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "AUTONOMOUS_IMPORT_OK"


def test_isolated_app_runs_geometry_and_web_contract_regressions(tmp_path):
    isolated = tmp_path / "isolated_regression"
    shutil.copytree(ROOT / "App", isolated / "App", ignore=shutil.ignore_patterns("__pycache__"))
    assert not (isolated / "experiments").exists()
    selected = [
        isolated / "App" / "tests" / "test_receipt_rectification.py",
        isolated / "App" / "tests" / "test_web_server.py",
    ]
    result = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-I", "-m", "pytest", "-q",
         *(str(path) for path in selected)],
        cwd=isolated, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout and "failed" not in result.stdout


def test_synthetic_production_components_and_event_contracts():
    image, polygon = synthetic_receipt()

    oriented, orientation = orient_receipt(image, mode="normal")
    assert orientation["status"] == "SKIPPED_PORTRAIT"
    assert orientation["rotation_clockwise"] == 0
    assert orientation["ocr_invocations"] == 0
    assert np.array_equal(oriented, image)

    necessity = assess_dewarp_necessity(image)
    assert necessity["decision"] in {"SKIP", "DEWARP", "UNCERTAIN"}
    assert necessity["needs_dewarp"] == (necessity["decision"] == "DEWARP")
    assert necessity["version"] == "dewarp-necessity-detector-1.1"

    rectified, map_x, map_y, metadata = rectify_table(image, polygon)
    assert rectified.ndim == 3 and rectified.shape[2] == 3
    assert map_x.shape == map_y.shape == rectified.shape[:2]
    assert map_x.dtype == map_y.dtype == np.float32
    assert metadata["method"] in {"contour_coons_patch", "wave_grid", "original_crop_fallback"}
    assert isinstance(metadata["wave_assessment"], dict)

    boundaries = [10, 60, 120]
    boxes = [np.array([[0, 20], [10, 20], [10, 30], [0, 30]], np.float32),
             np.array([[0, 80], [10, 80], [10, 90], [0, 90]], np.float32)]
    assert preliminary_row_count(boundaries) == 2
    assert completed_text_rows(boxes, 1, boundaries) == (1, 2)
    assert row_totals([{"kind": "product"}, {"kind": "tax"}, {"kind": "service"}]) == {
        "products": 1, "tax_rows": 1, "other_rows": 1,
    }

    job = EventJob()
    emit_status(job, "ocr", "OCR идёт", details="synthetic")
    emit_visual(job, "row_ready", row=raw_row({
        "row": 1, "kind": "product", "name_raw": "Товар",
        "quantity_raw": "2", "price_raw": "100",
    }))
    assert job.events == [
        {"type": "ui_status", "stage": "ocr", "title": "OCR идёт", "details": "synthetic"},
        {"type": "visual", "schema": SCHEMA_VERSION, "action": "row_ready", "row": {
            "row": 1, "kind": "product", "name": "Товар", "quantity": "2",
            "price": "100", "status": "raw", "target": "rectified",
        }},
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic_result.json").write_text(json.dumps({
        "orientation": orientation,
        "dewarp": necessity,
        "rectification_method": metadata["method"],
        "events": job.events,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
