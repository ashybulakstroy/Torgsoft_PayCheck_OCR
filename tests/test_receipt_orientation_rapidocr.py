# -*- coding: utf-8 -*-
"""Регрессия Orientation Detector 1.4. Fake OCR, без Tesseract и весов ONNX."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from experiments import receipt_orientation_detector as orient


def fake_receipt():
    image = np.full((640, 340, 3), 255, dtype=np.uint8)
    image[:30, :30, :] = (255, 0, 0)  # синяя метка шапки
    image[-22:, -25:, :] = (0, 255, 0)
    return image


def marker_position(im):
    blue = (im[:, :, 0] > 200) & (im[:, :, 1] < 70) & (im[:, :, 2] < 70)
    yy, xx = np.where(blue)
    if len(yy) < 3:
        return "none"
    row, col = float(yy.mean() / im.shape[0]), float(xx.mean() / im.shape[1])
    return ("T" if row < .5 else "B") + ("L" if col < .5 else "R")


def box(y, h, w):
    return [[10, y*h-8], [w-15, y*h-8], [w-15, y*h+8], [10, y*h+8]]


class FakeRapidOCR:
    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        h, w = img.shape[:2]
        corner = marker_position(img)
        if corner == "TL":
            rows = [
                [box(.18,h,w), "Тапсырыс №2142701789", .93],
                [box(.25,h,w), "Продавец Новый предзаказ", .92],
                [box(.42,h,w), "Тауардың атауы Саны Бағасы", .90],
                [box(.75,h,w), "Қорытынды", .95],
            ]
        elif corner == "BR":
            rows = [
                [box(.82,h,w), "Тапсырыс №2142701789", .93],
                [box(.75,h,w), "Продавец Новый предзаказ", .92],
                [box(.58,h,w), "Тауардың атауы Саны Бағасы", .90],
                [box(.22,h,w), "Қорытынды", .95],
            ]
        else:
            rows = [[box(.4,h,w), "ABXVZD", .25]]
        return rows, [0.01, 0.02, 0.03]


def fake_barcode(im, max_side=1600):
    corner = marker_position(im)
    y = {"TL": .85, "BR": .15}.get(corner)
    return {"found": y is not None,
            "source": "opencv_barcode" if y is not None else "none",
            "center_y_normalized": y}


class OrientationV14Tests(unittest.TestCase):
    def test_normal_portrait_zero_ocr_and_engine_not_required(self):
        image = fake_receipt()
        oriented, result = orient.orient_receipt(image, mode="normal")
        self.assertEqual(result["status"], "SKIPPED_PORTRAIT")
        self.assertEqual(result["rotation_clockwise"], 0)
        self.assertEqual(result["ocr_invocations"], 0)
        np.testing.assert_array_equal(image, oriented)

    def test_normal_portrait_180_is_skipped_by_design(self):
        image = orient.rotate_clockwise(fake_receipt(), 180)
        engine = FakeRapidOCR()
        oriented, result = orient.orient_receipt(image, engine=engine, mode="normal")
        self.assertEqual(result["status"], "SKIPPED_PORTRAIT")
        self.assertEqual(engine.calls, 0)
        np.testing.assert_array_equal(oriented, image)

    @patch.object(orient, "_barcode_location", side_effect=fake_barcode)
    def test_normal_landscape_90_270(self, _mock):
        original = fake_receipt()
        for rotation in (90, 270):
            with self.subTest(rotation=rotation):
                source = orient.rotate_clockwise(original, rotation)
                engine = FakeRapidOCR()
                got, result = orient.orient_receipt(source, mode="normal", engine=engine)
                self.assertEqual(result["status"], "OK")
                self.assertEqual(result["rotation_clockwise"], (-rotation) % 360)
                self.assertLessEqual(engine.calls, 4)
                np.testing.assert_array_equal(got, original)

    @patch.object(orient, "_barcode_location", side_effect=fake_barcode)
    def test_recovery_all_four_orientations(self, _mock):
        original = fake_receipt()
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation):
                source = orient.rotate_clockwise(original, rotation)
                engine = FakeRapidOCR()
                got, result = orient.orient_receipt(source, mode="recovery", engine=engine)
                self.assertEqual(result["status"], "OK")
                self.assertEqual(result["rotation_clockwise"], (-rotation) % 360)
                self.assertEqual(engine.calls, 4)
                self.assertEqual(result["ocr_invocations"], 4)
                np.testing.assert_array_equal(got, original)

    @patch.object(orient, "_barcode_location", side_effect=fake_barcode)
    def test_wide_canvas_valid_orientation_not_forced_to_rotate(self, _mock):
        canvas = np.full((800, 1300, 3), 255, dtype=np.uint8)
        canvas[80:720, 430:770] = fake_receipt()
        engine = FakeRapidOCR()
        got, result = orient.orient_receipt(canvas, engine=engine)
        self.assertEqual(result["rotation_clockwise"], 0)
        np.testing.assert_array_equal(got, canvas)

    def test_recovery_with_empty_ocr_is_uncertain(self):
        class EmptyOCR:
            def __call__(self, im):
                return None, None
        img = fake_receipt()
        with patch.object(orient, "_barcode_location", return_value={
            "found": False, "source": "none", "center_y_normalized": None,
        }):
            got, result = orient.orient_receipt(img, engine=EmptyOCR(), mode="recovery")
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["ocr_invocations"], 4)
        np.testing.assert_array_equal(got, img)

    def test_cli_portrait_does_not_import_model(self):
        # Запуск с реальным subprocess, без установленного rapidocr_onnxruntime.
        with tempfile.TemporaryDirectory() as td:
            im = fake_receipt()
            source = Path(td) / "photo.png"
            output = Path(td) / "out" / "oriented.png"
            cv2.imwrite(str(source), im)
            cmd = [sys.executable, "-m", "experiments.receipt_orientation_detector",
                   str(source), "--output", str(output)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(output.is_file())
            report = json.loads(output.with_suffix(".json").read_text("utf-8"))
            self.assertEqual(report["ocr_invocations"], 0)
            self.assertEqual(report["status"], "SKIPPED_PORTRAIT")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            orient.detect_orientation(fake_receipt(), mode="bad")


if __name__ == "__main__":
    unittest.main()

