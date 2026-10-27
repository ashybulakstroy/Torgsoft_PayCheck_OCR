"""Persistent OCR evidence and reports; R: is volatile scratch storage only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR_CACHE = ROOT / '.cache' / 'ocr_baselines'
OCR_REPORTS = ROOT / 'reports' / 'ocr_regression'
EXPERIMENT_OUTPUTS = ROOT / 'experiment_outputs'
