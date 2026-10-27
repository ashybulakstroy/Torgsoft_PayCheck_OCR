import json
from pathlib import Path

from App.batch_geometry_eval import evaluate_image
from App.geometry_review import build_review_rows


def test_build_review_rows_reads_summary():
    rows = build_review_rows("eval_outputs/summary.json")

    assert len(rows) > 0
    first = rows[0]
    assert "file" in first
    assert "method" in first
    assert "score" in first
    assert isinstance(first["score"], float)


def test_evaluate_image_writes_result_metadata():
    image_path = Path("data/in_data/WhatsApp Image 2026-10-23 at 09.34.47.jpeg")
    result = evaluate_image(image_path)
    meta_path = Path("eval_outputs") / image_path.stem / "result.json"

    assert meta_path.exists(), "Per-image result metadata must be written"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["file"] == image_path.name
    assert payload["method"] == result["method"]
    assert "safe_to_warp" in payload
