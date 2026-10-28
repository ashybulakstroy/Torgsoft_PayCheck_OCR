"""Run the current pipeline on a prioritized to_do receipt without touching web history."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from App.web_pipeline import ReceiptPipeline
from App.web_server import Job

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = "14e13295687b4112b32ce4b27a5c7189"
DEFAULT_OUT = Path(r"R:\Temp\ocr_reports\todo_header_postprocess")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = ROOT / "data" / "to_do" / args.case
    output = args.output / args.case
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "upload.bin", output / "upload.bin")
    result = ReceiptPipeline().process(Job(args.case, "image.png", output))
    summary = dict(
        rows=[[row["row"], row["kind"], row.get("name_raw", ""), row.get("quantity_raw", ""), row.get("price_raw", "")]
              for row in result["rows"]],
        product_count=result["product_count"], total=result["total_raw"],
        calculation=result["calculation"]["validation_status"],
    )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
