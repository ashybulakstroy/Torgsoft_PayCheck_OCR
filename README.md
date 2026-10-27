# Receipt table boundary detector

## Local web workspace

```powershell
& .\.venv\Scripts\python.exe -m App.web_server --host 0.0.0.0 --port 8023
```

Open **http://127.0.0.1:8023**. The port is configurable with `--port`. Alternatively: `.\start-web.ps1 -Port 8023`.

Upload by button, drag-and-drop, or clipboard paste. The left panel shows the original, crop and rectified images with live OCR/cell overlays. The right panel shows real processing progress and an electronic receipt, including separate VAT rows, matching confidence and JSON/CSV downloads. Web uploads are saved independently under `web_outputs/`.

See the [implementation plan](docs/web_ui_plan.md) and [web server instructions](docs/web_ui.md).

Detects the printed table boundary in mobile receipt photographs before OCR. The boundary is a polygon: curved top edges and the stepped total cell are preserved.

```powershell
python -m App.contour_review
python -m App --workers 1
```

The first command creates `debug_notes/contour_fix/index.html`, full-resolution overlays, a contact sheet and `results.json`. The second refreshes the existing `eval_outputs` batch artifacts.

- `App/table_boundary.py`: unified detector, `BoundaryOptions`, candidate diagnostics and explicit `needs_review`.
- `App/ocr_geometry.py`: compatibility functions, safe geometry entry point and manual quadrilateral mode.
- `App/batch_geometry_eval.py`: batch selection; complex/uncertain polygons keep the original.
- `App/tests/test_table_boundary.py`: real boundary landmarks, synthetic page/table/barcode separation and fallback regressions.

`detect_table_boundary(image)` returns `polygon`, `status`, `warnings`, algorithm version and candidate diagnostics. The two legacy functions `detect_table_polygon` and `detect_table_corners` now return the same polygon. A polygon is never reduced to four points merely to make perspective correction possible.

Boundary status `detected` means that a grid-supported candidate was found. It is separate from permission to warp and from OCR accuracy. Complex polygons require review; `rectify_receipt` returns the original pixels with `method=none`, `source_polygon`, and no fabricated `source_corners`.

See [baseline](docs/receipt_geometry_baseline.md) and [verification report with limitations](docs/contour_fix_report.md). The current corpus contains 22 photos, including three boundaries requiring review. It is not a completed production OCR pipeline.

Explicit contour rectification previews: run `python -m App.contour_unwarp`. This writes `rectified.png` and a `warped.png` compatibility copy inside each photo folder in `eval_outputs`, plus transformation maps and `rectified_review.html`. See [rectification report](docs/contour_unwarp_report.md).

Local OCR and catalog comparison: `python -m App.ocr_run`, then `python -m App.ocr_catalog_eval`. Raw OCR and row matches are saved in each photo folder. Open `eval_outputs/ocr_catalog_report.html` or `ocr_catalog_matches.xlsx`. See [OCR report](docs/ocr_catalog_report.md).

The catalog matching pass threshold is configured in `.env`: `SIMILARITY_PASS_THRESHOLD=87` (87% and above passes). `python-dotenv` loads it at startup; the code fallback is also 87.
