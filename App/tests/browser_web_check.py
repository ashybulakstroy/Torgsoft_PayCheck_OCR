"""Explicit real-browser check: run with the project .venv, server must be running."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web_outputs/browser_check"
EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=EDGE, headless=True, args=["--disable-gpu", "--no-first-run"])
        context = browser.new_context(viewport=dict(width=1440, height=1000), device_scale_factor=1)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error:errors.append(str(error)))
        page.goto("http://127.0.0.1:8023", wait_until="networkidle")
        page.screenshot(path=str(OUT / "empty.png"))
        first = ROOT / "data/in_data/WhatsApp Image 2026-10-24 at 13.33.50.jpeg"
        page.locator("#file-input").set_input_files(first)
        page.wait_for_function("document.querySelectorAll('.image-card').length>=2", timeout=60000)
        page.screenshot(path=str(OUT / "processing.png"))
        page.wait_for_function("state.result!==null", timeout=120000)
        result = page.evaluate("state.result")
        stream = page.request.get(f"http://127.0.0.1:8023/api/jobs/{result['job_id']}/events").text()
        events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]
        drafts = [event for event in events if event["type"] == "draft_rows"]
        assert len(drafts) >= 2
        assert all(row["stage"] == "ocr" and row["match"] is None for event in drafts for row in event["rows"])
        assert next(i for i,event in enumerate(events) if event["type"] == "header") < next(i for i,event in enumerate(events) if event["type"] == "draft_rows")
        assert [event["artifact"]["key"] for event in events if event["type"] == "artifact"] == ["original","header","crop","rectified"]
        assert result["calculation"]["status"] == "match"
        assert result["product_count"] == 1, result
        assert next(row for row in result["rows"] if row["kind"] == "product")["name_raw"].endswith("CT15074")
        assert next(row for row in result["rows"] if row["kind"] == "tax")["quantity_raw"] == "5056"
        assert page.locator(".image-card[data-key=rectified] .cell-box").count() == len(result["rows"])*4
        page.locator(".image-card[data-key=rectified]").scroll_into_view_if_needed()
        page.locator("#receipt-body tr").first.hover()
        assert page.locator(".image-card[data-key=rectified] .focused").count() == 4
        assert page.evaluate("window.scrollY") == 0
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "receipt.png"))
        page.locator(".image-card[data-key=rectified] .image-tools button").click()
        assert page.locator("dialog").is_visible()
        page.locator("#close-dialog").click()

        second = ROOT / "data/in_data/WhatsApp Image 2026-10-25 at 13.16.05.jpeg"
        data = dict(name=second.name, data=base64.b64encode(second.read_bytes()).decode())
        page.evaluate("""data=>{const dt=new DataTransfer();dt.items.add(new File([Uint8Array.from(atob(data.data),c=>c.charCodeAt(0))],data.name,{type:'image/jpeg'}));document.dispatchEvent(new DragEvent('drop',{dataTransfer:dt,bubbles:true,cancelable:true}));}""", data)
        page.wait_for_function("state.result!==null", timeout=120000)
        second_result = page.evaluate("state.result")
        assert second_result["product_count"] == 2
        assert [row for row in second_result["rows"] if row["kind"] == "product"][-1]["name_raw"].endswith("50951")
        assert next(row for row in second_result["rows"] if row["kind"] == "tax")["quantity_raw"] == "88656"
        page.locator("#review-toggle").click()
        assert page.locator("#review-toggle").get_attribute("aria-pressed") == "true"
        page.locator("#review-toggle").click()

        data = dict(name=first.name, data=base64.b64encode(first.read_bytes()).decode())
        page.evaluate("""data=>{const dt=new DataTransfer();dt.items.add(new File([Uint8Array.from(atob(data.data),c=>c.charCodeAt(0))],data.name,{type:'image/jpeg'}));document.dispatchEvent(new ClipboardEvent('paste',{clipboardData:dt,bubbles:true,cancelable:true}));}""", data)
        page.wait_for_function("state.result!==null", timeout=120000)
        assert page.evaluate("state.result.product_count") == 1
        page.reload(wait_until="networkidle")
        page.wait_for_function("state.result!==null", timeout=15000)
        assert page.locator("#exports").is_visible()
        page.set_viewport_size(dict(width=390, height=844))
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(OUT / "mobile.png"))
        assert not errors, errors
        (OUT / "results.json").write_text(json.dumps(dict(button_upload=True, drag_drop=True, paste=True, restore=True,
                                                         mobile=True, first_job=result["job_id"], second_job=second_result["job_id"], page_errors=errors), indent=2), encoding="utf-8")
        browser.close()
        print("Browser checks passed: button, drag/drop, paste, live cells, two tax regressions, zoom, reload, mobile.")


if __name__ == "__main__":
    main()
