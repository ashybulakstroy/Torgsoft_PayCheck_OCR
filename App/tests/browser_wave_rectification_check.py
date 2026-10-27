"""Real receipt upload through the running production web pipeline on port 8023."""
import json
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from App.wave_dewarp import find_rules

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT/"web_outputs/browser_check"
    out.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",headless=True)
        page = browser.new_page(viewport=dict(width=1600,height=1100))
        errors = []
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto("http://127.0.0.1:8023/?new=1",wait_until="networkidle")
        page.locator("#file-input").set_input_files(ROOT/"App/tests/fixtures/wavy_receipt_596753.png")
        page.wait_for_function("state.result!==null",timeout=120000)
        result = page.evaluate("state.result")
        assert result["rectification_method"] == "wave_grid",result
        assert result["product_count"] == 2
        products = [row for row in result["rows"] if row["kind"]=="product"]
        assert "ники Huter" in products[0]["name_raw"]
        assert "5,5" in products[1]["name_raw"] and "DY6500" in products[1]["name_raw"]
        assert result["total_raw"].replace(" ","").startswith("247780")
        directory = ROOT/"web_outputs"/result["job_id"]
        metadata = json.loads((directory/"rectification.json").read_text(encoding="utf-8"))
        assert metadata["grid_layout"]["grid_confirmed"]
        image = cv2.imdecode(np.frombuffer((directory/"rectified.png").read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        rules,_ = find_rules(image)
        assert max(rule.curvature for rule in rules)<1
        assert page.locator(".image-card[data-key=rectified] .cell-box").count()==8
        assert not page.locator("#debug-toggle").is_checked()
        page.locator("#debug-toggle").check()
        assert page.locator("#activity").is_visible()
        page.locator(".image-card[data-key=rectified]").scroll_into_view_if_needed()
        page.locator("#editor-panel").evaluate("el=>el.open=false")
        page.locator(".receipt-paper").scroll_into_view_if_needed()
        page.screenshot(path=str(out/"wave_receipt_production.png"))
        page.reload(wait_until="networkidle")
        page.wait_for_function("state.result!==null")
        assert page.evaluate("state.result.rectification_method") == "wave_grid"
        assert not page.locator("#debug-toggle").is_checked()
        assert not errors,errors
        (out/"wave_results.json").write_text(json.dumps(dict(job_id=result["job_id"],method=result["rectification_method"],
            product_count=result["product_count"],horizontal_curvature_px=max(rule.curvature for rule in rules),
            calculation_status=result["calculation"]["status"],page_errors=errors),indent=2),encoding="utf-8")
        browser.close()
        print("Production wave receipt passed: active wave_grid, two products, preserved footer, cells, debug, restore.")


if __name__=="__main__":
    main()
