"""Production upload: recover missing quantity, validate VAT, verify red/green UI."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=(Path("R:/TEMP") if Path("R:/TEMP").exists() else ROOT/"experiment_outputs")/"quantity_cell_recovery"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",headless=True)
        page=browser.new_page(viewport=dict(width=1600,height=1100))
        errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto("http://127.0.0.1:8023/?new=1",wait_until="networkidle")
        page.locator("#file-input").set_input_files(ROOT/"data/in_data/WhatsApp Image 2026-10-25 at 13.16.05.jpeg")
        page.wait_for_function("state.result!==null",timeout=120000)
        result=page.evaluate("state.result")
        products=[row for row in result["rows"] if row["kind"]=="product"]
        tax=next(row for row in result["rows"] if row["kind"]=="tax")
        assert products[1]["quantity"]=="1",result
        assert products[1]["quantity_raw"]==""
        assert products[1]["quantity_source"]=="cell_ocr"
        assert products[1]["price"]=="216000 T"
        assert result["calculation"]["validation_status"]=="match"
        assert result["calculation"]["vat"]["expected_tax"]=="88656.00"
        assert result["calculation"]["computed_total"]=="642756.00"
        assert page.locator(".receipt-total.math-passed").is_visible()
        assert page.locator("#receipt-body .vat-passed").count()==1
        assert page.locator("#receipt-body tr").count()==3
        page.locator("#editor-panel").evaluate("el=>el.open=false")
        page.locator(".receipt-paper").scroll_into_view_if_needed()
        page.locator(".image-card[data-key=rectified]").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/"production_green.png"))
        def edit(payload):
            revision=page.evaluate("state.result.revision")
            page.evaluate("payload=>sendEdit(payload)",payload)
            page.wait_for_function("r=>state.result.revision>r&&!state.saving",arg=revision)
        # A wrong price must turn the grand total red.
        edit(dict(row=products[1]["row"],price="216001"))
        assert page.locator(".receipt-total.math-failed").is_visible()
        assert page.locator("#receipt-total").evaluate("el=>getComputedStyle(el).color")=="rgb(180, 35, 50)"
        page.locator(".receipt-total").hover()
        assert page.locator(".receipt-total-focus").first.evaluate("el=>getComputedStyle(el).fill")=="rgb(180, 35, 50)"
        edit(dict(row=products[1]["row"],price="216000"))
        # Even equal grand totals cannot conceal an incorrect VAT amount.
        edit(dict(row=tax["row"],quantity="88655",expected_total="642755"))
        assert page.evaluate("state.result.calculation.status")=="match"
        assert page.evaluate("state.result.calculation.validation_status")=="vat_mismatch"
        assert page.locator("#receipt-body .vat-failed").count()==1
        assert page.locator(".receipt-total.math-failed").is_visible()
        page.screenshot(path=str(OUT/"production_vat_red.png"))
        edit(dict(row=tax["row"],quantity="88656",expected_total="642756"))
        assert page.locator(".receipt-total.math-passed").is_visible()
        assert page.locator("#receipt-body .vat-passed").count()==1
        page.reload(wait_until="networkidle")
        page.wait_for_function("state.result!==null")
        assert page.evaluate("state.result.calculation.validation_status")=="match"
        assert page.locator("#receipt-body tr").count()==3
        assert not errors,errors
        (OUT/"production_validation.json").write_text(json.dumps(dict(job_id=result["job_id"],quantity="1",
            raw_quantity_preserved=True,vat="88656.00",total="642756.00",red_and_green=True,page_errors=errors),indent=2),encoding="utf-8")
        browser.close()
        print("Production amount checks passed: quantity recovery, VAT, total, red/green, hover, restore.")


if __name__=="__main__":
    main()
