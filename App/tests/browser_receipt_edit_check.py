"""Real browser checks of receipt review, catalog selection and exact arithmetic."""
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"web_outputs/browser_check"
EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"


def main():
    jobs = json.loads((OUT/"results.json").read_text())
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EDGE, headless=True)
        page = browser.new_page(viewport=dict(width=1600,height=1100))
        errors = []
        page.on("pageerror", lambda error:errors.append(str(error)))
        page.goto("http://127.0.0.1:8023", wait_until="networkidle")

        def restore(job):
            page.evaluate("id=>localStorage.setItem('receipt-last-job',id)", job)
            page.reload(wait_until="networkidle")
            page.wait_for_function("state.result!==null")

        def save():
            revision = page.evaluate("state.result.revision")
            page.locator("#save-row").click()
            page.wait_for_function("r=>state.result.revision>r && !state.saving", arg=revision)

        restore(jobs["first_job"])
        assert page.locator(".image-card").evaluate_all("cards=>cards.map(c=>c.dataset.key)") == ["original","header","crop","rectified"]
        assert page.locator(".item-name").first.evaluate("el=>getComputedStyle(el).fontSize") == "18px"
        assert page.locator("#math-panel").is_hidden()
        assert "Артикул " not in page.locator("#receipt-body").inner_text()
        assert "Сходство " not in page.locator("#receipt-body").inner_text()
        assert page.locator(".receipt-total").evaluate("el=>el.classList.contains('math-passed')")
        assert page.locator("#receipt-body tr[data-row='1'] td:nth-child(3)").evaluate("el=>el.classList.contains('invalid-field')") is False
        raw = ROOT/"web_outputs"/jobs["first_job"]/"ocr_rectified.json"
        before = hashlib.sha256(raw.read_bytes()).hexdigest()
        # Actual catalog API and UI selection, preserving raw OCR.
        page.evaluate("editRow(state.result.rows.find(r=>r.kind==='product').row)")
        page.locator("#catalog-search").fill("CT15074")
        page.wait_for_timeout(700)
        page.locator(".catalog-option").filter(has_text="CT15074").first.click()
        page.locator("#edit-price").fill("30000")
        save()
        assert page.evaluate("state.result.calculation.status") == "mismatch"
        assert not page.locator(".receipt-total").evaluate("el=>el.classList.contains('math-passed')")
        assert page.locator("#finalize-receipt").is_disabled()
        page.locator("#edit-price").fill("31600")
        save()
        assert page.evaluate("state.result.calculation.computed_total") == "36656.00"
        assert page.locator("#finalize-receipt").is_enabled()
        page.locator("#finalize-receipt").click()
        page.wait_for_function("state.result.finalized===true && !state.saving")
        page.screenshot(path=str(OUT/"receipt_verified.png"))
        restore(jobs["first_job"])
        assert page.evaluate("state.result.finalized")
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == before

        restore(jobs["second_job"])
        assert page.evaluate("state.result.calculation.status") == "match"
        assert page.evaluate("state.result.rows.filter(r=>r.kind==='product')[1].quantity") == "1"
        assert page.locator("#receipt-body tr[data-row='1'] td:nth-child(3)").evaluate("el=>el.classList.contains('invalid-field')")
        assert page.locator(".receipt-total").evaluate("el=>el.classList.contains('math-error')")
        page.evaluate("editRow(state.result.rows.filter(r=>r.kind==='product')[1].row)")
        page.locator("#edit-quantity").fill("1")
        save()
        assert page.evaluate("state.result.calculation.computed_total") == "642756.00"
        assert page.evaluate("state.result.calculation.status") == "match"
        for row_id in page.evaluate("state.result.rows.filter(r=>r.needs_review).map(r=>r.row)"):
            page.evaluate("id=>editRow(id)", row_id)
            save()
        assert page.evaluate("state.result.review_count") == 0
        assert page.locator("#receipt-body .needs-check").count() == 0
        assert page.locator("#finalize-receipt").is_enabled()
        # Inclusive threshold controls yellow, independent of ambiguity review reason.
        classes = page.evaluate("""()=>{const row=structuredClone(state.result.rows[0]);row.needs_review=true;row.name_confirmed=false;row.similarity=state.result.pass_similarity;const equal=makeReceiptRow(row).classList.contains('needs-check');row.similarity-=.01;return {equal,below:makeReceiptRow(row).classList.contains('needs-check')};}""")
        assert classes == dict(equal=False, below=True)
        page.locator("#editor-panel").evaluate("el=>el.open=false")
        page.locator(".receipt-paper").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/"receipt_corrected.png"))
        page.set_viewport_size(dict(width=390,height=844))
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        assert page.locator(".receipt-table-wrap").evaluate("el=>el.scrollWidth>el.clientWidth")
        page.screenshot(path=str(OUT/"receipt_edit_mobile.png"))
        assert not errors, errors
        (OUT/"editing_results.json").write_text(json.dumps(dict(catalog_selection=True,original_ocr_unchanged=True,
            missing_quantity_corrected=True,mismatch_corrected=True,threshold_inclusive=True,finalize_restore=True,
            first_total="36656.00",second_total="642756.00",page_errors=errors),indent=2),encoding="utf-8")
        browser.close()
        print("Receipt browser checks passed: catalog, edits, totals, finalization, threshold, raw OCR, mobile.")


if __name__ == "__main__":
    main()
