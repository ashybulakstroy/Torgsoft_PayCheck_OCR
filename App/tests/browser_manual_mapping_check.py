"""Browser + HTTP + JSON learning check, isolated from the real knowledge base."""
import json
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

from App.manual_mappings import ManualMappings
from App.ocr_catalog_eval import CatalogMatcher, normalize_name, review_reason
from App.receipt_editing import save_receipt, update_receipt
from App.web_server import create_server


def main():
    with tempfile.TemporaryDirectory(prefix="receipt-knowledge-check-") as temporary:
        directory = Path(temporary)
        knowledge = ManualMappings(directory/"manual.json")
        target = dict(name="Пила CROWN CT15074", normalized=normalize_name("Пила CROWN CT15074"), sources=[dict(article="123")])
        other = dict(name="Дверь Лион 50951", normalized=normalize_name("Дверь Лион 50951"), sources=[dict(article="456")])
        entries = [target,other]
        matcher = CatalogMatcher(entries,knowledge=knowledge)
        raw = "Непонятный OCR жжж 987654"

        def process(job):
            observed = raw.upper().replace(" ", "") if job.filename == "repeat.png" else raw
            match = matcher.match(observed)
            row = dict(row=1,item_number_raw="1",kind="product",name_raw=observed,quantity_raw="1",price_raw="10Т",
                       match=match,stage="matched",similarity=match["best"]["score"],ocr_name_confidence_pct=80)
            row["review_reason"] = review_reason(row)
            result = update_receipt(dict(rows=[row],total_raw="10",job_id=job.id,filename=job.filename,
                elapsed_ms=1,catalog_names=len(entries),pass_similarity=87,warnings=[],raw_text=raw,
                order_number="",date="",downloads={"json":f"/api/jobs/{job.id}/files/receipt.json","csv":f"/api/jobs/{job.id}/files/receipt.csv"}))
            save_receipt(job.directory,result)
            return result

        server = create_server("127.0.0.1",0,directory/"jobs",process)
        pipeline = server.store.pipeline
        pipeline.knowledge = knowledge
        pipeline.matcher = matcher
        pipeline.catalog_index = {entry["normalized"]:entry for entry in entries}
        pipeline.ensure_catalog = lambda:None
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",headless=True)
                page = browser.new_page(viewport=dict(width=1440,height=1000))
                errors = []
                page.on("pageerror",lambda error:errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}",wait_until="networkidle")
                def upload(name="test.png"):
                    page.locator("#file-input").set_input_files(dict(name=name,mimeType="image/png",buffer=b"fixture"))
                    page.wait_for_function("state.result!==null")
                upload()
                assert page.locator("#receipt-body .needs-check").count() == 1
                page.locator(".catalog-option").filter(has_text=target["name"]).click()
                assert page.evaluate("state.catalogChosen")
                page.locator("#save-row").click()
                page.wait_for_function("state.result.revision===1 && !state.saving")
                assert knowledge.lookup(raw)["catalog_key"] == target["normalized"]
                assert page.locator("#receipt-body .needs-check").count() == 0
                upload("repeat.png")
                assert page.evaluate("state.result.rows[0].match.source") == "manual_mapping"
                assert page.evaluate("state.result.rows[0].name") == target["name"]
                assert page.evaluate("state.result.rows[0].name_raw") == raw.upper().replace(" ", "")
                assert page.evaluate("state.result.can_finalize")
                assert page.locator("#receipt-body .needs-check").count() == 0
                assert not errors,errors
                browser.close()
            print("Manual mapping browser check passed: choose, persist, repeat, reuse; real database untouched.")
        finally:
            server.shutdown()
            server.server_close()
            server.store.pool.shutdown(wait=True)
            thread.join(timeout=3)


if __name__ == "__main__":
    main()
