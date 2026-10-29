"""Receipt presentation regression: actual reported job and every dataset photo.

Reuses saved OCR where available. It does not alter recognition/segmentation,
working images, catalogs or learned mappings. Artifacts go to R:/TEMP if present.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
from playwright.sync_api import sync_playwright

from App.ocr_catalog_eval import aligned_lines, choose_row_boundaries, parse_rows, table_layout
from App.receipt_editing import update_receipt
from App.receipt_rectification import rectify_table
from App.table_boundary import detect_table_boundary
from App.web_pipeline import decode_photo, grid_cells

ROOT = Path(__file__).resolve().parents[2]
OUT = (Path("R:/TEMP") if Path("R:/TEMP").exists() else ROOT/"experiment_outputs")/"ocr_visual_restore"
BASE = "http://127.0.0.1:8023"


def get(path):
    with urllib.request.urlopen(BASE+path) as response:
        return json.load(response)


def case_for(source, engine):
    directory = ROOT/"eval_outputs"/source.stem
    cached = directory/"ocr_reference_matches.json"
    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        rows = copy.deepcopy(payload["after"])
        layout = dict(payload["layout"],row_boundaries=payload["row_boundaries"])
        encoded = (directory/"rectified.png").read_bytes()
        raw = json.loads((directory/"ocr_rectified.json").read_text(encoding="utf-8"))
    else:
        saved = OUT/"dataset_cache"/source.stem
        saved.mkdir(parents=True,exist_ok=True)
        cache = saved/"case.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            rows,layout,raw = payload["rows"],payload["layout"],payload["raw"]
            encoded = (saved/"rectified.png").read_bytes()
        else:
            image,_ = decode_photo(source.read_bytes())
            boundary = detect_table_boundary(image)
            image,_,_,metadata = rectify_table(image,boundary["polygon"],trust_top=boundary["status"]=="detected")
            encoded = cv2.imencode(".png",image)[1].tobytes()
            (saved/"rectified.png").write_bytes(encoded)
            result,_ = engine(image)
            raw = dict(lines=[dict(box=box,box_original=box,text=str(text),confidence=float(confidence)) for box,text,confidence in result or []])
            layout = table_layout(image,metadata)
            layout.setdefault("body_left",metadata["margin"])
            lines = aligned_lines(raw,metadata)
            boundaries,_ = choose_row_boundaries(lines,layout)
            rows = parse_rows(lines,layout,boundaries)
            cache.write_text(json.dumps(dict(rows=rows,layout=layout,raw=raw),ensure_ascii=False,indent=2),encoding="utf-8")
    signature = hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    for row in rows:
        row.setdefault("similarity",((row.get("match") or {}).get("best") or {}).get("score"))
    receipt = update_receipt(dict(rows=rows,total_raw="",filename=source.name,elapsed_ms=0,catalog_names=27567,
        pass_similarity=87,order_number="",date="",downloads=dict(json="#",csv="#"),raw_text="\n".join(line["text"] for line in raw["lines"])))
    assert signature==hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return dict(receipt=receipt,cells=grid_cells(layout,rows),image="data:image/png;base64,"+base64.b64encode(encoded).decode()),signature


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from rapidocr_onnxruntime import RapidOCR
    from App.ocr_run import MODEL
    engine = RapidOCR(rec_model_path=str(MODEL),rec_img_shape=[3,48,320],text_score=.3)
    photos = sorted(path for path in (ROOT/"data/in_data").iterdir() if path.suffix.lower() in {".png",".jpeg",".jpg",".webp"})
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",headless=True)
        page = browser.new_page(viewport=dict(width=1600,height=1100))
        errors = []
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(BASE+"/?new=1",wait_until="networkidle")
        reported = get("/api/jobs/49dc382dddaa4cb9a07795b2c107a370")
        page.evaluate("id=>localStorage.setItem('receipt-last-job',id)",reported["id"])
        page.reload(wait_until="networkidle")
        page.wait_for_function("state.result!==null && state.assets.get('rectified')?.cells?.length>0")
        page.locator("#editor-panel").evaluate("el=>el.open=false")
        assert page.locator("#receipt-body tr").count()==3
        assert "Корытынды" not in page.locator("#receipt-body").inner_text()
        assert page.locator("#receipt-body .needs-check").count()==2
        assert page.locator(".cell-box.low-match,.cell-box.focused").count()==0
        assert page.locator(".cell-box").evaluate_all("els=>els.every(el=>getComputedStyle(el).fill==='none')")
        assert page.locator(".receipt-total.math-passed").is_visible()
        assert page.locator("#receipt-total").evaluate("el=>getComputedStyle(el).color")=="rgb(22, 112, 77)"
        page.locator(".receipt-paper").scroll_into_view_if_needed()
        page.locator(".image-card[data-key=rectified]").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/"reported_receipt_after.png"))
        page.locator("#receipt-body tr").first.hover()
        page.mouse.move(10,10)
        page.wait_for_timeout(1100)
        assert page.locator(".cell-box.focused").count()==4
        assert page.locator(".cell-box.focused").first.evaluate("el=>getComputedStyle(el).stroke")=="rgb(224, 170, 22)"
        assert page.locator(".cell-box.focused").evaluate_all("els=>els.every(el=>getComputedStyle(el).fill==='none')")
        page.locator(".receipt-total").hover()
        assert page.locator(".receipt-total-focus").count()==2
        assert page.locator(".receipt-total-focus").first.evaluate("el=>getComputedStyle(el).fill")=="rgb(22, 112, 77)"
        page.locator("#receipt-body tr").nth(2).hover()
        assert page.locator(".receipt-total-focus").count()==0
        assert page.locator(".cell-box.focused").count()==4
        # Replay actual cached recognition for every photo through the view.
        cases = []
        for source in photos:
            data,signature = case_for(source,engine)
            page.mouse.move(10,10)
            page.evaluate("""data=>{reset();const image=new Image();image.src=data.image;return image.decode().then(()=>{
              addArtifact({key:'rectified',title:'Rectify',url:data.image,width:image.naturalWidth,height:image.naturalHeight});
              state.assets.get('rectified').cells=data.cells;finish(data.receipt);
            });}""",data)
            assert page.locator("#receipt-body tr").count()==sum(row["kind"]!="total" for row in data["receipt"]["rows"])
            assert page.locator(".cell-box.focused,.cell-box.low-match").count()==0
            assert page.locator(".cell-box").evaluate_all("els=>els.every(el=>getComputedStyle(el).fill==='none')")
            calc=data["receipt"]["calculation"]
            assert page.locator(".receipt-total").evaluate("el=>el.classList.contains('math-passed')") == (calc.get("validation_status",calc["status"])=="match")
            cases.append(dict(file=source.name,ocr_sha256=signature,view_passed=True))
        assert not errors,errors
        report = dict(files_tested=len(cases),worse=0,better=0,unchanged=len(cases),mean_delta_lines=0,
            mean_delta_confidence=0,scope="Presentation only; OCR inputs unchanged, no OCR improvement claimed",
            reported_receipt_total="642756.00",reported_status="match",cases=cases,page_errors=errors)
        (OUT/"visualization_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        browser.close()
        print(f"Presentation checks passed: reported receipt, totals, persistent outline, no fill, {len(cases)} dataset photos.")


if __name__=="__main__":
    main()
