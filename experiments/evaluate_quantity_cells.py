"""Full real-photo evaluation of isolated missing-quantity cell OCR recovery."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from App.ocr_catalog_eval import aligned_lines,choose_row_boundaries,parse_rows,table_layout
from App.ocr_run import MODEL
from App.receipt_editing import parse_number,update_receipt
from App.receipt_rectification import rectify_table
from App.table_boundary import detect_table_boundary
from App.web_pipeline import ReceiptPipeline,decode_photo
from experiments.quantity_cell_recovery import recover_quantities

ROOT = Path(__file__).resolve().parents[1]
from App.artifact_paths import OCR_CACHE
OUT = OCR_CACHE


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")


def baseline(source,pipeline):
    directory = OUT/source.stem
    directory.mkdir(parents=True,exist_ok=True)
    signature = hashlib.sha256(source.read_bytes()+MODEL.read_bytes()+
        inspect.getsource(ReceiptPipeline.recognize).encode()+inspect.getsource(parse_rows).encode()).hexdigest()
    saved = directory/"baseline.json"
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        if data.get("signature")==signature:
            image = cv2.imdecode(np.frombuffer((directory/"rectified.png").read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            return data,image,directory
    image,_ = decode_photo(source.read_bytes())
    boundary = detect_table_boundary(image)
    image,mx,my,metadata = rectify_table(image,boundary["polygon"],trust_top=boundary["status"]=="detected")
    (directory/"rectified.png").write_bytes(cv2.imencode(".png",image)[1].tobytes())
    raw = pipeline.recognize(directory/"rectified.png",SimpleNamespace(publish=lambda *args,**kwargs:None),"rectified",0)
    lines = [dict(box=box,box_original=box,text=str(text),confidence=float(confidence)) for box,text,confidence in raw]
    layout = table_layout(image,metadata)
    layout.setdefault("body_left",metadata["margin"])
    aligned = aligned_lines(dict(lines=lines),metadata)
    boundaries,_ = choose_row_boundaries(aligned,layout)
    rows = parse_rows(aligned,layout,boundaries)
    candidates = [line for line in aligned if line["center"][1]>=layout["body_bottom"]-5 and
                  line["center"][0]>layout["name_right"] and __import__('re').search(r"\d[.,]\d{2}",line["text"])]
    total = max(candidates,key=lambda line:line["center"][1])["text"] if candidates else ""
    receipt = update_receipt(dict(rows=rows,total_raw=total,filename=source.name))
    data = dict(signature=signature,receipt=receipt,layout=layout,lines=lines,method=metadata["method"])
    dump(saved,data)
    return data,image,directory


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pipeline = ReceiptPipeline()
    pipeline.engine = RapidOCR(rec_model_path=str(MODEL),rec_img_shape=[3,48,320],text_score=.3)
    files = sorted(path for path in (ROOT/"data/in_data").iterdir() if path.suffix.lower() in {".png",".jpg",".jpeg",".webp"})
    records = []
    for source in files:
        data,image,directory = baseline(source,pipeline)
        before = data["receipt"]
        after,diagnostics,crops = recover_quantities(before,image,data["layout"],lambda imgs:pipeline.engine.text_recognizer(imgs)[0])
        # Existing values and raw OCR evidence must stay exactly unchanged.
        harms = []
        changes = []
        for old,new in zip(before["rows"],after["rows"]):
            for key in ("name_raw","quantity_raw","price_raw","raw_lines"):
                if old.get(key)!=new.get(key):
                    harms.append(f"raw evidence changed: row {old['row']} {key}")
            if old["quantity"]!=new["quantity"]:
                if parse_number(old["quantity"]) is not None:
                    harms.append(f"valid quantity overwritten: row {old['row']}")
                changes.append(dict(row=new["row"],name=new["name_raw"],before=old["quantity"],after=new["quantity"],price=new["price"]))
            if old["price"]!=new["price"]:
                harms.append(f"price changed: row {old['row']}")
        if before["calculation"]["status"]=="match" and after["calculation"]["status"]!="match":
            harms.append("previous matching total no longer matches")
        status = "worse" if harms else "better" if changes else "unchanged"
        for crop in crops:
            for view in ("crop","tight"):
                (directory/f"row_{crop['row']}_{view}.png").write_bytes(cv2.imencode(".png",crop[view])[1].tobytes())
        record = dict(file=source.name,status=status,changes=changes,harms=harms,
                      before_total=before["calculation"],after_total=after["calculation"],recovery=diagnostics)
        records.append(record)
        dump(directory/"evaluation.json",record)
        dump(directory/"after.json",after)
        print(source.name,status,changes,flush=True)
        summary = dict(files_tested=len(records),worse=sum(r["status"]=="worse" for r in records),
            better=sum(r["status"]=="better" for r in records),unchanged=sum(r["status"]=="unchanged" for r in records),
            mean_delta_lines=0,mean_delta_confidence=0,
            metric_scope="Raw OCR lines/confidence are preserved; better/worse evaluate numeric fields and printed-total consistency",
            records=records)
        dump(OUT/"summary.json",summary)
        if status=="worse":
            print('FAIL: остановка на первом ухудшении',flush=True)
            raise SystemExit(1)
    summary["recommendation"] = "eligible_after_image_review" if summary["worse"]==0 and summary["better"]>0 else "do_not_integrate"
    dump(OUT/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="records"},ensure_ascii=False))


def web_candidate(case):
    directory=case['job']['path']
    metadata=json.loads((directory/'rectification.json').read_text(encoding='utf-8'))
    image=cv2.imdecode(np.frombuffer((directory/'rectified.png').read_bytes(),np.uint8),1)
    layout=table_layout(image,metadata)
    if not hasattr(web_candidate,'engine'):
        web_candidate.engine=RapidOCR(rec_model_path=str(MODEL),rec_img_shape=[3,48,320],text_score=.3)
    before=update_receipt(copy.deepcopy(case['receipt']))
    after,diagnostics,_=recover_quantities(before,image,layout,lambda imgs:web_candidate.engine.text_recognizer(imgs)[0])
    # Numerical recovery has independent OCR crops and a passing printed-total
    # check; persist the evidence for reviewing every proposed change.
    dump(case['output']/'quantity_cells.json',diagnostics)
    return dict(receipt=after,ocr_lines=case['ocr_lines'],
                verified=after['calculation'].get('validation_status')=='match',detail='OCR отдельных ячеек количества')


if __name__=="__main__":
    import sys
    if '--dataset' in sys.argv:main()
    else:
        from experiments.web_ocr_regression import main as web_main
        web_main('experiments.evaluate_quantity_cells:web_candidate')
