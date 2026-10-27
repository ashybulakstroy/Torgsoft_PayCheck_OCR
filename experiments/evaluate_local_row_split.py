"""Rebuild missing production OCR baselines, then test row splitting fail-fast."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import inspect
import json
import shutil
import copy
import argparse
import cv2
import numpy as np

from App.ocr_run import MODEL
from App.web_pipeline import ReceiptPipeline
from App.web_server import Job
from App.ocr_catalog_eval import parse_rows
from experiments.web_ocr_regression import raw_receipt
from experiments.local_row_split import repair
from experiments.ocr_failure_report import write_failure_report

ROOT=Path(__file__).resolve().parents[1]
from App.artifact_paths import OCR_CACHE,OCR_REPORTS
CACHE=OCR_CACHE


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def fingerprint(source):
    modules=['web_pipeline','ocr_catalog_eval','receipt_rectification','contour_unwarp',
             'wave_dewarp','table_boundary','quantity_cells','numeric_cell_split','receipt_editing','receipt_math']
    inputs=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                modules={name:hashlib.sha256((ROOT/'App'/f'{name}.py').read_bytes()).hexdigest() for name in modules},
                rapidocr_version=importlib.metadata.version('rapidocr_onnxruntime'),opencv_version=cv2.__version__)
    digest=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    return digest,inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-baseline',action='store_true',help='compare with saved pre-change results; verify source/model hashes')
    args=parser.parse_args()
    out=OCR_REPORTS/'local_row_split'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    pipeline=ReceiptPipeline(split_product_rows=False)
    sources=sorted(p for p in (ROOT/'data/in_data').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    summary=dict(status='running',scope='segmentation',requested=len(sources),available=len(sources),
                 files_tested=0,files_compared=0,worse=0,better=0,unchanged=0,errors=0,
                 mean_delta_lines=0,mean_delta_confidence=0,records=[])
    for index,source in enumerate(sources,1):
        directory=CACHE/source.stem;directory.mkdir(parents=True,exist_ok=True)
        case_dir=out/source.stem;case_dir.mkdir()
        job_info=dict(id=source.stem,path=directory,filename=source.name)
        print(f'[{index}/{len(sources)}] START {source.name}',flush=True)
        record=dict(id=source.stem,file=source.name,filename=source.name,status='running')
        try:
            digest,inputs=fingerprint(source)
            data=json.loads((directory/'baseline.json').read_text(encoding='utf-8')) if (directory/'baseline.json').exists() else {}
            cache_ok=data.get('production_fingerprint')==digest and all((directory/name).exists() for name in ['rectified.png','original.png','layout.json','receipt.json'])
            if args.frozen_baseline:
                previous=data.get('provenance',{})
                if not data or previous.get('source_sha256')!=inputs['source_sha256'] or previous.get('model_sha256')!=inputs['model_sha256']:
                    raise ValueError('Frozen baseline is missing or source/model hashes differ; not replacing it with a new result.')
                if data.get('baseline_kind')!='production_pipeline_without_local_row_split':
                    raise ValueError('Frozen baseline does not identify an independent pre-change pipeline.')
                cache_ok=all((directory/name).exists() for name in ['rectified.png','original.png','layout.json','receipt.json'])
                if not cache_ok:raise ValueError('Frozen baseline artifacts are incomplete.')
            if not cache_ok:
                print(f'[{index}/{len(sources)}] BASELINE: полный рабочий OCR без экспериментального разделения строк',flush=True)
                (directory/'upload.bin').write_bytes(source.read_bytes())
                job=Job(hashlib.sha256(source.name.encode()).hexdigest()[:32],source.name,directory)
                baseline=pipeline.process(job)
                lines=json.loads((directory/'ocr_rectified.json').read_text(encoding='utf-8'))['lines']
                layout=json.loads((directory/'layout.json').read_text(encoding='utf-8'))['layout']
                legacy=hashlib.sha256(source.read_bytes()+MODEL.read_bytes()+inspect.getsource(ReceiptPipeline.recognize).encode()+inspect.getsource(parse_rows).encode()).hexdigest()
                data=dict(signature=legacy,production_fingerprint=digest,provenance=inputs,
                          generated_at=datetime.now(timezone.utc).isoformat(),baseline_kind='production_pipeline_without_local_row_split',
                          receipt=baseline,lines=lines,layout=layout,method=baseline['rectification_method'])
                dump(directory/'baseline.json',data)
            before=raw_receipt(data['receipt'])
            dump(case_dir/'baseline.json',dict(receipt=before,ocr_lines=data['lines']))
            image=cv2.imdecode(np.frombuffer((directory/'rectified.png').read_bytes(),np.uint8),1)
            if image is None:raise ValueError('rectified.png cannot be decoded')
            snapshot=copy.deepcopy(before)
            after,diagnostics=repair(before,data['layout'],data['lines'],image)
            assert before==snapshot,'baseline mutated'
            dump(case_dir/'candidate.json',dict(receipt=after,ocr_lines=data['lines']))
            dump(case_dir/'row_split.json',diagnostics)
            changed=after!=before
            summary['files_compared']+=1
            record.update(status='worse' if changed else 'unchanged',diagnostics=diagnostics,
                          delta_lines=0,delta_confidence=0,cache_rebuilt=not cache_ok,
                          frozen_baseline=args.frozen_baseline,
                          reasons=['Изменение группировки требует независимого подтверждения по фото.'] if changed else [])
        except (Exception,KeyboardInterrupt) as error:
            record.update(status='error',reasons=[f'{type(error).__name__}: {error}'])
        summary['records'].append(record);summary['files_tested']=index
        for status in ['worse','better','unchanged']:summary[status]=sum(r['status']==status for r in summary['records'])
        summary['errors']=sum(r['status']=='error' for r in summary['records'])
        failed=record['status'] in {'worse','error'}
        summary['status']='failed' if failed else 'running'
        dump(out/'summary.json',summary)
        print(f'[{index}/{len(sources)}] {"FAIL" if failed else "PASS"} {source.name} '
              f'Δстрок=0 Δуверенности=0 '+('; '.join(record.get('reasons',[])) or 'без изменения группировки'),flush=True)
        if failed:
            path=write_failure_report(summary,out,job_info,open_browser=True)
            dump(out/'summary.json',summary)
            print(f'FAIL: прогон остановлен. Визуальный отчёт: {path}',flush=True)
            raise SystemExit(1)
    summary['status']='passed';dump(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False),flush=True)
    print(f'Отчёт: {out/"summary.json"}',flush=True)


if __name__=='__main__':main()
