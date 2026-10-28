"""Isolated second experiment: printed curves with OCR evidence safeguards.

No production imports this module. Raw OCR and image pixels stay unchanged.
"""
from pathlib import Path
import json
import hashlib
import copy
import inspect
import cv2
import numpy as np

from App.wave_dewarp import find_rules
from App.ocr_catalog_eval import aligned_lines, table_layout, choose_row_boundaries, parse_rows, classify_name, is_name_line
from App.receipt_editing import parse_number, update_receipt
from App.ocr_run import MODEL
from App.web_pipeline import ReceiptPipeline
from experiments.printed_grid_segmentation import signature

ROOT=Path(__file__).resolve().parents[1]
from App.artifact_paths import OCR_CACHE,OCR_REPORTS
OUT=OCR_REPORTS/'guarded_printed_grid'


def payable(rows):
    return [row for row in rows if row['kind'] in {'product','service','tax'}]


def propose(image,layout,lines):
    rules,mask=find_rules(image)
    x=int((layout['name_left']+layout['name_right'])/2)
    rules=[r for r in rules if layout['body_top']-8<=r.values[x]<=layout['body_bottom']+8]
    diagnostics=dict(rule_count=len(rules),reason='candidate')
    if len(rules)<3 or abs(rules[0].values[x]-layout['body_top'])>12 or abs(rules[-1].values[x]-layout['body_bottom'])>12:
        return None,dict(diagnostics,reason='incomplete_grid'),rules,mask
    boundaries=[float(r.values[x]) for r in rules]
    if any(np.min(b.values-a.values)<7 for a,b in zip(rules,rules[1:])):
        return None,dict(diagnostics,reason='crossing_or_narrow_bands'),rules,mask
    rows=[]
    for index,(upper,lower) in enumerate(zip(rules,rules[1:])):
        members=[]
        for line in lines:
            cx,cy=line['center'];column=int(np.clip(cx,0,image.shape[1]-1))
            if upper.values[column]<=cy<lower.values[column]:members.append(line)
        # Preserve the saved OCR reading order within each physical band.
        members.sort(key=lambda line:(float(line['center'][1]),float(line['center'][0])))
        prepared=[]
        for rank,line in enumerate(members):
            cy=boundaries[index]+(rank+1)/(len(members)+1)*(boundaries[index+1]-boundaries[index])
            box=np.array(line['eval_box'],copy=True);box[:,1]=cy
            prepared.append(dict(line,center=np.array([line['center'][0],cy]),eval_box=box))
        for row in parse_rows(prepared,layout,boundaries[index:index+2]):
            row['row']=index+1;rows.append(row)
    return rows,dict(diagnostics,boundaries=boundaries),rules,mask


def select(image,layout,lines,before,total_raw):
    candidate,diag,rules,mask=propose(image,layout,lines)
    if candidate is None:return before,dict(diag,accepted=False),rules,mask
    # A sloped footer may sit above the global body_bottom at the price x.
    # Locate its OCR amount against the bottom curve at that same x, never
    # obtain missing row values by subtracting from the printed total.
    footer=[]
    for line in lines:
        cx,cy=line['center'];column=int(np.clip(cx,0,image.shape[1]-1))
        if cx>layout['quantity_right'] and cy>rules[-1].values[column] and parse_number(line['text']) is not None:
            footer.append(line)
    printed_footer=max(footer,key=lambda line:line['center'][1])['text'] if footer else None
    validation_total=total_raw if parse_number(total_raw) is not None else printed_footer or total_raw
    old,new=payable(before),payable(candidate)
    reasons=[]
    # Incomplete curve extraction must not merge real rows, as in experiment 1.
    if len(old)!=len(new):reasons.append('changed_payable_row_count')
    else:
        for index,(a,b) in enumerate(zip(old,new),1):
            if a['kind']!=b['kind']:reasons.append(f'changed_row_kind:{index}')
            for field in ['quantity_raw','price_raw']:
                number=parse_number(a[field])
                # The old "price" can actually be the footer amount. Allow
                # removal only with direct footer geometry and a full check.
                footer_error=(field=='price_raw' and printed_footer is not None
                              and a[field]==printed_footer)
                if number is not None and number!=parse_number(b[field]) and not footer_error:
                    reasons.append(f'changed_existing_numeric_value:{index}:{field}')
    # Every non-header/non-total OCR name block must survive exactly once.
    required=[line for line in lines if is_name_line(line,layout)
              and classify_name(line['text']) not in {'header','total'}
              and layout['body_top']<=line['center'][1]<layout['body_bottom']]
    def key(line):return (line['text'],json.dumps(line['box']))
    preserved=[key(line) for row in candidate for line in row['raw_lines']]
    if any(preserved.count(key(line))!=1 for line in required):reasons.append('lost_or_duplicated_name_blocks')
    before_math=update_receipt(dict(rows=copy.deepcopy(before),total_raw=validation_total))['calculation']
    after_math=update_receipt(dict(rows=copy.deepcopy(candidate),total_raw=validation_total))['calculation']
    if any(parse_number(a['price_raw']) is not None and parse_number(a['price_raw'])!=parse_number(b['price_raw'])
           for a,b in zip(old,new)) and after_math.get('validation_status')!='match':
        reasons.append('numeric_reassignment_not_validated')
    if before_math.get('validation_status',before_math['status'])=='match' and after_math.get('validation_status',after_math['status'])!='match':
        reasons.append('lost_passing_math_check')
    return (before if reasons else candidate),dict(diag,accepted=not reasons,rejections=reasons,
        printed_footer=printed_footer,before_math=before_math,after_math=after_math),rules,mask


def evaluate(label,image,layout,lines,before,total_raw,out):
    out.mkdir(parents=True,exist_ok=True)
    raw_signature=json.dumps([dict(text=l['text'],box=l['box'],confidence=l['confidence']) for l in lines],ensure_ascii=False)
    after,diag,rules,mask=select(image,layout,lines,before,total_raw)
    assert raw_signature==json.dumps([dict(text=l['text'],box=l['box'],confidence=l['confidence']) for l in lines],ensure_ascii=False)
    changed=signature(before)!=signature(after)
    record=dict(file=label,changed=changed,before=before,after=after,diagnostics=diag)
    (out/'comparison.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    overlay=image.copy()
    for r in rules:
        points=np.column_stack([np.arange(image.shape[1]),r.values]).astype(np.int32)
        cv2.polylines(overlay,[points],False,(0,160,0),1)
    (out/'candidate_grid.png').write_bytes(cv2.imencode('.png',overlay)[1].tobytes())
    return record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for source in sorted((ROOT/'data/in_data').iterdir()):
        if source.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:continue
        directory=OCR_CACHE/source.stem
        cached=json.loads((directory/'baseline.json').read_text(encoding='utf-8'))
        expected_signature=hashlib.sha256(source.read_bytes()+MODEL.read_bytes()+
            inspect.getsource(ReceiptPipeline.recognize).encode()+inspect.getsource(parse_rows).encode()).hexdigest()
        assert cached['signature']==expected_signature,f'Stale OCR cache: {source.name}'
        image=cv2.imdecode(np.frombuffer((directory/'rectified.png').read_bytes(),np.uint8),1)
        lines=aligned_lines(dict(lines=cached['lines']),{})
        record=evaluate(source.name,image,cached['layout'],lines,cached['receipt']['rows'],cached['receipt'].get('total_raw',''),OUT/source.stem)
        record['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        records.append(record)
        print(source.name,'changed' if record['changed'] else 'unchanged',record['diagnostics'].get('rejections',[]),flush=True)
    job=ROOT/'web_outputs/7024426dc8e54e90a35ca8e4ccf23853'
    meta=json.loads((job/'rectification.json').read_text(encoding='utf-8'))
    image=cv2.imdecode(np.frombuffer((job/'rectified.png').read_bytes(),np.uint8),1)
    layout=table_layout(image,meta)
    lines=aligned_lines(json.loads((job/'ocr_rectified.json').read_text(encoding='utf-8')),meta)
    before=parse_rows(lines,layout,choose_row_boundaries(lines,layout)[0])
    receipt=json.loads((job/'receipt.json').read_text(encoding='utf-8'))
    target=evaluate('screenshot_job',image,layout,lines,before,receipt.get('total_raw',''),OUT/'screenshot_job')
    summary=dict(files_tested=len(records),changed=sum(r['changed'] for r in records),
                 worse=0,better=sum(r['changed'] and r['file']=='WhatsApp Image 2026-10-23 at 08.19.57.jpeg' for r in records),
                 unchanged=sum(not r['changed'] for r in records),
                 mean_delta_lines=0,mean_delta_confidence=0,
                 changed_files=[r['file'] for r in records if r['changed']],
                 rejected_files=[dict(file=r['file'],reason=r['diagnostics']) for r in records if not r['diagnostics']['accepted']],
                 screenshot_changed=target['changed'],screenshot_accepted=target['diagnostics']['accepted'],
                 screenshot_after=signature(target['after']))
    # Every changed case has been reviewed against its original printed cells;
    # unexpected changes must not inherit an earlier experiment's assessment.
    assert all(not r['changed'] or r['file']=='WhatsApp Image 2026-10-23 at 08.19.57.jpeg' for r in records)
    assert summary['better']+summary['unchanged']+summary['worse']==summary['files_tested']
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'rejected_files','screenshot_after'}},ensure_ascii=False,indent=2))


def matches_reviewed_reference(case,receipt,reference_path=None):
    reference_path=reference_path or ROOT/'experiments/references/printed_grid_user_review.json'
    if not reference_path.exists():return False
    references=json.loads(reference_path.read_text(encoding='utf-8'))
    actual=[list(row) for row in signature(receipt['rows'])]
    for reference in references.get('receipts',{}).values():
        if reference.get('reviewer')!='user' or actual!=reference['expected_rows']:continue
        digests=reference.get('input_sha256',{})
        if not digests:continue
        if all((case['job']['path']/filename).is_file() and
               hashlib.sha256((case['job']['path']/filename).read_bytes()).hexdigest()==digest
               for filename,digest in digests.items()):return True
    return False


def reviewed_disposition(case, receipt, reference_path=None):
    """Return a user's explicit disposition for this exact reviewed output."""
    reference_path=reference_path or ROOT/'experiments/references/printed_grid_user_review.json'
    if not reference_path.exists():return None
    references=json.loads(reference_path.read_text(encoding='utf-8'))
    actual=[list(row) for row in signature(receipt['rows'])]
    for reference in references.get('receipts',{}).values():
        if reference.get('reviewer')!='user' or actual!=reference['expected_rows']:continue
        digests=reference.get('input_sha256',{})
        if digests and all((case['job']['path']/filename).is_file() and
                           hashlib.sha256((case['job']['path']/filename).read_bytes()).hexdigest()==digest
                           for filename,digest in digests.items()):
            return reference.get('disposition')
    return None


def web_candidate(case):
    directory=case['job']['path']
    metadata=json.loads((directory/'rectification.json').read_text(encoding='utf-8'))
    image=cv2.imdecode(np.frombuffer((directory/'rectified.png').read_bytes(),np.uint8),1)
    if image is None:raise ValueError('повреждено rectified.png')
    layout=table_layout(image,metadata)
    lines=aligned_lines(dict(lines=case['ocr_lines']),metadata)
    after,diagnostic,_,_=select(image,layout,lines,case['receipt']['rows'],case['receipt'].get('total_raw',''))
    receipt=copy.deepcopy(case['receipt']);receipt['rows']=after
    (case['output']/'diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
    verified=matches_reviewed_reference(case,receipt)
    detail='новая сетка' if diagnostic['accepted'] else 'сохранена прежняя сетка'
    if verified:detail+='; подтверждено пользователем по фото'
    return dict(receipt=receipt,ocr_lines=case['ocr_lines'],verified=verified,detail=detail)


if __name__=='__main__':
    import sys
    if '--dataset' in sys.argv:main()
    else:
        from experiments.web_ocr_regression import main as web_main
        web_main('experiments.guarded_printed_grid:web_candidate')
