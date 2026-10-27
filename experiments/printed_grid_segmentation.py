"""Read-only experiment: compare printed-rule curves with legacy row segmentation.

Uses saved real OCR from every dataset photo; never changes production results.
Run with the project .venv: python -m experiments.printed_grid_segmentation
"""
from pathlib import Path
import json
import cv2
import numpy as np
from App.wave_dewarp import find_rules
from App.ocr_catalog_eval import aligned_lines, table_layout, choose_row_boundaries, parse_rows

ROOT = Path(__file__).resolve().parents[1]
from App.artifact_paths import OCR_CACHE,OCR_REPORTS
OUT = OCR_REPORTS/'printed_grid_segmentation'


def candidate(image, layout, lines):
    rules, mask = find_rules(image)
    x = int((layout['name_left']+layout['name_right'])/2)
    boundaries = [float(rule.values[x]) for rule in rules
                  if layout['body_top']-8 <= rule.values[x] <= layout['body_bottom']+8]
    if len(boundaries)<4:
        return None, dict(reason='too_few_rules'), mask
    if abs(boundaries[0]-layout['body_top'])>12 or abs(boundaries[-1]-layout['body_bottom'])>12:
        return None, dict(reason='incomplete_outer_grid',boundaries=boundaries), mask
    updated=dict(layout,row_boundaries=boundaries,grid_confirmed=True)
    # Classify every OCR block against the curves at its own x coordinate.
    # Parse each confirmed band separately so the price-anchor repair heuristic
    # cannot move a wrapped name into its neighbour after physical assignment.
    selected=[rule for rule in rules if layout['body_top']-8 <= rule.values[x] <= layout['body_bottom']+8]
    rows=[]
    for index,(upper,lower) in enumerate(zip(selected,selected[1:])):
        members=[]
        for line in lines:
            cx,cy=line['center'];column=int(np.clip(cx,0,image.shape[1]-1))
            if upper.values[column]<=cy<lower.values[column]:
                # Only the parser's working coordinates change; raw boxes and
                # raw text remain identical to the saved OCR evidence.
                fraction=(cy-upper.values[column])/(lower.values[column]-upper.values[column])
                mapped_y=boundaries[index]+fraction*(boundaries[index+1]-boundaries[index])
                item=dict(line,center=np.array([cx,mapped_y]),
                          eval_box=np.asarray(line['eval_box']).copy())
                item['eval_box'][:,1]=item['center'][1]
                members.append(item)
        for row in parse_rows(members,updated,boundaries[index:index+2]):
            row['row']=index+1;rows.append(row)
    return rows,dict(method='curved_printed_grid_per_column',boundaries=boundaries,
        detected_curves=[rule.public() for rule in rules]),mask


def signature(rows):
    return [(r['kind'],r['name_raw'],r['quantity_raw'],r['price_raw']) for r in rows if r['kind']!='total']


def evaluate(label,image,layout,lines,before,directory):
    directory.mkdir(parents=True,exist_ok=True)
    after,diagnostic,mask=candidate(image,layout,lines)
    changed=after is not None and signature(before)!=signature(after)
    # Changed grouping needs inspection against the real photo. No automatic
    # claim of improvement merely because geometry looks straighter.
    status='requires_review' if changed else 'unchanged'
    record=dict(file=label,status=status,diagnostic=diagnostic,before=before,after=after)
    (directory/'comparison.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    overlay=image.copy()
    for rule in diagnostic.get('detected_curves',[]):
        points=np.asarray(rule['curve']);xs=np.arange(len(points))*rule['step']
        cv2.polylines(overlay,[np.column_stack([xs,points]).astype(np.int32)],False,(0,150,0),1)
    for y in layout['row_boundaries']:
        cv2.line(overlay,(0,round(y)),(image.shape[1]-1,round(y)),(0,0,255),1)
    for name,im in [('comparison.png',overlay),('rule_mask.png',mask)]:
        (directory/name).write_bytes(cv2.imencode('.png',im)[1].tobytes())
    return record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    sources=sorted(p for p in (ROOT/'data/in_data').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    for source in sources:
        cached=OCR_CACHE/source.stem
        data=json.loads((cached/'baseline.json').read_text(encoding='utf-8'))
        image=cv2.imdecode(np.frombuffer((cached/'rectified.png').read_bytes(),np.uint8),1)
        lines=aligned_lines({'lines':data['lines']},{})
        record=evaluate(source.name,image,data['layout'],lines,data['receipt']['rows'],OUT/source.stem)
        records.append(record)
        print(source.name,record['status'],flush=True)
    # Also reproduce the exact screenshot job, without touching it.
    job=ROOT/'web_outputs/7024426dc8e54e90a35ca8e4ccf23853'
    metadata=json.loads((job/'rectification.json').read_text(encoding='utf-8'))
    image=cv2.imdecode(np.frombuffer((job/'rectified.png').read_bytes(),np.uint8),1)
    layout=table_layout(image,metadata)
    lines=aligned_lines(json.loads((job/'ocr_rectified.json').read_text(encoding='utf-8')),metadata)
    before=parse_rows(lines,layout,choose_row_boundaries(lines,layout)[0])
    target=evaluate('screenshot_job',image,layout,lines,before,OUT/'screenshot_job')
    # Reviewed against the actual printed receipts. Two candidates merge
    # distinct price rows; the service receipt restores separate VAT/total.
    reviewed={
        'WhatsApp Image 2026-10-22 at 13.52.58.jpeg':'worse',
        'WhatsApp Image 2026-10-23 at 17.55.11.jpeg':'worse',
        'WhatsApp Image 2026-10-23 at 08.19.57.jpeg':'better',
        # Reorders a spurious OCR "3" in the name but does not fix that error.
        'WhatsApp Image 2026-10-26 at 08.51.01.jpeg':'unchanged',
    }
    for record in records:
        record['assessment']=reviewed.get(record['file'],record['status'])
    summary=dict(files_tested=len(records),worse=sum(r['assessment']=='worse' for r in records),
                 better=sum(r['assessment']=='better' for r in records),
                 unchanged=sum(r['assessment']=='unchanged' for r in records),
                 requires_review=sum(r['assessment']=='requires_review' for r in records),
                 mean_delta_lines=0,mean_delta_confidence=0,
                 note='Raw OCR is unchanged; assessments concern grouping, not OCR confidence. Screenshot case improves separately; not counted twice in the dataset.',
                 assessments=[dict(file=r['file'],assessment=r['assessment']) for r in records],
                 screenshot_before=signature(target['before']),screenshot_after=signature(target['after'] or []))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def web_candidate(case):
    directory=case['job']['path']
    metadata=json.loads((directory/'rectification.json').read_text(encoding='utf-8'))
    image=cv2.imdecode(np.frombuffer((directory/'rectified.png').read_bytes(),np.uint8),1)
    layout=table_layout(image,metadata)
    lines=aligned_lines(dict(lines=case['ocr_lines']),metadata)
    rows,_,_=candidate(image,layout,lines)
    receipt=dict(case['receipt'],rows=rows if rows is not None else case['receipt']['rows'])
    return dict(receipt=receipt,ocr_lines=case['ocr_lines'],detail='первый эксперимент печатной сетки')


if __name__=='__main__':
    import sys
    if '--dataset' in sys.argv:main()
    else:
        from experiments.web_ocr_regression import main as web_main
        web_main('experiments.printed_grid_segmentation:web_candidate')
