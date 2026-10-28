"""Fail-fast regression harness for the latest 100 completed web receipts.

Candidate protocol: callable(case) -> {receipt, ocr_lines, verified?, detail?}.
Only independently reviewed references may set verified=True for changed text.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import json
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def latest_jobs(root,limit=100):
    jobs=[]
    for directory in root.iterdir():
        if not directory.is_dir() or not (directory/'state.json').exists():continue
        state=read(directory/'state.json')
        if state.get('status')=='done':
            jobs.append(dict(id=directory.name,path=directory,created_at=float(state['created_at']),
                             filename=state.get('filename','')))
    return sorted(jobs,key=lambda job:(job['created_at'],job['id']),reverse=True)[:limit]


def quality_gate_jobs(web_root,golden_root,limit=16):
    """Select unique golden receipts first, then newest completed web receipts."""
    selected=[];digests=set()
    def add(directory,created_at,filename,source):
        upload=directory/'upload.bin'
        if not upload.is_file():raise ValueError(f'нет upload.bin: {directory}')
        digest=hashlib.sha256(upload.read_bytes()).hexdigest()
        if digest in digests or len(selected)>=limit:return
        for name in ['receipt.json','ocr_rectified.json','original.png']:
            if not (directory/name).is_file():raise ValueError(f'нет артефакта {name}: {directory}')
        digests.add(digest)
        selected.append(dict(id=directory.name,path=directory,created_at=created_at,
                             filename=filename,source=source,upload_sha256=digest))
    golden_root=Path(golden_root)
    if golden_root.exists():
        for directory in sorted((path for path in golden_root.iterdir() if path.is_dir()),key=lambda p:p.name):
            receipt=read(directory/'receipt.json') if (directory/'receipt.json').is_file() else {}
            add(directory,0.,receipt.get('filename',''), 'golden')
    for job in latest_jobs(Path(web_root),100):
        add(job['path'],job['created_at'],job['filename'],'web')
    return selected


def raw_receipt(saved):
    result=copy.deepcopy(saved)
    # Compare OCR output, not a user's subsequent edits or catalogue names.
    for row in result['rows']:
        row['name']=row.get('name_raw','')
        row['quantity']=row.get('quantity_raw','')
        row['price']=row.get('price_raw','')
        row['name_confirmed']=False
    result['expected_total']=result.get('total_raw','')
    result['expected_total_source']='ocr'
    return result


def load_case(job):
    directory=job['path']
    for name in ['receipt.json','ocr_rectified.json','original.png']:
        if not (directory/name).is_file():raise ValueError(f'нет артефакта {name}')
    return dict(job=job,receipt=raw_receipt(read(directory/'receipt.json')),
                ocr_lines=read(directory/'ocr_rectified.json')['lines'])


def assess(case,candidate):
    from App.receipt_editing import parse_number,update_receipt
    from App.ocr_catalog_eval import classify_name,is_table_header_row
    before=case['receipt'];after=candidate['receipt']
    tolerated=candidate.get('review_disposition')=='tolerated_unchanged'
    old_all=[r for r in before['rows'] if r['kind'] in {'product','service','tax'}]
    removed_headers=[r for r in old_all if is_table_header_row(r)]
    old=[r for r in old_all if r not in removed_headers]
    new=[r for r in after['rows'] if r['kind'] in {'product','service','tax'}]
    reasons=[];changes=[]
    if removed_headers and not any(is_table_header_row(row) for row in new):
        changes.append('исключена строка шапки таблицы')
    if len(old)!=len(new):
        message=f'число строк {len(old)} → {len(new)}'
        if candidate.get('verified'):changes.append(message)
        else:reasons.append(message+': требуется эталон')
    def key(value):return ''.join(str(value).casefold().split())
    for index,(a,b) in enumerate(zip(old,new),1):
        if a['kind']!=b['kind']:reasons.append(f'строка {index}: изменён тип')
        if key(a.get('name_raw',''))!=key(b.get('name_raw','')):changes.append(f'название {index}')
        for field in ['quantity','price']:
            a_value=a.get(field,a.get(field+'_raw'));b_value=b.get(field,b.get(field+'_raw'))
            left,right=parse_number(a_value),parse_number(b_value)
            if left is not None and right!=left:reasons.append(f'строка {index}: изменено число {field}')
            elif left is None and right is not None:changes.append(f'{field} {index}')
        if classify_name(a.get('name_raw',''))=='tax' and b['kind']!='tax':reasons.append(f'потеря НДС {index}')
    old_calc=update_receipt(copy.deepcopy(before))['calculation']
    new_calc=update_receipt(copy.deepcopy(after))['calculation']
    if old_calc.get('validation_status',old_calc['status'])=='match' and new_calc.get('validation_status',new_calc['status'])!='match':
        reasons.append('потеря успешной проверки суммы/НДС')
    old_lines=case['ocr_lines'];new_lines=candidate['ocr_lines']
    delta=len(new_lines)-len(old_lines)
    mean=lambda lines:sum(float(l['confidence']) for l in lines)/len(lines) if lines else 0
    confidence=mean(new_lines)-mean(old_lines)
    old_ocr_texts=[str(line.get('text','')) for line in old_lines]
    new_ocr_texts=[str(line.get('text','')) for line in new_lines]
    # OCR may switch visually identical Cyrillic/Latin glyphs together with
    # letter case (for example currency "т" -> "T"). This is not a content
    # regression when the complete set of recognized blocks is unchanged.
    confusables=str.maketrans({'а':'a','в':'b','е':'e','к':'k','м':'m','н':'h',
                              'о':'o','р':'p','с':'c','т':'t','х':'x','у':'y'})
    def ocr_text_key(text):return text.casefold().translate(confusables)
    equivalent_ocr_text_change=(
        len(old_lines)==len(new_lines)
        and sorted(ocr_text_key(text) for text in old_ocr_texts)
            == sorted(ocr_text_key(text) for text in new_ocr_texts)
        and old_ocr_texts!=new_ocr_texts
    )
    if delta<0 and not tolerated:reasons.append('меньше OCR-блоков: требуется подтверждение')
    if confidence < -1e-6 and not tolerated and not equivalent_ocr_text_change:
        reasons.append('снизилась средняя уверенность OCR')
    tolerated=bool(changes) and tolerated
    non_header_changes=[change for change in changes if change != 'исключена строка шапки таблицы']
    if non_header_changes and not candidate.get('verified') and not tolerated:
        reasons.append('изменённые данные не подтверждены независимым эталоном')
    # A verified reference may approve corrected fields, but cannot override a
    # numeric regression or lost VAT/math check silently in this general runner.
    return dict(status='worse' if reasons else 'unchanged' if tolerated else 'better' if changes else 'unchanged',
                reasons=reasons,changes=changes,delta_lines=delta,delta_confidence=confidence,
                detail=('состав OCR-текста не изменился; отличается только регистр, похожий OCR-глиф или порядок блоков'
                        if equivalent_ocr_text_change else candidate.get('detail','')))


def run(candidate,root,out,limit=100,scope='segmentation',emit=print,open_report=False,start_index=1,jobs_override=None):
    out.mkdir(parents=True,exist_ok=True)
    summary=dict(requested=limit,files_tested=0,worse=0,better=0,unchanged=0,
                 mean_delta_lines=0,mean_delta_confidence=0,status='running',scope=scope,records=[])
    def save():
        rows=summary['records'];count=len(rows)
        summary['files_tested']=count
        for status in ['worse','better','unchanged']:summary[status]=sum(r['status']==status for r in rows)
        for name in ['lines','confidence']:
            summary['mean_delta_'+name]=sum(r.get('delta_'+name,0) for r in rows)/count if count else 0
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    def visual_stop(job=None):
        from experiments.ocr_failure_report import write_failure_report
        try:
            path=write_failure_report(summary,out,job,open_browser=open_report)
            emit(f'Визуальный отчёт: {path.resolve()}',flush=True)
            if summary.get('report_open_error'):emit(f'Не удалось открыть браузер: {summary["report_open_error"]}',flush=True)
        except Exception as error:
            summary['visual_report_error']=str(error)
            emit(f'Ошибка создания визуального отчёта: {error}. Данные остановки: {out/"summary.json"}',flush=True)
        save()
    try:jobs=list(jobs_override) if jobs_override is not None else latest_jobs(root,limit)
    except (Exception,KeyboardInterrupt) as error:
        summary.update(status='failed',error=f'Не удалось выбрать историю: {type(error).__name__}: {error}');save();emit(summary['error']);visual_stop();return 1,summary
    total_available=len(jobs)
    if start_index < 1 or start_index > total_available + 1:
        raise ValueError(f'некорректный start_index={start_index}; доступно {total_available} чеков')
    jobs=jobs[start_index-1:]
    summary['available']=total_available;summary['start_index']=start_index
    summary['coverage_complete']=start_index==1 and total_available==limit
    emit(f'Веб-история: выбрано {len(jobs)} чеков, начиная с {start_index}-го из {total_available}/{limit} завершённых.',flush=True)
    # A fixed selection makes the run reproducible if new jobs arrive mid-test.
    summary['selected_jobs']=[j['id'] for j in jobs];save()
    for relative_index,job in enumerate(jobs,1):
        index=start_index+relative_index-1
        emit(f'[{index}/{total_available}] START {job["id"]} {job["filename"]}',flush=True)
        try:
            case=load_case(job);case['output']=out/job['id'];case['output'].mkdir(exist_ok=True)
            baseline=copy.deepcopy(case)
            (case['output']/'baseline.json').write_text(json.dumps(dict(receipt=baseline['receipt'],ocr_lines=baseline['ocr_lines']),ensure_ascii=False,indent=2),encoding='utf-8')
            proposed=candidate(case)
            for filename,payload in [('baseline.json',dict(receipt=baseline['receipt'],ocr_lines=baseline['ocr_lines'])),
                                     ('candidate.json',proposed)]:
                (case['output']/filename).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            if scope=='full_pipeline' and proposed.get('scope')!='full_pipeline':
                raise ValueError('кандидат не подтвердил полный повторный OCR из исходного изображения')
            result=assess(baseline,proposed)
        except (Exception,KeyboardInterrupt) as error:
            result=dict(status='worse',reasons=[f'{type(error).__name__}: {error}'],changes=[],delta_lines=0,delta_confidence=0)
        record=dict(id=job['id'],filename=job['filename'],**result);summary['records'].append(record)
        reason='; '.join(result['reasons']) or result.get('detail') or result['status']
        emit(f'[{index}/{total_available}] {"FAIL" if result["status"]=="worse" else "PASS"} {job["id"]} '
             f'Δстрок={result["delta_lines"]:+d} Δуверенности={result["delta_confidence"]:+.4f} {reason}',flush=True)
        summary['status']='failed' if result['status']=='worse' else 'running';save()
        if result['status']=='worse':
            emit('FAIL: остановка на первом отрицательном результате.',flush=True);visual_stop(job);return 1,summary
    summary['status']='passed' if summary['coverage_complete'] else 'incomplete';save()
    emit(f'{summary["status"].upper()}: проверено {len(jobs)}/{limit}, хуже={summary["worse"]}, '
         f'лучше={summary["better"]}, без изменений={summary["unchanged"]}',flush=True)
    return (0 if summary['coverage_complete'] else 2),summary


def current_pipeline(case):
    """Run current production algorithm in an isolated output job, not live HTTP."""
    from App.web_server import Job
    from App.web_pipeline import ReceiptPipeline
    source=case['job']['path']/'upload.bin'
    if not source.exists():raise ValueError('нет исходного upload.bin для полного OCR')
    (case['output']/'upload.bin').write_bytes(source.read_bytes())
    if not hasattr(current_pipeline,'pipeline'):current_pipeline.pipeline=ReceiptPipeline()
    job=Job(case['job']['id'],case['job']['filename'],case['output'])
    receipt=current_pipeline.pipeline.process(job)
    return dict(receipt=raw_receipt(receipt),ocr_lines=read(case['output']/'ocr_rectified.json')['lines'],
                scope='full_pipeline',detail='полный OCR из upload.bin')


def main(default_candidate=None,default_scope='segmentation'):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',default=default_candidate,required=default_candidate is None,help='module:function')
    parser.add_argument('--jobs',type=Path,default=ROOT/'web_outputs')
    parser.add_argument('--golden',type=Path,default=ROOT/'data'/'golden')
    parser.add_argument('--limit',type=int,default=16,help='общий лимит golden + последние веб-чеки')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--scope',choices=['segmentation','full_pipeline'],default=default_scope)
    parser.add_argument('--start-index',type=int,default=1,help='начать с позиции в списке новых завершённых чеков (нумерация с 1)')
    parser.add_argument('--no-open-report',action='store_true',help='не открывать браузер (для запуска без UI)')
    args=parser.parse_args()
    module,name=args.candidate.split(':',1)
    candidate=getattr(importlib.import_module(module),name)
    from App.artifact_paths import OCR_REPORTS
    out=args.output or OCR_REPORTS/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    selected=quality_gate_jobs(args.jobs,args.golden,args.limit)
    code,summary=run(candidate,args.jobs,out,limit=args.limit,scope=args.scope,
                     open_report=not args.no_open_report,start_index=args.start_index,jobs_override=selected)
    print(f'Отчёт: {out/"summary.json"}',flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
