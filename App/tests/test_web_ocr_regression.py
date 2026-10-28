"""The web regression gate must stop before processing any job after failure."""
import copy
import json
from pathlib import Path
from experiments.web_ocr_regression import latest_jobs,quality_gate_jobs,run


def write_job(root,index,status='done'):
    directory=root/f'{index:032x}';directory.mkdir(parents=True)
    def dump(name,value):(directory/name).write_text(json.dumps(value),encoding='utf-8')
    dump('state.json',dict(created_at=index,status=status,filename=f'photo-{index}.jpg'))
    row=dict(row=1,kind='product',name_raw='Товар',quantity_raw='1',price_raw='10',raw_lines=[])
    dump('receipt.json',dict(rows=[row],total_raw='10'))
    dump('ocr_rectified.json',dict(lines=[dict(text='Товар',confidence=.9,box=[[0,0],[1,0],[1,1],[0,1]])]))
    (directory/'original.png').write_bytes(b'image')
    (directory/'upload.bin').write_bytes(f'upload-{index}'.encode())
    return directory


def unchanged(case):
    return dict(receipt=copy.deepcopy(case['receipt']),ocr_lines=case['ocr_lines'])


def quiet(*args,**kwargs):pass


def test_selects_latest_100_by_creation_not_filename_or_modification(tmp_path):
    for index in range(103):write_job(tmp_path,index)
    write_job(tmp_path,1000,status='failed')
    jobs=latest_jobs(tmp_path)
    assert len(jobs)==100
    assert [j['created_at'] for j in jobs]==list(reversed(range(3,103)))


def test_quality_gate_uses_unique_golden_then_latest_web_up_to_limit(tmp_path):
    golden=tmp_path/'golden';web=tmp_path/'web'
    first=write_job(golden,1);write_job(golden,2)
    duplicate=write_job(web,10)
    (duplicate/'upload.bin').write_bytes((first/'upload.bin').read_bytes())
    write_job(web,11);write_job(web,12);write_job(web,13)
    jobs=quality_gate_jobs(web,golden,limit=4)
    assert len(jobs)==4
    assert [job['source'] for job in jobs]==['golden','golden','web','web']
    assert jobs[2]['created_at']==13 and jobs[3]['created_at']==12
    assert len({job['upload_sha256'] for job in jobs})==4


def test_stops_immediately_and_saves_partial_report(tmp_path):
    root=tmp_path/'jobs'
    for index in range(3):write_job(root,index)
    called=[];messages=[]
    def candidate(case):
        called.append(case['job']['id']);result=unchanged(case)
        if len(called)==2:result['receipt']['rows'][0]['quantity']='2'
        return result
    code,summary=run(candidate,root,tmp_path/'out',limit=3,emit=lambda *a,**kw:messages.append(a[0]))
    assert code==1 and len(called)==2 and summary['worse']==1
    assert summary['status']=='failed' and summary['files_tested']==2
    assert any('[2/3] FAIL' in m for m in messages)
    assert json.loads((tmp_path/'out/summary.json').read_text())['files_tested']==2


def test_missing_artifact_fails_before_candidate(tmp_path):
    directory=write_job(tmp_path/'jobs',1);(directory/'ocr_rectified.json').unlink()
    def must_not_run(case):raise AssertionError('candidate ran')
    code,summary=run(must_not_run,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==1 and 'ocr_rectified.json' in summary['records'][0]['reasons'][0]


def test_insufficient_history_is_not_full_pass(tmp_path):
    write_job(tmp_path/'jobs',1)
    code,summary=run(unchanged,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==2 and summary['status']=='incomplete'
    assert summary['available']==1 and not summary['coverage_complete']


def test_full_success(tmp_path):
    for index in range(3):write_job(tmp_path/'jobs',index)
    code,summary=run(unchanged,tmp_path/'jobs',tmp_path/'out',limit=3,emit=quiet)
    assert code==0 and summary['status']=='passed' and summary['unchanged']==3


def test_unverified_regrouping_is_failure(tmp_path):
    write_job(tmp_path/'jobs',1)
    def changed(case):
        result=unchanged(case);result['receipt']['rows'][0]['name_raw']='Другой товар';return result
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==1 and 'эталоном' in ';'.join(summary['records'][0]['reasons'])


def test_removing_split_table_header_is_safe_improvement(tmp_path):
    write_job(tmp_path/'jobs',1)
    receipt_path=next((tmp_path/'jobs').glob('*/receipt.json'))
    receipt=json.loads(receipt_path.read_text())
    receipt['rows'].insert(0,dict(row=1,kind='product',name_raw='N',quantity_raw='Саны',price_raw='Багасы',raw_lines=[]))
    receipt_path.write_text(json.dumps(receipt))
    def without_header(case):
        result=unchanged(case)
        result['receipt']['rows']=result['receipt']['rows'][1:]
        return result
    code,summary=run(without_header,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==2 and summary['better']==1 and summary['worse']==0


def test_user_tolerated_name_change_is_unchanged(tmp_path):
    write_job(tmp_path/'jobs',1)
    def tolerated(case):
        result=unchanged(case)
        result['receipt']['rows'][0]['name_raw']='Незначительное изменение'
        result['review_disposition']='tolerated_unchanged'
        return result
    code,summary=run(tolerated,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==2 and summary['unchanged']==1 and summary['better']==0


def test_user_tolerated_missing_non_data_ocr_block_is_unchanged(tmp_path):
    write_job(tmp_path/'jobs',1)
    def tolerated(case):
        result=unchanged(case)
        result['ocr_lines']=result['ocr_lines'][:-1]
        result['review_disposition']='tolerated_unchanged'
        return result
    code,summary=run(tolerated,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==2 and summary['unchanged']==1 and summary['worse']==0


def test_start_index_skips_previous_jobs(tmp_path):
    for index in range(3):write_job(tmp_path/'jobs',index)
    code,summary=run(unchanged,tmp_path/'jobs',tmp_path/'out',limit=3,start_index=3,emit=quiet)
    assert code==2 and summary['files_tested']==1 and summary['start_index']==3


def test_verified_reference_can_approve_restored_row_count(tmp_path):
    write_job(tmp_path/'jobs',1)
    def changed(case):
        result=unchanged(case)
        result['receipt']['rows'].append(copy.deepcopy(result['receipt']['rows'][0]))
        result['receipt']['total_raw']='20'
        result['receipt']['expected_total']='20'
        result['verified']=True
        return result
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',limit=1,emit=quiet)
    assert code==0
    assert summary['better']==1 and summary['worse']==0


def test_exceptions_stop_not_skip(tmp_path):
    for index in range(2):write_job(tmp_path/'jobs',index)
    def crash(case):raise RuntimeError('OCR crashed')
    code,summary=run(crash,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==1 and summary['files_tested']==1


def test_candidate_cannot_modify_its_own_baseline(tmp_path):
    write_job(tmp_path/'jobs',1)
    def changed(case):
        case['receipt']['rows'][0]['price']='99'
        return dict(receipt=case['receipt'],ocr_lines=case['ocr_lines'])
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==1
    original=json.loads(next((tmp_path/'jobs').glob('*/receipt.json')).read_text())
    assert original['rows'][0]['price_raw']=='10'


def test_full_pipeline_scope_rejects_cached_candidate(tmp_path):
    write_job(tmp_path/'jobs',1)
    code,summary=run(unchanged,tmp_path/'jobs',tmp_path/'out',scope='full_pipeline',emit=quiet)
    assert code==1 and 'полный' in summary['records'][0]['reasons'][0]


def test_confidence_regression_fails(tmp_path):
    write_job(tmp_path/'jobs',1)
    def changed(case):
        result=unchanged(case);result['ocr_lines']=copy.deepcopy(result['ocr_lines'])
        result['ocr_lines'][0]['confidence']=.5;return result
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==1 and summary['mean_delta_confidence']<0


def test_letter_case_change_is_not_an_ocr_regression(tmp_path):
    write_job(tmp_path/'jobs',1)
    def changed(case):
        result=unchanged(case)
        result['ocr_lines']=copy.deepcopy(result['ocr_lines'])
        result['ocr_lines'][0]['text']='товар'
        result['ocr_lines'][0]['confidence']=.5
        return result
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==2
    assert summary['unchanged']==1 and summary['worse']==0
    assert 'регистр' in summary['records'][0]['detail']


def test_same_ocr_blocks_in_different_order_are_not_a_regression(tmp_path):
    directory=write_job(tmp_path/'jobs',1)
    payload=json.loads((directory/'ocr_rectified.json').read_text())
    payload['lines'].append(dict(text='НДС',confidence=.9,box=[[0,0],[1,0],[1,1],[0,1]]))
    (directory/'ocr_rectified.json').write_text(json.dumps(payload))
    def changed(case):
        result=unchanged(case)
        result['ocr_lines']=list(reversed(copy.deepcopy(result['ocr_lines'])))
        for line in result['ocr_lines']:line['confidence']-=.1
        return result
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==2
    assert summary['unchanged']==1 and summary['worse']==0


def test_cyrillic_lowercase_to_latin_uppercase_glyph_is_not_a_regression(tmp_path):
    directory=write_job(tmp_path/'jobs',1)
    payload=json.loads((directory/'ocr_rectified.json').read_text())
    payload['lines'][0]['text']='1 т'
    (directory/'ocr_rectified.json').write_text(json.dumps(payload))
    def changed(case):
        result=unchanged(case)
        result['ocr_lines']=copy.deepcopy(result['ocr_lines'])
        result['ocr_lines'][0].update(text='1 T',confidence=.5)
        return result
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==2
    assert summary['unchanged']==1 and summary['worse']==0


def test_user_edits_are_not_baseline_ocr(tmp_path):
    directory=write_job(tmp_path/'jobs',1)
    original=json.loads((directory/'receipt.json').read_text())
    original['rows'][0].update(name='Название из справочника',quantity='12',price='100')
    (directory/'receipt.json').write_text(json.dumps(original),encoding='utf-8')
    code,summary=run(unchanged,tmp_path/'jobs',tmp_path/'out',limit=1,emit=quiet)
    assert code==0
    baseline=json.loads(next((tmp_path/'out').glob('*/baseline.json')).read_text())
    assert baseline['receipt']['rows'][0]['quantity']=='1'


def test_corrupt_completed_job_is_not_silently_skipped(tmp_path):
    directory=write_job(tmp_path/'jobs',1)
    (directory/'state.json').write_text('{broken',encoding='utf-8')
    code,summary=run(unchanged,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==1 and summary['status']=='failed' and summary['files_tested']==0


def test_user_reference_is_bound_to_input_and_exact_reviewed_output(tmp_path):
    import hashlib
    from experiments.guarded_printed_grid import matches_reviewed_reference
    directory=write_job(tmp_path/'jobs',1)
    case=dict(job=dict(id=directory.name,path=directory))
    receipt=json.loads((directory/'receipt.json').read_text())
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(receipts={directory.name:dict(reviewer='user',
        input_sha256={'original.png':hashlib.sha256((directory/'original.png').read_bytes()).hexdigest()},
        expected_rows=[['product','Товар','1','10']])})),encoding='utf-8')
    assert matches_reviewed_reference(case,receipt,reference)
    # Another upload of byte-identical input is covered by the same review.
    case['job']['id']='another-upload'
    assert matches_reviewed_reference(case,receipt,reference)
    receipt['rows'][0]['name_raw']='Другой товар'
    assert not matches_reviewed_reference(case,receipt,reference)
    receipt['rows'][0]['name_raw']='Товар'
    (directory/'original.png').write_bytes(b'changed input')
    assert not matches_reviewed_reference(case,receipt,reference)


def test_failure_report_shows_before_after_and_escaped_text(tmp_path):
    write_job(tmp_path/'jobs',1)
    def changed(case):
        result=unchanged(case);result['receipt']['rows'][0]['name_raw']='<script>test</script>';return result
    code,summary=run(changed,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    report=Path(summary['visual_report']).read_text(encoding='utf-8')
    assert code==1 and 'До — сохранённый результат' in report and 'После — экспериментальный результат' in report
    assert '&lt;script&gt;test&lt;/script&gt;' in report and '<script>' not in report
    assert 'Строка 1' in report and 'изменённые данные' in report


def test_candidate_crash_still_has_baseline_and_visual_report(tmp_path):
    write_job(tmp_path/'jobs',1)
    def crash(case):raise RuntimeError('stage crashed')
    code,summary=run(crash,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    report=Path(summary['visual_report']).read_text(encoding='utf-8')
    assert code==1 and 'stage crashed' in report and 'Товар' in report
    assert 'Результат этого этапа не получен' in report


def test_interruption_reports_and_does_not_continue(tmp_path):
    for i in range(2):write_job(tmp_path/'jobs',i)
    def interrupted(case):raise KeyboardInterrupt()
    code,summary=run(interrupted,tmp_path/'jobs',tmp_path/'out',emit=quiet)
    assert code==1 and summary['files_tested']==1
    assert 'KeyboardInterrupt' in Path(summary['visual_report']).read_text(encoding='utf-8')


def test_failure_opens_os_browser_only_when_requested(tmp_path,monkeypatch):
    import os
    from experiments.ocr_failure_report import write_failure_report
    opened=[]
    monkeypatch.setattr(os,'startfile',lambda path:opened.append(path),raising=False)
    summary=dict(status='failed',error='test error')
    path=write_failure_report(summary,tmp_path,open_browser=True)
    assert summary['report_opened'] and opened==[str(path.resolve())]


def test_browser_error_does_not_hide_original_failure(tmp_path,monkeypatch):
    import os
    from experiments.ocr_failure_report import write_failure_report
    def blocked(path):raise OSError('browser unavailable')
    monkeypatch.setattr(os,'startfile',blocked,raising=False)
    summary=dict(status='failed',error='OCR failure')
    path=write_failure_report(summary,tmp_path,open_browser=True)
    assert path.exists() and summary['status']=='failed' and not summary['report_opened']
