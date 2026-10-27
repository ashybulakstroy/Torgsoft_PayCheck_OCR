"""Standalone, offline visual evidence for an interrupted OCR regression run."""
from pathlib import Path
import base64
import html
import json
import os
import webbrowser
from io import BytesIO
from PIL import Image


def read_optional(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def photo(path,rows=None,layout=None):
    if not path or not path.is_file():return '<p class="missing">Изображение недоступно: этап не завершён или файл отсутствует.</p>'
    try:
        data=path.read_bytes()
        with Image.open(BytesIO(data)) as image:width,height=image.size
    except Exception:return '<p class="missing">Сохранённое изображение повреждено или не читается.</p>'
    src='data:image/png;base64,'+base64.b64encode(data).decode()
    shapes=[]
    for row in rows or []:
        if row.get('kind')=='total' or not row.get('y_range'):continue
        start,end=row['y_range'];left=(layout or {}).get('body_left',0);right=(layout or {}).get('right',width)
        shapes.append(f'<rect x="{left}" y="{start}" width="{max(0,right-left)}" height="{max(0,end-start)}"/>')
    return f'<div class="photo"><img src="{src}" alt="Сохранённый чек"><svg viewBox="0 0 {width} {height}">'+''.join(shapes)+'</svg></div>'


def rows_of(payload):
    return [r for r in payload.get('receipt',{}).get('rows',[]) if r.get('kind')!='total']


def field_value(row,field):
    return row.get(field.removesuffix('_raw'),row.get(field,'')) if field in {'quantity_raw','price_raw'} else row.get(field,'')


def table(rows,other):
    if not rows:return '<p class="missing">Результат этого этапа не получен.</p>'
    body=[]
    fields=['name_raw','quantity_raw','price_raw']
    for index,row in enumerate(rows):
        neighbour=other[index] if index<len(other) else {}
        cells=[f'<td>{index+1}</td>']
        for field in fields:
            value=str(field_value(row,field) or '—')
            changed=field_value(row,field)!=field_value(neighbour,field)
            cells.append('<td'+(' class="changed"' if changed else '')+'>'+html.escape(value)+'</td>')
        body.append('<tr>'+''.join(cells)+'</tr>')
    return '<table><thead><tr><th>№</th><th>Название OCR</th><th>Кол-во</th><th>Цена</th></tr></thead><tbody>'+''.join(body)+'</tbody></table>'


def write_failure_report(summary,out,job=None,open_browser=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    directory=Path(job['path']) if job else None
    case_dir=out/job['id'] if job else out
    baseline=read_optional(case_dir/'baseline.json')
    candidate=read_optional(case_dir/'candidate.json')
    if not baseline and directory:
        baseline={'receipt':read_optional(directory/'receipt.json')}
    before,after=rows_of(baseline),rows_of(candidate)
    record=summary.get('records',[])[-1] if summary.get('records') else {}
    reasons=record.get('reasons') or [summary.get('error','Прогон остановлен до получения результата.')]
    layout=read_optional(directory/'layout.json').get('layout',{}) if directory else {}
    old_image=directory/'rectified.png' if directory else None
    new_image=case_dir/'rectified.png'
    reused=not new_image.is_file() and summary.get('scope')=='segmentation'
    if reused:new_image=old_image
    if not candidate:new_image=None
    changes=[]
    labels={'name_raw':'Название','quantity_raw':'Количество','price_raw':'Цена'}
    for i in range(max(len(before),len(after))):
        a=before[i] if i<len(before) else {};b=after[i] if i<len(after) else {}
        if not after:break
        for field,label in labels.items():
            if field_value(a,field)!=field_value(b,field):
                changes.append(f'Строка {i+1}, {label}: «{field_value(a,field) or "—"}» → «{field_value(b,field) or "—"}».')
    esc=html.escape
    title='OCR-тест остановлен — визуальный отчёт'
    content=f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>{title}</title><style>'
    content+='''*{box-sizing:border-box}body{font:16px Arial,sans-serif;color:#24374d;background:#f3f6fa;margin:0;padding:24px;max-width:1500px;margin:auto}h1{font-size:26px}h2{font-size:20px}p,li{line-height:1.5}.notice{padding:16px 20px;background:#fff0dc;border-left:5px solid #d49b23;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:white;border:1px solid #dae2ec;border-radius:10px;padding:18px;margin:16px 0}.photo{position:relative;max-width:750px;margin:auto}.photo img{display:block;width:100%;height:auto}.photo svg{position:absolute;inset:0;width:100%;height:100%;fill:none;stroke:#477de0;stroke-width:1.5;pointer-events:none}table{border-collapse:collapse;width:100%;font-size:14px;margin-top:16px}th,td{border:1px solid #dce3ed;padding:10px;text-align:left;vertical-align:top}th{background:#eef3f9}td:nth-child(3),td:nth-child(4){white-space:nowrap;font-weight:bold}.changed{background:#fff0be}.missing{padding:18px;background:#fff0ed;color:#a53838}details img{max-height:750px;width:auto;max-width:100%}.evidence{color:#61758b;font-size:14px}@media(max-width:850px){.grid{grid-template-columns:1fr}}@media print{body{background:white}.card{break-inside:avoid}}'''
    content+=f'</style><h1>{title}</h1><p>Чек: {esc(str((job or {}).get("filename","не определён")))} · ID {esc(str((job or {}).get("id","—")))}</p>'
    content+='<div class="notice"><b>Причина остановки</b><ul>'+''.join('<li>'+esc(str(r))+'</li>' for r in reasons)+'</ul>'
    content+='<p>FAIL означает, что проверка не пройдена. Неподтверждённое изменение само по себе ещё не доказывает ухудшение.</p></div>'
    if directory:
        content+='<details class="card"><summary>Исходное фото до подготовки изображения</summary>'+photo(directory/'original.png')+'</details>'
    content+='<div class="grid"><section class="card"><h2>До — сохранённый результат</h2>'+photo(old_image,before,layout)+table(before,after)+'</section>'
    content+='<section class="card"><h2>После — экспериментальный результат</h2>'+photo(new_image,after,layout)+table(after,before)
    if reused:content+='<p class="evidence">Проверялась группировка: изображение и OCR-блоки прежние, поверх показаны границы нового результата.</p>'
    content+='</section></div><section class="card"><h2>Что изменилось</h2>'
    content+='<ul>'+''.join('<li>'+esc(change)+'</li>' for change in changes)+'</ul>' if changes else '<p>Различия данных недоступны либо отсутствуют; причина остановки указана выше.</p>'
    content+=f'<p>Проверено: {summary.get("files_tested",0)} из {summary.get("available",summary.get("requested",100))}. После этой остановки следующие чеки не обрабатывались.</p>'
    content+='<p class="evidence">Жёлтым выделены отличающиеся поля. Это обозначение изменения, а не утверждение, что новый вариант правильный. Файлы веб-задания не изменялись.</p></section></html>'
    path=out/'failure_report.html';path.write_text(content,encoding='utf-8')
    summary['visual_report']=str(path.resolve())
    if open_browser:
        try:
            if os.name=='nt':os.startfile(str(path.resolve()))
            elif not webbrowser.open(path.resolve().as_uri()):raise RuntimeError('браузер недоступен')
            summary['report_opened']=True
        except Exception as error:summary.update(report_opened=False,report_open_error=str(error))
    return path
