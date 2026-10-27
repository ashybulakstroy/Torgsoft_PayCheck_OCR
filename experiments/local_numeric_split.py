"""Regression adapter for the production local quantity/price repair."""
import copy
import json
from experiments.guarded_printed_grid import matches_reviewed_reference

from App.numeric_cell_split import recover


def web_candidate(case):
    directory=case['job']['path']
    layout=json.loads((directory/'layout.json').read_text(encoding='utf-8'))['layout']
    before=case['receipt']
    after,diagnostics=recover(before,layout,case['ocr_lines'])
    for a,b in zip(before['rows'],after['rows']):
        for field in ['name_raw','quantity_raw','price_raw','raw_lines','row','y_range','kind']:
            assert a.get(field)==b.get(field),f'Unexpected non-local change: {field}'
    reference_view=copy.deepcopy(after)
    for row in reference_view['rows']:
        row['quantity_raw']=row.get('quantity',row.get('quantity_raw',''))
        row['price_raw']=row.get('price',row.get('price_raw',''))
    verified=matches_reviewed_reference(case,reference_view)
    (case['output']/'numeric_split.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    applied=sum(item['applied'] for item in diagnostics)
    detail=f'локально разделено пар: {applied}; названия и сетка сохранены'
    return dict(receipt=after,ocr_lines=case['ocr_lines'],verified=verified,detail=detail)


if __name__=='__main__':
    from experiments.web_ocr_regression import main
    main('experiments.local_numeric_split:web_candidate')
