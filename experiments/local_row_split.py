"""Regression adapter for the production local row-split guard."""
import json
import cv2
import numpy as np
from App.row_cell_split import repair



def web_candidate(case):
    directory=case['job']['path']
    layout=json.loads((directory/'layout.json').read_text(encoding='utf-8'))['layout']
    image=cv2.imdecode(np.frombuffer((directory/'rectified.png').read_bytes(),np.uint8),1)
    if image is None:raise ValueError('rectified image is missing or corrupt')
    receipt,diagnostics=repair(case['receipt'],layout,case['ocr_lines'],image)
    (case['output']/'row_split.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    count=sum(d['applied'] for d in diagnostics)
    return dict(receipt=receipt,ocr_lines=case['ocr_lines'],detail=f'разделено склеенных строк: {count}; соседние названия не переносились')


if __name__=='__main__':
    from experiments.web_ocr_regression import main
    main('experiments.local_row_split:web_candidate')
