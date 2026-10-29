"""Cache identity must change when a source, model, or production module changes."""
import pytest
import experiments.evaluate_local_row_split as evaluation


@pytest.mark.parametrize('changed',['source','model','module'])
def test_cache_fingerprint_tracks_real_inputs(tmp_path,monkeypatch,changed):
    production=['web_pipeline','ocr_catalog_eval','receipt_rectification','contour_unwarp',
                'wave_dewarp','table_boundary','quantity_cells','numeric_cell_split','receipt_editing','receipt_math']
    (tmp_path/'App').mkdir()
    for name in production:(tmp_path/'App'/f'{name}.py').write_text('baseline',encoding='utf-8')
    source=tmp_path/'photo.jpg';source.write_bytes(b'original photo')
    model=tmp_path/'recognizer.onnx';model.write_bytes(b'original model')
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    monkeypatch.setattr(evaluation,'MODEL',model)
    initial,_=evaluation.fingerprint(source)
    assert evaluation.fingerprint(source)[0]==initial
    path={'source':source,'model':model,'module':tmp_path/'App/web_pipeline.py'}[changed]
    path.write_bytes(path.read_bytes()+b' changed')
    assert evaluation.fingerprint(source)[0]!=initial
