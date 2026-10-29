r"""Run from project root: .\.venv\Scripts\python.exe -m pytest -q tests/test_selective_number_ocr.py"""
from copy import deepcopy

import cv2
import numpy as np

from App.selective_number_ocr import parse_index, suspect, make_views, recover_number_cells


def fixture():
    """Synthetic two-column outline; glyphs are actual pixels, never OCR labels."""
    image = np.full((220, 180, 3), 245, dtype=np.uint8)
    layout = {'body_left': 9, 'name_left': 80, 'name_right': 130,
              'quantity_right': 150, 'right': 175}
    rows = []
    for i in range(3):
        t, b = 22 + i * 55, 77 + i * 55
        cv2.rectangle(image, (9, t), (175, b), (70,70,70), 1)
        cv2.line(image, (80,t), (80,b), (70,70,70), 1)
        cv2.putText(image, str(i+1), (56, t + 37), cv2.FONT_HERSHEY_SIMPLEX,
                    1.05, (10,10,10), 2, cv2.LINE_AA)
        rows.append(dict(row=i+1, kind='product', item_number_raw=str(i+1),
                         y_range=[t,b], name_raw=f'Item {i+1}', quantity_raw='1', price_raw='20'))
    return image, layout, rows


def test_parse_only_factual_digits():
    assert parse_index('  3  ') == 3
    for bad in ('', 'I', 'l', 'No3', '3a', '0', '-2', '1000'):
        assert parse_index(bad) is None


def test_targets_only_bad_cell():
    image, layout, rows = fixture()
    rows[1]['item_number_raw'] = ''
    captured = []

    def fake(images):
        captured.extend(images)
        return [('2', .31), ('2', .28)]

    original = deepcopy(rows)
    updated, report = recover_number_cells(image, rows, layout, fake)
    assert len(captured) == 2
    assert all(view.ndim == 3 and view.shape[2] == 3 for view in captured)
    assert report['selected_cells'] == 1
    assert report['recognizer_calls'] == 1
    assert report['recovered'] == 1
    assert rows == original
    assert updated[1]['item_number_raw'] == ''
    assert updated[1]['item_number_recovered'] == '2'
    assert 'item_number_recovered' not in updated[0]
    assert updated[1]['price_raw'] == '20'


def test_no_unnecessary_ocr_when_all_read():
    image, layout, rows = fixture()
    def prohibited(_):
        raise AssertionError('OCR must not be called')
    result, report = recover_number_cells(image, rows, layout, prohibited)
    assert report['selected_cells'] == 0 and report['recognizer_calls'] == 0
    assert result == rows


def test_disagreement_keeps_original():
    image, layout, rows = fixture()
    rows[1]['item_number_raw'] = 'ВО'
    result, report = recover_number_cells(image, rows, layout, lambda imgs:[('2',.9),('3',.9)])
    assert report['attempts'][0]['status'] == 'unconfirmed_readings'
    assert 'item_number_recovered' not in result[1]


def test_low_confidence_keeps_original():
    image, layout, rows = fixture()
    rows[1]['item_number_raw'] = ''
    result, report = recover_number_cells(image, rows, layout, lambda imgs:[('2',.15),('2',.28)])
    assert report['attempts'][0]['status'] == 'insufficient_confidence'
    assert 'item_number_recovered' not in result[1]


def test_sequence_mismatch_keeps_original():
    image, layout, rows = fixture()
    rows[1]['item_number_raw'] = ''
    result, report = recover_number_cells(image, rows, layout, lambda imgs:[('7',.99),('7',.99)])
    assert report['attempts'][0]['status'] == 'sequence_conflict_after_ocr'
    assert 'item_number_recovered' not in result[1]


def test_conflicting_original_is_selected_but_not_overwritten():
    image, layout, rows = fixture()
    rows[1]['item_number_raw'] = '8'
    result, report = recover_number_cells(image, rows, layout, lambda imgs:[('2',.49),('2',.44)])
    assert report['selected_cells'] == 1
    assert result[1]['item_number_recovered'] == '2'
    assert result[1]['item_number_raw'] == '8'


def test_no_crop_does_not_trigger_model():
    image, layout, rows = fixture()
    rows[1]['item_number_raw'] = ''
    rows[1]['y_range'] = [40,40]
    result, report = recover_number_cells(image, rows, layout, lambda _: (_ for _ in ()).throw(AssertionError()))
    assert report['selected_cells'] == 0
    assert report['attempts'][0]['status'] == 'no_readable_crop'


def test_max_cells_restricts_expensive_work():
    image, layout, rows = fixture()
    for row in rows:
        row['item_number_raw'] = ''
    result, report = recover_number_cells(image, rows, layout, lambda imgs:[('1',.5),('1',.5)], max_cells=1)
    assert report['selected_cells'] == 1
    assert sum(x['status']=='skipped_limit' for x in report['attempts'])==2


def test_thin_digit_near_boundary_is_preserved():
    image, layout, rows = fixture()
    rows[0]['item_number_raw'] = ''
    crop = make_views(image, rows[0], layout)
    assert crop is not None
    assert crop['digit_box'][0] < 74
    assert crop['views'][0].shape[2] == 3
    # All training versions must preserve an actual ink component.
    assert crop['views'][0].min() < 100


def test_only_rows_does_not_scan_other_missing_numbers():
    image, layout, rows = fixture()
    for row in rows:
        row['item_number_raw'] = ''
    result, report = recover_number_cells(image, rows, layout,
                      lambda ims:[('2', .6),('2', .6)], only_rows={2})
    assert report['selected_cells'] == 1
    assert result[1]['item_number_recovered'] == '2'
    assert 'item_number_recovered' not in result[0]
    assert 'item_number_recovered' not in result[2]


def test_dominant_digit_beats_weak_nondigit_variant_only_with_sequence_confirmation():
    image, layout, rows = fixture()
    rows[0]['item_number_raw'] = ''
    result, report = recover_number_cells(
        image, rows, layout, lambda images: [('1', .49), ('и', .12)])
    assert report['attempts'][0]['status'] == 'recognized_dominant_view'
    assert result[0]['item_number_recovered'] == '1'


def test_dominant_digit_never_overrides_sequence_conflict():
    image, layout, rows = fixture()
    rows[1]['item_number_raw'] = ''
    result, report = recover_number_cells(
        image, rows, layout, lambda images: [('1', .49), ('и', .12)])
    assert report['attempts'][0]['status'] == 'sequence_conflict_after_ocr'
    assert 'item_number_recovered' not in result[1]

