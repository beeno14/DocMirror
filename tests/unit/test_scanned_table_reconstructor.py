from dataclasses import replace

import pytest

from docmirror.input.canonical import assemble_parse_result
from docmirror.input.extraction.scanned_table_reconstructor import (
    _block_to_token,
    _merged_cell_groups,
    _normalize_numeric_cell_punctuation,
    _normalize_single_cell_numeric_punctuation,
    _open_left_column_has_horizontal_dividers,
    _recover_arabic_ordinal_markers,
    _recover_cjk_ordinal_markers,
    _recover_dash_placeholder_cells,
    _recover_leading_cjk_section_labels,
    _recover_numeric_ocr_tokens,
    _recover_ordinal_cells,
    _recover_sparse_vertical_label_groups,
    _recover_weak_text_tokens,
    _split_tokens_at_supported_column_boundaries,
    _Token,
    _tokens_in_reading_order,
    needs_high_precision_grid_review,
    reconstruct_scanned_bordered_tables,
)
from docmirror.models.entities.domain import Block, PageLayout
from docmirror.models.sealed import seal_parse_result
from docmirror.output.mirror_projector import project_mirror


def _ocr_block(text: str, x0: float, y0: float, x1: float, y1: float, idx: int) -> Block:
    block_id = f"ocr:p0001:{idx:04d}"
    return Block(
        block_id=block_id,
        block_type="text",
        bbox=(x0, y0, x1, y1),
        page=1,
        raw_content=text,
        attrs={
            "ocr_source": "rapidocr_pdf_page",
            "confidence": 0.95,
            "ocr_rotation": 90,
            "ocr_orientation_score": 42.5,
            "normalized_page_width": 842.0,
            "normalized_page_height": 595.0,
        },
        evidence_ids=(block_id,),
    )


def test_weak_text_token_recovery_is_structural_and_audited(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recover_empty_quote_dash_from_image",
        lambda *_args, **_kwargs: CellRecognition(
            "说明：符号写作“-”并保留",
            0.94,
            "empty_quote_dash_shape",
            raw_text="说明：符号写作“”并保留",
            audit={"repairs": [{"reason": "empty_quote_horizontal_dash_shape"}]},
        ),
    )
    token = _Token("说明：符号写作“”并保留", (10.0, 10.0, 120.0, 24.0), "ev:text", 0.98)

    recovered, events = _recover_weak_text_tokens(
        np.full((50, 150, 3), 255, dtype=np.uint8),
        [token],
        page_width=150,
        page_height=50,
        page_number=2,
        table_index=1,
    )

    assert recovered[0].text == "说明：符号写作“-”并保留"
    assert recovered[0].confidence == 0.94
    assert events[0]["reason_codes"] == ["empty_quote_horizontal_dash_shape"]
    assert events[0]["source_ref"] == "ev:text"


def test_high_precision_review_requires_a_dense_numeric_grid():
    grid_blocks = []
    for row in range(4):
        grid_blocks.append(_ocr_block(f"row-{row}", 20, 20 + row * 30, 100, 40 + row * 30, row * 3))
        grid_blocks.append(_ocr_block(f"{row + 1}.00", 200, 20 + row * 30, 250, 40 + row * 30, row * 3 + 1))
        grid_blocks.append(_ocr_block(f"{row + 11}.00", 280, 20 + row * 30, 330, 40 + row * 30, row * 3 + 2))
    paragraph_blocks = [
        _ocr_block(f"paragraph-{index}", 20, 20 + index * 30, 140, 40 + index * 30, index) for index in range(12)
    ]

    assert needs_high_precision_grid_review(grid_blocks) is True
    assert needs_high_precision_grid_review(paragraph_blocks) is False


def test_reconstruct_scanned_bordered_tables_preserves_multiple_physical_grids():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((600, 400, 3), 255, dtype=np.uint8)
    for y in (50, 100, 150, 200):
        cv2.line(image, (20, y), (380, y), (0, 0, 0), 2)
    for x in (20, 140, 260, 380):
        cv2.line(image, (x, 50), (x, 200), (0, 0, 0), 2)
    for y in (300, 350, 400):
        cv2.line(image, (30, y), (370, y), (0, 0, 0), 2)
    for x in (30, 200, 370):
        cv2.line(image, (x, 300), (x, 400), (0, 0, 0), 2)

    values = [
        ("A", 30, 60),
        ("B", 150, 60),
        ("C", 270, 60),
        ("D", 30, 110),
        ("E", 40, 310),
        ("F", 220, 310),
        ("G", 40, 360),
    ]
    blocks = [_ocr_block(text, x, y, x + 50, y + 20, index) for index, (text, x, y) in enumerate(values)]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=400,
        page_height=600,
    )

    assert len(tables) == 2
    assert tables[0].attrs["extraction_layer"] == "scanned_image_line_grid"
    assert tables[0].attrs["preserve_headers"] is False
    assert tables[0].raw_content[0] == ["A", "B", "C"]
    assert tables[1].raw_content[0] == ["E", "F"]
    assert tables[0].attrs["geometry"]["cell_bboxes"][0][0]
    assert tables[0].attrs["geometry"]["cell_evidence_ids"][0][0]


def test_reconstruct_scanned_bordered_table_restores_open_outer_columns():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((260, 400, 3), 255, dtype=np.uint8)
    for y in (30, 100, 170):
        cv2.line(image, (20, y), (380, y), (0, 0, 0), 2)
    # Financial-note tables commonly omit their left and right vertical rules.
    for x in (140, 260):
        cv2.line(image, (x, 30), (x, 170), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 35, 50, 95, 70, 0),
        _ocr_block("年末余额", 155, 50, 225, 70, 1),
        _ocr_block("年初余额", 275, 50, 345, 70, 2),
        _ocr_block("存货", 35, 120, 95, 140, 3),
        _ocr_block("10.00", 165, 120, 215, 140, 4),
        _ocr_block("9.00", 285, 120, 335, 140, 5),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=400,
        page_height=260,
    )

    assert len(tables) == 1
    assert tables[0].raw_content == [
        ["项目", "年末余额", "年初余额"],
        ["存货", "10.00", "9.00"],
    ]


def test_reconstruct_scanned_bordered_table_includes_unruled_left_label_column():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((260, 400, 3), 255, dtype=np.uint8)
    for y in (30, 100, 170):
        cv2.line(image, (140, y), (380, y), (0, 0, 0), 2)
    for x in (140, 260, 380):
        cv2.line(image, (x, 30), (x, 170), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 80, 50, 130, 70, 0),
        _ocr_block("年末余额", 155, 50, 225, 70, 1),
        _ocr_block("年初余额", 275, 50, 345, 70, 2),
        _ocr_block("存货", 80, 120, 130, 140, 3),
        _ocr_block("10.00", 165, 120, 215, 140, 4),
        _ocr_block("9.00", 285, 120, 335, 140, 5),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=400,
        page_height=260,
    )

    assert len(tables) == 1
    assert tables[0].raw_content == [
        ["项目", "年末余额", "年初余额"],
        ["存货", "10.00", "9.00"],
    ]


def test_open_left_group_rules_allow_source_driven_vertical_cell_spans():
    np = pytest.importorskip("numpy")
    horizontal = np.zeros((181, 321), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    x_lines = [20, 100, 210, 320]
    y_lines = [20, 60, 100, 140, 180]
    for y in y_lines:
        horizontal[max(0, y - 1) : y + 2, 100:320] = 255
    horizontal[99:102, 20:100] = 255
    for x in x_lines[1:]:
        vertical[20:180, max(0, x - 1) : x + 2] = 255

    assert _open_left_column_has_horizontal_dividers(
        horizontal,
        x0=20,
        original_x0=100,
        y_lines=y_lines,
    )

    groups, _diagnostics = _merged_cell_groups(horizontal, vertical, x_lines, y_lines)

    assert {(0, 0), (1, 0)} in groups
    assert {(2, 0), (3, 0)} in groups


def test_sparse_first_column_label_is_anchored_to_numeric_detail_run():
    groups = [{(row, column)} for row in range(4) for column in range(3)]
    tokens = [_Token("分组", (5, 25, 15, 35), "label", 0.95)]
    for row in range(4):
        y0 = row * 10 + 1
        tokens.extend(
            [
                _Token(f"{row}.00", (25, y0, 35, y0 + 5), f"n{row}:1", 0.95),
                _Token(f"{row}.00", (45, y0, 55, y0 + 5), f"n{row}:2", 0.95),
            ]
        )

    recovered = _recover_sparse_vertical_label_groups(
        groups,
        tokens,
        x_lines=[0, 20, 40, 60],
        y_lines=[0, 10, 20, 30, 40],
        sx=1.0,
        sy=1.0,
    )

    assert {(0, 0), (1, 0), (2, 0), (3, 0)} in recovered


def test_reconstruct_scanned_bordered_table_records_merged_cell_span():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200):
        cv2.line(image, (20, y), (340, y), (0, 0, 0), 2)
    for x in (20, 180, 340):
        cv2.line(image, (x, 20), (x, 200), (0, 0, 0), 2)
    # Remove the first-row middle divider so the first cell spans two columns.
    cv2.line(image, (180, 23), (180, 77), (255, 255, 255), 7)
    blocks = [
        _ocr_block("merged", 40, 40, 150, 60, 0),
        _ocr_block("r2c1", 40, 100, 100, 120, 1),
        _ocr_block("r2c2", 210, 100, 270, 120, 2),
        _ocr_block("r3c1", 40, 160, 100, 180, 3),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=360,
        page_height=240,
    )

    assert len(tables) == 1
    spans = tables[0].attrs["geometry"]["cell_spans"]
    assert any(span["row"] == 0 and span["col"] == 0 and span["col_span"] == 2 for span in spans)
    result = assemble_parse_result((PageLayout(page_number=1, blocks=tuple(tables)),), {}, "")
    statuses = [cell.geometry_status for row in result.pages[0].tables[0].rows for cell in row.cells]
    assert set(statuses) <= {"exact", "derived"}
    mirror = project_mirror(seal_parse_result(result))
    grid = next(block for block in mirror["blocks"] if block["type"] == "table")["content"]["grid"]
    assert any(cell["row"] == 0 and cell["col"] == 0 and cell["col_span"] == 2 for cell in grid["cells"])


def test_reconstruct_scanned_bordered_table_splits_numeric_tokens_across_row_bands():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200):
        cv2.line(image, (20, y), (340, y), (0, 0, 0), 2)
    for x in (20, 180, 340):
        cv2.line(image, (x, 20), (x, 200), (0, 0, 0), 2)
    # The copied scan loses one amount-column divider, but the OCR tokens
    # still occupy distinct row bands.
    cv2.line(image, (183, 140), (337, 140), (255, 255, 255), 7)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("金额", 210, 40, 270, 60, 1),
        _ocr_block("存货", 40, 100, 100, 120, 2),
        _ocr_block("10.00", 210, 100, 270, 120, 3),
        _ocr_block("合计", 40, 160, 100, 180, 4),
        _ocr_block("20.00", 210, 160, 270, 180, 5),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=360,
        page_height=240,
    )

    assert tables[0].raw_content == [
        ["项目", "金额"],
        ["存货", "10.00"],
        ["合计", "20.00"],
    ]
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_vertical_merge_count"] == 1


def test_reconstruct_scanned_bordered_table_splits_token_at_supported_column_boundary():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 420, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (400, y), (0, 0, 0), 2)
    for x in (20, 140, 260, 400):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("年初余额", 165, 40, 235, 60, 1),
        _ocr_block("负债项目", 290, 40, 360, 60, 2),
        _ocr_block("A", 40, 100, 80, 120, 3),
        _ocr_block("100.00", 170, 100, 230, 120, 4),
        _ocr_block("应付账款", 290, 100, 360, 120, 5),
        _ocr_block("B", 40, 160, 80, 180, 6),
        _ocr_block("90.00", 170, 160, 230, 180, 7),
        _ocr_block("短期借款", 290, 160, 360, 180, 8),
        _ocr_block("C", 40, 220, 80, 240, 9),
        _ocr_block("80.00", 170, 220, 230, 240, 10),
        _ocr_block("实收资本", 290, 220, 360, 240, 11),
        _ocr_block("合计", 40, 280, 80, 300, 12),
        _ocr_block("70.00所有者权益合计", 190, 280, 340, 300, 13),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=420,
        page_height=340,
    )

    assert tables[0].raw_content[-1] == ["合计", "70.00", "所有者权益合计"]


def test_split_token_at_supported_boundary_recovers_two_concatenated_amounts():
    tokens = [
        *[
            _Token(value, bbox, f"peer:{index}", 0.95)
            for index, (value, bbox) in enumerate(
                (
                    ("391,401.55", (110.0, 20.0, 175.0, 35.0)),
                    ("302,161.68", (210.0, 20.0, 275.0, 35.0)),
                    ("75,665.60", (110.0, 40.0, 175.0, 55.0)),
                    ("210,073.61", (210.0, 40.0, 275.0, 55.0)),
                    ("467,067.15", (110.0, 60.0, 175.0, 75.0)),
                    ("285,212.48", (210.0, 60.0, 275.0, 75.0)),
                )
            )
        ],
        _Token("512,235.29979,302.44", (135.0, 80.0, 265.0, 95.0), "joined", 0.95),
    ]

    recovered = _split_tokens_at_supported_column_boundaries(
        tokens,
        x_lines=[0, 100, 200, 300],
        sx=1.0,
    )

    joined = [token for token in recovered if token.evidence_id == "joined"]
    assert [token.text for token in joined] == ["512,235.29", "979,302.44"]
    assert [token.bbox for token in joined] == [(135.0, 80.0, 200.0, 95.0), (200.0, 80.0, 265.0, 95.0)]


def test_split_token_at_supported_boundary_ignores_multilevel_header_noise():
    tokens = [
        _Token("坏账准备", (110.0, 10.0, 175.0, 25.0), "header:left", 0.95),
        _Token("账面价值", (210.0, 10.0, 275.0, 25.0), "header:right", 0.95),
        _Token("6.48", (175.0, 30.0, 205.0, 45.0), "left:1", 0.95),
        _Token("51,115,019.53", (210.0, 30.0, 285.0, 45.0), "right:1", 0.95),
        _Token("12.38", (175.0, 50.0, 205.0, 65.0), "left:2", 0.95),
        _Token("39,921,236.27", (210.0, 50.0, 285.0, 65.0), "right:2", 0.95),
        _Token("6.4851,115,019.53", (165.0, 70.0, 265.0, 85.0), "joined", 0.95),
    ]

    recovered = _split_tokens_at_supported_column_boundaries(
        tokens,
        x_lines=[0, 100, 200, 300],
        sx=1.0,
    )

    joined = [token for token in recovered if token.evidence_id == "joined"]
    assert [token.text for token in joined] == ["6.48", "51,115,019.53"]


def test_reconstruct_scanned_bordered_table_splits_amount_and_dash_across_missing_row_rule():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((300, 360, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260):
        cv2.line(image, (20, y), (340, y), (0, 0, 0), 2)
    for x in (20, 180, 340):
        cv2.line(image, (x, 20), (x, 260), (0, 0, 0), 2)
    cv2.line(image, (183, 140), (337, 140), (255, 255, 255), 7)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("期初余额", 210, 40, 290, 60, 1),
        _ocr_block("(4)住房公积金", 40, 100, 150, 120, 2),
        _ocr_block("8,553.00", 220, 100, 285, 120, 3),
        _ocr_block("(5)工会经费和职工教育经费", 35, 160, 170, 180, 4),
        _ocr_block("一", 245, 160, 255, 180, 5),
        _ocr_block("合计", 40, 220, 100, 240, 6),
        _ocr_block("189,687.98", 215, 220, 290, 240, 7),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=360,
        page_height=300,
    )

    assert tables[0].raw_content[1][1] == "8,553.00"
    assert tables[0].raw_content[2][0] == "(5)工会经费和职工教育经费"
    assert tables[0].raw_content[2][1] in {"一", "—"}
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_vertical_merge_count"] == 1


def test_reconstruct_scanned_bordered_table_restores_fully_missing_row_rule_from_token_baselines():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((300, 360, 3), 255, dtype=np.uint8)
    for y in (20, 80, 200, 260):
        cv2.line(image, (20, y), (340, y), (0, 0, 0), 2)
    for x in (20, 180, 340):
        cv2.line(image, (x, 20), (x, 260), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("期初余额", 210, 40, 290, 60, 1),
        _ocr_block("(4)住房公积金", 40, 100, 150, 120, 2),
        _ocr_block("8,553.00", 220, 100, 285, 120, 3),
        _ocr_block("(5)工会经费和职工教育经费", 35, 160, 170, 180, 4),
        _ocr_block("一", 245, 160, 255, 180, 5),
        _ocr_block("合计", 40, 220, 100, 240, 6),
        _ocr_block("189,687.98", 215, 220, 290, 240, 7),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=360,
        page_height=300,
    )

    assert tables[0].raw_content == [
        ["项目", "期初余额"],
        ["(4)住房公积金", "8,553.00"],
        ["(5)工会经费和职工教育经费", "一"],
        ["合计", "189,687.98"],
    ]


def test_reconstruct_scanned_bordered_table_restores_missing_column_from_repeated_token_lanes():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    for x in (20, 180, 380, 500):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("附注", 200, 40, 235, 60, 1),
        _ocr_block("期末余额", 290, 40, 355, 60, 2),
        _ocr_block("期初余额", 410, 40, 475, 60, 3),
    ]
    for index, (label, note, current, prior) in enumerate(
        (
            ("货币资金", "五、1", "1,160,635.89", "1,600,060.84"),
            ("应收账款", "五、2", "51,115,019.53", "39,921,236.27"),
            ("预付款项", "五、3", "48,382,867.35", "52,204,619.83"),
            ("其他应收款", "五、4", "23,830,411.99", "29,841,185.12"),
        ),
        start=1,
    ):
        y = 100 + (index - 1) * 60
        offset = 4 + (index - 1) * 4
        blocks.extend(
            (
                _ocr_block(label, 40, y, 120, y + 20, offset),
                _ocr_block(note, 200, y, 235, y + 20, offset + 1),
                _ocr_block(current, 290, y, 365, y + 20, offset + 2),
                _ocr_block(prior, 410, y, 485, y + 20, offset + 3),
            )
        )

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=340,
    )

    assert tables[0].raw_content[0] == ["项目", "附注", "期末余额", "期初余额"]
    assert tables[0].raw_content[2] == ["应收账款", "五、2", "51,115,019.53", "39,921,236.27"]
    diagnostics = tables[0].attrs["geometry"]["merge_diagnostics"]
    assert diagnostics["token_recovered_column_line_count"] == 1


def test_reconstruct_scanned_bordered_table_restores_local_divider_for_joined_amounts():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    for x in (20, 180, 340, 500):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    cv2.line(image, (260, 203), (260, 257), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("账面余额", 210, 40, 290, 60, 1),
        _ocr_block("坏账准备", 370, 40, 450, 60, 2),
        _ocr_block("按组合计提", 40, 100, 120, 120, 3),
        _ocr_block("371,611.90", 200, 100, 250, 120, 4),
        _ocr_block("244,210.02", 370, 100, 440, 120, 5),
        _ocr_block("风险组合", 40, 160, 120, 180, 6),
        _ocr_block("371,611.90", 200, 160, 250, 180, 7),
        _ocr_block("244,210.02", 370, 160, 440, 180, 8),
        _ocr_block("合计", 40, 220, 100, 240, 9),
        _ocr_block("371,611.901,470,933.60", 195, 220, 325, 240, 10),
        _ocr_block("1,598,335.48", 370, 220, 455, 240, 11),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=340,
    )

    assert tables[0].raw_content[3] == ["合计", "371,611.90", "1,470,933.60", "1,598,335.48"]


def test_reconstruct_scanned_bordered_table_extends_open_right_numeric_column():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (340, y), (0, 0, 0), 2)
    for x in (20, 180, 340):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("未分配利润", 210, 40, 300, 60, 1),
        _ocr_block("所有者权益合计", 350, 40, 415, 60, 2),
        _ocr_block("上年期末余额", 40, 100, 120, 120, 3),
        _ocr_block("9,039,934.16", 210, 100, 300, 120, 4),
        _ocr_block("51,134,564.17", 350, 100, 415, 120, 5),
        _ocr_block("本年期初余额", 40, 160, 120, 180, 6),
        _ocr_block("9,039,934.16", 210, 160, 300, 180, 7),
        _ocr_block("51,134,564.17", 350, 160, 415, 180, 8),
        _ocr_block("本期增加", 40, 220, 120, 240, 9),
        _ocr_block("1,806,359.88", 210, 220, 300, 240, 10),
        _ocr_block("2,007,066.53", 350, 220, 415, 240, 11),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=340,
    )

    assert tables[0].raw_content[0] == ["项目", "未分配利润", "所有者权益合计"]
    assert tables[0].raw_content[1] == ["上年期末余额", "9,039,934.16", "51,134,564.17"]


def test_reconstruct_scanned_bordered_table_restores_missing_header_data_row_rule():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 200, 260, 320):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    for x in (20, 180, 280, 380, 500):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    blocks = [
        _ocr_block("本期金额", 210, 40, 290, 60, 0),
        _ocr_block("项目", 40, 100, 100, 120, 1),
        _ocr_block("实收资本", 200, 100, 260, 120, 2),
        _ocr_block("资本公积", 300, 100, 360, 120, 3),
        _ocr_block("所有者权益合计", 400, 100, 480, 120, 4),
        _ocr_block("上年期末余额", 40, 160, 130, 180, 5),
        _ocr_block("27,366,313.00", 200, 160, 265, 180, 6),
        _ocr_block("12,144,181.02", 300, 160, 365, 180, 7),
        _ocr_block("51,134,564.17", 400, 160, 485, 180, 8),
        _ocr_block("本年期初余额", 40, 220, 130, 240, 9),
        _ocr_block("27,366,313.00", 200, 220, 265, 240, 10),
        _ocr_block("12,144,181.02", 300, 220, 365, 240, 11),
        _ocr_block("51,134,564.17", 400, 220, 485, 240, 12),
        _ocr_block("本期期末余额", 40, 280, 130, 300, 13),
        _ocr_block("27,366,313.00", 200, 280, 265, 300, 14),
        _ocr_block("12,144,181.02", 300, 280, 365, 300, 15),
        _ocr_block("51,134,564.17", 400, 280, 485, 300, 16),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=340,
    )

    assert tables[0].raw_content == [
        ["", "本期金额", "", ""],
        ["项目", "实收资本", "资本公积", "所有者权益合计"],
        ["上年期末余额", "27,366,313.00", "12,144,181.02", "51,134,564.17"],
        ["本年期初余额", "27,366,313.00", "12,144,181.02", "51,134,564.17"],
        ["本期期末余额", "27,366,313.00", "12,144,181.02", "51,134,564.17"],
    ]


def test_reconstruct_scanned_bordered_table_restores_numbered_label_rows_with_missing_ordinals():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((420, 420, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 380):
        cv2.line(image, (20, y), (400, y), (0, 0, 0), 2)
    for x in (20, 260, 400):
        cv2.line(image, (x, 20), (x, 380), (0, 0, 0), 2)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("期末余额", 290, 40, 360, 60, 1),
        _ocr_block("1.提取盈余公积", 40, 100, 180, 120, 2),
        _ocr_block("100.00", 300, 100, 350, 120, 3),
        _ocr_block("2.对所有者的分配", 40, 160, 180, 180, 4),
        _ocr_block("90.00", 300, 160, 350, 180, 5),
        _ocr_block("3.盈余公积弥补亏损", 40, 220, 190, 240, 6),
        _ocr_block("设定受益变动计划额结转留存收益", 40, 260, 220, 280, 7),
        _ocr_block("其他综合收益结转留存收益", 40, 300, 210, 320, 8),
        _ocr_block("6.其他", 40, 340, 100, 360, 9),
        _ocr_block("备注", 300, 340, 350, 360, 10),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=420,
        page_height=420,
    )

    assert [row[0] for row in tables[0].raw_content[-4:]] == [
        "3.盈余公积弥补亏损",
        "设定受益变动计划额结转留存收益",
        "其他综合收益结转留存收益",
        "6.其他",
    ]


def test_reconstruct_scanned_bordered_table_splits_header_data_rectangle_with_missing_rules():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((380, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320, 360):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    for x in (20, 120, 220, 360, 500):
        cv2.line(image, (x, 20), (x, 360), (0, 0, 0), 2)
    cv2.line(image, (220, 140), (500, 140), (255, 255, 255), 7)
    cv2.line(image, (360, 80), (360, 200), (255, 255, 255), 7)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("期末余额", 240, 100, 330, 120, 1),
        _ocr_block("期初余额", 390, 100, 470, 120, 2),
        _ocr_block("货币资金", 30, 160, 105, 180, 3),
        _ocr_block("100.00", 250, 160, 315, 180, 4),
        _ocr_block("90.00", 400, 160, 465, 180, 5),
        _ocr_block("应收账款", 30, 220, 105, 240, 6),
        _ocr_block("80.00", 250, 220, 315, 240, 7),
        _ocr_block("70.00", 400, 220, 465, 240, 8),
        _ocr_block("存货", 30, 280, 105, 300, 9),
        _ocr_block("60.00", 250, 280, 315, 300, 10),
        _ocr_block("50.00", 400, 280, 465, 300, 11),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=380,
    )

    assert any(row[2:] == ["期末余额", "期初余额"] for row in tables[0].raw_content), tables[0].raw_content
    assert any("100.00" in row and "90.00" in row for row in tables[0].raw_content)
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_rectangular_merge_count"] >= 1


def test_reconstruct_scanned_bordered_table_uses_full_table_numeric_profiles_for_merged_amounts():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((500, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320, 380, 440, 480):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    for x in (20, 140, 260, 380, 500):
        cv2.line(image, (x, 20), (x, 480), (0, 0, 0), 2)
    cv2.line(image, (380, 383), (380, 437), (255, 255, 255), 7)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("运输设备", 165, 40, 235, 60, 1),
        _ocr_block("电子设备", 285, 40, 355, 60, 2),
        _ocr_block("合计", 420, 40, 470, 60, 3),
    ]
    index = 4
    for row, values in enumerate(
        (
            ("期初余额", "391,401.55", "302,161.68", "693,563.23"),
            ("本期增加", "75,665.60", "210,073.61", "285,739.21"),
            ("本期计提", "75,665.60", "210,073.61", "285,739.21"),
        ),
        start=1,
    ):
        for column, value in enumerate(values):
            x0 = 40 + column * 120
            blocks.append(_ocr_block(value, x0, 100 + (row - 1) * 60, x0 + 75, 120 + (row - 1) * 60, index))
            index += 1
    blocks.extend(
        [
            _ocr_block("本期减少", 40, 280, 115, 300, index),
            _ocr_block("处置或报废", 40, 340, 115, 360, index + 1),
            _ocr_block("期末余额", 40, 400, 115, 420, index + 2),
            _ocr_block("467,067.15", 165, 400, 235, 420, index + 3),
            _ocr_block("512,235.29", 285, 400, 355, 420, index + 4),
            _ocr_block("979,302.44", 405, 400, 480, 420, index + 5),
        ]
    )

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=500,
    )

    closing_row = next(row for row in tables[0].raw_content if row[0] == "期末余额")
    assert closing_row == ["期末余额", "467,067.15", "512,235.29", "979,302.44"]


def test_reconstruct_scanned_bordered_table_reanchors_sparse_merged_amount_to_aligned_total_row():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((380, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320, 360):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    for x in (20, 180, 340, 500):
        cv2.line(image, (x, 20), (x, 360), (0, 0, 0), 2)
    for y in (143, 203, 263):
        cv2.line(image, (183, y), (337, y), (255, 255, 255), 7)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("期初余额", 210, 40, 290, 60, 1),
        _ocr_block("本期增加", 370, 40, 450, 60, 2),
        _ocr_block("(5)工会经费", 40, 100, 150, 120, 3),
        _ocr_block("(6)短期带薪缺勤", 40, 160, 150, 180, 4),
        _ocr_block("(7)短期利润分享计划", 40, 220, 160, 240, 5),
        _ocr_block("合计", 40, 280, 100, 300, 6),
        _ocr_block("189,687.98", 215, 280, 300, 300, 7),
        _ocr_block("2,662,445.69", 370, 280, 470, 300, 8),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=380,
    )

    assert tables[0].raw_content[4] == ["合计", "189,687.98", "2,662,445.69"]
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_reanchored_vertical_merge_count"] == 1


def test_reconstruct_scanned_bordered_table_splits_label_and_ordinal_header_by_column_evidence():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 420, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (400, y), (0, 0, 0), 2)
    for x in (20, 260, 400):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    cv2.line(image, (200, 80), (200, 320), (0, 0, 0), 2)
    blocks = [
        _ocr_block("资产次行", 90, 40, 225, 60, 0),
        _ocr_block("期末余额", 300, 40, 370, 60, 1),
        _ocr_block("货币资金", 60, 100, 130, 120, 2),
        _ocr_block("1", 220, 100, 235, 120, 3),
        _ocr_block("100.00", 300, 100, 360, 120, 4),
        _ocr_block("应收账款", 60, 160, 130, 180, 5),
        _ocr_block("2", 220, 160, 235, 180, 6),
        _ocr_block("90.00", 300, 160, 360, 180, 7),
        _ocr_block("存货", 60, 220, 110, 240, 8),
        _ocr_block("3", 220, 220, 235, 240, 9),
        _ocr_block("80.00", 300, 220, 360, 240, 10),
        _ocr_block("资产合计", 60, 280, 130, 300, 11),
        _ocr_block("4", 220, 280, 235, 300, 12),
        _ocr_block("270.00", 300, 280, 360, 300, 13),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=420,
        page_height=340,
    )

    assert tables[0].raw_content[0] == ["资产", "行次", "期末余额"]
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_horizontal_merge_count"] == 1


def test_reconstruct_scanned_bordered_table_splits_vertical_ordinal_header_tokens():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 420, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (400, y), (0, 0, 0), 2)
    for x in (20, 260, 400):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    cv2.line(image, (200, 80), (200, 320), (0, 0, 0), 2)
    blocks = [
        _ocr_block("资产", 80, 45, 120, 65, 0),
        _ocr_block("行", 220, 35, 235, 52, 1),
        _ocr_block("次", 220, 50, 235, 67, 2),
        _ocr_block("期末余额", 300, 40, 370, 60, 3),
        *[
            block
            for row, (label, line, amount) in enumerate(
                (("货币资金", "1", "100.00"), ("应收账款", "2", "90.00"), ("存货", "3", "80.00")),
                start=1,
            )
            for block in (
                _ocr_block(label, 60, 40 + row * 60, 130, 60 + row * 60, row * 3 + 1),
                _ocr_block(line, 220, 40 + row * 60, 235, 60 + row * 60, row * 3 + 2),
                _ocr_block(amount, 300, 40 + row * 60, 360, 60 + row * 60, row * 3 + 3),
            )
        ],
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=420,
        page_height=340,
    )

    assert tables[0].raw_content[0] == ["资产", "行次", "期末余额"]


def test_reconstruct_scanned_bordered_table_keeps_true_merged_label_header() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 420, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (400, y), (0, 0, 0), 2)
    for x in (20, 260, 400):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    cv2.line(image, (200, 80), (200, 320), (0, 0, 0), 2)
    blocks = [
        _ocr_block("资产项目", 90, 40, 225, 60, 0),
        _ocr_block("期末余额", 300, 40, 370, 60, 1),
        *[
            block
            for row, (label, line, amount) in enumerate(
                (
                    ("货币资金", "1", "100.00"),
                    ("应收账款", "2", "90.00"),
                    ("存货", "3", "80.00"),
                    ("合计", "4", "270.00"),
                ),
                start=1,
            )
            for block in (
                _ocr_block(label, 60, 40 + row * 60, 130, 60 + row * 60, row * 3),
                _ocr_block(line, 220, 40 + row * 60, 235, 60 + row * 60, row * 3 + 1),
                _ocr_block(amount, 300, 40 + row * 60, 360, 60 + row * 60, row * 3 + 2),
            )
        ],
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=420,
        page_height=340,
    )

    assert tables[0].raw_content[0] == ["资产项目", "", "期末余额"]
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_horizontal_merge_count"] == 0


def test_reconstruct_scanned_bordered_table_restores_numeric_cells_when_row_dividers_are_faint():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((340, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200, 260, 320):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    x_starts = (20, 140, 220, 300, 380, 460)
    for x in (*x_starts, 500):
        cv2.line(image, (x, 20), (x, 320), (0, 0, 0), 2)
    for x in (300, 380, 460):
        cv2.line(image, (x, 203), (x, 257), (255, 255, 255), 7)
    cv2.line(image, (140, 263), (140, 317), (255, 255, 255), 7)
    blocks = []
    values = [
        ["项目", "栏次", "本月数", "本年累计", "即征即退", "累计"],
        ["期初税额", "36", "0.00", "0.00", "—", "—"],
        ["本期税额", "37", "0.00", "0.00", "—", "—"],
        ["期末税额", "38=16-22-36-37", "0.00", "0.00", '"', "二"],
        ["即征即退实际退税额", "39", "0.00", "0.00", "0.00", "0.00"],
    ]
    index = 0
    for row, values_in_row in enumerate(values):
        for col, value in enumerate(values_in_row):
            x0 = x_starts[col] + 8
            y0 = 20 + row * 60 + 20
            blocks.append(_ocr_block(value, x0, y0, x0 + 40, y0 + 20, index))
            index += 1

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=340,
    )

    assert tables[0].raw_content[-2] == ["期末税额", "38=16-22-36-37", "0.00", "0.00", '"', "二"]
    assert tables[0].raw_content[-1] == ["即征即退实际退税额", "39", "0.00", "0.00", "0.00", "0.00"]
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_horizontal_merge_count"] == 2


def test_reconstruct_scanned_bordered_table_splits_label_tokens_with_aligned_amount_rows():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200):
        cv2.line(image, (20, y), (340, y), (0, 0, 0), 2)
    for x in (20, 180, 340):
        cv2.line(image, (x, 20), (x, 200), (0, 0, 0), 2)
    # The label-column divider is missing, while the amount column still
    # proves that two physical body rows exist at the same y bands.
    cv2.line(image, (23, 140), (177, 140), (255, 255, 255), 7)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("金额", 210, 40, 270, 60, 1),
        _ocr_block("甲公司", 40, 100, 100, 120, 2),
        _ocr_block("10.00", 210, 100, 270, 120, 3),
        _ocr_block("乙公司", 40, 160, 100, 180, 4),
        _ocr_block("20.00", 210, 160, 270, 180, 5),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=360,
        page_height=240,
    )

    assert tables[0].raw_content == [
        ["项目", "金额"],
        ["甲公司", "10.00"],
        ["乙公司", "20.00"],
    ]
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_vertical_merge_count"] == 1


def test_reconstruct_scanned_bordered_table_splits_subheader_from_first_amount():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((240, 520, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200):
        cv2.line(image, (20, y), (500, y), (0, 0, 0), 2)
    for x in (20, 180, 340, 500):
        cv2.line(image, (x, 20), (x, 200), (0, 0, 0), 2)
    # Only the first amount-column divider is lost. Its second-level header
    # and first body value must still land in their respective row bands.
    cv2.line(image, (183, 140), (337, 140), (255, 255, 255), 7)
    blocks = [
        _ocr_block("项目", 40, 40, 100, 60, 0),
        _ocr_block("年末余额", 370, 40, 450, 60, 1),
        _ocr_block("账面余额", 210, 100, 290, 120, 2),
        _ocr_block("账面价值", 370, 100, 450, 120, 3),
        _ocr_block("原材料", 40, 160, 100, 180, 4),
        _ocr_block("1,297,676.15", 205, 160, 310, 180, 5),
        _ocr_block("1,297,676.15", 365, 160, 470, 180, 6),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=520,
        page_height=240,
    )

    assert tables[0].raw_content == [
        ["项目", "", "年末余额"],
        ["", "账面余额", "账面价值"],
        ["原材料", "1,297,676.15", "1,297,676.15"],
    ]
    assert tables[0].attrs["geometry"]["merge_diagnostics"]["token_split_vertical_merge_count"] == 1


def test_reconstruct_scanned_bordered_tables_rejects_single_column_notice_frame():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (380, 280), (0, 0, 0), 2)
    for y in (90, 160, 230):
        cv2.line(image, (20, y), (380, y), (0, 0, 0), 2)
    blocks = [_ocr_block("notice", 40, 40, 120, 60, 0)]

    assert (
        reconstruct_scanned_bordered_tables(
            image,
            blocks,
            page_number=1,
            page_width=400,
            page_height=300,
        )
        == []
    )


def test_reconstruct_scanned_bordered_table_rejects_l_shaped_merge_component():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    for y in (20, 80, 140, 200):
        cv2.line(image, (20, y), (340, y), (0, 0, 0), 2)
    for x in (20, 180, 340):
        cv2.line(image, (x, 20), (x, 200), (0, 0, 0), 2)
    # Missing top vertical segment joins (0,0)-(0,1); missing left
    # horizontal segment joins (0,0)-(1,0), producing an L component.
    cv2.line(image, (180, 23), (180, 77), (255, 255, 255), 7)
    cv2.line(image, (23, 80), (177, 80), (255, 255, 255), 7)
    blocks = [
        _ocr_block("a", 40, 40, 80, 60, 0),
        _ocr_block("b", 210, 40, 250, 60, 1),
        _ocr_block("c", 40, 100, 80, 120, 2),
        _ocr_block("d", 210, 100, 250, 120, 3),
        _ocr_block("e", 40, 160, 80, 180, 4),
    ]

    tables = reconstruct_scanned_bordered_tables(
        image,
        blocks,
        page_number=1,
        page_width=360,
        page_height=240,
    )

    assert len(tables) == 1
    geometry = tables[0].attrs["geometry"]
    assert geometry["merge_diagnostics"]["rejected_non_rectangular_count"] >= 1
    assert geometry["cell_spans"] == []


def test_numeric_token_recovery_adopts_only_high_confidence_consensus(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    def recognize(*_args, **_kwargs):
        return CellRecognition(
            "12,540,311.58",
            0.96,
            "cell_crop_consensus",
            audit={"consensus_count": 3},
        )

    monkeypatch.setattr(cell_recognition, "recognize_micro_cell_from_image", recognize)
    tokens = [
        _Token("经营活动产生的现金流量", (10, 10, 150, 20), "ev:label", 0.99),
        _Token("12,540,311.", (200, 10, 280, 20), "ev:amount", 0.86),
        _Token("1.00", (200, 30, 240, 40), "ev:other1", 0.99),
        _Token("2.00", (200, 50, 240, 60), "ev:other2", 0.99),
    ]

    recovered, events = _recover_numeric_ocr_tokens(
        np.full((100, 300, 3), 255, dtype=np.uint8),
        tokens,
        page_width=300,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[1].text == "12,540,311.58"
    assert events[0]["input_text"] == "12,540,311."
    assert events[0]["source_ref"] == "ev:amount"


def test_numeric_token_recovery_rechecks_impossible_decimal_punctuation(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: CellRecognition(
            "47,319,683.79",
            0.97,
            "cell_crop_consensus",
            audit={"consensus_count": 3},
        ),
    )
    tokens = [
        _Token("所有者权益合计", (10, 20, 120, 30), "ev:label", 0.99),
        _Token("47.319,683.79", (200, 20, 280, 30), "ev:amount", 0.91),
        _Token("1.00", (200, 40, 240, 50), "ev:other1", 0.99),
        _Token("2.00", (200, 60, 240, 70), "ev:other2", 0.99),
    ]

    recovered, events = _recover_numeric_ocr_tokens(
        np.full((100, 300, 3), 255, dtype=np.uint8),
        tokens,
        page_width=300,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[1].text == "47,319,683.79"
    event = next(event for event in events if event["source_ref"] == "ev:amount")
    assert event["reason_codes"] == ["numeric_cell_crop_consensus"]


def test_numeric_punctuation_structure_requires_one_cell_geometry():
    tokens = [
        _Token("51,134.564.17", (10, 10, 90, 20), "ev:single", 0.97),
        _Token("371,611.901,470,933.60", (70, 30, 130, 40), "ev:joined", 0.97),
    ]

    recovered, events = _normalize_single_cell_numeric_punctuation(
        tokens,
        x_lines=[0, 100, 200],
        sx=1.0,
        page_number=1,
        table_index=0,
    )

    assert recovered[0].text == "51,134,564.17"
    assert recovered[1].text == "371,611.901,470,933.60"
    assert [event["source_ref"] for event in events] == ["ev:single"]


def test_materialized_numeric_cell_punctuation_is_repaired_without_joining_two_values():
    raw = [
        ["本年期初余额", "51,134.564.17"],
        ["合计", "371,611.901,470,933.60"],
        ["上年期末余额", "51,134,564.17"],
        ["本期期末余额", "53,141,630.70"],
    ]
    bboxes = [
        [[0.0, 0.0, 100.0, 10.0], [100.0, 0.0, 200.0, 10.0]],
        [[0.0, 10.0, 100.0, 20.0], [70.0, 10.0, 130.0, 20.0]],
        [[0.0, 20.0, 100.0, 30.0], [100.0, 20.0, 200.0, 30.0]],
        [[0.0, 30.0, 100.0, 40.0], [100.0, 30.0, 200.0, 40.0]],
    ]
    evidence_ids = [[[], ["ev:amount"]], [[], ["ev:joined"]], [[], ["ev:prior"]], [[], ["ev:final"]]]
    confidences = [[None, 0.97] for _row in raw]

    recovered, events = _normalize_numeric_cell_punctuation(
        raw,
        cell_bboxes=bboxes,
        cell_evidence_ids=evidence_ids,
        cell_confidences=confidences,
        column_bands=[{"x0": 0.0, "x1": 100.0}, {"x0": 110.0, "x1": 190.0}],
        page_number=1,
        table_index=0,
    )

    assert recovered[0][1] == "51,134,564.17"
    assert recovered[1][1] == "371,611.901,470,933.60"
    assert events[0]["source_ref"] == "ev:amount"


def test_numeric_token_recovery_removes_ocr_trailing_bracket(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: CellRecognition(
            "0.00",
            0.98,
            "cell_crop_consensus",
            audit={"consensus_count": 3},
        ),
    )
    tokens = [
        _Token("增值税纳税申报表", (10, 0, 150, 10), "ev:title", 0.99),
        _Token("即征即退实际退税额", (10, 20, 150, 30), "ev:label", 0.99),
        _Token("0.00]", (200, 20, 240, 30), "ev:amount", 0.88),
        _Token("1.00", (200, 40, 240, 50), "ev:other1", 0.99),
        _Token("2.00", (200, 60, 240, 70), "ev:other2", 0.99),
    ]

    recovered, events = _recover_numeric_ocr_tokens(
        np.full((100, 300, 3), 255, dtype=np.uint8),
        tokens,
        page_width=300,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[2].text == "0.00"
    assert [event["source_ref"] for event in events] == ["ev:amount"]


def test_numeric_token_recovery_accepts_three_vote_completion_of_one_decimal(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: CellRecognition(
            "0.00",
            0.846,
            "cell_crop_consensus",
            audit={"consensus_count": 3},
        ),
    )
    tokens = [
        _Token("增值税纳税申报表", (10, 0, 150, 10), "ev:title", 0.99),
        _Token("期末未缴查补税额", (10, 20, 150, 30), "ev:label", 0.99),
        _Token("0.0", (200, 20, 240, 30), "ev:amount", 0.848),
        _Token("1.00", (200, 40, 240, 50), "ev:other1", 0.99),
        _Token("2.00", (200, 60, 240, 70), "ev:other2", 0.99),
    ]

    recovered, events = _recover_numeric_ocr_tokens(
        np.full((100, 300, 3), 255, dtype=np.uint8),
        tokens,
        page_width=300,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[2].text == "0.00"
    assert [event["source_ref"] for event in events] == ["ev:amount"]


def test_numeric_token_recovery_uses_full_cell_digit_lattice_for_truncated_decimals(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    calls = 0

    def recognize(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CellRecognition(
                "1,123,17.69",
                0.94,
                "cell_crop_consensus",
                audit={"consensus_count": 3},
            )
        return CellRecognition(
            "1,12317968",
            0.55,
            "cell_crop_consensus",
            audit={
                "consensus_count": 2,
                "votes": [
                    {"text": "1,12317968", "confidence": 0.55},
                    {"text": "1,12317968", "confidence": 0.53},
                ],
            },
        )

    monkeypatch.setattr(cell_recognition, "recognize_micro_cell_from_image", recognize)
    tokens = [
        _Token("增值税纳税申报表", (10, 0, 150, 10), "ev:title", 0.99),
        _Token("本期已缴税额", (10, 20, 150, 30), "ev:label", 0.99),
        _Token("1,123,179.9", (200, 20, 240, 30), "ev:amount", 0.847),
        _Token("1.00", (200, 40, 240, 50), "ev:other1", 0.99),
        _Token("2.00", (200, 60, 240, 70), "ev:other2", 0.99),
    ]

    recovered, events = _recover_numeric_ocr_tokens(
        np.full((100, 300, 3), 255, dtype=np.uint8),
        tokens,
        page_width=300,
        page_height=100,
        page_number=1,
        table_index=0,
        grid_geometry=([0, 180, 300], [0, 35, 70, 100], 1.0, 1.0),
    )

    assert recovered[2].text == "1,123,179.68"
    assert events[0]["reason_codes"] == ["numeric_cell_digit_lattice_consensus"]


def test_numeric_token_recovery_repairs_low_confidence_zero_digit_confusion(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: CellRecognition(
            "0.",
            0.74,
            "cell_crop_consensus",
            audit={
                "consensus_count": 4,
                "votes": [
                    {"text": "0.", "confidence": 0.74},
                    {"text": "0.", "confidence": 0.72},
                    {"text": "0.", "confidence": 0.70},
                    {"text": "0.0", "confidence": 0.64},
                ],
            },
        ),
    )
    tokens = [_Token("明细表", (10, 0, 100, 10), "ev:title", 0.99)]
    tokens.extend(
        _Token("1.00", (200, 20 + index * 5, 240, 24 + index * 5), f"ev:noise:{index}", 0.89) for index in range(40)
    )
    tokens.extend(
        [
            _Token("明细A", (10, 240, 100, 250), "ev:label", 0.99),
            _Token("0.09", (200, 240, 240, 250), "ev:critical", 0.85),
        ]
    )

    recovered, events = _recover_numeric_ocr_tokens(
        np.full((300, 300, 3), 255, dtype=np.uint8),
        tokens,
        page_width=300,
        page_height=300,
        page_number=1,
        table_index=0,
        max_repairs=1,
        grid_geometry=([0, 150, 300], [0, 230, 270, 300], 1.0, 1.0),
    )

    assert recovered[-1].text == "0.00"
    assert [event["source_ref"] for event in events] == ["ev:critical"]
    assert events[0]["reason_codes"] == ["numeric_zero_glyph_consensus"]


def test_ordinal_recovery_requires_crop_confirmation_and_sequence(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: CellRecognition(
            "1",
            0.99,
            "cell_crop_consensus",
            audit={"consensus_count": 2},
        ),
    )
    raw = [
        ["项目", "行次", "本年累计金额", "本月金额"],
        ["一、经营活动产生的现金流量", "", "", ""],
        ["销售商品收到的现金", "", "100.00", "10.00"],
        ["收到其他现金", "2", "200.00", "20.00"],
    ]
    cell_bboxes = [
        [[float(column * 10), float(row * 10), float((column + 1) * 10), float((row + 1) * 10)] for column in range(4)]
        for row in range(4)
    ]
    cell_evidence_ids = [[[] for _column in range(4)] for _row in range(4)]
    cell_confidences = [[None for _column in range(4)] for _row in range(4)]
    cell_geometry_status = [["derived" for _column in range(4)] for _row in range(4)]
    cell_geometry_loss_reason = [["empty_ocr_cell" for _column in range(4)] for _row in range(4)]

    recovered, events = _recover_ordinal_cells(
        np.full((100, 100, 3), 255, dtype=np.uint8),
        raw,
        cell_bboxes=cell_bboxes,
        cell_evidence_ids=cell_evidence_ids,
        cell_confidences=cell_confidences,
        cell_geometry_status=cell_geometry_status,
        cell_geometry_loss_reason=cell_geometry_loss_reason,
        page_width=100,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[2][1] == "1"
    assert recovered[1][1] == ""
    assert events[0]["reason_codes"] == ["ordinal_cell_crop_consensus"]
    assert cell_geometry_status[2][1] == "exact"
    assert cell_evidence_ids[2][1] == ["table:p1:t0:r2:c1"]


def test_ordinal_recovery_repairs_invalid_glyph_from_strong_row_lattice(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: CellRecognition("", 0.0, "cell_crop_consensus"),
    )
    raw = [
        ["项目", "栏次", "本月数"],
        ["销售额", "1", "10.00"],
        ["其中", "2", "0.00"],
        ["劳务销售额", "n", "0.00"],
        ["纳税检查调整", "4", "0.00"],
        ["简易征收", "5", "0.00"],
    ]
    cell_bboxes = [
        [[float(column * 10), float(row * 10), float((column + 1) * 10), float((row + 1) * 10)] for column in range(3)]
        for row in range(6)
    ]
    cell_evidence_ids = [[[] for _column in range(3)] for _row in range(6)]
    cell_confidences = [[None for _column in range(3)] for _row in range(6)]
    cell_geometry_status = [["derived" for _column in range(3)] for _row in range(6)]
    cell_geometry_loss_reason = [["empty_ocr_cell" for _column in range(3)] for _row in range(6)]

    recovered, events = _recover_ordinal_cells(
        np.full((100, 100, 3), 255, dtype=np.uint8),
        raw,
        cell_bboxes=cell_bboxes,
        cell_evidence_ids=cell_evidence_ids,
        cell_confidences=cell_confidences,
        cell_geometry_status=cell_geometry_status,
        cell_geometry_loss_reason=cell_geometry_loss_reason,
        page_width=100,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[3][1] == "3"
    assert events[0]["input_text"] == "n"
    assert events[0]["reason_codes"] == ["ordinal_sequence_lattice"]


def test_dash_placeholder_recovery_uses_numeric_column_profile_and_crop_shape(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    monkeypatch.setattr(
        cell_recognition,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: CellRecognition(
            "—",
            0.94,
            "cell_crop_consensus",
            audit={
                "consensus_count": 1,
                "votes": [{"variant": "glyph_shape_dash", "text": "—", "confidence": 0.94}],
            },
        ),
    )
    raw = [
        ["项目", "栏次", "一般项目", "", "", "即征即退项目", ""],
        ["", "", "本月数", "本年累计", "", "本月数", "本年累计"],
        ["按适用税率计税销售额", "1", "100.00", "一", "", "", "一"],
        ["应税货物销售额", "2", "0.00", "0.00", "", "0.00", "0.00"],
        ["应税劳务销售额", "3", "0.00", "0.00", "", "0.00", "0.00"],
        ["纳税检查调整的销售额", "4", "0.00", "0.00", "", "0.00", "0.00"],
    ]
    cell_bboxes = [
        [[float(column * 10), float(row * 10), float((column + 1) * 10), float((row + 1) * 10)] for column in range(7)]
        for row in range(len(raw))
    ]
    cell_evidence_ids = [[[] for _column in range(7)] for _row in raw]
    cell_confidences = [[None for _column in range(7)] for _row in raw]
    cell_geometry_status = [["derived" for _column in range(7)] for _row in raw]
    cell_geometry_loss_reason = [["empty_ocr_cell" for _column in range(7)] for _row in raw]

    recovered, events = _recover_dash_placeholder_cells(
        np.full((100, 100, 3), 255, dtype=np.uint8),
        raw,
        cell_bboxes=cell_bboxes,
        cell_evidence_ids=cell_evidence_ids,
        cell_confidences=cell_confidences,
        cell_geometry_status=cell_geometry_status,
        cell_geometry_loss_reason=cell_geometry_loss_reason,
        page_width=100,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[2][2] == "100.00"
    assert recovered[2][3] == "—"
    assert recovered[2][5:] == ["—", "—"]
    assert len(events) == 3
    assert {event["reason_codes"][0] for event in events} == {"numeric_placeholder_cell_crop_shape"}


def test_dash_placeholder_recovery_supports_headerless_three_row_continuations(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    reviewed_bboxes: list[tuple[float, float, float, float]] = []

    def recognize(_image, bbox, **_kwargs):
        reviewed_bboxes.append(bbox)
        return CellRecognition(
            "—",
            0.94,
            "cell_crop_consensus",
            audit={"votes": [{"variant": "glyph_shape_dash", "text": "—", "confidence": 0.94}]},
        )

    monkeypatch.setattr(cell_recognition, "recognize_micro_cell_from_image", recognize)
    raw = [
        ["城市维护建设税", "39", "7.00", "77.00", "一", ""],
        ["教育费附加", "40", "4.00", "46.00", "一", "一"],
        ["地方教育附加", "41", "3.00", "30.00", "—", "一"],
    ]
    cell_bboxes = [
        [[float(column * 10), float(row * 10), float((column + 1) * 10), float((row + 1) * 10)] for column in range(6)]
        for row in range(3)
    ]
    cell_bboxes[1][4] = [40.0, 10.0, 60.0, 20.0]
    cell_bboxes[1][5] = [40.0, 10.0, 60.0, 20.0]
    cell_bboxes[0][5] = None
    evidence_ids = [[[] for _column in range(6)] for _row in raw]
    confidences = [[None for _column in range(6)] for _row in raw]
    geometry_status = [["derived" for _column in range(6)] for _row in raw]
    loss_reasons = [["empty_ocr_cell" for _column in range(6)] for _row in raw]

    recovered, _events = _recover_dash_placeholder_cells(
        np.full((100, 100, 3), 255, dtype=np.uint8),
        raw,
        cell_bboxes=cell_bboxes,
        cell_evidence_ids=evidence_ids,
        cell_confidences=confidences,
        cell_geometry_status=geometry_status,
        cell_geometry_loss_reason=loss_reasons,
        page_width=100,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert [row[4:] for row in recovered] == [["—", "—"], ["—", "—"], ["—", "—"]]
    assert (40.0, 10.0, 50.0, 20.0) in reviewed_bboxes
    assert (50.0, 10.0, 60.0, 20.0) in reviewed_bboxes
    assert (50.0, 0.0, 60.0, 10.0) in reviewed_bboxes
    assert geometry_status[1][4:] == ["derived", "derived"]


def test_leading_cjk_section_label_recovery_requires_matching_crop_consensus(monkeypatch):
    np = pytest.importorskip("numpy")
    import docmirror.ocr.micro_grid.cell_recognition as cell_recognition
    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

    def recognize(_image, bbox, **_kwargs):
        if bbox[1] < 20:
            return CellRecognition(
                "一、经营活动产生的现金流量：",
                0.96,
                "cell_crop_consensus",
                audit={"consensus_count": 3},
            )
        return CellRecognition(
            "二、投资活动收到的现金：",
            0.97,
            "cell_crop_consensus",
            audit={"consensus_count": 3},
        )

    monkeypatch.setattr(cell_recognition, "recognize_micro_cell_from_image", recognize)
    raw = [
        ["项目", "行次", "本年累计金额", "本月金额"],
        ["、经营活动产生的现金流量:", "", "", ""],
        ["、投资活动产生的现金流量:", "", "", ""],
        ["、现金净增加额", "20", "10.00", "1.00"],
    ]
    cell_bboxes = [
        [[float(column * 10), float(row * 10), float((column + 1) * 10), float((row + 1) * 10)] for column in range(4)]
        for row in range(len(raw))
    ]
    evidence_ids = [[[] for _column in range(4)] for _row in raw]
    evidence_ids[1][0] = ["ev:section-one"]
    confidences = [[None for _column in range(4)] for _row in raw]

    recovered, events = _recover_leading_cjk_section_labels(
        np.full((100, 100, 3), 255, dtype=np.uint8),
        raw,
        cell_bboxes=cell_bboxes,
        cell_evidence_ids=evidence_ids,
        cell_confidences=confidences,
        page_width=100,
        page_height=100,
        page_number=1,
        table_index=0,
    )

    assert recovered[1][0] == "一、经营活动产生的现金流量:"
    assert recovered[2][0] == "、投资活动产生的现金流量:"
    assert recovered[3][0] == "、现金净增加额"
    assert len(events) == 1
    assert events[0]["source_ref"] == "ev:section-one"
    assert events[0]["reason_codes"] == ["leading_cjk_ordinal_label_crop_consensus"]
    assert confidences[1][0] == 0.96


def test_cjk_ordinal_recovery_uses_only_same_column_sequence_evidence():
    raw = [
        ["项目", "栏次", "金额"],
        ["一）第一组", "1", "1.00"],
        ["普通项目", "2", "2.00"],
        ["）第二组", "3", "3.00"],
        ["（三）第三组", "4", "4.00"],
        ["（四）第四组", "5", "5.00"],
    ]
    cell_bboxes = [
        [[float(column * 10), float(row * 10), float((column + 1) * 10), float((row + 1) * 10)] for column in range(3)]
        for row in range(len(raw))
    ]
    evidence_ids = [[[] for _column in range(3)] for _row in raw]
    confidences = [[None for _column in range(3)] for _row in raw]

    recovered, events = _recover_cjk_ordinal_markers(
        raw,
        cell_bboxes=cell_bboxes,
        cell_evidence_ids=evidence_ids,
        cell_confidences=confidences,
        page_number=1,
        table_index=0,
    )

    assert recovered[1][0] == "（一）第一组"
    assert recovered[3][0] == "（二）第二组"
    assert {event["reason_codes"][0] for event in events} == {"cjk_ordinal_sequence_lattice"}


def test_arabic_ordinal_recovery_uses_section_bounded_sequence_evidence():
    raw = [
        ["(二)所有者投入和减少资本", ""],
        ["", ""],
        ["1.所有者投入的普通股", ""],
        ["2.其他权益工具投入资本", ""],
        ["．股份支付计入所有者权益的金额", ""],
        ["4.其他", ""],
        ["(三)利润分配", ""],
        ["1.提取盈余公积", ""],
        ["2.对所有者的分配", ""],
        [".其他", ""],
        ["(四)所有者权益内部结转", ""],
        ["1.资本公积转增资本", ""],
        [".盈余公积转增资本", ""],
        [".盈余公积弥补亏损", ""],
        ["1．设定受益变动计划额结转留存收益", ""],
        [".其他综合收益结转留存收益", ""],
        ["其他", ""],
        ["四、本期期末余额", "100.00"],
    ]
    cell_bboxes = [
        [
            [float(column * 100), float(row * 10), float((column + 1) * 100), float((row + 1) * 10)]
            for column in range(2)
        ]
        for row in range(len(raw))
    ]
    evidence_ids = [[[] for _column in range(2)] for _row in raw]
    confidences = [[None for _column in range(2)] for _row in raw]

    recovered, events = _recover_arabic_ordinal_markers(
        raw,
        cell_bboxes=cell_bboxes,
        cell_evidence_ids=evidence_ids,
        cell_confidences=confidences,
        page_number=1,
        table_index=0,
    )

    assert [recovered[index][0] for index in (4, 9, 12, 13, 14, 15, 16)] == [
        "3.股份支付计入所有者权益的金额",
        "3.其他",
        "2.盈余公积转增资本",
        "3.盈余公积弥补亏损",
        "4.设定受益变动计划额结转留存收益",
        "5.其他综合收益结转留存收益",
        "6.其他",
    ]
    assert len(events) == 7
    assert {event["reason_codes"][0] for event in events} == {"arabic_ordinal_section_sequence_lattice"}


def test_cell_tokens_follow_visual_lines_before_horizontal_position():
    tokens = [
        _Token("应补缴税额", (83.0, 363.0, 123.0, 373.0), "ev:bottom", 0.99),
        _Token("按简易计税办法计算的纳税检查", (83.0, 354.0, 192.0, 364.0), "ev:top", 0.99),
    ]

    assert [token.text for token in _tokens_in_reading_order(tokens)] == [
        "按简易计税办法计算的纳税检查",
        "应补缴税额",
    ]


def test_table_token_preserves_source_compatibility_characters():
    block = _ocr_block("1分次预缴税额", 10, 10, 100, 20, 1)
    block = replace(
        block,
        attrs={
            **block.attrs,
            "ocr_original_text": "①分次预缴税额",
            "ocr_correction": {
                "rule_id": "unicode.normalize",
                "reason_codes": ["unicode_normalization"],
            },
        },
    )

    assert _block_to_token(block).text == "①分次预缴税额"
