# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)


def _sealed_table(
    table_id: str,
    rows: list[list[str]],
    *,
    scale: float = 1.0,
    column_edges: tuple[float, ...] = (0.0, 44.0, 130.0, 173.0, 215.0, 301.0, 344.0),
    token_layout: dict[tuple[int, int], tuple[tuple[str, float, float], ...]] | None = None,
    source_cells: list[list[SimpleNamespace | None]] | None = None,
) -> SimpleNamespace:
    if any(len(row) != len(column_edges) - 1 for row in rows):
        raise ValueError("fixture rows and column lattice disagree")
    layouts = token_layout or {
        (0, 0): (("参缴地", 0.16, 0.84),),
        (0, 1): (("参缴日期", 0.07, 0.30), ("初缴月份", 0.48, 0.80)),
        (0, 2): (("缴至月份", 0.12, 0.84),),
        (0, 3): (("缴费状态", 0.16, 0.84),),
        (0, 4): (("月缴存额", 0.18, 0.44),),
        (0, 5): (("个人缴存比例单位缴存比例", 0.02, 0.94),),
        (1, 0): (("福建省漳州市", 0.08, 0.92),),
        (1, 1): (("2020.11.09", 0.05, 0.30),),
        (1, 2): (("2025.04", 0.22, 0.78),),
        (1, 3): (("缴交", 0.35, 0.65),),
        (1, 4): (("1,500", 0.10, 0.34), ("10%", 0.78, 0.92)),
        (1, 5): (("10%", 0.58, 0.86),),
        (2, 2): (("缴费单位", 0.18, 0.82),),
        (2, 5): (("信息更新日期", 0.08, 0.92),),
    }
    bboxes: list[list[list[float]]] = []
    evidence: list[list[list[str]]] = []
    atoms: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        bbox_row: list[list[float]] = []
        evidence_row: list[list[str]] = []
        for column, raw in enumerate(row):
            left = column_edges[column] * scale
            right = column_edges[column + 1] * scale
            bbox_row.append([left, row_index * 14.0 * scale, right, (row_index + 1) * 14.0 * scale])
            entries = layouts.get((row_index, column), ())
            ids: list[str] = []
            for token_index, (text, fraction_left, fraction_right) in enumerate(entries):
                token_id = f"e:{table_id}:{row_index}:{column}:{token_index}"
                ids.append(token_id)
                atoms.append(
                    {
                        "id": token_id,
                        "text": text,
                        "bbox": [
                            left + (right - left) * fraction_left,
                            (row_index * 14.0 + 2.0) * scale,
                            left + (right - left) * fraction_right,
                            (row_index * 14.0 + 12.0) * scale,
                        ],
                    }
                )
            if "".join(text for text, _left, _right in entries).replace(" ", "") != raw.replace(" ", ""):
                raise ValueError(f"fixture token text does not seal cell {(row_index, column)}")
            evidence_row.append(ids)
        bboxes.append(bbox_row)
        evidence.append(evidence_row)
    geometry = {
        "coordinate_system": "pdf_points",
        "row_bands": [
            {"index": row, "y0": row * 14.0 * scale, "y1": (row + 1) * 14.0 * scale}
            for row in range(len(rows))
        ],
        "col_bands": [
            {"index": column, "x0": column_edges[column] * scale, "x1": column_edges[column + 1] * scale}
            for column in range(len(column_edges) - 1)
        ],
        "cell_bboxes": bboxes,
        "cell_geometry_status": [["exact" for _cell in row] for row in rows],
        "cell_evidence_ids": evidence,
        "cell_token_ids": evidence,
        "cell_spans": [],
    }
    table = SimpleNamespace(
        table_id=table_id,
        bbox=[column_edges[0] * scale, 0.0, column_edges[-1] * scale, len(rows) * 14.0 * scale],
        metadata={
            "raw_rows": rows,
            "source_cell_bboxes": bboxes,
            "cell_evidence_ids": evidence,
            "cell_token_ids": evidence,
            "geometry": geometry,
        },
        source_cell_objects=source_cells,
    )
    table._test_atoms = atoms
    return table


def _scaled_housing_context(scale: float) -> SimpleNamespace:
    table = _sealed_table("housing", _housing_rows(), scale=scale)
    lines = [
        {
            "text": "漳州市玖一企业管理咨询有限公司 2025.05",
            "bbox": [55.0 * scale, 44.0 * scale, 336.0 * scale, 54.0 * scale],
            "evidence_ids": ["provider:employer", "provider:month"],
        }
    ]
    context = _housing_context(table=table, lines=lines)
    for atom in context.evidence_plane.evidence.text_atoms:
        if atom["id"] in {"provider:employer", "provider:month"}:
            atom["bbox"] = [value * scale for value in atom["bbox"]]
    return context


def _permute_housing_columns(order: tuple[int, ...]) -> SimpleNamespace:
    base = _sealed_table("housing", _housing_rows())
    rows = [[row[index] for index in order] for row in base.metadata["raw_rows"]]
    layouts: dict[tuple[int, int], tuple[tuple[str, float, float], ...]] = {}
    for row in range(len(rows)):
        for new_column, old_column in enumerate(order):
            old_left = base.metadata["geometry"]["col_bands"][old_column]["x0"]
            old_right = base.metadata["geometry"]["col_bands"][old_column]["x1"]
            width = old_right - old_left
            entries = []
            for atom in base._test_atoms:
                if atom["id"].startswith(f"e:housing:{row}:{old_column}:"):
                    entries.append(
                        (
                            atom["text"],
                            (atom["bbox"][0] - old_left) / width,
                            (atom["bbox"][2] - old_left) / width,
                        )
                    )
            if entries:
                layouts[(row, new_column)] = tuple(entries)
    return _sealed_table("housing-reordered", rows, token_layout=layouts)


def _housing_rows() -> list[list[str]]:
    return [
        ["参缴地", "参缴日期 初缴月份", "缴至月份", "缴费状态", "月缴存额", "个人缴存比例单位缴存比例"],
        ["福建省漳州市", "2020.11.09", "2025.04", "缴交", "1,500 10%", "10%"],
        ["", "", "缴费单位", "", "", "信息更新日期"],
    ]


def _housing_context(
    *,
    table: SimpleNamespace | None = None,
    lines: list[dict[str, object]] | None = None,
) -> SimpleNamespace:
    housing_table = table or _sealed_table("housing", _housing_rows())
    default_lines = [
        {
            "text": "漳州市玖一企业管理咨询有限公司 2025.05",
            "bbox": [55.0, 44.0, 336.0, 54.0],
            "evidence_ids": ["provider:employer", "provider:month"],
        }
    ]
    evidence_lines = default_lines if lines is None else lines
    atoms = list(getattr(housing_table, "_test_atoms", ()))
    atom_specs = {
        "provider:employer": ("漳州市玖一企业管理咨询有限公司", [55.0, 44.0, 220.0, 54.0]),
        "provider:month": ("2025.05", [307.0, 44.0, 336.0, 54.0]),
        "provider:employer:second": ("厦门市某某科技有限公司", [55.0, 58.0, 220.0, 68.0]),
        "provider:month:second": ("2025.06", [307.0, 58.0, 336.0, 68.0]),
        "provider:noise": ("其他", [150.0, 44.0, 180.0, 54.0]),
    }
    requested_ids = {
        str(value)
        for line in evidence_lines
        for value in line.get("evidence_ids", [])
    }
    for token_id in requested_ids:
        if token_id not in atom_specs:
            continue
        text, bbox = atom_specs[token_id]
        atoms.append({"id": token_id, "text": text, "bbox": bbox})
    page = SimpleNamespace(page_number=50, source_page_number=25, tables=[housing_table])
    return SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
        corrected_evidence_pages=lambda: [
            {"page": 50, "source_page": 25, "lines": evidence_lines}
        ],
        _personal_detail_extraction_issues=[],
    )


def test_compound_housing_fund_recovers_typed_values_without_date_shift() -> None:
    context = _housing_context()

    record = native_extraction._extract_public_records(context)[0]

    assert record["participation_date"] == "2020-11-09"
    assert record["paid_through_month"] == "2025-04"
    assert record["monthly_contribution"] == 1500
    assert record["personal_contribution_ratio"] == "10%"
    assert record["employer_contribution_ratio"] == "10%"
    assert record["employer"] == "漳州市玖一企业管理咨询有限公司"
    assert record["information_updated_month"] == "2025-05"
    assert "first_contribution_month" not in record
    assert any(
        issue["field_name"] == "first_contribution_month"
        for issue in context._personal_detail_extraction_issues
    )


def test_compound_housing_fund_signature_without_sealed_cells_fails_closed() -> None:
    rows = [
        ["参缴地", "参缴日期 初缴月份", "缴至月份", "缴费状态", "月缴存额", "个人缴存比例单位缴存比例"],
        ["Fuzhou", "2020.11.09", "2025.04", "active", "1,500 10%", "10%"],
        ["", "", "缴费单位", "", "", "信息更新日期"],
    ]
    table = SimpleNamespace(table_id="unsealed", bbox=[0, 0, 6, 3], metadata={"raw_rows": rows})
    context = SimpleNamespace(
        pages=[SimpleNamespace(page_number=50, source_page_number=25, tables=[table])],
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def test_compound_housing_fund_accepts_only_a_visually_proven_initial_month_dash() -> None:
    import cv2
    import numpy as np

    rows = _housing_rows()
    image = np.full((420, 3440, 3), 255, dtype=np.uint8)
    cv2.line(image, (930, 210), (1010, 210), (0, 0, 0), 3)
    cv2.line(image, (1050, 210), (1130, 210), (0, 0, 0), 3)
    source_cells: list[list[SimpleNamespace | None]] = [[None] * 6 for _ in range(3)]
    source_cells[1][1] = SimpleNamespace(
        bbox=[44.0, 14.0, 130.0, 28.0],
        geometry_status="exact",
        evidence_ids=["e:housing:1:1:0"],
    )
    table = _sealed_table("housing", rows, source_cells=source_cells)
    table.metadata["source_logical_page"] = 50
    context = _housing_context(table=table)
    context._page_image_resolver = lambda _page: {
        "image": image,
        "page_width": 344.0,
        "page_height": 42.0,
    }

    record = native_extraction._extract_public_records(context)[0]

    assert record["_source_absent_fields"] == ["first_contribution_month"]
    assert record["canonical_raw"]["first_contribution_month"] == "--"
    absence_ref = record["source_refs_by_field"]["first_contribution_month"][0]
    assert absence_ref["geometry_scope"] == "visual_subcell"
    left, right = absence_ref["horizontal_fraction"]
    assert 0.41 < left < 0.42
    assert 0.93 < right < 0.95
    assert 79.0 < absence_ref["bbox"][0] < 80.0
    assert 124.0 < absence_ref["bbox"][2] < 126.0
    assert absence_ref["visual_proof"] == "dash_pair_only"
    assert "evidence_ids" not in absence_ref
    assert not any(
        issue.get("field_name") == "first_contribution_month"
        for issue in context._personal_detail_extraction_issues
    )


def test_compound_housing_fund_does_not_treat_a_third_glyph_as_source_absence() -> None:
    import cv2
    import numpy as np

    rows = _housing_rows()
    image = np.full((420, 3440, 3), 255, dtype=np.uint8)
    cv2.line(image, (930, 210), (1010, 210), (0, 0, 0), 3)
    cv2.line(image, (1050, 210), (1130, 210), (0, 0, 0), 3)
    cv2.rectangle(image, (1160, 190), (1210, 230), (0, 0, 0), -1)
    source_cells: list[list[SimpleNamespace | None]] = [[None] * 6 for _ in range(3)]
    source_cells[1][1] = SimpleNamespace(
        bbox=[44.0, 14.0, 130.0, 28.0],
        geometry_status="exact",
        evidence_ids=["e:housing:1:1:0"],
    )
    table = _sealed_table("housing", rows, source_cells=source_cells)
    table.metadata["source_logical_page"] = 50
    context = _housing_context(table=table)
    context._page_image_resolver = lambda _page: {
        "image": image,
        "page_width": 344.0,
        "page_height": 42.0,
    }

    record = native_extraction._extract_public_records(context)[0]

    assert "_source_absent_fields" not in record
    assert any(
        issue.get("field_name") == "first_contribution_month"
        for issue in context._personal_detail_extraction_issues
    )


def test_compound_housing_fund_is_scale_independent() -> None:
    for scale in (0.5, 2.0):
        context = _scaled_housing_context(scale)
        records, consumed = native_extraction._bounded_compound_housing_fund_records(context)

        assert consumed == {"housing"}
        record = records[("housing_fund", 1)]
        assert record["participation_date"] == "2020-11-09"
        assert record["personal_contribution_ratio"] == "10%"
        assert record["employer"] == "漳州市玖一企业管理咨询有限公司"


def test_compound_housing_fund_accepts_unequal_seven_column_lattice() -> None:
    rows = [
        ["参缴地", "参缴日期 初缴月份", "缴至月份", "缴费状态", "月缴存额", "个人缴存比例", "单位缴存比例"],
        ["福建省漳州市", "2020.11.09", "2025.04", "缴交", "1,500", "10%", "10%"],
        ["", "", "缴费单位", "", "", "", "信息更新日期"],
    ]
    layout = {
        (0, 0): (("参缴地", 0.16, 0.84),),
        (0, 1): (("参缴日期", 0.06, 0.27), ("初缴月份", 0.47, 0.77)),
        (0, 2): (("缴至月份", 0.10, 0.86),),
        (0, 3): (("缴费状态", 0.16, 0.84),),
        (0, 4): (("月缴存额", 0.18, 0.72),),
        (0, 5): (("个人缴存比例", 0.05, 0.95),),
        (0, 6): (("单位缴存比例", 0.05, 0.95),),
        (1, 0): (("福建省漳州市", 0.08, 0.92),),
        (1, 1): (("2020.11.09", 0.05, 0.28),),
        (1, 2): (("2025.04", 0.20, 0.80),),
        (1, 3): (("缴交", 0.34, 0.66),),
        (1, 4): (("1,500", 0.22, 0.78),),
        (1, 5): (("10%", 0.30, 0.70),),
        (1, 6): (("10%", 0.30, 0.70),),
        (2, 2): (("缴费单位", 0.10, 0.90),),
        (2, 6): (("信息更新日期", 0.05, 0.95),),
    }
    table = _sealed_table(
        "housing-seven",
        rows,
        column_edges=(0.0, 39.0, 142.0, 181.0, 229.0, 277.0, 329.0, 401.0),
        token_layout=layout,
    )
    context = _housing_context(
        table=table,
        lines=[
            {
                "text": "漳州市玖一企业管理咨询有限公司 2025.05",
                "bbox": [45.0, 44.0, 390.0, 54.0],
                "evidence_ids": ["provider:employer", "provider:month"],
            }
        ],
    )
    for atom in context.evidence_plane.evidence.text_atoms:
        if atom["id"] == "provider:employer":
            atom["bbox"] = [45.0, 44.0, 250.0, 54.0]
        elif atom["id"] == "provider:month":
            atom["bbox"] = [355.0, 44.0, 390.0, 54.0]

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)

    assert consumed == {"housing-seven"}
    record = records[("housing_fund", 1)]
    assert record["monthly_contribution"] == 1500
    assert record["personal_contribution_ratio"] == "10%"
    assert record["employer_contribution_ratio"] == "10%"


def test_compound_housing_fund_reordered_columns_fail_closed_without_shifting() -> None:
    table = _permute_housing_columns((0, 2, 1, 3, 4, 5))
    context = _housing_context(table=table)

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def test_compound_housing_fund_reversed_provider_headers_fail_closed() -> None:
    table = _sealed_table("housing-provider-reversed", _housing_rows())
    rows = table.metadata["raw_rows"]
    rows[2][2], rows[2][5] = rows[2][5], rows[2][2]
    for atom in table._test_atoms:
        if atom["id"] == "e:housing-provider-reversed:2:2:0":
            atom["text"] = "淇℃伅鏇存柊鏃ユ湡"
        elif atom["id"] == "e:housing-provider-reversed:2:5:0":
            atom["text"] = "缂磋垂鍗曚綅"
    context = _housing_context(table=table)

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def test_compound_housing_fund_unknown_header_family_does_not_authorize() -> None:
    rows = _housing_rows()
    rows[0][3] = "缴费状态账户状态"
    layout = {
        (0, 0): (("参缴地", 0.16, 0.84),),
        (0, 1): (("参缴日期", 0.07, 0.30), ("初缴月份", 0.48, 0.80)),
        (0, 2): (("缴至月份", 0.12, 0.84),),
        (0, 3): (("缴费状态账户状态", 0.05, 0.95),),
        (0, 4): (("月缴存额", 0.18, 0.44),),
        (0, 5): (("个人缴存比例单位缴存比例", 0.02, 0.94),),
        (1, 0): (("福建省漳州市", 0.08, 0.92),),
        (1, 1): (("2020.11.09", 0.05, 0.30),),
        (1, 2): (("2025.04", 0.22, 0.78),),
        (1, 3): (("缴交", 0.35, 0.65),),
        (1, 4): (("1,500", 0.10, 0.34), ("10%", 0.78, 0.92)),
        (1, 5): (("10%", 0.58, 0.86),),
        (2, 2): (("缴费单位", 0.18, 0.82),),
        (2, 5): (("信息更新日期", 0.08, 0.92),),
    }
    context = _housing_context(table=_sealed_table("new-family", rows, token_layout=layout))

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def test_compound_housing_fund_duplicate_header_evidence_fails_closed() -> None:
    table = _sealed_table("housing", _housing_rows())
    duplicate = table.metadata["cell_evidence_ids"][0][0][0]
    table.metadata["cell_evidence_ids"][0][2] = [duplicate]
    table.metadata["cell_token_ids"][0][2] = [duplicate]
    table.metadata["geometry"]["cell_evidence_ids"][0][2] = [duplicate]
    table.metadata["geometry"]["cell_token_ids"][0][2] = [duplicate]
    context = _housing_context(table=table)

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def test_compound_housing_fund_registered_multi_role_header_may_cross_grid_divider() -> None:
    table = _sealed_table("housing-spilled-header", _housing_rows())
    atom = next(
        atom
        for atom in table._test_atoms
        if atom["id"] == "e:housing-spilled-header:0:5:0"
    )
    atom["bbox"] = [278.0, 1.0, 342.0, 15.0]
    context = _housing_context(table=table)

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)

    assert consumed == {"housing-spilled-header"}
    assert records[("housing_fund", 1)]["personal_contribution_ratio"] == "10%"
    assert records[("housing_fund", 1)]["employer_contribution_ratio"] == "10%"


def test_compound_housing_fund_multi_role_header_spill_across_row_fails_closed() -> None:
    table = _sealed_table("housing-cross-row-header", _housing_rows())
    atom = next(
        atom
        for atom in table._test_atoms
        if atom["id"] == "e:housing-cross-row-header:0:5:0"
    )
    atom["bbox"] = [250.0, 8.0, 342.0, 22.0]
    context = _housing_context(table=table)

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def test_compound_housing_fund_cross_role_value_token_fails_closed() -> None:
    table = _sealed_table("housing", _housing_rows())
    atom = next(atom for atom in table._test_atoms if atom["id"] == "e:housing:1:4:1")
    atom["bbox"] = [270.0, 16.0, 286.0, 26.0]
    context = _housing_context(table=table)

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def test_compound_housing_fund_foreign_colocated_token_does_not_authorize() -> None:
    rows = _housing_rows()
    rows[1][3] = "缴交批准"
    table = _sealed_table(
        "foreign-value",
        rows,
        token_layout={
            (0, 0): (("参缴地", 0.16, 0.84),),
            (0, 1): (("参缴日期", 0.07, 0.30), ("初缴月份", 0.48, 0.80)),
            (0, 2): (("缴至月份", 0.12, 0.84),),
            (0, 3): (("缴费状态", 0.16, 0.84),),
            (0, 4): (("月缴存额", 0.18, 0.44),),
            (0, 5): (("个人缴存比例单位缴存比例", 0.02, 0.94),),
            (1, 0): (("福建省漳州市", 0.08, 0.92),),
            (1, 1): (("2020.11.09", 0.05, 0.30),),
            (1, 2): (("2025.04", 0.22, 0.78),),
            (1, 3): (("缴交", 0.25, 0.48), ("批准", 0.53, 0.77)),
            (1, 4): (("1,500", 0.10, 0.34), ("10%", 0.78, 0.92)),
            (1, 5): (("10%", 0.58, 0.86),),
            (2, 2): (("缴费单位", 0.18, 0.82),),
            (2, 5): (("信息更新日期", 0.08, 0.92),),
        },
    )
    context = _housing_context(table=table)

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)

    assert consumed == {"foreign-value"}
    record = records[("housing_fund", 1)]
    assert "payment_status" not in record
    assert any(
        issue["field_name"] == "payment_status"
        for issue in context._personal_detail_extraction_issues
    )


def test_compound_housing_fund_provider_split_lines_are_geometry_paired() -> None:
    lines = [
        {
            "text": "漳州市玖一企业管理咨询有限公司",
            "bbox": [55.0, 44.0, 220.0, 54.0],
            "evidence_ids": ["provider:employer"],
        },
        {
            "text": "2025.05",
            "bbox": [307.0, 44.0, 336.0, 54.0],
            "evidence_ids": ["provider:month"],
        },
    ]
    context = _housing_context(lines=lines)

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)

    assert consumed == {"housing"}
    assert records[("housing_fund", 1)]["employer"] == "漳州市玖一企业管理咨询有限公司"
    assert records[("housing_fund", 1)]["information_updated_month"] == "2025-05"


def test_compound_housing_fund_foreign_provider_token_does_not_authorize() -> None:
    lines = [
        {
            "text": "婕冲窞甯傜帠涓€浼佷笟绠＄悊鍜ㄨ鏈夐檺鍏徃 鍏朵粬 2025.05",
            "bbox": [55.0, 44.0, 336.0, 54.0],
            "evidence_ids": ["provider:employer", "provider:noise", "provider:month"],
        }
    ]
    context = _housing_context(lines=lines)

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)
    record = records[("housing_fund", 1)]

    assert consumed == {"housing"}
    assert "employer" not in record
    assert "information_updated_month" not in record
    assert {issue["field_name"] for issue in context._personal_detail_extraction_issues} >= {
        "employer",
        "information_updated_month",
    }


def test_compound_housing_fund_later_typed_line_after_foreign_row_does_not_authorize() -> None:
    lines = [
        {
            "text": "鍏朵粬",
            "bbox": [150.0, 44.0, 180.0, 54.0],
            "evidence_ids": ["provider:noise"],
        },
        {
            "text": "婕冲窞甯傜帠涓€浼佷笟绠＄悊鍜ㄨ鏈夐檺鍏徃 2025.05",
            "bbox": [55.0, 58.0, 336.0, 68.0],
            "evidence_ids": ["provider:employer", "provider:month"],
        },
    ]
    context = _housing_context(lines=lines)
    for atom in context.evidence_plane.evidence.text_atoms:
        if atom["id"] == "provider:employer":
            atom["bbox"] = [55.0, 58.0, 220.0, 68.0]
        elif atom["id"] == "provider:month":
            atom["bbox"] = [307.0, 58.0, 336.0, 68.0]

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)
    record = records[("housing_fund", 1)]

    assert consumed == {"housing"}
    assert "employer" not in record
    assert "information_updated_month" not in record


def test_compound_housing_fund_without_visual_dash_reports_initial_month_unresolved() -> None:
    context = _housing_context()

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)
    record = records[("housing_fund", 1)]

    assert consumed == {"housing"}
    assert "first_contribution_month" not in record
    assert "first_contribution_month" in record["_unresolved_fields"]
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("field_name") == "first_contribution_month"
    ]
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "candidate_b_exact_slot_value_invalid"


def test_compound_housing_fund_duplicate_provider_rows_report_uncertainty() -> None:
    lines = [
        {
            "text": "漳州市玖一企业管理咨询有限公司 2025.05",
            "bbox": [55.0, 44.0, 336.0, 54.0],
            "evidence_ids": ["provider:employer", "provider:month"],
        },
        {
            "text": "厦门市某某科技有限公司 2025.06",
            "bbox": [55.0, 58.0, 336.0, 68.0],
            "evidence_ids": ["provider:employer:second", "provider:month:second"],
        },
    ]
    context = _housing_context(lines=lines)

    records, consumed = native_extraction._bounded_compound_housing_fund_records(context)
    record = records[("housing_fund", 1)]

    assert consumed == {"housing"}
    assert "employer" not in record
    assert "information_updated_month" not in record
    assert {issue["field_name"] for issue in context._personal_detail_extraction_issues} >= {
        "employer",
        "information_updated_month",
    }


def test_compound_housing_fund_foreign_exact_header_collision_without_lattice_fails_closed() -> None:
    rows = _housing_rows()
    table = SimpleNamespace(
        table_id="foreign-collision",
        bbox=[0.0, 0.0, 344.0, 42.0],
        metadata={"raw_rows": rows},
    )
    context = _housing_context(table=table)
    context.evidence_plane.evidence.text_atoms = []

    assert native_extraction._bounded_compound_housing_fund_records(context) == ({}, set())


def _summary_context(
    card_minimum: str,
    *,
    block_order: tuple[int, ...] = (0, 1, 2),
    scale: float = 1.0,
    extra_blocks: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (),
) -> SimpleNamespace:
    lines: list[dict[str, object]] = []
    atoms: list[dict[str, object]] = []
    blocks = (
        ("循环贷账户一信息汇总", "管理机构数 账户数 授信总额 余额 最近6个月平均应还款", "2 5 800,000 545,493 530"),
        ("循环贷账户二信息汇总", "管理机构数 账户数 授信总额 余额 最近6个月平均应还款", "3 3 614,000 299,428 120,495"),
        (
            "贷记卡账户信息汇总",
            "发卡机构数 账户数 授信总额 单家机构最高授信额 单家机构最低授信额 已用额度 最近6个月平均使用额度",
            f"12 16 690,800 226,000 {card_minimum} 26,101 38,562",
        ),
    )
    ordered_blocks = [blocks[index] for index in block_order]
    ordered_blocks.extend(
        (title, " ".join(headers), " ".join(values))
        for title, headers, values in extra_blocks
    )
    for index, block in enumerate(ordered_blocks):
        for offset, text in enumerate(block):
            evidence_ids: list[str] = []
            if offset == 0:
                evidence_ids = [f"e:{index}:title"]
                atoms.append(
                    {
                        "id": evidence_ids[0],
                        "text": text,
                        "bbox": [0, index * 30 * scale, 300 * scale, (index * 30 + 8) * scale],
                    }
                )
            else:
                parts = (
                    list(
                        next(
                            headers
                            for title, headers in native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS
                            if title == block[0]
                        )
                    )
                    if offset == 1 and block[0] in {title for title, _headers in native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS}
                    else text.split()
                )
                width = 300.0 * scale / len(parts)
                for column, part in enumerate(parts):
                    atom_id = f"e:{index}:{offset}:{column}"
                    evidence_ids.append(atom_id)
                    atoms.append(
                        {
                            "id": atom_id,
                            "text": part,
                            "bbox": [
                                column * width,
                                (index * 30 + offset * 10) * scale,
                                (column + 1) * width - 2,
                                (index * 30 + offset * 10 + 8) * scale,
                            ],
                        }
                    )
            lines.append(
                {
                    "text": text,
                    "bbox": [
                        0,
                        (index * 30 + offset * 10) * scale,
                        300 * scale,
                        (index * 30 + offset * 10 + 8) * scale,
                    ],
                    "evidence_ids": evidence_ids,
                }
            )
    return SimpleNamespace(
        pages=[],
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
        corrected_evidence_pages=lambda: [{"page": 3, "source_page": 2, "lines": lines}],
        _personal_detail_extraction_issues=[],
    )


def test_registered_scalar_summaries_recover_all_seventeen_slots() -> None:
    records, cells = native_extraction._extract_summary_datasets(_summary_context("10,000"))

    assert len(records) == 3
    assert len(cells) == 17
    assert {cell["value"] for cell in cells if cell["column_label"] == "单家机构最低授信额"} == {"10,000"}
    assert all(cell.get("value_status") != "unreadable" for cell in cells)


def test_registered_scalar_summary_single_block_is_independently_owned() -> None:
    records, cells = native_extraction._extract_summary_datasets(
        _summary_context("10,000", block_order=(1,))
    )

    second_title = native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[1][0]
    assert [record["title"] for record in records] == [second_title]
    assert len(cells) == 5


def test_registered_scalar_summary_blocks_can_be_reordered() -> None:
    records, cells = native_extraction._extract_summary_datasets(
        _summary_context("10,000", block_order=(2, 0, 1))
    )

    assert [record["title"] for record in records] == [
        native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[index][0]
        for index in (0, 1, 2)
    ]
    assert len(cells) == 17


def test_registered_scalar_summary_is_scale_independent() -> None:
    for scale in (0.5, 2.0):
        records, cells = native_extraction._extract_summary_datasets(
            _summary_context("10,000", block_order=(0,), scale=scale)
        )
        assert len(records) == 1
        assert len(cells) == 5


def test_unregistered_pboc_summary_block_cannot_authorize_registered_neighbour() -> None:
    quasi_title = "准贷记卡账户信息汇总"
    context = _summary_context(
        "10,000",
        block_order=(0,),
        extra_blocks=((quasi_title, ("账户数",), ("9",)),),
    )
    # Remove the registered block's own value evidence.  The extra exact PBOC
    # block must close the interval, not donate its scalar to the first block.
    context.evidence_plane.evidence.text_atoms = [
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if not str(atom["id"]).startswith("e:0:2:")
    ]

    records, cells = native_extraction._extract_summary_datasets(context)

    assert not any(record.get("title") == native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[0][0] for record in records)
    assert not any(cell.get("title") == native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[0][0] for cell in cells)


def test_registered_scalar_summary_title_collision_outside_pboc_form_fails_closed() -> None:
    context = _summary_context("10,000", block_order=(0,))
    page = context.corrected_evidence_pages()[0]
    page["lines"][0]["text"] = "合同附件循环贷账户一信息汇总"
    context.corrected_evidence_pages = lambda: [page]

    records, cells = native_extraction._extract_summary_datasets(context)

    assert records == []
    assert cells == []


def test_registered_scalar_summary_inside_foreign_registered_section_fails_closed() -> None:
    context = _summary_context("10,000", block_order=(0,))
    page = context.corrected_evidence_pages()[0]
    section = {
        "text": "（五）公共信息明细",
        "bbox": [0.0, -12.0, 300.0, -2.0],
        "evidence_ids": ["foreign-section"],
    }
    page["lines"].insert(0, section)
    context.evidence_plane.evidence.text_atoms.append(
        {"id": "foreign-section", "text": section["text"], "bbox": section["bbox"]}
    )
    context.corrected_evidence_pages = lambda: [page]

    records, cells = native_extraction._extract_summary_datasets(context)

    assert records == []
    assert cells == []


def test_registered_scalar_summary_rejects_duplicate_title_and_header_evidence() -> None:
    for duplicate_kind in ("title", "header"):
        context = _summary_context("10,000", block_order=(0,))
        page = context.corrected_evidence_pages()[0]
        source_index = 0 if duplicate_kind == "title" else 1
        duplicate = dict(page["lines"][source_index])
        duplicate["evidence_ids"] = [f"duplicate-{duplicate_kind}"]
        duplicate["bbox"] = list(duplicate["bbox"])
        page["lines"].append(duplicate)
        context.evidence_plane.evidence.text_atoms.append(
            {
                "id": duplicate["evidence_ids"][0],
                "text": duplicate["text"],
                "bbox": duplicate["bbox"],
            }
        )
        context.corrected_evidence_pages = lambda page=page: [page]

        records, cells = native_extraction._extract_summary_datasets(context)

        assert records == []
        assert cells == []


def test_registered_scalar_summary_rejects_duplicate_source_evidence_id() -> None:
    context = _summary_context("10,000", block_order=(0,))
    atoms = context.evidence_plane.evidence.text_atoms
    duplicate = dict(next(atom for atom in atoms if atom["id"] == "e:0:2:0"))
    atoms.append(duplicate)

    records, cells = native_extraction._extract_summary_datasets(context)

    assert records == []
    assert cells == []


def test_registered_scalar_summaries_survive_the_closed_world_projection() -> None:
    records, cells = native_extraction._extract_summary_datasets(_summary_context("10,000"))

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "personal_detail_summary_records": records,
                "personal_detail_summary_cells": cells,
            },
        }
    )
    metrics = prepared["datasets"]["personal_detail_credit_summary_metrics"]
    projected = project_personal_detail_datasets(prepared["datasets"])
    overview = projected["credit_business_overview"]

    assert len(metrics) == 17
    assert len(overview) == 17
    assert all(row.get("normalized", row)["mapping_status"] == "mapped" for row in overview)
    assert not [
        row
        for row in projected.get("extraction_issues", [])
        if row.get("normalized", row).get("target_dataset") == "credit_business_overview"
    ]


def test_registered_scalar_summary_reports_invalid_grouped_number_at_exact_metric() -> None:
    context = _summary_context("0,000")

    _records, cells = native_extraction._extract_summary_datasets(context)

    minimum = next(cell for cell in cells if cell["column_label"] == "单家机构最低授信额")
    assert minimum["value"] == "0,000"
    assert minimum["value_status"] == "unreadable"
    assert any(
        issue["issue_code"] == "candidate_b_registered_summary_scalar_unreadable"
        and issue["target_record_id"] == minimum["summary_cell_id"]
        for issue in context._personal_detail_extraction_issues
    )


def test_registered_scalar_summary_rejects_cross_column_value_tokens() -> None:
    context = _summary_context("10,000")
    atoms = context.evidence_plane.evidence.text_atoms
    first_values = [atom for atom in atoms if str(atom["id"]).startswith("e:0:2:")]
    assert len(first_values) == 5
    # Preserve the joined OCR line and token count, but displace two source
    # atoms into the same physical header band. Positional splitting would
    # silently accept this; exact column ownership must reject the block.
    first_values[0]["bbox"] = list(first_values[1]["bbox"])

    records, cells = native_extraction._extract_summary_datasets(context)

    first_title = native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[0][0]
    assert not any(record.get("title") == first_title for record in records)
    assert not any(cell.get("title") == first_title for cell in cells)


def test_registered_scalar_summary_rejects_value_token_spanning_columns() -> None:
    context = _summary_context("10,000", block_order=(0,))
    atoms = context.evidence_plane.evidence.text_atoms
    first_value = next(atom for atom in atoms if atom["id"] == "e:0:2:0")
    first_value["bbox"][2] = atoms[[atom["id"] for atom in atoms].index("e:0:2:1")]["bbox"][2]

    records, cells = native_extraction._extract_summary_datasets(context)

    assert records == []
    assert cells == []


def test_registered_scalar_summaries_recover_fragmented_lines_and_exact_table_tokens() -> None:
    import copy

    context = _summary_context("0,000")
    source_lines = context.corrected_evidence_pages()[0]["lines"]
    first_title, first_headers = native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[0]
    second_title, second_headers = native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[1]
    card_title, card_headers = native_extraction._REGISTERED_SCALAR_SUMMARY_BLOCKS[2]
    # First block: every header/value is its own exact corrected line.
    fragmented: list[dict[str, object]] = [
        {"text": first_title, "bbox": [0, 0, 300, 8], "evidence_ids": ["first-title"]}
    ]
    for column, header in enumerate(first_headers):
        fragmented.append(
            {
                "text": header,
                "bbox": [column * 50, 10, column * 50 + 45, 18],
                "evidence_ids": [f"first-header-{column}"],
            }
        )
    for column, value in enumerate(("2", "5", "800,000", "545,493", "530")):
        fragmented.append(
            {
                "text": value,
                "bbox": [column * 50, 20, column * 50 + 45, 28],
                "evidence_ids": [f"first-value-{column}"],
            }
        )
    fragmented.extend(
        line
        for line in copy.deepcopy(source_lines)
        if line["text"] in {second_title, card_title}
    )

    def table_for(
        table_id: str,
        title_y: float,
        headers: tuple[str, ...],
        values: tuple[str, ...],
    ) -> SimpleNamespace:
        atoms: list[dict[str, object]] = []
        cells: list[list[SimpleNamespace]] = [[], []]
        column_width = 40.0
        for row, row_values in enumerate((headers, values)):
            for column, text in enumerate(row_values):
                atom_id = f"{table_id}:{row}:{column}"
                bbox = [column * column_width, title_y + row * 10, (column + 1) * column_width, title_y + row * 10 + 8]
                atoms.append({"id": atom_id, "text": text, "bbox": bbox})
                cells[row].append(
                    SimpleNamespace(
                        text=text,
                        bbox=bbox,
                        geometry_status="exact",
                        evidence_ids=[atom_id],
                        token_ids=[atom_id],
                        row_span=1,
                        col_span=1,
                    )
                )
        table = SimpleNamespace(
            table_id=table_id,
            bbox=[0, title_y, len(headers) * column_width, title_y + 20],
            source_cell_objects=cells,
            metadata={
                "canonical_template_id": "information_summary",
                "geometry": {
                    "row_bands": [
                        {"index": 0, "y0": title_y, "y1": title_y + 10},
                        {"index": 1, "y0": title_y + 10, "y1": title_y + 20},
                    ],
                    "col_bands": [
                        {"index": index, "x0": index * column_width, "x1": (index + 1) * column_width}
                        for index in range(len(headers))
                    ],
                }
            },
        )
        return table, atoms

    second_table, second_atoms = table_for(
        "second-summary",
        40.0,
        second_headers,
        ("3", "3", "614,000", "299,428", "120,495"),
    )
    card_table, card_atoms = table_for(
        "card-summary",
        70.0,
        card_headers,
        ("12", "16", "690,800", "226,000", "0,000", "26,101", "38,562"),
    )
    context.pages = [
        SimpleNamespace(
            page_number=3,
            source_page_number=2,
            canonical_template_id="information_summary",
            tables=[second_table, card_table],
        )
    ]
    context.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                *[
                    {
                        "id": evidence_id,
                        "text": line["text"],
                        "bbox": line["bbox"],
                    }
                    for line in fragmented
                    for evidence_id in line.get("evidence_ids", [])
                ],
                *second_atoms,
                *card_atoms,
            ]
        )
    )
    context.corrected_evidence_pages = lambda: [
        {"page": 3, "source_page": 2, "lines": fragmented}
    ]

    records, cells = native_extraction._extract_summary_datasets(context)

    assert len(records) == 3
    assert len(cells) == 17
    assert sum(cell["source_refs"][0]["geometry_scope"] == "token" for cell in cells) == 12
    assert next(
        cell
        for cell in cells
        if cell["title"] == card_title and cell["column_label"] == card_headers[4]
    )["value_status"] == "unreadable"
