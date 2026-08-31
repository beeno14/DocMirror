# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
    _withhold_repayment_plane_conflicts,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
    apply_candidate_b_native_status_conflict_guard,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import project_personal_detail_datasets


def _cell(
    text: str,
    bbox: list[float] | None,
    *,
    row: int,
    col: int,
    row_span: int = 1,
    exact: bool = True,
) -> SimpleNamespace:
    evidence_ids = [f"native:{row}:{col}"] if text else []
    return SimpleNamespace(
        text=text,
        bbox=bbox,
        row_index=row,
        col_index=col,
        row_span=row_span,
        col_span=1,
        geometry_status="exact" if exact else "derived",
        evidence_ids=evidence_ids,
        token_ids=list(evidence_ids),
        source_cell_refs=[
            {
                "page": 21,
                "table_id": "pt_21_0",
                "row": row,
                "raw_row": row,
                "col": col,
            }
        ],
    )


def _headerless_continuation_table() -> SimpleNamespace:
    edges = [40.0 + 20.0 * index for index in range(14)]
    row_bands = ((40.0, 55.0), (55.0, 68.0), (68.0, 83.0))
    year_and_prior_amounts = [
        _cell(
            "2021",
            [edges[0], row_bands[0][0], edges[1], row_bands[2][1]],
            row=0,
            col=0,
            row_span=3,
        )
    ] + [
        _cell(
            "0",
            [edges[col], row_bands[0][0], edges[col + 1], row_bands[0][1]],
            row=0,
            col=col,
        )
        for col in range(1, 13)
    ]
    statuses = [_cell("", None, row=1, col=0, exact=False)] + [
        _cell(
            "#" if col == 9 else "*",
            [edges[col], row_bands[1][0], edges[col + 1], row_bands[1][1]],
            row=1,
            col=col,
        )
        for col in range(1, 13)
    ]
    amounts = [_cell("", None, row=2, col=0, exact=False)] + [
        _cell(
            "0",
            [edges[col], row_bands[2][0], edges[col + 1], row_bands[2][1]],
            row=2,
            col=col,
        )
        for col in range(1, 13)
    ]
    rows = [
        SimpleNamespace(cells=year_and_prior_amounts),
        SimpleNamespace(cells=statuses),
        SimpleNamespace(cells=amounts),
    ]
    geometry = {
        "geometry_source": "scanned_image_line_grid",
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": [[cell.bbox for cell in row.cells] for row in rows],
        "cell_geometry_status": [
            [cell.geometry_status for cell in row.cells] for row in rows
        ],
        "cell_spans": [
            {"row": 0, "col": 0, "row_span": 3, "col_span": 1}
        ],
        "row_bands": [
            {"index": index, "y0": band[0], "y1": band[1]}
            for index, band in enumerate(row_bands)
        ],
        "col_bands": [
            {"index": col, "x0": edges[col], "x1": edges[col + 1]}
            for col in range(13)
        ],
        "vertical_lines": edges,
        "horizontal_lines": [40.0, 55.0, 68.0, 83.0],
    }
    return SimpleNamespace(
        table_id="pt_21_0",
        bbox=[edges[0], 40.0, edges[-1], 83.0],
        extraction_layer="scanned_image_line_grid",
        metadata={"geometry": geometry, "preserve_headers": False},
        headers=[],
        rows=rows,
    )


def _provenance() -> dict[str, Any]:
    return {
        "selection_basis": "source_table_year_plus_twelve_ownership",
        "source": "source_table_geometry",
        "reason": "exact_source_table_continuation_lattice",
        "table_id": "pt_21_0",
        "continuation_logical_page": 21,
        "vertical_rule_count": 14,
        "rule_count": 14,
        "column_count": 13,
        "month_column_count": 12,
        "status_row_index": 1,
        "amount_row_index": 2,
        "coordinate_system": "pdf_points_top_left",
        "value_inputs_used": False,
    }


def _record(month: int, *, status: str = "*") -> dict[str, Any]:
    left = 40.0 + 20.0 * month
    shared = {
        "page": 21,
        "logical_page": 21,
        "geometry_scope": "cell",
        "geometry_status": "exact",
        "coordinate_system": "pdf_points_top_left",
        "grid_id": "mg_p20_repayment_1",
        "col": month,
        "geometry_provenance": _provenance(),
    }
    return {
        "repayment_id": f"mg_p20_repayment_1:2021-{month:02d}",
        "grid_id": "mg_p20_repayment_1",
        "account_id": "credit_account:test",
        "year": 2021,
        "month": month,
        "status": status,
        "overdue_amount": "0",
        "source_cell_refs": [
            {
                **deepcopy(shared),
                "row": 2,
                "field_name": "status",
                "bbox": [left, 55.0, left + 20.0, 68.0],
            },
            {
                **deepcopy(shared),
                "row": 3,
                "field_name": "overdue_amount",
                "bbox": [left, 68.0, left + 20.0, 83.0],
            },
        ],
    }


def _case(
    *,
    preserve_headers: bool = False,
) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    table = _headerless_continuation_table()
    records = [_record(8), _record(9)]
    if preserve_headers:
        # Model PageAssembler's preserved raw row 0: it is stored as headers,
        # not as typed cells.  The three typed rows keep indices 0/1/2 while
        # their year/status/amount raw geometry rows become 1/2/3.
        table.metadata["preserve_headers"] = True
        table.headers = ["还款记录"] + [""] * 12
        table.bbox[1] = 25.0
        geometry = table.metadata["geometry"]
        geometry["cell_bboxes"].insert(0, [None] * 13)
        geometry["cell_geometry_status"].insert(0, ["missing"] * 13)
        for span in geometry["cell_spans"]:
            span["row"] += 1
        geometry["row_bands"] = [
            {"index": 0, "y0": 25.0, "y1": 40.0},
            *[{**band, "index": band["index"] + 1} for band in geometry["row_bands"]],
        ]
        geometry["horizontal_lines"].insert(0, 25.0)
        for row in table.rows:
            for cell in row.cells:
                cell.source_cell_refs[0]["raw_row"] += 1
        for record in records:
            for ref in record["source_cell_refs"]:
                ref["geometry_provenance"]["status_row_index"] += 1
                ref["geometry_provenance"]["amount_row_index"] += 1
    page = SimpleNamespace(
        page_number=21,
        source_page_number=11,
        tables=[table],
    )
    context = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[page]),
        source_page_by_logical={21: 11},
    )
    return context, records


@pytest.mark.parametrize("preserve_headers", (False, True))
def test_headerless_continuation_binds_hash_to_september_and_never_august(
    preserve_headers: bool,
) -> None:
    context, records = _case(preserve_headers=preserve_headers)
    status_cell = context.parse_result.pages[0].tables[0].rows[1].cells[9]
    source_offset = 1 if preserve_headers else 0
    assert status_cell.row_index == 1
    assert status_cell.source_cell_refs[0]["raw_row"] == 1 + source_offset

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "*"
    assert records[0]["overdue_amount"] == "0"
    assert "status" not in records[1]
    assert records[1]["overdue_amount"] == "0"
    assert records[1]["canonical_raw"]["status"] == ["*", "#"]
    assert audit["agreements"] == 1
    assert audit["conflicts_withheld"] == 1
    issue = context._personal_detail_extraction_issues[0]
    assert issue["target_record_id"] == "mg_p20_repayment_1:2021-09"
    assert issue["observed_value"] == {
        "corrected_final": "*",
        "sealed_native_source_cell": "#",
        "paired_status_amount": "0",
    }
    assert {(ref["row"], ref["col"]) for ref in issue["source_refs"][2:]} == {
        (1 + source_offset, 9),
        (2 + source_offset, 9),
    }
    assert all(
        ref["logical_page"] == 21 and ref["source_page"] == 11
        for ref in issue["source_refs"][2:]
    )


@pytest.mark.parametrize("typed_row", (0, 1, 2))
@pytest.mark.parametrize("defect", ("missing_ref", "wrong_raw_row", "duplicate_owner"))
def test_headerless_continuation_requires_owned_year_status_and_amount_rows(
    typed_row: int,
    defect: str,
) -> None:
    context, records = _case(preserve_headers=True)
    table = context.parse_result.pages[0].tables[0]
    cell = table.rows[typed_row].cells[0 if typed_row == 0 else 9]
    if defect == "missing_ref":
        cell.source_cell_refs = []
    elif defect == "wrong_raw_row":
        cell.source_cell_refs[0]["raw_row"] += 10
    elif defect == "duplicate_owner":
        competitor = deepcopy(cell)
        competitor.row_index = 99
        competitor.source_cell_refs[0]["row"] = 99
        table.rows.append(SimpleNamespace(cells=[competitor]))

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records[1:],
        enabled=True,
    )

    assert records[1]["status"] == "*"
    assert audit["unique_native_witnesses"] == 0
    assert audit["conflicts_withheld"] == 0
    assert not hasattr(context, "_personal_detail_extraction_issues")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("table_id", "pt_21_other"),
        ("status_row_index", 0),
        ("value_inputs_used", True),
        ("continuation_logical_page", 20),
    ],
)
def test_headerless_continuation_requires_exact_value_free_provenance(
    key: str,
    value: Any,
) -> None:
    context, records = _case()
    for ref in records[1]["source_cell_refs"]:
        ref["geometry_provenance"][key] = value

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records[1:],
        enabled=True,
    )

    assert records[1]["status"] == "*"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


def test_headerless_source_cell_never_promotes_an_unknown_final_status() -> None:
    context, records = _case()
    records[1]["status"] = "unknown"

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records[1:],
        enabled=True,
    )

    assert records[1]["status"] == "unknown"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["records_checked"] == 0


@pytest.mark.parametrize("preserve_headers", (False, True))
def test_real_cross_plane_withholding_keeps_headerless_september_cell_in_projection(preserve_headers: bool) -> None:
    context, records = _case(preserve_headers=preserve_headers)
    native = deepcopy(records[1])
    native["status"] = "#"
    corrected = records[1]
    _withhold_repayment_plane_conflicts(
        context, {"repayment_records": [native]}, {"repayment_records": [corrected]},
    )
    assert "status" not in corrected
    assert context._personal_detail_extraction_issues[0]["status"] == "requires_review"
    audit = apply_candidate_b_native_status_conflict_guard(context, [corrected], enabled=True)
    assert audit["preserved_source_plane_conflicts"] == 1
    native_issue = next(issue for issue in context._personal_detail_extraction_issues if issue["issue_code"] == "candidate_b_native_source_cell_repayment_status_conflict")
    assert native_issue["observed_value"] == {
        "corrected_final": "*", "sealed_native_source_cell": "#",
        "paired_status_amount": "0", "corrected_final_already_withheld": True,
    }
    projected = project_personal_detail_datasets({
        "repayment_records": [corrected],
        "personal_detail_extraction_issues": context._personal_detail_extraction_issues,
    })
    [monthly] = projected["credit_account_monthly_performance"]
    assert monthly["monthly_performance_id"] == "mg_p20_repayment_1:2021-09"
    assert monthly["status_code"] is None and monthly["status_amount"] == "0"
