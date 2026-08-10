# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
    apply_candidate_b_native_status_conflict_guard,
)


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
        metadata={"geometry": geometry},
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


def _case() -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    page = SimpleNamespace(
        page_number=21,
        source_page_number=11,
        tables=[_headerless_continuation_table()],
    )
    context = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[page]),
        source_page_by_logical={21: 11},
    )
    return context, [_record(8), _record(9)]


def test_headerless_continuation_binds_hash_to_september_and_never_august() -> None:
    context, records = _case()

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
        (1, 9),
        (2, 9),
    }


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
