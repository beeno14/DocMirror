from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
    apply_candidate_b_native_status_conflict_guard,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)


def _x_edges() -> list[float]:
    return [45.0, 73.0, *[100.0 + 27.0 * index for index in range(12)]]


def _cell(
    text: str,
    bbox: list[float] | None,
    *,
    row: int,
    col: int,
    row_span: int = 1,
    exact: bool = True,
) -> SimpleNamespace:
    evidence = [f"ocr:p19:r{row}:c{col}"] if text and bbox is not None else []
    return SimpleNamespace(
        text=text,
        bbox=bbox,
        row_index=row,
        col_index=col,
        row_span=row_span,
        col_span=1,
        geometry_status="exact" if exact else "derived",
        geometry_source="scanned_image_line_grid",
        evidence_ids=evidence,
        token_ids=evidence,
    )


def _native_table() -> SimpleNamespace:
    edges = _x_edges()
    header_y = (157.5, 170.5)
    status_y = (170.5, 183.5)
    amount_y = (183.5, 196.5)
    header_text = ["", "28", "2", "3", "4", "5", "6", "7", "T.8", "9", "10", "11", "12"]
    header_cells = [
        _cell(
            header_text[col],
            [edges[col], header_y[0], edges[col + 1], header_y[1]],
            row=0,
            col=col,
        )
        for col in range(13)
    ]
    status_cells = [
        _cell(
            "2022 кя",
            [edges[0], status_y[0], edges[1], amount_y[1]],
            row=1,
            col=0,
            row_span=2,
        )
    ] + [
        _cell(
            "W" if col == 1 else "" if col == 7 else "N",
            [edges[col], status_y[0], edges[col + 1], status_y[1]],
            row=1,
            col=col,
        )
        for col in range(1, 13)
    ]
    amount_cells = [_cell("", None, row=2, col=0, exact=False)] + [
        _cell(
            "0",
            [edges[col], amount_y[0], edges[col + 1], amount_y[1]],
            row=2,
            col=col,
        )
        for col in range(1, 13)
    ]
    rows = [
        SimpleNamespace(cells=header_cells),
        SimpleNamespace(cells=status_cells),
        SimpleNamespace(cells=amount_cells),
    ]
    cell_bboxes = [
        [cell.bbox for cell in header_cells],
        [cell.bbox for cell in status_cells],
        [cell.bbox for cell in amount_cells],
    ]
    cell_statuses = [
        [cell.geometry_status for cell in header_cells],
        [cell.geometry_status for cell in status_cells],
        [cell.geometry_status for cell in amount_cells],
    ]
    geometry = {
        "geometry_source": "scanned_image_line_grid",
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": cell_bboxes,
        "cell_geometry_status": cell_statuses,
        "cell_spans": [{"row": 1, "col": 0, "row_span": 2, "col_span": 1}],
        "row_bands": [
            {"index": 0, "y0": header_y[0], "y1": header_y[1]},
            {"index": 1, "y0": status_y[0], "y1": status_y[1]},
            {"index": 2, "y0": amount_y[0], "y1": amount_y[1]},
        ],
        "col_bands": [{"index": col, "x0": edges[col], "x1": edges[col + 1]} for col in range(13)],
        "vertical_lines": edges,
        "horizontal_lines": [header_y[0], header_y[1], status_y[1], amount_y[1]],
    }
    return SimpleNamespace(
        table_id="pt_19_0",
        bbox=[edges[0], header_y[0], edges[-1], amount_y[1]],
        extraction_layer="scanned_image_line_grid",
        metadata={"geometry": geometry},
        rows=rows,
    )


def _provenance(
    *,
    basis: str = "year_plus_twelve_rule_ownership",
) -> dict[str, str]:
    return {
        "selection_basis": basis,
        "source": (
            "source_table_geometry"
            if basis == "source_table_year_plus_twelve_ownership"
            else "vertical_rule_projection"
        ),
    }


def _final_refs(
    *,
    basis: str = "year_plus_twelve_rule_ownership",
) -> list[dict[str, Any]]:
    shared = {
        "page": 19,
        "logical_page": 19,
        "geometry_scope": "cell",
        "coordinate_system": "pdf_points_top_left",
        "grid_id": "mg_p19_repayment_0",
        "col": 8,
        "geometry_provenance": _provenance(basis=basis),
    }
    return [
        {
            **shared,
            "row": 2,
            "field_name": "status",
            "bbox": [262.0326, 169.2853, 289.0286, 183.9927],
        },
        {
            **shared,
            "row": 3,
            "field_name": "overdue_amount",
            "bbox": [262.0326, 187.1571, 289.0286, 196.5991],
        },
    ]


def _case(
    *,
    final_status: str = "M",
    basis: str = "year_plus_twelve_rule_ownership",
) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    page = SimpleNamespace(
        page_number=19,
        source_page_number=10,
        tables=[_native_table()],
    )
    context = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[page]),
        source_page_by_logical={19: 10},
    )
    records = [
        {
            "repayment_id": "mg_p19_repayment_0:2022-08",
            "grid_id": "mg_p19_repayment_0",
            "account_id": "credit_account:test",
            "year": 2022,
            "month": 8,
            "status": final_status,
            "overdue_amount": "0",
            "source_cell_refs": _final_refs(basis=basis),
        }
    ]
    return context, records


def _source_owned_p19_case() -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    """Reproduce p19's row-8 header and row-9/10 source-owned lattice."""

    context, records = _case(basis="source_table_year_plus_twelve_ownership")
    table = context.parse_result.pages[0].tables[0]
    target_rows = table.rows
    for offset, row in enumerate(target_rows, start=8):
        for cell in row.cells:
            cell.row_index = offset
    year_cell = target_rows[1].cells[0]
    year_cell.text = "2022 搜"
    year_cell.token_ids = ["ocr:p19:year:2022", "ocr:p19:year:noise"]
    year_cell.evidence_ids = list(year_cell.token_ids)

    table.rows = (
        [SimpleNamespace(cells=[]) for _ in range(8)]
        + target_rows
        + [SimpleNamespace(cells=[]) for _ in range(6)]
    )
    geometry = table.metadata["geometry"]
    missing_boxes = [[None] * 13 for _ in range(8)]
    missing_statuses = [["missing"] * 13 for _ in range(8)]
    trailing_boxes = [[None] * 13 for _ in range(6)]
    trailing_statuses = [["missing"] * 13 for _ in range(6)]
    geometry["cell_bboxes"] = (
        missing_boxes + geometry["cell_bboxes"] + trailing_boxes
    )
    geometry["cell_geometry_status"] = (
        missing_statuses + geometry["cell_geometry_status"] + trailing_statuses
    )
    geometry["cell_spans"] = [
        {"row": 9, "col": 0, "row_span": 2, "col_span": 1}
    ]
    row_edges = [53.5 + 13.0 * index for index in range(18)]
    geometry["row_bands"] = [
        {"index": index, "y0": row_edges[index], "y1": row_edges[index + 1]}
        for index in range(17)
    ]
    geometry["horizontal_lines"] = row_edges
    table.bbox = [table.bbox[0], row_edges[0], table.bbox[2], row_edges[-1]]

    provenance = {
        "selection_basis": "source_table_year_plus_twelve_ownership",
        "source": "source_table_geometry",
        "reason": "exact_source_table_month_lattice_calibration",
        "table_id": "pt_19_0",
        "vertical_rule_count": 14,
        "rule_count": 14,
        "horizontal_rule_count": 18,
        "column_count": 13,
        "month_column_count": 12,
        "status_row_index": 9,
        "amount_row_index": 10,
        "year_anchor_row_index": 9,
        "year_anchor_mode": "spanning_year_cell",
        "year_row_span": 2,
        "active_cell_geometry_exact": True,
        "active_cell_rule_derived_count": 0,
        "coordinate_system": "pdf_points_top_left",
        "value_inputs_used": False,
        "logical_page": 19,
    }
    refs = records[0]["source_cell_refs"]
    refs[0].update(
        {
            "geometry_status": "exact",
            "bbox": [262.0, 170.5, 289.0, 183.5],
            "geometry_provenance": deepcopy(provenance),
        }
    )
    refs[1].update(
        {
            "geometry_status": "exact",
            "bbox": [262.0, 183.5, 289.0, 196.5],
            "geometry_provenance": deepcopy(provenance),
        }
    )
    return context, records


def test_same_cell_native_conflict_withholds_only_status_and_conserves_all_evidence() -> None:
    context, records = _case()

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    record = records[0]
    assert "status" not in record
    assert record["overdue_amount"] == "0"
    assert record["canonical_raw"]["status"] == ["M", "N"]
    assert record["_unresolved_fields"] == ["status"]
    assert record["extraction_status"] == "review"
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == ("candidate_b_native_source_cell_repayment_status_conflict")
    assert issue["target_record_id"] == "mg_p19_repayment_0:2022-08"
    assert issue["field_name"] == "status_code"
    assert issue["observed_value"] == {
        "corrected_final": "M",
        "paired_status_amount": "0",
        "sealed_native_source_cell": "N",
    }
    assert [(ref["evidence_plane"], ref["field_name"]) for ref in issue["source_refs"]] == [
        ("corrected_final", "status"),
        ("corrected_final", "overdue_amount"),
        ("sealed_native_source_table", "status"),
        ("sealed_native_source_table", "overdue_amount"),
    ]
    assert audit["conflicts_withheld"] == 1
    assert audit["unique_native_witnesses"] == 1


def test_source_owned_base_page_binds_actual_p19_shape_with_damaged_year_text() -> None:
    context, records = _source_owned_p19_case()
    header_cells = context.parse_result.pages[0].tables[0].rows[8].cells
    year_cell = context.parse_result.pages[0].tables[0].rows[9].cells[0]
    assert header_cells[1].text == "28"
    assert header_cells[8].text == "T.8"
    assert year_cell.text == "2022 搜"
    assert len(year_cell.token_ids) == 2

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert "status" not in records[0]
    assert records[0]["overdue_amount"] == "0"
    assert records[0]["canonical_raw"]["status"] == ["M", "N"]
    assert audit["unique_native_witnesses"] == 1
    assert audit["conflicts_withheld"] == 1
    issue = context._personal_detail_extraction_issues[0]
    assert issue["target_record_id"] == "mg_p19_repayment_0:2022-08"
    assert {(ref["row"], ref["col"]) for ref in issue["source_refs"][2:]} == {
        (9, 8),
        (10, 8),
    }


def test_source_owned_base_page_never_uses_geometry_only_when_typed_header_is_missing() -> None:
    context, records = _source_owned_p19_case()
    context.parse_result.pages[0].tables[0].rows[8].cells = []

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


@pytest.mark.parametrize(
    "defect",
    (
        "logical_page_mismatch",
        "table_mismatch",
        "status_row_mismatch",
        "year_anchor_mismatch",
        "header_row_mismatch",
        "shifted_header_bbox",
        "contradictory_rule_counts",
    ),
)
def test_source_owned_base_page_requires_all_declared_and_raw_rows_to_agree(
    defect: str,
) -> None:
    context, records = _source_owned_p19_case()
    refs = records[0]["source_cell_refs"]
    table = context.parse_result.pages[0].tables[0]
    if defect == "logical_page_mismatch":
        refs[1]["geometry_provenance"]["logical_page"] = 18
    elif defect == "table_mismatch":
        refs[1]["geometry_provenance"]["table_id"] = "pt_19_other"
    elif defect == "status_row_mismatch":
        refs[1]["geometry_provenance"]["status_row_index"] = 8
    elif defect == "year_anchor_mismatch":
        refs[1]["geometry_provenance"]["year_anchor_row_index"] = 8
    elif defect == "header_row_mismatch":
        table.rows[8].cells[8].row_index = 7
    elif defect == "shifted_header_bbox":
        table.rows[8].cells[8].bbox = list(table.rows[8].cells[9].bbox)
    elif defect == "contradictory_rule_counts":
        for ref in refs:
            ref["geometry_provenance"]["vertical_rule_count"] = 0

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


def test_source_owned_base_page_requires_one_unique_physical_lattice() -> None:
    context, records = _source_owned_p19_case()
    duplicate = deepcopy(context.parse_result.pages[0].tables[0])
    duplicate.table_id = "pt_19_duplicate"
    context.parse_result.pages[0].tables.append(duplicate)

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


@pytest.mark.parametrize(
    "defect",
    (
        "nonexact_header",
        "nonexact_status",
        "nonexact_amount",
        "multitoken_status",
        "multitoken_amount",
        "missing_header_evidence",
        "reordered_header_evidence",
    ),
)
def test_source_owned_base_page_requires_exact_single_token_raw_month_cells(
    defect: str,
) -> None:
    context, records = _source_owned_p19_case()
    table = context.parse_result.pages[0].tables[0]
    if defect == "nonexact_header":
        table.rows[8].cells[8].geometry_status = "derived"
    elif defect == "nonexact_status":
        table.rows[9].cells[8].geometry_status = "derived"
    elif defect == "nonexact_amount":
        table.rows[10].cells[8].geometry_status = "derived"
    elif defect == "multitoken_status":
        table.rows[9].cells[8].token_ids.append("ocr:p19:status:noise")
        table.rows[9].cells[8].evidence_ids.append("ocr:p19:status:noise")
    elif defect == "multitoken_amount":
        table.rows[10].cells[8].token_ids.append("ocr:p19:amount:noise")
        table.rows[10].cells[8].evidence_ids.append("ocr:p19:amount:noise")
    elif defect == "missing_header_evidence":
        table.rows[8].cells[8].evidence_ids = []
    elif defect == "reordered_header_evidence":
        table.rows[8].cells[8].token_ids = ["ocr:p19:header:8", "ocr:p19:header:noise"]
        table.rows[8].cells[8].evidence_ids = ["ocr:p19:header:noise", "ocr:p19:header:8"]

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


@pytest.mark.parametrize("year_text", ("2021 搜", "2022 2021"))
def test_source_owned_base_page_requires_one_matching_year(year_text: str) -> None:
    context, records = _source_owned_p19_case()
    context.parse_result.pages[0].tables[0].rows[9].cells[0].text = year_text

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


@pytest.mark.parametrize("defect", ("missing", "reordered"))
def test_source_owned_base_page_requires_matching_year_evidence(defect: str) -> None:
    context, records = _source_owned_p19_case()
    year_cell = context.parse_result.pages[0].tables[0].rows[9].cells[0]
    if defect == "missing":
        year_cell.evidence_ids = []
    elif defect == "reordered":
        year_cell.evidence_ids = list(reversed(year_cell.evidence_ids))

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


def test_same_cell_agreement_is_silent_and_default_scope_is_disabled() -> None:
    agreement_context, agreement_records = _case(final_status="N")

    agreement_audit = apply_candidate_b_native_status_conflict_guard(
        agreement_context,
        agreement_records,
        enabled=True,
    )

    assert agreement_records[0]["status"] == "N"
    assert not hasattr(agreement_context, "_personal_detail_extraction_issues")
    assert agreement_audit["agreements"] == 1

    disabled_context, disabled_records = _case()
    disabled_audit = apply_candidate_b_native_status_conflict_guard(
        disabled_context,
        disabled_records,
    )
    assert disabled_records[0]["status"] == "M"
    assert not hasattr(disabled_context, "_personal_detail_extraction_issues")
    assert disabled_audit["enabled"] is False


@pytest.mark.parametrize("zero", ("0", 0, 0.0, Decimal("0")))
def test_numeric_zero_forms_preserve_native_conflict_detection(zero: Any) -> None:
    context, records = _case()
    records[0]["overdue_amount"] = zero

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert "status" not in records[0]
    assert records[0]["overdue_amount"] == zero
    assert audit["conflicts_withheld"] == 1


def test_native_digit_with_zero_amount_cannot_override_symbolic_final_status() -> None:
    context, records = _case(final_status="*")
    context.parse_result.pages[0].tables[0].rows[1].cells[8].text = "3"

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "*"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["unique_native_witnesses"] == 1
    assert audit["native_numeric_witnesses_rejected_for_nonpositive_amount"] == 1
    assert audit["conflicts_withheld"] == 0


def test_native_digit_with_positive_amount_remains_a_conflicting_witness() -> None:
    context, records = _case(final_status="*")
    table = context.parse_result.pages[0].tables[0]
    table.rows[1].cells[8].text = "3"
    table.rows[2].cells[8].text = "1"
    records[0]["overdue_amount"] = "1"

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert "status" not in records[0]
    assert records[0]["canonical_raw"]["status"] == ["*", "3"]
    assert audit["native_numeric_witnesses_rejected_for_nonpositive_amount"] == 0
    assert audit["conflicts_withheld"] == 1


def test_boolean_amount_is_not_coerced_to_numeric_zero() -> None:
    context, records = _case()
    records[0]["overdue_amount"] = False

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert audit["conflicts_withheld"] == 0


@pytest.mark.parametrize(
    ("existing", "expected"),
    (
        ("M", ["M", "N"]),
        ("Q", ["Q", "M", "N"]),
        (["Q", "M"], ["Q", "M", "N"]),
    ),
)
def test_conflict_merges_existing_canonical_raw_status_without_evidence_loss(
    existing: Any,
    expected: list[Any],
) -> None:
    context, records = _case()
    records[0]["canonical_raw"] = {
        "status": deepcopy(existing),
        "other": "preserved",
    }

    apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["canonical_raw"] == {
        "status": expected,
        "other": "preserved",
    }


@pytest.mark.parametrize(
    "defect",
    (
        "no_status_ref",
        "no_amount_ref",
        "misaligned_amount_ref",
        "ambiguous_amount_ref",
        "tiny_amount_ref",
        "low_native_coverage_amount_ref",
        "derived_status_ref",
        "wrong_provenance_source",
    ),
)
def test_final_ref_defects_fail_closed_without_mutation(defect: str) -> None:
    context, records = _case()
    refs = records[0]["source_cell_refs"]
    if defect == "no_status_ref":
        records[0]["source_cell_refs"] = [refs[1]]
    elif defect == "no_amount_ref":
        records[0]["source_cell_refs"] = [refs[0]]
    elif defect == "misaligned_amount_ref":
        refs[1]["bbox"] = [289.0, 187.1571, 316.0, 196.5991]
    elif defect == "ambiguous_amount_ref":
        alternate = deepcopy(refs[1])
        alternate["bbox"] = [262.2, 187.2, 289.0, 196.5]
        refs.append(alternate)
    elif defect == "tiny_amount_ref":
        refs[1]["bbox"] = [262.2, 187.2, 288.8, 188.2]
    elif defect == "low_native_coverage_amount_ref":
        refs[1]["bbox"] = [262.1, 188.5, 288.9, 196.5]
    elif defect == "derived_status_ref":
        refs[0]["geometry_status"] = "derived"
    elif defect == "wrong_provenance_source":
        refs[0]["geometry_provenance"]["source"] = "unowned_geometry"

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


@pytest.mark.parametrize(
    "defect",
    (
        "shifted_status",
        "bad_lattice",
        "duplicate_native",
        "native_amount_mismatch",
        "incomplete_header",
        "missing_typed_amount",
        "shifted_raw_status",
        "wrong_status_row_index",
        "wrong_amount_row_index",
    ),
)
def test_native_binding_defects_and_incomplete_rows_fail_closed(defect: str) -> None:
    context, records = _case()
    table = context.parse_result.pages[0].tables[0]
    if defect == "shifted_status":
        records[0]["source_cell_refs"][0]["bbox"] = [
            289.0,
            169.2853,
            316.0,
            183.9927,
        ]
    elif defect == "bad_lattice":
        table.metadata["geometry"]["vertical_lines"].append(410.0)
    elif defect == "duplicate_native":
        duplicate = deepcopy(table)
        duplicate.table_id = "pt_19_duplicate"
        context.parse_result.pages[0].tables.append(duplicate)
    elif defect == "native_amount_mismatch":
        table.rows[2].cells[8].text = "1"
    elif defect == "incomplete_header":
        table.rows[0].cells = [cell for cell in table.rows[0].cells if cell.col_index != 8]
    elif defect == "missing_typed_amount":
        table.rows[2].cells = [cell for cell in table.rows[2].cells if cell.col_index != 8]
    elif defect == "shifted_raw_status":
        table.rows[1].cells[8].bbox = [289.0, 170.5, 316.0, 183.5]
    elif defect == "wrong_status_row_index":
        table.rows[1].cells[8].row_index = 9
    elif defect == "wrong_amount_row_index":
        table.rows[2].cells[8].row_index = 9

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")
    assert audit["conflicts_withheld"] == 0


def test_incomplete_source_table_geometry_never_falls_back_to_header_scan() -> None:
    context, records = _case(basis="source_table_year_plus_twelve_ownership")

    apply_candidate_b_native_status_conflict_guard(context, records, enabled=True)

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_mismatched_source_table_bindings_never_fall_back_to_header_scan() -> None:
    context, records = _case(basis="source_table_year_plus_twelve_ownership")
    for ref in records[0]["source_cell_refs"]:
        ref["geometry_provenance"].update(
            {
                "table_id": "pt_19_0",
                "continuation_logical_page": 19,
                "vertical_rule_count": 14,
                "column_count": 13,
                "month_column_count": 12,
                "status_row_index": 1,
                "amount_row_index": 2,
                "coordinate_system": "pdf_points_top_left",
                "value_inputs_used": False,
            }
        )
    records[0]["source_cell_refs"][1]["geometry_provenance"]["table_id"] = (
        "pt_19_other"
    )

    apply_candidate_b_native_status_conflict_guard(context, records, enabled=True)

    assert records[0]["status"] == "M"
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_conflict_survives_personal_detail_community_projection() -> None:
    context, records = _source_owned_p19_case()
    apply_candidate_b_native_status_conflict_guard(context, records, enabled=True)

    projected = project_personal_detail_datasets(
        {
            "repayment_records": records,
            "personal_detail_extraction_issues": list(context._personal_detail_extraction_issues),
        }
    )

    monthly_rows = projected["credit_account_monthly_performance"]
    assert len(monthly_rows) == 1
    monthly = monthly_rows[0]
    assert monthly["monthly_performance_id"] == "mg_p19_repayment_0:2022-08"
    assert monthly["status_code"] is None
    assert monthly["status_amount"] == "0"
    assert monthly["extraction_status"] == "review"
    assert monthly["review"]["status"] == "requires_review"

    issues = projected["extraction_issues"]
    issue = next(
        row
        for row in issues
        if row.get("issue_code")
        == "candidate_b_native_source_cell_repayment_status_conflict"
    )
    assert issue["target_record_id"] == "mg_p19_repayment_0:2022-08"
    assert issue["field_name"] == "status_code"
    assert {ref["field_name"] for ref in issue["source_refs"]} == {
        "status",
        "overdue_amount",
    }
    issue_evidence = [
        row
        for row in projected["extraction_issue_evidence"]
        if row["extraction_issue_id"] == issue["extraction_issue_id"]
    ]
    assert issue_evidence
    assert all(row["source"] == {"page_range": [19, 19]} for row in issue_evidence)
    assert all(
        "source_refs" not in row and "source_cell_refs" not in row
        for row in issue_evidence
    )
    paired_amount = next(
        row
        for row in issue_evidence
        if row["evidence_kind"] == "observed"
        and row["evidence_path"] == "paired_status_amount"
    )
    assert paired_amount["string_value"] == "0"


@pytest.mark.parametrize(
    "defect",
    (
        "wrong_issue_code",
        "wrong_target_record",
        "wrong_target_dataset",
        "wrong_target_field",
        "inactive_issue",
        "no_amount",
        "negative_amount",
        "nonfinite_amount",
        "invalid_amount",
        "account_missing",
        "grid_missing",
        "performance_month_missing",
    ),
)
def test_conflict_row_retention_requires_every_exact_bounded_proof(defect: str) -> None:
    context, records = _case()
    apply_candidate_b_native_status_conflict_guard(context, records, enabled=True)
    issue = context._personal_detail_extraction_issues[0]

    if defect == "wrong_issue_code":
        issue["issue_code"] = "candidate_b_monthly_status_grid_unresolved"
    elif defect == "wrong_target_record":
        issue["target_record_id"] = "mg_p19_repayment_0:2022-09"
    elif defect == "wrong_target_dataset":
        issue["target_dataset"] = "postpaid_payment_history"
    elif defect == "wrong_target_field":
        issue["field_name"] = "status"
    elif defect == "inactive_issue":
        issue["status"] = "resolved"
    elif defect == "no_amount":
        records[0].pop("overdue_amount")
    elif defect == "negative_amount":
        records[0]["overdue_amount"] = "-1"
    elif defect == "nonfinite_amount":
        records[0]["overdue_amount"] = "Infinity"
    elif defect == "invalid_amount":
        records[0]["overdue_amount"] = "not-an-amount"
    elif defect == "account_missing":
        records[0].pop("account_id")
    elif defect == "grid_missing":
        records[0].pop("grid_id")
    elif defect == "performance_month_missing":
        records[0].pop("year")

    projected = project_personal_detail_datasets(
        {
            "repayment_records": records,
            "personal_detail_extraction_issues": list(
                context._personal_detail_extraction_issues
            ),
        }
    )

    assert projected.get("credit_account_monthly_performance", []) == []
    assert any(
        row.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        for row in projected["extraction_issues"]
    )
