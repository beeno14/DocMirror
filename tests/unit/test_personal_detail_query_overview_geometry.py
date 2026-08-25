# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_summary_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)

_G1 = "最近1个月内的查询机构数"
_G2 = "最近1个月内的查询次数"
_G3 = "最近2年内的查询次数"
_SCHEMA = {
    _G1: ("贷款审批", "信用卡审批"),
    _G2: ("贷款审批", "信用卡审批", "本人查询"),
    _G3: ("贷后管理", "担保资格审查", "特约商户实名审查"),
}


def _query_context(
    *,
    scale: float = 1.0,
    group_order: tuple[str, ...] = (_G1, _G2, _G3),
    reverse_leaves: bool = False,
    values: dict[str, str | None] | None = None,
    residue: str | None = None,
    page_role: str = "information_summary",
    table_role: str = "information_summary",
    duplicate_leaf_evidence: bool = False,
    crossing_value: bool = False,
) -> SimpleNamespace:
    widths = (31.0, 67.0, 43.0, 89.0, 37.0, 106.0, 52.0, 83.0)
    edges = [0.0]
    for width in widths:
        edges.append(edges[-1] + width * scale)
    ordered_groups: list[tuple[str, tuple[str, ...]]] = []
    for group in group_order:
        leaves = _SCHEMA[group]
        ordered_groups.append((group, tuple(reversed(leaves)) if reverse_leaves else leaves))
    paths = [(group, leaf) for group, leaves in ordered_groups for leaf in leaves]
    defaults = {f"{group}/{leaf}": str(index + 7) for index, (group, leaf) in enumerate(paths)}
    defaults.update(values or {})

    width = len(paths)
    rows = [["" for _column in range(width)] for _row in range(3)]
    bboxes: list[list[list[float] | None]] = []
    statuses = [["exact" for _column in range(width)] for _row in range(3)]
    evidence = [[[] for _column in range(width)] for _row in range(3)]
    token_ids = [[[] for _column in range(width)] for _row in range(3)]
    atoms: list[dict[str, object]] = []
    for row in range(3):
        bboxes.append(
            [
                [edges[column], row * 18.0 * scale, edges[column + 1], (row + 1) * 18.0 * scale]
                for column in range(width)
            ]
        )

    cursor = 0
    group_segments: list[tuple[int, int, str]] = []
    for group, leaves in ordered_groups:
        start = cursor
        end = start + len(leaves)
        group_segments.append((start, end, group))
        token_id = f"group:{start}"
        rows[0][start] = group
        evidence[0][start] = [token_id]
        token_ids[0][start] = [token_id]
        left, right = edges[start], edges[end]
        atoms.append(
            {
                "id": token_id,
                "text": group,
                "bbox": [left + (right - left) * 0.08, 3.0 * scale, right - (right - left) * 0.08, 14.0 * scale],
            }
        )
        cursor = end

    if residue is not None:
        residue_column = group_segments[1][1] - 1
        token_id = "group:residue"
        rows[0][residue_column] = (rows[0][residue_column] + " " + residue).strip()
        evidence[0][residue_column].append(token_id)
        token_ids[0][residue_column].append(token_id)
        atoms.append(
            {
                "id": token_id,
                "text": residue,
                "bbox": [
                    edges[residue_column] + 0.35 * (edges[residue_column + 1] - edges[residue_column]),
                    3.0 * scale,
                    edges[residue_column] + 0.55 * (edges[residue_column + 1] - edges[residue_column]),
                    12.0 * scale,
                ],
            }
        )

    prior_leaf_id: str | None = None
    for column, (group, leaf) in enumerate(paths):
        token_id = prior_leaf_id if duplicate_leaf_evidence and column == 1 else f"leaf:{column}"
        assert token_id is not None
        rows[1][column] = leaf
        evidence[1][column] = [token_id]
        token_ids[1][column] = [token_id]
        if not (duplicate_leaf_evidence and column == 1):
            atoms.append(
                {
                    "id": token_id,
                    "text": leaf,
                    "bbox": [
                        edges[column] + 0.12 * (edges[column + 1] - edges[column]),
                        21.0 * scale,
                        edges[column + 1] - 0.12 * (edges[column + 1] - edges[column]),
                        32.0 * scale,
                    ],
                }
            )
        prior_leaf_id = token_id

    spans: list[dict[str, object]] = []
    for start, end, _group in group_segments:
        span_ids: list[str] = []
        raw_values: list[str] = []
        for column in range(start, end):
            group, leaf = paths[column]
            raw = defaults[f"{group}/{leaf}"]
            if raw is None:
                continue
            token_id = f"value:{column}"
            span_ids.append(token_id)
            raw_values.append(raw)
            token_left = edges[column] + 0.36 * (edges[column + 1] - edges[column])
            token_right = edges[column] + 0.62 * (edges[column + 1] - edges[column])
            if crossing_value and column == 0:
                token_right = edges[column + 1] + 0.2 * (edges[column + 2] - edges[column + 1])
            atoms.append(
                {
                    "id": token_id,
                    "text": raw,
                    "bbox": [token_left, 39.0 * scale, token_right, 50.0 * scale],
                }
            )
        rows[2][start] = " ".join(raw_values)
        evidence[2][start] = span_ids
        token_ids[2][start] = list(span_ids)
        bboxes[2][start] = [edges[start], 36.0 * scale, edges[end], 54.0 * scale]
        if end - start > 1:
            spans.append(
                {
                    "row": 2,
                    "col": start,
                    "row_span": 1,
                    "col_span": end - start,
                    "bbox": list(bboxes[2][start]),
                    "evidence_ids": list(span_ids),
                }
            )
            for covered in range(start + 1, end):
                statuses[2][covered] = "derived"
                bboxes[2][covered] = None

    geometry = {
        "coordinate_system": "pdf_points_top_left",
        "row_bands": [
            {"index": row, "y0": row * 18.0 * scale, "y1": (row + 1) * 18.0 * scale}
            for row in range(3)
        ],
        "col_bands": [
            {"index": column, "x0": edges[column], "x1": edges[column + 1]}
            for column in range(width)
        ],
        "cell_bboxes": bboxes,
        "cell_geometry_status": statuses,
        "cell_evidence_ids": evidence,
        "cell_token_ids": token_ids,
        "cell_spans": spans,
    }
    table = SimpleNamespace(
        table_id="query-overview",
        bbox=[edges[0], 0.0, edges[-1], 54.0 * scale],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": rows,
            "canonical_template_id": table_role,
            "source_logical_page": 7,
            "source_page": 4,
            "source_cell_bboxes": bboxes,
            "cell_evidence_ids": evidence,
            "cell_token_ids": token_ids,
            "geometry": geometry,
        },
    )
    page = SimpleNamespace(
        page_number=7,
        source_page_number=4,
        canonical_template_id=page_role,
        tables=[table],
        texts=[],
    )
    return SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
        reading_order_by_logical={7: 1},
        tables_continue=lambda _left, _right: False,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def _append_exact_empty_border(context: SimpleNamespace) -> None:
    table = context.pages[0].tables[0]
    geometry = table.metadata["geometry"]
    last = geometry["col_bands"][-1]["x1"]
    border_right = last + 3.0
    border_column = len(geometry["col_bands"])
    geometry["col_bands"].append({"index": border_column, "x0": last, "x1": border_right})
    table.metadata["raw_rows"][0].append("")
    table.metadata["raw_rows"][1].append("")
    table.metadata["raw_rows"][2].append("")
    for key in ("cell_evidence_ids", "cell_token_ids"):
        for row in geometry[key]:
            row.append([])
    geometry["cell_geometry_status"][0].append("exact")
    geometry["cell_geometry_status"][1].append("exact")
    geometry["cell_geometry_status"][2].append("derived")
    geometry["cell_bboxes"][0].append([last, 0.0, border_right, 18.0])
    geometry["cell_bboxes"][1].append([last, 18.0, border_right, 54.0])
    geometry["cell_bboxes"][2].append(None)
    geometry["cell_spans"].append(
        {
            "row": 1,
            "col": border_column,
            "row_span": 2,
            "col_span": 1,
            "bbox": [last, 18.0, border_right, 54.0],
            "evidence_ids": [],
        }
    )
    table.bbox[2] = border_right


def _collapse_query_header_rows(context: SimpleNamespace) -> None:
    table = context.pages[0].tables[0]
    geometry = table.metadata["geometry"]
    original_bands = geometry["row_bands"]
    header_y0 = original_bands[0]["y0"]
    header_y1 = original_bands[1]["y1"]
    value_y0 = original_bands[2]["y0"]
    value_y1 = original_bands[2]["y1"]
    rows = table.metadata["raw_rows"]
    combined = [
        " ".join(value for value in (rows[0][column], rows[1][column]) if value)
        for column in range(len(rows[0]))
    ]
    table.metadata["raw_rows"] = [combined, rows[2]]
    combined_evidence = [
        [*geometry["cell_evidence_ids"][0][column], *geometry["cell_evidence_ids"][1][column]]
        for column in range(len(combined))
    ]
    combined_tokens = [
        [*geometry["cell_token_ids"][0][column], *geometry["cell_token_ids"][1][column]]
        for column in range(len(combined))
    ]
    combined_boxes = [
        [band["x0"], header_y0, band["x1"], header_y1]
        for band in geometry["col_bands"]
    ]
    geometry["row_bands"] = [
        {"index": 0, "y0": header_y0, "y1": header_y1},
        {"index": 1, "y0": value_y0, "y1": value_y1},
    ]
    geometry["cell_bboxes"] = [combined_boxes, geometry["cell_bboxes"][2]]
    geometry["cell_geometry_status"] = [
        ["exact" for _column in combined],
        geometry["cell_geometry_status"][2],
    ]
    geometry["cell_evidence_ids"] = [combined_evidence, geometry["cell_evidence_ids"][2]]
    geometry["cell_token_ids"] = [combined_tokens, geometry["cell_token_ids"][2]]
    geometry["cell_spans"] = [
        {**span, "row": 1}
        for span in geometry["cell_spans"]
        if span["row"] == 2
    ]
    table.metadata["source_cell_bboxes"] = geometry["cell_bboxes"]
    table.metadata["cell_evidence_ids"] = geometry["cell_evidence_ids"]
    table.metadata["cell_token_ids"] = geometry["cell_token_ids"]


def _add_leaf_header_residue(context: SimpleNamespace, *, column: int, residue: str) -> None:
    table = context.pages[0].tables[0]
    geometry = table.metadata["geometry"]
    token_id = f"leaf-residue:{column}"
    table.metadata["raw_rows"][1][column] = f"{residue} {table.metadata['raw_rows'][1][column]}"
    geometry["cell_evidence_ids"][1][column].append(token_id)
    geometry["cell_token_ids"][1][column].append(token_id)
    box = geometry["cell_bboxes"][1][column]
    context.evidence_plane.evidence.text_atoms.append(
        {
            "id": token_id,
            "text": residue,
            "bbox": [box[0] + 1.0, box[1] + 1.0, box[0] + 4.0, box[1] + 8.0],
        }
    )


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_query_overview_uses_semantic_groups_and_exact_token_geometry(scale: float) -> None:
    context = _query_context(
        scale=scale,
        group_order=(_G3, _G1, _G2),
        reverse_leaves=True,
        values={f"{_G3}/贷后管理": "83"},
    )

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert records[0]["summary_type"] == "查询记录概要"
    assert len(cells) == 8
    assert {cell["column_label"] for cell in cells} == {
        f"{group}/{leaf}" for group, leaves in _SCHEMA.items() for leaf in leaves
    }
    assert next(cell for cell in cells if cell["column_label"] == f"{_G3}/贷后管理")["value"] == "83"

    content = {
        "facts": {},
        "datasets": {
            "personal_detail_summary_records": records,
            "personal_detail_summary_cells": cells,
            "personal_detail_extraction_issues": context._personal_detail_extraction_issues,
        },
    }
    prepare_personal_detail_source_collections(content)
    metrics = content["datasets"]["personal_detail_credit_summary_metrics"]
    assert len(metrics) == 8
    assert {metric["mapping_status"] for metric in metrics} == {"mapped"}
    assert next(metric for metric in metrics if metric["metric_code"] == "recent_2y_inquiry_count_post_loan_management")[
        "numeric_value"
    ] == "83"
    projected = project_personal_detail_datasets(content["datasets"])
    overview = projected["credit_business_overview"]
    assert len(overview) == 8
    assert all(row.get("normalized", row)["mapping_status"] == "mapped" for row in overview)


def test_query_overview_trims_only_an_exact_empty_terminal_border() -> None:
    context = _query_context()
    _append_exact_empty_border(context)

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert len(cells) == 8
    assert records[0]["source_column_count"] == 8


def test_query_overview_isolates_non_han_residue_from_a_leaf_header() -> None:
    context = _query_context(group_order=(_G3, _G1, _G2))
    _add_leaf_header_residue(context, column=3, residue="3")

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert len(cells) == 8
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_query_overview_header_residue_unresolved"
    )
    assert issue["observed_value"] == {"residue": ["3"]}


@pytest.mark.parametrize("scale", [0.75, 1.5])
def test_query_overview_recovers_a_token_sealed_two_tier_collapsed_header(scale: float) -> None:
    context = _query_context(
        scale=scale,
        group_order=(_G2, _G3, _G1),
        reverse_leaves=True,
        residue="2",
    )
    _collapse_query_header_rows(context)

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert len(cells) == 8
    residue_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_query_overview_header_residue_unresolved"
    )
    assert residue_issue["status"] == "requires_review"
    assert residue_issue["observed_value"] == {"residue": ["2"]}


def test_query_overview_withholds_missing_and_noninteger_values_but_conserves_slots() -> None:
    context = _query_context(
        values={
            f"{_G1}/贷款审批": None,
            f"{_G1}/信用卡审批": "0.",
        },
        residue="W",
    )

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert len(cells) == 8
    unresolved = [cell for cell in cells if cell.get("value_status") == "unreadable"]
    assert [(cell["column_label"], cell["value"]) for cell in unresolved] == [
        (f"{_G1}/贷款审批", None),
        (f"{_G1}/信用卡审批", "0."),
    ]
    assert sum(
        issue["issue_code"] == "candidate_b_query_overview_value_unresolved"
        for issue in context._personal_detail_extraction_issues
    ) == 2
    residue_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_query_overview_header_residue_ignored"
    )
    assert residue_issue["status"] == "resolved"
    assert residue_issue["observed_value"] == {"residue": ["W"]}

    content = {
        "facts": {},
        "datasets": {
            "personal_detail_summary_records": records,
            "personal_detail_summary_cells": cells,
            "personal_detail_extraction_issues": context._personal_detail_extraction_issues,
        },
    }
    prepare_personal_detail_source_collections(content)
    metrics = content["datasets"]["personal_detail_credit_summary_metrics"]
    assert len(metrics) == 8
    assert {metric["mapping_status"] for metric in metrics} == {"mapped"}
    assert sum(metric["reporting_status"] == "unknown" for metric in metrics) == 2
    assert {
        metric["metric_code"]
        for metric in metrics
        if metric["reporting_status"] == "unknown"
    } == {
        "recent_1m_institution_count_loan_approval",
        "recent_1m_institution_count_credit_card_approval",
    }
    projected = project_personal_detail_datasets(content["datasets"])
    overview = [row.get("normalized", row) for row in projected["credit_business_overview"]]
    assert len(overview) == 8
    assert sum(row["reporting_status"] == "unknown" for row in overview) == 2


@pytest.mark.parametrize("connected", [True, False])
def test_query_overview_integer_tokens_require_one_connected_glyph_run(connected: bool) -> None:
    context = _query_context(values={f"{_G3}/贷后管理": "83"})
    table = context.pages[0].tables[0]
    geometry = table.metadata["geometry"]
    old_id = "value:5"
    old_atom = next(atom for atom in context.evidence_plane.evidence.text_atoms if atom["id"] == old_id)
    context.evidence_plane.evidence.text_atoms.remove(old_atom)
    left, top, right, bottom = old_atom["bbox"]
    midpoint = (left + right) / 2.0
    if connected:
        boxes = ([left, top, midpoint - 0.5, bottom], [midpoint + 0.5, top, right, bottom])
    else:
        column_box = geometry["cell_bboxes"][1][5]
        boxes = (
            [column_box[0] + 2.0, top, column_box[0] + 7.0, bottom],
            [column_box[2] - 7.0, top, column_box[2] - 2.0, bottom],
        )
    replacements = [("value:5:a", "8", boxes[0]), ("value:5:b", "3", boxes[1])]
    context.evidence_plane.evidence.text_atoms.extend(
        {"id": token_id, "text": text, "bbox": bbox}
        for token_id, text, bbox in replacements
    )
    for key in ("cell_evidence_ids", "cell_token_ids"):
        ids = geometry[key][2][5]
        ids[ids.index(old_id) : ids.index(old_id) + 1] = [item[0] for item in replacements]

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    post_loan = next(cell for cell in cells if cell["column_label"] == f"{_G3}/贷后管理")
    if connected:
        assert post_loan["value"] == "83"
        assert "value_status" not in post_loan
    else:
        assert post_loan["value_status"] == "unreadable"
        assert any(
            issue["issue_code"] == "candidate_b_query_overview_value_unresolved"
            and issue["target_record_id"] == post_loan["summary_cell_id"]
            for issue in context._personal_detail_extraction_issues
        )


@pytest.mark.parametrize(
    "mutation",
    ["foreign_owner", "han_residue", "duplicate_evidence", "crossing_value"],
)
def test_query_overview_near_misses_fail_closed(mutation: str) -> None:
    kwargs: dict[str, object] = {}
    if mutation == "foreign_owner":
        kwargs.update(page_role="public_information", table_role="public_information")
    elif mutation == "han_residue":
        kwargs["residue"] = "未知"
    elif mutation == "duplicate_evidence":
        kwargs["duplicate_leaf_evidence"] = True
    elif mutation == "crossing_value":
        kwargs["crossing_value"] = True
    context = _query_context(**kwargs)

    records, cells = _extract_summary_datasets(context)

    assert records == []
    assert cells == []
    if mutation != "foreign_owner":
        assert any(
            issue["issue_code"] == "candidate_b_query_overview_layout_unresolved"
            for issue in context._personal_detail_extraction_issues
        )
