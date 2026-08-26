# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-side ownership checks for canonical PBOC tables.

Printed labels describe a schema; they do not identify the section that owns a
table.  Canonical projection supplies that ownership.  Consumers use this
module instead of falling back from a table to its page, especially on physical
pages containing more than one PBOC section.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from docmirror.plugins.credit_report.personal_detail_scanned.agreement_ocr import (
    CREDIT_AGREEMENT_CARD_HEADING_RE,
)
from docmirror.plugins.credit_report.personal_detail_scanned.section_headings import (
    REGISTERED_SECTION_TEMPLATE_BY_TITLE,
    canonical_account_family_heading,
    canonical_registered_section_heading,
    canonical_registered_subsection_heading,
)

MIXED_PBOC_PAGE_TEMPLATE_ID = "mixed_pboc_sections"
REGISTERED_PBOC_TABLE_ROLES = frozenset(
    REGISTERED_SECTION_TEMPLATE_BY_TITLE.values()
)
INQUIRY_AUTHORITY_CLOSED_PHYSICAL_ORDINAL = "closed_physical_ordinal"
INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY = "schema_carry_only"
_INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF = (
    "exact_printed_footer_schema_carry_bridge"
)

_SAME_PAGE_SECTION_BINDING = "exact_pboc_section_heading_and_table_schema"
_CROSS_PAGE_AGREEMENT_BINDING = (
    "terminal_prior_page_agreement_anchor_and_leading_exact_table"
)
_INQUIRY_SEED_BINDING = "exact_inquiry_subsection_and_bounded_header_residue"
_INQUIRY_CONTINUATION_BINDING = "authoritative_prior_inquiry_table_continuation"
_INQUIRY_HEADER_RESIDUE_RE = re.compile(
    r"^(?:\?查询日期查询机构X|\?查询日期查询机构|查询日期查询机构X)$"
)
_INQUIRY_ROLE_COLUMNS = {
    "sequence": 0,
    "inquiry_date": 1,
    "institution": 2,
    "reason": 3,
}
_PRINTED_PAGE_RE = re.compile(r"^第\s*\d+\s*页\s*[，,]\s*共\s*\d+\s*页$")
_PRINTED_PAGE_CAPTURE_RE = re.compile(
    r"^第\s*(?P<page>\d+)\s*页\s*[，,]\s*共\s*(?P<total>\d+)\s*页$"
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).rstrip("：:")


def _positive_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        box = tuple(float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _same_box(left: Any, right: Any) -> bool:
    left_box = _bbox(left)
    right_box = _bbox(right)
    return bool(
        left_box is not None
        and right_box is not None
        and all(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
            for a, b in zip(left_box, right_box, strict=True)
        )
    )


def _unique_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    ids = tuple(str(item) for item in value if str(item or ""))
    if not ids or len(ids) != len(set(ids)):
        return None
    return ids


def _raw_rows(table: Any) -> list[list[str]]:
    metadata = getattr(table, "metadata", None) or {}
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, Mapping) else None
    if isinstance(raw_rows, list) and raw_rows:
        return [
            [str(cell or "") for cell in row]
            for row in raw_rows
            if isinstance(row, list)
        ]
    headers = [str(value or "") for value in getattr(table, "headers", None) or ()]
    body = [
        [str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", None) or ()]
        for row in getattr(table, "rows", None) or ()
    ]
    return ([headers] if headers else []) + body


def _exact_cell_evidence(table: Any) -> dict[tuple[int, int], tuple[str, ...]] | None:
    """Resolve exact evidence IDs and reject conflicting metadata copies."""

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    geometry = metadata.get("geometry")
    owners = tuple(
        candidate
        for candidate in (geometry, metadata)
        if isinstance(candidate, Mapping)
    )
    observed: dict[tuple[int, int], set[tuple[str, ...]]] = {}
    for owner in owners:
        statuses = owner.get("cell_geometry_status")
        evidence = owner.get("cell_evidence_ids")
        if not isinstance(statuses, list) or not isinstance(evidence, list):
            continue
        for row_index, status_row in enumerate(statuses):
            if (
                not isinstance(status_row, list)
                or row_index >= len(evidence)
                or not isinstance(evidence[row_index], list)
            ):
                continue
            for column, status in enumerate(status_row):
                if str(status or "") != "exact" or column >= len(evidence[row_index]):
                    continue
                ids = _unique_ids(evidence[row_index][column])
                if ids is not None:
                    observed.setdefault((row_index, column), set()).add(ids)
    if not observed or any(len(candidates) != 1 for candidates in observed.values()):
        return None
    return {key: next(iter(candidates)) for key, candidates in observed.items()}


def _schema_owner_is_consistent(table: Any, owner: Mapping[str, Any]) -> bool:
    table_id = str(getattr(table, "table_id", "") or "")
    rows = _raw_rows(table)
    raw_header_row = owner.get("header_row")
    header_row = (
        _positive_int(raw_header_row + 1)
        if isinstance(raw_header_row, int) and not isinstance(raw_header_row, bool)
        else None
    )
    raw_witness_row = owner.get("population_witness_row")
    witness_row = (
        _positive_int(raw_witness_row + 1)
        if isinstance(raw_witness_row, int) and not isinstance(raw_witness_row, bool)
        else None
    )
    if (
        not table_id
        or str(owner.get("table_id") or "") != table_id
        or header_row is None
        or witness_row is None
    ):
        return False
    header_index = header_row - 1
    witness_index = witness_row - 1
    if not (0 <= header_index < witness_index < len(rows)):
        return False
    populated = [
        (column, _compact(value))
        for column, value in enumerate(rows[header_index])
        if _compact(value)
    ]
    labels = tuple(value for _column, value in populated)
    owner_labels = owner.get("header_labels")
    if (
        not labels
        or len(labels) != len(set(labels))
        or not isinstance(owner_labels, list)
        or tuple(_compact(value) for value in owner_labels) != labels
        or any(
            column >= len(rows[witness_index])
            or not _compact(rows[witness_index][column])
            for column, _label in populated
        )
        or _bbox(owner.get("table_bbox")) is None
    ):
        return False
    exact = _exact_cell_evidence(table)
    if exact is None:
        return False
    selected: list[str] = []
    for row_index in (header_index, witness_index):
        for column, _label in populated:
            ids = exact.get((row_index, column))
            if ids is None:
                return False
            selected.extend(ids)
    owner_ids = _unique_ids(owner.get("evidence_ids"))
    owner_box = _bbox(owner.get("table_bbox"))
    source_cell_boxes = metadata_source_cell_boxes(table)
    if owner_box is None or source_cell_boxes is None:
        return False
    selected_boxes = [
        source_cell_boxes.get((row_index, column))
        for row_index in (header_index, witness_index)
        for column, _label in populated
    ]
    if any(box is None for box in selected_boxes):
        return False
    if any(
        box[0] < owner_box[0] - 1e-6
        or box[1] < owner_box[1] - 1e-6
        or box[2] > owner_box[2] + 1e-6
        or box[3] > owner_box[3] + 1e-6
        for box in selected_boxes
        if box is not None
    ):
        return False
    return bool(
        owner_ids is not None
        and len(selected) == len(set(selected))
        and set(owner_ids) == set(selected)
    )


def metadata_source_cell_boxes(
    table: Any,
) -> dict[tuple[int, int], tuple[float, float, float, float]] | None:
    """Resolve one non-conflicting source-coordinate cell grid."""

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    geometry = metadata.get("geometry")
    grids = [
        candidate
        for candidate in (
            metadata.get("source_cell_bboxes"),
            geometry.get("source_cell_bboxes") if isinstance(geometry, Mapping) else None,
            geometry.get("cell_bboxes") if isinstance(geometry, Mapping) else None,
        )
        if isinstance(candidate, list)
    ]
    if not grids:
        return None
    resolved: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for grid in grids:
        for row, cells in enumerate(grid):
            if not isinstance(cells, list):
                continue
            for column, raw_box in enumerate(cells):
                box = _bbox(raw_box)
                if box is None:
                    continue
                prior = resolved.get((row, column))
                if prior is not None and not _same_box(prior, box):
                    return None
                resolved[(row, column)] = box
    return resolved or None


def _text_values(text_item: Any) -> tuple[str, Any, tuple[str, ...] | None]:
    if isinstance(text_item, Mapping):
        text = text_item.get("content") or text_item.get("text") or ""
        box = text_item.get("source_bbox") or text_item.get("bbox")
        ids = _unique_ids(text_item.get("evidence_ids"))
    else:
        text = getattr(text_item, "content", "") or getattr(text_item, "text", "")
        box = getattr(text_item, "source_bbox", None) or getattr(text_item, "bbox", None)
        ids = _unique_ids(getattr(text_item, "evidence_ids", None))
    return str(text), box, ids


def _heading_role(value: Any) -> tuple[str, str] | None:
    title = canonical_registered_section_heading(value)
    if title is not None:
        return REGISTERED_SECTION_TEMPLATE_BY_TITLE[title], title
    subsection = canonical_registered_subsection_heading(value)
    if subsection is not None:
        return subsection
    account_family = canonical_account_family_heading(value)
    if account_family is not None:
        return "credit_account_detail", _compact(value)
    return None


def _same_page_owner_is_consistent(
    page: Any,
    owner: Mapping[str, Any],
    *,
    role: str,
) -> bool:
    heading_ids = _unique_ids(owner.get("heading_evidence_ids"))
    owner_ids = _unique_ids(owner.get("evidence_ids"))
    heading_box = _bbox(owner.get("heading_bbox"))
    if (
        heading_ids is None
        or owner_ids is None
        or heading_box is None
        or set(heading_ids).intersection(owner_ids)
    ):
        return False
    candidates: list[tuple[str, str]] = []
    for text_item in getattr(page, "texts", None) or ():
        text, box, ids = _text_values(text_item)
        resolved = _heading_role(text)
        if (
            ids is not None
            and set(ids) == set(heading_ids)
            and _same_box(box, heading_box)
            and resolved is not None
        ):
            candidates.append(resolved)
    return bool(
        len(candidates) == 1
        and candidates[0][0] == role
        and _compact(candidates[0][1]) == _compact(owner.get("heading_title"))
    )


def _ordered_page_rank(context: Any, logical_page: int) -> int | None:
    order = getattr(context, "reading_order_by_logical", None)
    if not isinstance(order, Mapping):
        return None
    rank = order.get(logical_page)
    if not isinstance(rank, int) or isinstance(rank, bool):
        return None
    return rank


def _cross_page_owner_is_consistent(
    context: Any,
    page: Any,
    table: Any,
    owner: Mapping[str, Any],
) -> bool:
    if context is None or str(owner.get("template_id") or "") != "credit_agreement":
        return False
    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return False
    current_logical = _positive_int(metadata.get("source_logical_page"))
    prior_logical = _positive_int(owner.get("heading_source_logical_page"))
    sequence = _positive_int(owner.get("printed_sequence"))
    heading_ids = _unique_ids(owner.get("heading_evidence_ids"))
    owner_ids = _unique_ids(owner.get("evidence_ids"))
    heading_box = _bbox(owner.get("heading_bbox"))
    if (
        current_logical is None
        or current_logical != _positive_int(getattr(page, "page_number", None))
        or prior_logical is None
        or prior_logical == current_logical
        or sequence is None
        or heading_ids is None
        or owner_ids is None
        or heading_box is None
        or set(heading_ids).intersection(owner_ids)
    ):
        return False
    prior_rank = _ordered_page_rank(context, prior_logical)
    current_rank = _ordered_page_rank(context, current_logical)
    if prior_rank is None or current_rank is None or current_rank != prior_rank + 1:
        return False
    prior_pages = [
        candidate
        for candidate in getattr(context, "pages", None) or ()
        if _positive_int(getattr(candidate, "page_number", None)) == prior_logical
    ]
    if (
        len(prior_pages) != 1
        or str(getattr(prior_pages[0], "canonical_template_id", "") or "")
        != "credit_agreement"
    ):
        return False
    prior_page = prior_pages[0]
    section_heading_owners: list[tuple[str, ...]] = []
    for text_item in getattr(prior_page, "texts", None) or ():
        text, box, ids = _text_values(text_item)
        if (
            _bbox(box) is not None
            and ids is not None
            and canonical_registered_section_heading(text) == "授信协议信息"
        ):
            section_heading_owners.append(ids)
    if len(section_heading_owners) != 1:
        return False
    anchors: list[tuple[float, int, tuple[str, ...], tuple[float, float, float, float]]] = []
    for text_item in getattr(prior_page, "texts", None) or ():
        text, box, ids = _text_values(text_item)
        match = CREDIT_AGREEMENT_CARD_HEADING_RE.fullmatch(_compact(text))
        exact_box = _bbox(box)
        if match is not None and exact_box is not None and ids is not None:
            anchors.append((exact_box[1], int(match.group("sequence")), ids, exact_box))
    anchors.sort(key=lambda item: item[0])
    sequences = [item[1] for item in anchors]
    all_anchor_ids = [evidence_id for item in anchors for evidence_id in item[2]]
    matching = [
        item
        for item in anchors
        if item[1] == sequence
        and set(item[2]) == set(heading_ids)
        and _same_box(item[3], heading_box)
    ]
    if (
        len(anchors) < 2
        or len(sequences) != len(set(sequences))
        or sequences != list(range(sequences[0], sequences[0] + len(sequences)))
        or len(all_anchor_ids) != len(set(all_anchor_ids))
        or set(section_heading_owners[0]).intersection(
            {*all_anchor_ids, *owner_ids}
        )
        or len(matching) != 1
        or matching[0] is not anchors[-1]
    ):
        return False
    for text_item in getattr(prior_page, "texts", None) or ():
        text, box, _ids = _text_values(text_item)
        exact_box = _bbox(box)
        if exact_box is None or exact_box[1] < heading_box[3] - 1e-6:
            continue
        if not _PRINTED_PAGE_RE.fullmatch(re.sub(r"\s+", "", text)):
            return False

    table_box = _bbox(getattr(table, "bbox", None))
    other_table_tops = [
        candidate_box[1]
        for candidate in getattr(page, "tables", None) or ()
        if candidate is not table
        and (candidate_box := _bbox(getattr(candidate, "bbox", None))) is not None
    ]
    if table_box is None or any(top <= table_box[1] + 1e-6 for top in other_table_tops):
        return False
    following_headings: list[float] = []
    for text_item in getattr(page, "texts", None) or ():
        text, box, ids = _text_values(text_item)
        exact_box = _bbox(box)
        if ids is not None and exact_box is not None and _heading_role(text) is not None:
            following_headings.append(exact_box[1])
    return bool(following_headings and table_box[3] <= min(following_headings) + 1e-6)


def _exact_cell_statuses(table: Any) -> dict[tuple[int, int], str] | None:
    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    geometry = metadata.get("geometry")
    owners = tuple(
        candidate
        for candidate in (geometry, metadata)
        if isinstance(candidate, Mapping)
    )
    observed: dict[tuple[int, int], set[str]] = {}
    for owner in owners:
        statuses = owner.get("cell_geometry_status")
        if not isinstance(statuses, list):
            continue
        for row, values in enumerate(statuses):
            if not isinstance(values, list):
                continue
            for column, value in enumerate(values):
                status = str(value or "")
                if status:
                    observed.setdefault((row, column), set()).add(status)
    if not observed or any(len(values) != 1 for values in observed.values()):
        return None
    return {key: next(iter(values)) for key, values in observed.items()}


def _boxes_have_disjoint_interiors(
    boxes: Sequence[tuple[float, float, float, float]],
) -> bool:
    return all(
        max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        <= 1e-8
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
    )


def _exact_owner_cell_lattice(
    table: Any,
) -> tuple[
    dict[tuple[int, int], tuple[str, ...]],
    dict[tuple[int, int], tuple[float, float, float, float]],
    dict[tuple[int, int], str],
] | None:
    evidence = _exact_cell_evidence(table)
    boxes = metadata_source_cell_boxes(table)
    statuses = _exact_cell_statuses(table)
    if evidence is None or boxes is None or statuses is None:
        return None
    exact_boxes = [
        box
        for key, box in boxes.items()
        if statuses.get(key) == "exact"
    ]
    if not exact_boxes or not _boxes_have_disjoint_interiors(exact_boxes):
        return None
    return evidence, boxes, statuses


def _derived_span_coverage(
    table: Any,
    exact_evidence: Mapping[tuple[int, int], tuple[str, ...]],
    boxes: Mapping[tuple[int, int], tuple[float, float, float, float]],
    statuses: Mapping[tuple[int, int], str],
) -> set[tuple[int, int]] | None:
    metadata = getattr(table, "metadata", None) or {}
    geometry = metadata.get("geometry") if isinstance(metadata, Mapping) else None
    spans = geometry.get("cell_spans") if isinstance(geometry, Mapping) else None
    if not isinstance(spans, list):
        return set()
    covered: set[tuple[int, int]] = set()
    for span in spans:
        if not isinstance(span, Mapping):
            return None
        row = span.get("row")
        column = span.get("col")
        row_span = span.get("row_span")
        column_span = span.get("col_span")
        if (
            not isinstance(row, int)
            or isinstance(row, bool)
            or not isinstance(column, int)
            or isinstance(column, bool)
            or not isinstance(row_span, int)
            or isinstance(row_span, bool)
            or not isinstance(column_span, int)
            or isinstance(column_span, bool)
            or row < 0
            or column < 0
            or row_span <= 0
            or column_span <= 1
            or statuses.get((row, column)) != "exact"
            or (row, column) not in boxes
            or _unique_ids(span.get("evidence_ids"))
            != exact_evidence.get((row, column))
        ):
            return None
        span_box = span.get("bbox")
        if span_box is not None and not _same_box(span_box, boxes[(row, column)]):
            return None
        for covered_row in range(row, row + row_span):
            for covered_column in range(column, column + column_span):
                key = (covered_row, covered_column)
                if key == (row, column):
                    continue
                if (
                    key in covered
                    or statuses.get(key) != "derived"
                    or key in boxes
                    or key in exact_evidence
                ):
                    return None
                covered.add(key)
    return covered


def _owner_anomalies(
    owner: Mapping[str, Any],
    rows: Sequence[Sequence[str]],
    *,
    body_start: int,
    population_start: int,
    sequence_column: int,
) -> list[dict[str, Any]] | None:
    expected = [
        {
            "row": row_index,
            "expected_sequence": population_start + offset,
            "raw_sequence": raw_sequence,
            "status": (
                "physical_field_omission"
                if not raw_sequence
                else "unparsed_raw_sequence"
            ),
        }
        for offset, row_index in enumerate(range(body_start, len(rows)))
        for raw_sequence in (_compact(rows[row_index][sequence_column]),)
        if raw_sequence != str(population_start + offset)
    ]
    observed = owner.get("sequence_field_anomalies")
    return expected if isinstance(observed, list) and observed == expected else None


def _owner_role_columns(owner: Mapping[str, Any]) -> dict[str, int] | None:
    raw = owner.get("inquiry_role_columns")
    if not isinstance(raw, Mapping) or dict(raw) != _INQUIRY_ROLE_COLUMNS:
        return None
    labels = owner.get("header_labels")
    if not isinstance(labels, list) or tuple(_compact(value) for value in labels) != (
        "编号",
        "查询日期",
        "查询机构",
        "查询原因",
    ):
        return None
    return dict(_INQUIRY_ROLE_COLUMNS)


def _owner_evidence_is_complete(
    owner: Mapping[str, Any],
    exact_evidence: Mapping[tuple[int, int], tuple[str, ...]],
    selected_cells: Sequence[tuple[int, int]],
) -> bool:
    selected = [
        evidence_id
        for cell in selected_cells
        for evidence_id in exact_evidence.get(cell, ())
    ]
    owner_ids = _unique_ids(owner.get("evidence_ids"))
    return bool(
        owner_ids is not None
        and selected
        and len(selected) == len(set(selected))
        and set(owner_ids) == set(selected)
    )


def _bounded_inquiry_seed_owner_is_consistent(
    page: Any,
    table: Any,
    owner: Mapping[str, Any],
) -> dict[str, Any] | None:
    rows = _raw_rows(table)
    roles = _owner_role_columns(owner)
    lattice = _exact_owner_cell_lattice(table)
    population_start = _positive_int(owner.get("population_start"))
    population_endpoint = _positive_int(owner.get("population_endpoint"))
    if (
        roles is None
        or lattice is None
        or population_start != 1
        or population_endpoint != len(rows) - 1
        or len(rows) < 2
        or any(len(row) != 4 for row in rows)
        or owner.get("header_row") != 0
        or owner.get("header_rows") != [0]
        or owner.get("population_witness_row") != 1
        or owner.get("population_endpoint_row") != len(rows) - 1
        or owner.get("header_binding")
        != "exact_bounded_residue_collapsed_header_lattice"
        or _compact(rows[0][0]) != "编号"
        or _INQUIRY_HEADER_RESIDUE_RE.fullmatch(_compact(rows[0][1])) is None
        or _compact(rows[0][2])
        or _compact(rows[0][3]) != "查询原因"
        or _compact(rows[1][0]) != "1"
        or _compact(rows[-1][0]) != str(population_endpoint)
        or not _same_page_owner_is_consistent(
            page,
            owner,
            role="annotations_and_inquiries",
        )
    ):
        return None
    exact_evidence, boxes, statuses = lattice
    covered = _derived_span_coverage(table, exact_evidence, boxes, statuses)
    if covered is None:
        return None
    selected_cells = [(0, column) for column in (0, 1, 3)]
    omission_rows: list[int] = []
    for row_index in range(1, len(rows)):
        if not all(_compact(rows[row_index][column]) for column in (1, 2, 3)):
            return None
        for column in range(4):
            if statuses.get((row_index, column)) != "exact" or (row_index, column) not in boxes:
                return None
            if _compact(rows[row_index][column]):
                if (row_index, column) not in exact_evidence:
                    return None
                selected_cells.append((row_index, column))
            elif column == 0 and (row_index, column) not in exact_evidence:
                omission_rows.append(row_index)
            else:
                return None
    if (
        statuses.get((0, 0)) != "exact"
        or statuses.get((0, 1)) != "exact"
        or statuses.get((0, 2)) != "derived"
        or statuses.get((0, 3)) != "exact"
        or (0, 2) in boxes
        or (0, 2) not in covered
        or owner.get("physical_field_omission_rows") != omission_rows
        or _owner_anomalies(
            owner,
            rows,
            body_start=1,
            population_start=1,
            sequence_column=roles["sequence"],
        )
        is None
        or not _owner_evidence_is_complete(owner, exact_evidence, selected_cells)
    ):
        return None
    heading_title = _compact(owner.get("heading_title"))
    inquiry_type = (
        "personal"
        if heading_title == "本人查询记录明细"
        else "institution"
        if heading_title in {
            "机构查询记录明细",
            "查询记录机构查询记录明细",
        }
        else ""
    )
    if not inquiry_type:
        return None
    return {
        "template_id": "annotations_and_inquiries",
        "binding": _INQUIRY_SEED_BINDING,
        "authority_mode": INQUIRY_AUTHORITY_CLOSED_PHYSICAL_ORDINAL,
        "body_start": 1,
        "population_start": population_start,
        "population_endpoint": population_endpoint,
        "inquiry_role_columns": roles,
        "inquiry_type": inquiry_type,
        "sequence_field_anomalies": list(owner["sequence_field_anomalies"]),
        "evidence_ids": list(owner["evidence_ids"]),
    }


def _ordinary_exact_inquiry_seed_authority(
    page: Any,
    table: Any,
) -> dict[str, Any] | None:
    """Validate an exact ordinary seed without granting ordinal closure."""

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    page_role = str(getattr(page, "canonical_template_id", "") or "")
    table_role = str(metadata.get("canonical_template_id") or "")
    canonical_page_role = str(metadata.get("canonical_page_template_id") or "")
    owner = metadata.get("canonical_section_owner")
    if page_role == MIXED_PBOC_PAGE_TEMPLATE_ID:
        if (
            table_role != "annotations_and_inquiries"
            or canonical_page_role != MIXED_PBOC_PAGE_TEMPLATE_ID
            or not isinstance(owner, Mapping)
            or owner.get("binding") != _SAME_PAGE_SECTION_BINDING
            or owner.get("header_binding") != "exact_single_row_header_lattice"
            or owner.get("header_row") != 0
            or owner.get("header_rows") != [0]
            or any(
                key in owner
                for key in ("authority_mode", "inquiry_authority_mode")
            )
            or not _schema_owner_is_consistent(table, owner)
            or not _same_page_owner_is_consistent(
                page,
                owner,
                role="annotations_and_inquiries",
            )
            or _owner_role_columns(owner) is None
        ):
            return None
    elif (
        page_role != "annotations_and_inquiries"
        or table_role != "annotations_and_inquiries"
        or canonical_page_role not in {"", "annotations_and_inquiries"}
        or owner is not None
    ):
        return None

    rows = _raw_rows(table)
    lattice = _exact_owner_cell_lattice(table)
    if (
        lattice is None
        or len(rows) < 2
        or any(len(row) != 4 for row in rows)
        or tuple(_compact(value) for value in rows[0])
        != ("编号", "查询日期", "查询机构", "查询原因")
    ):
        return None
    exact_evidence, boxes, statuses = lattice
    selected_cells: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        if row_index > 0 and not any(_compact(value) for value in row):
            return None
        for column in range(4):
            key = (row_index, column)
            derived_sequence_omission = bool(
                row_index > 0
                and column == _INQUIRY_ROLE_COLUMNS["sequence"]
                and not _compact(row[column])
                and statuses.get(key) == "derived"
                and key not in boxes
                and key not in exact_evidence
            )
            if derived_sequence_omission:
                continue
            if statuses.get(key) != "exact" or key not in boxes:
                return None
            if _compact(row[column]):
                if key not in exact_evidence:
                    return None
                selected_cells.append(key)
            elif key in exact_evidence:
                return None
    first_sequence = _compact(rows[1][0])
    last_sequence = _compact(rows[-1][0])
    endpoint = len(rows) - 1
    if first_sequence != "1" or last_sequence != str(endpoint):
        return None
    evidence_ids = [
        evidence_id
        for cell in selected_cells
        for evidence_id in exact_evidence[cell]
    ]
    if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
        return None
    return {
        "template_id": "annotations_and_inquiries",
        "binding": "ordinary_exact_inquiry_seed",
        "authority_mode": INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY,
        "body_start": 1,
        "population_start": 1,
        "population_endpoint": endpoint,
        "inquiry_role_columns": dict(_INQUIRY_ROLE_COLUMNS),
        "evidence_ids": evidence_ids,
    }


def _page_at_prior_rank(context: Any, page: Any) -> Any | None:
    logical = _positive_int(getattr(page, "page_number", None))
    rank = _ordered_page_rank(context, logical) if logical is not None else None
    if rank is None:
        return None
    candidates = [
        candidate
        for candidate in getattr(context, "pages", None) or ()
        if (candidate_logical := _positive_int(getattr(candidate, "page_number", None)))
        is not None
        and _ordered_page_rank(context, candidate_logical) == rank - 1
    ]
    return candidates[0] if len(candidates) == 1 else None


def _logical_int_mapping(value: Any) -> dict[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[int, int] = {}
    for raw_logical, raw_printed in value.items():
        if isinstance(raw_logical, bool):
            return None
        try:
            logical = int(raw_logical)
        except (TypeError, ValueError):
            return None
        printed = _positive_int(raw_printed)
        if logical <= 0 or printed is None or logical in result:
            return None
        result[logical] = printed
    return result


def _bottom_footer_geometry(value: Any, *, page_height: Any) -> bool:
    box = _bbox(value)
    try:
        height = float(page_height)
    except (TypeError, ValueError):
        return False
    if box is None or not math.isfinite(height) or height <= 0.0:
        return False
    tolerance = max(2.0, height * 0.01)
    return bool(
        box[1] >= height * 0.85
        and box[3] >= height * 0.90
        and box[3] <= height + tolerance
        and box[3] - box[1] <= height * 0.08
    )


def _exact_page_printed_identity(page: Any) -> tuple[int, int] | None:
    identities: list[tuple[int, int]] = []
    page_height = getattr(page, "height", 0)
    for text_item in getattr(page, "texts", None) or ():
        text, _source_box, ids = _text_values(text_item)
        box = (
            text_item.get("bbox")
            if isinstance(text_item, Mapping)
            else getattr(text_item, "bbox", None)
        )
        match = _PRINTED_PAGE_CAPTURE_RE.fullmatch(re.sub(r"\s+", "", text))
        if (
            match is None
            or not _bottom_footer_geometry(box, page_height=page_height)
            or ids is None
        ):
            continue
        page_number = _positive_int(int(match.group("page")))
        total = _positive_int(int(match.group("total")))
        if page_number is not None and total is not None and page_number <= total:
            identities.append((page_number, total))
    return identities[0] if len(identities) == 1 else None


def _proof_page_identity_is_consistent(
    context: Any,
    page: Any,
    proof: Any,
) -> tuple[int, int] | None:
    if not isinstance(proof, Mapping):
        return None
    logical = _positive_int(proof.get("logical_page"))
    source = _positive_int(proof.get("source_page"))
    printed = _positive_int(proof.get("printed_page"))
    total = _positive_int(proof.get("printed_total"))
    geometry = proof.get("geometry")
    if not isinstance(geometry, Mapping):
        return None
    if (
        logical is None
        or source is None
        or printed is None
        or total is None
        or printed > total
        or logical != _positive_int(getattr(page, "page_number", None))
        or source != _positive_int(getattr(page, "source_page_number", None))
    ):
        return None
    try:
        width = float(geometry.get("width"))
        height = float(geometry.get("height"))
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0.0
        or height <= 0.0
    ):
        return None
    geometry_kind = str(geometry.get("kind") or "")
    if geometry_kind == "page":
        if not (
            math.isclose(
                width,
                float(getattr(page, "width", 0) or 0),
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
            and math.isclose(
                height,
                float(getattr(page, "height", 0) or 0),
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
        ):
            return None
    elif geometry_kind == "topology":
        topology = getattr(context, "page_topology", None)
        try:
            current = topology.geometry(logical) if topology is not None else None
        except (AttributeError, TypeError, ValueError):
            return None
        crop = getattr(current, "source_crop_bbox", None) if current is not None else None
        proof_crop = geometry.get("source_crop_bbox")
        if (
            current is None
            or _positive_int(getattr(current, "source_page", None)) != source
            or not math.isclose(
                float(getattr(current, "width", 0) or 0),
                width,
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(getattr(current, "height", 0) or 0),
                height,
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
            or str(getattr(current, "split_kind", "") or "")
            != str(geometry.get("split_kind") or "")
            or getattr(current, "segment_index", None)
            != geometry.get("segment_index")
            or int(getattr(current, "selected_rotation", 0) or 0)
            != geometry.get("selected_rotation")
            or (getattr(current, "transform_usable", None) is True)
            is not (geometry.get("transform_usable") is True)
            or (
                crop is None
                and proof_crop is not None
            )
            or (
                crop is not None
                and not _same_box(crop, proof_crop)
            )
        ):
            return None
    else:
        return None
    return printed, total


def _context_evidence_page(context: Any, page: Any) -> Mapping[str, Any] | None:
    logical = _positive_int(getattr(page, "page_number", None))
    loader = getattr(context, "corrected_evidence_pages", None)
    if logical is not None and callable(loader):
        try:
            candidates = [
                candidate
                for candidate in loader()
                if isinstance(candidate, Mapping)
                and _positive_int(candidate.get("page")) == logical
            ]
        except (AttributeError, TypeError, ValueError):
            return None
        if len(candidates) == 1:
            return candidates[0]
    if logical is None:
        return None
    return {
        "page": logical,
        "source_page": _positive_int(getattr(page, "source_page_number", None)),
        "page_width": float(getattr(page, "width", 0) or 0),
        "page_height": float(getattr(page, "height", 0) or 0),
        "lines": [
            {
                "text": _text_values(text_item)[0],
                "bbox": list(_bbox(_text_values(text_item)[1]) or ()),
                "evidence_ids": list(_text_values(text_item)[2] or ()),
            }
            for text_item in getattr(page, "texts", None) or ()
        ],
    }


def _inquiry_continuation_adjacency_is_consistent(
    context: Any,
    prior_page: Any,
    prior_table: Any,
    prior_owner: Mapping[str, Any],
    page: Any,
    table: Any,
    owner: Mapping[str, Any],
    *,
    prior_authority_mode: str,
) -> bool:
    proof = owner.get("adjacency_proof")
    if not isinstance(proof, Mapping):
        return False
    kind = str(proof.get("kind") or "")
    identity_kind = str(proof.get("identity_kind") or "")
    previous_printed = _proof_page_identity_is_consistent(
        context,
        prior_page,
        proof.get("previous"),
    )
    current_printed = _proof_page_identity_is_consistent(
        context,
        page,
        proof.get("current"),
    )
    if (
        previous_printed is None
        or current_printed is None
        or previous_printed[1] != current_printed[1]
        or current_printed[0] != previous_printed[0] + 1
    ):
        return False
    resolution = getattr(context, "reading_order_resolution", None)
    tables_continue = getattr(context, "tables_continue", None)
    if not callable(tables_continue):
        return False
    try:
        continuation_decision = tables_continue(
            str(getattr(prior_table, "table_id", "") or ""),
            str(getattr(table, "table_id", "") or ""),
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if continuation_decision is not True and continuation_decision is not False:
        return False
    if (
        kind == "exact_printed_footer_table_edge"
        and continuation_decision is not True
    ) or (
        # The entity graph may truthfully keep consecutive physical tables
        # separate.  That false edge can carry only the already validated
        # ordinary schema; it cannot be upgraded to physical ordinal closure.
        kind == _INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF
        and (
            continuation_decision is not False
            or prior_authority_mode != INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY
        )
    ) or (
        kind == "local_paired_topology_entity_edge"
        and identity_kind == "exact_footer_pair"
        and continuation_decision is not False
    ):
        return False
    if kind in {
        "exact_printed_footer_table_edge",
        _INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF,
    } and (
        resolution is not None
        or kind == _INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF
    ):
        if not isinstance(resolution, Mapping):
            return False
        printed_total = _positive_int(resolution.get("printed_total"))
        printed_by_logical = _logical_int_mapping(
            resolution.get("printed_page_by_logical")
        )
        previous_logical = _positive_int(getattr(prior_page, "page_number", None))
        current_logical = _positive_int(getattr(page, "page_number", None))
        if (
            resolution.get("resolved") is not True
            or resolution.get("authoritative") is not True
            or resolution.get("identity_fallback") is True
            or printed_total != previous_printed[1]
            or printed_by_logical is None
            or previous_logical is None
            or current_logical is None
            or printed_by_logical.get(previous_logical) != previous_printed[0]
            or printed_by_logical.get(current_logical) != current_printed[0]
        ):
            return False
    if kind in {
        "exact_printed_footer_table_edge",
        _INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF,
    }:
        return bool(
            identity_kind == "exact_footer_pair"
            and _exact_page_printed_identity(prior_page) == previous_printed
            and _exact_page_printed_identity(page) == current_printed
        )
    if kind != "local_paired_topology_entity_edge" or identity_kind not in {
        "exact_footer_pair",
        "paired_inferred_current_footer",
    }:
        return False
    if (
        _exact_page_printed_identity(prior_page) != previous_printed
        or (
            identity_kind == "exact_footer_pair"
            and _exact_page_printed_identity(page) != current_printed
        )
        or (
            identity_kind == "paired_inferred_current_footer"
            and _exact_page_printed_identity(page) is not None
        )
    ):
        return False
    previous_evidence = _context_evidence_page(context, prior_page)
    current_evidence = _context_evidence_page(context, page)
    topology = getattr(context, "page_topology", None)
    audit_loader = getattr(context, "page_topology_audit", None)
    entity_context = getattr(context, "entity_context", None)
    if previous_evidence is None or current_evidence is None:
        return False
    from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
        _local_exact_spread_printed_adjacency,
        _local_paired_inquiry_entity_continuation_proved,
        _local_paired_printed_adjacency,
    )

    topology_proved = (
        _local_exact_spread_printed_adjacency(
            prior_page,
            previous_evidence,
            page,
            current_evidence,
            previous_printed=previous_printed,
            current_printed=current_printed,
            reading_order_resolution=resolution,
            topology=topology,
            frozen_topology_audit_loader=audit_loader,
        )
        if identity_kind == "exact_footer_pair"
        else _local_paired_printed_adjacency(
            prior_page,
            previous_evidence,
            page,
            current_evidence,
            previous_printed=previous_printed,
            current_printed=None,
            reading_order_resolution=resolution,
            topology=topology,
            frozen_topology_audit_loader=audit_loader,
        )
    )
    return bool(
        topology_proved
        and _local_paired_inquiry_entity_continuation_proved(
            entity_context,
            prior_page,
            prior_table,
            prior_owner,
            page,
            table,
            {"header_binding": owner.get("header_binding")},
        )
    )


def _validated_inquiry_population_owner(
    context: Any,
    page: Any,
    table: Any,
    owner: Mapping[str, Any],
    *,
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    table_id = str(getattr(table, "table_id", "") or "")
    if not table_id or table_id in seen:
        return None
    binding = str(owner.get("binding") or "")
    if binding == _INQUIRY_SEED_BINDING:
        return _bounded_inquiry_seed_owner_is_consistent(page, table, owner)
    if binding != _INQUIRY_CONTINUATION_BINDING:
        return None
    rows = _raw_rows(table)
    roles = _owner_role_columns(owner)
    lattice = _exact_owner_cell_lattice(table)
    population_start = _positive_int(owner.get("population_start"))
    population_endpoint = _positive_int(owner.get("population_endpoint"))
    if (
        context is None
        or roles is None
        or lattice is None
        or not rows
        or any(len(row) != 4 for row in rows)
        or population_start is None
        or population_endpoint != population_start + len(rows) - 1
        or owner.get("header_row") is not None
        or owner.get("population_endpoint_row") != len(rows) - 1
        or owner.get("header_binding") != "inherited_exact_four_role_lattice"
        or owner.get("population_witness_row") not in range(len(rows))
        or any(
            key in owner
            for key in ("authority_mode", "inquiry_authority_mode")
        )
    ):
        return None
    first = _compact(rows[0][roles["sequence"]])
    last = _compact(rows[-1][roles["sequence"]])
    first_sequence = int(first) if re.fullmatch(r"[1-9]\d{0,3}", first) else None
    last_sequence = int(last) if re.fullmatch(r"[1-9]\d{0,3}", last) else None
    if (
        first_sequence is not None and first_sequence != population_start
    ) or (
        last_sequence is not None and last_sequence != population_endpoint
    ) or (first_sequence is None and last_sequence is None):
        return None
    exact_evidence, boxes, statuses = lattice
    covered = _derived_span_coverage(table, exact_evidence, boxes, statuses)
    if covered is None:
        return None
    selected_cells: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        if not any(_compact(value) for value in row):
            return None
        for column in range(4):
            key = (row_index, column)
            if key in covered:
                if _compact(row[column]):
                    return None
                continue
            if statuses.get(key) != "exact" or key not in boxes:
                return None
            if _compact(row[column]):
                if key not in exact_evidence:
                    return None
                selected_cells.append(key)
            elif key in exact_evidence:
                return None
    if (
        _owner_anomalies(
            owner,
            rows,
            body_start=0,
            population_start=population_start,
            sequence_column=roles["sequence"],
        )
        is None
        or not _owner_evidence_is_complete(owner, exact_evidence, selected_cells)
    ):
        return None
    prior_page = _page_at_prior_rank(context, page)
    prior_table_id = str(owner.get("prior_table_id") or "")
    prior_tables = [
        candidate
        for candidate in getattr(prior_page, "tables", None) or ()
        if str(getattr(candidate, "table_id", "") or "") == prior_table_id
    ] if prior_page is not None else []
    if len(prior_tables) != 1:
        return None
    prior_table = prior_tables[0]
    prior_metadata = getattr(prior_table, "metadata", None) or {}
    prior_owner = (
        prior_metadata.get("canonical_section_owner")
        if isinstance(prior_metadata, Mapping)
        else None
    )
    if not isinstance(prior_owner, Mapping):
        return None
    prior_validated = _validated_inquiry_population_owner(
        context,
        prior_page,
        prior_table,
        prior_owner,
        seen=seen | {table_id},
    )
    if prior_validated is None:
        prior_validated = _ordinary_exact_inquiry_seed_authority(
            prior_page,
            prior_table,
        )
    prior_ids = (
        _unique_ids(prior_validated.get("evidence_ids"))
        if prior_validated is not None
        else None
    )
    current_ids = _unique_ids(owner.get("evidence_ids"))
    if (
        prior_validated is None
        or prior_validated["population_endpoint"] + 1 != population_start
        or prior_ids is None
        or current_ids is None
        or set(prior_ids).intersection(current_ids)
        or not _inquiry_continuation_adjacency_is_consistent(
            context,
            prior_page,
            prior_table,
            prior_owner,
            page,
            table,
            owner,
            prior_authority_mode=str(prior_validated["authority_mode"]),
        )
    ):
        return None
    return {
        "template_id": "annotations_and_inquiries",
        "binding": _INQUIRY_CONTINUATION_BINDING,
        "authority_mode": prior_validated["authority_mode"],
        "body_start": 0,
        "population_start": population_start,
        "population_endpoint": population_endpoint,
        "inquiry_role_columns": roles,
        "sequence_field_anomalies": list(owner["sequence_field_anomalies"]),
        "prior_table_id": prior_table_id,
        "evidence_ids": list(owner["evidence_ids"]),
        **(
            {"inquiry_type": prior_validated["inquiry_type"]}
            if prior_validated.get("inquiry_type") in {"institution", "personal"}
            else {}
        ),
    }


def canonical_inquiry_population_metadata(
    context: Any,
    page: Any,
    table: Any,
) -> dict[str, Any] | None:
    """Return validated population metadata for a special mixed inquiry owner."""

    metadata = getattr(table, "metadata", None) or {}
    owner = metadata.get("canonical_section_owner") if isinstance(metadata, Mapping) else None
    if (
        str(getattr(page, "canonical_template_id", "") or "")
        != MIXED_PBOC_PAGE_TEMPLATE_ID
        or str(metadata.get("canonical_template_id") or "")
        != "annotations_and_inquiries"
        or str(metadata.get("canonical_page_template_id") or "")
        != MIXED_PBOC_PAGE_TEMPLATE_ID
        or not isinstance(owner, Mapping)
        or str(owner.get("template_id") or "")
        != "annotations_and_inquiries"
        or str(owner.get("table_id") or "")
        != str(getattr(table, "table_id", "") or "")
    ):
        return None
    return _validated_inquiry_population_owner(context, page, table, owner)


def canonical_table_role(context: Any, page: Any, table: Any) -> str | None:
    """Return the uniquely verified canonical owner role for ``table``.

    Ordinary pages require explicit page/table agreement.  Mixed pages never
    inherit a role from the page; they require the table-local ownership graph
    emitted by canonical layout projection.
    """

    page_role = str(getattr(page, "canonical_template_id", "") or "")
    metadata = getattr(table, "metadata", None) or {}
    table_id = str(getattr(table, "table_id", "") or "")
    if not isinstance(metadata, Mapping) or not table_id:
        return None
    table_role = str(metadata.get("canonical_template_id") or "")
    canonical_page_role = str(metadata.get("canonical_page_template_id") or "")
    owner = metadata.get("canonical_section_owner")

    if page_role != MIXED_PBOC_PAGE_TEMPLATE_ID:
        if (
            page_role not in REGISTERED_PBOC_TABLE_ROLES
            or table_role != page_role
            or canonical_page_role not in {"", page_role}
            or owner is not None
        ):
            return None
        return table_role

    if (
        table_role not in REGISTERED_PBOC_TABLE_ROLES
        or canonical_page_role != MIXED_PBOC_PAGE_TEMPLATE_ID
        or not isinstance(owner, Mapping)
        or str(owner.get("table_id") or "") != table_id
        or str(owner.get("template_id") or "") != table_role
    ):
        return None
    binding = str(owner.get("binding") or "")
    if binding in {_INQUIRY_SEED_BINDING, _INQUIRY_CONTINUATION_BINDING}:
        return (
            table_role
            if table_role == "annotations_and_inquiries"
            and _validated_inquiry_population_owner(
                context,
                page,
                table,
                owner,
            )
            is not None
            else None
        )
    if not _schema_owner_is_consistent(table, owner):
        return None
    if binding == _SAME_PAGE_SECTION_BINDING:
        return table_role if _same_page_owner_is_consistent(page, owner, role=table_role) else None
    if binding == _CROSS_PAGE_AGREEMENT_BINDING:
        return (
            table_role
            if _cross_page_owner_is_consistent(context, page, table, owner)
            else None
        )
    return None


def canonical_table_owned_by(
    context: Any,
    page: Any,
    table: Any,
    role: str,
) -> bool:
    return canonical_table_role(context, page, table) == role


def canonical_cross_page_agreement_anchor(
    context: Any,
    page: Any,
    table: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Return the exact prior-page card anchor after full owner validation."""

    if canonical_table_role(context, page, table) != "credit_agreement":
        return None
    metadata = getattr(table, "metadata", None) or {}
    owner = metadata.get("canonical_section_owner") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(owner, Mapping)
        or owner.get("binding") != _CROSS_PAGE_AGREEMENT_BINDING
    ):
        return None
    sequence = _positive_int(owner.get("printed_sequence"))
    logical_page = _positive_int(owner.get("heading_source_logical_page"))
    heading_ids = _unique_ids(owner.get("heading_evidence_ids"))
    heading_box = _bbox(owner.get("heading_bbox"))
    if sequence is None or logical_page is None or heading_ids is None or heading_box is None:
        return None
    prior_pages = [
        candidate
        for candidate in getattr(context, "pages", None) or ()
        if _positive_int(getattr(candidate, "page_number", None)) == logical_page
    ]
    if len(prior_pages) != 1:
        return None
    return str(sequence), {
        "source": "candidate_b_canonical_section_owner",
        "logical_page": logical_page,
        "source_page": int(
            getattr(prior_pages[0], "source_page_number", 0) or logical_page
        ),
        "bbox": list(heading_box),
        "geometry_scope": "text",
        "field_name": "sequence",
        "binding": "canonical_card_anchor",
        "binding_quality": "exact_cross_page_section_owner",
        "evidence_ids": list(heading_ids),
    }


__all__ = [
    "INQUIRY_AUTHORITY_CLOSED_PHYSICAL_ORDINAL",
    "INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY",
    "MIXED_PBOC_PAGE_TEMPLATE_ID",
    "REGISTERED_PBOC_TABLE_ROLES",
    "canonical_cross_page_agreement_anchor",
    "canonical_inquiry_population_metadata",
    "canonical_table_owned_by",
    "canonical_table_role",
]
