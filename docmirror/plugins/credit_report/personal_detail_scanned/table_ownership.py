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

_SAME_PAGE_SECTION_BINDING = "exact_pboc_section_heading_and_table_schema"
_CROSS_PAGE_AGREEMENT_BINDING = (
    "terminal_prior_page_agreement_anchor_and_leading_exact_table"
)
_PRINTED_PAGE_RE = re.compile(r"^第\s*\d+\s*页\s*[，,]\s*共\s*\d+\s*页$")


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
        or not _schema_owner_is_consistent(table, owner)
    ):
        return None
    binding = str(owner.get("binding") or "")
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
    "MIXED_PBOC_PAGE_TEMPLATE_ID",
    "REGISTERED_PBOC_TABLE_ROLES",
    "canonical_cross_page_agreement_anchor",
    "canonical_table_owned_by",
    "canonical_table_role",
]
