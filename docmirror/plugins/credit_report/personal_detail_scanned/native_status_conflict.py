# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed monthly status checks against sealed physical table cells.

This is deliberately a conflict guard, not another recognizer.  A final
Candidate-B status is compared only with the single sealed source-table cell
at the same physical location after both sides independently prove exact
year-plus-twelve-month ownership.  Disagreement withholds the status; this
module never promotes the native token or mutates ``ParseResult``.
"""

from __future__ import annotations

import re
from copy import deepcopy
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any
from types import SimpleNamespace

_CANONICAL_PBOC_STATUSES = frozenset(
    {
        "*",
        "/",
        "#",
        "N",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "A",
        "B",
        "C",
        "D",
        "G",
        "M",
        "Z",
    }
)
_STATUS_FIELDS = ("status", "status_code")
_AMOUNT_FIELDS = ("overdue_amount", "status_amount")
_OWNED_MONTH_GEOMETRIES = frozenset(
    {
        "year_plus_twelve_rule_ownership",
        "source_table_year_plus_twelve_ownership",
    }
)
_TOP_LEFT_PDF_COORDINATES = "pdf_points_top_left"
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_DECIMAL_RE = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")

# This is an OCR-reading confidence threshold, not a table/geometry score.
# It matches the existing independent-page candidate acceptance threshold:
# an observation too weak to repair a field must not silently become a final
# status merely because its glyph happens to be in the legal vocabulary.
MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM = 0.72
_MONTHLY_SLOT_SOURCE = "sealed_native_monthly_field_slot"
_MONTHLY_SLOT_SCHEMA = "sealed_monthly_field_slot_v1"


def _owned_geometry_provenance(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    basis = str(value.get("selection_basis") or "")
    required_source = {
        "year_plus_twelve_rule_ownership": "vertical_rule_projection",
        "source_table_year_plus_twelve_ownership": "source_table_geometry",
    }.get(basis)
    return basis if required_source and str(value.get("source") or "") == required_source else None


def _get(owner: Any, key: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
        return owner.get(key, default)
    return getattr(owner, key, default)


def _values(record: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = record.get("normalized")
    return normalized if isinstance(normalized, Mapping) else record


def _mutable_values(record: dict[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized")
    return normalized if isinstance(normalized, dict) else record


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (all(isfinite(item) for item in box) and box[2] > box[0] and box[3] > box[1]):
        return None
    return box


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _matrix_get(matrix: Any, row: int, col: int, default: Any = None) -> Any:
    if not isinstance(matrix, (list, tuple)) or not (0 <= row < len(matrix)):
        return default
    values = matrix[row]
    if not isinstance(values, (list, tuple)) or not (0 <= col < len(values)):
        return default
    return values[col]


def _canonical_status(value: Any) -> str | None:
    token = str(value or "").strip().upper()
    return token if token in _CANONICAL_PBOC_STATUSES else None


def _single_status(record: Mapping[str, Any]) -> tuple[str, str] | None:
    values = _values(record)
    observations = [
        (field_name, token)
        for owner in (values, record)
        for field_name in _STATUS_FIELDS
        if (token := _canonical_status(owner.get(field_name))) is not None
    ]
    distinct = {token for _field_name, token in observations}
    if len(distinct) != 1:
        return None
    token = next(iter(distinct))
    field_name = next(field for field, observed in observations if observed == token)
    return field_name, token


def _performance_year_month(record: Mapping[str, Any]) -> tuple[int, int] | None:
    values = _values(record)
    performance_month = str(values.get("performance_month") or "").strip()
    match = re.fullmatch(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])", performance_month)
    if match is not None:
        return int(match.group(1)), int(match.group(2))
    year = _integer(values.get("year"))
    month = _integer(values.get("month"))
    if year is None or month is None or not (1900 <= year <= 2099 and 1 <= month <= 12):
        return None
    return year, month


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _DECIMAL_RE.fullmatch(text):
        return None
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def _single_amount(record: Mapping[str, Any]) -> Decimal | None:
    values = _values(record)
    observed = [
        amount
        for owner in (values, record)
        for field_name in _AMOUNT_FIELDS
        if (amount := _decimal(owner.get(field_name))) is not None
    ]
    return observed[0] if observed and len(set(observed)) == 1 else None


def _record_refs(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    refs: list[Mapping[str, Any]] = []
    seen_owners: set[int] = set()
    for owner in (_values(record), record):
        if id(owner) in seen_owners:
            continue
        seen_owners.add(id(owner))
        by_field = owner.get("source_refs_by_field")
        if isinstance(by_field, Mapping):
            for field_name in (*_STATUS_FIELDS, *_AMOUNT_FIELDS):
                values = by_field.get(field_name)
                if isinstance(values, (list, tuple)):
                    refs.extend(ref for ref in values if isinstance(ref, Mapping))
        for pool_name in ("source_cell_refs", "source_refs"):
            values = owner.get(pool_name)
            if isinstance(values, (list, tuple)):
                refs.extend(ref for ref in values if isinstance(ref, Mapping))
    return refs


def _exact_final_status_ref(
    record: Mapping[str, Any],
    *,
    month: int,
) -> dict[str, Any] | None:
    grid_id = str(_values(record).get("grid_id") or record.get("grid_id") or "").strip()
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for ref in _record_refs(record):
        ref_field = str(ref.get("field_name") or "").strip()
        geometry = ref.get("geometry_provenance")
        box = _bbox(ref.get("bbox"))
        logical_page = _integer(ref.get("logical_page") or ref.get("page"))
        row = _integer(ref.get("row"))
        column = _integer(ref.get("col"))
        if not (
            ref_field in _STATUS_FIELDS
            and str(ref.get("geometry_scope") or "") == "cell"
            and str(ref.get("coordinate_system") or "") == _TOP_LEFT_PDF_COORDINATES
            and str(ref.get("geometry_status") or "") in {"", "exact"}
            and _owned_geometry_provenance(geometry) in _OWNED_MONTH_GEOMETRIES
            and box is not None
            and logical_page is not None
            and logical_page > 0
            and row is not None
            and row >= 0
            and column == month
            and (not grid_id or str(ref.get("grid_id") or "").strip() == grid_id)
        ):
            continue
        marker = (logical_page, row, column, *(round(value, 5) for value in box))
        candidates.setdefault(marker, dict(ref))
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _exact_final_amount_ref(
    record: Mapping[str, Any],
    *,
    month: int,
    status_ref: Mapping[str, Any],
) -> dict[str, Any] | None:
    grid_id = str(_values(record).get("grid_id") or record.get("grid_id") or "").strip()
    status_box = _bbox(status_ref.get("bbox"))
    status_page = _integer(status_ref.get("logical_page") or status_ref.get("page"))
    status_row = _integer(status_ref.get("row"))
    status_geometry = status_ref.get("geometry_provenance")
    if not (
        status_box is not None
        and status_page is not None
        and status_row is not None
        and isinstance(status_geometry, Mapping)
    ):
        return None
    status_basis = _owned_geometry_provenance(status_geometry)
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for ref in _record_refs(record):
        ref_field = str(ref.get("field_name") or "").strip()
        geometry = ref.get("geometry_provenance")
        box = _bbox(ref.get("bbox"))
        logical_page = _integer(ref.get("logical_page") or ref.get("page"))
        row = _integer(ref.get("row"))
        column = _integer(ref.get("col"))
        if not (
            ref_field in _AMOUNT_FIELDS
            and str(ref.get("geometry_scope") or "") == "cell"
            and str(ref.get("coordinate_system") or "") == _TOP_LEFT_PDF_COORDINATES
            and str(ref.get("geometry_status") or "") in {"", "exact"}
            and _owned_geometry_provenance(geometry) == status_basis
            and status_basis in _OWNED_MONTH_GEOMETRIES
            and box is not None
            and logical_page == status_page
            and row == status_row + 1
            and column == month
            and (not grid_id or str(ref.get("grid_id") or "").strip() == grid_id)
            and abs(box[0] - status_box[0]) <= 1.5
            and abs(box[2] - status_box[2]) <= 1.5
            and 0.0 <= box[1] - status_box[3] <= 5.0
            and box[3] > status_box[3]
        ):
            continue
        marker = (logical_page, row, column, *(round(value, 5) for value in box))
        candidates.setdefault(marker, dict(ref))
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _rows(table: Any) -> list[Any]:
    rows = _get(table, "rows")
    if not isinstance(rows, (list, tuple)):
        rows = _get(table, "row_models")
    return list(rows) if isinstance(rows, (list, tuple)) else []


def _owned_cell_index(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _typed_cells_by_raw_geometry_row(
    table: Any,
    *,
    coordinate_page: int,
) -> dict[int, dict[int, Any]] | None:
    """Map raw lattice rows to typed cells only through exact source ownership.

    ``TableBlock.rows`` contains typed rows.  When ``preserve_headers`` is true,
    its first typed row owns raw lattice row 1 rather than row 0, so a raw
    geometry index must never be used as a typed-container index.  The cell's
    single ``source_cell_refs`` entry is the only admissible bridge.  ``page``
    names the canonical coordinate page; the physical source-page origin is a
    separate property of the page and does not participate in this row map.
    """

    if (
        not isinstance(coordinate_page, int)
        or isinstance(coordinate_page, bool)
        or coordinate_page <= 0
    ):
        return None
    table_id = str(_get(table, "table_id") or _get(table, "id") or "").strip()
    rows = _rows(table)
    if not table_id or not rows:
        return None

    # Empty typed rows make no ownership claim.  Invalid or competing owners
    # affect only the raw rows they claim; an unrelated empty/damaged row must
    # not disable an otherwise exact native witness elsewhere in the table.
    owners_by_raw_row: dict[int, list[dict[int, Any] | None]] = {}
    for row in rows:
        cells = _get(row, "cells")
        if not isinstance(cells, (list, tuple)) or not cells:
            continue
        row_raw_indices: set[int] = set()
        row_typed_indices: set[int] = set()
        owned_cells: dict[int, Any] = {}
        valid_owner = True
        for cell in cells:
            typed_row = _owned_cell_index(_get(cell, "row_index"))
            column = _owned_cell_index(_get(cell, "col_index"))
            refs = _get(cell, "source_cell_refs")
            if typed_row is not None:
                row_typed_indices.add(typed_row)
            if isinstance(refs, (list, tuple)):
                # Keep even malformed claims local to their declared raw row.
                # In particular, a duplicate ref must not disappear and leave
                # another owner looking unique.
                for candidate_ref in refs:
                    if isinstance(candidate_ref, Mapping):
                        claimed_row = _owned_cell_index(candidate_ref.get("raw_row"))
                        if claimed_row is not None:
                            row_raw_indices.add(claimed_row)
            if (
                typed_row is None
                or column is None
                or not isinstance(refs, (list, tuple))
                or len(refs) != 1
            ):
                valid_owner = False
                continue
            [ref] = refs
            if not isinstance(ref, Mapping):
                valid_owner = False
                continue
            ref_row = _owned_cell_index(ref.get("row"))
            raw_row = _owned_cell_index(ref.get("raw_row"))
            ref_column = _owned_cell_index(ref.get("col"))
            ref_page = _owned_cell_index(ref.get("page"))
            if not (
                str(ref.get("table_id") or "").strip() == table_id
                and ref_row == typed_row
                and raw_row is not None
                and ref_column == column
                and ref_page == coordinate_page
            ):
                valid_owner = False
                continue
            if column in owned_cells:
                valid_owner = False
            owned_cells[column] = cell
        if len(row_raw_indices) != 1 or len(row_typed_indices) != 1:
            valid_owner = False
        for raw_row in row_raw_indices:
            owners_by_raw_row.setdefault(raw_row, []).append(
                owned_cells if valid_owner else None
            )
    # Two typed containers claiming one raw row are ambiguous even when their
    # columns do not overlap.  Required rows absent from this index fail closed
    # at the candidate lookup; there is no typed-position fallback.
    cells_by_raw_row: dict[int, dict[int, Any]] = {}
    for raw_row, owners in owners_by_raw_row.items():
        if len(owners) == 1:
            row_owner = owners[0]
            if row_owner is not None:
                cells_by_raw_row[raw_row] = row_owner
    return cells_by_raw_row


def _exact_cell(cell: Any, *, require_single_token: bool = False) -> bool:
    if (
        str(_get(cell, "geometry_status") or "") != "exact"
        or _bbox(_get(cell, "bbox")) is None
        or _integer(_get(cell, "col_span", 1)) != 1
    ):
        return False
    if not require_single_token:
        return True
    token_ids = [str(item) for item in _get(cell, "token_ids", ()) or () if item]
    evidence_ids = [str(item) for item in _get(cell, "evidence_ids", ()) or () if item]
    return len(token_ids) == 1 and len(evidence_ids) == 1 and token_ids == evidence_ids


def _nonempty_equal_token_evidence(cell: Any) -> bool:
    token_ids = [str(item) for item in _get(cell, "token_ids", ()) or () if item]
    evidence_ids = [
        str(item) for item in _get(cell, "evidence_ids", ()) or () if item
    ]
    return bool(token_ids) and token_ids == evidence_ids


def _header_proves_month_ordinals(cells: Mapping[int, Any]) -> bool:
    if set(range(1, 13)).difference(cells):
        return False
    matched = 0
    for month in range(1, 13):
        text = str(_get(cells[month], "text") or "").strip()
        if re.fullmatch(r"0?\d{1,2}", text):
            value = int(text)
            if 1 <= value <= 12 and value != month:
                return False
            if value == month:
                matched += 1
    # The physical lattice is primary; this threshold tolerates up to three
    # damaged header glyphs while rejecting an unowned or shifted 12-cell row.
    return matched >= 9


def _year_cell_proves_row(cell: Any, expected_year: int) -> bool:
    if not _exact_cell(cell):
        return False
    if _integer(_get(cell, "row_span", 1)) != 2:
        return False
    years = _YEAR_RE.findall(str(_get(cell, "text") or ""))
    return len(years) == 1 and int(years[0]) == expected_year


def _year_anchor_proves_pair(
    cells_by_raw_row: Mapping[int, Mapping[int, Any]],
    *,
    status_row: int,
    amount_row: int,
    expected_year: int,
) -> tuple[int, Any] | None:
    """Bind one typed year anchor to a headerless continuation row pair."""

    matches: list[tuple[int, Any]] = []
    for raw_row in sorted(cells_by_raw_row):
        if raw_row > status_row:
            continue
        cells = cells_by_raw_row[raw_row]
        cell = cells.get(0)
        if cell is None:
            continue
        row_span = _integer(_get(cell, "row_span", 1)) or 1
        col_span = _integer(_get(cell, "col_span", 1)) or 1
        years = _YEAR_RE.findall(str(_get(cell, "text") or ""))
        if not (
            _exact_cell(cell, require_single_token=True)
            and _owned_cell_index(_get(cell, "col_index")) == 0
            and col_span == 1
            and row_span >= 2
            and raw_row + row_span - 1 >= amount_row
            and len(years) == 1
            and int(years[0]) == expected_year
        ):
            continue
        matches.append((raw_row, cell))
    return matches[0] if len(matches) == 1 else None


def _overlap_is_same_cell(
    native_box: tuple[float, float, float, float],
    final_box: tuple[float, float, float, float],
) -> bool:
    left = max(native_box[0], final_box[0])
    top = max(native_box[1], final_box[1])
    right = min(native_box[2], final_box[2])
    bottom = min(native_box[3], final_box[3])
    if right <= left or bottom <= top:
        return False
    intersection = (right - left) * (bottom - top)
    native_area = (native_box[2] - native_box[0]) * (native_box[3] - native_box[1])
    final_area = (final_box[2] - final_box[0]) * (final_box[3] - final_box[1])
    union = native_area + final_area - intersection
    x_overlap = right - left
    y_overlap = bottom - top
    return (
        intersection / native_area >= 0.80
        and intersection / final_area >= 0.80
        and intersection / union >= 0.70
        and x_overlap / min(native_box[2] - native_box[0], final_box[2] - final_box[0]) >= 0.92
        and y_overlap / min(native_box[3] - native_box[1], final_box[3] - final_box[1]) >= 0.80
    )


def _amount_overlap_is_same_cell(
    native_box: tuple[float, float, float, float],
    final_box: tuple[float, float, float, float],
) -> bool:
    """Bind a vertically trimmed final amount crop to one native table cell."""

    left = max(native_box[0], final_box[0])
    top = max(native_box[1], final_box[1])
    right = min(native_box[2], final_box[2])
    bottom = min(native_box[3], final_box[3])
    if right <= left or bottom <= top:
        return False
    intersection = (right - left) * (bottom - top)
    native_width = native_box[2] - native_box[0]
    native_height = native_box[3] - native_box[1]
    final_width = final_box[2] - final_box[0]
    final_height = final_box[3] - final_box[1]
    native_area = native_width * native_height
    final_area = final_width * final_height
    union = native_area + final_area - intersection
    height_ratio = final_height / native_height
    return (
        intersection / native_area >= 0.65
        and intersection / final_area >= 0.80
        and intersection / union >= 0.60
        and (right - left) / min(native_width, final_width) >= 0.92
        and 0.65 <= height_ratio <= 1.35
    )


def _page_number(page: Any, *names: str) -> int | None:
    for name in names:
        value = _integer(_get(page, name))
        if value is not None and value > 0:
            return value
    return None


def _candidate_pages(context: Any, ref: Mapping[str, Any]) -> list[Any]:
    parse_result = getattr(context, "parse_result", None)
    pages = getattr(parse_result, "pages", None)
    if not isinstance(pages, (list, tuple)):
        return []
    logical_page = _integer(ref.get("logical_page") or ref.get("page"))
    if logical_page is None:
        return []
    expected_source_page = _integer(ref.get("source_page"))
    source_page_map = getattr(context, "source_page_by_logical", None)
    mapped_source_page = _integer(source_page_map.get(logical_page)) if isinstance(source_page_map, Mapping) else None
    if (
        expected_source_page is not None
        and mapped_source_page is not None
        and expected_source_page != mapped_source_page
    ):
        return []
    expected_source_page = expected_source_page or mapped_source_page
    matched: list[Any] = []
    for page in pages:
        if _page_number(page, "page_number", "logical_page") != logical_page:
            continue
        source_page = _page_number(page, "source_page_number", "source_page")
        if expected_source_page is not None and source_page is not None and source_page != expected_source_page:
            continue
        matched.append(page)
    return matched


def _native_source_ref(
    *,
    page: Any,
    table: Any,
    row_index: int,
    column: int,
    cell: Any,
    logical_bbox: tuple[float, float, float, float],
    field_name: str,
) -> dict[str, Any]:
    source_bbox = _bbox(_get(cell, "bbox"))
    ref: dict[str, Any] = {
        "source": "sealed_native_physical_table_cell",
        "evidence_plane": "sealed_native_source_table",
        "logical_page": _page_number(page, "page_number", "logical_page"),
        "source_page": _page_number(page, "source_page_number", "source_page"),
        "table_id": str(_get(table, "table_id") or _get(table, "id") or ""),
        "row": row_index,
        "col": column,
        "field_name": field_name,
        "geometry_scope": "cell",
        "geometry_status": "exact",
        "coordinate_system": _TOP_LEFT_PDF_COORDINATES,
        "bbox": list(logical_bbox),
        "source_bbox": (list(source_bbox) if source_bbox is not None and source_bbox != logical_bbox else None),
        "evidence_ids": [str(item) for item in _get(cell, "evidence_ids", ()) or ()],
        "token_ids": [str(item) for item in _get(cell, "token_ids", ()) or ()],
    }
    return {key: value for key, value in ref.items() if value not in (None, "", [])}


def _source_geometry_row_binding(
    final_ref: Mapping[str, Any],
    final_amount_ref: Mapping[str, Any],
    *,
    logical_page: int,
) -> tuple[str, int, int] | None:
    """Read the value-free physical row binding carried by continuation refs."""

    status_geometry = final_ref.get("geometry_provenance")
    amount_geometry = final_amount_ref.get("geometry_provenance")
    if not isinstance(status_geometry, Mapping) or not isinstance(amount_geometry, Mapping):
        return None
    required_pairs = {
        "selection_basis": "source_table_year_plus_twelve_ownership",
        "source": "source_table_geometry",
        "coordinate_system": _TOP_LEFT_PDF_COORDINATES,
        "column_count": 13,
        "month_column_count": 12,
    }
    if any(
        status_geometry.get(key) != expected or amount_geometry.get(key) != expected
        for key, expected in required_pairs.items()
    ):
        return None
    if status_geometry.get("value_inputs_used") is not False or amount_geometry.get("value_inputs_used") is not False:
        return None
    table_id = str(status_geometry.get("table_id") or "").strip()
    amount_table_id = str(amount_geometry.get("table_id") or "").strip()
    status_row = _integer(status_geometry.get("status_row_index"))
    amount_row = _integer(status_geometry.get("amount_row_index"))
    if not (
        table_id
        and amount_table_id == table_id
        and status_row is not None
        and status_row >= 0
        and amount_row == status_row + 1
        and _integer(amount_geometry.get("status_row_index")) == status_row
        and _integer(amount_geometry.get("amount_row_index")) == amount_row
    ):
        return None
    for geometry in (status_geometry, amount_geometry):
        continuation_page = _integer(geometry.get("continuation_logical_page"))
        rule_count = _integer(geometry.get("vertical_rule_count") or geometry.get("rule_count"))
        if continuation_page != logical_page or rule_count != 14:
            return None
    return table_id, status_row, amount_row


def _base_source_geometry_row_binding(
    final_ref: Mapping[str, Any],
    final_amount_ref: Mapping[str, Any],
    *,
    logical_page: int,
) -> tuple[str, int, int, int] | None:
    """Read one exact, value-free base-page source-table row binding."""

    status_geometry = final_ref.get("geometry_provenance")
    amount_geometry = final_amount_ref.get("geometry_provenance")
    if not isinstance(status_geometry, Mapping) or not isinstance(amount_geometry, Mapping):
        return None
    required_pairs = {
        "selection_basis": "source_table_year_plus_twelve_ownership",
        "source": "source_table_geometry",
        "coordinate_system": _TOP_LEFT_PDF_COORDINATES,
        "column_count": 13,
        "month_column_count": 12,
    }
    if any(
        status_geometry.get(key) != expected or amount_geometry.get(key) != expected
        for key, expected in required_pairs.items()
    ):
        return None
    if status_geometry.get("value_inputs_used") is not False or amount_geometry.get("value_inputs_used") is not False:
        return None
    # This path is deliberately disjoint from headerless continuation binding.
    # Even a null continuation marker is not accepted as base-page provenance.
    if any("continuation_logical_page" in geometry for geometry in (status_geometry, amount_geometry)):
        return None
    table_id = str(status_geometry.get("table_id") or "").strip()
    amount_table_id = str(amount_geometry.get("table_id") or "").strip()
    status_row = _integer(status_geometry.get("status_row_index"))
    amount_row = _integer(status_geometry.get("amount_row_index"))
    if not (
        table_id
        and amount_table_id == table_id
        and status_row is not None
        and status_row >= 1
        and amount_row == status_row + 1
        and _integer(amount_geometry.get("status_row_index")) == status_row
        and _integer(amount_geometry.get("amount_row_index")) == amount_row
    ):
        return None
    for geometry in (status_geometry, amount_geometry):
        vertical_rule_count = _integer(geometry.get("vertical_rule_count"))
        rule_count = _integer(geometry.get("rule_count"))
        if (
            vertical_rule_count is not None
            and rule_count is not None
            and vertical_rule_count != rule_count
        ):
            return None
        effective_rule_count = (
            vertical_rule_count if vertical_rule_count is not None else rule_count
        )
        if not (
            _integer(geometry.get("logical_page")) == logical_page
            and effective_rule_count == 14
            and _integer(geometry.get("year_anchor_row_index")) == status_row
            and str(geometry.get("year_anchor_mode") or "") == "spanning_year_cell"
            and _integer(geometry.get("year_row_span")) == 2
            and geometry.get("active_cell_geometry_exact") is True
            and _integer(geometry.get("active_cell_rule_derived_count")) == 0
        ):
            return None
    return table_id, status_row - 1, status_row, amount_row


def _declares_source_table_geometry(ref: Mapping[str, Any]) -> bool:
    geometry = ref.get("geometry_provenance")
    return bool(
        isinstance(geometry, Mapping)
        and str(geometry.get("selection_basis") or "")
        == "source_table_year_plus_twelve_ownership"
    )


def _base_provenance_bound_native_candidate(
    *,
    page: Any,
    table: Any,
    geometry_tables: list[Mapping[str, Any]],
    table_geometry: Mapping[str, Any],
    final_box: tuple[float, float, float, float],
    final_amount_box: tuple[float, float, float, float],
    logical_page: int,
    expected_year: int,
    expected_month: int,
    expected_amount: Decimal,
    header_row: int,
    status_row: int,
    amount_row: int,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Bind raw cells through a source-owned base-page header and row pair."""

    from docmirror.plugins.credit_report.source_table_month_lattice import (
        resolve_unique_source_table_year_plus_twelve_ownership,
    )

    coordinate_page = _page_number(page, "page_number", "logical_page")
    if coordinate_page is None or coordinate_page != logical_page:
        return None
    cells_by_raw_row = _typed_cells_by_raw_geometry_row(
        table,
        coordinate_page=coordinate_page,
    )
    if cells_by_raw_row is None or not (
        header_row + 1 == status_row
        and status_row + 1 == amount_row
    ):
        return None
    header_cells = cells_by_raw_row.get(header_row)
    status_cells = cells_by_raw_row.get(status_row)
    amount_cells = cells_by_raw_row.get(amount_row)
    if not (
        header_cells
        and status_cells
        and amount_cells
        and set(status_cells) == set(range(13))
        and expected_month in amount_cells
        and _header_proves_month_ordinals(header_cells)
        and _year_cell_proves_row(status_cells[0], expected_year)
    ):
        return None
    # The bound base-page path accepts damaged header text, but never inferred
    # header geometry or a header cell owned by a different physical row/col.
    for month in range(1, 13):
        header_cell = header_cells[month]
        raw_header_box = _bbox(_get(header_cell, "bbox"))
        logical_header_box = _bbox(
            _matrix_get(table_geometry.get("cell_bboxes"), header_row, month)
        )
        if not (
            _exact_cell(header_cell)
            and _owned_cell_index(_get(header_cell, "col_index")) == month
            and raw_header_box is not None
            and logical_header_box is not None
            and _overlap_is_same_cell(raw_header_box, logical_header_box)
            and _nonempty_equal_token_evidence(header_cell)
        ):
            return None
    year_cell = status_cells[0]
    if not (
        _owned_cell_index(_get(year_cell, "col_index")) == 0
        and _nonempty_equal_token_evidence(year_cell)
    ):
        return None
    logical_year_box = _bbox(_matrix_get(table_geometry.get("cell_bboxes"), status_row, 0))
    raw_year_box = _bbox(_get(year_cell, "bbox"))
    if not (
        logical_year_box is not None
        and raw_year_box is not None
        and _overlap_is_same_cell(raw_year_box, logical_year_box)
    ):
        return None
    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        geometry_tables,
        logical_page=logical_page,
        expected_year=expected_year,
        active_months=(expected_month,),
        year_bbox=logical_year_box,
        status_bbox=final_box,
    )
    table_id = str(_get(table, "table_id") or _get(table, "id") or "")
    if not (
        lattice is not None
        and lattice.table_id == table_id
        and lattice.year_anchor_row_index == status_row
        and lattice.header_row_index == header_row
        and lattice.status_row_index == status_row
        and lattice.amount_row_index == amount_row
    ):
        return None
    status_cell = status_cells[expected_month]
    amount_cell = amount_cells[expected_month]
    token = _canonical_status(_get(status_cell, "text"))
    native_amount = _decimal(_get(amount_cell, "text"))
    native_box = lattice.month_bboxes[expected_month - 1]
    native_amount_box = lattice.amount_bboxes[expected_month - 1]
    raw_status_box = _bbox(_get(status_cell, "bbox"))
    raw_amount_box = _bbox(_get(amount_cell, "bbox"))
    if not (
        token is not None
        and _exact_cell(status_cell, require_single_token=True)
        and _exact_cell(amount_cell, require_single_token=True)
        and native_amount == expected_amount
        and raw_status_box is not None
        and raw_amount_box is not None
        and _overlap_is_same_cell(raw_status_box, native_box)
        and _overlap_is_same_cell(raw_amount_box, native_amount_box)
        and _overlap_is_same_cell(native_box, final_box)
        and _amount_overlap_is_same_cell(native_amount_box, final_amount_box)
        and _owned_cell_index(_get(status_cell, "col_index")) == expected_month
        and _owned_cell_index(_get(amount_cell, "col_index")) == expected_month
    ):
        return None
    return (
        token,
        _native_source_ref(
            page=page,
            table=table,
            row_index=status_row,
            column=expected_month,
            cell=status_cell,
            logical_bbox=native_box,
            field_name="status",
        ),
        _native_source_ref(
            page=page,
            table=table,
            row_index=amount_row,
            column=expected_month,
            cell=amount_cell,
            logical_bbox=native_amount_box,
            field_name="overdue_amount",
        ),
    )


def _provenance_bound_native_candidate(
    *,
    page: Any,
    table: Any,
    geometry_tables: list[Mapping[str, Any]],
    table_geometry: Mapping[str, Any],
    final_box: tuple[float, float, float, float],
    final_amount_box: tuple[float, float, float, float],
    logical_page: int,
    expected_year: int,
    expected_month: int,
    expected_amount: Decimal,
    status_row: int,
    amount_row: int,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Bind raw cells through an already-proved value-free continuation lattice."""

    from docmirror.plugins.credit_report.source_table_month_lattice import (
        resolve_unique_source_table_year_plus_twelve_ownership,
    )

    coordinate_page = _page_number(page, "page_number", "logical_page")
    if coordinate_page is None or coordinate_page != logical_page:
        return None
    cells_by_raw_row = _typed_cells_by_raw_geometry_row(
        table,
        coordinate_page=coordinate_page,
    )
    if cells_by_raw_row is None:
        return None
    status_cells = cells_by_raw_row.get(status_row)
    amount_cells = cells_by_raw_row.get(amount_row)
    year_anchor = _year_anchor_proves_pair(
        cells_by_raw_row,
        status_row=status_row,
        amount_row=amount_row,
        expected_year=expected_year,
    )
    if not (
        status_cells
        and amount_cells
        and expected_month in status_cells
        and expected_month in amount_cells
        and year_anchor is not None
    ):
        return None
    year_anchor_row, year_cell = year_anchor
    logical_year_box = _bbox(
        _matrix_get(table_geometry.get("cell_bboxes"), year_anchor_row, 0)
    )
    raw_year_box = _bbox(_get(year_cell, "bbox"))
    if not (
        logical_year_box is not None
        and raw_year_box is not None
        and _overlap_is_same_cell(raw_year_box, logical_year_box)
    ):
        return None
    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        geometry_tables,
        logical_page=logical_page,
        expected_year=expected_year,
        active_months=(expected_month,),
        year_bbox=logical_year_box,
        status_bbox=final_box,
    )
    table_id = str(_get(table, "table_id") or _get(table, "id") or "")
    if not (
        lattice is not None
        and lattice.table_id == table_id
        and lattice.year_anchor_row_index == year_anchor_row
        and lattice.status_row_index == status_row
        and lattice.amount_row_index == amount_row
    ):
        return None
    status_cell = status_cells[expected_month]
    amount_cell = amount_cells[expected_month]
    token = _canonical_status(_get(status_cell, "text"))
    native_amount = _decimal(_get(amount_cell, "text"))
    native_box = lattice.month_bboxes[expected_month - 1]
    native_amount_box = lattice.amount_bboxes[expected_month - 1]
    raw_status_box = _bbox(_get(status_cell, "bbox"))
    raw_amount_box = _bbox(_get(amount_cell, "bbox"))
    if not (
        token is not None
        and _exact_cell(status_cell, require_single_token=True)
        and _exact_cell(amount_cell, require_single_token=True)
        and native_amount == expected_amount
        and raw_status_box is not None
        and raw_amount_box is not None
        and _overlap_is_same_cell(raw_status_box, native_box)
        and _overlap_is_same_cell(raw_amount_box, native_amount_box)
        and _overlap_is_same_cell(native_box, final_box)
        and _amount_overlap_is_same_cell(native_amount_box, final_amount_box)
        and _owned_cell_index(_get(status_cell, "col_index")) == expected_month
        and _owned_cell_index(_get(amount_cell, "col_index")) == expected_month
    ):
        return None
    return (
        token,
        _native_source_ref(
            page=page,
            table=table,
            row_index=status_row,
            column=expected_month,
            cell=status_cell,
            logical_bbox=native_box,
            field_name="status",
        ),
        _native_source_ref(
            page=page,
            table=table,
            row_index=amount_row,
            column=expected_month,
            cell=amount_cell,
            logical_bbox=native_amount_box,
            field_name="overdue_amount",
        ),
    )


def _native_candidates(
    context: Any,
    *,
    final_ref: Mapping[str, Any],
    final_amount_ref: Mapping[str, Any],
    expected_year: int,
    expected_month: int,
    expected_amount: Decimal,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    from docmirror.plugins.credit_report.source_table_month_lattice import (
        detached_source_table_geometry_by_page,
        resolve_unique_source_table_year_plus_twelve_ownership,
    )

    final_box = _bbox(final_ref.get("bbox"))
    final_amount_box = _bbox(final_amount_ref.get("bbox"))
    logical_page = _integer(final_ref.get("logical_page") or final_ref.get("page"))
    if final_box is None or final_amount_box is None or logical_page is None:
        return []
    source_binding = _source_geometry_row_binding(
        final_ref,
        final_amount_ref,
        logical_page=logical_page,
    )
    base_source_binding = _base_source_geometry_row_binding(
        final_ref,
        final_amount_ref,
        logical_page=logical_page,
    )
    if (
        _declares_source_table_geometry(final_ref)
        or _declares_source_table_geometry(final_amount_ref)
    ) and (source_binding is None) == (base_source_binding is None):
        # A source-owned ref must carry exactly one complete, mutually agreeing
        # continuation or base-page binding.  Never fall back to a nearby row.
        return []
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for page in _candidate_pages(context, final_ref):
        coordinate_page = _page_number(page, "page_number", "logical_page")
        if coordinate_page is None or coordinate_page != logical_page:
            continue
        tables = _get(page, "tables")
        if not isinstance(tables, (list, tuple)):
            continue
        geometry_tables = detached_source_table_geometry_by_page([page]).get(logical_page, [])
        geometry_by_id: dict[str, list[Mapping[str, Any]]] = {}
        for geometry_table in geometry_tables:
            geometry_by_id.setdefault(str(geometry_table.get("table_id") or ""), []).append(geometry_table)
        for table in tables:
            table_id = str(_get(table, "table_id") or _get(table, "id") or "")
            table_geometry = geometry_by_id.get(table_id, [])
            if len(table_geometry) != 1:
                continue
            if source_binding is not None:
                bound_table_id, status_row, amount_row = source_binding
                if table_id != bound_table_id:
                    continue
                candidate = _provenance_bound_native_candidate(
                    page=page,
                    table=table,
                    geometry_tables=geometry_tables,
                    table_geometry=table_geometry[0],
                    final_box=final_box,
                    final_amount_box=final_amount_box,
                    logical_page=logical_page,
                    expected_year=expected_year,
                    expected_month=expected_month,
                    expected_amount=expected_amount,
                    status_row=status_row,
                    amount_row=amount_row,
                )
                if candidate is not None:
                    candidates.append(candidate)
                # A complete source-lattice binding is the only admissible
                # ownership proof for this continuation ref.  Never fall back
                # to a nearby header/table when its bound raw cell is unusable.
                continue
            if base_source_binding is not None:
                bound_table_id, header_row, status_row, amount_row = base_source_binding
                if table_id != bound_table_id:
                    continue
                candidate = _base_provenance_bound_native_candidate(
                    page=page,
                    table=table,
                    geometry_tables=geometry_tables,
                    table_geometry=table_geometry[0],
                    final_box=final_box,
                    final_amount_box=final_amount_box,
                    logical_page=logical_page,
                    expected_year=expected_year,
                    expected_month=expected_month,
                    expected_amount=expected_amount,
                    header_row=header_row,
                    status_row=status_row,
                    amount_row=amount_row,
                )
                if candidate is not None:
                    candidates.append(candidate)
                # Base-page source ownership is equally binding: malformed or
                # ambiguous raw evidence may not trigger the generic scan.
                continue
            cells_by_raw_row = _typed_cells_by_raw_geometry_row(
                table,
                coordinate_page=coordinate_page,
            )
            raw_cell_bboxes = table_geometry[0].get("cell_bboxes")
            if cells_by_raw_row is None or not isinstance(raw_cell_bboxes, (list, tuple)):
                continue
            for row_position in range(1, len(raw_cell_bboxes) - 1):
                header_cells = cells_by_raw_row.get(row_position - 1)
                status_cells = cells_by_raw_row.get(row_position)
                amount_cells = cells_by_raw_row.get(row_position + 1)
                if not header_cells or not status_cells or not amount_cells:
                    continue
                if not (
                    set(status_cells) == set(range(13))
                    and expected_month in amount_cells
                    and _header_proves_month_ordinals(header_cells)
                    and _year_cell_proves_row(status_cells[0], expected_year)
                ):
                    continue
                logical_year_box = _bbox(
                    _matrix_get(
                        table_geometry[0].get("cell_bboxes"),
                        row_position,
                        0,
                    )
                )
                if logical_year_box is None:
                    continue
                lattice = resolve_unique_source_table_year_plus_twelve_ownership(
                    geometry_tables,
                    logical_page=logical_page,
                    expected_year=expected_year,
                    active_months=(expected_month,),
                    year_bbox=logical_year_box,
                    status_bbox=final_box,
                )
                if not (
                    lattice is not None
                    and lattice.table_id == table_id
                    and lattice.status_row_index == row_position
                    and lattice.header_row_index == row_position - 1
                ):
                    continue
                status_cell = status_cells[expected_month]
                amount_cell = amount_cells[expected_month]
                token = _canonical_status(_get(status_cell, "text"))
                native_amount = _decimal(_get(amount_cell, "text"))
                native_box = lattice.month_bboxes[expected_month - 1]
                native_amount_box = lattice.amount_bboxes[expected_month - 1]
                raw_status_box = _bbox(_get(status_cell, "bbox"))
                raw_amount_box = _bbox(_get(amount_cell, "bbox"))
                if not (
                    token is not None
                    and _exact_cell(status_cell, require_single_token=True)
                    and _exact_cell(amount_cell, require_single_token=True)
                    and native_amount == expected_amount
                    and native_box is not None
                    and raw_status_box is not None
                    and raw_amount_box is not None
                    and _overlap_is_same_cell(raw_status_box, native_box)
                    and _overlap_is_same_cell(raw_amount_box, native_amount_box)
                    and _overlap_is_same_cell(native_box, final_box)
                    and _amount_overlap_is_same_cell(
                        native_amount_box,
                        final_amount_box,
                    )
                    and _owned_cell_index(_get(status_cell, "col_index")) == expected_month
                    and _owned_cell_index(_get(amount_cell, "col_index")) == expected_month
                ):
                    continue
                candidates.append(
                    (
                        token,
                        _native_source_ref(
                            page=page,
                            table=table,
                            row_index=row_position,
                            column=expected_month,
                            cell=status_cell,
                            logical_bbox=native_box,
                            field_name="status",
                        ),
                        _native_source_ref(
                            page=page,
                            table=table,
                            row_index=row_position + 1,
                            column=expected_month,
                            cell=amount_cell,
                            logical_bbox=native_amount_box,
                            field_name="overdue_amount",
                        ),
                    )
                )
    return candidates


class _MonthlySourceEvidence:
    """Read-only indexes over one sealed acquisition, shared by a repair pass.

    The exact resolver still decides admissibility. The index only narrows its
    input to *all* atoms claiming the requested IDs, including malformed and
    non-token claims, so indexing cannot hide duplicates or a competing owner.
    """

    def __init__(self, parse_result: Any) -> None:
        from docmirror.plugins.credit_report.source_table_month_lattice import (
            detached_source_table_geometry_by_page,
        )

        self.parse_result = parse_result
        self.geometry = detached_source_table_geometry_by_page(parse_result)
        self.tables: dict[tuple[int, str], list[tuple[Any, Any]]] = {}
        for page in _get(parse_result, "pages", ()) or ():
            logical = _owned_cell_index(_get(page, "page_number"))
            if logical is None or logical <= 0:
                continue
            for table in _get(page, "tables", ()) or ():
                table_id = str(_get(table, "table_id") or _get(table, "id") or "")
                self.tables.setdefault((logical, table_id), []).append((page, table))
        self.atoms: dict[str, list[Any]] = {}
        plane = _get(parse_result, "evidence_plane")
        evidence = _get(plane, "evidence")
        for atom in _get(evidence, "text_atoms", ()) or ():
            claims = {str(_get(atom, "id") or "")}
            refs = _get(atom, "source_refs", ())
            if isinstance(refs, Mapping):
                refs = refs.keys()
            elif isinstance(refs, str):
                refs = (refs,)
            elif not isinstance(refs, (list, tuple)):
                refs = ()
            claims.update(ref for ref in refs if isinstance(ref, str))
            for claim in claims - {""}:
                self.atoms.setdefault(claim, []).append(atom)
        self.bundle_tokens: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        domain = _get(_get(parse_result, "entities"), "domain_specific")
        bundles = domain.get("_page_evidence_bundles", ()) if isinstance(domain, Mapping) else ()
        observations: dict[str, list[tuple[tuple[int, str], int, dict[str, Any]]]] = {}
        for bundle_index, bundle in enumerate(bundles or ()):
            if not isinstance(bundle, Mapping):
                continue
            page = bundle.get("page")
            if _owned_cell_index(page) is None or page <= 0:
                continue
            for pool_name, pool in (("tokens", bundle), *(
                (name, bundle.get(name)) for name in ("local_structure_evidence", "micro_grid_evidence")
            )):
                if not isinstance(pool, Mapping) or pool.get("page", page) != page:
                    continue
                for token in pool.get("tokens", ()) or ():
                    if not isinstance(token, Mapping) or not isinstance(token.get("token_id"), str):
                        continue
                    source = str(token.get("source") or "").lower()
                    if any(part in source for part in ("semantic", "synthetic", "expanded", "table_cell", "line_projection")):
                        continue
                    observations.setdefault(token["token_id"], []).append(((bundle_index, pool_name), page, dict(token)))
        for token_id, claims in observations.items():
            # The same immutable token is often serialized in both producer
            # views. Collapse only identical cross-view aliases; repetitions
            # within one view and conflicting text/score/geometry remain
            # duplicate claims and therefore fail the exact resolver.
            signatures = {
                repr((page, sorted(token.items(), key=lambda item: str(item[0]))))
                for _pool, page, token in claims
            }
            pools = [pool for pool, _page, _token in claims]
            kept = claims[:1] if len(signatures) == 1 and len(set(pools)) == len(pools) else claims
            self.bundle_tokens[token_id] = [(page, token) for _pool, page, token in kept]
        self.token_cache: dict[tuple[int, tuple[str, ...]], Any] = {}
        self.row_cache: dict[tuple[int, str], Any] = {}
        self.witness_cache: dict[tuple[Any, ...], Any] = {}

    @staticmethod
    def ids(value: Any, *, allow_empty: bool = False) -> tuple[str, ...] | None:
        if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
            return None
        if any(not isinstance(item, str) or not item or item != item.strip() for item in value):
            return None
        return tuple(value) if len(set(value)) == len(value) else None

    def tokens(self, logical_page: int, ids: tuple[str, ...]) -> Any:
        from docmirror.plugins.credit_report.personal_detail_scanned.exact_evidence import (
            resolve_exact_page_token_atoms,
        )

        key = (logical_page, ids)
        if key not in self.token_cache:
            if any(len(self.bundle_tokens.get(token_id, ())) > 1 for token_id in ids):
                # The evidence builder may coalesce visually identical OCR
                # payloads. It cannot erase conflicting immutable ID/score
                # claims that remain in the original producer views.
                self.token_cache[key] = None
                return None
            atoms: list[Any] = []
            seen: set[int] = set()
            for token_id in ids:
                for atom in self.atoms.get(token_id, ()):
                    if id(atom) not in seen:
                        seen.add(id(atom))
                        atoms.append(atom)
            view = SimpleNamespace(
                evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
                entities=SimpleNamespace(domain_specific={
                    "_page_evidence_bundles": [
                        {"page": page, "tokens": [token]}
                        for token_id in ids for page, token in self.bundle_tokens.get(token_id, ())
                    ],
                }),
            )
            self.token_cache[key] = resolve_exact_page_token_atoms(
                view, ids, logical_page=logical_page, require_raw_tokens=True,
            )
        return self.token_cache[key]

    def confidence(self, logical_page: int, token: tuple[Any, ...]) -> float | None:
        text, box, token_id = token
        observations: list[Any] = []
        for atom in self.atoms.get(token_id, ()):
            source_kind = str(_get(atom, "source_kind") or "").lower()
            metadata = _get(atom, "metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            raw_kind = source_kind == "metadata_ocr_token" or (
                metadata.get("granularity") == "token"
                and ("ocr" in source_kind or source_kind in {
                    "metadata", "micro_grid_evidence_token", "local_structure_evidence_token",
                })
                and not any(part in source_kind for part in ("table", "cell", "line", "semantic"))
            )
            if raw_kind and _get(atom, "page_id") == f"page:{logical_page:04d}":
                observations.append(atom)
        if not observations:
            observations.extend(token for page, token in self.bundle_tokens.get(token_id, ()) if page == logical_page)
        if len(observations) != 1:
            return None
        observation = observations[0]
        if (
            str(_get(observation, "text") or _get(observation, "content") or "").strip() != text
            or _bbox(_get(observation, "bbox")) != box
        ):
            return None
        score = _get(observation, "confidence")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return None
        return float(score) if isfinite(score) and 0.0 <= score <= 1.0 else None


def _slot_contains(outer: tuple[float, ...], inner: tuple[float, ...]) -> bool:
    # A one-point allowance covers rounded PDF coordinates, not neighbouring
    # month cells or vertically merged observations.
    return bool(
        outer[0] - 1.0 <= inner[0] < inner[2] <= outer[2] + 1.0
        and outer[1] - 1.0 <= inner[1] < inner[3] <= outer[3] + 1.0
    )


def _monthly_raw_cell_tokens(
    evidence: _MonthlySourceEvidence, cell: Any, *, logical_page: int,
) -> tuple[tuple[Any, ...], ...] | None:
    ids = evidence.ids(_get(cell, "evidence_ids"), allow_empty=True)
    token_ids = evidence.ids(_get(cell, "token_ids"), allow_empty=True)
    if ids is None or token_ids is None or set(ids) != set(token_ids):
        return None
    if not ids:
        return ()
    box = _bbox(_get(cell, "bbox"))
    tokens = evidence.tokens(logical_page, ids)
    if box is None or tokens is None or any(not _slot_contains(box, token[1]) for token in tokens):
        return None
    return tokens


def _monthly_header_rows(table: Any, rows: Mapping[int, Any], geometry: Mapping[str, Any]) -> dict[int, Any]:
    """Retain the assembler's raw header owner when it is not a typed row.

    ``preserve_headers`` moves raw row zero into ``TableBlock.headers``. Its
    token IDs remain in the sealed geometry matrices; they are not replaced by
    IDs synthesized from header text or typed-container positions.
    """

    result = dict(rows)
    metadata = _get(table, "metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    headers = _get(table, "headers")
    if 0 in rows or metadata.get("preserve_headers", True) is not True or not isinstance(headers, (list, tuple)) or len(headers) != 13:
        return result
    # An invalid/duplicate typed claim must not be laundered through fallback.
    if any(
        isinstance(ref, Mapping) and ref.get("raw_row") == 0
        for row in _rows(table) for cell in _get(row, "cells", ()) or ()
        for ref in _get(cell, "source_cell_refs", ()) or ()
    ):
        return result
    raw_geometry = metadata.get("geometry")
    raw_geometry = raw_geometry if isinstance(raw_geometry, Mapping) else metadata
    cells: dict[int, dict[str, Any]] = {}
    # The twelve ordinal cells establish the header. Column zero may be an
    # unused blank label; retain it only when exact so a year anchored there
    # still has to pass the separate, unchanged year-owner proof.
    for col in (*range(1, 13), 0):
        box = _bbox(_matrix_get(raw_geometry.get("cell_bboxes"), 0, col))
        if (
            box is None or box != _bbox(_matrix_get(geometry.get("cell_bboxes"), 0, col))
            or _matrix_get(raw_geometry.get("cell_geometry_status"), 0, col) != "exact"
        ):
            if col == 0:
                continue
            return result
        spans = [
            span for span in raw_geometry.get("cell_spans", ()) or ()
            if isinstance(span, Mapping) and span.get("row") == 0 and span.get("col") == col
        ]
        if len(spans) > 1 or any(
            _owned_cell_index(span.get("row_span", 1)) is None or span.get("row_span", 1) < 1
            or (col != 0 and span.get("row_span", 1) != 1) or span.get("col_span", 1) != 1
            for span in spans
        ):
            if col == 0:
                continue
            return result
        cells[col] = {
            "bbox": box, "geometry_status": "exact", "row_span": spans[0].get("row_span", 1) if spans else 1, "col_span": 1,
            "evidence_ids": _matrix_get(raw_geometry.get("cell_evidence_ids"), 0, col),
            "token_ids": _matrix_get(raw_geometry.get("cell_token_ids"), 0, col),
        }
    result[0] = cells
    return result


def _monthly_raw_pair(
    evidence: _MonthlySourceEvidence,
    *, logical_page: int, table_id: str, year: int, month: int,
    year_row: int, status_row: int, amount_row: int,
) -> dict[str, Any] | None:
    """Authenticate a physical row pair without trusting a published value."""

    from docmirror.plugins.credit_report.source_table_month_lattice import (
        resolve_unique_source_table_year_plus_twelve_ownership,
    )

    cache_key = (logical_page, table_id, year, month, year_row, status_row, amount_row)
    if cache_key in evidence.witness_cache:
        return evidence.witness_cache[cache_key]
    evidence.witness_cache[cache_key] = None
    owners = evidence.tables.get((logical_page, table_id), ())
    geometries = [
        table for table in evidence.geometry.get(logical_page, ())
        if table.get("table_id") == table_id
    ]
    if len(owners) != 1 or len(geometries) != 1 or amount_row != status_row + 1:
        return None
    page, table = owners[0]
    source_page = _owned_cell_index(_get(page, "source_page_number"))
    if source_page is None or source_page <= 0:
        return None
    geometry = geometries[0]
    if (
        geometry.get("source_logical_page") != logical_page or geometry.get("canonical_geometry_registered")
        or geometry.get("source_page") != source_page
    ):
        return None
    row_key = (logical_page, table_id)
    if row_key not in evidence.row_cache:
        evidence.row_cache[row_key] = _typed_cells_by_raw_geometry_row(table, coordinate_page=logical_page)
    rows = evidence.row_cache[row_key]
    if not isinstance(rows, Mapping):
        return None
    rows = _monthly_header_rows(table, rows, geometry)
    year_cell = rows.get(year_row, {}).get(0)
    if not _exact_cell(year_cell):
        return None
    year_tokens = _monthly_raw_cell_tokens(evidence, year_cell, logical_page=logical_page)
    year_text = " ".join(token[0] for token in year_tokens or ())
    years = _YEAR_RE.findall(year_text)
    if year_tokens is None or len(years) != 1 or int(years[0]) != year:
        return None
    year_box = _bbox(_get(year_cell, "bbox"))
    if year_box is None or _bbox(_matrix_get(geometry.get("cell_bboxes"), year_row, 0)) != year_box:
        return None
    bands = [band for band in geometry.get("row_bands", ()) or () if isinstance(band, Mapping) and band.get("index") == status_row]
    columns = [band for band in geometry.get("col_bands", ()) or () if isinstance(band, Mapping)]
    column_bands = {band.get("index"): band for band in columns}
    if len(bands) != 1 or len(columns) != 13 or set(column_bands) != set(range(13)):
        # Effective split-column layouts are not silently treated as raw
        # thirteen-column tables. They retain their explicit uncertainty.
        return None
    try:
        status_box = tuple(float(value) for value in (
            column_bands[1]["x0"], bands[0]["y0"], column_bands[12]["x1"], bands[0]["y1"],
        ))
    except (KeyError, TypeError, ValueError):
        return None
    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        geometries, logical_page=logical_page, expected_year=year,
        active_months=(month,), year_bbox=year_box, status_bbox=status_box,
    )
    if (
        lattice is None or lattice.status_row_index != status_row
        or lattice.amount_row_index != amount_row or lattice.year_anchor_row_index != year_row
    ):
        return None
    header_candidates: list[tuple[int, tuple[str, ...]]] = []
    for raw_row, cells in rows.items():
        if raw_row >= status_row or not set(range(1, 13)).issubset(cells):
            continue
        matched: list[str] = []
        matched_columns = 0
        contradiction = False
        for ordinal in range(1, 13):
            cell = cells[ordinal]
            if (
                not _exact_cell(cell) or _integer(_get(cell, "row_span", 1)) != 1
                or _bbox(_matrix_get(geometry.get("cell_bboxes"), raw_row, ordinal)) != _bbox(_get(cell, "bbox"))
            ):
                continue
            tokens = _monthly_raw_cell_tokens(evidence, cell, logical_page=logical_page)
            text = "".join(token[0] for token in tokens or ()).strip()
            if not re.fullmatch(r"0?\d{1,2}", text):
                continue
            if 1 <= int(text) <= 12 and int(text) != ordinal:
                contradiction = True
            if int(text) == ordinal and tokens:
                matched_columns += 1
                matched.extend(token[2] for token in tokens)
        if not contradiction and matched_columns >= 9 and len(set(matched)) == len(matched):
            header_candidates.append((raw_row, tuple(matched)))
    if not header_candidates:
        return None
    header_row, header_ids = max(header_candidates, key=lambda item: item[0])
    result: dict[str, Any] = {
        "logical_page": logical_page, "source_page": source_page, "table_id": table_id,
        "year": year, "month": month, "year_row": year_row,
        "status_row": status_row, "amount_row": amount_row,
        "header_row": header_row, "year_evidence_ids": tuple(token[2] for token in year_tokens),
        "header_evidence_ids": header_ids, "lattice_provenance": lattice.provenance_dict(), "fields": {},
    }
    for field_name, raw_row, slot in (
        ("status", status_row, lattice.month_bboxes[month - 1]),
        ("overdue_amount", amount_row, lattice.amount_bboxes[month - 1]),
    ):
        cell = rows.get(raw_row, {}).get(month)
        if cell is None or not _exact_cell(cell):
            continue
        tokens = _monthly_raw_cell_tokens(evidence, cell, logical_page=logical_page)
        if tokens is None:
            continue
        contained = tuple(token for token in tokens if _slot_contains(slot, token[1]))
        ambiguous = any(
            min(slot[2], token[1][2]) > max(slot[0], token[1][0])
            and min(slot[3], token[1][3]) > max(slot[1], token[1][1])
            and token not in contained for token in tokens
        )
        if ambiguous:
            continue
        scores = [evidence.confidence(logical_page, token) for token in contained]
        result["fields"][field_name] = {
            "bbox": tuple(slot), "evidence_ids": tuple(token[2] for token in contained),
            "parent_evidence_ids": tuple(token[2] for token in tokens),
            "raw_value": " ".join(token[0] for token in sorted(contained, key=lambda item: (item[1][1], item[1][0]))),
            "confidence": min(scores) if scores and all(score is not None for score in scores) else None,
        }
    evidence.witness_cache[cache_key] = result
    return result


def monthly_field_slot_identity(ref: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Identify a narrow monthly slot; authentication is a separate operation."""

    proof = ref.get("monthly_slot_proof")
    if ref.get("source") != _MONTHLY_SLOT_SOURCE or not isinstance(proof, Mapping) or proof.get("schema") != _MONTHLY_SLOT_SCHEMA:
        return None
    field_name = ref.get("field_name")
    canonical_field = "status" if field_name in _STATUS_FIELDS else "overdue_amount" if field_name in _AMOUNT_FIELDS else ""
    indices = [ref.get("logical_page"), ref.get("source_page"), proof.get("year"), proof.get("month"), proof.get("year_row"), proof.get("status_row"), proof.get("amount_row")]
    if (
        not canonical_field or any(_owned_cell_index(value) is None for value in indices)
        or not isinstance(proof.get("account_id"), str) or not proof["account_id"].strip()
        or indices[0] <= 0 or indices[1] <= 0 or not 1900 <= indices[2] <= 2099
        or not 1 <= indices[3] <= 12 or indices[6] != indices[5] + 1
        or _owned_cell_index(ref.get("column")) != indices[3]
        or _owned_cell_index(ref.get("row")) != proof.get("status_row" if canonical_field == "status" else "amount_row")
        or not isinstance(ref.get("table_id"), str) or not ref["table_id"]
        or _bbox(ref.get("bbox")) is None or ref.get("geometry_scope") != "cell"
        or ref.get("geometry_status") != "exact" or ref.get("coordinate_system") != _TOP_LEFT_PDF_COORDINATES
        or _MonthlySourceEvidence.ids(ref.get("evidence_ids"), allow_empty=True) is None
    ):
        return None
    return (_MONTHLY_SLOT_SCHEMA, *indices, ref["table_id"], canonical_field, _bbox(ref["bbox"]), proof["account_id"])


def resolve_sealed_monthly_field_slot(
    parse_result: Any, ref: Mapping[str, Any], *, evidence: _MonthlySourceEvidence | None = None,
) -> dict[str, Any] | None:
    """Re-prove a blank/populated slot against the actual sealed source table."""

    if monthly_field_slot_identity(ref) is None:
        return None
    proof = ref["monthly_slot_proof"]
    evidence = evidence or _MonthlySourceEvidence(parse_result)
    pair = _monthly_raw_pair(
        evidence, logical_page=ref["logical_page"], table_id=ref["table_id"],
        year=proof["year"], month=proof["month"], year_row=proof["year_row"],
        status_row=proof["status_row"], amount_row=proof["amount_row"],
    )
    canonical_field = "status" if ref["field_name"] in _STATUS_FIELDS else "overdue_amount"
    field = pair.get("fields", {}).get(canonical_field) if pair is not None else None
    if (
        field is None or pair["source_page"] != ref["source_page"]
        or field["bbox"] != _bbox(ref["bbox"])
        or set(field["evidence_ids"]) != set(_MonthlySourceEvidence.ids(ref.get("evidence_ids"), allow_empty=True) or ())
        or any(proof.get(key) != pair[key] for key in ("header_row",))
        or any(tuple(proof.get(key) or ()) != pair[key] for key in ("year_evidence_ids", "header_evidence_ids"))
        or tuple(proof.get("parent_evidence_ids") or ()) != field["parent_evidence_ids"]
    ):
        return None
    return {**field, "source_page": pair["source_page"]}


def authenticated_monthly_field_slots(
    context: Any, record: Mapping[str, Any], *, evidence: _MonthlySourceEvidence | None = None,
) -> dict[str, dict[str, Any]]:
    """Map a registered grid/month to independently proven raw source slots.

    The transform comes from the canonical table object, never from a caller's
    field ref. All repair selection subsequently happens in the raw origin
    page, while the original registered locator remains available for audits.
    """

    parse_result = _get(context, "parse_result", context)
    evidence = evidence or _MonthlySourceEvidence(parse_result)
    year_month = _performance_year_month(record)
    values = _values(record)
    account_ids = {str(owner.get("account_id") or "").strip() for owner in (values, record)} - {""}
    if year_month is None or len(account_ids) != 1:
        return {}
    year, month = year_month
    pages = _get(context, "pages")
    if not isinstance(pages, (list, tuple)):
        pages = _get(parse_result, "pages", ())
    found: dict[tuple[Any, ...], tuple[dict[str, Any], tuple[float, ...], int, dict[str, Any]]] = {}
    for ref in _record_refs(record):
        field_name = str(ref.get("field_name") or "")
        if field_name not in (*_STATUS_FIELDS, *_AMOUNT_FIELDS) or ref.get("geometry_scope") != "cell":
            continue
        logical = _owned_cell_index(ref.get("logical_page") or ref.get("page"))
        box = _bbox(ref.get("bbox"))
        provenance = ref.get("geometry_provenance")
        if (
            logical is None or logical <= 0 or box is None or ref.get("col") != month
            or ref.get("geometry_status") not in (None, "", "exact")
            or ref.get("coordinate_system") != _TOP_LEFT_PDF_COORDINATES
            or not isinstance(provenance, Mapping)
            or _owned_geometry_provenance(provenance) not in _OWNED_MONTH_GEOMETRIES
        ):
            continue
        status_row = _owned_cell_index(provenance.get("status_row_index"))
        amount_row = _owned_cell_index(provenance.get("amount_row_index"))
        year_row = _owned_cell_index(provenance.get("year_anchor_row_index"))
        if status_row is None or amount_row is None or year_row is None:
            continue
        for page in pages or ():
            if _get(page, "page_number") != logical:
                continue
            for table in _get(page, "tables", ()) or ():
                table_id = str(_get(table, "table_id") or _get(table, "id") or "")
                if table_id != provenance.get("table_id"):
                    continue
                metadata = _get(table, "metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                origin = metadata.get("source_logical_page", logical)
                if _owned_cell_index(origin) is None or origin <= 0 or metadata.get("coordinate_logical_page", logical) != logical:
                    continue
                raw_affine = metadata.get("source_to_canonical_affine")
                if raw_affine is None and origin == logical and not metadata.get("canonical_geometry"):
                    affine = (1.0, 1.0, 0.0, 0.0)
                elif isinstance(raw_affine, Mapping):
                    try:
                        affine = tuple(float(raw_affine[key]) for key in ("scale_x", "scale_y", "offset_x", "offset_y"))
                    except (KeyError, TypeError, ValueError):
                        continue
                else:
                    continue
                if not all(isfinite(value) for value in affine) or affine[0] <= 0.0 or affine[1] <= 0.0:
                    continue
                pair = _monthly_raw_pair(
                    evidence, logical_page=origin, table_id=table_id, year=year, month=month,
                    year_row=year_row, status_row=status_row, amount_row=amount_row,
                )
                if pair is None or any(
                    owner.get(key) is not None and owner.get(key) != expected
                    for owner in (metadata, provenance, ref)
                    for key, expected in (("source_page", pair["source_page"]), ("source_logical_page", origin))
                ):
                    continue
                canonical_field = "status" if field_name in _STATUS_FIELDS else "overdue_amount"
                physical = pair.get("fields", {}).get(canonical_field) if pair is not None else None
                if physical is None:
                    continue
                sx, sy, ox, oy = affine
                raw_box = physical["bbox"]
                registered = (raw_box[0] * sx + ox, raw_box[1] * sy + oy, raw_box[2] * sx + ox, raw_box[3] * sy + oy)
                matcher = _overlap_is_same_cell if canonical_field == "status" else _amount_overlap_is_same_cell
                if not matcher(registered, box):
                    continue
                marker = (origin, table_id, year, month, year_row, status_row, amount_row, logical, affine)
                entry = found.setdefault(marker, (pair, affine, logical, {}))
                field_refs = entry[3].setdefault(canonical_field, {})
                # Trimmed/full-box aliases have already independently passed
                # same-cell overlap. They may share one detector row, but two
                # different row owners must not be collapsed.
                field_marker = _owned_cell_index(ref.get("row"))
                field_refs.setdefault(field_marker, dict(ref))
    if len(found) != 1:
        return {}
    pair, affine, registered_page, field_witnesses = next(iter(found.values()))
    if any(len(witnesses) != 1 for witnesses in field_witnesses.values()):
        return {}
    fallback_ref = next(iter(next(iter(field_witnesses.values())).values()))
    result: dict[str, dict[str, Any]] = {}
    for canonical_field, field in pair["fields"].items():
        witnesses = field_witnesses.get(canonical_field)
        registered_ref = next(iter(witnesses.values())) if witnesses else fallback_ref
        actual_field = next((name for name in (_STATUS_FIELDS if canonical_field == "status" else _AMOUNT_FIELDS) if name in values), canonical_field)
        sx, sy, ox, oy = affine
        box = field["bbox"]
        proof = {
            "schema": _MONTHLY_SLOT_SCHEMA,
            "account_id": next(iter(account_ids)),
            **{key: pair[key] for key in ("year", "month", "year_row", "status_row", "amount_row", "header_row")},
            "year_evidence_ids": list(pair["year_evidence_ids"]),
            "header_evidence_ids": list(pair["header_evidence_ids"]),
            "parent_evidence_ids": list(field["parent_evidence_ids"]),
        }
        registered_box = [box[0] * sx + ox, box[1] * sy + oy, box[2] * sx + ox, box[3] * sy + oy]
        registered_row = _owned_cell_index(registered_ref.get("row"))
        if registered_row is None:
            continue
        registered_is_status = registered_ref.get("field_name") in _STATUS_FIELDS
        relative_row = (0 if canonical_field == "status" else 1) - (0 if registered_is_status else 1)
        if registered_row + relative_row < 0:
            continue
        registered_field_ref = {
            **deepcopy(registered_ref), "field_name": actual_field,
            "row": registered_row + relative_row, "col": month, "bbox": registered_box,
            "geometry_scope": "cell", "geometry_status": "exact",
            "geometry_provenance": {
                **deepcopy(registered_ref.get("geometry_provenance") or {}),
                **deepcopy(pair["lattice_provenance"]),
                "selection_basis": "source_table_year_plus_twelve_ownership", "source": "source_table_geometry",
                "table_id": pair["table_id"], "status_row_index": pair["status_row"], "amount_row_index": pair["amount_row"],
                "year_anchor_row_index": pair["year_row"], "coordinate_system": _TOP_LEFT_PDF_COORDINATES,
                "logical_page": registered_page, "source_logical_page": pair["logical_page"],
                "source_page": pair["source_page"], "value_inputs_used": False,
            },
        }
        if not witnesses:
            # A missing field may borrow its sibling's layout anchor, never
            # that sibling's token/acquisition identity.
            for key in ("token_ids", "evidence_ids", "acquisition_id"):
                registered_field_ref.pop(key, None)
            registered_field_ref["evidence_ids"] = list(field["evidence_ids"])
        result[actual_field] = {
            "source": _MONTHLY_SLOT_SOURCE, "evidence_plane": "sealed_native_source_table",
            "logical_page": pair["logical_page"], "source_page": pair["source_page"],
            "table_id": pair["table_id"], "row": pair["status_row" if canonical_field == "status" else "amount_row"],
            "column": month, "field_name": actual_field, "geometry_scope": "cell", "geometry_status": "exact",
            "coordinate_system": _TOP_LEFT_PDF_COORDINATES, "bbox": list(box),
            "evidence_ids": list(field["evidence_ids"]), "monthly_slot_proof": proof,
            "registered_logical_page": registered_page,
            "registered_bbox": registered_box,
            "registered_source_ref": registered_field_ref,
            "source_ocr_confidence": field["confidence"], "observed_raw": field["raw_value"],
        }
    return result


def _target_record_id(record: Mapping[str, Any], *, year: int, month: int) -> str:
    values = _values(record)
    return str(
        values.get("repayment_id")
        or record.get("repayment_id")
        or values.get("monthly_performance_id")
        or record.get("record_id")
        or f"{values.get('grid_id') or record.get('grid_id') or 'repayment'}:{year:04d}-{month:02d}"
    ).strip()


def _withhold_status(
    record: dict[str, Any],
    *,
    final_token: str,
    native_token: str,
) -> None:
    values = _mutable_values(record)
    for field_name in _STATUS_FIELDS:
        values.pop(field_name, None)
        if values is not record:
            record.pop(field_name, None)
    unresolved = record.setdefault("_unresolved_fields", [])
    if not isinstance(unresolved, list):
        unresolved = list(unresolved or ())
        record["_unresolved_fields"] = unresolved
    if "status" not in unresolved:
        unresolved.append("status")
    record["extraction_status"] = "review"
    if values is not record:
        values["extraction_status"] = "review"
    canonical_raw = record.setdefault("canonical_raw", {})
    if not isinstance(canonical_raw, dict):
        canonical_raw = {}
        record["canonical_raw"] = canonical_raw
    existing = canonical_raw.get("status")
    observations = (
        list(existing) if isinstance(existing, (list, tuple)) else [] if existing in (None, "") else [existing]
    )
    for candidate in (final_token, native_token):
        if not any(candidate == observed for observed in observations):
            observations.append(candidate)
    canonical_raw["status"] = observations


def _preserved_monthly_status_observation(
    context: Any, record: Mapping[str, Any], *, native_token: str,
    year: int, month: int, registered_ref: Mapping[str, Any],
) -> str | None:
    """Recover a diagnostic observation, never invent an already-null value."""

    observations: set[str] = set()
    record_audit = record.get("audit")
    if isinstance(record_audit, Mapping) and record_audit.get("reason") == "corrected_status_planes_disagree":
        raw = record_audit.get("observations")
        raw = raw if isinstance(raw, Mapping) else {}
        fallback, exact = raw.get("fallback"), raw.get("exact_source_cell")
        if (
            isinstance(fallback, (list, tuple)) and len(fallback) == 1
            and isinstance(exact, (list, tuple)) and len(exact) == 1
            and _canonical_status(exact[0]) == native_token
            and (token := _canonical_status(fallback[0])) is not None and token != native_token
        ):
            observations.add(token)
    target = _target_record_id(record, year=year, month=month)
    registered_box = _bbox(registered_ref.get("bbox"))
    registered_page = registered_ref.get("logical_page") or registered_ref.get("page")
    for issue in getattr(context, "_personal_detail_extraction_issues", ()) or ():
        if not isinstance(issue, Mapping):
            continue
        values = issue.get("normalized")
        values = values if isinstance(values, Mapping) else issue
        if not (
            values.get("issue_code") == "candidate_b_independent_plane_repayment_status_conflict"
            and values.get("parser_stage") == "candidate_b_cross_plane_repayment_reconciliation"
            and values.get("target_dataset") == "repayment_records"
            and values.get("target_record_id") == target and values.get("field_name") in _STATUS_FIELDS
            and values.get("status", "requires_review") in {"open", "requires_review"}
        ):
            continue
        raw = values.get("observed_value")
        if not isinstance(raw, Mapping) or _canonical_status(raw.get("native_static")) != native_token:
            continue
        corrected = _canonical_status(raw.get("corrected_page"))
        if corrected is None or corrected == native_token or registered_box is None:
            continue
        # The earlier two-plane issue must still name this very physical
        # month on both planes; matching only a grid-local record ID is not
        # enough to upgrade it into an exact sealed-cell conflict.
        planes: set[str] = set()
        for prior_ref in values.get("source_refs", ()) or ():
            if not isinstance(prior_ref, Mapping):
                continue
            box = _bbox(prior_ref.get("bbox"))
            if (
                prior_ref.get("field_name") in _STATUS_FIELDS and prior_ref.get("geometry_scope") == "cell"
                and prior_ref.get("geometry_status") in (None, "", "exact")
                and prior_ref.get("coordinate_system") == _TOP_LEFT_PDF_COORDINATES
                and (prior_ref.get("logical_page") or prior_ref.get("page")) == registered_page
                and prior_ref.get("col") == month and box is not None
                and _overlap_is_same_cell(registered_box, box)
            ):
                planes.add(str(prior_ref.get("evidence_plane") or ""))
        if {"native_static", "corrected_page"}.issubset(planes):
            observations.add(corrected)
    return next(iter(observations)) if len(observations) == 1 else None


def _guard_monthly_source_uncertainties(
    context: Any, records: list[dict[str, Any]], audit: dict[str, Any],
) -> set[int]:
    """Keep weak readings and earlier field-local contradictions explicit."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    evidence = _MonthlySourceEvidence(context.parse_result)
    handled: set[int] = set()
    overlay = getattr(context, "_ocr_correction_overlay", None)
    confirmer = getattr(overlay, "monthly_field_confirmation", None)
    for record in records:
        slots = authenticated_monthly_field_slots(context, record, evidence=evidence)
        status_refs = [ref for field, ref in slots.items() if field in _STATUS_FIELDS]
        if len(status_refs) != 1:
            continue
        ref = status_refs[0]
        native_token = _canonical_status(ref.get("observed_raw"))
        current = _single_status(record)
        current_token = current[1] if current is not None else ""
        score = ref.get("source_ocr_confidence")
        weak_native = bool(
            native_token is not None and isinstance(score, (int, float)) and not isinstance(score, bool)
            and score < MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM
        )
        confirmation = confirmer(ref, value=current_token) if current_token and callable(confirmer) else None
        if weak_native and confirmation is not None and confirmation.confidence > score:
            handled.add(id(record))
            audit["independent_monthly_field_confirmations"] += 1
            continue
        proof = ref["monthly_slot_proof"]
        target = _target_record_id(record, year=proof["year"], month=proof["month"])
        amount = _single_amount(record)
        native_is_invalid_digit = bool(native_token in {"1", "2", "3", "4", "5", "6", "7"} and amount is not None and amount <= 0)
        conflict_token = (
            current_token if weak_native and current_token and current_token != native_token
            else _preserved_monthly_status_observation(
                context, record, native_token=native_token, year=proof["year"], month=proof["month"],
                registered_ref=ref["registered_source_ref"],
            )
            if not current_token and native_token is not None else None
        )
        if native_is_invalid_digit:
            conflict_token = None
            # A weak but semantically impossible raw digit is not a competing
            # reading against a valid symbolic status. Withhold the digit
            # itself if published; leave a separately valid symbol unchanged.
            if current_token and current_token != native_token:
                audit["native_numeric_witnesses_rejected_for_nonpositive_amount"] += 1
                handled.add(id(record))
                continue
        if weak_native:
            _withhold_status(record, final_token=current_token or native_token, native_token=native_token)
            audit["low_source_ocr_confidence_withheld"] += 1
            record_issue(context, make_issue(
                category="ocr_cell_level_error",
                issue_code="candidate_b_monthly_source_ocr_confidence_unresolved",
                message="A legal monthly status had insufficient source OCR confidence and no independently confirmed same-slot page reading; only the status was withheld.",
                parser_stage="candidate_b_final_native_source_cell_guard",
                target_dataset="repayment_records", target_record_id=target, field_name="status_code",
                observed_value={
                    "observed_status": native_token, "source_ocr_confidence": score,
                    "minimum_source_ocr_confidence": MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM,
                },
                candidate_value={"resolution": "withheld_pending_independent_page_evidence"},
                source_refs=({**deepcopy(ref), "field_name": "status_code"}, deepcopy(ref["registered_source_ref"])),
                reason_codes=("low_source_ocr_confidence", "exact_monthly_source_slot", "independent_page_confirmation_missing", "normalized_value_withheld"),
            ))
            handled.add(id(record))
        if conflict_token is None or native_token is None:
            continue
        # The currently withheld field has no value. Report the preserved
        # conflicting observation explicitly, never fabricate a current one.
        _withhold_status(record, final_token=conflict_token, native_token=native_token)
        native_ref = {
            **deepcopy(ref), "source": "sealed_native_physical_table_cell",
            "col": ref["column"], "field_name": "status",
            "token_ids": list(ref["evidence_ids"]),
        }
        record_issue(context, make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_native_source_cell_repayment_status_conflict",
            message="A preserved corrected-plane monthly status disagreed with the unique sealed source-cell reading; the already-withheld field remains unresolved.",
            parser_stage="candidate_b_final_native_source_cell_guard",
            target_dataset="repayment_records", target_record_id=target, field_name="status_code",
            observed_value={
                "corrected_final": conflict_token, "sealed_native_source_cell": native_token,
                "paired_status_amount": format(amount, "f") if amount is not None else None,
                "corrected_final_already_withheld": not bool(current_token),
            },
            candidate_value={"resolution": "withheld_pending_review"},
            source_refs=(deepcopy(ref["registered_source_ref"]), native_ref),
            reason_codes=("exact_same_month_source_cell", "preserved_source_plane_conflict", "monthly_status_conflict", "normalized_value_withheld"),
        ))
        audit["conflicts_withheld"] += 1
        audit["preserved_source_plane_conflicts"] += 1
        handled.add(id(record))
    return handled


def apply_candidate_b_native_status_conflict_guard(
    context: Any,
    records: Iterable[dict[str, Any]],
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Withhold exact same-cell disagreements; do nothing in every other case.

    ``enabled`` defaults to false so importing this helper cannot change the
    digital-personal, enterprise, or non-Candidate-B paths accidentally.
    """

    audit = {
        "enabled": bool(enabled),
        "records_checked": 0,
        "unique_native_witnesses": 0,
        "native_numeric_witnesses_rejected_for_nonpositive_amount": 0,
        "agreements": 0,
        "conflicts_withheld": 0,
        "low_source_ocr_confidence_withheld": 0,
        "independent_monthly_field_confirmations": 0,
        "preserved_source_plane_conflicts": 0,
    }
    if not enabled:
        return audit
    parse_result = getattr(context, "parse_result", None)
    if parse_result is None:
        return audit

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    record_list = [record for record in records if isinstance(record, dict)]
    handled = _guard_monthly_source_uncertainties(context, record_list, audit)
    for record in record_list:
        if id(record) in handled:
            continue
        status = _single_status(record)
        year_month = _performance_year_month(record)
        amount = _single_amount(record)
        if year_month is None or amount is None:
            continue
        year, month = year_month
        if status is None:
            record_audit = record.get("audit")
            has_preserved_audit = isinstance(record_audit, Mapping) and record_audit.get("reason") == "corrected_status_planes_disagree"
            target = _target_record_id(record, year=year, month=month)
            has_preserved_issue = any(
                values.get("issue_code") == "candidate_b_independent_plane_repayment_status_conflict"
                and values.get("target_record_id") == target
                for issue in getattr(context, "_personal_detail_extraction_issues", ()) or ()
                if isinstance(issue, Mapping)
                for values in (issue.get("normalized") if isinstance(issue.get("normalized"), Mapping) else issue,)
            )
            if not has_preserved_audit and not has_preserved_issue:
                continue
        final_ref = _exact_final_status_ref(record, month=month)
        if final_ref is None:
            continue
        final_amount_ref = _exact_final_amount_ref(
            record,
            month=month,
            status_ref=final_ref,
        )
        if final_amount_ref is None:
            continue
        audit["records_checked"] += 1
        native = _native_candidates(
            context,
            final_ref=final_ref,
            final_amount_ref=final_amount_ref,
            expected_year=year,
            expected_month=month,
            expected_amount=amount,
        )
        if len(native) != 1:
            continue
        audit["unique_native_witnesses"] += 1
        native_token, native_ref, native_amount_ref = native[0]
        if native_token in {"1", "2", "3", "4", "5", "6", "7"} and amount <= 0:
            # A digit status denotes positive delinquency aging.  The monthly
            # grid contract already rejects a digit paired with a zero or
            # negative amount, so such a native OCR token cannot serve as a
            # competing witness against a valid symbolic final status.
            audit[
                "native_numeric_witnesses_rejected_for_nonpositive_amount"
            ] += 1
            continue
        already_withheld = status is None
        final_token = status[1] if status is not None else _preserved_monthly_status_observation(
            context, record, native_token=native_token, year=year, month=month, registered_ref=final_ref,
        )
        if final_token is None:
            continue
        if native_token == final_token:
            audit["agreements"] += 1
            continue

        target_record_id = _target_record_id(record, year=year, month=month)
        corrected_ref = {**dict(final_ref), "evidence_plane": "corrected_final"}
        corrected_amount_ref = {
            **dict(final_amount_ref),
            "evidence_plane": "corrected_final",
        }
        _withhold_status(
            record,
            final_token=final_token,
            native_token=native_token,
        )
        record_issue(
            context,
            make_issue(
                category="ocr_cell_level_error",
                issue_code="candidate_b_native_source_cell_repayment_status_conflict",
                message=(
                    "The final monthly status disagreed with the unique exact sealed "
                    "source-table cell at the same physical month location; the status "
                    "was withheld without choosing either value."
                ),
                parser_stage="candidate_b_final_native_source_cell_guard",
                target_dataset="repayment_records",
                target_record_id=target_record_id,
                field_name="status_code",
                observed_value={
                    "corrected_final": final_token,
                    "sealed_native_source_cell": native_token,
                    "paired_status_amount": format(amount, "f"),
                    **({"corrected_final_already_withheld": True} if already_withheld else {}),
                },
                candidate_value=None,
                source_refs=(
                    corrected_ref,
                    corrected_amount_ref,
                    native_ref,
                    native_amount_ref,
                ),
                reason_codes=(
                    "exact_year_plus_twelve_month_ownership",
                    "unique_same_cell_native_witness",
                    "paired_amount_agreement",
                    "monthly_status_conflict",
                    "normalized_value_withheld",
                ),
            ),
        )
        audit["conflicts_withheld"] += 1
        if already_withheld:
            audit["preserved_source_plane_conflicts"] += 1
    return audit


__all__ = [
    "MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM",
    "apply_candidate_b_native_status_conflict_guard",
    "authenticated_monthly_field_slots",
    "monthly_field_slot_identity",
    "resolve_sealed_monthly_field_slot",
]
