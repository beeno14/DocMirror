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
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

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
_DECIMAL_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")


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


def _row_cell_map(row: Any) -> dict[int, Any] | None:
    cells = _get(row, "cells")
    if not isinstance(cells, (list, tuple)):
        return None
    result: dict[int, Any] = {}
    for fallback_index, cell in enumerate(cells):
        column = _integer(_get(cell, "col_index"))
        if column is None:
            column = fallback_index
        if column in result:
            return None
        result[column] = cell
    return result


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
    rows: list[Any],
    *,
    status_row: int,
    amount_row: int,
    expected_year: int,
) -> tuple[int, Any] | None:
    """Bind one typed year anchor to a headerless continuation row pair."""

    matches: list[tuple[int, Any]] = []
    for row_position in range(0, min(status_row, len(rows) - 1) + 1):
        cells = _row_cell_map(rows[row_position])
        cell = cells.get(0) if cells else None
        if cell is None:
            continue
        explicit_row = _integer(_get(cell, "row_index"))
        explicit_col = _integer(_get(cell, "col_index"))
        row_span = _integer(_get(cell, "row_span", 1)) or 1
        col_span = _integer(_get(cell, "col_span", 1)) or 1
        years = _YEAR_RE.findall(str(_get(cell, "text") or ""))
        if not (
            _exact_cell(cell, require_single_token=True)
            and (explicit_row is None or explicit_row == row_position)
            and (explicit_col is None or explicit_col == 0)
            and col_span == 1
            and row_span >= 2
            and row_position <= status_row
            and row_position + row_span - 1 >= amount_row
            and len(years) == 1
            and int(years[0]) == expected_year
        ):
            continue
        matches.append((row_position, cell))
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


def _declares_source_table_geometry(ref: Mapping[str, Any]) -> bool:
    geometry = ref.get("geometry_provenance")
    return bool(
        isinstance(geometry, Mapping)
        and str(geometry.get("selection_basis") or "")
        == "source_table_year_plus_twelve_ownership"
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

    rows = _rows(table)
    if not (0 <= status_row < len(rows) and 0 <= amount_row < len(rows)):
        return None
    status_cells = _row_cell_map(rows[status_row])
    amount_cells = _row_cell_map(rows[amount_row])
    year_anchor = _year_anchor_proves_pair(
        rows,
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
    explicit_status_row = _integer(_get(status_cell, "row_index"))
    explicit_amount_row = _integer(_get(amount_cell, "row_index"))
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
        and (explicit_status_row is None or explicit_status_row == status_row)
        and (explicit_amount_row is None or explicit_amount_row == amount_row)
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
    if (
        _declares_source_table_geometry(final_ref)
        or _declares_source_table_geometry(final_amount_ref)
    ) and source_binding is None:
        # A source-table-owned continuation must carry one complete, mutually
        # agreeing value-free binding.  Falling back to a nearby header table
        # would silently detach the final cell from its declared ownership.
        return []
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for page in _candidate_pages(context, final_ref):
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
            rows = _rows(table)
            for row_position in range(1, len(rows) - 1):
                header_cells = _row_cell_map(rows[row_position - 1])
                status_cells = _row_cell_map(rows[row_position])
                amount_cells = _row_cell_map(rows[row_position + 1])
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
                explicit_status_row = _integer(_get(status_cell, "row_index"))
                explicit_amount_row = _integer(_get(amount_cell, "row_index"))
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
                    and (explicit_status_row is None or explicit_status_row == row_position)
                    and (explicit_amount_row is None or explicit_amount_row == row_position + 1)
                ):
                    continue
                row_index = explicit_status_row if explicit_status_row is not None else row_position
                candidates.append(
                    (
                        token,
                        _native_source_ref(
                            page=page,
                            table=table,
                            row_index=row_index,
                            column=expected_month,
                            cell=status_cell,
                            logical_bbox=native_box,
                            field_name="status",
                        ),
                        _native_source_ref(
                            page=page,
                            table=table,
                            row_index=(explicit_amount_row if explicit_amount_row is not None else row_position + 1),
                            column=expected_month,
                            cell=amount_cell,
                            logical_bbox=native_amount_box,
                            field_name="overdue_amount",
                        ),
                    )
                )
    return candidates


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
        "agreements": 0,
        "conflicts_withheld": 0,
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

    for record in records:
        if not isinstance(record, dict):
            continue
        status = _single_status(record)
        year_month = _performance_year_month(record)
        amount = _single_amount(record)
        if status is None or year_month is None or amount is None:
            continue
        _field_name, final_token = status
        year, month = year_month
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
    return audit


__all__ = ["apply_candidate_b_native_status_conflict_guard"]
