# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value-free ownership proof for physical year-plus-twelve source tables.

The public resolver consumes detached geometry only.  It does not inspect OCR
text, status glyphs, amounts, account identity, or neighbouring business
values.  Candidate-B consumers may use the resulting physical bands either to
repair continuation geometry or, in a separate stricter layer, to bind raw
source cells for conflict detection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceTableMonthLattice:
    """One uniquely owned physical year/status/amount row pair."""

    table_id: str
    logical_page: int
    source_page: int | None
    expected_year: int
    year_anchor_row_index: int
    header_row_index: int
    status_row_index: int
    amount_row_index: int
    year_bbox: BBox
    month_bboxes: tuple[BBox, ...]
    amount_bboxes: tuple[BBox, ...]
    coordinate_system: str
    geometry_source: str
    provenance: tuple[tuple[str, Any], ...]

    def month_col_bands(self) -> list[dict[str, Any]]:
        """Return field-grid-compatible, detached physical month bands."""

        return [
            {
                "index": month,
                "header": str(month),
                "role": "month",
                "bbox": list(box),
            }
            for month, box in enumerate(self.month_bboxes, start=1)
        ]

    def provenance_dict(self) -> dict[str, Any]:
        return dict(self.provenance)


def _get(owner: Any, key: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
        return owner.get(key, default)
    return getattr(owner, key, default)


def _integer(value: Any) -> int | None:
    """Return only a native integer; geometry ownership never coerces types."""

    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (all(isfinite(item) for item in box) and box[2] > box[0] and box[3] > box[1]):
        return None
    return box


def _matrix_get(matrix: Any, row: int, col: int, default: Any = None) -> Any:
    if not isinstance(matrix, (list, tuple)) or not (0 <= row < len(matrix)):
        return default
    values = matrix[row]
    if not isinstance(values, (list, tuple)) or not (0 <= col < len(values)):
        return default
    return values[col]


def _geometry(table: Any) -> dict[str, Any]:
    metadata = _get(table, "metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    geometry = metadata.get("geometry")
    geometry = dict(geometry) if isinstance(geometry, Mapping) else {}
    for key in (
        "cell_bboxes",
        "source_cell_bboxes",
        "cell_geometry_status",
        "cell_geometry_loss_reason",
        "cell_spans",
        "row_bands",
        "col_bands",
        "vertical_lines",
        "horizontal_lines",
        "coordinate_system",
        "geometry_source",
    ):
        if key not in geometry and key in metadata:
            geometry[key] = metadata[key]
    return geometry


def _typed_cell_matrices(table: Any) -> tuple[list[list[Any]], list[list[Any]], list[dict[str, int]]]:
    rows = _get(table, "rows")
    if not isinstance(rows, (list, tuple)):
        rows = _get(table, "row_models")
    if not isinstance(rows, (list, tuple)):
        return [], [], []
    boxes: list[list[Any]] = []
    statuses: list[list[Any]] = []
    spans: list[dict[str, int]] = []
    for fallback_row, row in enumerate(rows):
        cells = _get(row, "cells")
        if not isinstance(cells, (list, tuple)):
            boxes.append([])
            statuses.append([])
            continue
        width = max(
            ((_integer(_get(cell, "col_index")) or fallback_col) + 1 for fallback_col, cell in enumerate(cells)),
            default=0,
        )
        box_row: list[Any] = [None] * width
        status_row: list[Any] = [None] * width
        for fallback_col, cell in enumerate(cells):
            row_index = _integer(_get(cell, "row_index"))
            col_index = _integer(_get(cell, "col_index"))
            row_index = fallback_row if row_index is None else row_index
            col_index = fallback_col if col_index is None else col_index
            if row_index != fallback_row or not (0 <= col_index < width):
                continue
            box_row[col_index] = list(_get(cell, "bbox") or ()) or None
            status_row[col_index] = str(_get(cell, "geometry_status") or "")
            row_span = _integer(_get(cell, "row_span", 1)) or 1
            col_span = _integer(_get(cell, "col_span", 1)) or 1
            if row_span > 1 or col_span > 1:
                spans.append(
                    {
                        "row": row_index,
                        "col": col_index,
                        "row_span": row_span,
                        "col_span": col_span,
                    }
                )
        boxes.append(box_row)
        statuses.append(status_row)
    return boxes, statuses, spans


def _normalized_spans(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, (list, tuple)):
        return []
    spans: list[dict[str, int]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row = _integer(item.get("row"))
        col = _integer(item.get("col"))
        row_span = _integer(item.get("row_span")) or 1
        col_span = _integer(item.get("col_span")) or 1
        if row is None or col is None or row < 0 or col < 0:
            continue
        spans.append(
            {
                "row": row,
                "col": col,
                "row_span": max(1, row_span),
                "col_span": max(1, col_span),
            }
        )
    return spans


def _page_sequence(parse_result_or_pages: Any) -> list[Any]:
    pages = _get(parse_result_or_pages, "pages")
    if isinstance(pages, (list, tuple)):
        return list(pages)
    if isinstance(parse_result_or_pages, (list, tuple)):
        return list(parse_result_or_pages)
    return []


def detached_source_table_geometry_by_page(
    parse_result_or_pages: Any,
) -> dict[int, list[dict[str, Any]]]:
    """Copy only physical table geometry from sealed/canonical page objects.

    ``cell_bboxes`` is intentionally preferred over ``source_cell_bboxes``:
    Candidate-B repayment refs use registered logical-page coordinates.  The
    source matrix remains attached under its explicit name for diagnostics but
    is never used by the ownership resolver.
    """

    result: dict[int, list[dict[str, Any]]] = {}
    for fallback_page, page in enumerate(_page_sequence(parse_result_or_pages), start=1):
        logical_page = _integer(_get(page, "page_number")) or fallback_page
        source_page = _integer(_get(page, "source_page_number") or _get(page, "source_page"))
        tables = _get(page, "tables")
        if not isinstance(tables, (list, tuple)):
            continue
        for table in tables:
            geometry = _geometry(table)
            typed_boxes, typed_statuses, typed_spans = _typed_cell_matrices(table)
            cell_bboxes = geometry.get("cell_bboxes")
            if not isinstance(cell_bboxes, (list, tuple)) or not cell_bboxes:
                cell_bboxes = typed_boxes
            cell_statuses = geometry.get("cell_geometry_status")
            if not isinstance(cell_statuses, (list, tuple)) or not cell_statuses:
                cell_statuses = typed_statuses
            spans = _normalized_spans(geometry.get("cell_spans")) or typed_spans
            if not cell_bboxes or not cell_statuses:
                continue
            table_id = str(_get(table, "table_id") or _get(table, "id") or "").strip()
            result.setdefault(logical_page, []).append(
                {
                    "table_id": table_id,
                    "logical_page": logical_page,
                    "source_page": source_page,
                    "bbox": deepcopy(_get(table, "bbox")),
                    "extraction_layer": str(_get(table, "extraction_layer") or ""),
                    "geometry_source": str(geometry.get("geometry_source") or _get(table, "extraction_layer") or ""),
                    "coordinate_system": str(geometry.get("coordinate_system") or "pdf_points_top_left"),
                    "cell_bboxes": deepcopy(list(cell_bboxes)),
                    "source_cell_bboxes": deepcopy(geometry.get("source_cell_bboxes") or []),
                    "cell_geometry_status": deepcopy(list(cell_statuses)),
                    "cell_spans": deepcopy(spans),
                    "row_bands": deepcopy(geometry.get("row_bands") or []),
                    "col_bands": deepcopy(geometry.get("col_bands") or []),
                    "vertical_lines": deepcopy(geometry.get("vertical_lines") or []),
                    "horizontal_lines": deepcopy(geometry.get("horizontal_lines") or []),
                }
            )
    return result


def _span_at(table: Mapping[str, Any], row: int, col: int) -> tuple[int, int]:
    matches = [
        item
        for item in table.get("cell_spans") or ()
        if isinstance(item, Mapping) and _integer(item.get("row")) == row and _integer(item.get("col")) == col
    ]
    if len(matches) > 1:
        return 0, 0
    if not matches:
        return 1, 1
    return (
        _integer(matches[0].get("row_span")) or 1,
        _integer(matches[0].get("col_span")) or 1,
    )


def _cell_exact(table: Mapping[str, Any], row: int, col: int) -> bool:
    return (
        _bbox(_matrix_get(table.get("cell_bboxes"), row, col)) is not None
        and str(_matrix_get(table.get("cell_geometry_status"), row, col) or "") == "exact"
        and _span_at(table, row, col) == (1, 1)
    )


def _anchor_span(table: Mapping[str, Any], row: int, col: int) -> tuple[int, int]:
    """Return an explicit anchor span; ambiguity is represented as ``(0, 0)``."""

    return _span_at(table, row, col)


def _cell_bbox(table: Mapping[str, Any], row: int, col: int) -> BBox | None:
    return _bbox(_matrix_get(table.get("cell_bboxes"), row, col))


def _table_bbox(table: Mapping[str, Any]) -> BBox | None:
    explicit = _bbox(table.get("bbox"))
    if explicit is not None:
        return explicit
    boxes = [
        box
        for row in table.get("cell_bboxes") or ()
        if isinstance(row, (list, tuple))
        for value in row
        if (box := _bbox(value)) is not None
    ]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _band_map(
    value: Any,
    *,
    axis: str,
    table_bbox: BBox,
) -> dict[int, BBox] | None:
    if not isinstance(value, (list, tuple)):
        return None
    bands: dict[int, BBox] = {}
    for fallback_index, item in enumerate(value):
        if not isinstance(item, Mapping):
            return None
        index = _integer(item.get("index"))
        index = fallback_index if index is None else index
        box = _bbox(item.get("bbox"))
        if box is None:
            try:
                if axis == "row":
                    box = (
                        table_bbox[0],
                        float(item["y0"]),
                        table_bbox[2],
                        float(item["y1"]),
                    )
                elif axis == "col":
                    box = (
                        float(item["x0"]),
                        table_bbox[1],
                        float(item["x1"]),
                        table_bbox[3],
                    )
            except (KeyError, TypeError, ValueError):
                box = None
            box = _bbox(box)
        if box is None or index in bands:
            return None
        bands[index] = box
    return bands


def _vertical_edges(table: Mapping[str, Any]) -> tuple[float, ...] | None:
    raw_lines = table.get("vertical_lines")
    if not isinstance(raw_lines, (list, tuple)):
        return None
    lines: list[float] = []
    for value in raw_lines:
        if isinstance(value, Mapping):
            value = value.get("x") or value.get("position")
        try:
            coordinate = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(coordinate):
            return None
        lines.append(coordinate)
    ordered = sorted(lines)
    if len(ordered) != 14 or any(right - left <= 1.0 for left, right in zip(ordered, ordered[1:])):
        return None
    return tuple(ordered)


def _raw_column_geometry(
    table: Mapping[str, Any],
) -> tuple[tuple[BBox, ...], tuple[float, ...]] | None:
    """Return the scanner's primitive column bands without assigning roles."""

    if str(table.get("geometry_source") or "") != "scanned_image_line_grid":
        return None
    table_box = _table_bbox(table)
    raw_lines = table.get("vertical_lines")
    if table_box is None or not isinstance(raw_lines, (list, tuple)):
        return None
    lines: list[float] = []
    for value in raw_lines:
        if isinstance(value, Mapping):
            value = value.get("x") or value.get("position")
        try:
            coordinate = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(coordinate):
            return None
        lines.append(coordinate)
    ordered = tuple(sorted(lines))
    if (
        not 14 <= len(ordered) <= 19
        or any(right - left <= 1.0 for left, right in zip(ordered, ordered[1:]))
    ):
        return None
    col_bands = _band_map(
        table.get("col_bands"),
        axis="col",
        table_bbox=table_box,
    )
    column_count = len(ordered) - 1
    if col_bands is None or set(col_bands) != set(range(column_count)):
        return None
    columns = tuple(col_bands[index] for index in range(column_count))
    if not all(
        abs(columns[index][0] - ordered[index]) <= 2.0
        and abs(columns[index][2] - ordered[index + 1]) <= 2.0
        for index in range(column_count)
    ):
        return None
    return columns, ordered


def _contiguous_column_partitions(
    column_count: int,
    *,
    group_count: int = 13,
) -> list[tuple[tuple[int, ...], ...]]:
    """Enumerate small contiguous primitive-to-effective partitions."""

    partitions: list[tuple[tuple[int, ...], ...]] = []

    def visit(
        start: int,
        groups: list[tuple[int, ...]],
    ) -> None:
        remaining_columns = column_count - start
        remaining_groups = group_count - len(groups)
        if remaining_groups == 0:
            if remaining_columns == 0:
                partitions.append(tuple(groups))
            return
        if remaining_columns < remaining_groups or remaining_columns > remaining_groups * 2:
            return
        for width in (1, 2):
            if start + width <= column_count:
                visit(
                    start + width,
                    [*groups, tuple(range(start, start + width))],
                )

    visit(0, [])
    return partitions


def _uniform_effective_widths(
    columns: Sequence[BBox],
    groups: Sequence[Sequence[int]],
) -> bool:
    widths = [
        columns[group[-1]][2] - columns[group[0]][0]
        for group in groups
        if group
    ]
    if len(widths) != 13:
        return False
    middle = median(widths)
    return bool(
        middle > 0.0
        and all(abs(width - middle) / middle <= 0.15 for width in widths)
    )


def _exact_pair_span_rows(
    table: Mapping[str, Any],
    *,
    first_col: int,
    columns: Sequence[BBox],
) -> set[int]:
    table_box = _table_bbox(table)
    row_bands = (
        _band_map(
            table.get("row_bands"),
            axis="row",
            table_bbox=table_box,
        )
        if table_box is not None
        else None
    )
    if row_bands is None or first_col + 1 >= len(columns):
        return set()
    rows: set[int] = set()
    for item in table.get("cell_spans") or ():
        if not isinstance(item, Mapping):
            continue
        row = _integer(item.get("row"))
        col = _integer(item.get("col"))
        row_span = _integer(item.get("row_span")) or 1
        col_span = _integer(item.get("col_span")) or 1
        if (
            row is None
            or col != first_col
            or row_span != 1
            or col_span != 2
            or _span_at(table, row, col) != (1, 2)
            or row not in row_bands
            or str(
                _matrix_get(
                    table.get("cell_geometry_status"),
                    row,
                    col,
                )
                or ""
            )
            != "exact"
            or _cell_bbox(table, row, col) is None
        ):
            continue
        observed = _cell_bbox(table, row, col)
        expected = (
            columns[first_col][0],
            row_bands[row][1],
            columns[first_col + 1][2],
            row_bands[row][3],
        )
        if observed is None or not _same_cell(observed, expected):
            continue
        if any(
            str(
                _matrix_get(
                    table.get("cell_geometry_status"),
                    row,
                    covered_col,
                )
                or ""
            )
            == "exact"
            and _cell_bbox(table, row, covered_col) is not None
            for covered_col in range(first_col + 1, first_col + col_span)
        ):
            continue
        rows.add(row)
    return rows


def _partition_from_repeated_exact_spans(
    table: Mapping[str, Any],
    columns: Sequence[BBox],
) -> tuple[tuple[int, ...], ...] | None:
    valid: list[tuple[tuple[int, ...], ...]] = []
    for groups in _contiguous_column_partitions(len(columns)):
        if not _uniform_effective_widths(columns, groups):
            continue
        pair_rows = [
            _exact_pair_span_rows(
                table,
                first_col=group[0],
                columns=columns,
            )
            for group in groups
            if len(group) == 2
        ]
        if not pair_rows:
            continue
        common_rows = set.intersection(*pair_rows)
        if len(common_rows) < 3 or not any(
            {start, start + 1, start + 2}.issubset(common_rows)
            for start in common_rows
        ):
            continue
        valid.append(groups)
    return valid[0] if len(valid) == 1 else None


def _terminal_sliver_partition(
    table: Mapping[str, Any],
    columns: Sequence[BBox],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]] | None:
    if len(columns) != 14:
        return None
    candidates: list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] = []
    for ignored_col, kept in (
        (0, tuple(range(1, 14))),
        (13, tuple(range(13))),
    ):
        groups = tuple((column,) for column in kept)
        if not _uniform_effective_widths(columns, groups):
            continue
        kept_widths = [columns[column][2] - columns[column][0] for column in kept]
        ignored_width = columns[ignored_col][2] - columns[ignored_col][0]
        if ignored_width > median(kept_widths) * 0.20:
            continue
        kept_left = columns[kept[0]][0]
        kept_right = columns[kept[-1]][2]
        crossing_span = False
        for item in table.get("cell_spans") or ():
            if not isinstance(item, Mapping):
                continue
            col = _integer(item.get("col"))
            col_span = _integer(item.get("col_span")) or 1
            if col is None:
                continue
            owned = set(range(col, col + col_span))
            if ignored_col in owned and owned.intersection(kept):
                crossing_span = True
                break
        if crossing_span:
            continue
        disjoint = True
        matrix = table.get("cell_bboxes")
        if not isinstance(matrix, (list, tuple)):
            continue
        for row in matrix:
            if not isinstance(row, (list, tuple)) or ignored_col >= len(row):
                continue
            box = _bbox(row[ignored_col])
            if box is None:
                continue
            if ignored_col == 0:
                disjoint = disjoint and box[2] <= kept_left + 1.0
            else:
                disjoint = disjoint and box[0] >= kept_right - 1.0
        if disjoint:
            candidates.append((groups, (ignored_col,)))
    return candidates[0] if len(candidates) == 1 else None


def _remapped_span(
    span: Mapping[str, Any],
    groups: Sequence[Sequence[int]],
) -> dict[str, int] | None:
    row = _integer(span.get("row"))
    col = _integer(span.get("col"))
    row_span = _integer(span.get("row_span")) or 1
    col_span = _integer(span.get("col_span")) or 1
    if row is None or col is None:
        return None
    raw_columns = set(range(col, col + col_span))
    matching = [
        index
        for index, group in enumerate(groups)
        if raw_columns.intersection(group)
    ]
    if not matching:
        return None
    if matching != list(range(matching[0], matching[-1] + 1)):
        return None
    owned_groups = [groups[index] for index in matching]
    if raw_columns != {
        raw_col
        for group in owned_groups
        for raw_col in group
    }:
        return None
    normalized = {
        "row": row,
        "col": matching[0],
        "row_span": row_span,
        "col_span": len(matching),
    }
    return normalized if row_span > 1 or len(matching) > 1 else None


def _effective_column_table(
    table: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Collapse only uniquely repeated physical subdivisions to 13 cells."""

    raw_geometry = _raw_column_geometry(table)
    if raw_geometry is None:
        return None
    columns, _raw_edges = raw_geometry
    if len(columns) == 13:
        return table
    groups = _partition_from_repeated_exact_spans(table, columns)
    ignored: tuple[int, ...] = ()
    mode = "repeated_exact_span_partition"
    if groups is None:
        sliver = _terminal_sliver_partition(table, columns)
        if sliver is None:
            return None
        groups, ignored = sliver
        mode = "disjoint_terminal_sliver"

    exact_pair_rows = {
        group[0]: _exact_pair_span_rows(
            table,
            first_col=group[0],
            columns=columns,
        )
        for group in groups
        if len(group) == 2
    }

    matrix = table.get("cell_bboxes")
    statuses = table.get("cell_geometry_status")
    if not isinstance(matrix, (list, tuple)) or not isinstance(statuses, (list, tuple)):
        return None
    normalized_boxes: list[list[Any]] = []
    normalized_statuses: list[list[Any]] = []
    for row_index, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            return None
        box_row: list[Any] = []
        status_row: list[Any] = []
        for group in groups:
            first_col = group[0]
            box = _matrix_get(matrix, row_index, first_col)
            status = _matrix_get(statuses, row_index, first_col, "")
            if len(group) == 2:
                if row_index not in exact_pair_rows.get(first_col, set()):
                    box = None
                    status = "missing"
            box_row.append(deepcopy(box))
            status_row.append(deepcopy(status))
        normalized_boxes.append(box_row)
        normalized_statuses.append(status_row)

    normalized_spans = [
        remapped
        for item in table.get("cell_spans") or ()
        if isinstance(item, Mapping)
        and (remapped := _remapped_span(item, groups)) is not None
    ]
    effective_bands = [
        {
            "index": index,
            "x0": columns[group[0]][0],
            "x1": columns[group[-1]][2],
        }
        for index, group in enumerate(groups)
    ]
    effective_edges = [effective_bands[0]["x0"]] + [
        band["x1"] for band in effective_bands
    ]
    table_box = _table_bbox(table)
    if table_box is None:
        return None
    normalized = dict(table)
    normalized.update(
        {
            "bbox": [
                effective_edges[0],
                table_box[1],
                effective_edges[-1],
                table_box[3],
            ],
            "cell_bboxes": normalized_boxes,
            "cell_geometry_status": normalized_statuses,
            "cell_spans": normalized_spans,
            "col_bands": effective_bands,
            "vertical_lines": effective_edges,
            "effective_column_canonicalization": {
                "mode": mode,
                "raw_column_count": len(columns),
                "effective_column_count": len(groups),
                "collapsed_group_count": sum(len(group) > 1 for group in groups),
                "ignored_terminal_column_count": len(ignored),
            },
        }
    )
    return normalized


def _owned_columns(
    table: Mapping[str, Any],
) -> tuple[tuple[BBox, ...], tuple[float, ...]] | None:
    if str(table.get("geometry_source") or "") != "scanned_image_line_grid":
        return None
    table_box = _table_bbox(table)
    if table_box is None:
        return None
    edges = _vertical_edges(table)
    col_bands = _band_map(
        table.get("col_bands"),
        axis="col",
        table_bbox=table_box,
    )
    if edges is None or col_bands is None or set(col_bands) != set(range(13)):
        return None
    columns = tuple(col_bands[index] for index in range(13))
    if not all(
        abs(columns[index][0] - edges[index]) <= 2.0 and abs(columns[index][2] - edges[index + 1]) <= 2.0
        for index in range(13)
    ):
        return None
    return columns, edges


def _rows_follow_horizontal_rules(
    table: Mapping[str, Any],
    row_bands: Mapping[int, BBox],
) -> bool:
    """Require one global physical rule at every detached row-band edge."""

    raw_lines = table.get("horizontal_lines")
    if (
        not isinstance(raw_lines, (list, tuple))
        or set(row_bands) != set(range(len(row_bands)))
    ):
        return False
    lines: list[float] = []
    for value in raw_lines:
        if isinstance(value, Mapping):
            value = value.get("y") or value.get("position")
        try:
            coordinate = float(value)
        except (TypeError, ValueError):
            return False
        if not isfinite(coordinate):
            return False
        lines.append(coordinate)
    ordered = sorted(lines)
    if (
        len(ordered) != len(row_bands) + 1
        or any(right - left <= 1.0 for left, right in zip(ordered, ordered[1:]))
    ):
        return False
    return all(
        abs(row_bands[index][1] - ordered[index]) <= 2.0
        and abs(row_bands[index][3] - ordered[index + 1]) <= 2.0
        for index in range(len(row_bands))
    )


def _covering_spans(
    table: Mapping[str, Any],
    *,
    row: int,
    col: int,
) -> list[dict[str, int]]:
    return [
        {
            "row": span_row,
            "col": span_col,
            "row_span": row_span,
            "col_span": col_span,
        }
        for item in table.get("cell_spans") or ()
        if isinstance(item, Mapping)
        and (span_row := _integer(item.get("row"))) is not None
        and (span_col := _integer(item.get("col"))) is not None
        and (row_span := _integer(item.get("row_span")) or 1) >= 1
        and (col_span := _integer(item.get("col_span")) or 1) >= 1
        and span_row <= row < span_row + row_span
        and span_col <= col < span_col + col_span
    ]


def _has_competing_exact_cell(
    table: Mapping[str, Any],
    *,
    row: int,
    col: int,
    target: BBox,
    allowed_span_anchor: tuple[int, int] | None = None,
) -> bool:
    """Return whether another exact matrix cell materially owns ``target``."""

    matrix = table.get("cell_bboxes")
    statuses = table.get("cell_geometry_status")
    if not isinstance(matrix, (list, tuple)):
        return True
    for other_row, cells in enumerate(matrix):
        if not isinstance(cells, (list, tuple)):
            continue
        for other_col, value in enumerate(cells):
            if (other_row, other_col) == (row, col) or (
                allowed_span_anchor is not None
                and (other_row, other_col) == allowed_span_anchor
            ):
                continue
            if str(_matrix_get(statuses, other_row, other_col) or "") != "exact":
                continue
            other = _bbox(value)
            if other is None:
                continue
            target_coverage, _other_coverage, _iou = _intersection_ratio(
                target,
                other,
            )
            if target_coverage >= 0.50:
                return True
    return False


def _cell_supported_by_physical_rules(
    table: Mapping[str, Any],
    *,
    row: int,
    col: int,
    row_bands: Mapping[int, BBox],
    columns: Sequence[BBox],
) -> tuple[bool, bool]:
    """Validate one cell, allowing only explicit merged-rule occlusion.

    The second return value reports whether an exact 1x1 source cell was
    available.  A scanner may mark a watermark-obscured cell as covered by one
    explicit span even though the global horizontal and vertical rules remain
    exact; in that case the rule intersection still supplies value-free cell
    ownership.
    """

    derived = _derived_cell(row_bands[row], columns[col])
    coverings = _covering_spans(table, row=row, col=col)
    if _cell_exact(table, row, col):
        # An exact 1x1 cell cannot simultaneously be owned by a second merged
        # representation.  Checking this before accepting the exact cell keeps
        # a wide row span from hiding behind the intact 1x1 matrix.
        if coverings and coverings != [
            {
                "row": row,
                "col": col,
                "row_span": 1,
                "col_span": 1,
            }
        ]:
            return False, False
        observed = _cell_bbox(table, row, col)
        return bool(
            observed is not None
            and _same_cell(observed, derived)
            and not _has_competing_exact_cell(
                table,
                row=row,
                col=col,
                target=derived,
            )
        ), True
    if len(coverings) != 1:
        return False, False
    covering = coverings[0]
    if covering["col_span"] != 1 or covering["row_span"] > 3:
        # A wide merged row erases physical month ownership; it cannot be
        # reconstructed merely because table-level rules exist elsewhere.
        return False, False
    anchor_row = covering["row"]
    anchor_col = covering["col"]
    final_row = anchor_row + covering["row_span"] - 1
    final_col = anchor_col + covering["col_span"] - 1
    if (
        anchor_row not in row_bands
        or final_row not in row_bands
        or not (0 <= anchor_col <= final_col < len(columns))
        or str(
            _matrix_get(
                table.get("cell_geometry_status"), anchor_row, anchor_col
            )
            or ""
        )
        != "exact"
    ):
        return False, False
    anchor_box = _cell_bbox(table, anchor_row, anchor_col)
    expected_span = (
        columns[anchor_col][0],
        row_bands[anchor_row][1],
        columns[final_col][2],
        row_bands[final_row][3],
    )
    return bool(
        anchor_box is not None
        and _same_cell(anchor_box, expected_span)
        and not _has_competing_exact_cell(
            table,
            row=row,
            col=col,
            target=derived,
            allowed_span_anchor=(anchor_row, anchor_col),
        )
    ), False


def _derived_cell(row_band: BBox, col_band: BBox) -> BBox:
    return col_band[0], row_band[1], col_band[2], row_band[3]


def _target_matches_year_row_pair(target: BBox, row_pair_year_box: BBox) -> bool:
    """Bind a pair-height year target to one exact physical row pair."""

    target_coverage, pair_coverage, _iou = _intersection_ratio(
        target,
        row_pair_year_box,
    )
    center_x = (target[0] + target[2]) / 2.0
    center_y = (target[1] + target[3]) / 2.0
    return (
        row_pair_year_box[0] - 1.0 <= center_x <= row_pair_year_box[2] + 1.0
        and row_pair_year_box[1] - 2.0 <= center_y <= row_pair_year_box[3] + 2.0
        and target_coverage >= 0.70
        and pair_coverage >= 0.35
    )


def _year_anchor_for_pair(
    table: Mapping[str, Any],
    *,
    status_row: int,
    amount_row: int,
    year_bbox: BBox,
    row_bands: Mapping[int, BBox],
    year_column: BBox,
) -> tuple[int, BBox, int, str] | None:
    """Resolve the physical year cell that owns one status/amount row pair.

    Most scanner tables preserve the printed two-row year span.  A smaller
    class of otherwise exact line grids splits that span and places the year
    glyph in either the status or amount row.  The detached target bbox can
    disambiguate that shape without consulting the source cell's text.
    """

    matrix = table.get("cell_bboxes")
    if not isinstance(matrix, (list, tuple)):
        return None
    row_pair_year_box = (
        year_column[0],
        row_bands[status_row][1],
        year_column[2],
        row_bands[amount_row][3],
    )
    for candidate_row in range(0, min(amount_row + 1, len(matrix))):
        if (
            str(
                _matrix_get(
                    table.get("cell_geometry_status"),
                    candidate_row,
                    0,
                )
                or ""
            )
            != "exact"
        ):
            continue
        candidate_box = _cell_bbox(table, candidate_row, 0)
        row_span, col_span = _anchor_span(table, candidate_row, 0)
        final_row = candidate_row + row_span - 1
        declared_box = (
            (
                year_column[0],
                row_bands[candidate_row][1],
                year_column[2],
                row_bands[final_row][3],
            )
            if candidate_row in row_bands and final_row in row_bands
            else None
        )
        geometry_consistent = bool(
            candidate_box is not None
            and col_span == 1
            and declared_box is not None
            and _same_cell(candidate_box, declared_box)
        )
        if geometry_consistent or candidate_box is None:
            continue
        pair_coverage, cell_coverage, _iou = _intersection_ratio(
            row_pair_year_box,
            candidate_box,
        )
        if pair_coverage >= 0.40 or cell_coverage >= 0.70:
            return None

    matches: list[tuple[int, BBox, int, str]] = []
    for anchor_row in range(0, min(amount_row + 1, len(matrix))):
        row_span, col_span = _anchor_span(table, anchor_row, 0)
        box = _cell_bbox(table, anchor_row, 0)
        final_row = anchor_row + row_span - 1
        declared_span_box = (
            (
                year_column[0],
                row_bands[anchor_row][1],
                year_column[2],
                row_bands[final_row][3],
            )
            if anchor_row in row_bands and final_row in row_bands
            else None
        )
        if not (
            box is not None
            and str(_matrix_get(table.get("cell_geometry_status"), anchor_row, 0) or "") == "exact"
            and col_span == 1
            and declared_span_box is not None
            and _same_cell(box, declared_span_box)
            and _target_matches_year_cell(year_bbox, box)
        ):
            continue
        if (
            row_span >= 2
            and anchor_row <= status_row
            and anchor_row + row_span - 1 >= amount_row
        ):
            matches.append((anchor_row, box, row_span, "spanning_year_cell"))
        elif row_span == 1 and anchor_row in {status_row, amount_row}:
            matches.append((anchor_row, box, row_span, "target_bound_singleton_year_cell"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None

    # Some exact scanner grids preserve the year column and both horizontal
    # row rules but split the printed two-row year cell into singletons.  A
    # pair-height target then covers only half of either singleton and fails
    # the ordinary 70% target-to-cell check.  Bind the exact row-pair union
    # only when at least one geometrically consistent singleton corroborates
    # column zero; the caller still has to prove the global row/column rules
    # and one unique status/amount lattice before this candidate is returned.
    singleton_support = 0
    for anchor_row in (status_row, amount_row):
        row_span, col_span = _anchor_span(table, anchor_row, 0)
        box = _cell_bbox(table, anchor_row, 0)
        declared_box = (
            (
                year_column[0],
                row_bands[anchor_row][1],
                year_column[2],
                row_bands[anchor_row][3],
            )
            if anchor_row in row_bands
            else None
        )
        if (
            row_span == 1
            and col_span == 1
            and box is not None
            and declared_box is not None
            and str(
                _matrix_get(
                    table.get("cell_geometry_status"),
                    anchor_row,
                    0,
                )
                or ""
            )
            == "exact"
            and _same_cell(box, declared_box)
        ):
            singleton_support += 1
    if singleton_support and _target_matches_year_row_pair(
        year_bbox,
        row_pair_year_box,
    ):
        return (
            status_row,
            row_pair_year_box,
            2,
            "row_pair_year_column",
        )
    return None


def _same_x(first: BBox, second: BBox, *, tolerance: float = 1.5) -> bool:
    return abs(first[0] - second[0]) <= tolerance and abs(first[2] - second[2]) <= tolerance


def _vertical_neighbors(upper: BBox, lower: BBox, *, tolerance: float = 2.0) -> bool:
    return abs(upper[3] - lower[1]) <= tolerance and lower[3] > upper[3]


def _intersection_ratio(first: BBox, second: BBox) -> tuple[float, float, float]:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0, 0.0, 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / first_area, intersection / second_area, intersection / union


def _same_cell(first: BBox, second: BBox) -> bool:
    first_coverage, second_coverage, intersection_over_union = _intersection_ratio(first, second)
    x_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    y_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return (
        first_coverage >= 0.80
        and second_coverage >= 0.80
        and intersection_over_union >= 0.70
        and x_overlap / min(first[2] - first[0], second[2] - second[0]) >= 0.92
        and y_overlap / min(first[3] - first[1], second[3] - second[1]) >= 0.80
    )


def _target_owns_year_column(target: BBox, year_cell: BBox, row_pair: BBox) -> bool:
    center_x = (target[0] + target[2]) / 2.0
    center_y = (target[1] + target[3]) / 2.0
    return year_cell[0] - 1.0 <= center_x <= year_cell[2] + 1.0 and row_pair[1] - 2.0 <= center_y <= row_pair[3] + 2.0


def _target_matches_year_cell(target: BBox, year_cell: BBox) -> bool:
    """Bind a detached year glyph bbox to exactly one physical column-zero cell."""

    center_x = (target[0] + target[2]) / 2.0
    center_y = (target[1] + target[3]) / 2.0
    y_overlap = max(0.0, min(target[3], year_cell[3]) - max(target[1], year_cell[1]))
    return (
        year_cell[0] - 1.0 <= center_x <= year_cell[2] + 1.0
        and year_cell[1] - 2.0 <= center_y <= year_cell[3] + 2.0
        and y_overlap / max(target[3] - target[1], 1e-6) >= 0.70
    )


def _status_target_matches_row(target: BBox, months: Sequence[BBox]) -> bool:
    row_box = (months[0][0], min(box[1] for box in months), months[-1][2], max(box[3] for box in months))
    row_height = row_box[3] - row_box[1]
    target_width = target[2] - target[0]
    median_month_width = median(box[2] - box[0] for box in months)
    if target_width >= median_month_width * 2.0:
        x_overlap = max(0.0, min(target[2], row_box[2]) - max(target[0], row_box[0]))
        y_overlap = max(0.0, min(target[3], row_box[3]) - max(target[1], row_box[1]))
        target_center_x = (target[0] + target[2]) / 2.0
        return (
            x_overlap / min(target_width, row_box[2] - row_box[0]) >= 0.75
            and row_box[0] <= target_center_x <= row_box[2]
            and y_overlap / min(target[3] - target[1], row_height) >= 0.75
        )
    return sum(_same_cell(target, month_box) for month_box in months) == 1


def _candidate_lattices(
    table: Mapping[str, Any],
    *,
    logical_page: int,
    expected_year: int,
    active_months: frozenset[int],
    year_bbox: BBox,
    status_bbox: BBox,
) -> list[SourceTableMonthLattice]:
    effective_table = _effective_column_table(table)
    if effective_table is None:
        return []
    table = effective_table
    if (
        _integer(table.get("logical_page")) != logical_page
        or str(table.get("coordinate_system") or "") != "pdf_points_top_left"
    ):
        return []
    owned_columns = _owned_columns(table)
    table_box = _table_bbox(table)
    row_bands = (
        _band_map(
            table.get("row_bands"),
            axis="row",
            table_bbox=table_box,
        )
        if table_box is not None
        else None
    )
    if (
        owned_columns is None
        or row_bands is None
        or not _rows_follow_horizontal_rules(table, row_bands)
    ):
        return []
    columns, _edges = owned_columns
    matrix = table.get("cell_bboxes")
    if not isinstance(matrix, (list, tuple)):
        return []
    candidates: list[SourceTableMonthLattice] = []
    for status_row in range(0, len(matrix) - 1):
        header_row = status_row - 1
        amount_row = status_row + 1
        if status_row not in row_bands or amount_row not in row_bands:
            continue
        year_anchor = _year_anchor_for_pair(
            table,
            status_row=status_row,
            amount_row=amount_row,
            year_bbox=year_bbox,
            row_bands=row_bands,
            year_column=columns[0],
        )
        if year_anchor is None:
            continue
        year_anchor_row, year_cell, year_row_span, year_anchor_mode = year_anchor
        status_cells = tuple(_derived_cell(row_bands[status_row], columns[month]) for month in range(1, 13))
        amount_cells = tuple(_derived_cell(row_bands[amount_row], columns[month]) for month in range(1, 13))
        if not _vertical_neighbors(row_bands[status_row], row_bands[amount_row]):
            continue
        active_cell_support = [
            _cell_supported_by_physical_rules(
                table,
                row=row,
                col=month,
                row_bands=row_bands,
                columns=columns,
            )
            for row in (status_row, amount_row)
            for month in active_months
        ]
        if not all(supported for supported, _exact in active_cell_support):
            continue
        exact_active_cell_count = sum(
            exact for _supported, exact in active_cell_support
        )
        covered_active_cell_count = len(active_cell_support) - exact_active_cell_count
        widths = [box[2] - box[0] for box in status_cells]
        median_width = median(widths)
        if median_width <= 0.0 or any(abs(width - median_width) / median_width > 0.15 for width in widths):
            continue
        if not (
            abs(year_cell[0] - columns[0][0]) <= 2.0
            and abs(year_cell[2] - columns[0][2]) <= 2.0
            and abs(year_cell[2] - status_cells[0][0]) <= 2.0
            and 0.70 * median_width <= year_cell[2] - year_cell[0] <= 1.35 * median_width
        ):
            continue
        exact_header = bool(
            header_row >= 0
            and header_row in row_bands
            and all(_cell_exact(table, header_row, month) for month in range(1, 13))
        )
        if exact_header:
            header_boxes = tuple(_cell_bbox(table, header_row, month) for month in range(1, 13))
            exact_header = bool(
                all(header_boxes)
                and all(
                    _same_cell(
                        header_boxes[index],
                        _derived_cell(row_bands[header_row], columns[index + 1]),
                    )
                    for index in range(12)
                )
                and _vertical_neighbors(row_bands[header_row], row_bands[status_row])
            )
        canonicalization = table.get("effective_column_canonicalization")
        row_pair = (
            year_cell[0],
            row_bands[status_row][1],
            status_cells[-1][2],
            row_bands[amount_row][3],
        )
        if not (
            _target_owns_year_column(year_bbox, year_cell, row_pair)
            and _status_target_matches_row(status_bbox, status_cells)
        ):
            continue
        provenance = {
            "selection_basis": "source_table_year_plus_twelve_ownership",
            "source": "source_table_geometry",
            "value_inputs_used": False,
            "rule_count": 14,
            "column_count": 13,
            "month_column_count": 12,
            "status_amount_row_pair": True,
            "horizontal_rule_count": len(row_bands) + 1,
            "active_cell_geometry_exact": covered_active_cell_count == 0,
            "active_cell_rule_derived_count": covered_active_cell_count,
            "year_row_span": year_row_span,
            "year_anchor_mode": year_anchor_mode,
            "year_anchor_row_index": year_anchor_row,
            "exact_preceding_header": exact_header,
            "active_months": sorted(active_months),
        }
        if isinstance(canonicalization, Mapping):
            provenance.update(
                {
                    "effective_column_canonicalization": str(
                        canonicalization.get("mode") or ""
                    ),
                    "raw_column_count": _integer(
                        canonicalization.get("raw_column_count")
                    ),
                    "effective_column_count": _integer(
                        canonicalization.get("effective_column_count")
                    ),
                    "collapsed_raw_column_group_count": _integer(
                        canonicalization.get("collapsed_group_count")
                    ),
                    "ignored_terminal_column_count": _integer(
                        canonicalization.get("ignored_terminal_column_count")
                    ),
                }
            )
        candidates.append(
            SourceTableMonthLattice(
                table_id=str(table.get("table_id") or ""),
                logical_page=logical_page,
                source_page=_integer(table.get("source_page")),
                expected_year=expected_year,
                year_anchor_row_index=year_anchor_row,
                header_row_index=header_row if exact_header else -1,
                status_row_index=status_row,
                amount_row_index=amount_row,
                year_bbox=year_cell,
                month_bboxes=status_cells,
                amount_bboxes=amount_cells,
                coordinate_system="pdf_points_top_left",
                geometry_source=str(table.get("geometry_source") or ""),
                provenance=tuple(sorted(provenance.items())),
            )
        )
    return candidates


def resolve_unique_source_table_year_plus_twelve_ownership(
    tables: Iterable[Mapping[str, Any]],
    *,
    logical_page: int,
    expected_year: int,
    active_months: Iterable[int],
    year_bbox: Sequence[float],
    status_bbox: Sequence[float],
) -> SourceTableMonthLattice | None:
    """Return one exact value-free lattice, or ``None`` on any ambiguity."""

    normalized_logical_page = _integer(logical_page)
    normalized_expected_year = _integer(expected_year)
    year_box = _bbox(year_bbox)
    status_box = _bbox(status_bbox)
    if isinstance(active_months, (str, bytes, Mapping)):
        return None
    try:
        month_values = iter(active_months)
    except TypeError:
        return None
    normalized_months = [_integer(value) for value in month_values]
    if any(month is None or not 1 <= month <= 12 for month in normalized_months):
        return None
    months = frozenset(
        month for month in normalized_months if month is not None
    )
    if (
        year_box is None
        or status_box is None
        or not months
        or normalized_expected_year is None
        or normalized_logical_page is None
        or not (1900 <= normalized_expected_year <= 2099)
        or normalized_logical_page <= 0
    ):
        return None
    if isinstance(tables, (str, bytes, Mapping)):
        return None
    try:
        table_values = iter(tables)
    except TypeError:
        return None
    candidates = [
        candidate
        for table in table_values
        if isinstance(table, Mapping)
        for candidate in _candidate_lattices(
            table,
            logical_page=normalized_logical_page,
            expected_year=normalized_expected_year,
            active_months=months,
            year_bbox=year_box,
            status_bbox=status_box,
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


__all__ = [
    "SourceTableMonthLattice",
    "detached_source_table_geometry_by_page",
    "resolve_unique_source_table_year_plus_twelve_ownership",
]
