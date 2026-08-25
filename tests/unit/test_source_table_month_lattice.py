from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.source_table_month_lattice import (
    detached_source_table_geometry_by_page,
    resolve_unique_source_table_year_plus_twelve_ownership,
)


def _x_edges() -> list[float]:
    return [45.0, 73.0, *[100.0 + 27.0 * index for index in range(12)]]


def _scanner_table(
    *,
    year_anchor_row: int = 1,
    year_row_span: int = 2,
    status_row: int = 1,
    active_months: range = range(1, 13),
) -> dict[str, object]:
    edges = _x_edges()
    row_edges = [150.0, 163.0, 176.0, 189.0, 202.0]
    row_count = len(row_edges) - 1
    cells: list[list[list[float] | None]] = [[None for _column in range(13)] for _row in range(row_count)]
    statuses = [["missing" for _column in range(13)] for _row in range(row_count)]
    year_bottom = row_edges[year_anchor_row + year_row_span]
    cells[year_anchor_row][0] = [45.0, row_edges[year_anchor_row], 73.0, year_bottom]
    statuses[year_anchor_row][0] = "exact"
    header_row = status_row - 1
    if header_row >= 0:
        for month in range(1, 13):
            cells[header_row][month] = [
                edges[month],
                row_edges[header_row],
                edges[month + 1],
                row_edges[header_row + 1],
            ]
            statuses[header_row][month] = "exact"
    for month in active_months:
        cells[status_row][month] = [
            edges[month],
            row_edges[status_row],
            edges[month + 1],
            row_edges[status_row + 1],
        ]
        statuses[status_row][month] = "exact"
        cells[status_row + 1][month] = [
            edges[month],
            row_edges[status_row + 1],
            edges[month + 1],
            row_edges[status_row + 2],
        ]
        statuses[status_row + 1][month] = "exact"
    return {
        "table_id": "pt_test_0",
        "logical_page": 19,
        "source_page": 10,
        "bbox": [45.0, row_edges[0], edges[-1], row_edges[-1]],
        "geometry_source": "scanned_image_line_grid",
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": cells,
        "cell_geometry_status": statuses,
        "cell_spans": [
            {
                "row": year_anchor_row,
                "col": 0,
                "row_span": year_row_span,
                "col_span": 1,
            }
        ],
        # These are the actual scanned-table metadata shapes: axis bounds,
        # not synthetic full bboxes.
        "row_bands": [
            {"index": index, "y0": row_edges[index], "y1": row_edges[index + 1]} for index in range(row_count)
        ],
        "col_bands": [{"index": index, "x0": edges[index], "x1": edges[index + 1]} for index in range(13)],
        "vertical_lines": edges,
        "horizontal_lines": row_edges,
    }


def _boundary_split_year_table(
    *,
    table_id: str,
    logical_page: int,
    left: float,
    year_right: float,
    right: float,
    row_edges: tuple[float, float, float],
) -> dict[str, object]:
    """Build the exact two-singleton year-column shape seen on Lin p6/p17."""

    pitch = (right - year_right) / 12.0
    column_edges = [left, year_right, *[year_right + pitch * index for index in range(1, 13)]]
    cell_bboxes = [
        [
            [
                column_edges[column],
                row_edges[row],
                column_edges[column + 1],
                row_edges[row + 1],
            ]
            for column in range(13)
        ]
        for row in range(2)
    ]
    return {
        "table_id": table_id,
        "logical_page": logical_page,
        "source_page": logical_page,
        "bbox": [left, row_edges[0], right, row_edges[-1]],
        "geometry_source": "scanned_image_line_grid",
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": cell_bboxes,
        "cell_geometry_status": [["exact"] * 13 for _row in range(2)],
        "cell_spans": [
            {"row": row, "col": 0, "row_span": 1, "col_span": 1}
            for row in range(2)
        ],
        "row_bands": [
            {"index": row, "y0": row_edges[row], "y1": row_edges[row + 1]}
            for row in range(2)
        ],
        "col_bands": [
            {
                "index": column,
                "x0": column_edges[column],
                "x1": column_edges[column + 1],
            }
            for column in range(13)
        ],
        "vertical_lines": column_edges,
        "horizontal_lines": list(row_edges),
    }


def _primitive_split_table(
    split_effective_columns: tuple[int, ...],
    *,
    terminal_sliver: bool = False,
) -> dict[str, object]:
    """Represent one 13-cell lattice with exact scanner subdivisions."""

    table = _scanner_table()
    original_boxes = table["cell_bboxes"]
    original_statuses = table["cell_geometry_status"]
    original_bands = table["col_bands"]
    assert isinstance(original_boxes, list)
    assert isinstance(original_statuses, list)
    assert isinstance(original_bands, list)

    groups: list[tuple[int, ...]] = []
    raw_bands: list[dict[str, float | int]] = []
    raw_index = 0
    for effective_col, band in enumerate(original_bands):
        x0 = float(band["x0"])
        x1 = float(band["x1"])
        if effective_col in split_effective_columns:
            midpoint = x0 + (x1 - x0) * 0.48
            raw_bands.extend(
                (
                    {"index": raw_index, "x0": x0, "x1": midpoint},
                    {"index": raw_index + 1, "x0": midpoint, "x1": x1},
                )
            )
            groups.append((raw_index, raw_index + 1))
            raw_index += 2
        else:
            raw_bands.append({"index": raw_index, "x0": x0, "x1": x1})
            groups.append((raw_index,))
            raw_index += 1

    raw_boxes: list[list[list[float] | None]] = []
    raw_statuses: list[list[str]] = []
    spans: list[dict[str, int]] = []
    original_spans = {
        (int(span["row"]), int(span["col"])): span
        for span in table["cell_spans"]  # type: ignore[union-attr]
    }
    for row_index, (box_row, status_row) in enumerate(
        zip(original_boxes, original_statuses, strict=True)
    ):
        next_boxes: list[list[float] | None] = [None] * len(raw_bands)
        next_statuses = ["missing"] * len(raw_bands)
        for effective_col, group in enumerate(groups):
            next_boxes[group[0]] = deepcopy(box_row[effective_col])
            next_statuses[group[0]] = str(status_row[effective_col])
            original_span = original_spans.get((row_index, effective_col))
            row_span = int(original_span.get("row_span", 1)) if original_span else 1
            if len(group) == 2 and str(status_row[effective_col]) == "exact":
                next_statuses[group[1]] = "derived"
                spans.append(
                    {
                        "row": row_index,
                        "col": group[0],
                        "row_span": row_span,
                        "col_span": 2,
                    }
                )
            elif original_span is not None:
                spans.append(
                    {
                        "row": row_index,
                        "col": group[0],
                        "row_span": row_span,
                        "col_span": 1,
                    }
                )
        raw_boxes.append(next_boxes)
        raw_statuses.append(next_statuses)

    if terminal_sliver:
        sliver_left = float(raw_bands[-1]["x1"])
        raw_bands.append(
            {
                "index": len(raw_bands),
                "x0": sliver_left,
                "x1": sliver_left + 4.0,
            }
        )
        for row_index, row in enumerate(raw_boxes):
            row.append(
                [
                    sliver_left,
                    150.0 + 13.0 * row_index,
                    sliver_left + 4.0,
                    163.0 + 13.0 * row_index,
                ]
            )
            raw_statuses[row_index].append("exact")

    table["cell_bboxes"] = raw_boxes
    table["cell_geometry_status"] = raw_statuses
    table["cell_spans"] = spans
    table["col_bands"] = raw_bands
    table["vertical_lines"] = [raw_bands[0]["x0"], *[band["x1"] for band in raw_bands]]
    table["bbox"] = [45.0, 150.0, raw_bands[-1]["x1"], 202.0]
    return table


def test_resolver_accepts_exact_scanner_shape_and_returns_consumer_bands() -> None:
    table = _scanner_table()

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=[50.0, 164.0, 69.0, 187.0],
        status_bbox=[262.03, 162.0, 289.03, 177.0],
    )

    assert lattice is not None
    assert lattice.year_anchor_row_index == 1
    assert lattice.status_row_index == 1
    assert lattice.amount_row_index == 2
    assert lattice.header_row_index == 0
    bands = lattice.month_col_bands()
    assert [band["index"] for band in bands] == list(range(1, 13))
    assert [int(band["header"]) for band in bands] == list(range(1, 13))
    assert all(band["role"] == "month" for band in bands)
    assert lattice.provenance_dict()["selection_basis"] == ("source_table_year_plus_twelve_ownership")
    assert lattice.provenance_dict()["value_inputs_used"] is False


@pytest.mark.parametrize(
    ("saved_shape", "split_columns", "expected_collapses"),
    (
        ("p5_raw_18", (2, 4, 6, 8, 10), 5),
        ("p6_raw_14", (6,), 1),
        ("p9_grid_0_raw_14", (6,), 1),
        ("p10_grid_1_raw_14", (6,), 1),
    ),
)
def test_resolver_collapses_only_repeated_exact_saved_shape_subdivisions(
    saved_shape: str,
    split_columns: tuple[int, ...],
    expected_collapses: int,
) -> None:
    table = _primitive_split_table(split_columns)
    table["table_id"] = saved_shape

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=[50.0, 164.0, 69.0, 187.0],
        status_bbox=[262.03, 162.0, 289.03, 177.0],
    )

    assert lattice is not None
    provenance = lattice.provenance_dict()
    assert provenance["effective_column_canonicalization"] == (
        "repeated_exact_span_partition"
    )
    assert provenance["raw_column_count"] == 13 + expected_collapses
    assert provenance["effective_column_count"] == 13
    assert provenance["collapsed_raw_column_group_count"] == expected_collapses
    assert provenance["value_inputs_used"] is False


def test_resolver_ignores_only_a_strictly_disjoint_p17_terminal_sliver() -> None:
    table = _primitive_split_table((), terminal_sliver=True)
    table["table_id"] = "pt_17_saved_shape"

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=[50.0, 164.0, 69.0, 187.0],
        status_bbox=[262.03, 162.0, 289.03, 177.0],
    )

    assert lattice is not None
    provenance = lattice.provenance_dict()
    assert provenance["effective_column_canonicalization"] == (
        "disjoint_terminal_sliver"
    )
    assert provenance["ignored_terminal_column_count"] == 1
    assert lattice.month_bboxes[-1][2] == pytest.approx(397.0)


@pytest.mark.parametrize("saved_shape", ("p17", "p18"))
def test_resolver_binds_unique_saved_shape_pair_height_year_target(
    saved_shape: str,
) -> None:
    table = _scanner_table(
        year_anchor_row=2,
        year_row_span=1,
        status_row=1,
    )
    table["table_id"] = f"pt_{saved_shape}_singleton_year"

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=[45.0, 163.0, 73.0, 189.0],
        status_bbox=[262.03, 162.0, 289.03, 177.0],
    )

    assert lattice is not None
    provenance = lattice.provenance_dict()
    assert provenance["year_anchor_mode"] == "row_pair_year_column"
    assert provenance["year_row_span"] == 2
    assert provenance["value_inputs_used"] is False


@pytest.mark.parametrize(
    (
        "table_id",
        "logical_page",
        "expected_year",
        "left",
        "year_right",
        "right",
        "row_edges",
        "year_bbox",
    ),
    (
        (
            "pt_6_0",
            6,
            2022,
            53.5,
            81.0,
            403.5,
            (43.5, 58.5, 73.5),
            [59.5, 53.5, 77.0, 63.0],
        ),
        (
            "pt_17_0",
            17,
            2019,
            34.0,
            62.0,
            389.0,
            (38.5, 53.0, 65.5),
            [40.5, 47.5, 58.5, 58.5],
        ),
    ),
    ids=("lin_p6_2022", "lin_p17_2019"),
)
def test_resolver_binds_real_boundary_straddling_year_singletons(
    table_id: str,
    logical_page: int,
    expected_year: int,
    left: float,
    year_right: float,
    right: float,
    row_edges: tuple[float, float, float],
    year_bbox: list[float],
) -> None:
    table = _boundary_split_year_table(
        table_id=table_id,
        logical_page=logical_page,
        left=left,
        year_right=year_right,
        right=right,
        row_edges=row_edges,
    )

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=logical_page,
        expected_year=expected_year,
        active_months=range(1, 13),
        year_bbox=year_bbox,
        status_bbox=[year_right, row_edges[0], right, row_edges[1]],
    )

    assert lattice is not None
    assert lattice.table_id == table_id
    assert lattice.year_anchor_row_index == 0
    assert lattice.status_row_index == 0
    assert lattice.amount_row_index == 1
    assert lattice.year_bbox == (left, row_edges[0], year_right, row_edges[2])
    provenance = lattice.provenance_dict()
    assert provenance["year_anchor_mode"] == (
        "boundary_straddling_singleton_year_cells"
    )
    assert provenance["year_row_span"] == 2
    assert provenance["active_cell_geometry_exact"] is True
    assert provenance["value_inputs_used"] is False


@pytest.mark.parametrize(
    "failure_mode",
    (
        "outside_row_pair",
        "month_column_target",
        "tiny_boundary_speck",
        "status_non_exact",
        "amount_missing",
        "amount_inset",
        "status_spanned",
        "duplicate_candidate",
    ),
)
def test_boundary_straddling_year_singleton_fallback_fails_closed(
    failure_mode: str,
) -> None:
    table = _boundary_split_year_table(
        table_id="pt_6_0",
        logical_page=6,
        left=53.5,
        year_right=81.0,
        right=403.5,
        row_edges=(43.5, 58.5, 73.5),
    )
    year_bbox = [59.5, 53.5, 77.0, 63.0]
    tables = [table]
    if failure_mode == "outside_row_pair":
        year_bbox = [59.5, 30.0, 77.0, 40.0]
    elif failure_mode == "month_column_target":
        year_bbox = [90.0, 53.5, 107.5, 63.0]
    elif failure_mode == "tiny_boundary_speck":
        year_bbox = [59.5, 57.9, 77.0, 59.1]
    elif failure_mode == "status_non_exact":
        table["cell_geometry_status"][0][0] = "derived"  # type: ignore[index]
    elif failure_mode == "amount_missing":
        table["cell_bboxes"][1][0] = None  # type: ignore[index]
        table["cell_geometry_status"][1][0] = "missing"  # type: ignore[index]
    elif failure_mode == "amount_inset":
        table["cell_bboxes"][1][0] = [53.5, 59.0, 81.0, 73.5]  # type: ignore[index]
    elif failure_mode == "status_spanned":
        table["cell_spans"][0]["row_span"] = 2  # type: ignore[index]
    else:
        duplicate = deepcopy(table)
        duplicate["table_id"] = "pt_6_1"
        tables.append(duplicate)

    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            tables,
            logical_page=6,
            expected_year=2022,
            active_months=range(1, 13),
            year_bbox=year_bbox,
            status_bbox=[81.0, 43.5, 403.5, 58.5],
        )
        is None
    )


@pytest.mark.parametrize(
    "year_bbox",
    (
        [59.5, 45.0, 77.0, 55.0],
        [59.5, 50.0, 77.0, 60.0],
    ),
    ids=("wholly_status_row", "sub_twenty_percent_boundary_graze"),
)
def test_year_target_owned_by_one_singleton_keeps_existing_mode(
    year_bbox: list[float],
) -> None:
    table = _boundary_split_year_table(
        table_id="pt_6_0",
        logical_page=6,
        left=53.5,
        year_right=81.0,
        right=403.5,
        row_edges=(43.5, 58.5, 73.5),
    )

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=6,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=year_bbox,
        status_bbox=[81.0, 43.5, 403.5, 58.5],
    )

    assert lattice is not None
    assert lattice.provenance_dict()["year_anchor_mode"] == (
        "target_bound_singleton_year_cell"
    )


def test_resolver_rejects_non_repeated_or_competing_primitive_split() -> None:
    missing_amount_span = _primitive_split_table((6,))
    missing_amount_span["cell_spans"] = [  # type: ignore[index]
        span
        for span in missing_amount_span["cell_spans"]  # type: ignore[union-attr]
        if not (span["row"] == 2 and span["col"] == 6)
    ]
    competing = _primitive_split_table((6,))
    competing["cell_bboxes"][1][7] = [  # type: ignore[index]
        235.5,
        163.0,
        249.0,
        176.0,
    ]
    competing["cell_geometry_status"][1][7] = "exact"  # type: ignore[index]
    mismatched_union = _primitive_split_table((6,))
    mismatched_union["cell_bboxes"][0][6][2] = (  # type: ignore[index]
        mismatched_union["col_bands"][6]["x1"]  # type: ignore[index]
    )
    duplicate_span = _primitive_split_table((6,))
    duplicated = next(
        span
        for span in duplicate_span["cell_spans"]  # type: ignore[union-attr]
        if span["row"] == 1 and span["col"] == 6
    )
    duplicate_span["cell_spans"].append(deepcopy(duplicated))  # type: ignore[union-attr]

    kwargs = {
        "logical_page": 19,
        "expected_year": 2022,
        "active_months": range(1, 13),
        "year_bbox": [50.0, 164.0, 69.0, 187.0],
        "status_bbox": [262.03, 162.0, 289.03, 177.0],
    }
    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [missing_amount_span],
            **kwargs,
        )
        is None
    )
    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [competing],
            **kwargs,
        )
        is None
    )
    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [mismatched_union],
            **kwargs,
        )
        is None
    )
    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [duplicate_span],
            **kwargs,
        )
        is None
    )


def test_resolver_rejects_disjoint_row_evidence_for_collapsed_groups() -> None:
    table = _primitive_split_table((2, 6))
    boxes = table["cell_bboxes"]
    statuses = table["cell_geometry_status"]
    spans = table["cell_spans"]
    row_bands = table["row_bands"]
    col_bands = table["col_bands"]
    assert isinstance(boxes, list)
    assert isinstance(statuses, list)
    assert isinstance(spans, list)
    assert isinstance(row_bands, list)
    assert isinstance(col_bands, list)

    split_anchors = sorted(
        {
            int(span["col"])
            for span in spans
            if span.get("col_span") == 2
        }
    )
    assert split_anchors == [2, 7]
    second_anchor = split_anchors[1]
    spans[:] = [
        span
        for span in spans
        if not (span.get("col") == second_anchor and int(span["row"]) < 3)
    ]
    for row in range(3):
        boxes[row][second_anchor] = None
        statuses[row][second_anchor] = "missing"

    for row, (y0, y1) in enumerate(
        ((189.0, 202.0), (202.0, 215.0), (215.0, 228.0)),
        start=3,
    ):
        if row >= len(boxes):
            boxes.append([None] * len(col_bands))
            statuses.append(["missing"] * len(col_bands))
            row_bands.append({"index": row, "y0": y0, "y1": y1})
        boxes[row][second_anchor] = [
            col_bands[second_anchor]["x0"],
            y0,
            col_bands[second_anchor + 1]["x1"],
            y1,
        ]
        statuses[row][second_anchor] = "exact"
        statuses[row][second_anchor + 1] = "derived"
        spans.append(
            {
                "row": row,
                "col": second_anchor,
                "row_span": 1,
                "col_span": 2,
            }
        )
    table["horizontal_lines"] = [150.0, 163.0, 176.0, 189.0, 202.0, 215.0, 228.0]
    table["bbox"][3] = 228.0  # type: ignore[index]

    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [table],
            logical_page=19,
            expected_year=2022,
            active_months=(2,),
            year_bbox=[50.0, 164.0, 69.0, 187.0],
            status_bbox=[127.0, 162.0, 154.0, 177.0],
        )
        is None
    )


@pytest.mark.parametrize(
    ("table_id", "column_count"),
    (
        ("pt_9_1", 3),
        ("pt_13_1", 6),
        ("pt_14_2", 6),
        ("pt_16_1", 6),
    ),
)
def test_resolver_preserves_saved_coarse_table_withholding(
    table_id: str,
    column_count: int,
) -> None:
    coarse = _scanner_table()
    coarse["table_id"] = table_id
    coarse["cell_bboxes"] = [  # type: ignore[index]
        row[:column_count] for row in coarse["cell_bboxes"]
    ]
    coarse["cell_geometry_status"] = [  # type: ignore[index]
        row[:column_count]
        for row in coarse["cell_geometry_status"]  # type: ignore[union-attr]
    ]
    coarse["cell_spans"] = []
    coarse["col_bands"] = coarse["col_bands"][:column_count]  # type: ignore[index]
    coarse["vertical_lines"] = coarse["vertical_lines"][: column_count + 1]  # type: ignore[index]
    coarse["bbox"][2] = coarse["vertical_lines"][-1]  # type: ignore[index]

    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [coarse],
            logical_page=19,
            expected_year=2022,
            active_months=(1, 2),
            year_bbox=[50.0, 164.0, 69.0, 187.0],
            status_bbox=[73.0, 163.0, 127.0, 176.0],
        )
        is None
    )


def test_resolver_supports_headerless_span_three_and_sparse_active_months() -> None:
    table = _scanner_table(
        year_anchor_row=0,
        year_row_span=3,
        status_row=1,
        active_months=range(5, 13),
    )
    # Row zero is a spanning year anchor, not a complete month header.
    for month in range(1, 5):
        table["cell_bboxes"][0][month] = None  # type: ignore[index]
        table["cell_geometry_status"][0][month] = "missing"  # type: ignore[index]

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2021,
        active_months=range(5, 13),
        year_bbox=[50.0, 151.0, 69.0, 187.0],
        status_bbox=[181.0, 163.0, 208.0, 177.0],
    )

    assert lattice is not None
    assert lattice.year_anchor_row_index == 0
    assert lattice.header_row_index == -1
    assert lattice.provenance_dict()["year_row_span"] == 3
    assert len(lattice.month_bboxes) == 12


@pytest.mark.parametrize("year_anchor_row", (1, 2))
def test_resolver_binds_split_year_cell_by_target_geometry_only(
    year_anchor_row: int,
) -> None:
    table = _scanner_table(
        year_anchor_row=year_anchor_row,
        year_row_span=1,
        status_row=1,
    )
    row_edges = [150.0, 163.0, 176.0, 189.0, 202.0]

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=[50.0, row_edges[year_anchor_row] + 1.0, 69.0, row_edges[year_anchor_row + 1] - 1.0],
        status_bbox=[73.0, 164.0, 397.0, 175.0],
    )

    assert lattice is not None
    assert lattice.year_anchor_row_index == year_anchor_row
    assert lattice.provenance_dict()["year_anchor_mode"] == (
        "target_bound_singleton_year_cell"
    )
    assert lattice.provenance_dict()["value_inputs_used"] is False


def test_resolver_uses_global_rules_for_one_explicitly_merged_active_cell() -> None:
    table = _scanner_table(year_anchor_row=1, year_row_span=1, status_row=1)
    table["cell_bboxes"][1][1] = [73.0, 163.0, 100.0, 189.0]  # type: ignore[index]
    table["cell_bboxes"][2][1] = None  # type: ignore[index]
    table["cell_geometry_status"][2][1] = "derived"  # type: ignore[index]
    table["cell_spans"].append(  # type: ignore[union-attr]
        {"row": 1, "col": 1, "row_span": 2, "col_span": 1}
    )

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=[50.0, 164.0, 69.0, 175.0],
        status_bbox=[73.0, 163.0, 397.0, 176.0],
    )

    assert lattice is not None
    provenance = lattice.provenance_dict()
    assert provenance["active_cell_geometry_exact"] is False
    assert provenance["active_cell_rule_derived_count"] == 2
    assert provenance["horizontal_rule_count"] == 5
    assert provenance["year_anchor_mode"] == "target_bound_singleton_year_cell"
    assert provenance["value_inputs_used"] is False


def test_resolver_does_not_invent_month_cells_inside_a_wide_merged_row() -> None:
    table = _scanner_table()
    table["cell_bboxes"][1][1] = [73.0, 163.0, 397.0, 189.0]  # type: ignore[index]
    for row in (1, 2):
        for month in range(2 if row == 1 else 1, 13):
            table["cell_bboxes"][row][month] = None  # type: ignore[index]
            table["cell_geometry_status"][row][month] = "derived"  # type: ignore[index]
    table["cell_spans"].append(  # type: ignore[union-attr]
        {"row": 1, "col": 1, "row_span": 2, "col_span": 12}
    )

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(1, 13),
        year_bbox=[50.0, 164.0, 69.0, 187.0],
        status_bbox=[73.0, 163.0, 397.0, 176.0],
    )

    assert lattice is None


@pytest.mark.parametrize("contradiction", ("wide_span", "extra_exact_cell"))
def test_resolver_rejects_competing_geometry_over_intact_month_cells(
    contradiction: str,
) -> None:
    table = _scanner_table()
    if contradiction == "wide_span":
        table["cell_spans"].append(  # type: ignore[union-attr]
            {"row": 1, "col": 1, "row_span": 2, "col_span": 12}
        )
    else:
        table["cell_bboxes"][1].append(  # type: ignore[index]
            deepcopy(table["cell_bboxes"][1][8])  # type: ignore[index]
        )
        table["cell_geometry_status"][1].append("exact")  # type: ignore[index]

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=(8,),
        year_bbox=[50.0, 164.0, 69.0, 187.0],
        status_bbox=[262.03, 162.0, 289.03, 177.0],
    )

    assert lattice is None


def test_resolver_rejects_declared_spanning_year_cell_with_inset_bbox() -> None:
    table = _scanner_table()
    # The metadata still declares a two-row span, but its exact bbox covers
    # only the status row.  The target glyph fitting that inset must not make
    # the contradictory ownership proof valid.
    table["cell_bboxes"][1][0] = [45.0, 163.0, 73.0, 176.0]  # type: ignore[index]

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=(8,),
        year_bbox=[50.0, 164.0, 69.0, 175.0],
        status_bbox=[262.03, 162.0, 289.03, 177.0],
    )

    assert lattice is None


def test_resolver_rejects_split_year_cell_when_target_geometry_is_ambiguous() -> None:
    table = _scanner_table(year_anchor_row=1, year_row_span=1, status_row=1)
    table["cell_bboxes"][2][0] = [45.0, 163.0, 73.0, 189.0]  # type: ignore[index]
    table["cell_geometry_status"][2][0] = "exact"  # type: ignore[index]

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=(8,),
        year_bbox=[50.0, 164.0, 69.0, 175.0],
        status_bbox=[262.03, 162.0, 289.03, 177.0],
    )

    assert lattice is None


def test_resolver_accepts_a_wide_partial_status_row_inside_the_month_lattice() -> None:
    table = _scanner_table(active_months=range(5, 13))

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=range(5, 13),
        year_bbox=[50.0, 164.0, 69.0, 187.0],
        status_bbox=[181.0, 163.0, 397.0, 176.0],
    )

    assert lattice is not None
    assert lattice.status_row_index == 1


@pytest.mark.parametrize(
    "status_bbox",
    (
        [0.0, 163.0, 60.0, 176.0],
        [500.0, 163.0, 600.0, 176.0],
    ),
)
def test_resolver_rejects_wide_same_y_targets_outside_the_month_lattice(
    status_bbox: list[float],
) -> None:
    table = _scanner_table()

    lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        [table],
        logical_page=19,
        expected_year=2022,
        active_months=(8,),
        year_bbox=[50.0, 164.0, 69.0, 187.0],
        status_bbox=status_bbox,
    )

    assert lattice is None


def test_resolver_rejects_ambiguous_shifted_and_non_exact_lattices() -> None:
    clean = _scanner_table()
    duplicate = deepcopy(clean)
    duplicate["table_id"] = "pt_test_duplicate"
    shifted = deepcopy(clean)
    shifted["cell_bboxes"][1][8] = [289.0, 163.0, 316.0, 176.0]  # type: ignore[index]
    extra_rule = deepcopy(clean)
    extra_rule["vertical_lines"] = [*extra_rule["vertical_lines"], 410.0]  # type: ignore[index]
    missing_column = deepcopy(clean)
    missing_column["col_bands"] = missing_column["col_bands"][:-1]  # type: ignore[index]
    wrong_source = deepcopy(clean)
    wrong_source["geometry_source"] = "estimated"
    missing_horizontal_rule = deepcopy(clean)
    missing_horizontal_rule["horizontal_lines"] = missing_horizontal_rule[
        "horizontal_lines"
    ][:-1]  # type: ignore[index]

    kwargs = {
        "logical_page": 19,
        "expected_year": 2022,
        "active_months": (8,),
        "year_bbox": [50.0, 164.0, 69.0, 187.0],
        "status_bbox": [262.03, 162.0, 289.03, 177.0],
    }
    assert resolve_unique_source_table_year_plus_twelve_ownership([clean, duplicate], **kwargs) is None
    for invalid in (
        shifted,
        extra_rule,
        missing_column,
        wrong_source,
        missing_horizontal_rule,
    ):
        assert resolve_unique_source_table_year_plus_twelve_ownership([invalid], **kwargs) is None


def test_resolver_malformed_public_arguments_fail_closed() -> None:
    table = _scanner_table()
    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [table],
            logical_page="bad",  # type: ignore[arg-type]
            expected_year="bad",  # type: ignore[arg-type]
            active_months=(8,),
            year_bbox=[50.0, 164.0, 69.0, 187.0],
            status_bbox=[262.03, 162.0, 289.03, 177.0],
        )
        is None
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("logical_page", 19),
        ("expected_year", 2022),
        ("active_month", 8),
    ),
)
def test_resolver_accepts_only_native_integer_public_ordinals(
    argument: str,
    value: object,
) -> None:
    table = _scanner_table()
    kwargs: dict[str, object] = {
        "logical_page": 19,
        "expected_year": 2022,
        "active_months": (8,),
        "year_bbox": [50.0, 164.0, 69.0, 187.0],
        "status_bbox": [262.03, 162.0, 289.03, 177.0],
    }
    if argument == "active_month":
        kwargs["active_months"] = (value,)
    else:
        kwargs[argument] = value

    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [table],
            **kwargs,  # type: ignore[arg-type]
        )
        is not None
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("logical_page", True),
        ("logical_page", 19.0),
        ("logical_page", 19.5),
        ("logical_page", float("nan")),
        ("logical_page", float("inf")),
        ("logical_page", "19"),
        ("logical_page", " +19 "),
        ("logical_page", "19.5"),
        ("logical_page", "junk"),
        ("logical_page", 0),
        ("logical_page", -19),
        ("expected_year", True),
        ("expected_year", 2022.0),
        ("expected_year", 2022.5),
        ("expected_year", float("nan")),
        ("expected_year", float("inf")),
        ("expected_year", "2022"),
        ("expected_year", " +2022 "),
        ("expected_year", "2022.5"),
        ("expected_year", "junk"),
        ("expected_year", 1899),
        ("expected_year", 2100),
        ("active_month", True),
        ("active_month", 8.0),
        ("active_month", 8.5),
        ("active_month", float("nan")),
        ("active_month", float("inf")),
        ("active_month", "8"),
        ("active_month", "+8"),
        ("active_month", "8.5"),
        ("active_month", "junk"),
        ("active_month", 0),
        ("active_month", 13),
    ),
)
def test_resolver_rejects_inexact_or_invalid_public_ordinals(
    argument: str,
    value: object,
) -> None:
    table = _scanner_table()
    kwargs: dict[str, object] = {
        "logical_page": 19,
        "expected_year": 2022,
        "active_months": (8,),
        "year_bbox": [50.0, 164.0, 69.0, 187.0],
        "status_bbox": [262.03, 162.0, 289.03, 177.0],
    }
    if argument == "active_month":
        kwargs["active_months"] = (value,)
    else:
        kwargs[argument] = value

    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [table],
            **kwargs,  # type: ignore[arg-type]
        )
        is None
    )


@pytest.mark.parametrize(
    "active_months",
    (None, 8, "8", b"8", {"month": 8}, (8, "junk")),
)
def test_resolver_invalid_active_month_collections_fail_closed(
    active_months: object,
) -> None:
    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            [_scanner_table()],
            logical_page=19,
            expected_year=2022,
            active_months=active_months,  # type: ignore[arg-type]
            year_bbox=[50.0, 164.0, 69.0, 187.0],
            status_bbox=[262.03, 162.0, 289.03, 177.0],
        )
        is None
    )


@pytest.mark.parametrize("tables", (None, 17, 19.0, True, "table", b"table", {}))
def test_resolver_invalid_table_collections_fail_closed(tables: object) -> None:
    assert (
        resolve_unique_source_table_year_plus_twelve_ownership(
            tables,  # type: ignore[arg-type]
            logical_page=19,
            expected_year=2022,
            active_months=(8,),
            year_bbox=[50.0, 164.0, 69.0, 187.0],
            status_bbox=[262.03, 162.0, 289.03, 177.0],
        )
        is None
    )


def test_sanitizer_prefers_logical_cell_geometry_and_emits_no_values() -> None:
    geometry = _scanner_table()
    table = SimpleNamespace(
        table_id="pt_test_0",
        bbox=geometry["bbox"],
        extraction_layer="scanned_image_line_grid",
        metadata={
            "geometry": {
                key: deepcopy(value)
                for key, value in geometry.items()
                if key
                in {
                    "cell_bboxes",
                    "cell_geometry_status",
                    "cell_spans",
                    "row_bands",
                    "col_bands",
                    "vertical_lines",
                    "horizontal_lines",
                    "geometry_source",
                    "coordinate_system",
                }
            },
            "source_cell_bboxes": [[[1.0, 1.0, 2.0, 2.0]]],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=19,
        source_page_number=10,
        tables=[table],
    )

    detached = detached_source_table_geometry_by_page([page])[19][0]

    assert detached["cell_bboxes"][1][8] == geometry["cell_bboxes"][1][8]
    assert detached["source_cell_bboxes"] == [[[1.0, 1.0, 2.0, 2.0]]]
    assert not {"text", "raw_rows", "evidence_ids", "token_ids"}.intersection(detached)
