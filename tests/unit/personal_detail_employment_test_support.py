from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace


def own_employment_table(
    table: SimpleNamespace,
    *,
    logical_page: int,
    source_page: int,
) -> SimpleNamespace:
    """Attach generic exact PBOC profile ownership to an employment fixture."""

    rows = table.metadata["raw_rows"]
    columns = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (columns - len(row)))
    widths = [41.0 + ((column * 47) % 113) for column in range(columns)]
    column_edges = [17.0]
    for width in widths:
        column_edges.append(column_edges[-1] + width)
    row_edges = [23.0]
    for row in range(len(rows)):
        row_edges.append(row_edges[-1] + 13.0 + (row % 3))
    evidence = [
        [
            [f"employment:{table.table_id}:{row}:{column}"] if str(value).strip() else []
            for column, value in enumerate(values)
        ]
        for row, values in enumerate(rows)
    ]
    table.metadata.update(
        {
            "canonical_template_id": "report_header_and_identity",
            "source_logical_page": logical_page,
            "source_page": source_page,
            "geometry": {
                "coordinate_system": "pdf_points_top_left",
                "row_bands": [
                    {"index": row, "y0": row_edges[row], "y1": row_edges[row + 1]}
                    for row in range(len(rows))
                ],
                "col_bands": [
                    {
                        "index": column,
                        "x0": column_edges[column],
                        "x1": column_edges[column + 1],
                    }
                    for column in range(columns)
                ],
                "cell_bboxes": [
                    [
                        [
                            column_edges[column],
                            row_edges[row],
                            column_edges[column + 1],
                            row_edges[row + 1],
                        ]
                        for column in range(columns)
                    ]
                    for row in range(len(rows))
                ],
                "cell_geometry_status": [["exact"] * columns for _row in rows],
                "cell_evidence_ids": evidence,
                "cell_token_ids": deepcopy(evidence),
                "cell_spans": [],
            },
        }
    )
    table.bbox = [column_edges[0], row_edges[0], column_edges[-1], row_edges[-1]]
    return table


def employment_page(
    *tables: SimpleNamespace,
    logical_page: int,
    source_page: int | None = None,
) -> SimpleNamespace:
    source_page = logical_page if source_page is None else source_page
    for table in tables:
        own_employment_table(
            table,
            logical_page=logical_page,
            source_page=source_page,
        )
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=source_page,
        canonical_template_id="report_header_and_identity",
        canonical_fragment_logical_pages=(logical_page,),
        coordinate_transform={"source_page_numbers": [source_page]},
        tables=list(tables),
        texts=[],
    )
