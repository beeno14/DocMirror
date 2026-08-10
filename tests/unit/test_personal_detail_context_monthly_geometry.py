from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)


def _table(
    table_id: str,
    *,
    left: float,
    raw_value: str = "SOURCE_VALUE_MUST_NOT_LEAK",
) -> SimpleNamespace:
    cell_bbox = [left, 100.0, left + 20.0, 120.0]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[left, 100.0, left + 20.0, 120.0],
        extraction_layer="static-slice-grid",
        rows=[[raw_value]],
        metadata={
            "raw_rows": [[raw_value]],
            "geometry_source": "static-slice-grid",
            "coordinate_system": "pdf_points_top_left",
            "cell_bboxes": [[cell_bbox]],
            "cell_geometry_status": [["exact"]],
            "cell_spans": [],
            "row_bands": [{"index": 0, "bbox": [left, 100.0, left + 20.0, 120.0]}],
            "col_bands": [{"index": 0, "bbox": [left, 100.0, left + 20.0, 120.0]}],
        },
    )


def _page(page: int, *tables: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=page,
        source_page_number=page,
        width=600.0,
        height=800.0,
        tables=list(tables),
        texts=[],
    )


def _unusable_table(table_id: str, *, left: float) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        bbox=[left, 100.0, left + 20.0, 120.0],
        extraction_layer="static-slice-grid",
        rows=[],
        metadata={
            "raw_rows": [["FILTERED_CONFLICT_MUST_NOT_LEAK"]],
            "cell_bboxes": [[[left, 100.0, left + 20.0, 120.0]]],
        },
    )


def _context(
    *,
    sealed_pages: list[SimpleNamespace],
    canonical_pages: list[SimpleNamespace] | None,
) -> PersonalDetailExtractionContext:
    evidence_pages = canonical_pages if canonical_pages is not None else sealed_pages
    logical_pages = sorted(
        {
            page.page_number
            for page in evidence_pages
            if isinstance(page.page_number, int) and page.page_number > 0
        }
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}
    context.reading_order_by_logical = {
        page: order for order, page in enumerate(logical_pages, start=1)
    }
    context.reading_order_resolution = {
        "resolved": False,
        "authoritative": False,
        "basis": "single_page_test",
    }
    context._page_image_resolver = None
    context.parse_result = SimpleNamespace(
        pages=sealed_pages,
        entities=SimpleNamespace(
            domain_specific={
                "_page_evidence_bundles": [
                    {
                        "page": page,
                        "local_structure_evidence": {"page": page},
                        "micro_grid_evidence": {"page": page},
                    }
                    for page in logical_pages
                ]
            }
        ),
    )
    context._canonical_layout_projection_cache = (
        None
        if canonical_pages is None
        else SimpleNamespace(pages=canonical_pages)
    )
    context.corrected_evidence_pages = lambda: [
        {
            "page": page,
            "source_page": page,
            "page_width": 600.0,
            "page_height": 800.0,
            "lines": [],
        }
        for page in logical_pages
    ]
    return context


def _captured_primary_input(
    monkeypatch: pytest.MonkeyPatch,
    context: PersonalDetailExtractionContext,
) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []

    def materialize(detached: dict[str, Any], **_kwargs: Any) -> None:
        captured.append(deepcopy(detached))

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize."
        "materialize_credit_repayment_micro_grids_from_bundles",
        materialize,
    )
    assert context.corrected_repayment_records() == []
    assert len(captured) == 2
    return captured[0]


def _source_geometry(
    detached: dict[str, Any],
    *,
    page: int = 1,
) -> list[dict[str, Any]]:
    bundle = next(
        bundle
        for bundle in detached["_page_evidence_bundles"]
        if bundle["page"] == page
    )
    return bundle["micro_grid_evidence"]["source_table_geometry"]


def test_monthly_geometry_uses_selected_static_canonical_page_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_table = _table("static-slice-grid", left=210.0)
    context = _context(
        sealed_pages=[_page(1)],
        canonical_pages=[_page(1, canonical_table)],
    )

    geometry = _source_geometry(_captured_primary_input(monkeypatch, context))

    assert [table["table_id"] for table in geometry] == ["static-slice-grid"]
    assert geometry[0]["cell_bboxes"] == [[[210.0, 100.0, 230.0, 120.0]]]
    assert "SOURCE_VALUE_MUST_NOT_LEAK" not in repr(geometry)
    assert "raw_rows" not in geometry[0]


@pytest.mark.parametrize(
    "canonical_pages",
    [
        [_page(1, _table("grid-a", left=100.0)), _page(1, _table("grid-b", left=200.0))],
        [
            _page(
                1,
                _table("duplicate-grid", left=100.0),
                _table("duplicate-grid", left=200.0),
            )
        ],
        [
            _page(
                1,
                _table("filtered-duplicate-grid", left=100.0),
                _unusable_table("filtered-duplicate-grid", left=200.0),
            )
        ],
        [
            _page(
                1,
                _table("exact-grid", left=100.0),
                _unusable_table("filtered-grid", left=100.0),
            )
        ],
    ],
    ids=(
        "duplicate_page",
        "conflicting_duplicate_table",
        "filtered_conflicting_duplicate_table",
        "filtered_duplicate_physical_geometry",
    ),
)
def test_monthly_geometry_fails_closed_for_nonunique_canonical_ownership(
    monkeypatch: pytest.MonkeyPatch,
    canonical_pages: list[SimpleNamespace],
) -> None:
    context = _context(
        sealed_pages=[_page(1, _table("sealed-fallback", left=10.0))],
        canonical_pages=canonical_pages,
    )

    geometry = _source_geometry(_captured_primary_input(monkeypatch, context))

    assert geometry == []


def test_monthly_geometry_allows_same_layout_on_distinct_canonical_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        sealed_pages=[_page(1), _page(2)],
        canonical_pages=[
            _page(1, _table("page-1-grid", left=100.0)),
            _page(2, _table("page-2-grid", left=100.0)),
        ],
    )

    detached = _captured_primary_input(monkeypatch, context)

    assert [table["table_id"] for table in _source_geometry(detached, page=1)] == [
        "page-1-grid"
    ]
    assert [table["table_id"] for table in _source_geometry(detached, page=2)] == [
        "page-2-grid"
    ]


def test_monthly_geometry_rejects_one_table_identity_claiming_two_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        sealed_pages=[_page(1), _page(2)],
        canonical_pages=[
            _page(1, _table("cross-page-duplicate", left=100.0)),
            _page(2, _table("cross-page-duplicate", left=200.0)),
        ],
    )

    detached = _captured_primary_input(monkeypatch, context)

    assert _source_geometry(detached, page=1) == []
    assert _source_geometry(detached, page=2) == []


def test_monthly_geometry_preserves_sealed_ordinary_page_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        sealed_pages=[_page(1, _table("ordinary-grid", left=40.0, raw_value="N"))],
        canonical_pages=None,
    )

    geometry = _source_geometry(_captured_primary_input(monkeypatch, context))

    assert [table["table_id"] for table in geometry] == ["ordinary-grid"]
    assert geometry[0]["cell_bboxes"] == [[[40.0, 100.0, 60.0, 120.0]]]
    assert "N" not in repr(geometry)
