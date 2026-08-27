from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.evidence.plane import EvidencePlaneBuilder
from docmirror.models.entities.parse_result import CellValue, PageContent, ParseResult, TableBlock, TableRow
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    _authenticated_printed_monthly_anchors,
    _exact_source_table_repair_tokens_by_page,
    _lines_with_printed_monthly_anchors,
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
        {page.page_number for page in evidence_pages if isinstance(page.page_number, int) and page.page_number > 0}
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}
    context.reading_order_by_logical = {page: order for order, page in enumerate(logical_pages, start=1)}
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
        None if canonical_pages is None else SimpleNamespace(pages=canonical_pages)
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
        "docmirror.plugins.credit_report.micro_grid_materialize.materialize_credit_repayment_micro_grids_from_bundles",
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
    bundle = next(bundle for bundle in detached["_page_evidence_bundles"] if bundle["page"] == page)
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

    assert [table["table_id"] for table in _source_geometry(detached, page=1)] == ["page-1-grid"]
    assert [table["table_id"] for table in _source_geometry(detached, page=2)] == ["page-2-grid"]


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
        sealed_pages=[_page(1, _table("ordinary-grid", left=40.0, raw_value="PRIVATE_RAW_STATUS_N"))],
        canonical_pages=None,
    )

    geometry = _source_geometry(_captured_primary_input(monkeypatch, context))

    assert [table["table_id"] for table in geometry] == ["ordinary-grid"]
    assert geometry[0]["cell_bboxes"] == [[[40.0, 100.0, 60.0, 120.0]]]
    assert "PRIVATE_RAW_STATUS_N" not in repr(geometry)


def test_monthly_repair_deploys_only_exact_source_table_field_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical table atoms, not table values, reach the field repair seam."""

    status_bbox = [120.0, 100.0, 140.0, 120.0]
    amount_bbox = [120.0, 120.0, 140.0, 140.0]
    table = SimpleNamespace(
        table_id="repayment-grid",
        bbox=[100.0, 100.0, 360.0, 140.0],
        extraction_layer="scanned_image_line_grid",
        rows=[["2024", "PRIVATE_STATUS_VALUE"], ["", "PRIVATE_AMOUNT_VALUE"]],
        metadata={
            "geometry": {
                "geometry_source": "scanned_image_line_grid",
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": [
                    [[100.0, 100.0, 120.0, 140.0], status_bbox],
                    [None, amount_bbox],
                ],
                "cell_geometry_status": [
                    ["exact", "exact"],
                    ["derived", "exact"],
                ],
                "cell_evidence_ids": [
                    [["year-token"], ["status-token"]],
                    [[], ["amount-token"]],
                ],
                "cell_token_ids": [
                    [["year-token"], ["status-token"]],
                    [[], ["amount-token"]],
                ],
                "cell_spans": [{"row": 0, "col": 0, "row_span": 2, "col_span": 1}],
                "row_bands": [
                    {"index": 0, "y0": 100.0, "y1": 120.0},
                    {"index": 1, "y0": 120.0, "y1": 140.0},
                ],
                "col_bands": [
                    {"index": 0, "x0": 100.0, "x1": 120.0},
                    {"index": 1, "x0": 120.0, "x1": 140.0},
                ],
            }
        },
    )
    context = _context(
        sealed_pages=[_page(1)],
        canonical_pages=[_page(1, table)],
    )
    context.parse_result.entities.domain_specific["_page_evidence_bundles"][0]["tokens"] = [
        {
            "token_id": "year-token",
            "text": "2024",
            "bbox": [101.0, 105.0, 117.0, 115.0],
            "confidence": 0.99,
            "page": 1,
            "evidence_ids": ["year-token"],
        },
        {
            "token_id": "status-token",
            "text": "N",
            "bbox": [125.0, 104.0, 135.0, 116.0],
            "confidence": 0.97,
            "page": 1,
            "evidence_ids": ["status-token"],
        },
        {
            "token_id": "amount-token",
            "text": "0",
            "bbox": [125.0, 124.0, 135.0, 136.0],
            "confidence": 0.98,
            "page": 1,
            "evidence_ids": ["amount-token"],
        },
    ]
    context.parse_result.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                SimpleNamespace(
                    id="ev:0001:text:000001",
                    text="2024",
                    bbox=[101.0, 105.0, 117.0, 115.0],
                    confidence=0.99,
                    source_refs=["year-token"],
                ),
                SimpleNamespace(
                    id="ev:0001:text:000002",
                    text="N",
                    bbox=[125.0, 104.0, 135.0, 116.0],
                    confidence=0.97,
                    source_refs=["status-token"],
                ),
                SimpleNamespace(
                    id="ev:0001:text:000003",
                    text="0",
                    bbox=[125.0, 124.0, 135.0, 136.0],
                    confidence=0.98,
                    source_refs=["amount-token"],
                ),
                SimpleNamespace(
                    id="ev:0001:text:000004",
                    text="PRIVATE_STATUS_VALUE",
                    bbox=status_bbox,
                    confidence=0.99,
                    source_refs=["status-token"],
                ),
            ]
        )
    )

    detached = _captured_primary_input(monkeypatch, context)
    [bundle] = detached["_page_evidence_bundles"]
    tokens = bundle["micro_grid_evidence"]["tokens"]

    assert tokens == [
        {
            "token_id": "status-token",
            "content": "N",
            "bbox": [125.0, 104.0, 135.0, 116.0],
            "confidence": 0.97,
            "page": 1,
            "source_logical_page": 1,
            "source_origin_logical_page": 1,
            "coordinate_system": "pdf_points_top_left",
            "source": "exact_native_source_table_status_cell",
            "evidence_ids": ["status-token"],
        },
        {
            "token_id": "amount-token",
            "content": "0",
            "bbox": [125.0, 124.0, 135.0, 136.0],
            "confidence": 0.98,
            "page": 1,
            "source_logical_page": 1,
            "source_origin_logical_page": 1,
            "coordinate_system": "pdf_points_top_left",
            "source": "exact_native_source_table_amount_cell",
            "evidence_ids": ["amount-token"],
        },
    ]
    assert "PRIVATE_STATUS_VALUE" not in repr(tokens)
    assert "PRIVATE_AMOUNT_VALUE" not in repr(tokens)


def test_monthly_repair_uses_exact_corrected_cell_atom_when_raw_token_is_noisy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrected cell stays usable while raw OCR proves only its ownership."""

    cell_bbox = [120.0, 100.0, 140.0, 120.0]
    table = SimpleNamespace(
        table_id="corrected-repayment-grid",
        bbox=cell_bbox,
        extraction_layer="scanned_image_line_grid",
        rows=[["PRIVATE_TABLE_VALUE_MUST_NOT_LEAK"]],
        metadata={
            "geometry": {
                "geometry_source": "scanned_image_line_grid",
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": [[cell_bbox]],
                "cell_geometry_status": [["exact"]],
                "cell_evidence_ids": [[["status-token"]]],
                "cell_token_ids": [[["status-token"]]],
                "cell_spans": [],
                "row_bands": [{"index": 0, "y0": 100.0, "y1": 120.0}],
                "col_bands": [{"index": 0, "x0": 120.0, "x1": 140.0}],
            }
        },
    )
    context = _context(
        sealed_pages=[_page(1)],
        canonical_pages=[_page(1, table)],
    )
    context.parse_result.parser_info = SimpleNamespace(
        options={
            "ocr_corrections": {
                "events": [
                    {
                        "event_id": "correction-event-1",
                        "rule_id": "monthly-status-confusable",
                        "source_ref": "status-token",
                        "action": "applied",
                        "original": "?",
                        "corrected": "N",
                    }
                ]
            }
        }
    )
    context.parse_result.entities.domain_specific["_page_evidence_bundles"][0]["tokens"] = [
        {
            "token_id": "status-token",
            "text": "?",
            "bbox": [125.0, 104.0, 135.0, 116.0],
            "confidence": 0.41,
            "page": 1,
            "evidence_ids": ["status-token"],
        }
    ]
    context.parse_result.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                SimpleNamespace(
                    id="ev:0001:text:raw",
                    source_kind="metadata_ocr_token",
                    page_id="page:0001",
                    text="?",
                    bbox=[125.0, 104.0, 135.0, 116.0],
                    confidence=0.41,
                    source_refs=["status-token"],
                    metadata={},
                ),
                SimpleNamespace(
                    id="ev:0001:text:corrected",
                    source_kind="parse_result_table_cell",
                    page_id="page:0001",
                    text="N",
                    bbox=cell_bbox,
                    confidence=0.96,
                    source_refs=["status-token"],
                    metadata={
                        "table_id": "corrected-repayment-grid",
                        "row_index": 0,
                        "col_index": 0,
                        "geometry_status": "exact",
                        "token_ids": ["status-token"],
                        "source_cell_refs": [
                            {
                                "table_id": "corrected-repayment-grid",
                                "page": 1,
                                "row": 0,
                                "raw_row": 0,
                                "col": 0,
                            }
                        ],
                        "ocr_original_text": "?",
                        "ocr_corrected_text": "N",
                        "ocr_correction_id": "correction-event-1",
                        "ocr_correction_action": "applied",
                        "ocr_correction_rule_id": "monthly-status-confusable",
                    },
                ),
                SimpleNamespace(
                    id="ev:0001:text:wrong-cell",
                    source_kind="parse_result_table_cell",
                    page_id="page:0001",
                    text="M",
                    bbox=cell_bbox,
                    confidence=0.99,
                    source_refs=["status-token"],
                    metadata={
                        "table_id": "different-grid",
                        "row_index": 0,
                        "col_index": 0,
                        "geometry_status": "exact",
                        "token_ids": ["status-token"],
                    },
                ),
            ]
        )
    )

    detached = _captured_primary_input(monkeypatch, context)
    [bundle] = detached["_page_evidence_bundles"]
    tokens = bundle["micro_grid_evidence"]["tokens"]

    assert tokens == [
        {
            "token_id": "status-token",
            "content": "N",
            "bbox": cell_bbox,
            "confidence": 0.96,
            "page": 1,
            "source_logical_page": 1,
            "source_origin_logical_page": 1,
            "coordinate_system": "pdf_points_top_left",
            "source": "exact_corrected_source_table_status_cell",
            "evidence_ids": ["status-token", "ev:0001:text:corrected"],
        }
    ]
    assert "PRIVATE_TABLE_VALUE_MUST_NOT_LEAK" not in repr(tokens)


def test_monthly_repair_never_uses_canonical_typed_cell_as_correction_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed CellValue alone cannot change its immutable raw OCR token."""

    cell_bbox = [120.0, 100.0, 140.0, 120.0]
    typed_cell = SimpleNamespace(
        text="N",
        bbox=cell_bbox,
        confidence=0.94,
        geometry_confidence=0.93,
        row_index=0,
        col_index=0,
        geometry_status="exact",
        evidence_ids=["status-token"],
        token_ids=["status-token"],
        source_cell_refs=[
            {
                "table_id": "canonical-repayment-grid",
                "page": 1,
                "row": 0,
                "raw_row": 0,
                "col": 0,
            }
        ],
    )
    table = SimpleNamespace(
        table_id="canonical-repayment-grid",
        bbox=cell_bbox,
        extraction_layer="scanned_image_line_grid",
        rows=[SimpleNamespace(cells=[typed_cell])],
        metadata={
            "geometry": {
                "geometry_source": "scanned_image_line_grid",
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": [[cell_bbox]],
                "cell_geometry_status": [["exact"]],
                "cell_evidence_ids": [[["status-token"]]],
                "cell_token_ids": [[["status-token"]]],
                "cell_spans": [],
                "row_bands": [{"index": 0, "y0": 100.0, "y1": 120.0}],
                "col_bands": [{"index": 0, "x0": 120.0, "x1": 140.0}],
            }
        },
    )
    context = _context(
        sealed_pages=[_page(1)],
        canonical_pages=[_page(1, table)],
    )
    context.parse_result.entities.domain_specific["_page_evidence_bundles"][0]["tokens"] = [
        {
            "token_id": "status-token",
            "text": "?",
            "bbox": [125.0, 104.0, 135.0, 116.0],
            "confidence": 0.41,
            "page": 1,
            "evidence_ids": ["status-token"],
        }
    ]
    context.parse_result.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                SimpleNamespace(
                    id="ev:0001:text:raw",
                    source_kind="metadata_ocr_token",
                    page_id="page:0001",
                    text="?",
                    bbox=[125.0, 104.0, 135.0, 116.0],
                    confidence=0.41,
                    source_refs=["status-token"],
                    metadata={},
                )
            ]
        )
    )

    detached = _captured_primary_input(monkeypatch, context)
    [bundle] = detached["_page_evidence_bundles"]

    assert bundle["micro_grid_evidence"]["tokens"] == []


def test_monthly_repair_uses_applied_atom_with_header_offset_typed_corroboration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw geometry row 1 maps to typed row 0 only through raw_row provenance."""

    header_bbox = [100.0, 80.0, 140.0, 100.0]
    cell_bbox = [100.0, 100.0, 140.0, 120.0]
    typed_cell = SimpleNamespace(
        text="N",
        bbox=cell_bbox,
        confidence=0.94,
        geometry_confidence=0.93,
        row_index=0,
        col_index=0,
        geometry_status="exact",
        evidence_ids=["status-token"],
        token_ids=["status-token"],
        source_cell_refs=[
            {
                "table_id": "header-offset-grid",
                "page": 1,
                "row": 0,
                "raw_row": 1,
                "col": 0,
            }
        ],
    )
    table = SimpleNamespace(
        table_id="header-offset-grid",
        bbox=[100.0, 80.0, 140.0, 120.0],
        extraction_layer="scanned_image_line_grid",
        rows=[SimpleNamespace(cells=[typed_cell])],
        metadata={
            "raw_rows": [["月份"], ["PRIVATE_TABLE_VALUE_MUST_NOT_LEAK"]],
            "geometry": {
                "geometry_source": "scanned_image_line_grid",
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": [[header_bbox], [cell_bbox]],
                "cell_geometry_status": [["exact"], ["exact"]],
                "cell_evidence_ids": [[["header-token"]], [["status-token"]]],
                "cell_token_ids": [[["header-token"]], [["status-token"]]],
                "cell_spans": [],
                "row_bands": [
                    {"index": 0, "y0": 80.0, "y1": 100.0},
                    {"index": 1, "y0": 100.0, "y1": 120.0},
                ],
                "col_bands": [{"index": 0, "x0": 100.0, "x1": 140.0}],
            },
        },
    )
    context = _context(
        sealed_pages=[_page(1)],
        canonical_pages=[_page(1, table)],
    )
    context.parse_result.parser_info = SimpleNamespace(
        options={
            "ocr_corrections": {
                "events": [
                    {
                        "event_id": "header-offset-event",
                        "rule_id": "monthly-status-confusable",
                        "source_ref": "status-token",
                        "action": "applied",
                        "original": "?",
                        "corrected": "N",
                    }
                ]
            }
        }
    )
    context.parse_result.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                SimpleNamespace(
                    id="ev:0001:text:raw",
                    source_kind="metadata_ocr_token",
                    page_id="page:0001",
                    text="?",
                    bbox=[112.0, 104.0, 128.0, 116.0],
                    confidence=0.41,
                    source_refs=["status-token"],
                    metadata={},
                ),
                SimpleNamespace(
                    id="ev:0001:text:corrected",
                    source_kind="parse_result_table_cell",
                    page_id="page:0001",
                    text="N",
                    bbox=cell_bbox,
                    confidence=0.96,
                    source_refs=["status-token"],
                    metadata={
                        "table_id": "header-offset-grid",
                        "row_index": 0,
                        "col_index": 0,
                        "geometry_status": "exact",
                        "token_ids": ["status-token"],
                        "source_cell_refs": [
                            {
                                "table_id": "header-offset-grid",
                                "page": 1,
                                "row": 0,
                                "raw_row": 1,
                                "col": 0,
                            }
                        ],
                        "ocr_original_text": "?",
                        "ocr_corrected_text": "N",
                        "ocr_correction_id": "header-offset-event",
                        "ocr_correction_action": "applied",
                        "ocr_correction_rule_id": "monthly-status-confusable",
                    },
                ),
            ]
        )
    )

    detached = _captured_primary_input(monkeypatch, context)
    [bundle] = detached["_page_evidence_bundles"]

    assert bundle["micro_grid_evidence"]["tokens"] == [
        {
            "token_id": "status-token",
            "content": "N",
            "bbox": cell_bbox,
            "confidence": 0.93,
            "page": 1,
            "source_logical_page": 1,
            "source_origin_logical_page": 1,
            "coordinate_system": "pdf_points_top_left",
            "source": "exact_corrected_source_table_status_cell",
            "evidence_ids": ["status-token", "ev:0001:text:corrected"],
        }
    ]


def _direct_corrected_repair_case() -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    dict[str, Any],
]:
    cell_bbox = [100.0, 100.0, 140.0, 120.0]
    table = SimpleNamespace(
        table_id="direct-corrected-grid",
        bbox=cell_bbox,
        rows=[["PRIVATE_TABLE_VALUE_MUST_NOT_LEAK"]],
        metadata={
            "geometry": {
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": [[cell_bbox]],
                "cell_geometry_status": [["exact"]],
                "cell_evidence_ids": [[["status-token"]]],
                "cell_token_ids": [[["status-token"]]],
                "cell_spans": [],
                "row_bands": [{"index": 0, "y0": 100.0, "y1": 120.0}],
                "col_bands": [{"index": 0, "x0": 100.0, "x1": 140.0}],
            }
        },
    )
    event = {
        "event_id": "direct-event",
        "rule_id": "monthly-status-confusable",
        "source_ref": "status-token",
        "action": "applied",
        "original": "?",
        "corrected": "N",
    }
    corrected_atom = SimpleNamespace(
        id="ev:0001:text:corrected",
        source_kind="parse_result_table_cell",
        page_id="page:0001",
        text="N",
        bbox=cell_bbox,
        confidence=0.96,
        source_refs=["status-token"],
        metadata={
            "table_id": "direct-corrected-grid",
            "row_index": 0,
            "col_index": 0,
            "geometry_status": "exact",
            "token_ids": ["status-token"],
            "source_cell_refs": [
                {
                    "table_id": "direct-corrected-grid",
                    "page": 1,
                    "row": 0,
                    "raw_row": 0,
                    "col": 0,
                }
            ],
            "ocr_original_text": "?",
            "ocr_corrected_text": "N",
            "ocr_correction_id": "direct-event",
            "ocr_correction_action": "applied",
            "ocr_correction_rule_id": "monthly-status-confusable",
        },
    )
    plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                SimpleNamespace(
                    id="ev:0001:text:raw",
                    source_kind="metadata_ocr_token",
                    page_id="page:0001",
                    text="?",
                    bbox=[112.0, 104.0, 128.0, 116.0],
                    confidence=0.41,
                    source_refs=["status-token"],
                    metadata={},
                ),
                corrected_atom,
            ]
        )
    )
    owner = SimpleNamespace(
        parser_info=SimpleNamespace(
            options={"ocr_corrections": {"events": [event]}}
        ),
        evidence_plane=plane,
    )
    return owner, _page(1, table), corrected_atom, event


@pytest.mark.parametrize(
    "mutation",
    ("missing_event", "mismatched_event", "wrong_page", "duplicate_atom"),
)
def test_monthly_corrected_repair_claims_fail_closed(
    mutation: str,
) -> None:
    owner, page, corrected_atom, event = _direct_corrected_repair_case()
    events = owner.parser_info.options["ocr_corrections"]["events"]
    if mutation == "missing_event":
        events.clear()
    elif mutation == "mismatched_event":
        event["corrected"] = "M"
    elif mutation == "wrong_page":
        corrected_atom.page_id = "page:0002"
    elif mutation == "duplicate_atom":
        duplicate = deepcopy(corrected_atom)
        duplicate.id = "ev:0001:text:corrected-duplicate"
        owner.evidence_plane.evidence.text_atoms.append(duplicate)

    assert _exact_source_table_repair_tokens_by_page(owner, [page], {1}) == {}


def test_monthly_corrected_repair_withholds_present_typed_cell_disagreement() -> None:
    owner, page, _corrected_atom, _event = _direct_corrected_repair_case()
    [table] = page.tables
    table.metadata["raw_rows"] = [["PRIVATE_TABLE_VALUE_MUST_NOT_LEAK"]]
    table.rows = [
        SimpleNamespace(
            cells=[
                SimpleNamespace(
                    text="M",
                    bbox=[100.0, 100.0, 140.0, 120.0],
                    confidence=0.95,
                    geometry_confidence=0.95,
                    row_index=0,
                    col_index=0,
                    geometry_status="exact",
                    evidence_ids=["status-token"],
                    token_ids=["status-token"],
                    source_cell_refs=[
                        {
                            "table_id": "direct-corrected-grid",
                            "page": 1,
                            "row": 0,
                            "raw_row": 0,
                            "col": 0,
                        }
                    ],
                )
            ]
        )
    ]

    assert _exact_source_table_repair_tokens_by_page(owner, [page], {1}) == {}


def _uncorrected_monthly_table_case(
    evidence_key: str = "micro_grid_evidence",
) -> tuple[SimpleNamespace, PageContent, list[dict[str, Any]]]:
    """Build a printed year/status/amount excerpt through the real atom builder."""

    table_id = "uncorrected-monthly-grid"
    year_bbox = [100.0, 100.0, 120.0, 140.0]
    cell_bboxes: list[list[Any]] = [[year_bbox], [None]]
    cell_statuses = [["exact"], ["derived"]]
    cell_token_ids: list[list[list[str]]] = [[["year-token"]], [[]]]
    typed_rows = [
        [
            CellValue(
                text="2024",
                bbox=year_bbox,
                row_index=0,
                col_index=0,
                row_span=2,
                geometry_status="exact",
                evidence_ids=["year-token"],
                token_ids=["year-token"],
            )
        ],
        [CellValue(text="", row_index=1, col_index=0, geometry_status="derived")],
    ]
    raw_tokens: list[dict[str, Any]] = []
    for row, role in enumerate(("status", "amount")):
        y0 = 100.0 + 20.0 * row
        for month in range(1, 13):
            x0 = 100.0 + 20.0 * month
            cell_bbox = [x0, y0, x0 + 20.0, y0 + 20.0]
            token_id = f"{role}-token-{month:02d}"
            value = "N" if role == "status" else "0"
            cell_bboxes[row].append(cell_bbox)
            cell_statuses[row].append("exact")
            cell_token_ids[row].append([token_id])
            typed_rows[row].append(
                CellValue(
                    text=value,
                    bbox=cell_bbox,
                    row_index=row,
                    col_index=month,
                    geometry_status="exact",
                    confidence=0.99,
                    geometry_confidence=0.99,
                    evidence_ids=[token_id],
                    token_ids=[token_id],
                    source_cell_refs=[
                        {
                            "table_id": table_id,
                            "page": 1,
                            "row": row,
                            "raw_row": row,
                            "col": month,
                        }
                    ],
                )
            )
            raw_tokens.append(
                {
                    "token_id": token_id,
                    "text": value,
                    "bbox": [x0 + 5.0, y0 + 4.0, x0 + 15.0, y0 + 16.0],
                    "confidence": 0.97,
                    "page": 1,
                    "evidence_ids": [token_id],
                }
            )
    table = TableBlock(
        table_id=table_id,
        bbox=[100.0, 100.0, 360.0, 140.0],
        caption="2024年1月至2024年12月还款记录（状态及逾期金额）",
        extraction_layer="scanned_image_line_grid",
        rows=[TableRow(cells=cells) for cells in typed_rows],
        metadata={
            "raw_rows": [[cell.text for cell in cells] for cells in typed_rows],
            "geometry": {
                "geometry_source": "scanned_image_line_grid",
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": cell_statuses,
                "cell_evidence_ids": deepcopy(cell_token_ids),
                "cell_token_ids": cell_token_ids,
                "cell_spans": [{"row": 0, "col": 0, "row_span": 2, "col_span": 1}],
                "row_bands": [
                    {"index": row, "y0": 100.0 + 20.0 * row, "y1": 120.0 + 20.0 * row}
                    for row in range(2)
                ],
                "col_bands": [
                    {"index": col, "x0": 100.0 + 20.0 * col, "x1": 120.0 + 20.0 * col}
                    for col in range(13)
                ],
            },
        },
    )
    result = ParseResult(pages=[PageContent(page_number=1, width=600, height=800, tables=[table])])
    result.entities.domain_specific["_page_evidence_bundles"] = [
        {"page": 1, evidence_key: {"page": 1, "tokens": deepcopy(raw_tokens)}}
    ]
    owner = SimpleNamespace(parse_result=result, evidence_plane=EvidencePlaneBuilder().build(result))
    return owner, result.pages[0], raw_tokens


@pytest.mark.parametrize("evidence_key", ("micro_grid_evidence", "local_structure_evidence"))
def test_monthly_repair_preserves_uncorrected_typed_cells_from_real_evidence_plane(
    evidence_key: str,
) -> None:
    """Ordinary table atoms corroborate raw glyphs; they do not claim repairs."""

    owner, page, raw_tokens = _uncorrected_monthly_table_case(evidence_key)
    original_result = owner.parse_result.model_dump()
    original_atoms = deepcopy(owner.evidence_plane.evidence.text_atoms)
    table_atoms = [
        atom
        for atom in owner.evidence_plane.evidence.text_atoms
        if atom.source_kind == "parse_result_table_cell" and atom.source_refs != ["year-token"]
    ]
    assert len(table_atoms) == 24
    assert all("ocr_correction_action" not in atom.metadata for atom in table_atoms)
    assert all(atom.id not in atom.source_refs for atom in table_atoms)

    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {1})

    assert set(repaired) == {1}
    by_token = {token["token_id"]: token for token in repaired[1]}
    assert set(by_token) == {token["token_id"] for token in raw_tokens}
    for raw in raw_tokens:
        token_id = raw["token_id"]
        expected_source = (
            "exact_native_source_table_status_cell"
            if token_id.startswith("status-")
            else "exact_native_source_table_amount_cell"
        )
        assert by_token[token_id] == {
            "token_id": token_id,
            "content": raw["text"],
            "bbox": raw["bbox"],
            "confidence": 0.97,
            "page": 1,
            "source_logical_page": 1,
            "source_origin_logical_page": 1,
            "coordinate_system": "pdf_points_top_left",
            "source": expected_source,
            "evidence_ids": [token_id],
        }
    assert owner.parse_result.model_dump() == original_result
    assert owner.evidence_plane.evidence.text_atoms == original_atoms


@pytest.mark.parametrize(
    "claim",
    ("incomplete_applied_metadata", "empty_correction_id", "missing_action", "event_without_atom"),
)
def test_monthly_repair_withholds_asserted_correction_without_erasing_uncorrected_siblings(
    claim: str,
) -> None:
    owner, page, raw_tokens = _uncorrected_monthly_table_case()
    target_id = "status-token-01"
    target_atom = next(
        atom
        for atom in owner.evidence_plane.evidence.text_atoms
        if atom.source_kind == "parse_result_table_cell" and atom.source_refs == [target_id]
    )
    if claim == "incomplete_applied_metadata":
        target_atom.metadata["ocr_correction_action"] = "applied"
    elif claim == "empty_correction_id":
        target_atom.metadata["ocr_correction_id"] = ""
    elif claim == "missing_action":
        target_atom.metadata.update({"ocr_original_text": "N", "ocr_corrected_text": "M"})
    else:
        owner.parse_result.parser_info.options["ocr_corrections"] = {
            "events": [
                {
                    "event_id": "uncorroborated-event",
                    "rule_id": "monthly-status-confusable",
                    "source_ref": target_id,
                    "action": "applied",
                    "original": "N",
                    "corrected": "M",
                }
            ]
        }

    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {1})

    assert {token["token_id"] for token in repaired[1]} == {
        raw["token_id"] for raw in raw_tokens if raw["token_id"] != target_id
    }


def _noisy_monthly_source_cell_case(
    shape: str,
) -> tuple[SimpleNamespace, PageContent, dict[str, Any]]:
    """Use saved Ye cell geometry with explicitly simulated raw OCR word boxes.

    The saved Primary artifacts retain the 3x1 N cell and the 福 N cell, not
    their raw OCR word boxes. These independent word observations therefore
    test the supported ownership contract, not an asserted Primary recovery.
    Surrounding status/zero rows are simulated business-document context.
    """

    if shape == "vertical_span":
        table_id, status_row, target_col, year = "pt_17_0", 10, 1, 2020
        y_edges = [56.5, 71, 83.5, 96.5, 109.5, 122.5, 135.5, 148, 161.5, 174.5, 187.5, 200.5, 213.5, 226.5, 241]
        x_edges = [42, 69, 96, 123, 149, 176, 203, 229.5, 256, 283, 309, 336, 362.5, 391]
        target_box = [69.0, 187.5, 96.0, 226.5]
        raw_target_box = [78.0, 190.5, 86.0, 198.5]
        spans = [{"row": 10, "col": 1, "row_span": 3, "col_span": 1}]
    else:
        assert shape == "multiword"
        table_id, status_row, target_col, year = "pt_17_1", 9, 3, 2024
        y_edges = [263, 278, 303, 316, 335, 348, 361, 374, 387, 400, 413, 425.5, 438.5, 451, 464, 477, 490, 503, 516, 529, 542, 557]
        x_edges = [41, 68, 95, 122, 148, 175, 202, 228, 255, 281, 308, 334.5, 361, 388]
        target_box = [122.0, 400.0, 148.0, 413.0]
        raw_target_box = [136.0, 402.0, 144.0, 411.0]
        spans = []
    target_id = f"simulated:{table_id}:status:{target_col}"
    noise_id = f"simulated:{table_id}:watermark:{target_col}"
    cell_bboxes: list[list[Any]] = []
    cell_statuses: list[list[str]] = []
    cell_token_ids: list[list[list[str]]] = []
    typed_rows: list[TableRow] = []
    raw_tokens: list[dict[str, Any]] = []
    for row in range(len(y_edges) - 1):
        boxes, statuses, token_ids_row, cells = [], [], [], []
        for col in range(13):
            bbox = [float(x_edges[col]), float(y_edges[row]), float(x_edges[col + 1]), float(y_edges[row + 1])]
            text, ids, row_span, geometry_status = "", [], 1, "exact"
            if row == status_row - 1 and col:
                text = str(col)
            if col == 0 and row == status_row + int(shape == "multiword"):
                text = str(year)
            covered = shape == "vertical_span" and col == target_col and row in (status_row + 1, status_row + 2)
            if covered:
                bbox, geometry_status = None, "derived"
            elif col and row in (status_row, status_row + 1):
                role = "status" if row == status_row else "amount"
                text = "N" if role == "status" else "0"
                token_id = f"simulated:{table_id}:{role}:{col}"
                ids = [token_id]
                word_box = [bbox[0] + 6.0, bbox[1] + 3.0, bbox[2] - 6.0, bbox[3] - 2.0]
                if row == status_row and col == target_col:
                    bbox, word_box = list(target_box), list(raw_target_box)
                    if shape == "vertical_span":
                        row_span = 3
                    else:
                        text, ids = "福 N", [noise_id, target_id]
                        raw_tokens.append(
                            {
                                "token_id": noise_id,
                                "text": "福",
                                "bbox": [124.0, 402.0, 132.0, 411.0],
                                "confidence": 0.82,
                                "page": 17,
                                "evidence_ids": [noise_id],
                            }
                        )
                raw_tokens.append(
                    {
                        "token_id": token_id,
                        "text": "N" if role == "status" else "0",
                        "bbox": word_box,
                        "confidence": 0.97,
                        "page": 17,
                        "evidence_ids": [token_id],
                    }
                )
            boxes.append(bbox)
            statuses.append(geometry_status)
            token_ids_row.append(ids)
            cells.append(
                CellValue(
                    text=text,
                    bbox=bbox,
                    row_index=row,
                    col_index=col,
                    row_span=row_span,
                    geometry_status=geometry_status,
                    confidence=0.99,
                    geometry_confidence=0.99,
                    token_ids=list(ids),
                    evidence_ids=list(ids),
                    source_cell_refs=[{"table_id": table_id, "page": 17, "raw_row": row, "row": row, "col": col}],
                )
            )
        cell_bboxes.append(boxes)
        cell_statuses.append(statuses)
        cell_token_ids.append(token_ids_row)
        typed_rows.append(TableRow(cells=cells))
    table = TableBlock(
        table_id=table_id,
        bbox=[float(x_edges[0]), float(y_edges[0]), float(x_edges[-1]), float(y_edges[-1])],
        caption=f"{year}年1月至{year}年12月还款记录（状态及逾期金额）",
        extraction_layer="scanned_image_line_grid",
        rows=typed_rows,
        metadata={
            "source_logical_page": 17,
            "source_page": 9,
            "preserve_headers": False,
            "raw_rows": [[cell.text for cell in row.cells] for row in typed_rows],
            "geometry": {
                "geometry_source": "scanned_image_line_grid",
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": cell_statuses,
                "cell_token_ids": cell_token_ids,
                "cell_evidence_ids": deepcopy(cell_token_ids),
                "cell_spans": spans,
                "row_bands": [
                    {"index": row, "y0": float(y_edges[row]), "y1": float(y_edges[row + 1])}
                    for row in range(len(y_edges) - 1)
                ],
                "col_bands": [
                    {"index": col, "x0": float(x_edges[col]), "x1": float(x_edges[col + 1])}
                    for col in range(13)
                ],
            },
        },
    )
    result = ParseResult(pages=[PageContent(page_number=17, width=600, height=800, tables=[table])])
    result.entities.domain_specific["_page_evidence_bundles"] = [
        {"page": 17, "micro_grid_evidence": {"page": 17, "tokens": deepcopy(raw_tokens)}}
    ]
    owner = SimpleNamespace(parse_result=result, evidence_plane=EvidencePlaneBuilder().build(result))
    return owner, result.pages[0], {
        "target_id": target_id,
        "noise_id": noise_id,
        "target_col": target_col,
        "status_row": status_row,
        "year": year,
        "raw_target_bbox": raw_target_box,
        "raw_tokens": raw_tokens,
    }


def _source_word_atom(owner: SimpleNamespace, token_id: str) -> Any:
    return next(
        atom
        for atom in owner.evidence_plane.evidence.text_atoms
        if atom.source_kind == "micro_grid_evidence_token" and atom.source_refs == [token_id]
    )


@pytest.mark.parametrize("shape", ("vertical_span", "multiword"))
def test_monthly_repair_exposes_independent_word_inside_exact_noisy_cell(shape: str) -> None:
    owner, page, case = _noisy_monthly_source_cell_case(shape)
    before_result = owner.parse_result.model_dump()
    before_atoms = deepcopy(owner.evidence_plane.evidence.text_atoms)

    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {17})

    by_id = {token["token_id"]: token for token in repaired[17]}
    expected_ids = {token["token_id"] for token in case["raw_tokens"]} - {case["noise_id"]}
    assert set(by_id) == expected_ids
    assert by_id[case["target_id"]] == {
        "token_id": case["target_id"],
        "content": "N",
        "bbox": case["raw_target_bbox"],
        "confidence": 0.97,
        "page": 17,
        "source_logical_page": 17,
        "source_origin_logical_page": 17,
        "coordinate_system": "pdf_points_top_left",
        "source": "exact_native_source_table_status_cell",
        "evidence_ids": [case["target_id"]],
    }
    assert owner.parse_result.model_dump() == before_result
    assert owner.evidence_plane.evidence.text_atoms == before_atoms


@pytest.mark.parametrize(
    "damage",
    ("missing_span", "overlapping_span", "typed_span", "covered_bbox", "covered_status", "covered_tokens", "horizontal_span"),
)
def test_monthly_repair_rejects_contradictory_vertical_span_ownership(damage: str) -> None:
    owner, page, case = _noisy_monthly_source_cell_case("vertical_span")
    table = page.tables[0]
    geometry = table.metadata["geometry"]
    row, col = case["status_row"], case["target_col"]
    if damage == "missing_span":
        geometry["cell_spans"] = []
    elif damage == "overlapping_span":
        geometry["cell_spans"].append({"row": row + 1, "col": col, "row_span": 2, "col_span": 1})
    elif damage == "typed_span":
        table.rows[row].cells[col].row_span = 2
    elif damage == "covered_bbox":
        geometry["cell_bboxes"][row + 1][col] = [69.0, 200.5, 96.0, 213.5]
    elif damage == "covered_status":
        geometry["cell_geometry_status"][row + 1][col] = "exact"
    elif damage == "covered_tokens":
        geometry["cell_token_ids"][row + 1][col] = [case["target_id"]]
    else:
        geometry["cell_spans"][0]["col_span"] = 2

    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {17})

    ids = {token["token_id"] for token in repaired[17]}
    assert case["target_id"] not in ids
    assert "simulated:pt_17_0:status:3" in ids


@pytest.mark.parametrize(
    "damage",
    ("partial_plane", "partial_evidence_ids", "partial_typed_ids", "duplicate_id", "duplicate_evidence_id", "duplicate_noise_owner", "two_eligible_words", "joined_word", "missing_noise_bbox"),
)
def test_monthly_repair_requires_complete_unambiguous_multiword_cell(damage: str) -> None:
    owner, page, case = _noisy_monthly_source_cell_case("multiword")
    table = page.tables[0]
    geometry = table.metadata["geometry"]
    row, col = case["status_row"], case["target_col"]
    target = table.rows[row].cells[col]
    if damage == "partial_plane":
        noise_atom = _source_word_atom(owner, case["noise_id"])
        owner.evidence_plane.evidence.text_atoms.remove(noise_atom)
    elif damage == "partial_evidence_ids":
        geometry["cell_evidence_ids"][row][col] = [case["target_id"]]
    elif damage == "partial_typed_ids":
        target.token_ids = [case["target_id"]]
    elif damage == "duplicate_id":
        geometry["cell_token_ids"][row][col].append(case["target_id"])
    elif damage == "duplicate_evidence_id":
        geometry["cell_evidence_ids"][row][col].append(case["target_id"])
    elif damage == "duplicate_noise_owner":
        # This incomplete second owner must not disappear before uniqueness
        # is checked merely because it cannot itself emit a repair glyph.
        geometry["cell_token_ids"][0][0] = [case["noise_id"]]
    elif damage == "two_eligible_words":
        _source_word_atom(owner, case["noise_id"]).text = "M"
        target.text = "M N"
    elif damage == "joined_word":
        geometry["cell_token_ids"][row][col] = [case["target_id"]]
        geometry["cell_evidence_ids"][row][col] = [case["target_id"]]
        target.token_ids = target.evidence_ids = [case["target_id"]]
        _source_word_atom(owner, case["target_id"]).text = "福 N"
    else:
        _source_word_atom(owner, case["noise_id"]).bbox = None

    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {17})

    ids = {token["token_id"] for token in repaired[17]}
    assert case["target_id"] not in ids
    assert "simulated:pt_17_1:status:2" in ids


@pytest.mark.parametrize("shape", ("vertical_span", "multiword"))
@pytest.mark.parametrize("damage", ("wrong_raw_page", "wrong_typed_page", "cell_correction_claim", "outside_cell"))
def test_monthly_repair_noisy_cell_cannot_bypass_page_or_raw_value_ownership(shape: str, damage: str) -> None:
    owner, page, case = _noisy_monthly_source_cell_case(shape)
    cell = page.tables[0].rows[case["status_row"]].cells[case["target_col"]]
    if damage == "wrong_raw_page":
        _source_word_atom(owner, case["target_id"]).page_id = "page:0018"
    elif damage == "wrong_typed_page":
        cell.source_cell_refs[0]["page"] = 18
    elif damage == "cell_correction_claim":
        atom = next(
            atom
            for atom in owner.evidence_plane.evidence.text_atoms
            if atom.source_kind == "parse_result_table_cell" and case["target_id"] in atom.source_refs
        )
        atom.metadata["ocr_correction_action"] = "applied"
    else:
        _source_word_atom(owner, case["target_id"]).bbox = [500.0, 600.0, 510.0, 610.0]

    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {17})

    ids = {token["token_id"] for token in repaired[17]}
    assert case["target_id"] not in ids
    assert f"simulated:{page.tables[0].table_id}:status:2" in ids


@pytest.mark.parametrize("raw_row", (10, 11, 12))
def test_monthly_repair_merged_cell_word_box_does_not_move_into_another_status_row(raw_row: int) -> None:
    from docmirror.plugins.credit_report.repayment_grid import (
        _candidate_b_exact_active_source_status_row,
        _coerce_tokens,
    )
    from docmirror.plugins.credit_report.source_table_month_lattice import SourceTableMonthLattice

    owner, page, case = _noisy_monthly_source_cell_case("vertical_span")
    table = page.tables[0]
    geometry = table.metadata["geometry"]
    raw_atom = _source_word_atom(owner, case["target_id"])
    raw_atom.bbox = [78.0, geometry["row_bands"][raw_row]["y0"] + 3.0, 86.0, geometry["row_bands"][raw_row]["y1"] - 2.0]
    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {17})
    target = next(token for token in repaired[17] if token["token_id"] == case["target_id"])
    assert target["bbox"] == list(raw_atom.bbox)
    # This fixture isolates the adapter-to-field handoff. The independently
    # proved status/amount bands do not borrow the merged cell's full height.
    lattice = SourceTableMonthLattice(
        table_id=table.table_id,
        logical_page=17,
        source_logical_page=17,
        source_page=9,
        expected_year=2020,
        year_anchor_row_index=10,
        header_row_index=9,
        status_row_index=10,
        amount_row_index=11,
        year_bbox=(42.0, 187.5, 69.0, 200.5),
        month_bboxes=tuple((col["x0"], 187.5, col["x1"], 200.5) for col in geometry["col_bands"][1:]),
        amount_bboxes=tuple((col["x0"], 200.5, col["x1"], 213.5) for col in geometry["col_bands"][1:]),
        coordinate_system="pdf_points_top_left",
        geometry_source="scanned_image_line_grid",
        provenance=(),
    )

    selected = _candidate_b_exact_active_source_status_row(
        _coerce_tokens([target], page=17),
        {"idx": 10, "source_logical_page": 17},
        source_lattice=lattice,
        active_months=[1],
        status_charset={"N", "M"},
        page=17,
        base_page_height=None,
    )

    assert selected is not None
    assert selected["candidate_b_status_token_months"] == ([1] if raw_row == 10 else [])
    assert selected["candidate_b_status_unresolved_months"] == ([] if raw_row == 10 else [1])


@pytest.mark.parametrize("shape", ("vertical_span", "multiword"))
@pytest.mark.parametrize("copy_projection", (False, True))
def test_monthly_repair_noisy_cell_preserves_registered_raw_origin(shape: str, copy_projection: bool) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import _project_table
    from docmirror.plugins.credit_report.source_table_month_lattice import detached_source_table_geometry_by_page

    owner, page, case = _noisy_monthly_source_cell_case(shape)
    table = page.tables[0]

    def project(box: Any) -> Any:
        return [box[0] * 1.5 + 20.0, box[1] * 1.5 + 40.0, box[2] * 1.5 + 20.0, box[3] * 1.5 + 40.0] if box else None

    projected = _project_table(table, template_id="credit_account_detail", transform=project)
    projected.metadata["coordinate_logical_page"] = 18
    if copy_projection:
        projected = deepcopy(projected)
    canonical_page = SimpleNamespace(page_number=18, source_page_number=9, tables=[projected])
    [geometry] = detached_source_table_geometry_by_page([canonical_page])[18]
    row, col = case["status_row"], case["target_col"]
    assert projected.rows == []
    assert projected.source_cell_objects[row][col].model_dump() == table.rows[row].cells[col].model_dump()
    assert geometry["cell_bboxes"][row][col] == project(table.rows[row].cells[col].bbox)
    assert geometry["source_logical_page"] == 17
    assert geometry["coordinate_logical_page"] == 18
    assert "raw_rows" not in geometry

    repaired = _exact_source_table_repair_tokens_by_page(owner, [canonical_page], {18})

    target = next(token for token in repaired[18] if token["token_id"] == case["target_id"])
    assert target["bbox"] == project(case["raw_target_bbox"])
    assert target["page"] == target["source_logical_page"] == 18
    assert target["source_origin_logical_page"] == 17


def _printed_monthly_anchor_case(
    *, first_anchor_parts: tuple[str, str] | None = None,
) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    """Two same-range business grids have distinct sealed physical anchors."""

    lines, tokens = [], []
    for index, y0 in enumerate((100.0, 300.0)):
        ids = [f"sealed-anchor:{index}:start", f"sealed-anchor:{index}:end"]
        pieces = first_anchor_parts if index == 0 and first_anchor_parts is not None else ("2019年10月至", "2024年9月还款记录")
        for token_id, text, x0, x1 in zip(ids, pieces, (50.0, 164.0), (160.0, 330.0), strict=True):
            tokens.append(
                {
                    "token_id": token_id,
                    "text": text,
                    "bbox": [x0, y0, x1, y0 + 12.0],
                    "confidence": 0.98,
                    "page": 7,
                    "evidence_ids": [token_id],
                }
            )
        lines.append(
            {
                "line_id": f"printed-range-{index}",
                "text": " ".join(pieces),
                "content": " ".join(pieces),
                "bbox": [50.0, y0, 330.0, y0 + 12.0],
                # Deliberately different physical-PDF coordinates. This old
                # field cannot substitute for the raw logical token boxes.
                "source_bbox": [550.0, y0, 830.0, y0 + 12.0],
                "confidence": 0.98,
                "page": 7,
                "evidence_ids": ids,
                "token_ids": ids,
            }
        )
    result = ParseResult(pages=[PageContent(page_number=7, source_page_number=4, width=600, height=800)])
    raw_evidence = {"page": 7, "source_page": 4, "page_width": 600, "page_height": 800, "lines": lines, "tokens": tokens}
    result.entities.domain_specific["_page_evidence_bundles"] = [
        {
            "page": 7,
            "source_page_number": 4,
            "local_structure_evidence": deepcopy(raw_evidence),
            "micro_grid_evidence": deepcopy(raw_evidence),
        }
    ]
    owner = SimpleNamespace(parse_result=result, evidence_plane=EvidencePlaneBuilder().build(result))
    return owner, lines


def _printed_monthly_anchor_context(owner: SimpleNamespace) -> PersonalDetailExtractionContext:
    context = _context(sealed_pages=[_page(7)], canonical_pages=[_page(7)])
    context.parse_result = owner.parse_result
    context.evidence_plane = owner.evidence_plane
    context._construct_static_topology_pages = lambda pages: pages
    return context


def test_printed_monthly_anchor_identity_comes_only_from_complete_raw_evidence() -> None:
    owner, lines = _printed_monthly_anchor_case()
    before = owner.parse_result.model_dump()

    anchors = _authenticated_printed_monthly_anchors(owner)
    annotated = _lines_with_printed_monthly_anchors(lines, logical_page=7, source_page=4, anchors=anchors)

    assert len(anchors) == 2
    for line, stamped in zip(lines, annotated, strict=True):
        assert stamped["printed_anchor_identity"] == {
            "coordinate_system": "pdf_points_top_left",
            "coordinate_plane": "raw_logical_page",
            "source_logical_page": 7,
            "source_page": 4,
            "evidence_ids": sorted(line["evidence_ids"]),
            "bbox": line["bbox"],
            "date_range": [2019, 10, 2024, 9],
        }
        assert stamped["printed_anchor_identity"]["bbox"] != line["source_bbox"]
    assert annotated[0]["printed_anchor_identity"] != annotated[1]["printed_anchor_identity"]
    assert owner.parse_result.model_dump() == before
    assert all("printed_anchor_identity" not in line for line in lines)


def test_printed_monthly_anchor_inventory_uses_actual_raw_bundle_producer_without_detector_grids() -> None:
    from docmirror.input.extraction.scanned_evidence import build_scanned_page_evidence_bundle

    owner, _lines = _printed_monthly_anchor_case()
    raw_tokens = owner.parse_result.entities.domain_specific["_page_evidence_bundles"][0]["micro_grid_evidence"]["tokens"]
    bundle = build_scanned_page_evidence_bundle(
        [
            SimpleNamespace(
                raw_content=token["text"], bbox=token["bbox"], evidence_ids=token["evidence_ids"],
                attrs={"confidence": token["confidence"], "ocr_source": "rapidocr_pdf_logical_page"},
            )
            for token in raw_tokens
        ],
        page=7, source_page=4, page_width=600, page_height=800,
    )
    owner.parse_result.entities.domain_specific["_page_evidence_bundles"] = [bundle]
    owner.evidence_plane = EvidencePlaneBuilder().build(owner.parse_result)
    before = owner.parse_result.model_dump()
    context = _printed_monthly_anchor_context(owner)

    [source_page] = context._build_source_evidence_pages()

    assert context._candidate_b_printed_anchor_inventory_complete is True
    assert len(context._candidate_b_printed_anchor_inventory) == 2
    assert context._candidate_b_printed_anchor_inventory == [line["printed_anchor_identity"] for line in source_page["lines"]]
    assert all(anchor["date_range"] == [2019, 10, 2024, 9] for anchor in context._candidate_b_printed_anchor_inventory)
    assert all(anchor["source_logical_page"] == 7 and anchor["source_page"] == 4 for anchor in context._candidate_b_printed_anchor_inventory)
    assert "micro_grid_structures" not in bundle
    assert not hasattr(context, "_corrected_repayment_micro_grids")
    assert owner.parse_result.model_dump() == before


def test_printed_monthly_anchor_inventory_deduplicates_only_identical_full_raw_anchors() -> None:
    owner, lines = _printed_monthly_anchor_case()
    bundle = owner.parse_result.entities.domain_specific["_page_evidence_bundles"][0]
    for key in ("local_structure_evidence", "micro_grid_evidence"):
        duplicate = deepcopy(bundle[key]["lines"][0])
        duplicate["line_id"] = "another-view-of-the-same-sealed-anchor"
        bundle[key]["lines"].append(duplicate)
    context = _printed_monthly_anchor_context(owner)
    before = owner.parse_result.model_dump()

    [source_page] = context._build_source_evidence_pages()

    inventory = deepcopy(context._candidate_b_printed_anchor_inventory)
    assert context._candidate_b_printed_anchor_inventory_complete is True
    assert len(inventory) == 2 and len(source_page["lines"]) == 3
    assert {tuple(anchor["evidence_ids"]) for anchor in inventory} == {tuple(sorted(line["evidence_ids"])) for line in lines}
    assert inventory[0]["date_range"] == inventory[1]["date_range"]
    assert inventory[0]["bbox"] != inventory[1]["bbox"]
    context._candidate_b_printed_anchor_inventory[0]["bbox"][0] = -999.0
    context._sealed_printed_monthly_anchor_index()
    assert context._candidate_b_printed_anchor_inventory == inventory
    assert owner.parse_result.model_dump() == before


@pytest.mark.parametrize("damage", ("partial_ids", "changed_raw_text", "changed_range", "wrong_bbox", "wrong_page", "new_ocr_ids"))
def test_printed_monthly_anchor_inventory_completeness_cannot_borrow_a_valid_duplicate_view(damage: str) -> None:
    owner, _lines = _printed_monthly_anchor_case()
    bundle = owner.parse_result.entities.domain_specific["_page_evidence_bundles"][0]
    # The local view still authenticates this exact anchor. The original micro
    # view must also pass; a valid twin cannot certify an altered source line.
    line = bundle["micro_grid_evidence"]["lines"][0]
    if damage == "partial_ids":
        line["evidence_ids"] = line["evidence_ids"][:1]
    elif damage == "changed_raw_text":
        line["text"] = line["content"] = "0" + line["text"]
    elif damage == "changed_range":
        line["text"] = line["content"] = line["text"].replace("2019", "2018")
    elif damage == "wrong_bbox":
        line["bbox"] = list(line["source_bbox"])
    elif damage == "wrong_page":
        line["page"] = 8
    else:
        line["evidence_ids"] = line["token_ids"] = ["new-reocr-start", "new-reocr-end"]
    context = _printed_monthly_anchor_context(owner)

    context._build_source_evidence_pages()

    assert len(context._candidate_b_printed_anchor_inventory) == 2
    assert context._candidate_b_printed_anchor_inventory_complete is False


@pytest.mark.parametrize("damage", ("no_bundles", "duplicate_bundle", "missing_views", "malformed_lines", "uncovered_raw_page"))
def test_printed_monthly_anchor_inventory_requires_an_inspectable_closed_raw_population(damage: str) -> None:
    owner, _lines = _printed_monthly_anchor_case()
    bundles = owner.parse_result.entities.domain_specific["_page_evidence_bundles"]
    if damage == "no_bundles":
        bundles.clear()
    elif damage == "duplicate_bundle":
        bundles.append(deepcopy(bundles[0]))
    elif damage == "missing_views":
        bundles[0].pop("local_structure_evidence")
        bundles[0].pop("micro_grid_evidence")
    elif damage == "malformed_lines":
        bundles[0]["micro_grid_evidence"]["lines"] = {"uninspectable": "2019年10月至2024年9月还款记录"}
    else:
        owner.parse_result.pages.append(PageContent(page_number=8, source_page_number=4, width=600, height=800))
    context = _printed_monthly_anchor_context(owner)

    context._sealed_printed_monthly_anchor_index()

    assert context._candidate_b_printed_anchor_inventory_complete is False


@pytest.mark.parametrize("separator", ("", "-", "—", "–", "－", "至", "到", "一"))
def test_printed_monthly_anchor_inventory_uses_existing_accepted_range_separators(separator: str) -> None:
    owner, _lines = _printed_monthly_anchor_case(first_anchor_parts=(f"2019年10月{separator}", "2024年9月还款记录"))
    context = _printed_monthly_anchor_context(owner)

    context._build_source_evidence_pages()

    assert context._candidate_b_printed_anchor_inventory_complete is True
    assert len(context._candidate_b_printed_anchor_inventory) == 2
    assert all(anchor["date_range"] == [2019, 10, 2024, 9] for anchor in context._candidate_b_printed_anchor_inventory)


@pytest.mark.parametrize(
    ("parts", "expected_range"),
    [
        (("2024年9月", "还款记录"), [2024, 9, 2024, 9]),
        (("0 2019年10月至", "2024年9月还款记录"), [2019, 10, 2024, 9]),
        (("2019年10月至", "2024年9月的还款记录"), [2019, 10, 2024, 9]),
        (("2014年9月至", "2024年9月还款记录"), [2014, 9, 2024, 9]),
    ],
    ids=("one-endpoint", "one-prefix-glyph", "existing-de-suffix", "existing-maximum-span"),
)
def test_printed_monthly_anchor_inventory_retains_existing_bounded_range_contract(parts, expected_range) -> None:
    owner, lines = _printed_monthly_anchor_case(first_anchor_parts=parts)
    context = _printed_monthly_anchor_context(owner)

    anchors = context._sealed_printed_monthly_anchor_index()

    assert context._candidate_b_printed_anchor_inventory_complete is True
    assert anchors[(7, tuple(sorted(lines[0]["evidence_ids"])))]["date_range"] == expected_range


@pytest.mark.parametrize(
    "parts",
    [
        ("2019年10月至2021年1月至", "2024年9月还款记录"),
        ("账户2019年10月至", "2024年9月还款记录"),
        ("2019年10月至", "2024年9月还款记录（状态及逾期金额）"),
        ("2014年8月至", "2024年9月还款记录"),
        ("2019年10月~", "2024年9月还款记录"),
        ("2019年13月至", "2024年9月还款记录"),
    ],
    ids=("third-endpoint", "word-prefix", "extra-suffix", "excessive-span", "unaccepted-separator", "bad-month"),
)
def test_printed_monthly_anchor_inventory_never_certifies_a_partial_or_unaccepted_raw_range(parts) -> None:
    # Both raw OCR words and both producer views contain this exact text. This
    # checks the business range decoder, not a convenient raw-text mismatch.
    owner, lines = _printed_monthly_anchor_case(first_anchor_parts=parts)
    context = _printed_monthly_anchor_context(owner)

    anchors = context._sealed_printed_monthly_anchor_index()

    assert (7, tuple(sorted(lines[0]["evidence_ids"]))) not in anchors
    assert len(context._candidate_b_printed_anchor_inventory) == 1
    assert context._candidate_b_printed_anchor_inventory_complete is False


def test_printed_monthly_anchor_inventory_excludes_new_phase_local_ocr_anchors() -> None:
    owner, lines = _printed_monthly_anchor_case()
    context = _printed_monthly_anchor_context(owner)
    replacement = deepcopy(lines[0])
    replacement["text"] = replacement["content"] = "2024年10月至2025年9月还款记录"
    replacement["evidence_ids"] = replacement["token_ids"] = ["new-reocr-start", "new-reocr-end"]
    context._business_repair_active = True
    context._business_repair_evidence_by_page = {7: {"page": 7, "source_page": 4, "lines": [replacement]}}
    context.corrected_evidence_pages = lambda: [deepcopy(context._business_repair_evidence_by_page[7])]

    context._build_source_evidence_pages()

    assert context._candidate_b_printed_anchor_inventory_complete is True
    assert len(context._candidate_b_printed_anchor_inventory) == 2
    assert all(anchor["date_range"] == [2019, 10, 2024, 9] for anchor in context._candidate_b_printed_anchor_inventory)
    assert "new-reocr" not in repr(context._candidate_b_printed_anchor_inventory)


@pytest.mark.parametrize("with_shifted_alias", (False, True))
def test_printed_monthly_anchor_inventory_keeps_distinct_raw_logical_pages_on_one_pdf_page(with_shifted_alias: bool) -> None:
    owner, _lines = _printed_monthly_anchor_case()
    bundles = owner.parse_result.entities.domain_specific["_page_evidence_bundles"]
    second = deepcopy(bundles[0])
    second["page"] = 8
    for key in ("local_structure_evidence", "micro_grid_evidence"):
        view = second[key]
        view["page"] = 8
        for line in view["lines"]:
            line["page"] = 8
            line["evidence_ids"] = [f"{value}:page8" for value in line["evidence_ids"]]
            line["token_ids"] = [f"{value}:page8" for value in line["token_ids"]]
        for token in view["tokens"]:
            token["page"] = 8
            token["token_id"] += ":page8"
            token["evidence_ids"] = [f"{value}:page8" for value in token["evidence_ids"]]
    bundles.append(second)
    owner.parse_result.pages.append(PageContent(page_number=8, source_page_number=4, width=600, height=800))
    owner.evidence_plane = EvidencePlaneBuilder().build(owner.parse_result)
    if with_shifted_alias:
        alias = deepcopy(second["micro_grid_evidence"]["lines"][0])
        alias["source_logical_page"] = 8
        alias["coordinate_status"] = "cross_page_y_shift"
        alias["bbox"][1] += 800.0
        alias["bbox"][3] += 800.0
        bundles[0]["micro_grid_evidence"]["lines"].append(alias)
    context = _printed_monthly_anchor_context(owner)

    context._build_source_evidence_pages()

    inventory = context._candidate_b_printed_anchor_inventory
    assert context._candidate_b_printed_anchor_inventory_complete is True
    assert len(inventory) == 4
    assert {anchor["source_logical_page"] for anchor in inventory} == {7, 8}
    assert {anchor["source_page"] for anchor in inventory} == {4}
    assert len({tuple(anchor["bbox"]) for anchor in inventory}) == 2
    assert len({(anchor["source_logical_page"], tuple(anchor["evidence_ids"])) for anchor in inventory}) == 4


@pytest.mark.parametrize(
    "damage",
    ("missing_raw_token", "duplicate_raw_token", "partial_line_ids", "wrong_raw_page", "wrong_source_page", "duplicate_bundle", "wrong_raw_bbox", "changed_range", "wrong_coordinate_system"),
)
def test_printed_monthly_anchor_proof_rejects_incomplete_or_contradictory_raw_sources(damage: str) -> None:
    owner, lines = _printed_monthly_anchor_case()
    bundle = owner.parse_result.entities.domain_specific["_page_evidence_bundles"][0]
    target_id = lines[0]["evidence_ids"][0]
    if damage == "missing_raw_token":
        owner.evidence_plane.evidence.text_atoms.remove(_source_word_atom(owner, target_id))
    elif damage == "duplicate_raw_token":
        duplicate = deepcopy(_source_word_atom(owner, target_id))
        duplicate.id += ":duplicate"
        owner.evidence_plane.evidence.text_atoms.append(duplicate)
    elif damage == "wrong_raw_page":
        _source_word_atom(owner, target_id).page_id = "page:0008"
    elif damage == "wrong_source_page":
        bundle["micro_grid_evidence"]["source_page"] = 99
    elif damage == "duplicate_bundle":
        owner.parse_result.entities.domain_specific["_page_evidence_bundles"].append(deepcopy(bundle))
    else:
        for key in ("local_structure_evidence", "micro_grid_evidence"):
            line = bundle[key]["lines"][0]
            if damage == "partial_line_ids":
                line["evidence_ids"] = [target_id]
            elif damage == "wrong_raw_bbox":
                line["bbox"] = list(line["source_bbox"])
            elif damage == "changed_range":
                line["text"] = line["content"] = line["text"].replace("2019", "2018")
            else:
                line["coordinate_system"] = "image_pixels"

    anchors = _authenticated_printed_monthly_anchors(owner)

    assert (7, tuple(sorted(lines[0]["evidence_ids"]))) not in anchors
    if damage in {"wrong_source_page", "duplicate_bundle"}:
        assert anchors == {}
    else:
        assert len(anchors) == 1


@pytest.mark.parametrize("damage", ("partial_ids", "transitive_ids", "new_ocr_ids", "wrong_page", "physical_pdf_bbox"))
def test_printed_monthly_anchor_annotation_cannot_borrow_another_identity(damage: str) -> None:
    owner, lines = _printed_monthly_anchor_case()
    anchors = _authenticated_printed_monthly_anchors(owner)
    candidate = deepcopy(lines[0])
    candidate["printed_anchor_identity"] = deepcopy(next(iter(anchors.values())))
    page = 7
    if damage == "partial_ids":
        candidate["evidence_ids"] = candidate["token_ids"] = lines[0]["evidence_ids"][:1]
    elif damage == "transitive_ids":
        candidate["evidence_ids"] = candidate["token_ids"] = [lines[0]["evidence_ids"][0], lines[1]["evidence_ids"][1]]
    elif damage == "new_ocr_ids":
        candidate["evidence_ids"] = candidate["token_ids"] = ["page-reocr-start", "page-reocr-end"]
    elif damage == "wrong_page":
        page = 8
    else:
        candidate["bbox"] = list(candidate["source_bbox"])

    [annotated] = _lines_with_printed_monthly_anchors([candidate], logical_page=page, source_page=4, anchors=anchors)

    assert "printed_anchor_identity" not in annotated


@pytest.mark.parametrize("canonical_only_grid", (False, True))
def test_monthly_canonical_and_detached_inputs_share_authenticated_printed_anchor(
    monkeypatch: pytest.MonkeyPatch,
    canonical_only_grid: bool,
) -> None:
    from docmirror.plugins.credit_report.micro_grid_materialize import _date_range_only_grid
    from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import PBOCCanonicalTemplateAssembler

    owner, _lines = _printed_monthly_anchor_case()
    before = owner.parse_result.model_dump()
    context = _printed_monthly_anchor_context(owner)
    source_pages = context._build_source_evidence_pages()
    original_inventory = deepcopy(context._candidate_b_printed_anchor_inventory)
    assert context._candidate_b_printed_anchor_inventory_complete is True
    source_pages[0]["source_crop_bbox"] = [50.0, 60.0, 950.0, 1260.0]
    assembler = PBOCCanonicalTemplateAssembler(
        owner.parse_result,
        topology=None,
        reading_order_by_logical={7: 1},
        source_evidence_loader=lambda: deepcopy(source_pages),
        issue_owner=context,
    )
    canonical_page, registered_evidence, _audit = assembler._assemble_group(
        {"logical_pages": [7], "canonical_page": 1, "template_id": "credit_account_detail"},
        {7: owner.parse_result.pages[0]},
        {7: source_pages[0]},
        {7: {}},
    )
    context._canonical_layout_projection_cache = SimpleNamespace(pages=[canonical_page])
    context.corrected_evidence_pages = lambda: [deepcopy(registered_evidence)]
    captured: list[dict[str, Any]] = []

    def materialize(detached: dict[str, Any], **_kwargs: Any) -> None:
        captured.append(deepcopy(detached))
        if canonical_only_grid and len(captured) == 1:
            bundle = detached["_page_evidence_bundles"][0]
            anchor = bundle["micro_grid_evidence"]["lines"][0]
            grid = _date_range_only_grid(anchor, page=7, grid_index=0, page_width=600)
            assert grid is not None
            bundle["micro_grid_structures"] = [grid]

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize.materialize_credit_repayment_micro_grids_from_bundles",
        materialize,
    )

    records = context.corrected_repayment_records()

    assert len(records) == (60 if canonical_only_grid else 0)
    assert len(captured) == 2
    assert context._candidate_b_printed_grid_census_required is True
    assert context._candidate_b_printed_anchor_inventory_complete is True
    assert context._candidate_b_printed_anchor_inventory == original_inventory
    assert len(original_inventory) == 2
    assert context._candidate_b_monthly_source_structure_grids == []
    assert len(context._corrected_repayment_micro_grids) == int(canonical_only_grid)
    if canonical_only_grid:
        assert context._corrected_repayment_micro_grids[0]["audit"]["printed_anchor_provenance"] in original_inventory
    canonical_lines = captured[0]["_page_evidence_bundles"][0]["micro_grid_evidence"]["lines"]
    detached_lines = captured[1]["_page_evidence_bundles"][0]["micro_grid_evidence"]["lines"]
    for canonical, detached in zip(canonical_lines, detached_lines, strict=True):
        assert canonical["bbox"] != detached["bbox"]
        assert canonical["printed_anchor_identity"] == detached["printed_anchor_identity"]
        assert canonical["printed_anchor_identity"]["bbox"] == detached["bbox"]
        assert canonical["printed_anchor_identity"]["source_page"] == 4
    assert owner.parse_result.model_dump() == before
