from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    _printed_reading_order,
)
from docmirror.plugins.credit_report.personal_detail_scanned.page_topology import (
    PersonalDetailLogicalPageImageResolver,
    PersonalDetailPageTopology,
)


def _transform(
    *,
    source: int,
    kind: str,
    segment: int,
    crop: list[float],
    rotation: int = 0,
    confidence: float = 0.0,
) -> dict[str, object]:
    return {
        "source_page_number": source,
        "source_crop_bbox": crop,
        "display_width": crop[2] - crop[0],
        "display_height": crop[3] - crop[1],
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "inverse_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "decomposition": {
            "kind": kind,
            "segment_index": segment,
            "selected_rotation": rotation,
            "confidence": confidence,
        },
    }


def _page(
    logical: int,
    *,
    source: int,
    segment: int,
    crop: list[float],
    footer: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical,
        source_page_number=source,
        width=crop[2] - crop[0],
        height=crop[3] - crop[1],
        coordinate_transform=_transform(
            source=source,
            kind="two_page_spread",
            segment=segment,
            crop=crop,
        ),
        texts=[SimpleNamespace(content=footer)] if footer else [],
    )


def test_topology_orders_spread_siblings_by_core_segment_geometry() -> None:
    result = SimpleNamespace(
        pages=[
            _page(20, source=1, segment=0, crop=[0, 0, 300, 800]),
            _page(10, source=1, segment=1, crop=[300, 0, 600, 800]),
        ]
    )

    topology = PersonalDetailPageTopology(result)

    assert topology.ordered_pair((10, 20)) == (20, 10)
    assert topology.audit()["valid"] is True
    assert topology.audit()["double_page_sources"] == 1


def test_topology_preserves_any_number_of_nonoverlapping_core_fragments() -> None:
    result = SimpleNamespace(
        pages=[
            _page(30, source=1, segment=2, crop=[0, 600, 600, 900]),
            _page(10, source=1, segment=0, crop=[0, 0, 600, 300]),
            _page(20, source=1, segment=1, crop=[0, 300, 600, 600]),
        ]
    )

    topology = PersonalDetailPageTopology(result)

    assert topology.ordered_fragments(1) == (10, 20, 30)
    assert topology.audit()["valid"] is True
    assert topology.audit()["fragmented_sources"] == 1


def test_printed_page_inference_uses_geometry_instead_of_logical_ids() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                20,
                source=1,
                segment=0,
                crop=[0, 0, 300, 800],
                footer="第 1 页，共 2 页",
            ),
            _page(10, source=1, segment=1, crop=[300, 0, 600, 800]),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )
    topology = PersonalDetailPageTopology(result)

    assert _printed_reading_order(result, topology) == {20: 1, 10: 2}


def test_ambiguous_spread_geometry_disables_sibling_page_inference() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                20,
                source=1,
                segment=0,
                crop=[0, 0, 350, 800],
                footer="第 1 页，共 2 页",
            ),
            _page(10, source=1, segment=0, crop=[250, 0, 600, 800]),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )
    topology = PersonalDetailPageTopology(result)

    assert topology.audit()["valid"] is False
    assert topology.ordered_pair((10, 20)) is None
    assert _printed_reading_order(result, topology) == {10: 10, 20: 20}


def test_single_page_and_final_nonblank_spread_half_are_valid_layouts() -> None:
    single = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=600,
        height=800,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 600, 800],
        ),
        texts=[],
    )
    partial = _page(2, source=2, segment=0, crop=[0, 0, 300, 800])

    topology = PersonalDetailPageTopology(SimpleNamespace(pages=[single, partial]))
    audit = topology.audit()

    assert audit["valid"] is True
    assert audit["single_page_sources"] == 1
    assert audit["partial_spread_sources"] == 1


def test_resolver_never_uses_an_invalid_stored_transform() -> None:
    page = _page(1, source=1, segment=0, crop=[0, 0, 300, 800])
    page.coordinate_transform["matrix"] = None
    result = SimpleNamespace(pages=[page], file_path="missing.pdf")
    resolver = PersonalDetailLogicalPageImageResolver(result)

    assert resolver(1) is None
    assert resolver.audit()["valid"] is False


def test_resolver_does_not_reconstruct_pages_with_invalid_core_transforms(tmp_path: Path) -> None:
    import fitz

    pdf_path = tmp_path / "spread.pdf"
    with fitz.open() as document:
        source = document.new_page(width=1200, height=850)
        for y in range(40, 810, 55):
            source.draw_rect(fitz.Rect(40, y, 590, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
            source.draw_rect(fitz.Rect(610, y, 1160, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
        document.save(pdf_path)

    left = _page(1, source=1, segment=0, crop=[0, 0, 600, 850])
    right = _page(2, source=1, segment=1, crop=[600, 0, 1200, 850])
    left.coordinate_transform["matrix"] = None
    right.coordinate_transform["matrix"] = None
    result = SimpleNamespace(pages=[left, right], file_path=str(pdf_path))
    resolver = PersonalDetailLogicalPageImageResolver(result, zoom=1.0)

    assert resolver(1) is None
    assert resolver(2) is None
    assert resolver.audit()["valid"] is False


def test_static_subpages_use_split_result_without_ocr_or_footer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fitz

    pdf_path = tmp_path / "unsplit-spread.pdf"
    with fitz.open() as document:
        source = document.new_page(width=1200, height=850)
        for y in range(40, 810, 55):
            source.draw_rect(fitz.Rect(40, y, 590, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
            source.draw_rect(fitz.Rect(610, y, 1160, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
        document.save(pdf_path)

    monkeypatch.setattr(
        "docmirror.ocr.repair.recognizers.rapidocr_recognize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("topology must not call OCR")),
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 1200, 850],
        ),
        texts=[],
    )
    resolver = PersonalDetailLogicalPageImageResolver(
        SimpleNamespace(pages=[page], file_path=str(pdf_path)),
        zoom=1.0,
    )

    recovered = resolver.static_split_slices({1})

    assert [item["segment_index"] for item in recovered] == [0, 1]
    assert all(item["subpage_basis"] == "static_split_validator" for item in recovered)
    assert all(item["page_key"].startswith("source:1:crop:") for item in recovered)
    assert all(item["split_confidence"] >= 0.72 for item in recovered)


def test_static_split_uses_existing_split_results_as_consensus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fitz

    pdf_path = tmp_path / "consensus-spread.pdf"
    with fitz.open() as document:
        source = document.new_page(width=1200, height=850)
        for y in range(40, 810, 55):
            source.draw_rect(fitz.Rect(40, y, 590, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
            source.draw_rect(fitz.Rect(610, y, 1160, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
        document.save(pdf_path)

    pages = [
        SimpleNamespace(
            page_number=1,
            source_page_number=1,
            width=1200,
            height=850,
            coordinate_transform=_transform(
                source=1,
                kind="none",
                segment=0,
                crop=[0, 0, 1200, 850],
            ),
            texts=[],
        ),
        _page(2, source=2, segment=0, crop=[0, 0, 600, 850]),
        _page(3, source=2, segment=1, crop=[600, 0, 1200, 850]),
        _page(4, source=3, segment=0, crop=[0, 0, 600, 850]),
        _page(5, source=3, segment=1, crop=[600, 0, 1200, 850]),
    ]
    observed_boosts: list[float] = []
    from docmirror.input.extraction import page_splitter

    original = page_splitter.decision_from_analyses

    def record_boost(*args, **kwargs):
        observed_boosts.append(float(kwargs.get("consensus_boost") or 0.0))
        return original(*args, **kwargs)

    monkeypatch.setattr(page_splitter, "decision_from_analyses", record_boost)
    monkeypatch.setattr(
        "docmirror.ocr.repair.recognizers.rapidocr_recognize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("topology must not call OCR")),
    )
    resolver = PersonalDetailLogicalPageImageResolver(
        SimpleNamespace(pages=pages, file_path=str(pdf_path)),
        zoom=1.0,
    )

    recovered = resolver.static_split_slices({1})

    assert observed_boosts == [0.05]
    assert [item["segment_index"] for item in recovered] == [0, 1]
    assert all(item["split_consensus_boost"] == 0.05 for item in recovered)


def test_static_split_validator_uses_the_scored_off_centre_gutter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fitz

    pdf_path = tmp_path / "off-centre-spread.pdf"
    with fitz.open() as document:
        source = document.new_page(width=1200, height=850)
        for y in range(40, 810, 55):
            source.draw_rect(fitz.Rect(30, y, 515, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
            source.draw_rect(fitz.Rect(545, y, 1170, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
        document.save(pdf_path)

    monkeypatch.setattr(
        "docmirror.ocr.repair.recognizers.rapidocr_recognize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("topology must not call OCR")),
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 1200, 850],
            confidence=0.9,
        ),
        texts=[],
    )
    resolver = PersonalDetailLogicalPageImageResolver(
        SimpleNamespace(pages=[page], file_path=str(pdf_path)),
        zoom=1.0,
    )

    recovered = resolver.static_split_slices({1})

    assert len(recovered) == 2
    assert recovered[0]["split_position"] == pytest.approx(530, abs=20)
    assert recovered[0]["split_ratio"] == pytest.approx(530 / 1200, abs=0.02)
    assert recovered[0]["page_width"] != pytest.approx(600, abs=10)


def test_static_split_validator_uses_core_rotation_without_ocr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fitz

    pdf_path = tmp_path / "sideways-spread.pdf"
    with fitz.open() as document:
        source = document.new_page(width=850, height=1200)
        for x in range(40, 810, 55):
            source.draw_rect(fitz.Rect(x, 40, x + 12, 590), color=(0, 0, 0), fill=(0, 0, 0))
            source.draw_rect(fitz.Rect(x, 610, x + 12, 1160), color=(0, 0, 0), fill=(0, 0, 0))
        document.save(pdf_path)

    monkeypatch.setattr(
        "docmirror.ocr.repair.recognizers.rapidocr_recognize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("topology must not call OCR")),
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 850, 1200],
            rotation=90,
            confidence=0.9,
        ),
        texts=[],
    )
    resolver = PersonalDetailLogicalPageImageResolver(
        SimpleNamespace(pages=[page], file_path=str(pdf_path)),
        zoom=1.0,
    )

    recovered = resolver.static_split_slices({1})

    assert [item["segment_index"] for item in recovered] == [0, 1]
    assert all(item["selected_rotation"] == 90 for item in recovered)
    assert all(item["coordinate_transform"]["decomposition"]["selected_rotation"] == 90 for item in recovered)


def test_static_split_validator_rejects_one_wide_landscape_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fitz

    pdf_path = tmp_path / "single-landscape.pdf"
    with fitz.open() as document:
        source = document.new_page(width=1200, height=850)
        for y in range(40, 810, 55):
            source.draw_rect(fitz.Rect(40, y, 1160, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
        document.save(pdf_path)

    monkeypatch.setattr(
        "docmirror.ocr.repair.recognizers.rapidocr_recognize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("topology must not call OCR")),
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(source=1, kind="none", segment=0, crop=[0, 0, 1200, 850]),
        texts=[],
    )
    resolver = PersonalDetailLogicalPageImageResolver(
        SimpleNamespace(pages=[page], file_path=str(pdf_path)),
        zoom=1.0,
    )

    assert resolver.static_split_slices({1}) == []


def test_static_topology_partitions_existing_evidence_without_ocr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fitz

    pdf_path = tmp_path / "partition-spread.pdf"
    with fitz.open() as document:
        source = document.new_page(width=1200, height=850)
        for y in range(40, 810, 55):
            source.draw_rect(fitz.Rect(40, y, 590, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
            source.draw_rect(fitz.Rect(610, y, 1160, y + 12), color=(0, 0, 0), fill=(0, 0, 0))
        document.save(pdf_path)

    monkeypatch.setattr(
        "docmirror.ocr.repair.recognizers.rapidocr_recognize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("topology must not call OCR")),
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 1200, 850],
            confidence=0.9,
        ),
        texts=[],
        tables=[],
    )
    result = SimpleNamespace(pages=[page], file_path=str(pdf_path))
    topology = PersonalDetailPageTopology(result)
    context = object.__new__(PersonalDetailExtractionContext)
    context.page_topology = topology
    context._page_image_resolver = PersonalDetailLogicalPageImageResolver(result, topology=topology, zoom=1.0)
    context._frozen_logical_pages = {1: page}
    context._topology_recovery_issues = []
    context.source_page_by_logical = {1: 1}
    context.reading_order_by_logical = {1: 1}

    pages = context._construct_static_topology_pages(
        [
            {
                "page": 1,
                "source_page": 1,
                "page_width": 1200,
                "page_height": 850,
                "lines": [
                    {"text": "left", "bbox": [100, 100, 180, 120]},
                    {"text": "right", "bbox": [700, 100, 790, 120]},
                    {"text": "crossing", "bbox": [580, 100, 620, 120]},
                    {"text": "unlocated"},
                ],
            }
        ]
    )

    assert len(pages) == 2
    assert [line["text"] for line in pages[0]["lines"]] == ["left"]
    assert [line["text"] for line in pages[1]["lines"]] == ["right"]
    assert pages[1]["lines"][0]["bbox"][0] == pytest.approx(100, abs=15)
    assert all(line["text"] != "crossing" for item in pages for line in item["lines"])
    assert context._topology_recovery_issues[0]["code"] == "static_split_boundary_ambiguous"
    assert context._topology_recovery_issues[0]["ambiguous_item_count"] == 2
    assert sorted(context._frozen_logical_pages) == [1, 2]


def test_static_validator_only_renders_core_potential_split_pages() -> None:
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=600,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 600, 850],
            confidence=0.4,
        ),
        texts=[],
        tables=[],
    )
    topology = PersonalDetailPageTopology(SimpleNamespace(pages=[page]))
    context = object.__new__(PersonalDetailExtractionContext)
    context.page_topology = topology
    context._page_image_resolver = SimpleNamespace(
        static_split_slices=lambda _pages: (_ for _ in ()).throw(
            AssertionError("low-confidence non-spread must not be rendered")
        )
    )

    evidence = [{"page": 1, "source_page": 1, "lines": []}]

    assert context._construct_static_topology_pages(evidence) == evidence


def test_static_validator_failure_is_reported_without_changing_original_page() -> None:
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 1200, 850],
            confidence=0.9,
        ),
        texts=[],
        tables=[],
    )
    topology = PersonalDetailPageTopology(SimpleNamespace(pages=[page]))
    context = object.__new__(PersonalDetailExtractionContext)
    context.page_topology = topology
    context._topology_recovery_issues = []
    context._page_image_resolver = SimpleNamespace(
        static_split_slices=lambda _pages: [],
        audit=lambda: {
            "static_split_decisions": [
                {
                    "source_page": 1,
                    "status": "failed",
                    "reason": "source_page_not_renderable",
                }
            ]
        },
    )
    evidence = [{"page": 1, "source_page": 1, "lines": []}]

    assert context._construct_static_topology_pages(evidence) == evidence
    assert context._topology_recovery_issues == [
        {
            "code": "static_page_split_validation_failed",
            "message": "Static page-split validation could not run; the original logical page was preserved.",
            "source_page": 1,
            "status": "failed",
            "reason": "source_page_not_renderable",
        }
    ]


def test_static_validator_records_unrenderable_source_as_terminal_failure() -> None:
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 1200, 850],
            confidence=0.9,
        ),
        texts=[],
    )
    resolver = PersonalDetailLogicalPageImageResolver(
        SimpleNamespace(pages=[page], file_path="missing.pdf")
    )

    assert resolver.static_split_slices({1}) == []
    assert resolver.audit()["static_split_decisions"] == [
        {
            "source_page": 1,
            "status": "failed",
            "reason": "source_page_not_renderable",
        }
    ]


def test_static_split_preserves_unsplit_evidence_when_projection_transform_is_unusable() -> None:
    topology_page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=1200,
        height=850,
        coordinate_transform=_transform(
            source=1,
            kind="none",
            segment=0,
            crop=[0, 0, 1200, 850],
            confidence=0.9,
        ),
        texts=[],
        tables=[],
    )
    topology = PersonalDetailPageTopology(SimpleNamespace(pages=[topology_page]))
    invalid_page = SimpleNamespace(
        coordinate_transform={"inverse_matrix": None},
        tables=[],
    )
    split_pages = [
        {
            "source_page": 1,
            "segment_index": segment,
            "source_crop_bbox": [segment * 600, 0, (segment + 1) * 600, 850],
            "source_to_logical": [[1, 0, -segment * 600], [0, 1, 0], [0, 0, 1]],
        }
        for segment in (0, 1)
    ]
    context = object.__new__(PersonalDetailExtractionContext)
    context.page_topology = topology
    context._page_image_resolver = SimpleNamespace(
        static_split_slices=lambda _pages: split_pages,
        audit=lambda: {"static_split_decisions": []},
    )
    context._frozen_logical_pages = {1: invalid_page}
    context._topology_recovery_issues = []
    context.source_page_by_logical = {1: 1}
    context.reading_order_by_logical = {1: 1}
    evidence = [{"page": 1, "source_page": 1, "lines": [{"text": "preserve"}]}]

    assert context._construct_static_topology_pages(evidence) == evidence
    assert context._topology_recovery_issues[0]["code"] == "static_split_evidence_transform_unusable"
