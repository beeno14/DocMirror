from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    _printed_reading_order,
    _printed_reading_order_resolution,
    build_personal_detail_extraction_context,
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
    width = crop[2] - crop[0]
    height = crop[3] - crop[1]
    return SimpleNamespace(
        page_number=logical,
        source_page_number=source,
        width=width,
        height=height,
        coordinate_transform=_transform(
            source=source,
            kind="two_page_spread",
            segment=segment,
            crop=crop,
        ),
        texts=(
            [
                SimpleNamespace(
                    content=footer,
                    bbox=[width * 0.35, height - 20.0, width * 0.65, height - 5.0],
                )
            ]
            if footer
            else []
        ),
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


def test_one_printed_spread_pair_cannot_infer_an_unread_sibling() -> None:
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
    result.pages[1].texts = [
        SimpleNamespace(content="账户 1", bbox=[20.0, 100.0, 100.0, 120.0])
    ]
    topology = PersonalDetailPageTopology(result)

    order, resolution = _printed_reading_order_resolution(result, topology)

    assert order == {10: 10, 20: 20}
    assert resolution["resolved"] is False
    assert resolution["reason"] == "logical_page_footer_unresolved"
    assert resolution["paired_inferred_logical_pages"] == []


def test_two_fully_printed_spreads_prove_consecutive_left_to_right_inference() -> None:
    result = SimpleNamespace(
        pages=[
            _page(10, source=1, segment=0, crop=[0, 0, 300, 800], footer="第 1 页，共 6 页"),
            _page(20, source=1, segment=1, crop=[300, 0, 600, 800], footer="第 2 页，共 6 页"),
            _page(30, source=2, segment=0, crop=[0, 0, 300, 800], footer="第 3 页，共 6 页"),
            _page(40, source=2, segment=1, crop=[300, 0, 600, 800], footer="第 4 页，共 6 页"),
            _page(50, source=3, segment=0, crop=[0, 0, 300, 800], footer="第 5 页，共 6 页"),
            _page(60, source=3, segment=1, crop=[300, 0, 600, 800]),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )
    result.pages[-1].texts = [
        SimpleNamespace(content="账户 6", bbox=[20.0, 100.0, 100.0, 120.0])
    ]
    topology = PersonalDetailPageTopology(result)

    order, resolution = _printed_reading_order_resolution(result, topology)

    assert order == {10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6}
    assert resolution["resolved"] is True
    assert resolution["paired_inferred_logical_pages"] == [60]


def test_different_split_ratios_do_not_prove_one_spread_imposition_profile() -> None:
    result = SimpleNamespace(
        pages=[
            _page(10, source=1, segment=0, crop=[0, 0, 300, 800], footer="第 1 页，共 6 页"),
            _page(20, source=1, segment=1, crop=[300, 0, 600, 800], footer="第 2 页，共 6 页"),
            _page(30, source=2, segment=0, crop=[0, 0, 120, 800], footer="第 3 页，共 6 页"),
            _page(40, source=2, segment=1, crop=[120, 0, 600, 800], footer="第 4 页，共 6 页"),
            _page(50, source=3, segment=0, crop=[0, 0, 300, 800], footer="第 5 页，共 6 页"),
            _page(60, source=3, segment=1, crop=[300, 0, 600, 800]),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )
    result.pages[-1].texts = [
        SimpleNamespace(content="账户 6", bbox=[20.0, 100.0, 100.0, 120.0])
    ]
    topology = PersonalDetailPageTopology(result)

    _order, resolution = _printed_reading_order_resolution(result, topology)

    assert resolution["resolved"] is False
    assert resolution["paired_inferred_logical_pages"] == []


@pytest.mark.parametrize(
    "second_pair_footers",
    (
        ("第 3 页，共 6 页", "第 5 页，共 6 页"),
        ("第 4 页，共 6 页", "第 3 页，共 6 页"),
    ),
)
def test_nonconsecutive_or_reordered_printed_spread_disproves_inference(
    second_pair_footers: tuple[str, str],
) -> None:
    result = SimpleNamespace(
        pages=[
            _page(10, source=1, segment=0, crop=[0, 0, 300, 800], footer="第 1 页，共 6 页"),
            _page(20, source=1, segment=1, crop=[300, 0, 600, 800], footer="第 2 页，共 6 页"),
            _page(30, source=2, segment=0, crop=[0, 0, 300, 800], footer=second_pair_footers[0]),
            _page(40, source=2, segment=1, crop=[300, 0, 600, 800], footer=second_pair_footers[1]),
            _page(50, source=3, segment=0, crop=[0, 0, 300, 800], footer="第 5 页，共 6 页"),
            _page(60, source=3, segment=1, crop=[300, 0, 600, 800]),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )
    result.pages[-1].texts = [SimpleNamespace(content="账户", bbox=[20, 100, 100, 120])]
    topology = PersonalDetailPageTopology(result)

    _order, resolution = _printed_reading_order_resolution(result, topology)

    assert resolution["resolved"] is False
    assert resolution["paired_inferred_logical_pages"] == []


def test_top_bottom_or_right_to_left_spreads_cannot_prove_left_to_right_inference() -> None:
    cases = (
        ([0, 0, 600, 400], [0, 400, 600, 800]),
        ([300, 0, 600, 800], [0, 0, 300, 800]),
    )
    for left_crop, right_crop in cases:
        result = SimpleNamespace(
            pages=[
                _page(10, source=1, segment=0, crop=left_crop, footer="第 1 页，共 6 页"),
                _page(20, source=1, segment=1, crop=right_crop, footer="第 2 页，共 6 页"),
                _page(30, source=2, segment=0, crop=left_crop, footer="第 3 页，共 6 页"),
                _page(40, source=2, segment=1, crop=right_crop, footer="第 4 页，共 6 页"),
                _page(50, source=3, segment=0, crop=left_crop, footer="第 5 页，共 6 页"),
                _page(60, source=3, segment=1, crop=right_crop),
            ],
            entities=SimpleNamespace(domain_specific={}),
        )
        result.pages[-1].texts = [SimpleNamespace(content="账户", bbox=[20, 100, 100, 120])]
        topology = PersonalDetailPageTopology(result)

        _order, resolution = _printed_reading_order_resolution(result, topology)

        assert resolution["resolved"] is False
        assert resolution["paired_inferred_logical_pages"] == []


@pytest.mark.parametrize("marker_bbox", ([200, 100, 400, 120], None))
def test_printed_page_order_rejects_body_or_unlocated_markers(
    marker_bbox: list[int] | None,
) -> None:
    def marker(content: str) -> SimpleNamespace:
        return SimpleNamespace(content=content, **({"bbox": marker_bbox} if marker_bbox else {}))

    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[marker("第2页，共2页")],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                texts=[marker("第1页，共2页")],
            ),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )

    order, resolution = _printed_reading_order_resolution(result)

    assert order == {1: 1, 2: 2}
    assert resolution["resolved"] is False
    assert resolution["authoritative"] is False


def test_printed_page_order_accepts_bottom_full_and_page_only_footers() -> None:
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=20,
                source_page_number=1,
                width=600,
                height=800,
                texts=[SimpleNamespace(content="第1页，共2页", bbox=[200, 770, 400, 790])],
            ),
            SimpleNamespace(
                page_number=10,
                source_page_number=2,
                width=600,
                height=800,
                texts=[SimpleNamespace(content="第2页", bbox=[200, 770, 400, 790])],
            ),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )

    order, resolution = _printed_reading_order_resolution(result)

    assert order == {20: 1, 10: 2}
    assert resolution["resolved"] is True
    assert resolution["authoritative"] is True
    assert resolution["page_only_footer_logical_pages"] == [10]


def test_printed_page_order_accepts_only_a_source_empty_extra_scan_half() -> None:
    blank = _page(30, source=3, segment=0, crop=[0, 0, 300, 800])
    blank.tables = []
    result = SimpleNamespace(
        pages=[
            _page(
                20,
                source=1,
                segment=0,
                crop=[0, 0, 300, 800],
                footer="第 1 页，共 2 页",
            ),
            _page(
                10,
                source=2,
                segment=0,
                crop=[0, 0, 300, 800],
                footer="第 2 页，共 2 页",
            ),
            blank,
        ],
        entities=SimpleNamespace(domain_specific={}),
    )

    order, resolution = _printed_reading_order_resolution(result)

    assert order == {20: 1, 10: 2, 30: 3}
    assert resolution["resolved"] is True
    assert resolution["authoritative"] is True
    assert resolution["basis"] == "complete_unique_printed_page_permutation_with_blank_tail"
    assert resolution["printed_page_by_logical"] == {10: 2, 20: 1}
    assert resolution["blank_logical_pages"] == [30]


@pytest.mark.parametrize(
    "extra_kind",
    (
        "text",
        "table",
        "typed_table_shadowed_by_empty_raw_rows",
        "mapping_row_models_shadowed_by_empty_raw_rows",
        "mapping_top_level_raw_rows",
        "key_value",
        "bundle_structure",
        "region_candidate",
    ),
)
def test_printed_page_order_rejects_nonempty_unprinted_scan_half(
    extra_kind: str,
) -> None:
    extra = _page(3, source=3, segment=0, crop=[0, 0, 300, 800])
    extra.tables = []
    if extra_kind == "text":
        extra.texts = [SimpleNamespace(content="查询记录", bbox=[20, 100, 120, 120])]
    elif extra_kind == "table":
        extra.tables = [
            SimpleNamespace(
                metadata={"raw_rows": [["查询日期"], ["2024-01-01"]]},
                headers=[],
                rows=[],
            )
        ]
    elif extra_kind == "typed_table_shadowed_by_empty_raw_rows":
        extra.tables = [
            SimpleNamespace(
                metadata={"raw_rows": [[""]]},
                headers=[],
                rows=[
                    SimpleNamespace(
                        cells=[SimpleNamespace(text="查询记录")],
                    )
                ],
            )
        ]
    elif extra_kind == "mapping_row_models_shadowed_by_empty_raw_rows":
        extra.tables = [
            {
                "metadata": {"raw_rows": [[""]]},
                "headers": [],
                "row_models": [
                    {
                        "cells": [
                            {
                                "text": "",
                                "cleaned": "",
                                "numeric": 0,
                            }
                        ]
                    }
                ],
            }
        ]
    elif extra_kind == "mapping_top_level_raw_rows":
        extra.tables = [
            {
                "metadata": {},
                "raw_rows": [["查询记录"]],
                "headers": [],
                "rows": [],
            }
        ]
    elif extra_kind == "key_value":
        extra.key_values = [SimpleNamespace(key=" ", value="张三")]
    domain_specific = {}
    if extra_kind == "bundle_structure":
        domain_specific = {
            "_page_evidence_bundles": [
                {
                    "page": 3,
                    "source_page_number": 3,
                    "tokens": [{"text": "还款记录"}],
                }
            ]
        }
    elif extra_kind == "region_candidate":
        domain_specific = {
            "_page_evidence_bundles": [
                {
                    "page": 3,
                    "region_detect": {
                        "region_detect_candidates": [
                            {"kind": "micro_grid", "score": 0.91}
                        ]
                    },
                }
            ]
        }
    result = SimpleNamespace(
        pages=[
            _page(
                1,
                source=1,
                segment=0,
                crop=[0, 0, 300, 800],
                footer="第 1 页，共 2 页",
            ),
            _page(
                2,
                source=2,
                segment=0,
                crop=[0, 0, 300, 800],
                footer="第 2 页，共 2 页",
            ),
            extra,
        ],
        entities=SimpleNamespace(domain_specific=domain_specific),
    )

    order, resolution = _printed_reading_order_resolution(result)

    assert order == {1: 1, 2: 2, 3: 3}
    assert resolution["resolved"] is False
    assert resolution["reason"] == "logical_page_footer_unresolved"
    assert resolution["blank_logical_pages"] == []


def test_empty_spread_sibling_cannot_fill_a_missing_printed_page() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                20,
                source=1,
                segment=0,
                crop=[0, 0, 300, 800],
                footer="第 1 页，共 3 页",
            ),
            _page(10, source=1, segment=1, crop=[300, 0, 600, 800]),
            _page(
                30,
                source=2,
                segment=0,
                crop=[0, 0, 300, 800],
                footer="第 3 页，共 3 页",
            ),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )
    topology = PersonalDetailPageTopology(result)

    order, resolution = _printed_reading_order_resolution(result, topology)

    assert order == {10: 10, 20: 20, 30: 30}
    assert resolution["resolved"] is False
    assert resolution["reason"] == "printed_page_permutation_incomplete"
    assert resolution["paired_inferred_logical_pages"] == []
    assert resolution["blank_logical_pages"] == [10]


@pytest.mark.parametrize(
    ("footers", "reason"),
    (
        (("第 1 页，共 3 页", "第 3 页，共 3 页"), "printed_page_permutation_incomplete"),
        (("第 1 页，共 2 页", "第 1 页，共 2 页"), "printed_page_nonunique"),
    ),
)
def test_blank_scan_half_does_not_hide_an_invalid_printed_permutation(
    footers: tuple[str, str],
    reason: str,
) -> None:
    result = SimpleNamespace(
        pages=[
            _page(1, source=1, segment=0, crop=[0, 0, 300, 800], footer=footers[0]),
            _page(2, source=2, segment=0, crop=[0, 0, 300, 800], footer=footers[1]),
            _page(3, source=3, segment=0, crop=[0, 0, 300, 800]),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )

    order, resolution = _printed_reading_order_resolution(result)

    assert order == {1: 1, 2: 2, 3: 3}
    assert resolution["resolved"] is False
    assert resolution["reason"] == reason
    assert resolution["blank_logical_pages"] == [3]


def test_printed_page_order_rejects_conflicting_bottom_markers() -> None:
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[
                    SimpleNamespace(content="第1页，共2页", bbox=[100, 765, 280, 785]),
                    SimpleNamespace(content="第2页，共2页", bbox=[320, 765, 500, 785]),
                ],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                texts=[SimpleNamespace(content="第2页，共2页", bbox=[200, 770, 400, 790])],
            ),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )

    order, resolution = _printed_reading_order_resolution(result)

    assert order == {1: 1, 2: 2}
    assert resolution["resolved"] is False
    assert resolution["reason"] == "ambiguous_full_footer"


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
    context.parse_result = SimpleNamespace(
        entities=SimpleNamespace(
            domain_specific={
                "_page_evidence_bundles": [
                    {
                        "page": 1,
                        "source_page_number": 1,
                        "local_structure_evidence": {
                            "page": 1,
                            "source_page": 1,
                            "lines": [{"text": "original", "bbox": [100, 100, 180, 120]}],
                        },
                    }
                ]
            }
        )
    )
    context._conserved_corrected_evidence_pages_cache = tuple(pages)
    context._conserved_corrected_evidence_sha256 = "static-test"
    conservation = context.corrected_evidence_conservation_audit()
    assert conservation["valid"] is True
    assert conservation["raw_bundle_count"] == 1
    assert conservation["conserved_page_count"] == 2
    assert conservation["static_replacement_sources"] == [1]
    assert conservation["source_mappings"][0]["status"] == (
        "registered_static_two_way_replacement"
    )


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


def test_conserved_corrected_plane_survives_canonical_filter_and_repair_without_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source_page(logical: int, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            page_number=logical,
            source_page_number=logical,
            width=600.0,
            height=800.0,
            coordinate_transform=_transform(
                source=logical,
                kind="none",
                segment=0,
                crop=[0.0, 0.0, 600.0, 800.0],
            ),
            tables=[],
            texts=[
                SimpleNamespace(
                    content=text,
                    bbox=[20.0, 40.0, 500.0, 70.0],
                    evidence_ids=[f"sealed:{logical}"],
                )
            ],
        )

    def bundle(logical: int, text: str) -> dict[str, object]:
        return {
            "page": logical,
            "source_page_number": logical,
            "local_structure_evidence": {
                "page": logical,
                "source_page": logical,
                "page_width": 600.0,
                "page_height": 800.0,
                "lines": [
                    {
                        "text": text,
                        "bbox": [20.0, 40.0, 500.0, 70.0],
                        "evidence_ids": [f"sealed:{logical}"],
                    }
                ],
            },
        }

    original_texts = {
        1: "信息概要",
        2: "未注册的业务页面内容足够长",
        3: "报告说明",
    }
    result = SimpleNamespace(
        pages=[source_page(logical, text) for logical, text in original_texts.items()],
        entities=SimpleNamespace(
            domain_specific={
                "_page_evidence_bundles": [
                    bundle(logical, text) for logical, text in original_texts.items()
                ]
            }
        ),
    )
    context = build_personal_detail_extraction_context(result)
    monkeypatch.setattr(
        "docmirror.ocr.repair.recognizers.rapidocr_recognize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("page conservation and canonical registration must not call OCR")
        ),
    )

    conserved_before = context.conserved_corrected_evidence_pages()
    discovery_pages = context.corrected_evidence_pages()
    discovery_audit = context.canonical_layout_audit()

    assert [page["page"] for page in conserved_before] == [1, 2, 3]
    assert [page["page"] for page in discovery_pages] == [1, 3]
    assert discovery_audit["corrected_evidence_conservation"]["valid"] is True
    assert discovery_audit["canonical_subset_conservation"]["withheld_logical_pages"] == [2]
    assert discovery_audit["canonical_subset_conservation"]["withheld_pages"][0][
        "localization_issue_code"
    ] == "canonical_page_registration_failed"

    repaired_page = bundle(2, "报告说明")["local_structure_evidence"]
    plan = SimpleNamespace(
        affected_pages=(2,),
        page_evidence={2: repaired_page},
        requires_second_pass=True,
        audit=lambda: {
            "architecture": "schema_triggered_page_repair_v1",
            "unresolved_template_pages": [2],
            "affected_pages": [2],
            "second_schema_pass_required": True,
            "page_decisions": [
                {
                    "logical_page": 2,
                    "mode": "existing_complete_page_evidence",
                    "ocr_invocations": 0,
                    "reason": "business_schema_template_unresolved",
                    "target_count": 1,
                }
            ],
        },
    )

    class Coordinator:
        def __init__(self, _parse_result: object, *, monthly_context: object) -> None:
            assert monthly_context is context
            assert monthly_context.parse_result is _parse_result

        def plan(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return plan

        def resolve_page_evidence(
            self,
            candidate: SimpleNamespace,
            **_kwargs: object,
        ) -> SimpleNamespace:
            return candidate

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.business_repair."
        "BusinessUncertaintyRepairCoordinator",
        Coordinator,
    )

    assert context.prepare_candidate_b_business_repair({}) is True
    repaired_pages = context.corrected_evidence_pages()
    repaired_audit = context.canonical_layout_audit()
    ocr_audit = context.ocr_correction_audit()
    conserved_after = context.conserved_corrected_evidence_pages()

    assert [page["page"] for page in repaired_pages] == [1, 2, 3]
    assert conserved_after == conserved_before
    assert conserved_after[1]["lines"][0]["text"] == original_texts[2]
    assert repaired_audit["canonical_subset_conservation"]["withheld_logical_pages"] == []
    history = repaired_audit["canonical_projection_phase_history"]
    assert [row["phase"] for row in history] == ["discovery", "business_repair"]
    assert history[0]["withheld_logical_pages"] == [2]
    assert history[1]["withheld_logical_pages"] == []
    assert {row["conserved_plane_sha256"] for row in history} == {
        repaired_audit["corrected_evidence_conservation"]["conserved_plane_sha256"]
    }
    assert ocr_audit["corrected_evidence_conservation"]["valid"] is True
    assert ocr_audit["canonical_projection_phase_history"] == history
