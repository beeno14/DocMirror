from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
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
    assert resolver.audit()["recovered_logical_pages"] == []


def test_resolver_recovers_existing_halves_with_the_core_splitter(tmp_path: Path) -> None:
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

    rendered_left = resolver(1)
    rendered_right = resolver(2)

    assert rendered_left is not None and rendered_left["recovered_with_core_splitter"] is True
    assert rendered_right is not None and rendered_right["recovered_with_core_splitter"] is True
    assert rendered_left["coordinate_transform"]["decomposition"]["segment_index"] == 0
    assert rendered_right["coordinate_transform"]["decomposition"]["segment_index"] == 1
    assert resolver.audit()["recovered_logical_pages"] == [1, 2]
