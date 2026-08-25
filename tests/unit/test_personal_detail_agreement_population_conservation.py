from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.agreement_ocr import (
    canonical_credit_agreement_heading,
    canonical_credit_agreement_section_heading,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _sealed_agreement_population_census,
    _source_completeness_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)


def _text(
    content: str,
    top: float,
    evidence_id: str,
    *,
    scale: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        bbox=[17.0 * scale, top * scale, 143.0 * scale, (top + 9.0) * scale],
        evidence_ids=[evidence_id],
    )


def _page(
    logical_page: int,
    source_page: int,
    *texts: SimpleNamespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=source_page,
        texts=list(texts),
        tables=[],
    )


def _context(
    pages: list[SimpleNamespace],
    *,
    order: dict[int, int] | None = None,
    resolved: bool = True,
    authoritative: bool = True,
) -> SimpleNamespace:
    registered_order = order or {
        page.page_number: index for index, page in enumerate(pages, start=1)
    }
    return SimpleNamespace(
        parse_result=SimpleNamespace(pages=pages),
        _frozen_logical_pages={page.page_number: page for page in pages},
        pages=[],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical=registered_order,
        reading_order_resolution={
            "resolved": resolved,
            "authoritative": authoritative,
        },
        tables_continue=lambda _left, _right: None,
        _personal_detail_extraction_issues=[],
    )


def _dense_context(
    *,
    count: int = 6,
    scale: float = 1.0,
    section_heading: str = "（十二）投信协议信息",
) -> SimpleNamespace:
    first = _page(
        41,
        23,
        _text(section_heading, 20.0, "section", scale=scale),
        *(
            _text(
                ("投伯协议" if sequence % 2 else "授值协议") + str(sequence),
                40.0 + sequence * 18.0,
                f"card:{sequence}",
                scale=scale,
            )
            for sequence in range(1, 4)
        ),
    )
    second = _page(
        9,
        24,
        *(
            _text(
                ("投值协这" if sequence == 4 else "授信协议") + str(sequence),
                30.0 + (sequence - 4) * 18.0,
                f"card:{sequence}",
                scale=scale,
            )
            for sequence in range(4, count + 1)
        ),
        _text("机构查询记录明细", 150.0, "boundary", scale=scale),
    )
    # The sealed list order is intentionally not the printed reading order.
    return _context([second, first], order={41: 1, 9: 2})


def _install_fake_topology(
    context: SimpleNamespace,
    groups: dict[int, tuple[int, ...]],
    geometries: dict[int, SimpleNamespace],
    *,
    valid: bool = True,
) -> None:
    def ordered_pair(logicals: tuple[int, ...]) -> tuple[int, int] | None:
        requested = set(logicals)
        for group in groups.values():
            if len(group) == 2 and set(group) == requested:
                return group
        return None

    context.page_topology = SimpleNamespace(
        audit=lambda: {"valid": valid},
        geometry=lambda logical: geometries.get(logical),
        ordered_fragments=lambda source: groups.get(source, ()),
        ordered_pair=ordered_pair,
    )


def _local_topology_context() -> SimpleNamespace:
    pages = [
        _page(
            50,
            10,
            _text("（六）投信协议信息", 20.0, "section"),
            _text("授信协议1", 40.0, "card:1"),
        ),
        _page(12, 10, _text("投伯协议2", 30.0, "card:2")),
        _page(88, 11, _text("授值协议3", 30.0, "card:3")),
        _page(
            4,
            11,
            _text("投值协议4", 30.0, "card:4"),
            _text("机构查询记录明细", 70.0, "boundary"),
        ),
    ]
    context = _context(
        [pages[2], pages[0], pages[3], pages[1]],
        order={50: 50, 12: 12, 88: 88, 4: 4},
        resolved=False,
        authoritative=False,
    )
    context.source_page_by_logical = {50: 10, 12: 10, 88: 11, 4: 11}
    groups = {10: (50, 12), 11: (88, 4)}
    geometries = {
        logical: SimpleNamespace(
            source_page=source,
            width=200.0,
            height=300.0,
            split_kind="two_page_spread",
            segment_index=segment,
            selected_rotation=0,
            source_crop_bbox=(segment * 100.0, 0.0, (segment + 1) * 100.0, 300.0),
            transform_usable=True,
        )
        for source, group in groups.items()
        for segment, logical in enumerate(group)
    }
    context._test_topology_groups = groups
    context._test_topology_geometries = geometries
    _install_fake_topology(context, groups, geometries)
    return context


def _append_partial_spread_page(context: SimpleNamespace) -> SimpleNamespace:
    page = _page(90, 12, _text("普通正文", 30.0, "outside:prose"))
    context.parse_result.pages.append(page)
    context._frozen_logical_pages[90] = page
    context.source_page_by_logical[90] = 12
    context._test_topology_groups[12] = (90,)
    context._test_topology_geometries[90] = SimpleNamespace(
        source_page=12,
        width=200.0,
        height=300.0,
        split_kind="two_page_spread",
        segment_index=0,
        selected_rotation=0,
        source_crop_bbox=(0.0, 0.0, 100.0, 300.0),
        transform_usable=True,
    )
    return page


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.25])
def test_sealed_agreement_census_uses_authoritative_order_not_page_numbers(
    scale: float,
) -> None:
    census = _sealed_agreement_population_census(_dense_context(scale=scale))

    assert census is not None
    assert census["sequences"] == [1, 2, 3, 4, 5, 6]
    assert set(census["ordinal_observations"]) == set(range(1, 7))
    assert all(
        observation["source_refs"][0]["binding"]
        == "printed_credit_agreement_ordinal"
        for observation in census["ordinal_observations"].values()
    )


def test_frozen_local_topology_can_prove_order_without_identity_fallback() -> None:
    context = _local_topology_context()

    census = _sealed_agreement_population_census(context)

    assert census is not None
    assert census["sequences"] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "heading",
    [
        "授信协议信息",
        "十授伯协议信息",
        "（十二）投值协议信息：",
        "(九)投值协这信息",
    ],
)
def test_agreement_section_heading_uses_only_finite_complete_aliases(
    heading: str,
) -> None:
    assert canonical_credit_agreement_section_heading(heading) == "授信协议信息"


@pytest.mark.parametrize(
    "heading",
    [
        "（十二投信协议信息",
        "十二）投信协议信息",
        "张（十二）投信协议信息",
        "（十二）授偿协议信息",
        "（十二）投信协议",
        "目录（十二）投信协议信息",
    ],
)
def test_agreement_section_heading_near_matches_fail_closed(heading: str) -> None:
    assert canonical_credit_agreement_section_heading(heading) is None


@pytest.mark.parametrize("heading", ["授信协议0", "授信协议01", "授信协议1000"])
def test_non_positive_or_non_canonical_card_ordinals_fail_closed(
    heading: str,
) -> None:
    assert canonical_credit_agreement_heading(heading) is None


def test_dense_heading_census_proves_identity_but_no_business_fields() -> None:
    ledger = _source_completeness_ledger(_dense_context(count=5))

    assert ledger["credit_agreements"] == 5
    assert ledger["credit_agreement_sequence_endpoint"] == 5
    assert ledger["credit_agreement_observed_sequences"] == [1, 2, 3, 4, 5]
    for observation in ledger["credit_agreement_ordinal_observations"].values():
        assert "printed_fields" not in observation
        assert "field_source_refs" not in observation


def test_mutable_unfrozen_raw_pages_never_establish_population() -> None:
    context = _dense_context(count=4)
    del context._frozen_logical_pages

    assert _sealed_agreement_population_census(context) is None


def test_every_unpublished_exact_heading_gets_one_source_local_omission() -> None:
    ledger = _source_completeness_ledger(_dense_context(count=4))
    content = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {"credit_lines": []},
        }
    )

    omissions = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_credit_agreement_record_omitted"
    ]
    assert [issue["target_record_id"] for issue in omissions] == [
        "credit_agreement:1",
        "credit_agreement:2",
        "credit_agreement:3",
        "credit_agreement:4",
    ]
    for sequence, issue in enumerate(omissions, start=1):
        assert issue["observed_value"] == {"credit_agreement_sequence": sequence}
        assert len(issue["source_refs"]) == 1
        ref = issue["source_refs"][0]
        assert ref["source"] == "candidate_b_source_coverage_ledger"
        assert ref["geometry_scope"] == "line"
        assert ref["binding"] == "printed_credit_agreement_ordinal"
        assert ref["binding_quality"] == "printed_credit_agreement_ordinal"
        assert ref["sequence"] == sequence
        assert ref["logical_page"] > 0
        assert ref["source_page"] > 0
        assert len(ref["bbox"]) == 4
        assert ref["evidence_ids"]
    assert not any(
        issue.get("issue_code") == "source_credit_agreement_field_omitted"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


def _single_page_context(*texts: SimpleNamespace) -> SimpleNamespace:
    return _context([_page(7, 4, *texts)])


@pytest.mark.parametrize(
    "texts",
    [
        # TOC/prose cannot open a registered business envelope.
        (
            _text("目录（六）授信协议信息", 10.0, "toc"),
            _text("详见授信协议1", 30.0, "prose"),
            _text("机构查询记录明细", 50.0, "boundary"),
        ),
        # A second exact start makes section ownership ambiguous.
        (
            _text("（六）授信协议信息", 10.0, "start:1"),
            _text("授信协议1", 30.0, "card:1"),
            _text("六投信协议信息", 50.0, "start:2"),
            _text("机构查询记录明细", 70.0, "boundary"),
        ),
        # Without an exact following boundary, the envelope is not sealed.
        (
            _text("（六）授信协议信息", 10.0, "start"),
            _text("授信协议1", 30.0, "card:1"),
            _text("后续未知章节", 50.0, "unknown"),
        ),
        # Sparse/high ordinals cannot imply intervening cards.
        (
            _text("（六）授信协议信息", 10.0, "start"),
            _text("授信协议1", 30.0, "card:1"),
            _text("授信协议2", 50.0, "card:2"),
            _text("授信协议91", 70.0, "card:91"),
            _text("机构查询记录明细", 90.0, "boundary"),
        ),
        # Duplicate card headings are identity conflicts even with new OCR IDs.
        (
            _text("（六）授信协议信息", 10.0, "start"),
            _text("授信协议1", 30.0, "card:1:a"),
            _text("投信协议1", 50.0, "card:1:b"),
            _text("机构查询记录明细", 70.0, "boundary"),
        ),
        # A card outside the agreement interval is a cross-section conflict.
        (
            _text("（六）授信协议信息", 10.0, "start"),
            _text("授信协议1", 30.0, "card:1"),
            _text("机构查询记录明细", 50.0, "boundary"),
            _text("授信协议2", 70.0, "card:2"),
        ),
    ],
)
def test_unsealed_or_ambiguous_source_populations_fail_closed(
    texts: tuple[SimpleNamespace, ...],
) -> None:
    assert _sealed_agreement_population_census(_single_page_context(*texts)) is None


def test_replayed_structural_evidence_fails_closed() -> None:
    context = _single_page_context(
        _text("（六）授信协议信息", 10.0, "replayed"),
        _text("授信协议1", 30.0, "replayed"),
        _text("机构查询记录明细", 50.0, "boundary"),
    )

    assert _sealed_agreement_population_census(context) is None


def test_structural_evidence_replayed_by_prose_fails_closed() -> None:
    context = _dense_context(count=4)
    page = context._frozen_logical_pages[41]
    page.texts.append(_text("普通正文", 140.0, "card:1"))

    assert _sealed_agreement_population_census(context) is None


@pytest.mark.parametrize(
    ("resolved", "authoritative"),
    [(False, True), (True, False), (False, False)],
)
def test_non_authoritative_reading_order_never_counts_agreement_headings(
    resolved: bool,
    authoritative: bool,
) -> None:
    context = _dense_context(count=4)
    context.reading_order_resolution = {
        "resolved": resolved,
        "authoritative": authoritative,
    }

    assert _sealed_agreement_population_census(context) is None


def test_partial_or_duplicate_reading_order_fails_closed() -> None:
    partial = _dense_context(count=4)
    partial.reading_order_by_logical = {41: 1}
    duplicate = _dense_context(count=4)
    duplicate.reading_order_by_logical = {41: 3, 9: 3}

    assert _sealed_agreement_population_census(partial) is None
    assert _sealed_agreement_population_census(duplicate) is None


@pytest.mark.parametrize(
    "order",
    [
        {41: 1, 9: 2, 99: 3},
        {41: 2, 9: 4},
    ],
)
def test_extra_or_gapped_global_reading_order_fails_closed(
    order: dict[int, int],
) -> None:
    context = _dense_context(count=4)
    context.reading_order_by_logical = order

    assert _sealed_agreement_population_census(context) is None


def test_local_topology_rejects_an_incomplete_two_up_pair() -> None:
    context = _local_topology_context()
    context._frozen_logical_pages.pop(12)
    context.parse_result.pages = [
        page for page in context.parse_result.pages if page.page_number != 12
    ]

    assert _sealed_agreement_population_census(context) is None


def test_local_topology_allows_a_partial_spread_after_the_closed_envelope() -> None:
    context = _local_topology_context()
    _append_partial_spread_page(context)

    census = _sealed_agreement_population_census(context)

    assert census is not None
    assert census["sequences"] == [1, 2, 3, 4]


def test_local_topology_rejects_a_partial_spread_inside_the_envelope() -> None:
    context = _local_topology_context()
    partial_page = _append_partial_spread_page(context)
    complete_boundary_page = context._frozen_logical_pages[4]
    complete_boundary_page.texts = [
        text
        for text in complete_boundary_page.texts
        if text.content != "机构查询记录明细"
    ]
    partial_page.texts.append(_text("机构查询记录明细", 70.0, "partial:boundary"))

    assert _sealed_agreement_population_census(context) is None


def test_local_topology_rejects_a_partial_spread_before_the_envelope() -> None:
    context = _local_topology_context()
    partial_page = _append_partial_spread_page(context)
    context._test_topology_groups[9] = context._test_topology_groups.pop(12)
    partial_page.source_page_number = 9
    context.source_page_by_logical[90] = 9
    context._test_topology_geometries[90].source_page = 9

    assert _sealed_agreement_population_census(context) is None


@pytest.mark.parametrize("defect", ["overlapping_crop", "unusable_transform"])
def test_local_topology_rejects_crop_or_transform_defects(defect: str) -> None:
    context = _local_topology_context()
    geometry = context._test_topology_geometries[12]
    if defect == "overlapping_crop":
        geometry.source_crop_bbox = (90.0, 0.0, 190.0, 300.0)
    else:
        geometry.transform_usable = False

    assert _sealed_agreement_population_census(context) is None


def test_local_topology_rejects_duplicate_segments() -> None:
    context = _local_topology_context()
    context._test_topology_geometries[12].segment_index = 0

    assert _sealed_agreement_population_census(context) is None


def test_local_topology_rejects_a_physical_source_gap() -> None:
    context = _local_topology_context()
    groups = context._test_topology_groups
    geometries = context._test_topology_geometries
    groups[12] = groups.pop(11)
    for logical in groups[12]:
        context._frozen_logical_pages[logical].source_page_number = 12
        context.source_page_by_logical[logical] = 12
        geometries[logical].source_page = 12

    assert _sealed_agreement_population_census(context) is None


def test_local_topology_rejects_physical_source_reversal() -> None:
    context = _local_topology_context()
    groups = context._test_topology_groups
    geometries = context._test_topology_geometries
    groups.clear()
    groups.update({12: (50, 12), 10: (88, 4)})
    for logical in (50, 12):
        context._frozen_logical_pages[logical].source_page_number = 12
        context.source_page_by_logical[logical] = 12
        geometries[logical].source_page = 12
    for logical in (88, 4):
        context._frozen_logical_pages[logical].source_page_number = 10
        context.source_page_by_logical[logical] = 10
        geometries[logical].source_page = 10

    assert _sealed_agreement_population_census(context) is None


def test_local_topology_rejects_an_extra_registered_logical_page() -> None:
    context = _local_topology_context()
    context._test_topology_groups[10] = (50, 12, 99)

    assert _sealed_agreement_population_census(context) is None


@pytest.mark.parametrize("defect", ["unfrozen", "invalid_audit"])
def test_local_topology_must_be_frozen_and_valid(defect: str) -> None:
    context = _local_topology_context()
    if defect == "unfrozen":
        context._frozen_logical_pages = {}
    else:
        _install_fake_topology(
            context,
            context._test_topology_groups,
            context._test_topology_geometries,
            valid=False,
        )

    assert _sealed_agreement_population_census(context) is None


@pytest.mark.parametrize("defect", ["bbox", "evidence", "overlap"])
def test_unsealed_heading_geometry_or_evidence_fails_closed(defect: str) -> None:
    context = _dense_context(count=4)
    pages = list(context._frozen_logical_pages.values())
    first = next(page for page in pages if page.page_number == 41)
    if defect == "bbox":
        first.texts[1].bbox = [10.0, 30.0, float("nan"), 40.0]
    elif defect == "evidence":
        first.texts[1].evidence_ids = []
    else:
        first.texts[1].bbox = deepcopy(first.texts[0].bbox)

    assert _sealed_agreement_population_census(context) is None
