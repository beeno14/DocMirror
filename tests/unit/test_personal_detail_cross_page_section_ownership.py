from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    PBOCCanonicalTemplateAssembler,
    _sealed_account_card_continuation_proved,
)
from docmirror.plugins.credit_report.personal_detail_scanned.page_topology import (
    PersonalDetailPageTopology,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntity,
    CreditReportEntityContext,
    CreditReportUnit,
    EntityTransitionDecision,
    TransitionHypothesis,
)


def _line(text: str, bbox: list[float], *, owner: str | None = None) -> dict[str, object]:
    return {
        "text": text,
        "bbox": bbox,
        "evidence_ids": [owner or f"line:{bbox[0]}:{bbox[1]}:{text}"],
    }


def _table(
    table_id: str,
    *,
    top: float,
    header: tuple[str, ...],
    values: tuple[tuple[str, ...], ...],
    widths: tuple[float, ...] | None = None,
) -> SimpleNamespace:
    assert values and all(len(row) == len(header) for row in values)
    widths = widths or tuple(float(index + 2) for index in range(len(header)))
    assert len(widths) == len(header)
    left = 30.0
    scale = 500.0 / sum(widths)
    bands: list[tuple[float, float]] = []
    for width in widths:
        right = left + width * scale
        bands.append((left, right))
        left = right
    rows = [header, *values]
    row_height = 20.0
    cell_bboxes = [
        [
            [x0, top + row * row_height, x1, top + (row + 1) * row_height]
            for x0, x1 in bands
        ]
        for row in range(len(rows))
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[30.0, top, 530.0, top + len(rows) * row_height],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": [list(row) for row in rows],
            "geometry": {
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": [
                    ["exact"] * len(header) for _row in rows
                ],
                "cell_evidence_ids": [
                    [
                        [f"{table_id}:r{row}:c{column}"]
                        for column in range(len(header))
                    ]
                    for row in range(len(rows))
                ],
                "coordinate_system": "pdf_points_top_left",
            },
        },
    )


def _page(number: int, *tables: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=number,
        source_page_number=number,
        width=560.0,
        height=700.0,
        tables=list(tables),
        texts=[],
    )


def _evidence(number: int, *lines: dict[str, object]) -> dict[str, object]:
    return {
        "page": number,
        "source_page": number,
        "page_width": 560.0,
        "page_height": 700.0,
        "lines": [*lines, _line(f"第{number}页，共9页", [220.0, 660.0, 340.0, 680.0])],
    }


def _build(
    pages: list[SimpleNamespace],
    evidence: list[dict[str, object]],
    *,
    continuations: set[tuple[str, str]] | None = None,
    topology: object | None = None,
    reading_order_resolution: dict[str, object] | None = None,
    entity_context: object | None = None,
    force_table_continuation_false: bool = False,
):
    topology = topology or SimpleNamespace(geometry=lambda _logical: None)
    owner = SimpleNamespace(
        tables_continue=lambda left, right: False
        if force_table_continuation_false
        else (left, right) in (continuations or set()),
        reading_order_resolution=reading_order_resolution,
        page_topology=topology,
        page_topology_audit=(
            lambda: {
                **topology.audit(),
                "topology_frozen_before_reocr": True,
            }
        )
        if callable(getattr(topology, "audit", None))
        else None,
        entity_context=entity_context,
    )
    return PBOCCanonicalTemplateAssembler(
        SimpleNamespace(pages=pages),
        topology=topology,
        reading_order_by_logical={
            page.page_number: index for index, page in enumerate(pages, start=1)
        },
        source_evidence_loader=lambda: evidence,
        issue_owner=owner,
    ).build()


PUBLIC_HEADER = ("编号", "主管税务机关", "欠税总额", "欠税统计日期")
INQUIRY_HEADER = ("编号", "查询日期", "查询机构", "查询原因")
AGREEMENT_HEADER = (
    "管理机构",
    "授信协议标识",
    "生效日期",
    "到期日期",
    "授信额度用途",
)


def test_top_level_heading_cannot_swallow_later_exact_subsection_table() -> None:
    public = _table(
        "public",
        top=70.0,
        header=PUBLIC_HEADER,
        values=(("1", "主管机关", "100", "2025.01.01"),),
        widths=(2.0, 8.0, 4.0, 6.0),
    )
    inquiry = _table(
        "inquiry",
        top=260.0,
        header=INQUIRY_HEADER,
        values=(("1", "2025.02.03", "机构甲", "贷后管理"),),
        widths=(2.0, 6.0, 9.0, 7.0),
    )
    page = _page(1, public, inquiry)
    evidence = _evidence(
        1,
        _line("公共信息明细", [180.0, 25.0, 360.0, 50.0]),
        _line("欠税记录", [200.0, 52.0, 340.0, 67.0]),
        _line("查询记录", [200.0, 220.0, 340.0, 240.0]),
    )

    projection = _build([page], [evidence])

    assert projection.unresolved_pages == ()
    projected = projection.pages[0]
    assert projected.canonical_template_id == "mixed_pboc_sections"
    assert {
        table.table_id: table.metadata["canonical_template_id"]
        for table in projected.tables
    } == {
        "public": "public_information",
        "inquiry": "annotations_and_inquiries",
    }


def _agreement_page_case() -> tuple[list[SimpleNamespace], list[dict[str, object]]]:
    previous_tables = [
        _table(
            f"agreement-{sequence}",
            top=70.0 + (sequence - 1) * 120.0,
            header=AGREEMENT_HEADER,
            values=(("机构甲", f"A{sequence}", "2024.01.01", "长期", "贷款"),),
        )
        for sequence in (1, 2)
    ]
    current_tables = [
        _table(
            "agreement-3",
            top=65.0,
            header=("管理机构", "授信协议标识", "生效日期附注", "到期日期", "授信额度用途"),
            values=(("机构乙", "A3", "2024.02.01", "长期", "贷款"),),
        ),
        _table(
            "agreement-4",
            top=175.0,
            header=AGREEMENT_HEADER,
            values=(("机构丙", "A4", "2024.03.01", "长期", "贷款"),),
        ),
        _table(
            "inquiry-after-agreement",
            top=330.0,
            header=INQUIRY_HEADER,
            values=(("1", "2025.03.04", "机构丁", "贷后管理"),),
        ),
    ]
    pages = [_page(3, *previous_tables), _page(4, *current_tables)]
    evidence = [
        _evidence(
            3,
            _line("授信协议信息", [170.0, 15.0, 370.0, 38.0]),
            _line("授信协议1", [35.0, 50.0, 130.0, 68.0]),
            _line("授信协议2", [35.0, 170.0, 130.0, 188.0]),
        ),
        _evidence(
            4,
            _line("授信协议3", [35.0, 48.0, 130.0, 67.0]),
            _line("授信协议4", [35.0, 158.0, 130.0, 177.0]),
            _line("查询记录", [190.0, 290.0, 350.0, 315.0]),
        ),
    ]
    return pages, evidence


def test_dense_numbered_agreement_cards_and_inquiry_get_table_local_owners() -> None:
    pages, evidence = _agreement_page_case()

    projection = _build(pages, evidence)

    current = next(page for page in projection.pages if page.page_number == 4)
    assert current.canonical_template_id == "mixed_pboc_sections"
    assert {
        table.table_id: table.metadata["canonical_template_id"]
        for table in current.tables
    } == {
        "agreement-3": "credit_agreement",
        "agreement-4": "credit_agreement",
        "inquiry-after-agreement": "annotations_and_inquiries",
    }
    assert current.tables[0].metadata["canonical_section_owner"]["header_binding"] == (
        "preceding_exact_agreement_role_map"
    )


@pytest.mark.parametrize("defect", ["ordinal_gap", "unsealed_anchor", "printed_gap"])
def test_agreement_continuation_requires_dense_sealed_consecutive_proof(defect: str) -> None:
    pages, evidence = _agreement_page_case()
    if defect == "ordinal_gap":
        evidence[1]["lines"][0]["text"] = "授信协议5"
    elif defect == "unsealed_anchor":
        evidence[1]["lines"][0]["evidence_ids"] = []
    else:
        evidence[1]["lines"][-1]["text"] = "第6页，共9页"

    projection = _build(pages, evidence)

    current = next(page for page in projection.pages if page.page_number == 4)
    # The bad agreement continuation is withheld, but its failure cannot erase
    # the independently owned exact inquiry table lower on the same page.
    assert [table.table_id for table in current.tables] == [
        "inquiry-after-agreement"
    ]
    assert current.tables[0].metadata["canonical_template_id"] == (
        "annotations_and_inquiries"
    )


def _inquiry_continuation_case():
    public = _table(
        "public-first",
        top=60.0,
        header=PUBLIC_HEADER,
        values=(("1", "主管机关", "0", "2025.01.01"),),
    )
    first_inquiry = _table(
        "inquiry-first",
        top=240.0,
        header=INQUIRY_HEADER,
        values=(("1", "2025.02.01", "机构甲", "贷后管理"),),
    )
    second_inquiry = _table(
        "inquiry-second",
        top=50.0,
        header=("", "", "", ""),
        values=(
            ("2", "2025.01.20", "机构乙", "贷款审批"),
            ("3", "2025.01.10", "机构丙", "贷后管理"),
        ),
    )
    # Remove the synthetic blank header row: this is a true headerless table.
    second_inquiry.metadata["raw_rows"] = second_inquiry.metadata["raw_rows"][1:]
    for key in ("cell_bboxes", "cell_geometry_status", "cell_evidence_ids"):
        second_inquiry.metadata["geometry"][key] = second_inquiry.metadata["geometry"][key][1:]
    third_inquiry = _table(
        "inquiry-third",
        top=45.0,
        header=("", "", "", ""),
        values=(("4", "2025.01.02", "机构丁", "贷后管理"),),
    )
    third_inquiry.metadata["raw_rows"] = third_inquiry.metadata["raw_rows"][1:]
    for key in ("cell_bboxes", "cell_geometry_status", "cell_evidence_ids"):
        third_inquiry.metadata["geometry"][key] = third_inquiry.metadata["geometry"][key][1:]
    pages = [
        _page(1, public, first_inquiry),
        _page(2, second_inquiry),
        _page(3, third_inquiry),
    ]
    evidence = [
        _evidence(
            1,
            _line("公共信息明细", [180.0, 15.0, 360.0, 38.0]),
            _line("欠税记录", [200.0, 40.0, 340.0, 55.0]),
            _line("查询记录", [200.0, 200.0, 340.0, 220.0]),
        ),
        _evidence(2),
        _evidence(3),
    ]
    return pages, evidence


def _append_terminal_border_band(
    table: SimpleNamespace,
    *,
    sealed: bool = False,
) -> None:
    geometry = table.metadata["geometry"]
    prior_boxes = geometry["cell_bboxes"][-1]
    top = max(box[3] for box in prior_boxes)
    bottom = top + 4.0
    geometry["cell_bboxes"].append(
        [[box[0], top, box[2], bottom] for box in prior_boxes]
    )
    geometry["cell_geometry_status"].append(["exact"] * len(prior_boxes))
    geometry["cell_evidence_ids"].append(
        [
            [f"{table.table_id}:terminal:{column}"] if sealed else []
            for column in range(len(prior_boxes))
        ]
    )
    table.metadata["raw_rows"].append([""] * len(prior_boxes))
    table.bbox[3] = bottom


def test_exact_inquiry_table_owner_propagates_across_variable_length_pages() -> None:
    pages, evidence = _inquiry_continuation_case()

    projection = _build(
        pages,
        evidence,
        continuations={
            ("inquiry-first", "inquiry-second"),
            ("inquiry-second", "inquiry-third"),
        },
    )

    assert projection.unresolved_pages == ()
    continued = [page for page in projection.pages if page.page_number in {2, 3}]
    assert len(continued) == 2
    assert all(page.canonical_template_id == "mixed_pboc_sections" for page in continued)
    assert all(
        table.metadata["canonical_section_owner"]["binding"]
        == "authoritative_prior_inquiry_table_continuation"
        for page in continued
        for table in page.tables
    )


def _spread_topology_page(
    logical: int,
    *,
    source: int,
    segment: int,
    crop: list[float],
    matrix: list[list[float]] | None = None,
) -> SimpleNamespace:
    matrix = matrix or [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return SimpleNamespace(
        page_number=logical,
        source_page_number=source,
        width=crop[2] - crop[0],
        height=crop[3] - crop[1],
        coordinate_transform={
            "source_page_number": source,
            "source_crop_bbox": crop,
            "display_width": crop[2] - crop[0],
            "display_height": crop[3] - crop[1],
            "matrix": matrix,
            "inverse_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "decomposition": {
                "kind": "two_page_spread",
                "segment_index": segment,
                "selected_rotation": 0,
                "confidence": 0.99,
            },
        },
        tables=[],
        texts=[],
    )


def _paired_inferred_inquiry_case(
    defect: str = "",
) -> tuple[
    list[SimpleNamespace],
    list[dict[str, object]],
    PersonalDetailPageTopology,
    dict[str, object],
    set[tuple[str, str]],
]:
    pages, evidence = _inquiry_continuation_case()
    pages = pages[:2]
    evidence = evidence[:2]
    for page, logical in zip(pages, (50, 60), strict=True):
        page.page_number = logical
        page.source_page_number = 3
    for page_evidence, logical in zip(evidence, (50, 60), strict=True):
        page_evidence["page"] = logical
        page_evidence["source_page"] = 3
    evidence[0]["lines"][-1]["text"] = "第5页，共9页"
    evidence[1]["lines"] = []

    target_left_crop = [0.0, 0.0, 300.0, 800.0]
    target_right_crop = [300.0, 0.0, 600.0, 800.0]
    target_left_source = 3
    target_right_source = 3
    target_left_segment = 0
    target_right_segment = 1
    target_right_matrix = None
    if defect == "different_source_pair":
        target_right_source = 4
    elif defect == "profile_mismatch":
        target_left_crop = [0.0, 0.0, 180.0, 800.0]
        target_right_crop = [180.0, 0.0, 600.0, 800.0]
    elif defect == "transform_mismatch":
        target_right_matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    elif defect == "reverse_segment":
        target_left_crop, target_right_crop = (
            target_right_crop,
            target_left_crop,
        )
        target_left_segment, target_right_segment = (1, 0)

    topology = PersonalDetailPageTopology(
        SimpleNamespace(
            pages=[
                _spread_topology_page(
                    10,
                    source=1,
                    segment=0,
                    crop=[0.0, 0.0, 300.0, 800.0],
                ),
                _spread_topology_page(
                    20,
                    source=1,
                    segment=1,
                    crop=[300.0, 0.0, 600.0, 800.0],
                ),
                _spread_topology_page(
                    30,
                    source=2,
                    segment=0,
                    crop=[0.0, 0.0, 300.0, 800.0],
                ),
                _spread_topology_page(
                    40,
                    source=2,
                    segment=1,
                    crop=[300.0, 0.0, 600.0, 800.0],
                ),
                _spread_topology_page(
                    50,
                    source=target_left_source,
                    segment=target_left_segment,
                    crop=target_left_crop,
                ),
                _spread_topology_page(
                    60,
                    source=target_right_source,
                    segment=target_right_segment,
                    crop=target_right_crop,
                    matrix=target_right_matrix,
                ),
            ]
        )
    )
    resolution: dict[str, object] = {
        "resolved": False,
        "authoritative": False,
        "identity_fallback": True,
        "printed_total": 9,
        "printed_page_by_logical": {
            10: 1,
            20: 2,
            30: 3,
            40: 4,
            50: 5,
            60: 6,
        },
        "observed_logical_pages": [10, 20, 30, 40, 50, 60],
        "full_footer_logical_pages": [10, 20, 30, 40, 50],
        "page_only_footer_logical_pages": [],
        "paired_inferred_logical_pages": [60],
        "blank_logical_pages": [],
        "unresolved_logical_pages": [],
    }
    continuations = {("inquiry-first", "inquiry-second")}
    if defect == "nonconsecutive_mapping":
        resolution["printed_page_by_logical"][60] = 7
    elif defect == "current_only_fallback":
        resolution["paired_inferred_logical_pages"] = []
    elif defect == "missing_edge":
        continuations = set()
    elif defect == "conflicting_footer":
        evidence[1]["lines"] = [
            _line("第8页，共9页", [220.0, 660.0, 340.0, 680.0])
        ]
    elif defect == "ambiguous_footer":
        evidence[1]["lines"] = [
            _line("第6页，共9页", [180.0, 660.0, 260.0, 680.0]),
            _line("第8页，共9页", [280.0, 660.0, 360.0, 680.0]),
        ]
    return pages, evidence, topology, resolution, continuations


def _paired_inquiry_entity_context(
    pages: list[SimpleNamespace],
    *,
    defect: str = "",
) -> CreditReportEntityContext:
    units: list[CreditReportUnit] = []
    for page in pages:
        for order, table in enumerate(page.tables):
            rows = tuple(tuple(str(value) for value in row) for row in table.metadata["raw_rows"])
            units.append(
                CreditReportUnit(
                    unit_id=f"unit:{table.table_id}",
                    page=page.page_number,
                    order=order,
                    source_index=order,
                    kind="table",
                    text="\n".join(" | ".join(row) for row in rows),
                    bbox=tuple(table.bbox),
                    page_width=page.width,
                    page_height=page.height,
                    table_id=table.table_id,
                    rows=rows,
                )
            )
    by_table = {unit.table_id: unit for unit in units}
    previous = by_table["inquiry-first"]
    current = by_table["inquiry-second"]
    if defect == "duplicate_unit":
        units.append(replace(previous, unit_id="unit:duplicate-inquiry-first"))
    elif defect == "role_mismatch":
        current_index = units.index(current)
        current = replace(current, kind="text")
        units[current_index] = current

    previous_entity = CreditReportEntity(
        entity_id="entity:inquiry-chain",
        kind="table",
        unit_ids=(previous.unit_id, current.unit_id),
        pages=(previous.page, current.page),
        confidence=0.98,
    )
    if defect == "wrong_entity":
        previous_entity = replace(
            previous_entity,
            unit_ids=(previous.unit_id,),
            pages=(previous.page,),
        )
        current_entities = (
            CreditReportEntity(
                entity_id="entity:other-inquiry",
                kind="table",
                unit_ids=(current.unit_id,),
                pages=(current.page,),
                confidence=0.98,
            ),
        )
    elif defect == "missing_entity":
        previous_entity = replace(
            previous_entity,
            unit_ids=(previous.unit_id,),
            pages=(previous.page,),
        )
        current_entities = ()
    else:
        current_entities = ()

    public = by_table["public-first"]
    entities = (
        CreditReportEntity(
            entity_id="entity:public",
            kind="table",
            unit_ids=(public.unit_id,),
            pages=(public.page,),
            confidence=0.99,
        ),
        previous_entity,
        *current_entities,
    )
    hypotheses = (
        TransitionHypothesis(
            action="same_table",
            score=0.98,
            signals=("exact_inquiry_continuation",),
        ),
        TransitionHypothesis(
            action="different_table",
            score=0.02,
            signals=(),
        ),
    )
    pair_decision = EntityTransitionDecision(
        left_unit_id=previous.unit_id,
        right_unit_id=current.unit_id,
        from_page=previous.page,
        to_page=current.page,
        hypotheses=hypotheses,
        selected="same_table",
        confidence=0.98,
    )
    decisions = [] if defect == "missing_edge" else [pair_decision]
    if defect == "competing_edge":
        decisions.append(
            replace(
                pair_decision,
                right_unit_id=public.unit_id,
                to_page=public.page,
            )
        )
    return CreditReportEntityContext(
        report_family="personal_detail",
        units=tuple(units),
        furniture_unit_ids=(),
        entities=entities,
        decisions=tuple(decisions),
        unassigned_unit_ids=(),
    )


def test_paired_inferred_footer_proves_only_local_inquiry_continuation() -> None:
    pages, evidence, topology, resolution, continuations = (
        _paired_inferred_inquiry_case()
    )

    projection = _build(
        pages,
        evidence,
        continuations=continuations,
        topology=topology,
        reading_order_resolution=resolution,
        entity_context=_paired_inquiry_entity_context(pages),
        force_table_continuation_false=True,
    )

    current = next(page for page in projection.pages if page.page_number == 60)
    owner = current.tables[0].metadata["canonical_section_owner"]
    assert owner["binding"] == "authoritative_prior_inquiry_table_continuation"
    assert resolution["authoritative"] is False


@pytest.mark.parametrize(
    "defect",
    (
        "different_source_pair",
        "profile_mismatch",
        "transform_mismatch",
        "nonconsecutive_mapping",
        "current_only_fallback",
        "missing_edge",
        "reverse_segment",
        "conflicting_footer",
        "ambiguous_footer",
    ),
)
def test_paired_inferred_footer_edge_fails_closed(defect: str) -> None:
    pages, evidence, topology, resolution, continuations = (
        _paired_inferred_inquiry_case(defect)
    )

    projection = _build(
        pages,
        evidence,
        continuations=continuations,
        topology=topology,
        reading_order_resolution=resolution,
        entity_context=_paired_inquiry_entity_context(
            pages,
            defect="missing_edge" if defect == "missing_edge" else "",
        ),
        force_table_continuation_false=True,
    )

    current = next(
        (page for page in projection.pages if page.page_number == 60),
        None,
    )
    assert current is None or not current.tables
    assert 60 in projection.unresolved_pages or (
        current is not None
        and current.canonical_template_id == "blank_fragment"
    )


@pytest.mark.parametrize(
    "defect",
    (
        "missing_entity",
        "wrong_entity",
        "duplicate_unit",
        "competing_edge",
        "role_mismatch",
        "geometry_reversal",
    ),
)
def test_local_paired_entity_continuation_fails_closed(defect: str) -> None:
    pages, evidence, topology, resolution, continuations = (
        _paired_inferred_inquiry_case()
    )
    entity_defect = defect
    if defect == "geometry_reversal":
        pages[0].tables.append(
            _table(
                "table-below-prior-inquiry",
                top=330.0,
                header=PUBLIC_HEADER,
                values=(("2", "主管机关乙", "0", "2025.01.02"),),
            )
        )
        entity_defect = ""

    projection = _build(
        pages,
        evidence,
        continuations=continuations,
        topology=topology,
        reading_order_resolution=resolution,
        entity_context=_paired_inquiry_entity_context(
            pages,
            defect=entity_defect,
        ),
        force_table_continuation_false=True,
    )

    current = next(
        (page for page in projection.pages if page.page_number == 60),
        None,
    )
    assert current is None or not current.tables
    assert 60 in projection.unresolved_pages or (
        current is not None
        and current.canonical_template_id == "blank_fragment"
    )


def test_headerless_inquiry_ignores_only_unowned_exact_terminal_border_band() -> None:
    pages, evidence = _inquiry_continuation_case()
    _append_terminal_border_band(pages[1].tables[0])

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    second = next(page for page in projection.pages if page.page_number == 2)
    assert second.tables[0].metadata["canonical_section_owner"]["binding"] == (
        "authoritative_prior_inquiry_table_continuation"
    )


def test_headerless_inquiry_rejects_terminal_blank_band_with_source_evidence() -> None:
    pages, evidence = _inquiry_continuation_case()
    _append_terminal_border_band(pages[1].tables[0], sealed=True)

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    assert all(page.page_number != 2 for page in projection.pages)
    assert 2 in projection.unresolved_pages


def _merge_headerless_role_cells(
    table: SimpleNamespace,
    *,
    row: int,
    left: int,
    right: int,
) -> None:
    assert right == left + 1
    geometry = table.metadata["geometry"]
    left_box = geometry["cell_bboxes"][row][left]
    right_box = geometry["cell_bboxes"][row][right]
    merged_ids = [
        *geometry["cell_evidence_ids"][row][left],
        *geometry["cell_evidence_ids"][row][right],
    ]
    table.metadata["raw_rows"][row][left] = (
        f"{table.metadata['raw_rows'][row][left]} "
        f"{table.metadata['raw_rows'][row][right]}"
    )
    table.metadata["raw_rows"][row][right] = ""
    geometry["cell_bboxes"][row][left] = [
        left_box[0],
        left_box[1],
        right_box[2],
        left_box[3],
    ]
    geometry["cell_bboxes"][row][right] = None
    geometry["cell_geometry_status"][row][right] = "derived"
    geometry["cell_evidence_ids"][row][left] = merged_ids
    geometry["cell_evidence_ids"][row][right] = []
    geometry.setdefault("cell_spans", []).append(
        {
            "row": row,
            "col": left,
            "row_span": 1,
            "col_span": 2,
            "bbox": list(geometry["cell_bboxes"][row][left]),
            "evidence_ids": list(merged_ids),
        }
    )


def test_headerless_inquiry_preserves_exact_empty_and_colspan_role_slots() -> None:
    pages, evidence = _inquiry_continuation_case()
    continuation = pages[1].tables[0]
    _merge_headerless_role_cells(continuation, row=0, left=0, right=1)

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    second = next(page for page in projection.pages if page.page_number == 2)
    owner = second.tables[0].metadata["canonical_section_owner"]
    assert owner["header_binding"] == "inherited_exact_four_role_lattice"
    assert owner["physical_field_omission_rows"] == [0]


def test_headerless_inquiry_preserves_one_exact_empty_role_slot() -> None:
    pages, evidence = _inquiry_continuation_case()
    continuation = pages[1].tables[0]
    continuation.metadata["raw_rows"][0][0] = ""
    continuation.metadata["geometry"]["cell_evidence_ids"][0][0] = []

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    second = next(page for page in projection.pages if page.page_number == 2)
    owner = second.tables[0].metadata["canonical_section_owner"]
    assert owner["header_binding"] == "inherited_exact_four_role_lattice"
    assert owner["physical_field_omission_rows"] == [0]


@pytest.mark.parametrize("defect", ["wrong_span", "foreign_role"])
def test_headerless_inquiry_rejects_wrong_span_or_foreign_role(defect: str) -> None:
    pages, evidence = _inquiry_continuation_case()
    continuation = pages[1].tables[0]
    if defect == "wrong_span":
        _merge_headerless_role_cells(continuation, row=0, left=0, right=1)
        continuation.metadata["geometry"]["cell_spans"][0]["evidence_ids"] = [
            "replayed-or-foreign"
        ]
    else:
        continuation.metadata["raw_rows"][0][2] = "管理机构"

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    assert all(page.page_number != 2 for page in projection.pages)
    assert 2 in projection.unresolved_pages


def _collapse_headerless_sequence_and_date(table: SimpleNamespace) -> None:
    geometry = table.metadata["geometry"]
    collapsed_rows: list[list[str]] = []
    collapsed_boxes: list[list[list[float]]] = []
    collapsed_statuses: list[list[str]] = []
    collapsed_ids: list[list[list[str]]] = []
    for row_index, row in enumerate(table.metadata["raw_rows"]):
        first_box, second_box, institution_box, reason_box = geometry[
            "cell_bboxes"
        ][row_index]
        collapsed_rows.append(
            [f"{row[0]} {row[1]}", row[2], row[3]]
        )
        collapsed_boxes.append(
            [
                [first_box[0], first_box[1], second_box[2], first_box[3]],
                institution_box,
                reason_box,
            ]
        )
        collapsed_statuses.append(["exact", "exact", "exact"])
        collapsed_ids.append(
            [
                [
                    *geometry["cell_evidence_ids"][row_index][0],
                    *geometry["cell_evidence_ids"][row_index][1],
                ],
                geometry["cell_evidence_ids"][row_index][2],
                geometry["cell_evidence_ids"][row_index][3],
            ]
        )
    table.metadata["raw_rows"] = collapsed_rows
    geometry["cell_bboxes"] = collapsed_boxes
    geometry["cell_geometry_status"] = collapsed_statuses
    geometry["cell_evidence_ids"] = collapsed_ids
    geometry["cell_spans"] = []


def test_headerless_three_column_sequence_date_merge_requires_unique_types() -> None:
    pages, evidence = _inquiry_continuation_case()
    continuation = pages[1].tables[0]
    _collapse_headerless_sequence_and_date(continuation)

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    second = next(page for page in projection.pages if page.page_number == 2)
    owner = second.tables[0].metadata["canonical_section_owner"]
    assert owner["header_binding"] == (
        "inherited_exact_sequence_date_collapsed_lattice"
    )
    assert owner["physical_role_columns"] == [
        ["sequence", "inquiry_date"],
        ["institution"],
        ["reason"],
    ]


def test_headerless_three_column_ambiguous_sequence_date_tokens_fail_closed() -> None:
    pages, evidence = _inquiry_continuation_case()
    continuation = pages[1].tables[0]
    _collapse_headerless_sequence_and_date(continuation)
    for row in continuation.metadata["raw_rows"]:
        row[0] = f"{row[0]} 999"

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    assert all(page.page_number != 2 for page in projection.pages)
    assert 2 in projection.unresolved_pages


def test_split_personal_header_and_leading_institution_tail_are_both_owned() -> None:
    pages, evidence = _inquiry_continuation_case()
    leading = pages[1].tables[0]
    personal = _table(
        "personal-split",
        top=260.0,
        header=("编号", "", "", ""),
        values=(
            ("", "查询日期", "查询机构", "查询原因"),
            ("1", "2025.01.05", "本人", "本人查询(自助查询机)"),
        ),
    )
    personal.metadata["geometry"]["cell_spans"] = [
        {"row": 0, "col": 0, "row_span": 2, "col_span": 1}
    ]
    pages[1].tables = [leading, personal]
    evidence[1]["lines"].insert(
        0,
        _line("本人查询记录明细", [180.0, 245.0, 370.0, 262.0]),
    )

    projection = _build(
        pages[:2],
        evidence[:2],
        continuations={("inquiry-first", "inquiry-second")},
    )

    second = next(page for page in projection.pages if page.page_number == 2)
    assert {
        table.table_id: table.metadata["canonical_template_id"]
        for table in second.tables
    } == {
        "inquiry-second": "annotations_and_inquiries",
        "personal-split": "annotations_and_inquiries",
    }
    assert second.tables[1].metadata["canonical_section_owner"][
        "header_binding"
    ] == "exact_complementary_header_lattice"


def test_headerless_inquiry_page_rejects_missing_edge_or_inexact_cell() -> None:
    for defect in ("missing_edge", "inexact_cell", "nonfinite_bbox"):
        pages, evidence = _inquiry_continuation_case()
        if defect == "inexact_cell":
            pages[1].tables[0].metadata["geometry"]["cell_geometry_status"][0][0] = (
                "derived"
            )
        elif defect == "nonfinite_bbox":
            pages[1].tables[0].metadata["geometry"]["cell_bboxes"][0][0][2] = (
                float("inf")
            )
        continuations = (
            set()
            if defect == "missing_edge"
            else {("inquiry-first", "inquiry-second")}
        )

        projection = _build(pages[:2], evidence[:2], continuations=continuations)

        projected = next(
            (page for page in projection.pages if page.page_number == 2),
            None,
        )
        assert projected is None or not projected.tables
        assert 2 in projection.unresolved_pages or (
            projected is not None
            and projected.canonical_template_id == "blank_fragment"
        )


def test_account_anchor_may_touch_but_not_enter_the_table_header_band() -> None:
    account = _table(
        "account-card",
        top=100.0,
        header=("管理机构", "账户标识", "开立日期", "业务种类", "账户币种", "担保方式"),
        values=(("机构甲", "A1", "2024.01.01", "贷款", "人民币", "信用"),),
    )
    page = _page(2, account)
    evidence = _evidence(2, _line("账户21", [35.0, 90.0, 125.0, 101.0]))

    assert _sealed_account_card_continuation_proved(page, evidence) is True

    evidence["lines"][0]["bbox"] = [35.0, 90.0, 125.0, 105.0]
    assert _sealed_account_card_continuation_proved(page, evidence) is False


def _sealed_account_continuation_case(
    *,
    population: int,
    start: int,
    scale: float,
    reverse_tables: bool,
) -> tuple[SimpleNamespace, dict[str, object]]:
    header = ("管理机构", "账户标识", "开立日期", "到期日期", "借款金额", "账户币种")
    widths = (3.0, 9.0, 4.0, 5.0, 7.0, 2.0)
    tables = [
        _table(
            f"account-{index}",
            top=100.0 + index * 90.0,
            header=header,
            values=(("机构甲", f"A{index}", "2024.01.01", "--", "1000", "人民币元"),),
            widths=widths,
        )
        for index in range(population)
    ]
    lines = [
        _line(
            f"账户{start + index}",
            [35.0, 90.0 + index * 90.0, 125.0, 101.0 + index * 90.0],
        )
        for index in range(population)
    ]
    page = _page(2, *(reversed(tables) if reverse_tables else tables))
    evidence = _evidence(2, *lines)
    if scale != 1.0:
        page.width *= scale
        page.height *= scale
        for table in tables:
            table.bbox = [value * scale for value in table.bbox]
            geometry = table.metadata["geometry"]
            geometry["cell_bboxes"] = [
                [[value * scale for value in bbox] for bbox in row]
                for row in geometry["cell_bboxes"]
            ]
        for line in evidence["lines"]:
            line["bbox"] = [value * scale for value in line["bbox"]]
        evidence["page_width"] = float(evidence["page_width"]) * scale
        evidence["page_height"] = float(evidence["page_height"]) * scale
    return page, evidence


@pytest.mark.parametrize(
    ("population", "start", "scale", "reverse_tables"),
    [
        (1, 4, 0.55, False),
        (2, 31, 1.0, True),
        (4, 207, 1.75, False),
    ],
)
def test_account_continuation_requires_complete_sealed_bijection_at_any_scale(
    population: int,
    start: int,
    scale: float,
    reverse_tables: bool,
) -> None:
    page, evidence = _sealed_account_continuation_case(
        population=population,
        start=start,
        scale=scale,
        reverse_tables=reverse_tables,
    )

    assert _sealed_account_card_continuation_proved(page, evidence) is True


def test_account_continuation_does_not_assume_locally_consecutive_ordinals() -> None:
    page, evidence = _sealed_account_continuation_case(
        population=2,
        start=40,
        scale=1.0,
        reverse_tables=False,
    )
    evidence["lines"][1]["text"] = "账户43"

    assert _sealed_account_card_continuation_proved(page, evidence) is True


@pytest.mark.parametrize(
    "defect",
    [
        "missing_table",
        "extra_dense_table",
        "duplicate_table_id",
        "ambiguous_interval",
        "ordinal_reversal",
        "duplicate_anchor_evidence",
        "replayed_table_evidence",
        "replayed_noncandidate_table_evidence",
        "replayed_line_evidence",
        "non_string_header_evidence",
        "blank_header_evidence_member",
        "inexact_header_cell",
        "foreign_section_heading",
    ],
)
def test_account_continuation_rejects_incomplete_ambiguous_or_replayed_owners(
    defect: str,
) -> None:
    page, evidence = _sealed_account_continuation_case(
        population=2,
        start=40,
        scale=1.0,
        reverse_tables=False,
    )
    if defect == "missing_table":
        page.tables.pop()
    elif defect == "extra_dense_table":
        extra, _unused = _sealed_account_continuation_case(
            population=1,
            start=1,
            scale=1.0,
            reverse_tables=False,
        )
        extra_table = extra.tables[0]
        extra_table.table_id = "extra-account"
        page.tables.append(extra_table)
    elif defect == "duplicate_table_id":
        page.tables[1].table_id = page.tables[0].table_id
    elif defect == "ambiguous_interval":
        second = page.tables[1]
        shift = -85.0
        second.bbox[1] += shift
        second.bbox[3] += shift
        second.metadata["geometry"]["cell_bboxes"] = [
            [
                [bbox[0], bbox[1] + shift, bbox[2], bbox[3] + shift]
                for bbox in row
            ]
            for row in second.metadata["geometry"]["cell_bboxes"]
        ]
    elif defect == "ordinal_reversal":
        evidence["lines"][0]["text"] = "账户42"
        evidence["lines"][1]["text"] = "账户41"
    elif defect == "duplicate_anchor_evidence":
        evidence["lines"][1]["evidence_ids"] = list(
            evidence["lines"][0]["evidence_ids"]
        )
    elif defect == "replayed_table_evidence":
        page.tables[1].metadata["geometry"]["cell_evidence_ids"][0][0] = list(
            page.tables[0].metadata["geometry"]["cell_evidence_ids"][0][0]
        )
    elif defect == "replayed_noncandidate_table_evidence":
        unrelated = _table(
            "unrelated",
            top=500.0,
            header=("字段甲", "字段乙"),
            values=(("值甲", "值乙"),),
        )
        unrelated.metadata["geometry"]["cell_evidence_ids"][0][0] = list(
            page.tables[0].metadata["geometry"]["cell_evidence_ids"][0][0]
        )
        page.tables.append(unrelated)
    elif defect == "replayed_line_evidence":
        evidence["lines"].insert(
            -1,
            _line(
                "旁注",
                [180.0, 500.0, 230.0, 515.0],
                owner=page.tables[0].metadata["geometry"]["cell_evidence_ids"][0][0][0],
            ),
        )
    elif defect == "non_string_header_evidence":
        page.tables[0].metadata["geometry"]["cell_evidence_ids"][0][0] = [17]
    elif defect == "blank_header_evidence_member":
        page.tables[0].metadata["geometry"]["cell_evidence_ids"][0][0].append(" ")
    elif defect == "inexact_header_cell":
        page.tables[0].metadata["geometry"]["cell_geometry_status"][0][0] = (
            "derived"
        )
    else:
        evidence["lines"].insert(
            -1,
            _line("授信协议信息", [180.0, 300.0, 360.0, 315.0]),
        )

    assert _sealed_account_card_continuation_proved(page, evidence) is False
