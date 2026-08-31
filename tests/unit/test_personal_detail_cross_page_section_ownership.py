from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessUncertaintyRepairCoordinator,
)
from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    PBOCCanonicalTemplateAssembler,
    _sealed_account_card_continuation_proved,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_inquiries,
    _inquiry_source_coverage,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)
from docmirror.plugins.credit_report.personal_detail_scanned.page_topology import (
    PersonalDetailPageTopology,
)
from docmirror.plugins.credit_report.personal_detail_scanned.table_ownership import (
    INQUIRY_AUTHORITY_CLOSED_PHYSICAL_ORDINAL,
    INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY,
    canonical_inquiry_population_metadata,
    canonical_table_role,
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
    force_table_continuation_unknown: bool = False,
):
    topology = topology or SimpleNamespace(geometry=lambda _logical: None)
    owner = SimpleNamespace(
        tables_continue=lambda left, right: (
            None
            if force_table_continuation_unknown
            else (
                False
                if force_table_continuation_false
                else (left, right) in (continuations or set())
            )
        ),
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


def _ye_shaped_spread_topology(*, profile_mismatch: bool = False) -> PersonalDetailPageTopology:
    second_left = [0.0, 0.0, 280.0, 800.0] if profile_mismatch else [0.0, 0.0, 300.0, 800.0]
    second_right = [280.0, 0.0, 600.0, 800.0] if profile_mismatch else [300.0, 0.0, 600.0, 800.0]
    return PersonalDetailPageTopology(
        SimpleNamespace(
            pages=[
                _spread_topology_page(27, source=14, segment=0, crop=[0.0, 0.0, 300.0, 800.0]),
                _spread_topology_page(28, source=14, segment=1, crop=[300.0, 0.0, 600.0, 800.0]),
                _spread_topology_page(29, source=15, segment=0, crop=second_left),
                _spread_topology_page(30, source=15, segment=1, crop=second_right),
            ]
        )
    )


def _ye_shaped_inquiry_entity_context(
    pages: list[SimpleNamespace],
    *,
    missing_first_edge: bool = False,
    missing_second_edge: bool = False,
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
                    source_index=len(units),
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
    institution_ids = (
        "ye-inquiry:1-36",
        "ye-inquiry:37-76",
        "ye-inquiry:77-96",
    )
    institution_units = tuple(by_table[table_id] for table_id in institution_ids)
    personal = by_table["ye-inquiry:personal-1-16"]
    entities = (
        CreditReportEntity(
            entity_id="entity:ye-institution-inquiry-chain",
            kind="table",
            unit_ids=tuple(unit.unit_id for unit in institution_units),
            pages=tuple(unit.page for unit in institution_units),
            confidence=0.99,
        ),
        CreditReportEntity(
            entity_id="entity:ye-personal-inquiry",
            kind="table",
            unit_ids=(personal.unit_id,),
            pages=(personal.page,),
            confidence=0.99,
        ),
    )
    hypotheses = (
        TransitionHypothesis(
            action="same_table",
            score=0.99,
            signals=("exact_inquiry_continuation",),
        ),
        TransitionHypothesis(
            action="different_table",
            score=0.01,
            signals=(),
        ),
    )
    decisions = [
        EntityTransitionDecision(
            left_unit_id=left.unit_id,
            right_unit_id=right.unit_id,
            from_page=left.page,
            to_page=right.page,
            hypotheses=hypotheses,
            selected="same_table",
            confidence=0.99,
        )
        for left, right in zip(institution_units[:-1], institution_units[1:], strict=True)
    ]
    if missing_first_edge:
        decisions = decisions[1:]
    if missing_second_edge:
        decisions = decisions[:1]
    return CreditReportEntityContext(
        report_family="personal_detail",
        units=tuple(units),
        furniture_unit_ids=(),
        entities=entities,
        decisions=tuple(decisions),
        unassigned_unit_ids=(),
    )


def _ye_shaped_inquiry_registration_case(
    defect: str = "",
) -> tuple[
    list[SimpleNamespace],
    list[dict[str, object]],
    PersonalDetailPageTopology,
    CreditReportEntityContext,
]:
    def values(first: int, last: int) -> tuple[tuple[str, ...], ...]:
        return tuple(
            (
                str(sequence),
                f"2024.{(sequence - 1) // 28 + 1:02d}.{(sequence - 1) % 28 + 1:02d}",
                f"查询机构{sequence}",
                "贷后管理" if sequence % 2 else "贷款审批",
            )
            for sequence in range(first, last + 1)
        )

    institution_first = _table(
        "ye-inquiry:1-36",
        top=80.0,
        header=INQUIRY_HEADER,
        values=values(1, 36),
    )
    first_geometry = institution_first.metadata["geometry"]
    first_geometry["cell_bboxes"][0][1] = [
        first_geometry["cell_bboxes"][0][1][0],
        first_geometry["cell_bboxes"][0][1][1],
        first_geometry["cell_bboxes"][0][2][2],
        first_geometry["cell_bboxes"][0][1][3],
    ]
    first_geometry["cell_evidence_ids"][0][1] = [
        *first_geometry["cell_evidence_ids"][0][1],
        *first_geometry["cell_evidence_ids"][0][2],
    ]
    first_geometry["cell_bboxes"][0][2] = None
    first_geometry["cell_geometry_status"][0][2] = "derived"
    first_geometry["cell_evidence_ids"][0][2] = []
    first_geometry["cell_spans"] = [
        {
            "row": 0,
            "col": 1,
            "row_span": 1,
            "col_span": 2,
            "bbox": list(first_geometry["cell_bboxes"][0][1]),
            "evidence_ids": list(first_geometry["cell_evidence_ids"][0][1]),
        }
    ]
    institution_first.metadata["raw_rows"][0] = [
        "编号",
        "? 查询日期 查询机构 X",
        "",
        "查询原因",
    ]
    for row_index in (5, 8, 9, 29):
        institution_first.metadata["raw_rows"][row_index][0] = ""
        first_geometry["cell_evidence_ids"][row_index][0] = []
    institution_first.metadata["raw_rows"][27][0] = "27 多"
    institution_first.metadata["raw_rows"][19][1] += "Pd"
    institution_first.metadata["raw_rows"][22][1] = (
        "心" + institution_first.metadata["raw_rows"][22][1]
    )
    institution_first.metadata["raw_rows"][26][1] = (
        "证" + institution_first.metadata["raw_rows"][26][1]
    )

    institution_second = _table(
        "ye-inquiry:37-76",
        top=60.0,
        header=("", "", "", ""),
        values=values(37, 76),
    )
    institution_third = _table(
        "ye-inquiry:77-96",
        top=60.0,
        header=("", "", "", ""),
        values=values(77, 96),
    )
    for table in (institution_second, institution_third):
        table.metadata["raw_rows"] = table.metadata["raw_rows"][1:]
        for key in ("cell_bboxes", "cell_geometry_status", "cell_evidence_ids"):
            table.metadata["geometry"][key] = table.metadata["geometry"][key][1:]
    institution_second.metadata["raw_rows"][11][0] = "福 48"
    institution_second.metadata["raw_rows"][30][0] = ""
    institution_second.metadata["geometry"]["cell_evidence_ids"][30][0] = []
    institution_second.metadata["raw_rows"][32][0] = "K69"
    institution_third.metadata["raw_rows"][12][0] = "789"

    personal = _table(
        "ye-inquiry:personal-1-16",
        top=600.0,
        header=INQUIRY_HEADER,
        values=tuple(
            (
                str(sequence),
                f"2023.{(sequence - 1) // 28 + 1:02d}.{(sequence - 1) % 28 + 1:02d}",
                "本人",
                "本人查询(自助查询机)",
            )
            for sequence in range(1, 17)
        ),
    )
    pages = [
        _page(27, institution_first),
        _page(28, institution_second),
        _page(29, institution_third, personal),
    ]
    for page, source in zip(pages, (14, 14, 15), strict=True):
        page.source_page_number = source
        page.height = 1200.0

    def page_evidence(
        logical: int,
        source: int,
        *lines: dict[str, object],
    ) -> dict[str, object]:
        return {
            "page": logical,
            "source_page": source,
            "page_width": 560.0,
            "page_height": 1200.0,
            "lines": [
                *lines,
                _line(
                    f"第{logical}页,共31页",
                    [220.0, 1160.0, 340.0, 1180.0],
                ),
            ],
        }

    evidence = [
        page_evidence(
            27,
            14,
            _line(
                "机构查询记录明细",
                [190.0, 58.0, 370.0, 80.5],
            ),
        ),
        page_evidence(28, 14),
        page_evidence(
            29,
            15,
            _line("本人查询记录明细", [190.0, 575.0, 370.0, 600.0]),
        ),
    ]
    if defect == "wrong_first_endpoint":
        institution_first.metadata["raw_rows"][1][0] = "2"
    elif defect == "wrong_last_endpoint":
        institution_first.metadata["raw_rows"][-1][0] = "35"
    elif defect == "interior_geometry_gap":
        institution_first.metadata["geometry"]["cell_geometry_status"][10][2] = "derived"
    elif defect == "malformed_non_sequence_role":
        institution_first.metadata["raw_rows"][10][1] = "2024.99"
    elif defect == "ambiguous_cell_owner":
        institution_first.metadata["geometry"]["cell_evidence_ids"][10][2] = list(
            institution_first.metadata["geometry"]["cell_evidence_ids"][11][2]
        )
    elif defect == "overlapping_cell_geometry":
        institution_first.metadata["geometry"]["cell_bboxes"][10][2] = list(
            institution_first.metadata["geometry"]["cell_bboxes"][10][1]
        )
    elif defect == "foreign_header_residue":
        institution_first.metadata["raw_rows"][0][1] = "? 查询日期 查询机构 账户"
    elif defect == "partial_subsection_heading":
        evidence[0]["lines"][0]["text"] = "机构查询记录"
    elif defect == "nonconsecutive_footer":
        evidence[1]["lines"][-1]["text"] = "第30页,共31页"
    elif defect == "restart_second_population":
        institution_second.metadata["raw_rows"][0][0] = "1"
    elif defect == "gap_second_population":
        institution_second.metadata["raw_rows"][0][0] = "38"
    elif defect == "wrong_third_start":
        institution_third.metadata["raw_rows"][0][0] = "78"
    elif defect == "merged_date_institution_tail":
        # Production-shaped Ye tail: the final four institutional rows retain
        # four physical columns, but the date cell spans the otherwise empty
        # institution slot.  The source text remains lossless and the exact
        # colspan is part of the immutable cell lattice.
        geometry = institution_third.metadata["geometry"]
        geometry["cell_spans"] = []
        for row_index in range(16, 20):
            row = institution_third.metadata["raw_rows"][row_index]
            row[1] = f"{row[1]} {row[2]}"
            row[2] = ""
            left_box = geometry["cell_bboxes"][row_index][1]
            right_box = geometry["cell_bboxes"][row_index][2]
            merged_box = [left_box[0], left_box[1], right_box[2], left_box[3]]
            merged_ids = [
                *geometry["cell_evidence_ids"][row_index][1],
                *geometry["cell_evidence_ids"][row_index][2],
            ]
            geometry["cell_bboxes"][row_index][1] = merged_box
            geometry["cell_evidence_ids"][row_index][1] = merged_ids
            geometry["cell_bboxes"][row_index][2] = None
            geometry["cell_geometry_status"][row_index][2] = "derived"
            geometry["cell_evidence_ids"][row_index][2] = []
            geometry["cell_spans"].append(
                {
                    "row": row_index,
                    "col": 1,
                    "row_span": 1,
                    "col_span": 2,
                    "bbox": list(merged_box),
                    "evidence_ids": list(merged_ids),
                }
            )
    topology = _ye_shaped_spread_topology(
        profile_mismatch=defect == "topology_profile_mismatch"
    )
    entity_context = _ye_shaped_inquiry_entity_context(
        pages,
        missing_first_edge=defect == "missing_first_edge",
        missing_second_edge=defect == "missing_second_edge",
    )
    return pages, evidence, topology, entity_context


def _compress_table_height(table: SimpleNamespace, factor: float) -> None:
    top = float(table.bbox[1])
    geometry = table.metadata["geometry"]
    geometry["cell_bboxes"] = [
        [
            [
                box[0],
                top + (box[1] - top) * factor,
                box[2],
                top + (box[3] - top) * factor,
            ]
            for box in row
        ]
        for row in geometry["cell_bboxes"]
    ]
    table.bbox[3] = top + (float(table.bbox[3]) - top) * factor


def _make_headerless(table: SimpleNamespace) -> None:
    table.metadata["raw_rows"] = table.metadata["raw_rows"][1:]
    geometry = table.metadata["geometry"]
    for key in ("cell_bboxes", "cell_geometry_status", "cell_evidence_ids"):
        geometry[key] = geometry[key][1:]
    table.bbox[1] = min(box[1] for box in geometry["cell_bboxes"][0])
    for row_index, row in enumerate(table.metadata["raw_rows"]):
        if not str(row[0] or "").strip():
            geometry["cell_evidence_ids"][row_index][0] = []


def _lin_shaped_inquiry_case():
    def values(first: int, last: int) -> tuple[tuple[str, ...], ...]:
        return tuple(
            (
                str(sequence),
                f"2022.{(sequence - 1) // 28 + 1:02d}.{(sequence - 1) % 28 + 1:02d}",
                f"查询机构{sequence}",
                "贷后管理" if sequence % 2 else "贷款审批",
            )
            for sequence in range(first, last + 1)
        )

    seed_values = [list(row) for row in values(1, 16)]
    seed_values[3][0] = ""
    seed_values[4][0] = ""
    seed = _table(
        "ordinary-inquiry-seed",
        top=80.0,
        header=INQUIRY_HEADER,
        values=tuple(tuple(row) for row in seed_values),
    )
    seed.metadata["geometry"]["cell_evidence_ids"][4][0] = []
    seed.metadata["geometry"]["cell_evidence_ids"][5][0] = []

    middle_values = [list(row) for row in values(17, 54)]
    middle_values[10][0] = "%"
    middle_values[29][0] = "花拾"
    middle = _table(
        "ordinary-inquiry-continuation-a",
        top=20.0,
        header=("", "", "", ""),
        values=tuple(tuple(row) for row in middle_values),
    )
    _compress_table_height(middle, 0.75)
    _make_headerless(middle)

    tail_values = [list(row) for row in values(55, 89)]
    tail_values[11][0] = "6%"
    tail_values[12][0] = ""
    tail_values[32][0] = "0k87"
    tail = _table(
        "ordinary-inquiry-continuation-b",
        top=20.0,
        header=("", "", "", ""),
        values=tuple(tuple(row) for row in tail_values),
    )
    _compress_table_height(tail, 0.6)
    _make_headerless(tail)

    personal = _table(
        "ordinary-personal-inquiry",
        top=540.0,
        header=INQUIRY_HEADER,
        values=(("1", "2023.01.01", "本人", "本人查询"),),
    )
    pages = [
        _page(5, seed),
        _page(6, middle),
        _page(7, tail, personal),
    ]
    evidence = [
        _evidence(5, _line("查询记录", [190.0, 45.0, 350.0, 65.0])),
        _evidence(6),
        _evidence(
            7,
            _line("本人查询记录明细", [180.0, 510.0, 370.0, 525.0]),
        ),
    ]
    return pages, evidence


def _lin_shaped_inquiry_projection():
    pages, evidence = _lin_shaped_inquiry_case()
    return (
        _build(
            pages,
            evidence,
            continuations={
                ("ordinary-inquiry-seed", "ordinary-inquiry-continuation-a"),
                (
                    "ordinary-inquiry-continuation-a",
                    "ordinary-inquiry-continuation-b",
                ),
            },
        ),
        evidence,
    )


def _fit_table_to_bbox(table: SimpleNamespace, target: list[float]) -> None:
    """Affine-fit a synthetic lattice to one production table rectangle."""

    source = [float(value) for value in table.bbox]
    x_scale = (target[2] - target[0]) / (source[2] - source[0])
    y_scale = (target[3] - target[1]) / (source[3] - source[1])
    for row in table.metadata["geometry"]["cell_bboxes"]:
        for index, box in enumerate(row):
            if box is None:
                continue
            row[index] = [
                target[0] + (box[0] - source[0]) * x_scale,
                target[1] + (box[1] - source[1]) * y_scale,
                target[0] + (box[2] - source[0]) * x_scale,
                target[1] + (box[3] - source[1]) * y_scale,
            ]
    table.bbox = list(target)


def _retarget_table_id(table: SimpleNamespace, table_id: str) -> None:
    old_id = table.table_id
    table.table_id = table_id
    evidence = table.metadata["geometry"]["cell_evidence_ids"]
    for row in evidence:
        for column, ids in enumerate(row):
            row[column] = [
                str(value).replace(f"{old_id}:", f"{table_id}:", 1)
                for value in ids
            ]


def _lin_production_shaped_inquiry_case():
    """Replay the small page/table slice persisted by the Lin live audit.

    The live report is a two-logical-pages-per-source scan.  Its ordinary
    inquiry seed ends on logical page 26, while the two exact headerless
    tables start on pages 27 and 28.  The entity graph keeps all three
    physical tables separate, so ``tables_continue`` truthfully returns
    ``False`` even though the exact inquiry schema carries across them.
    """

    pages, _synthetic_evidence = _lin_shaped_inquiry_case()
    page_numbers = (26, 27, 28)
    source_pages = (13, 14, 14)
    page_widths = (447.0, 404.0, 438.0)
    table_ids = {
        "ordinary-inquiry-seed": "pt_26_3",
        "ordinary-inquiry-continuation-a": "pt_27_0",
        "ordinary-inquiry-continuation-b": "pt_28_0",
        "ordinary-personal-inquiry": "pt_28_1",
    }
    target_boxes = {
        "pt_26_3": [50.5, 325.0, 401.0, 561.5],
        "pt_27_0": [44.5, 35.0, 398.5, 557.0],
        "pt_28_0": [52.0, 34.5, 402.5, 487.5],
        "pt_28_1": [53.0, 512.0, 402.5, 540.0],
    }
    for page, logical, source, width in zip(
        pages,
        page_numbers,
        source_pages,
        page_widths,
        strict=True,
    ):
        page.page_number = logical
        page.source_page_number = source
        page.width = width
        page.height = 595.5
        for table in page.tables:
            _retarget_table_id(table, table_ids[table.table_id])
            _fit_table_to_bbox(table, target_boxes[table.table_id])

    # The live seed has two source-preserving, geometry-derived empty sequence
    # cells.  All remaining populated cells retain exact boxes and evidence.
    seed = pages[0].tables[0]
    seed.metadata["raw_rows"][4] = [
        "",
        "2022.1214",
        "平安普惠融资担保有限公司",
        "担保资格审查",
    ]
    seed.metadata["raw_rows"][5] = [
        "",
        "2022.12.13",
        "浙江网商银行股份有限公司",
        "贷后管理",
    ]
    seed.metadata["raw_rows"][8][0] = "8 游"
    seed.metadata["raw_rows"][9][0] = "敬9"
    for row in (4, 5):
        seed.metadata["geometry"]["cell_geometry_status"][row][0] = "derived"
        seed.metadata["geometry"]["cell_bboxes"][row][0] = None
        seed.metadata["geometry"]["cell_evidence_ids"][row][0] = []

    evidence = [
        {
            "page": 26,
            "source_page": 13,
            "page_width": 447.0,
            "page_height": 595.5,
            "lines": [
                _line(
                    "机构查询记录明细",
                    [197.0, 315.0, 257.0, 326.5],
                    owner="lin:p26:institution-heading",
                ),
                _line(
                    "第26页,共30页",
                    [199.0, 577.5, 248.5, 586.0],
                    owner="lin:p26:footer",
                ),
            ],
        },
        {
            "page": 27,
            "source_page": 14,
            "page_width": 404.0,
            "page_height": 595.5,
            "lines": [
                _line(
                    "第27页,共30页",
                    [194.0, 564.5, 245.5, 574.5],
                    owner="lin:p27:footer",
                ),
            ],
        },
        {
            "page": 28,
            "source_page": 14,
            "page_width": 438.0,
            "page_height": 595.5,
            "lines": [
                _line(
                    "本人查询记录明细",
                    [198.0, 502.0, 258.5, 512.0],
                    owner="lin:p28:personal-heading",
                ),
                _line(
                    "拉",
                    [266.5, 502.5, 280.5, 514.0],
                    owner="lin:p28:noise",
                ),
                _line(
                    "第28页,共30页",
                    [201.0, 563.0, 252.0, 574.5],
                    owner="lin:p28:footer",
                ),
            ],
        },
    ]
    for page, page_evidence in zip(pages, evidence, strict=True):
        page.texts = deepcopy(page_evidence["lines"])
    resolution = {
        "resolved": True,
        "authoritative": True,
        "identity_fallback": False,
        "printed_total": 30,
        "printed_page_by_logical": {26: 26, 27: 27, 28: 28},
    }
    return pages, evidence, resolution


def _lin_production_shaped_projection():
    pages, evidence, resolution = _lin_production_shaped_inquiry_case()
    projection = _build(
        pages,
        evidence,
        force_table_continuation_false=True,
        reading_order_resolution=resolution,
    )
    return projection, evidence, resolution


def _lin_production_shaped_consumer_context(projection, evidence, resolution):
    consumer_evidence = deepcopy(evidence)
    consumer_evidence[0]["canonical_template_id"] = "annotations_and_inquiries"
    consumer_evidence[0]["lines"].append(
        _line(
            "5 2022.12.13 浙江网商银行股份有限公司 贷后管理",
            [55.0, 382.0, 390.0, 393.0],
            owner="lin:p26:sequence-5-line",
        )
    )
    return SimpleNamespace(
        pages=list(projection.pages),
        reading_order_by_logical={26: 1, 27: 2, 28: 3},
        reading_order_resolution=deepcopy(resolution),
        page_topology=SimpleNamespace(geometry=lambda _logical: None),
        tables_continue=lambda _left, _right: False,
        corrected_evidence_pages=lambda: deepcopy(consumer_evidence),
    )


def _lin_shaped_consumer_context(projection, evidence):
    consumer_evidence = deepcopy(evidence)
    consumer_evidence[0]["canonical_template_id"] = "annotations_and_inquiries"
    consumer_evidence[0]["lines"].append(
        _line(
            "5 2022.01.05 查询机构5 贷后管理",
            [35.0, 181.0, 500.0, 195.0],
        )
    )
    return SimpleNamespace(
        pages=list(projection.pages),
        reading_order_by_logical={5: 1, 6: 2, 7: 3},
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
            "identity_fallback": False,
            "printed_total": 9,
            "printed_page_by_logical": {5: 5, 6: 6, 7: 7},
        },
        tables_continue=lambda left, right: (left, right)
        in {
            (
                "ordinary-inquiry-seed",
                "ordinary-inquiry-continuation-a",
            ),
            (
                "ordinary-inquiry-continuation-a",
                "ordinary-inquiry-continuation-b",
            ),
        },
        corrected_evidence_pages=lambda: deepcopy(consumer_evidence),
    )


def _ye_shaped_consumer_context(projection, evidence, topology):
    return SimpleNamespace(
        pages=list(projection.pages),
        reading_order_by_logical={27: 1, 28: 2, 29: 3},
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
            "identity_fallback": False,
            "printed_total": 31,
            "printed_page_by_logical": {27: 27, 28: 28, 29: 29},
        },
        page_topology=topology,
        tables_continue=lambda _left, _right: False,
        page_topology_audit=lambda: {
            **topology.audit(),
            "topology_frozen_before_reocr": True,
        },
        entity_context=_ye_shaped_inquiry_entity_context(
            list(projection.pages)
        ),
        corrected_evidence_pages=lambda: deepcopy(evidence),
    )


def test_lin_shaped_ordinary_seed_carries_schema_without_closing_ordinals() -> None:
    projection, evidence = _lin_shaped_inquiry_projection()
    context = _lin_shaped_consumer_context(projection, evidence)
    continuations = {
        table.table_id: (page, table)
        for page in context.pages
        for table in page.tables
        if table.table_id.startswith("ordinary-inquiry-continuation-")
    }

    validated = {
        table_id: canonical_inquiry_population_metadata(context, page, table)
        for table_id, (page, table) in continuations.items()
    }
    assert {
        table_id: metadata["authority_mode"] if metadata is not None else None
        for table_id, metadata in validated.items()
    } == {
        "ordinary-inquiry-continuation-a": INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY,
        "ordinary-inquiry-continuation-b": INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY,
    }
    assert {
        table.metadata["canonical_section_owner"]["adjacency_proof"]["kind"]
        for _page, table in continuations.values()
    } == {"exact_printed_footer_table_edge"}

    records = _extract_inquiries(context)
    coverage = _inquiry_source_coverage(context)
    institution_sequences = sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "institution"
    )
    personal_sequences = sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "personal"
    )

    assert len(records) == 87
    assert institution_sequences == [
        sequence
        for sequence in range(1, 90)
        if sequence not in {4, 66, 67}
    ]
    assert personal_sequences == [1]
    assert len(institution_sequences) == len(set(institution_sequences))
    assert coverage["sequence_endpoints"] == {
        "institution": 89,
        "personal": 1,
    }
    inferred = {
        issue["candidate_value"]["normalized_sequence"]
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_sequence_inferred_from_row_order"
    }
    assert inferred == {27, 46, 87}
    assert not any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_owner_ordinal_corrected"
        for issue in context._personal_detail_extraction_issues
    )

    emitted_business_rows = {
        (
            row["inquiry_date"],
            row["institution"],
            row["reason"],
        )
        for row in records
        if row["inquiry_type"] == "institution"
    }
    localized_omissions = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_multiple_missing_sequences_unresolved"
        and (
            issue["observed_value"]["row"]["inquiry_date"],
            issue["observed_value"]["row"]["institution"],
            issue["observed_value"]["row"]["reason"],
        )
        not in emitted_business_rows
    ]
    assert {
        issue["observed_value"]["row"]["raw_inquiry_date"]
        for issue in localized_omissions
    } == {"2022.01.04", "2022.03.10", "2022.03.11"}
    assert all(
        issue["reason_codes"][-1] == "record_not_emitted"
        for issue in localized_omissions
    )


def test_lin_production_shaped_replay_carries_all_headerless_rows() -> None:
    projection, evidence, resolution = _lin_production_shaped_projection()
    context = _lin_production_shaped_consumer_context(
        projection,
        evidence,
        resolution,
    )

    owners = {
        table.table_id: table.metadata.get("canonical_section_owner")
        for page in context.pages
        for table in page.tables
    }
    assert set(owners) == {"pt_26_3", "pt_27_0", "pt_28_0", "pt_28_1"}
    assert all(owner is not None for owner in owners.values())
    assert {
        owners[table_id]["adjacency_proof"]["kind"]
        for table_id in ("pt_27_0", "pt_28_0")
    } == {"exact_printed_footer_schema_carry_bridge"}
    for page in context.pages:
        for table in page.tables:
            if table.table_id not in {"pt_27_0", "pt_28_0"}:
                continue
            metadata = canonical_inquiry_population_metadata(
                context,
                page,
                table,
            )
            assert metadata is not None
            assert (
                metadata["authority_mode"]
                == INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY
            )

    records = _extract_inquiries(context)
    coverage = _inquiry_source_coverage(context)
    institution_sequences = sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "institution"
    )
    # Discovery conserves bounded sequence noise as field-local uncertainty;
    # the deterministic repair pass materializes 8 and 9.  This test exercises
    # schema carry only, so it must not bypass the agreed repair boundary.
    assert len(records) == 85
    assert institution_sequences == [
        sequence
        for sequence in range(1, 90)
        if sequence not in {4, 8, 9, 66, 67}
    ]
    assert [
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "personal"
    ] == [1]
    assert coverage["sequence_endpoints"] == {
        "institution": 89,
        "personal": 1,
    }
    assert {
        issue["candidate_value"]["normalized_sequence"]
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_sequence_inferred_from_row_order"
    } == {27, 46, 87}


@pytest.mark.parametrize(
    "defect",
    (
        "footer_geometry",
        "proof_kind",
        "resolution_missing",
        "resolution_non_authoritative",
        "source_page",
    ),
)
def test_lin_production_schema_bridge_fails_closed_after_consumer_tamper(
    defect: str,
) -> None:
    projection, evidence, resolution = _lin_production_shaped_projection()
    context = _lin_production_shaped_consumer_context(
        projection,
        evidence,
        resolution,
    )
    page = next(page for page in context.pages if page.page_number == 27)
    table = next(table for table in page.tables if table.table_id == "pt_27_0")

    if defect == "footer_geometry":
        page.texts[-1].bbox = [100.0, 10.0, 200.0, 20.0]
    elif defect == "proof_kind":
        table.metadata["canonical_section_owner"]["adjacency_proof"][
            "kind"
        ] = "exact_printed_footer_table_edge"
    elif defect == "resolution_missing":
        context.reading_order_resolution = None
    elif defect == "resolution_non_authoritative":
        context.reading_order_resolution["authoritative"] = False
    elif defect == "source_page":
        page.source_page_number = 15
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(defect)

    assert canonical_inquiry_population_metadata(context, page, table) is None
    assert canonical_table_role(context, page, table) is None


@pytest.mark.parametrize("continuation_decision", [None, True])
def test_lin_production_schema_bridge_ignores_entity_merge_vote(
    continuation_decision: bool | None,
) -> None:
    projection, evidence, resolution = _lin_production_shaped_projection()
    context = _lin_production_shaped_consumer_context(
        projection,
        evidence,
        resolution,
    )
    page = next(page for page in context.pages if page.page_number == 27)
    table = next(table for table in page.tables if table.table_id == "pt_27_0")
    context.tables_continue = lambda _left, _right: continuation_decision

    metadata = canonical_inquiry_population_metadata(context, page, table)

    assert metadata is not None
    assert metadata["authority_mode"] == INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY
    assert canonical_table_role(context, page, table) == "annotations_and_inquiries"


@pytest.mark.parametrize(
    "defect",
    (
        "derived_non_sequence_cell",
        "derived_sequence_has_text",
        "unresolved_sequence_geometry",
    ),
)
def test_lin_production_schema_bridge_rejects_unsealed_seed_cells(
    defect: str,
) -> None:
    projection, evidence, resolution = _lin_production_shaped_projection()
    context = _lin_production_shaped_consumer_context(
        projection,
        evidence,
        resolution,
    )
    seed_page = next(page for page in context.pages if page.page_number == 26)
    seed = next(table for table in seed_page.tables if table.table_id == "pt_26_3")
    page = next(page for page in context.pages if page.page_number == 27)
    table = next(table for table in page.tables if table.table_id == "pt_27_0")
    geometry = seed.metadata["geometry"]

    if defect == "derived_non_sequence_cell":
        geometry["cell_geometry_status"][4][1] = "derived"
        geometry["cell_bboxes"][4][1] = None
        geometry["cell_evidence_ids"][4][1] = []
    elif defect == "derived_sequence_has_text":
        seed.metadata["raw_rows"][4][0] = "4"
    elif defect == "unresolved_sequence_geometry":
        geometry["cell_geometry_status"][4][0] = "unresolved"
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(defect)

    assert canonical_inquiry_population_metadata(context, page, table) is None
    assert canonical_table_role(context, page, table) is None


@pytest.mark.parametrize(
    "defect",
    (
        "footer_gap",
        "footer_geometry",
        "missing_resolution",
        "non_authoritative_resolution",
    ),
)
def test_lin_production_schema_bridge_requires_sealed_projection_order(
    defect: str,
) -> None:
    pages, evidence, resolution = _lin_production_shaped_inquiry_case()
    if defect == "footer_gap":
        evidence[1]["lines"][-1]["text"] = "第29页,共30页"
        pages[1].texts[-1]["text"] = "第29页,共30页"
    elif defect == "footer_geometry":
        evidence[1]["lines"][-1]["bbox"] = [100.0, 10.0, 200.0, 20.0]
        pages[1].texts[-1]["bbox"] = [100.0, 10.0, 200.0, 20.0]
    elif defect == "missing_resolution":
        resolution = None
    elif defect == "non_authoritative_resolution":
        resolution["authoritative"] = False
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(defect)

    projection = _build(
        pages,
        evidence,
        force_table_continuation_false=True,
        reading_order_resolution=resolution,
    )
    assert not any(
        table.table_id == "pt_27_0"
        for page in projection.pages
        for table in page.tables
    )


@pytest.mark.parametrize(
    "defect",
    (
        "authority_escalation",
        "contradictory_printed_mapping",
        "continuation_decision_false",
        "continuation_decision_missing",
        "continuation_decision_unresolved",
        "footer_geometry_tamper",
        "header_lattice_tamper",
        "non_authoritative_resolution",
        "prior_table_tamper",
        "proof_kind_tamper",
        "proof_kind_local_tamper",
        "role_slot_tamper",
        "source_page_tamper",
    ),
)
def test_lin_schema_carry_continuation_fails_closed_after_projection_tamper(
    defect: str,
) -> None:
    projection, evidence = _lin_shaped_inquiry_projection()
    context = _lin_shaped_consumer_context(projection, evidence)
    seed_page = next(page for page in context.pages if page.page_number == 5)
    target_page = next(page for page in context.pages if page.page_number == 6)
    seed = next(
        table
        for table in seed_page.tables
        if table.table_id == "ordinary-inquiry-seed"
    )
    target = next(
        table
        for table in target_page.tables
        if table.table_id == "ordinary-inquiry-continuation-a"
    )
    owner = target.metadata["canonical_section_owner"]

    if defect == "authority_escalation":
        owner["authority_mode"] = INQUIRY_AUTHORITY_CLOSED_PHYSICAL_ORDINAL
    elif defect == "contradictory_printed_mapping":
        context.reading_order_resolution["printed_page_by_logical"][6] = 7
    elif defect == "continuation_decision_false":
        context.tables_continue = lambda _left, _right: False
    elif defect == "continuation_decision_missing":
        context.tables_continue = None
    elif defect == "continuation_decision_unresolved":
        context.tables_continue = lambda _left, _right: None
    elif defect == "footer_geometry_tamper":
        target_page.texts[-1].bbox = [100.0, 10.0, 200.0, 20.0]
    elif defect == "header_lattice_tamper":
        seed.metadata["geometry"]["cell_geometry_status"][0][0] = "derived"
    elif defect == "non_authoritative_resolution":
        context.reading_order_resolution["authoritative"] = False
    elif defect == "prior_table_tamper":
        owner["prior_table_id"] = "unrelated-table"
    elif defect == "proof_kind_tamper":
        owner["adjacency_proof"]["kind"] = "rank_only"
    elif defect == "proof_kind_local_tamper":
        owner["adjacency_proof"]["kind"] = (
            "local_paired_topology_entity_edge"
        )
    elif defect == "role_slot_tamper":
        owner["inquiry_role_columns"] = {
            "sequence": 1,
            "inquiry_date": 0,
            "institution": 2,
            "reason": 3,
        }
    elif defect == "source_page_tamper":
        target_page.source_page_number = 99
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(defect)

    assert canonical_inquiry_population_metadata(context, target_page, target) is None
    assert canonical_table_role(context, target_page, target) is None


def test_ye_shaped_inquiry_chain_registers_all_112_physical_rows() -> None:
    pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case()

    projection = _build(
        pages,
        evidence,
        topology=topology,
        entity_context=entity_context,
        force_table_continuation_false=True,
    )

    assert projection.unresolved_pages == ()
    owners = {
        table.table_id: table.metadata["canonical_section_owner"] for page in projection.pages for table in page.tables
    }
    assert set(owners) == {
        "ye-inquiry:1-36",
        "ye-inquiry:37-76",
        "ye-inquiry:77-96",
        "ye-inquiry:personal-1-16",
    }
    assert owners["ye-inquiry:1-36"]["binding"] == ("exact_inquiry_subsection_and_bounded_header_residue")
    assert owners["ye-inquiry:1-36"]["physical_field_omission_rows"] == [5, 8, 9, 29]
    assert owners["ye-inquiry:1-36"]["sequence_field_anomalies"] == [
        {
            "row": row,
            "expected_sequence": row,
            "raw_sequence": "",
            "status": "physical_field_omission",
        }
        for row in (5, 8, 9)
    ] + [
        {
            "row": 27,
            "expected_sequence": 27,
            "raw_sequence": "27多",
            "status": "unparsed_raw_sequence",
        },
        {
            "row": 29,
            "expected_sequence": 29,
            "raw_sequence": "",
            "status": "physical_field_omission",
        },
    ]
    first_projected = next(
        table
        for page in projection.pages
        for table in page.tables
        if table.table_id == "ye-inquiry:1-36"
    )
    assert [first_projected.metadata["raw_rows"][row][0] for row in (5, 8, 9, 27, 29)] == [
        "",
        "",
        "",
        "27 多",
        "",
    ]
    assert owners["ye-inquiry:37-76"]["binding"] == ("authoritative_prior_inquiry_table_continuation")
    assert owners["ye-inquiry:77-96"]["binding"] == ("authoritative_prior_inquiry_table_continuation")
    assert owners["ye-inquiry:37-76"]["sequence_field_anomalies"] == [
        {
            "row": 11,
            "expected_sequence": 48,
            "raw_sequence": "福48",
            "status": "unparsed_raw_sequence",
        },
        {
            "row": 30,
            "expected_sequence": 67,
            "raw_sequence": "",
            "status": "physical_field_omission",
        },
        {
            "row": 32,
            "expected_sequence": 69,
            "raw_sequence": "K69",
            "status": "unparsed_raw_sequence",
        },
    ]
    assert owners["ye-inquiry:77-96"]["sequence_field_anomalies"] == [
        {
            "row": 12,
            "expected_sequence": 89,
            "raw_sequence": "789",
            "status": "unparsed_raw_sequence",
        }
    ]
    assert (
        sum(
            len(table.metadata["raw_rows"])
            - (1 if table.table_id in {"ye-inquiry:1-36", "ye-inquiry:personal-1-16"} else 0)
            for page in projection.pages
            for table in page.tables
        )
        == 112
    )


@pytest.mark.parametrize("continuation_decision", [False, None, True])
def test_ye_shaped_exact_footer_chain_does_not_require_entity_merge(
    continuation_decision: bool | None,
) -> None:
    pages, evidence, topology, _entity_context = (
        _ye_shaped_inquiry_registration_case()
    )

    projection = _build(
        pages,
        evidence,
        topology=topology,
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
            "identity_fallback": False,
            "printed_total": 31,
            "printed_page_by_logical": {27: 27, 28: 28, 29: 29},
        },
        entity_context=None,
        force_table_continuation_false=continuation_decision is False,
        force_table_continuation_unknown=continuation_decision is None,
    )

    assert projection.unresolved_pages == ()
    owners = {
        table.table_id: table.metadata["canonical_section_owner"]
        for page in projection.pages
        for table in page.tables
    }
    assert owners["ye-inquiry:37-76"]["adjacency_proof"]["kind"] == (
        "exact_printed_footer_schema_carry_bridge"
    )
    assert owners["ye-inquiry:77-96"]["adjacency_proof"]["kind"] == (
        "exact_printed_footer_schema_carry_bridge"
    )

    context = _ye_shaped_consumer_context(projection, evidence, topology)
    context.entity_context = None
    context.tables_continue = lambda _left, _right: continuation_decision
    continuation_tables = [
        (page, table)
        for page in context.pages
        for table in page.tables
        if table.table_id in {"ye-inquiry:37-76", "ye-inquiry:77-96"}
    ]
    assert all(
        (metadata := canonical_inquiry_population_metadata(context, page, table))
        is not None
        and metadata["authority_mode"] == INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY
        for page, table in continuation_tables
    )
    records = _extract_inquiries(context)
    assert len(records) == 112
    assert sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "institution"
    ) == list(range(1, 97))
    assert sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "personal"
    ) == list(range(1, 17))


def test_ye_shaped_institution_tail_accepts_exact_date_institution_colspans() -> None:
    pages, evidence, topology, _entity_context = (
        _ye_shaped_inquiry_registration_case("merged_date_institution_tail")
    )

    projection = _build(
        pages,
        evidence,
        topology=topology,
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
            "identity_fallback": False,
            "printed_total": 31,
            "printed_page_by_logical": {27: 27, 28: 28, 29: 29},
        },
        entity_context=None,
        force_table_continuation_false=True,
    )

    owners = {
        table.table_id: table.metadata["canonical_section_owner"]
        for page in projection.pages
        for table in page.tables
    }
    assert projection.unresolved_pages == ()
    assert owners["ye-inquiry:77-96"]["population_start"] == 77
    assert owners["ye-inquiry:77-96"]["population_endpoint"] == 96
    assert owners["ye-inquiry:77-96"]["physical_field_omission_rows"][-4:] == [
        16,
        17,
        18,
        19,
    ]


def test_ye_shaped_exact_collapsed_personal_header_is_consumable_without_profile_vote() -> None:
    pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case()
    personal = next(
        table
        for table in pages[-1].tables
        if table.table_id == "ye-inquiry:personal-1-16"
    )
    personal.metadata["raw_rows"][0] = [
        "编号 查询日期",
        "",
        "查询机构",
        "查询原因",
    ]
    geometry = personal.metadata["geometry"]
    left_box = geometry["cell_bboxes"][0][0]
    right_box = geometry["cell_bboxes"][0][1]
    merged_box = [left_box[0], left_box[1], right_box[2], left_box[3]]
    merged_ids = [
        *geometry["cell_evidence_ids"][0][0],
        *geometry["cell_evidence_ids"][0][1],
    ]
    geometry["cell_bboxes"][0][0] = merged_box
    geometry["cell_evidence_ids"][0][0] = merged_ids
    geometry["cell_bboxes"][0][1] = None
    geometry["cell_geometry_status"][0][1] = "derived"
    geometry["cell_evidence_ids"][0][1] = []
    geometry["cell_spans"] = [
        {
            "row": 0,
            "col": 0,
            "row_span": 1,
            "col_span": 2,
            "bbox": list(merged_box),
            "evidence_ids": list(merged_ids),
        }
    ]
    dates = [
        "2024.01.15",
        "2023.10.28",
        "2023.09.22",
        "2023.08.28",
        "2023.07.20",
        "2023.06.16",
        "离 2023.04.25",
        "2023.03.26",
        "2023.02.17",
        "2023.01.04",
        "2022.12.11",
        "2022.10.11",
        "2022.07.21",
        "2022.07.05",
        "2022.05.30",
        "2022.04.27",
    ]
    institutions = ["本人"] * 16
    institutions[11:14] = ["您 本人 业", "真 本人", "苏 本人 6"]
    for sequence, (date, institution) in enumerate(
        zip(dates, institutions, strict=True),
        start=1,
    ):
        row = personal.metadata["raw_rows"][sequence]
        row[0] = "8 囍" if sequence == 8 else "版" if sequence == 9 else str(sequence)
        row[1] = date
        row[2] = institution
        row[3] = "本人查询(自助查询机)"

    projection = _build(
        pages,
        evidence,
        topology=topology,
        entity_context=entity_context,
        force_table_continuation_false=True,
    )
    context = _ye_shaped_consumer_context(projection, evidence, topology)
    page = next(page for page in context.pages if page.page_number == 29)
    table = next(
        table
        for table in page.tables
        if table.table_id == "ye-inquiry:personal-1-16"
    )

    metadata = canonical_inquiry_population_metadata(context, page, table)
    discovery_records = _extract_inquiries(context)
    discovery_issues = deepcopy(context._personal_detail_extraction_issues)

    assert table.metadata["canonical_section_owner"]["header_binding"] == (
        "exact_collapsed_colspan_header_lattice"
    )
    assert metadata is not None
    assert metadata["authority_mode"] == INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY
    assert canonical_table_role(context, page, table) == "annotations_and_inquiries"
    assert len(discovery_records) == 112
    unresolved_date_row = next(
        row
        for row in discovery_records
        if row["inquiry_type"] == "personal" and row["sequence"] == 7
    )
    assert unresolved_date_row["inquiry_date"] is None
    assert unresolved_date_row["institution"] == "本人"
    assert unresolved_date_row["extraction_status"] == "review"
    assert unresolved_date_row["_unresolved_fields"] == ["inquiry_date"]
    unresolved_date_issue = next(
        issue
        for issue in discovery_issues
        if issue.get("issue_code") == "candidate_b_inquiry_row_cells_unresolved"
        and issue.get("observed_value", {}).get("sequence") == 7
        and issue.get("field_name") == "inquiry_date"
    )
    assert unresolved_date_issue["target_record_id"] == unresolved_date_row["inquiry_id"]
    assert unresolved_date_issue["reason_codes"][-2:] == [
        "physical_record_identity_conserved",
        "field_local_value_withheld",
    ]

    coordinator = BusinessUncertaintyRepairCoordinator(context)
    plan = coordinator.plan(
        {"inquiry_records": deepcopy(discovery_records)},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=discovery_issues,
    )
    date_repair = next(
        repair
        for repair in plan.field_repairs
        if repair.field_name == "inquiry_date"
        and repair.observed_value == "离 2023.04.25"
    )
    assert date_repair.mode == "deterministic"
    assert date_repair.candidate_value == "2023-04-25"
    institution_repairs = [
        repair
        for repair in plan.field_repairs
        if repair.field_name == "institution"
        and repair.observed_value in {"您 本人 业", "真 本人", "苏 本人 6"}
    ]
    assert len(institution_repairs) == 3
    assert {repair.mode for repair in institution_repairs} == {
        "context_rich_reocr"
    }

    ocr_calls: list[tuple[set[int], str]] = []

    def page_ocr_loader(
        logical_pages: set[int],
        *,
        reason: str,
    ) -> list[dict[str, object]]:
        ocr_calls.append((set(logical_pages), reason))
        acquired: list[dict[str, object]] = []
        for logical_page in sorted(logical_pages):
            repairs = [
                repair
                for repair in institution_repairs
                if any(
                    int(ref.get("logical_page") or 0) == logical_page
                    for ref in repair.source_refs
                )
            ]
            if not repairs:
                continue
            source_page = int(repairs[0].source_refs[0]["source_page"])
            lines = [
                {
                    "text": "本人查询记录明细",
                    "content": "本人查询记录明细",
                    "confidence": 0.99,
                    "bbox": [190.0, 575.0, 370.0, 600.0],
                    "evidence_ids": [
                        f"personal_detail_page_reocr:ye-{logical_page}:w0"
                    ],
                    "source": "personal_detail_page_reocr_once",
                }
            ]
            for index, repair in enumerate(repairs, start=1):
                ref = repair.source_refs[0]
                lines.append(
                    {
                        "text": "本人",
                        "content": "本人",
                        "confidence": 0.99,
                        "bbox": list(ref["bbox"]),
                        "evidence_ids": [
                            f"personal_detail_page_reocr:ye-{logical_page}:w{index}"
                        ],
                        "source": "personal_detail_page_reocr_once",
                    }
                )
            acquired.append(
                {
                    "page": logical_page,
                    "logical_page": logical_page,
                    "source_page": source_page,
                    "page_key": f"ye-{logical_page}",
                    "lines": lines,
                }
            )
        return acquired

    coordinator.resolve_page_evidence(
        plan,
        source_pages=evidence,
        page_ocr_loader=page_ocr_loader,
    )
    context._business_repair_plan = plan
    context._business_repair_active = True
    context._personal_detail_extraction_issues = []
    context._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(context)
    context.candidate_b_planned_field_repair = (
        PersonalDetailExtractionContext.candidate_b_planned_field_repair.__get__(
            context,
            type(context),
        )
    )
    context.candidate_b_field_repair = (
        PersonalDetailExtractionContext.candidate_b_field_repair.__get__(
            context,
            type(context),
        )
    )
    context._ocr_correction_overlay.install_business_repair_evidence(
        plan.page_evidence.values(),
        affected_pages=plan.affected_pages,
        allowed_target_refs=(
            {**dict(ref), "field_name": repair.field_name}
            for repair in plan.field_repairs
            for ref in repair.source_refs
        ),
    )

    records = _extract_inquiries(context)

    assert ({29}, "business_field_context_rich_reocr_required") in ocr_calls
    assert all(
        reason == "business_field_context_rich_reocr_required"
        for _pages, reason in ocr_calls
    )
    assert len(records) == 112
    assert sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "personal"
    ) == list(range(1, 17))
    personal_records = sorted(
        (row for row in records if row["inquiry_type"] == "personal"),
        key=lambda row: row["sequence"],
    )
    assert [row["inquiry_date"] for row in personal_records] == [
        date.replace("离 ", "").replace(".", "-") for date in dates
    ]
    assert {row["institution"] for row in personal_records} == {"本人"}


def test_ye_shaped_owned_populations_emit_all_112_rows_with_local_sequence_repairs() -> None:
    pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case()
    projection = _build(
        pages,
        evidence,
        topology=topology,
        entity_context=entity_context,
        force_table_continuation_false=True,
    )
    context = _ye_shaped_consumer_context(projection, evidence, topology)

    continuation_tables = [
        (page, table)
        for page in context.pages
        for table in page.tables
        if table.table_id in {"ye-inquiry:37-76", "ye-inquiry:77-96"}
    ]
    assert {
        table.metadata["canonical_section_owner"]["adjacency_proof"]["kind"]
        for _page, table in continuation_tables
    } == {"local_paired_topology_entity_edge"}
    assert all(
        (metadata := canonical_inquiry_population_metadata(context, page, table))
        is not None
        and metadata["authority_mode"]
        == INQUIRY_AUTHORITY_CLOSED_PHYSICAL_ORDINAL
        for page, table in continuation_tables
    )

    records = _extract_inquiries(context)
    coverage = _inquiry_source_coverage(context)

    assert len(records) == 112
    assert sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "institution"
    ) == list(range(1, 97))
    assert sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "personal"
    ) == list(range(1, 17))
    assert coverage["sequence_endpoints"] == {
        "institution": 96,
        "personal": 16,
    }
    assert {
        inquiry_type: len(sequences)
        for inquiry_type, sequences in coverage["observed_sequences"].items()
    } == {
        "institution": 96,
        "personal": 16,
    }
    repaired = {
        row["sequence"]: row["canonical_raw"]["sequence"]
        for row in records
        if "sequence" in (row.get("canonical_raw") or {})
    }
    assert repaired == {
        5: "",
        8: "",
        9: "",
        27: "27 多",
        29: "",
        48: "福 48",
        67: "",
        69: "K69",
        89: "789",
    }
    repaired_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_sequence_owner_ordinal_corrected"
    ]
    assert {
        issue["candidate_value"]["normalized_sequence"]
        for issue in repaired_issues
    } == set(repaired)
    assert all(
        len(issue["source_refs"]) == 2
        and issue["source_refs"][1]["field_name"] == "sequence"
        and issue["source_refs"][1]["binding"]
        == "validated_canonical_population_column"
        for issue in repaired_issues
    )
    unresolved_dates = {
        row["sequence"]: row["canonical_raw"]["inquiry_date"]
        for row in records
        if "inquiry_date" in (row.get("canonical_raw") or {})
    }
    assert unresolved_dates == {
        19: "2024.01.19Pd",
        22: "心2024.01.22",
        26: "证2024.01.26",
    }
    assert all(
        row["inquiry_date"] is None
        for row in records
        if row["sequence"] in unresolved_dates
        and row["inquiry_type"] == "institution"
    )
    assert {
        issue["target_record_id"]
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_owned_inquiry_row_field_unresolved"
    } == {
        row["inquiry_id"]
        for row in records
        if row["sequence"] in unresolved_dates
        and row["inquiry_type"] == "institution"
    }


@pytest.mark.parametrize(
    "defect",
    (
        "contradictory_printed_mapping",
        "entity_edge_tamper",
        "footer_geometry_tamper",
        "identity_kind_tamper",
        "non_authoritative_resolution",
        "source_page_tamper",
        "topology_audit_tamper",
    ),
)
def test_ye_local_paired_consumer_fails_closed_after_projection_tamper(
    defect: str,
) -> None:
    pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case()
    projection = _build(
        pages,
        evidence,
        topology=topology,
        entity_context=entity_context,
        force_table_continuation_false=True,
    )
    context = _ye_shaped_consumer_context(projection, evidence, topology)
    target_page = next(page for page in context.pages if page.page_number == 28)
    target = next(
        table
        for table in target_page.tables
        if table.table_id == "ye-inquiry:37-76"
    )
    baseline = canonical_inquiry_population_metadata(context, target_page, target)
    assert baseline is not None
    assert baseline["authority_mode"] == INQUIRY_AUTHORITY_CLOSED_PHYSICAL_ORDINAL

    if defect == "contradictory_printed_mapping":
        context.reading_order_resolution["printed_page_by_logical"][28] = 30
    elif defect == "entity_edge_tamper":
        context.entity_context = _ye_shaped_inquiry_entity_context(
            list(context.pages),
            missing_first_edge=True,
        )
    elif defect == "footer_geometry_tamper":
        footer = next(
            text
            for text in target_page.texts
            if "28" in str(getattr(text, "content", ""))
        )
        footer.bbox = [100.0, 10.0, 200.0, 20.0]
    elif defect == "identity_kind_tamper":
        target.metadata["canonical_section_owner"]["adjacency_proof"][
            "identity_kind"
        ] = "paired_inferred_current_footer"
    elif defect == "non_authoritative_resolution":
        context.reading_order_resolution.update(
            {
                "resolved": False,
                "authoritative": False,
                "identity_fallback": True,
            }
        )
    elif defect == "source_page_tamper":
        target_page.source_page_number = 99
    elif defect == "topology_audit_tamper":
        context.page_topology_audit = lambda: {
            **topology.audit(),
            "topology_frozen_before_reocr": False,
        }
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(defect)

    assert canonical_inquiry_population_metadata(context, target_page, target) is None
    assert canonical_table_role(context, target_page, target) is None


def test_ye_local_paired_proof_kind_cannot_escalate_to_exact_footer() -> None:
    pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case()
    projection = _build(
        pages,
        evidence,
        topology=topology,
        entity_context=entity_context,
        force_table_continuation_false=True,
    )
    context = _ye_shaped_consumer_context(projection, evidence, topology)
    target_page = next(page for page in context.pages if page.page_number == 28)
    target = next(
        table
        for table in target_page.tables
        if table.table_id == "ye-inquiry:37-76"
    )
    assert canonical_inquiry_population_metadata(context, target_page, target) is not None

    target.metadata["canonical_section_owner"]["adjacency_proof"]["kind"] = (
        "exact_printed_footer_table_edge"
    )
    context.entity_context = _ye_shaped_inquiry_entity_context(
        list(context.pages),
        missing_first_edge=True,
    )
    context.page_topology_audit = lambda: {
        **topology.audit(),
        "topology_frozen_before_reocr": False,
    }

    assert context.reading_order_resolution["authoritative"] is True
    assert context.tables_continue(
        "ye-inquiry:1-36",
        "ye-inquiry:37-76",
    ) is False
    assert canonical_inquiry_population_metadata(context, target_page, target) is None
    assert canonical_table_role(context, target_page, target) is None


def test_ye_local_exact_footer_proof_requires_false_continuation_decision() -> None:
    pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case()
    projection = _build(
        pages,
        evidence,
        topology=topology,
        entity_context=entity_context,
        force_table_continuation_false=True,
    )
    context = _ye_shaped_consumer_context(projection, evidence, topology)
    target_page = next(page for page in context.pages if page.page_number == 28)
    target = next(
        table
        for table in target_page.tables
        if table.table_id == "ye-inquiry:37-76"
    )
    owner = target.metadata["canonical_section_owner"]
    assert owner["adjacency_proof"]["kind"] == (
        "local_paired_topology_entity_edge"
    )
    assert owner["adjacency_proof"]["identity_kind"] == "exact_footer_pair"
    assert canonical_inquiry_population_metadata(context, target_page, target) is not None

    context.tables_continue = lambda _left, _right: True

    assert canonical_inquiry_population_metadata(context, target_page, target) is None
    assert canonical_table_role(context, target_page, target) is None


@pytest.mark.parametrize(
    ("table_id", "field", "value"),
    (
        ("ye-inquiry:1-36", "binding", "authoritative_prior_inquiry_table_continuation"),
        ("ye-inquiry:1-36", "population_endpoint", 35),
        ("ye-inquiry:37-76", "binding", "exact_inquiry_subsection_and_bounded_header_residue"),
        ("ye-inquiry:37-76", "prior_table_id", "unrelated-table"),
        ("ye-inquiry:37-76", "population_start", 38),
    ),
)
def test_ye_shaped_inquiry_binding_tampering_is_rejected(
    table_id: str,
    field: str,
    value: object,
) -> None:
    pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case()
    projection = _build(
        pages,
        evidence,
        topology=topology,
        entity_context=entity_context,
        force_table_continuation_false=True,
    )
    projected_pages = deepcopy(list(projection.pages))
    context = _ye_shaped_consumer_context(
        SimpleNamespace(pages=projected_pages),
        evidence,
        topology,
    )
    target_page = next(
        page
        for page in projected_pages
        if any(table.table_id == table_id for table in page.tables)
    )
    target_table = next(table for table in target_page.tables if table.table_id == table_id)
    assert canonical_table_role(context, target_page, target_table) == (
        "annotations_and_inquiries"
    )
    target_table.metadata["canonical_section_owner"][field] = value

    assert canonical_table_role(context, target_page, target_table) is None


def test_ye_shaped_inquiry_chain_fails_closed_on_seed_or_neighbor_defect() -> None:
    seed_defects = {
        "ambiguous_cell_owner",
        "interior_geometry_gap",
        "foreign_header_residue",
        "malformed_non_sequence_role",
        "overlapping_cell_geometry",
        "partial_subsection_heading",
        "wrong_first_endpoint",
        "wrong_last_endpoint",
    }
    for defect in (
        *sorted(seed_defects),
        "missing_first_edge",
        "missing_second_edge",
        "non_authoritative_identity",
        "nonconsecutive_footer",
        "restart_second_population",
        "gap_second_population",
        "topology_profile_mismatch",
        "wrong_third_start",
    ):
        pages, evidence, topology, entity_context = _ye_shaped_inquiry_registration_case(defect)

        projection = _build(
            pages,
            evidence,
            topology=topology,
            reading_order_resolution=(
                {
                    "resolved": False,
                    "authoritative": False,
                    "identity_fallback": True,
                    "printed_total": 31,
                    "printed_page_by_logical": {27: 27, 28: 28, 29: 29},
                }
                if defect == "non_authoritative_identity"
                else None
            ),
            entity_context=entity_context,
            continuations=(
                {
                    ("ye-inquiry:1-36", "ye-inquiry:37-76"),
                    ("ye-inquiry:37-76", "ye-inquiry:77-96"),
                }
                if defect == "non_authoritative_identity"
                else None
            ),
            force_table_continuation_false=defect != "non_authoritative_identity",
        )

        projected_table_ids = {table.table_id for page in projection.pages for table in page.tables}
        assert "ye-inquiry:77-96" not in projected_table_ids, defect
        if defect not in {
            "missing_second_edge",
            "topology_profile_mismatch",
            "wrong_third_start",
        }:
            assert "ye-inquiry:37-76" not in projected_table_ids, defect
        if defect in seed_defects:
            assert "ye-inquiry:1-36" not in projected_table_ids, defect
            assert 27 in projection.unresolved_pages, defect


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


def _paired_inferred_consumer_context(
    projection,
    evidence,
    topology,
    resolution,
):
    return SimpleNamespace(
        pages=list(projection.pages),
        reading_order_by_logical={50: 1, 60: 2},
        reading_order_resolution=deepcopy(resolution),
        page_topology=topology,
        tables_continue=lambda _left, _right: False,
        page_topology_audit=lambda: {
            **topology.audit(),
            "topology_frozen_before_reocr": True,
        },
        entity_context=_paired_inquiry_entity_context(
            list(projection.pages)
        ),
        corrected_evidence_pages=lambda: deepcopy(evidence),
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
    assert owner["adjacency_proof"]["kind"] == (
        "local_paired_topology_entity_edge"
    )
    assert owner["adjacency_proof"]["identity_kind"] == (
        "paired_inferred_current_footer"
    )
    assert resolution["authoritative"] is False

    context = _paired_inferred_consumer_context(
        projection,
        evidence,
        topology,
        resolution,
    )
    metadata = canonical_inquiry_population_metadata(
        context,
        current,
        current.tables[0],
    )
    assert metadata is not None
    assert metadata["authority_mode"] == INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY
    assert canonical_table_role(context, current, current.tables[0]) == (
        "annotations_and_inquiries"
    )
    assert [row["sequence"] for row in _extract_inquiries(context)] == [1, 2, 3]
    context.tables_continue = lambda _left, _right: True
    assert canonical_inquiry_population_metadata(
        context,
        current,
        current.tables[0],
    ) is not None


@pytest.mark.parametrize(
    "defect",
    (
        "contradictory_printed_mapping",
        "continuation_decision_tamper",
        "entity_edge_tamper",
        "identity_kind_tamper",
        "inferred_page_set_tamper",
        "source_page_tamper",
        "topology_audit_tamper",
    ),
)
def test_paired_inferred_consumer_fails_closed_after_projection_tamper(
    defect: str,
) -> None:
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
    context = _paired_inferred_consumer_context(
        projection,
        evidence,
        topology,
        resolution,
    )
    target_page = next(page for page in context.pages if page.page_number == 60)
    target = target_page.tables[0]
    assert canonical_inquiry_population_metadata(context, target_page, target) is not None

    if defect == "contradictory_printed_mapping":
        context.reading_order_resolution["printed_page_by_logical"][60] = 7
    elif defect == "continuation_decision_tamper":
        context.tables_continue = lambda _left, _right: None
    elif defect == "entity_edge_tamper":
        context.entity_context = _paired_inquiry_entity_context(
            list(context.pages),
            defect="missing_edge",
        )
    elif defect == "identity_kind_tamper":
        target.metadata["canonical_section_owner"]["adjacency_proof"][
            "identity_kind"
        ] = "exact_footer_pair"
    elif defect == "inferred_page_set_tamper":
        context.reading_order_resolution["paired_inferred_logical_pages"] = []
    elif defect == "source_page_tamper":
        target_page.source_page_number = 4
    elif defect == "topology_audit_tamper":
        context.page_topology_audit = lambda: {
            **topology.audit(),
            "topology_frozen_before_reocr": False,
        }
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(defect)

    assert canonical_inquiry_population_metadata(context, target_page, target) is None
    assert canonical_table_role(context, target_page, target) is None


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
