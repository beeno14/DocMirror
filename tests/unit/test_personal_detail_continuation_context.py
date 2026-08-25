from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    PersonalDetailTransitionPolicy,
    build_personal_detail_extraction_context,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _account_heading_for_table,
    _extract_accounts,
    _extract_employment_records,
    _extract_profile_detail_records,
    _extract_summary_datasets,
    _extract_table_accounts,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_residence_records as _extract_raw_residence_records,
)
from docmirror.plugins.credit_report.scanned_business import (
    _extract_residence_records as _extract_scanned_residence_records,
)
from docmirror.plugins.credit_report.scanned_business import extract_scanned_credit_accounts
from docmirror.plugins.credit_report.shared.entity_decoder import CreditReportUnit
from tests.unit.personal_detail_employment_test_support import employment_page


def _exact_grid_metadata(
    rows: list[list[str]],
    *,
    top: float = 20.0,
    widths: list[float] | None = None,
) -> dict[str, object]:
    width = max(len(row) for row in rows)
    if widths is None:
        widths = [100.0] * width
    assert len(widths) == width
    row_bands = [
        {"index": row, "y0": top + row * 10.0, "y1": top + (row + 1) * 10.0}
        for row in range(len(rows))
    ]
    col_bands: list[dict[str, float | int]] = []
    left = 20.0
    for column, column_width in enumerate(widths):
        col_bands.append({"index": column, "x0": left, "x1": left + column_width})
        left += column_width
    boxes: list[list[list[float] | None]] = []
    statuses: list[list[str]] = []
    evidence: list[list[list[str]]] = []
    for row_index, row in enumerate(rows):
        row_boxes: list[list[float] | None] = []
        row_statuses: list[str] = []
        row_evidence: list[list[str]] = []
        for column in range(width):
            row_boxes.append(
                [
                    col_bands[column]["x0"],
                    row_bands[row_index]["y0"],
                    col_bands[column]["x1"],
                    row_bands[row_index]["y1"],
                ]
            )
            row_statuses.append("exact")
            row_evidence.append([f"residence:{row_index}:{column}"] if row[column] else [])
        boxes.append(row_boxes)
        statuses.append(row_statuses)
        evidence.append(row_evidence)
    return {
        "raw_rows": rows,
        "geometry": {
            "row_bands": row_bands,
            "col_bands": col_bands,
            "cell_bboxes": boxes,
            "cell_geometry_status": statuses,
            "cell_evidence_ids": evidence,
            "cell_token_ids": evidence,
            "cell_spans": [],
        },
    }


def _extract_residence_records(parse_result: object) -> list[dict[str, object]]:
    """Attach the ordinary PBOC identity owner to legacy residence fixtures."""

    for page in getattr(parse_result, "pages", ()):
        page.canonical_template_id = "report_header_and_identity"
        for table in getattr(page, "tables", ()):
            table.metadata.setdefault(
                "canonical_template_id",
                "report_header_and_identity",
            )
    return _extract_raw_residence_records(parse_result)


def _unit(
    unit_id: str,
    page: int,
    kind: str,
    text: str,
    *,
    bbox: tuple[float, float, float, float],
    rows: tuple[tuple[str, ...], ...] = (),
) -> CreditReportUnit:
    return CreditReportUnit(
        unit_id=unit_id,
        page=page,
        order=0,
        source_index=0,
        kind=kind,  # type: ignore[arg-type]
        text=text,
        bbox=bbox,
        page_width=600,
        page_height=800,
        table_id=unit_id if kind == "table" else "",
        rows=rows,
    )


def _same_entity_context(
    table_order_by_id: dict[str, int],
    *,
    reading_order_by_logical: dict[int, int],
) -> PersonalDetailExtractionContext:
    context = object.__new__(PersonalDetailExtractionContext)
    units_by_table = {
        table_id: SimpleNamespace(
            unit_id=f"unit:{table_id}",
            page=order,
        )
        for table_id, order in table_order_by_id.items()
    }
    shared_entity = SimpleNamespace(entity_id="same-entity")
    unit_ids = {unit.unit_id for unit in units_by_table.values()}
    context.entity_context = SimpleNamespace(
        units=tuple(units_by_table.values()),
        table_unit_id=lambda table_id: (
            units_by_table[table_id].unit_id
            if table_id in units_by_table
            else None
        ),
        entity_for_unit=lambda unit_id: (
            shared_entity if unit_id in unit_ids else None
        ),
    )
    context.reading_order_by_logical = dict(reading_order_by_logical)
    context.source_page_by_logical = {
        page: page for page in reading_order_by_logical
    }
    context.reading_order_resolution = {
        "resolved": True,
        "authoritative": True,
        "basis": "complete_unique_printed_page_permutation",
    }
    context.evidence_unit_ids = {}
    context.parse_result = SimpleNamespace()
    return context


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            _unit(
                "table-left",
                1,
                "table",
                "账户标识 管理机构 账户状态",
                bbox=(20, 620, 580, 780),
                rows=(("账户标识", "管理机构", "账户状态"), ("A1", "示例银行", "正常")),
            ),
            _unit(
                "table-right",
                2,
                "table",
                "账户标识 管理机构 账户状态",
                bbox=(20, 20, 580, 220),
                rows=(("账户标识", "管理机构", "账户状态"), ("A1", "示例银行", "正常")),
            ),
            "same_table",
        ),
        (
            _unit(
                "table-left",
                1,
                "table",
                "账户标识 管理机构 账户状态",
                bbox=(20, 620, 580, 780),
                rows=(("账户标识", "管理机构", "账户状态"), ("A1", "示例银行", "正常")),
            ),
            _unit(
                "text-right",
                2,
                "text",
                "账户状态正常，管理机构示例银行",
                bbox=(20, 20, 580, 80),
            ),
            "table_to_text_related",
        ),
        (
            _unit(
                "text-left",
                1,
                "text",
                "账户标识A1，管理机构",
                bbox=(20, 700, 580, 780),
            ),
            _unit(
                "table-right",
                2,
                "table",
                "账户标识 管理机构 账户状态",
                bbox=(20, 20, 580, 220),
                rows=(("账户标识", "管理机构", "账户状态"), ("A1", "示例银行", "正常")),
            ),
            "text_to_table_related",
        ),
        (
            _unit(
                "text-left",
                1,
                "text",
                "账户标识A1，管理机构",
                bbox=(20, 700, 580, 780),
            ),
            _unit(
                "text-right",
                2,
                "text",
                "账户状态正常，管理机构示例银行",
                bbox=(20, 20, 580, 80),
            ),
            "same_text_section",
        ),
    ],
)
def test_personal_detail_policy_scores_all_cross_page_modalities(
    left: CreditReportUnit,
    right: CreditReportUnit,
    expected: str,
) -> None:
    decision = PersonalDetailTransitionPolicy().score((left,), right, None)

    assert decision[0].action == expected
    assert decision[0].score >= decision[1].score


def test_personal_detail_policy_semantically_vetoes_unrelated_tables() -> None:
    account = _unit(
        "account",
        1,
        "table",
        "账户标识 管理机构 账户状态",
        bbox=(20, 620, 580, 780),
        rows=(("账户标识", "管理机构", "账户状态"), ("A1", "示例银行", "正常")),
    )
    inquiry = _unit(
        "inquiry",
        2,
        "table",
        "查询日期 查询机构 查询原因",
        bbox=(20, 20, 580, 220),
        rows=(("查询日期", "查询机构", "查询原因"), ("2026-01-01", "示例银行", "贷后管理")),
    )

    hypotheses = PersonalDetailTransitionPolicy().score((account,), inquiry, None)

    assert hypotheses[0].action == "different_table"
    assert "personal_detail_semantic_veto" in hypotheses[0].signals


def test_personal_detail_context_uses_logical_pages_and_suppresses_table_owned_text() -> None:
    table_1 = SimpleNamespace(
        table_id="account-head",
        metadata={"raw_rows": [["账户标识", "管理机构", "账户状态"], ["A1", "示例银行", "正常"]]},
        headers=[],
        rows=[],
        bbox=[20, 600, 580, 780],
    )
    table_2 = SimpleNamespace(
        table_id="account-tail",
        metadata={"raw_rows": [["账户标识", "管理机构", "账户状态"], ["A1", "示例银行", "正常"]]},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 220],
    )
    evidence_line_1 = {"text": "账户标识 A1", "bbox": [30, 620, 200, 650]}
    evidence_line_2 = {"text": "账户状态 正常", "bbox": [30, 40, 200, 70]}
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=10,
                source_page_number=5,
                width=600,
                height=800,
                tables=[table_1],
                texts=[
                    SimpleNamespace(content="账户标识 A1", bbox=[30, 620, 200, 650]),
                    SimpleNamespace(content="第1页，共2页", bbox=[250, 780, 350, 795]),
                ],
            ),
            SimpleNamespace(
                page_number=11,
                source_page_number=5,
                width=600,
                height=800,
                tables=[table_2],
                texts=[
                    SimpleNamespace(content="账户状态 正常", bbox=[30, 40, 200, 70]),
                    SimpleNamespace(content="第2页，共2页", bbox=[250, 780, 350, 795]),
                ],
            ),
        ],
        entities=SimpleNamespace(
            domain_specific={
                "_page_evidence_bundles": [
                    {"page": 10, "source_page_number": 5, "local_structure_evidence": {"lines": [evidence_line_1]}},
                    {"page": 11, "source_page_number": 5, "local_structure_evidence": {"lines": [evidence_line_2]}},
                ]
            }
        ),
    )

    context = build_personal_detail_extraction_context(result)

    assert context.entity_context.content_conserved is True
    assert {unit.kind for unit in context.entity_context.units} == {"table"}
    assert context.source_page_by_logical == {10: 5, 11: 5}
    assert context.reading_order_by_logical == {10: 1, 11: 2}
    assert context.tables_continue("account-head", "account-tail") is True
    assert context.entity_context.entity_for_unit("personal_detail:table:p10:account-head").pages == (1, 2)
    assert context.allows_scanned_line_transition(10, evidence_line_1, 0, 11, evidence_line_2, 0) is True

    # Entity identity is evidence, but it cannot authorize a cross-page edge
    # after printed order has become explicitly non-authoritative.
    context.reading_order_resolution = {
        "resolved": False,
        "authoritative": False,
        "basis": "unresolved_identity_fallback",
    }
    assert (
        context.entity_context.entity_for_unit("personal_detail:table:p10:account-head")
        == context.entity_context.entity_for_unit("personal_detail:table:p11:account-tail")
    )
    assert context.tables_continue("account-head", "account-tail") is False
    assert context.allows_scanned_line_transition(10, evidence_line_1, 0, 11, evidence_line_2, 0) is False
    assert context.allows_scanned_line_transition(10, evidence_line_1, 0, 10, evidence_line_1, 0) is True


def test_cross_page_entity_identity_requires_forward_registered_adjacency() -> None:
    context = _same_entity_context(
        {"page-1": 1, "page-2": 2, "page-3": 3},
        reading_order_by_logical={1: 1, 2: 2, 3: 3},
    )
    lines = {
        page: {
            "text": f"page {page}",
            "bbox": [20, 20, 200, 40],
            "evidence_ids": [f"page-{page}"],
        }
        for page in (1, 2, 3)
    }
    context.evidence_unit_ids = {
        f"evidence:page-{page}": f"unit:page-{page}" for page in (1, 2, 3)
    }

    assert context.tables_continue("page-1", "page-2") is True
    assert context.tables_continue("page-1", "page-3") is False
    assert context.tables_continue("page-2", "page-1") is False
    assert (
        context.allows_scanned_line_transition(1, lines[1], 0, 2, lines[2], 0)
        is True
    )
    assert (
        context.allows_scanned_line_transition(1, lines[1], 0, 3, lines[3], 0)
        is False
    )
    assert (
        context.allows_scanned_line_transition(2, lines[2], 0, 1, lines[1], 0)
        is False
    )

    context.reading_order_by_logical = {1: 1, 3: 3}
    assert context.pages_adjacent_in_reading_order(1, 2) is False
    context.reading_order_by_logical = {1: 1, 2: 1, 3: 2}
    assert context.pages_adjacent_in_reading_order(2, 3) is False
    context.reading_order_by_logical = {1: 1, 2: 2, 3: 3}
    assert context.pages_adjacent_in_reading_order(True, 2) is False


def test_page_edge_geometry_cannot_create_a_cross_section_line_owner() -> None:
    context = _same_entity_context(
        {},
        reading_order_by_logical={1: 1, 2: 2},
    )
    context.page_topology = SimpleNamespace(
        geometry=lambda page: SimpleNamespace(height=800.0)
        if page in {1, 2}
        else None
    )
    public_tail = {
        "text": "住房公积金信息更新日期",
        "bbox": [20, 700, 580, 790],
        "evidence_ids": ["public-tail"],
    }
    inquiry_head = {
        "text": "机构查询记录明细",
        "bbox": [20, 10, 580, 80],
        "evidence_ids": ["inquiry-head"],
    }

    # Printed adjacency and matching edge geometry are not record or section
    # identity.  With no registered evidence-unit owners this stays unknown.
    assert (
        context.allows_scanned_line_transition(
            1,
            public_tail,
            0,
            2,
            inquiry_head,
            0,
        )
        is None
    )


def test_entity_unit_positions_preserve_nonidentity_logical_page_order() -> None:
    context = _same_entity_context(
        {"logical-20": 1, "logical-17": 2},
        reading_order_by_logical={20: 1, 17: 2},
    )
    left = {
        "text": "logical 20",
        "bbox": [20, 700, 200, 720],
        "evidence_ids": ["logical-20"],
    }
    right = {
        "text": "logical 17",
        "bbox": [20, 20, 200, 40],
        "evidence_ids": ["logical-17"],
    }
    context.evidence_unit_ids = {
        "evidence:logical-20": "unit:logical-20",
        "evidence:logical-17": "unit:logical-17",
    }

    assert context.tables_continue("logical-20", "logical-17") is True
    assert context.pages_adjacent_in_reading_order(20, 17) is True
    assert context.allows_scanned_line_transition(20, left, 0, 17, right, 0) is True


def test_residence_provider_does_not_join_across_authoritative_page_gap() -> None:
    residence = SimpleNamespace(
        table_id="residence-head",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "某市某区一号", "--", "自置", "2025.01.02"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 600, 580, 780],
    )
    provider = SimpleNamespace(
        table_id="residence-provider",
        metadata={
            "raw_rows": [
                ["编号", "数据发生机构名称"],
                ["1", "跨页错误银行"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 180],
    )
    pages = (
        SimpleNamespace(page_number=1, source_page_number=1, tables=[residence]),
        SimpleNamespace(page_number=2, source_page_number=2, tables=[]),
        SimpleNamespace(page_number=3, source_page_number=3, tables=[provider]),
    )
    context = _same_entity_context(
        {"residence-head": 1, "residence-provider": 3},
        reading_order_by_logical={1: 1, 2: 2, 3: 3},
    )
    context._canonical_layout_projection_cache = SimpleNamespace(pages=pages)
    context._personal_detail_extraction_issues = []

    records = _extract_residence_records(context)

    assert len(records) == 1
    assert records[0].get("data_provider") is None
    assert records[0]["extraction_status"] == "review"
    assert any(
        issue["issue_code"] == "candidate_b_residence_provider_missing"
        and issue["field_name"] == "data_provider"
        and issue["target_record_id"] == records[0]["residence_record_id"]
        for issue in context._personal_detail_extraction_issues
    )


def test_personal_detail_context_cache_is_single_pass_and_copy_on_read() -> None:
    empty = SimpleNamespace(pages=[], entities=SimpleNamespace(domain_specific={}))
    context = build_personal_detail_extraction_context(empty)
    calls = 0

    def build() -> dict[str, list[int]]:
        nonlocal calls
        calls += 1
        return {"rows": [1]}

    first = context.cached("sample", build)
    first["rows"].append(2)
    second = context.cached("sample", build)

    assert isinstance(context, PersonalDetailExtractionContext)
    assert calls == 1
    assert second == {"rows": [1]}


def test_personal_detail_context_exposes_lossless_runtime_evidence_plane() -> None:
    runtime_plane = SimpleNamespace(evidence=SimpleNamespace(text_atoms=[]))
    canonical_plane = SimpleNamespace(to_runtime=lambda: runtime_plane)
    source = SimpleNamespace(
        pages=[],
        entities=SimpleNamespace(domain_specific={}),
        evidence_plane=canonical_plane,
    )

    context = build_personal_detail_extraction_context(source)

    assert context.evidence_plane is runtime_plane


def test_personal_detail_context_removes_repeated_edge_furniture() -> None:
    pages = [
        SimpleNamespace(
            page_number=page,
            source_page_number=page,
            width=600,
            height=800,
            tables=[],
            texts=[SimpleNamespace(content="中国人民银行征信中心", bbox=[20, 10, 200, 30])],
        )
        for page in (1, 2)
    ]
    context = build_personal_detail_extraction_context(
        SimpleNamespace(pages=pages, entities=SimpleNamespace(domain_specific={}))
    )

    assert context.entity_context.units == ()
    assert len(context.entity_context.furniture_unit_ids) == 2


def test_native_account_extraction_obeys_cross_page_entity_veto() -> None:
    account = SimpleNamespace(
        table_id="account",
        metadata={"raw_rows": [["账户标识", "管理机构", "余额"], ["A1", "示例银行", "100"]]},
        headers=[],
        rows=[],
        bbox=[20, 600, 580, 780],
    )
    unrelated = SimpleNamespace(
        table_id="employment",
        metadata={"raw_rows": [["工作单位", "单位地址", "余额"], ["示例公司", "示例地址", "999"]]},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 220],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                tables=[account],
                texts=[],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                tables=[unrelated],
                texts=[],
            ),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )
    context = build_personal_detail_extraction_context(result)

    table_accounts, _repayments, _events = _extract_table_accounts(context)
    accounts, _repayments, _events = _extract_accounts(context)

    # The closed-world registration layer excludes the unrelated unregistered
    # table before entity construction.  An unavailable relation is treated as
    # a split by the account extractor, never as permission to merge.
    assert context.tables_continue("account", "employment") is False
    assert table_accounts == []
    # An unregistered table is not even a source-owned account observation;
    # neither its labels nor its proximity to another page can grant that role.
    assert accounts == []


def test_native_account_extraction_rejects_repayment_liability_table() -> None:
    liability = SimpleNamespace(
        table_id="repayment-liability",
        metadata={
            "raw_rows": [
                ["管理机构", "账户标识", "责任人类型", "还款责任金额", "保证合同编号"],
                ["样例银行", "A1", "保证人", "4,000,000", "G-001"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 220],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[liability], texts=[])],
    )

    accounts, repayments, events = _extract_accounts(result)

    assert accounts == []
    assert repayments == []
    assert events == []


def test_scanned_account_extraction_obeys_cross_page_entity_veto() -> None:
    bundles = [
        {
            "page": 1,
            "source_page_number": 7,
            "local_structure_evidence": {
                "page": 1,
                "source_page": 7,
                "width": 600,
                "height": 800,
                "lines": [
                    {
                        "text": "非循环贷账户",
                        "bbox": [20, 650, 580, 680],
                        "evidence_ids": ["account-family"],
                    },
                    {
                        "text": "账户 1 （通用贷款）",
                        "bbox": [20, 690, 580, 720],
                        "evidence_ids": ["account-anchor"],
                    },
                    {
                        "text": "账户状态 正常 管理机构 某银行",
                        "bbox": [20, 740, 580, 780],
                        "evidence_ids": ["account-detail"],
                    },
                ],
            },
        },
        {
            "page": 2,
            "source_page_number": 7,
            "local_structure_evidence": {
                "page": 2,
                "source_page": 7,
                "width": 600,
                "height": 800,
                "lines": [
                    {
                        "text": "工作单位 某公司 单位地址 某地址",
                        "bbox": [20, 20, 580, 60],
                        "evidence_ids": ["employment-detail"],
                    },
                ],
            },
        },
    ]
    result = SimpleNamespace(
        pages=[],
        entities=SimpleNamespace(domain_specific={"_page_evidence_bundles": bundles}),
    )
    context = build_personal_detail_extraction_context(result)

    accounts = extract_scanned_credit_accounts(context)

    assert len(accounts) == 1
    assert "工作单位" not in accounts[0]["raw_detail_text"]
    assert {line["logical_page"] for line in accounts[0]["raw_detail_lines"]} == {1}


@pytest.mark.parametrize(
    ("heading_ids", "anchor_ids", "anchor_bbox"),
    [
        ([], ["anchor"], [20, 40, 580, 65]),
        (["heading"], [], [20, 40, 580, 65]),
        (["replayed"], ["replayed"], [20, 40, 580, 65]),
        (["heading"], ["anchor"], [20, 40, float("inf"), 65]),
        (["heading"], ["anchor", "anchor"], [20, 40, 580, 65]),
    ],
)
def test_scanned_account_anchors_require_distinct_exact_source_owners(
    heading_ids: list[str],
    anchor_ids: list[str],
    anchor_bbox: list[float],
) -> None:
    result = SimpleNamespace(
        entities=SimpleNamespace(
            domain_specific={
                "_page_evidence_bundles": [
                    {
                        "page": 1,
                        "source_page_number": 1,
                        "local_structure_evidence": {
                            "lines": [
                                {
                                    "text": "（二）循环贷账户一",
                                    "bbox": [20, 10, 580, 35],
                                    "evidence_ids": heading_ids,
                                },
                                {
                                    "text": "账户 37（授信协议标识：GENERIC37）",
                                    "bbox": anchor_bbox,
                                    "evidence_ids": anchor_ids,
                                },
                            ]
                        },
                    }
                ]
            }
        )
    )

    assert extract_scanned_credit_accounts(result) == []


def test_personal_detail_subsection_heading_closes_previous_account_entity() -> None:
    first = SimpleNamespace(
        table_id="loan-account",
        metadata={"raw_rows": [["账户标识", "管理机构", "余额"], ["A1", "样例银行", "100"]]},
        headers=[],
        rows=[],
        bbox=[20, 620, 580, 780],
    )
    second = SimpleNamespace(
        table_id="revolving-account",
        metadata={"raw_rows": [["账户标识", "管理机构", "余额"], ["B1", "样例银行", "200"]]},
        headers=[],
        rows=[],
        bbox=[20, 60, 580, 220],
    )
    heading = SimpleNamespace(content="（三）循环贷账户一", bbox=[200, 20, 400, 45])
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                tables=[first],
                texts=[],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                tables=[second],
                texts=[heading],
            ),
        ],
        entities=SimpleNamespace(domain_specific={}),
    )

    context = build_personal_detail_extraction_context(result)

    assert context.tables_continue("loan-account", "revolving-account") is False
    heading_unit = next(unit for unit in context.entity_context.units if unit.text == "（三）循环贷账户一")
    assert heading_unit.kind == "heading"


def test_native_account_extraction_treats_unknown_cross_page_mapping_as_split() -> None:
    account = SimpleNamespace(
        table_id="account",
        metadata={"raw_rows": [["账户标识", "管理机构", "余额"], ["A1", "样例银行", "100"]]},
        headers=[],
        rows=[],
        bbox=[20, 600, 580, 780],
    )
    unknown = SimpleNamespace(
        table_id="unknown",
        metadata={"raw_rows": [["余额"], ["999"]]},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 100],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[account], texts=[]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[unknown], texts=[]),
        ],
        tables_continue=lambda _left, _right: None,
    )

    table_accounts, _repayments, _events = _extract_table_accounts(result)
    accounts, _repayments, _events = _extract_accounts(result)

    assert len(table_accounts) == 1
    assert table_accounts[0]["balance"] == 100
    assert accounts == []
    issues = result._personal_detail_extraction_issues
    assert any(
        issue["issue_code"]
        == "candidate_b_account_table_fragment_owner_unresolved"
        and issue["observed_value"]["continuation_decision"] is None
        and "table_fragment_withheld" in issue["reason_codes"]
        for issue in issues
    )
    assert any(
        issue["issue_code"] == "candidate_b_account_anchor_population_missing"
        and issue["target_record_id"]
        == table_accounts[0]["_table_observation_instance_id"]
        for issue in issues
    )


def test_summary_extraction_consumes_headerless_cross_page_fragment() -> None:
    head = SimpleNamespace(
        table_id="summary-head",
        metadata={
            "canonical_template_id": "information_summary",
            "raw_rows": [
                ["逾期（透支）信息汇总", "", "", "", ""],
                ["账户类型", "账户数", "月份数", "单月最高逾期/透支总额", "最长逾期/透支月数"],
                ["贷记卡账户", "2", "3", "25,484", "2"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 620, 580, 780],
    )
    tail = SimpleNamespace(
        table_id="summary-tail",
        metadata={
            "canonical_template_id": "information_summary",
            "raw_rows": [["准贷记卡账户", "--", "--", "--", "--"]],
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 80],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="information_summary",
                tables=[head],
                texts=[],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                canonical_template_id="information_summary",
                tables=[tail],
                texts=[],
            ),
        ],
        tables_continue=lambda left, right: (left, right) == ("summary-head", "summary-tail"),
    )

    records, cells = _extract_summary_datasets(result)

    assert len(records) == 1
    assert records[0]["source_row_count"] == 2
    assert len(records[0]["source_refs"]) == 2
    assert any(
        cell["value"] == "准贷记卡账户" and cell["column_label"] == "账户类型" and cell["row_index"] == 2
        for cell in cells
    )
    assert not any(cell["value"] == "账户类型" for cell in cells)


def test_summary_extraction_withholds_values_under_unknown_or_shifted_header() -> None:
    table = SimpleNamespace(
        table_id="summary-damaged",
        metadata={
            "canonical_template_id": "information_summary",
            "raw_rows": [
                ["逾期（透支）信息汇总", "", ""],
                ["", "账户数", "OCR损坏的金额标题"],
                ["贷记卡账户", "2", "25,484"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 220],
    )
    context = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="information_summary",
                tables=[table],
                texts=[],
            )
        ],
        _personal_detail_extraction_issues=[],
    )

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert cells == []
    assert any(
        issue.get("issue_code") == "candidate_b_summary_layout_unresolved"
        and "header_fill_inference_forbidden" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_card_summary_uses_finite_columns_when_leaf_headers_are_blank() -> None:
    table = SimpleNamespace(
        table_id="credit-card-summary",
        metadata={
            "canonical_template_id": "information_summary",
            "raw_rows": [
                ["贷记卡账户信息汇总", "", "", "", "", "", ""],
                ["发卡机构数", "账户数", "授信总额", "", "", "已用额度", "最近6个月平均使用额度"],
                ["2", "3", "62,000", "50,000", "12,000", "55,000", "45,000"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 220],
    )
    context = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="information_summary",
                tables=[table],
                texts=[],
            )
        ],
        _personal_detail_extraction_issues=[],
    )

    _records, cells = _extract_summary_datasets(context)

    assert [(cell["column_label"], cell["value"]) for cell in cells] == [
        ("发卡机构数", "2"),
        ("账户数", "3"),
        ("授信总额", "62,000"),
        ("单家机构最高授信额", "50,000"),
        ("单家机构最低授信额", "12,000"),
        ("已用额度", "55,000"),
        ("最近6个月平均使用额度", "45,000"),
    ]
    assert context._personal_detail_extraction_issues == []


def test_credit_business_overview_does_not_emit_merged_group_label_as_business_type() -> None:
    table = SimpleNamespace(
        table_id="business-overview",
        metadata={
            "canonical_template_id": "information_summary",
            "raw_rows": [
                ["信用业务概要", "", "", ""],
                ["", "业务类型", "账户数", "首笔业务发放月份"],
                ["贷款", "个人住房贷款", "2", "2017.06"],
                ["信用卡", "2 贷记卡 n", "22", "2007.01"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 220],
    )
    context = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="information_summary",
                tables=[table],
                texts=[],
            )
        ],
        _personal_detail_extraction_issues=[],
    )

    _records, cells = _extract_summary_datasets(context)

    assert not any(cell["value"] in {"贷款", "信用卡"} for cell in cells)
    assert [cell["column_label"] for cell in cells if cell["column_index"] == 2] == [
        "业务类型",
        "业务类型",
    ]


def test_residence_provider_continuation_uses_entity_and_sequence_not_page_number() -> None:
    residence_rows = [
        ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
        ["7", "某市某区某路7号", "010-12345678", "租房", "2025.01.02"],
    ]
    residence = SimpleNamespace(
        table_id="residence",
        metadata=_exact_grid_metadata(residence_rows, widths=[55.0, 231.0, 104.0, 81.0, 119.0]),
        headers=[],
        rows=[],
        bbox=[20, 600, 580, 780],
    )
    provider = SimpleNamespace(
        table_id="provider",
        metadata=_exact_grid_metadata([["7", "样例银行"]], widths=[49.0, 307.0]),
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 80],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=10, source_page_number=10, tables=[residence]),
            SimpleNamespace(page_number=11, source_page_number=11, tables=[provider]),
        ],
        tables_continue=lambda left, right: (left, right) == ("residence", "provider"),
    )

    records = _extract_residence_records(result)

    assert len(records) == 1
    assert records[0]["sequence"] == 7
    assert records[0]["data_provider"] == "样例银行"


@pytest.mark.parametrize(
    ("page_role", "table_role", "canonical_page_role"),
    [
        ("public_information", "public_information", ""),
        ("report_header_and_identity", "public_information", ""),
        ("public_information", "report_header_and_identity", ""),
        (
            "mixed_pboc_sections",
            "report_header_and_identity",
            "mixed_pboc_sections",
        ),
    ],
)
def test_residence_labels_cannot_bypass_canonical_identity_owner(
    page_role: str,
    table_role: str,
    canonical_page_role: str,
) -> None:
    metadata: dict[str, object] = {
        "raw_rows": [
            ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
            ["1", "某市某区一号", "010-12345678", "自置", "2025.01.02"],
        ],
        "canonical_template_id": table_role,
    }
    if canonical_page_role:
        metadata["canonical_page_template_id"] = canonical_page_role
    table = SimpleNamespace(
        table_id="foreign-residence-shaped",
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 160],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=7,
                source_page_number=7,
                canonical_template_id=page_role,
                tables=[table],
                texts=[],
            )
        ],
        tables_continue=lambda _left, _right: False,
    )

    assert _extract_raw_residence_records(result) == []


@pytest.mark.parametrize(
    ("header", "population", "widths"),
    [
        (
            ["信息更新日期", "居住状况", "居住地址", "编号", "住宅电话"],
            1,
            [117.0, 83.0, 229.0, 47.0, 109.0],
        ),
        (
            ["住宅电话", "编号", "居住地址", "信息更新日期", "居住状况"],
            3,
            [103.0, 53.0, 241.0, 127.0, 79.0],
        ),
    ],
)
def test_identity_owned_residence_accepts_reordered_variable_population(
    header: list[str],
    population: int,
    widths: list[float],
) -> None:
    rows = [header]
    expected_addresses: list[str] = []
    for sequence in range(1, population + 1):
        address = f"某市某区泛化路{sequence}号"
        expected_addresses.append(address)
        values = {
            "编号": str(sequence),
            "居住地址": address,
            "住宅电话": f"010-{sequence:08d}",
            "居住状况": "自置" if sequence % 2 else "租房",
            "信息更新日期": f"2025.02.{sequence:02d}",
        }
        rows.append([values[label] for label in header])
    rows.append(["编号", "数据发生机构名称", "", "", ""])
    rows.extend(
        [[str(sequence), f"样例机构{sequence}", "", "", ""] for sequence in range(1, population + 1)]
    )
    metadata = _exact_grid_metadata(rows, widths=widths)
    metadata["canonical_template_id"] = "report_header_and_identity"
    table = SimpleNamespace(
        table_id=f"owned-reordered-residence-{population}",
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=[20, 20, 620, 20 + len(rows) * 10],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=13,
                source_page_number=29,
                canonical_template_id="report_header_and_identity",
                tables=[table],
                texts=[],
            )
        ],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_raw_residence_records(result)

    assert [record["sequence"] for record in records] == list(
        range(1, population + 1)
    )
    assert [record["address"] for record in records] == expected_addresses
    assert [record["data_provider"] for record in records] == [
        f"样例机构{sequence}" for sequence in range(1, population + 1)
    ]


@pytest.mark.parametrize(
    "phone",
    (
        "$01012345678",
        "010-12345678?",
        "号01012345678",
        "010--12345678",
        "010-12-345-678",
        "12345",
    ),
)
def test_residence_phone_with_unowned_glyph_residue_is_withheld(phone: str) -> None:
    rows = [
        ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
        ["1", "某市某区泛化路1号", phone, "自置", "2025.02.01"],
    ]
    metadata = _exact_grid_metadata(rows, widths=[53.0, 241.0, 109.0, 79.0, 127.0])
    metadata["canonical_template_id"] = "report_header_and_identity"
    table = SimpleNamespace(
        table_id="owned-residence-invalid-phone",
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=[20, 20, 629, 40],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=13,
                source_page_number=29,
                canonical_template_id="report_header_and_identity",
                tables=[table],
                texts=[],
            )
        ],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_raw_residence_records(result)

    assert len(records) == 1
    assert "residential_phone" not in records[0]
    assert records[0]["canonical_raw"]["residential_phone"] == [phone]
    assert (
        records[0]["source_refs_by_field"]["residential_phone"][0]["geometry_scope"]
        == "cell"
    )
    assert any(
        issue.get("target_record_id") == records[0]["residence_record_id"]
        and issue.get("field_name") == "residential_phone"
        and issue.get("observed_value") == [phone]
        and issue.get("source_refs", [{}])[0].get("geometry_scope") == "cell"
        for issue in result._personal_detail_extraction_issues
    )


def test_residence_phone_accepts_only_registered_presentation_separators() -> None:
    rows = [
        ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
        ["1", "某市某区泛化路1号", "(010) 12345678", "自置", "2025.02.01"],
    ]
    metadata = _exact_grid_metadata(rows, widths=[53.0, 241.0, 109.0, 79.0, 127.0])
    metadata["canonical_template_id"] = "report_header_and_identity"
    table = SimpleNamespace(
        table_id="owned-residence-presented-phone",
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=[20, 20, 629, 40],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=13,
                source_page_number=29,
                canonical_template_id="report_header_and_identity",
                tables=[table],
                texts=[],
            )
        ],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_raw_residence_records(result)

    assert records[0]["residential_phone"] == "01012345678"
    assert records[0]["canonical_raw"]["residential_phone"] == "(010) 12345678"


def test_employment_fragments_join_by_header_columns_and_printed_sequence() -> None:
    basic = SimpleNamespace(
        table_id="employment-basic",
        metadata={
            "raw_rows": [
                ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
                ["2", "样例科技有限公司", "私营企业", "样例路2号", "010-12345678"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 180],
    )
    detail = SimpleNamespace(
        table_id="employment-detail",
        metadata={
            "raw_rows": [
                ["编号", "职业", "行业", "职务", "职称", "进入本单位年份", "信息更新日期"],
                [
                    "2",
                    "专业技术人员",
                    "信息传输、计算机服务和软件业",
                    "一般员工",
                    "中级",
                    "2020",
                    "2025.01.02",
                ],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 180],
    )
    provider = SimpleNamespace(
        table_id="employment-provider",
        metadata={"raw_rows": [["2", "样例银行股份有限公司"]]},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 80],
    )
    continuations = {
        ("employment-basic", "employment-detail"),
        ("employment-detail", "employment-provider"),
    }
    result = SimpleNamespace(
        pages=[
            employment_page(basic, logical_page=1),
            employment_page(detail, logical_page=2),
            employment_page(provider, logical_page=3),
        ],
        tables_continue=lambda left, right: (left, right) in continuations,
    )

    records = _extract_employment_records(result)

    assert len(records) == 1
    assert records[0]["sequence"] == 2
    assert records[0]["employer"] == "样例科技有限公司"
    assert records[0]["occupation"] == "专业技术人员"
    assert records[0]["entry_year"] == 2020
    assert records[0]["data_provider"] == "样例银行股份有限公司"
    assert not any(
        issue["issue_code"] == "candidate_b_employment_component_missing"
        for issue in getattr(result, "_personal_detail_extraction_issues", [])
    )


def test_employment_basic_only_is_retained_but_explicitly_marked_incomplete() -> None:
    basic = SimpleNamespace(
        table_id="employment-basic",
        metadata={
            "raw_rows": [
                ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
                ["1", "样例科技有限公司", "私营企业", "样例路1号", "010-12345678"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 180],
    )
    result = SimpleNamespace(
        pages=[employment_page(basic, logical_page=1)],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_employment_records(result)

    assert len(records) == 1
    assert records[0]["employer"] == "样例科技有限公司"
    assert records[0]["extraction_status"] == "review"
    assert any(
        issue["issue_code"] == "candidate_b_employment_component_missing"
        and issue["target_record_id"] == records[0]["employment_record_id"]
        and issue["candidate_value"]["missing_components"] == ["detail", "provider"]
        for issue in result._personal_detail_extraction_issues
    )


def test_residence_provider_table_cannot_activate_employment_extraction() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "某市某区某路1号", "--", "租房", "2025.01.02"],
            ]
        },
        headers=[],
        rows=[],
    )
    provider = SimpleNamespace(
        table_id="residence-provider",
        metadata={
            "raw_rows": [
                ["编号", "数据发生机构名称"],
                ["1", "样例银行股份有限公司"],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[residence]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[provider]),
        ],
        tables_continue=lambda left, right: (left, right)
        == ("residence", "residence-provider"),
    )

    assert _extract_employment_records(result) == []


def test_residence_text_shaped_date_and_unkeyed_provider_suffix_fail_closed() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "2025.01.02 某市某区某路1号", "--", "租房", ""],
                ["2", "2024.12.03 某市某区某路2号", "--", "自置", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    provider = SimpleNamespace(
        table_id="residence-provider",
        metadata={
            "raw_rows": [
                ["X", "样例银行股份有限公司2025.01.02"],
                ["Y", "样例消费金融有限公司"],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[residence]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[provider]),
        ],
        tables_continue=lambda left, right: (left, right)
        == ("residence", "residence-provider"),
    )

    records = _extract_residence_records(result)

    assert [record["address"] for record in records] == [
        "2025.01.02 某市某区某路1号",
        "2024.12.03 某市某区某路2号",
    ]
    assert all("information_updated_date" not in record for record in records)
    assert all("data_provider" not in record for record in records)
    issues = getattr(result, "_personal_detail_extraction_issues", [])
    assert {
        issue["target_record_id"]
        for issue in issues
        if issue["issue_code"] == "candidate_b_residence_provider_missing"
    } == {record["residence_record_id"] for record in records}
    assert {
        issue["target_record_id"]
        for issue in issues
        if issue.get("field_name") == "information_updated_date"
    } >= {record["residence_record_id"] for record in records}


def test_residence_embedded_date_inside_address_is_preserved_and_reported() -> None:
    address = "朝阳区2025.01.02公园路1号"
    residence = SimpleNamespace(
        table_id="residence-embedded-address-date",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["41", address, "--", "租房", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=8, source_page_number=3, tables=[residence])],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_residence_records(result)

    assert records[0]["address"] == address
    assert "information_updated_date" not in records[0]
    assert any(
        issue.get("field_name") == "information_updated_date"
        and issue["target_record_id"] == records[0]["residence_record_id"]
        for issue in result._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    ("header", "value", "address_column"),
    [
        (
            ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
            ["17", "朝阳区公园路17号 2025.01.02", "--", "租房", ""],
            1,
        ),
        (
            ["信息更新日期", "居住状况", "居住地址", "编号", "住宅电话"],
            ["", "租房", "2025.01.02 朝阳区公园路17号", "17", "--"],
            2,
        ),
    ],
)
def test_residence_exact_address_date_tokens_support_reordering_and_variable_widths(
    header: list[str],
    value: list[str],
    address_column: int,
) -> None:
    rows = [header, value]
    metadata = _exact_grid_metadata(
        rows,
        widths=[62.0, 173.0, 91.0, 138.0, 117.0],
    )
    geometry = metadata["geometry"]
    assert isinstance(geometry, dict)
    address_tokens = ["residence-address-token", "residence-date-token"]
    geometry["cell_evidence_ids"][1][address_column] = address_tokens
    geometry["cell_token_ids"][1][address_column] = address_tokens
    address_box = geometry["cell_bboxes"][1][address_column]
    assert isinstance(address_box, list)
    split = address_box[0] + (address_box[2] - address_box[0]) * 0.62
    date_first = value[address_column].lstrip().startswith("2025.01.02")
    left_text, left_id = (
        ("2025.01.02", "residence-date-token")
        if date_first
        else ("朝阳区公园路17号", "residence-address-token")
    )
    right_text, right_id = (
        ("朝阳区公园路17号", "residence-address-token")
        if date_first
        else ("2025.01.02", "residence-date-token")
    )
    atoms = [
        {
            "id": left_id,
            "text": left_text,
            "bbox": [address_box[0] + 2.0, address_box[1] + 1.0, split - 2.0, address_box[3] - 1.0],
        },
        {
            "id": right_id,
            "text": right_text,
            "bbox": [split + 2.0, address_box[1] + 1.0, address_box[2] - 2.0, address_box[3] - 1.0],
        },
    ]
    table = SimpleNamespace(
        table_id="residence-exact-address-date",
        metadata=metadata,
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=4, source_page_number=9, tables=[table])],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
    )

    records = _extract_residence_records(result)

    assert records[0]["address"] == "朝阳区公园路17号"
    assert records[0]["information_updated_date"] == "2025-01-02"


def test_residence_unowned_exact_tokens_do_not_authorize_address_date_split() -> None:
    address = "朝阳区公园路17号 2025.01.02"
    rows = [
        ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
        ["17", address, "--", "租房", ""],
    ]
    metadata = _exact_grid_metadata(rows, widths=[62.0, 173.0, 91.0, 138.0, 117.0])
    geometry = metadata["geometry"]
    assert isinstance(geometry, dict)
    geometry["cell_evidence_ids"][1][1] = ["address-token", "date-token"]
    geometry["cell_token_ids"][1][1] = ["address-token", "date-token"]
    # A plausible header string is not a registered semantic owner when one
    # label cell has no exact evidence.
    geometry["cell_evidence_ids"][0][4] = []
    address_box = geometry["cell_bboxes"][1][1]
    assert isinstance(address_box, list)
    split = (address_box[0] + address_box[2]) / 2.0
    atoms = [
        {
            "id": "address-token",
            "text": "朝阳区公园路17号",
            "bbox": [address_box[0] + 2.0, address_box[1] + 1.0, split - 2.0, address_box[3] - 1.0],
        },
        {
            "id": "date-token",
            "text": "2025.01.02",
            "bbox": [split + 2.0, address_box[1] + 1.0, address_box[2] - 2.0, address_box[3] - 1.0],
        },
    ]
    table = SimpleNamespace(table_id="unowned-address-date", metadata=metadata, headers=[], rows=[])
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=4, source_page_number=9, tables=[table])],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
    )

    records = _extract_residence_records(result)

    assert records[0]["address"] == address
    assert "information_updated_date" not in records[0]


def test_residence_provider_uses_reordered_semantic_columns_and_exact_record_key() -> None:
    residence_rows = [
        ["居住状况", "编号", "信息更新日期", "居住地址", "住宅电话"],
        ["租房", "73", "2025.01.02", "朝阳区公园路73号", "--"],
    ]
    provider_rows = [
        ["数据发生机构名称", "备注", "编号"],
        ["某征信服务有限公司2025.01.02", "--", "73"],
    ]
    residence = SimpleNamespace(
        table_id="residence-reordered",
        metadata=_exact_grid_metadata(
            residence_rows,
            widths=[84.0, 53.0, 127.0, 219.0, 98.0],
        ),
        headers=[],
        rows=[],
    )
    provider = SimpleNamespace(
        table_id="residence-provider-reordered",
        metadata=_exact_grid_metadata(provider_rows, widths=[238.0, 71.0, 49.0]),
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=2, source_page_number=7, tables=[residence]),
            SimpleNamespace(page_number=3, source_page_number=8, tables=[provider]),
        ],
        tables_continue=lambda left, right: (left, right)
        == ("residence-reordered", "residence-provider-reordered"),
    )

    records = _extract_residence_records(result)

    assert records[0]["sequence"] == 73
    assert records[0]["data_provider"] == "某征信服务有限公司2025.01.02"


def test_two_cell_residence_continuation_is_not_misclassified_as_provider() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "某市某区某路1号", "--", "租房", "2025.01.02"],
            ]
        },
        headers=[],
        rows=[],
    )
    continuation = SimpleNamespace(
        table_id="residence-continuation",
        metadata={"raw_rows": [["2", "某市某区某路2号", "", "", ""]]},
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[residence]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[continuation]),
        ],
        tables_continue=lambda left, right: (left, right)
        == ("residence", "residence-continuation"),
    )

    records = _extract_residence_records(result)

    assert [(record["sequence"], record.get("address")) for record in records] == [
        (1, "某市某区某路1号"),
        (2, "某市某区某路2号"),
    ]


def test_two_cell_employment_continuation_is_not_misclassified_as_provider() -> None:
    basic = SimpleNamespace(
        table_id="employment-basic",
        metadata={
            "raw_rows": [
                ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
                ["1", "样例科技有限公司一", "私营企业", "样例路1号", "010-12345678"],
            ]
        },
        headers=[],
        rows=[],
    )
    continuation = SimpleNamespace(
        table_id="employment-continuation",
        metadata={"raw_rows": [["2", "样例科技有限公司二", "", "", ""]]},
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            employment_page(basic, logical_page=1),
            employment_page(continuation, logical_page=2),
        ],
        tables_continue=lambda left, right: (left, right)
        == ("employment-basic", "employment-continuation"),
    )

    records = _extract_employment_records(result)

    assert [(record["sequence"], record.get("employer")) for record in records] == [
        (1, "样例科技有限公司一"),
        (2, "样例科技有限公司二"),
    ]
    assert all("data_provider" not in record for record in records)


def test_employment_state_terminates_before_continued_residence_provider_table() -> None:
    employment = SimpleNamespace(
        table_id="employment",
        metadata={
            "raw_rows": [
                ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
                ["1", "样例科技有限公司", "私营企业", "样例路1号", "010-12345678"],
            ]
        },
        headers=[],
        rows=[],
    )
    residence_provider = SimpleNamespace(
        table_id="residence-provider",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["编号", "数据发生机构名称", "", "", ""],
                ["1", "样例银行股份有限公司", "", "", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            employment_page(employment, logical_page=1),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[residence_provider]),
        ],
        tables_continue=lambda left, right: (left, right)
        == ("employment", "residence-provider"),
    )

    records = _extract_employment_records(result)

    assert len(records) == 1
    assert records[0]["employer"] == "样例科技有限公司"
    assert "data_provider" not in records[0]


@pytest.mark.parametrize("provider", ["样例银行", "样例合作机构"])
def test_same_table_unkeyed_residence_provider_is_reported(provider: str) -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "某市某区某路1号", "--", "租房", "2025.01.02"],
                ["编号", "数据发生机构名称", "", "", ""],
                ["P", provider, "", "", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[residence])],
        tables_continue=lambda _left, _right: False,
    )

    _extract_residence_records(result)

    assert any(
        issue["issue_code"] == "candidate_b_continuation_sequence_unresolved"
        and provider in issue["observed_value"]["physical_cells"]
        for issue in result._personal_detail_extraction_issues
    )


@pytest.mark.parametrize("ordinal", ["X", "Y", "第1项", "99"])
def test_residence_provider_requires_exact_known_record_ordinal(ordinal: str) -> None:
    provider = "某征信服务有限公司2025.01.02"
    residence = SimpleNamespace(
        table_id="residence-provider-key-negative",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "朝阳区公园路1号", "--", "租房", "2025.01.02"],
                ["数据发生机构名称", "编号", "", "", ""],
                [provider, ordinal, "", "", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=6, source_page_number=2, tables=[residence])],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_residence_records(result)

    assert "data_provider" not in records[0]
    assert any(
        issue["issue_code"] == "candidate_b_continuation_sequence_unresolved"
        and provider in issue["observed_value"]["physical_cells"]
        for issue in result._personal_detail_extraction_issues
    )


def test_residence_provider_duplicate_semantic_owner_fails_closed() -> None:
    residence = SimpleNamespace(
        table_id="residence-provider-duplicate-owner",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "朝阳区公园路1号", "--", "租房", "2025.01.02"],
                ["编号", "数据发生机构名称", "数据发生机构名称", "", ""],
                ["1", "机构甲", "机构乙", "", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=6, source_page_number=2, tables=[residence])],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_residence_records(result)

    assert "data_provider" not in records[0]
    assert any(
        issue["issue_code"] == "candidate_b_canonical_header_graph_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


def test_occupied_invalid_residence_date_slot_is_not_silently_repaired_from_address() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "2025.01.02 某市某区某路1号", "--", "租房", "无法识别"],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[residence])],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_residence_records(result)

    assert len(records) == 1
    assert any(
        issue.get("field_name") == "information_updated_date"
        and issue["target_record_id"] == records[0]["residence_record_id"]
        for issue in result._personal_detail_extraction_issues
    )


def test_explicit_absent_residence_provider_is_not_reported_as_missing() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "某市某区某路1号", "--", "租房", "2025.01.02"],
            ]
        },
        headers=[],
        rows=[],
    )
    provider = SimpleNamespace(
        table_id="residence-provider",
        metadata={"raw_rows": [["编号", "数据发生机构名称"], ["1", "--"]]},
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[residence]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[provider]),
        ],
        tables_continue=lambda left, right: (left, right)
        == ("residence", "residence-provider"),
    )

    records = _extract_residence_records(result)

    assert len(records) == 1
    assert "data_provider" not in records[0]
    assert not any(
        issue["issue_code"] == "candidate_b_residence_provider_missing"
        for issue in getattr(result, "_personal_detail_extraction_issues", [])
    )


def test_scanned_residence_unknown_continuation_does_not_use_structural_fallback() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "某市某区某路1号", "", "租房", "2025.01.02"],
            ]
        },
        headers=[],
        rows=[],
    )
    plausible_tail = SimpleNamespace(
        table_id="plausible-tail",
        metadata={"raw_rows": [["2", "2024.12.03 某市某区某路2号"]]},
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[residence]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[plausible_tail]),
        ],
        tables_continue=lambda _left, _right: None,
    )

    records = _extract_scanned_residence_records(result)

    assert [record["sequence"] for record in records] == [1]


def test_account_heading_uses_nearest_account_anchor_even_without_agreement() -> None:
    page = SimpleNamespace(
        height=800,
        texts=[
            SimpleNamespace(
                content="账户1（授信协议标识：AGREEMENT1）",
                bbox=[20, 20, 300, 40],
            ),
            SimpleNamespace(content="账户2", bbox=[20, 300, 100, 320]),
        ],
    )
    table = SimpleNamespace(bbox=[20, 325, 580, 600])

    assert _account_heading_for_table(page, table) == {}


def test_printed_page_footers_restore_plugin_continuation_order() -> None:
    def bundle(logical: int, printed: int, *texts: str) -> dict[str, object]:
        lines = [
            {"text": text, "bbox": [20, 40 + index * 30, 580, 65 + index * 30]} for index, text in enumerate(texts)
        ]
        lines.append(
            {
                "text": f"第 {printed} 页，共 4 页",
                "bbox": [220, 760, 380, 785],
            }
        )
        return {
            "page": logical,
            "source_page_number": (logical + 1) // 2,
            "local_structure_evidence": {
                "page": logical,
                "page_width": 600,
                "page_height": 800,
                "lines": lines,
            },
        }

    # The sealed logical order is 1, 4, 2, 3. The plugin must use printed
    # order 1, 2, 3, 4 without rewriting logical/source provenance.
    bundles = [
        bundle(1, 1, "（四）贷记卡账户", "账户 1（授信协议标识：A1）"),
        bundle(2, 4, "（五）授信协议信息", "授信协议 1"),
        bundle(3, 2, "账户 2（授信协议标识：A2）"),
        bundle(4, 3, "账户 3（授信协议标识：A3）"),
    ]
    result = SimpleNamespace(
        pages=[],
        entities=SimpleNamespace(domain_specific={"_page_evidence_bundles": bundles}),
    )

    context = build_personal_detail_extraction_context(result)
    accounts = extract_scanned_credit_accounts(context)

    assert dict(context.reading_order_by_logical) == {1: 1, 3: 2, 4: 3, 2: 4}
    # Unsealed bundle text may establish an ordering hint, but it cannot own
    # canonical account pages or records.  Exact evidence IDs and bounded
    # geometry are required before the projection adopts them.
    assert context.corrected_evidence_pages() == []
    assert accounts == []


def test_explicit_empty_canonical_account_plane_does_not_reopen_sealed_bundles() -> None:
    sealed_bundle = {
        "page": 1,
        "source_page_number": 1,
        "local_structure_evidence": {
            "lines": [
                {
                    "text": "（四）贷记卡账户",
                    "bbox": [20, 20, 580, 45],
                    "evidence_ids": ["sealed-family"],
                },
                {
                    "text": "账户 1（授信协议标识：GENERIC1）",
                    "bbox": [20, 50, 580, 75],
                    "evidence_ids": ["sealed-anchor"],
                },
            ]
        },
    }
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [],
        entities=SimpleNamespace(
            domain_specific={"_page_evidence_bundles": [sealed_bundle]}
        ),
    )

    assert extract_scanned_credit_accounts(context) == []


def test_native_profile_tables_preserve_empty_cells_and_embedded_subtables() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        bbox=[20, 20, 580, 300],
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "某市某区一号", "", "按揭", "2023.11.07"],
                ["2.", "某市某区二号", "13800138000", "其他", "2023.08.15"],
                ["编号", "数据发生机构名称", "", "", ""],
                ["1", "示例银行一", "", "", ""],
                ["敬", "示例银行二", "", "", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    employment = SimpleNamespace(
        table_id="employment",
        bbox=[20, 320, 580, 700],
        metadata={
            "raw_rows": [
                ["编号", "单位地址 工作单位 单位性质 单位电话", "", ""],
                ["1", "示例粮油有限公司 国有企业 某市某路60号 059100000000", "", ""],
                ["编号", "行业 职业", "职务", "职称 进入本单位年份 信息更新日期"],
                ["1", "商业、服务业人员 批发和零售业", "一般员工", "-- 2022.05.31"],
                ["编号", "数据发生机构名称", "", ""],
                ["P", "示例银行", "", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    employment_page(employment, logical_page=2, source_page=1)
    profile = SimpleNamespace(
        table_id="profile",
        bbox=[20, 20, 580, 300],
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["编号 手机号码", "", "信息更新日期", "数据发生机构名称"],
                ["13799911561", "", "2023.11.07", "示例银行"],
            ]
        },
        headers=[],
        rows=[],
    )
    spouse = SimpleNamespace(
        table_id="spouse",
        bbox=[20, 320, 580, 500],
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["姓名", "证件类型", "证件号码", "工作单位", "联系电话"],
                ["林航", "", "--", "", "13763822211"],
                ["数据发生机构名称", "", "", "", ""],
                ["示例消费金融有限公司", "", "", "", ""],
            ]
        },
        headers=[],
        rows=[],
    )
    employment_owner_page = employment_page(
        employment,
        logical_page=2,
        source_page=1,
    )
    employment_owner_page.tables = [residence, employment]
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="report_header_and_identity",
                tables=[profile, spouse],
            ),
            employment_owner_page,
        ],
        tables_continue=lambda _left, _right: False,
    )

    residences = _extract_residence_records(result)
    employments = _extract_employment_records(result)
    details = _extract_profile_detail_records(result)

    assert [(row["address"], row.get("residential_phone"), row.get("data_provider")) for row in residences] == [
        ("某市某区一号", None, "示例银行一"),
        ("某市某区二号", "13800138000", None),
    ]
    assert len(employments) == 1
    assert {
        field: employments[0][field]
        for field in ("employer", "employer_type", "employer_address", "employer_phone")
    } == {
        "employer": "示例粮油有限公司",
        "employer_type": "国有企业",
        "employer_address": "某市某路60号",
        "employer_phone": "059100000000",
    }
    # The basic component is safely retained. A geometry-free adjacent provider
    # row cannot prove the spouse provider role and remains withheld.
    assert employments[0]["extraction_status"] == "review"
    assert details["mobile_phone_records"] == []
    assert details["spouse_records"][0]["name"] == "林航"
    assert details["spouse_records"][0]["phone"] == "13763822211"
    assert "data_provider" not in details["spouse_records"][0]
    issue_codes = {row["issue_code"] for row in result._personal_detail_extraction_issues}
    assert "candidate_b_canonical_header_graph_unresolved" in issue_codes
    assert "candidate_b_continuation_sequence_unresolved" in issue_codes


def test_mobile_and_spouse_partial_headers_ignore_other_canonical_sections() -> None:
    rows = (
        ["被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因"],
        ["编号", "居住地址 G", "住宅电话", "居住状况", "信息更新日期"],
        ["编号", "? 职业 行业", "", "职务 职称 进入本单位年份", "信息更新日期", ""],
        [
            "主业务借款人",
            "",
            "",
            "主业务借款人证件类型",
            "",
            "",
            "",
            "主业务借款人证件号码",
            "",
            "",
        ],
    )
    tables = [
        SimpleNamespace(
            table_id=f"other-canonical-section:{index}",
            bbox=[20, 20, 580, 100],
            metadata={"raw_rows": [row]},
            headers=[],
            rows=[],
        )
        for index, row in enumerate(rows, start=1)
    ]
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="report_header_and_identity",
                tables=tables,
            )
        ],
        tables_continue=lambda _left, _right: False,
        _personal_detail_extraction_issues=[],
    )

    details = _extract_profile_detail_records(result)

    assert details == {"mobile_phone_records": [], "spouse_records": []}
    assert not any(
        issue["issue_code"] == "candidate_b_canonical_header_graph_unresolved"
        and issue["target_dataset"] in {"mobile_phone_records", "spouse_records"}
        for issue in result._personal_detail_extraction_issues
    )


def test_mobile_and_spouse_exclusive_merged_headers_remain_reported() -> None:
    tables = [
        SimpleNamespace(
            table_id="mobile-merged-header",
            bbox=[20, 20, 580, 100],
            metadata={
                "canonical_template_id": "report_header_and_identity",
                "raw_rows": [
                    ["编号 手机号码", "", "信息更新日期", "数据发生机构名称"]
                ]
            },
            headers=[],
            rows=[],
        ),
        SimpleNamespace(
            table_id="spouse-merged-header",
            bbox=[20, 120, 580, 200],
            metadata={
                "canonical_template_id": "report_header_and_identity",
                "raw_rows": [["姓名 证件类型", "证件号码", "工作单位", "联系电话"]]
            },
            headers=[],
            rows=[],
        ),
    ]
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="report_header_and_identity",
                tables=tables,
            )
        ],
        tables_continue=lambda _left, _right: False,
        _personal_detail_extraction_issues=[],
    )

    _extract_profile_detail_records(result)

    assert {
        issue["target_dataset"]
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_canonical_header_graph_unresolved"
    } == {"mobile_phone_records", "spouse_records"}


def test_account_fact_graph_never_shifts_values_across_an_empty_cell() -> None:
    table = SimpleNamespace(
        table_id="account",
        metadata={
            "raw_rows": [
                ["管理机构", "账户标识", "开立日期", "借款金额", "账户币种"],
                ["样例银行", "", "2024.01.02", "5000", "人民币元"],
            ],
            "cell_bboxes": [
                [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10], [30, 0, 40, 10], [40, 0, 50, 10]],
                [[0, 10, 10, 20], [10, 10, 20, 20], [20, 10, 30, 20], [30, 10, 40, 20], [40, 10, 50, 20]],
            ],
        },
        headers=[],
        rows=[],
        bbox=[0, 0, 50, 20],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table], texts=[])],
        corrected_evidence_pages=lambda: [],
    )

    table_accounts, _repayments, _events = _extract_table_accounts(result)
    accounts, _repayments, _events = _extract_accounts(result)

    assert len(table_accounts) == 1
    assert table_accounts[0]["management_institution"] == "样例银行"
    assert table_accounts[0]["open_date"] == "2024-01-02"
    assert table_accounts[0]["loan_amount"] == 5000
    assert "account_identifier" not in table_accounts[0]
    assert (
        table_accounts[0]["source_refs_by_field"]["loan_amount"][0]["binding"]
        == "canonical_field_slot"
    )
    assert (
        table_accounts[0]["source_refs_by_field"]["loan_amount"][0]["geometry_scope"]
        == "cell"
    )
    # Exact cells are decoded without shifting, but the table observation is
    # not promoted to a population record without a printed account anchor.
    assert accounts == []
    assert any(
        issue["field_name"] == "account_identifier"
        and issue["issue_code"] == "candidate_b_exact_slot_value_unreadable"
        for issue in result._personal_detail_extraction_issues
    )
    assert any(
        issue["issue_code"] == "candidate_b_account_anchor_population_missing"
        and issue["target_record_id"]
        == table_accounts[0]["_table_observation_instance_id"]
        for issue in result._personal_detail_extraction_issues
    )


def test_account_fact_conflict_is_withheld_instead_of_last_write_wins() -> None:
    table = SimpleNamespace(
        table_id="account",
        metadata={
            "raw_rows": [
                ["管理机构", "账户标识", "开立日期"],
                ["样例银行甲", "A12345678", "2024.01.02"],
                ["管理机构", "账户标识", "开立日期"],
                ["样例银行乙", "A12345678", "2024.01.02"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[0, 0, 50, 20],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table], texts=[])],
        corrected_evidence_pages=lambda: [],
    )

    table_accounts, _repayments, _events = _extract_table_accounts(result)
    accounts, _repayments, _events = _extract_accounts(result)

    assert len(table_accounts) == 1
    assert "management_institution" not in table_accounts[0]
    assert table_accounts[0]["canonical_raw"]["management_institution"] == [
        "样例银行甲",
        "样例银行乙",
    ]
    assert accounts == []
    assert any(
        issue["field_name"] == "management_institution"
        and issue["issue_code"] == "candidate_b_exact_slot_value_conflict"
        for issue in result._personal_detail_extraction_issues
    )
    assert any(
        issue["issue_code"] == "candidate_b_account_anchor_population_missing"
        and issue["target_record_id"]
        == table_accounts[0]["_table_observation_instance_id"]
        for issue in result._personal_detail_extraction_issues
    )


def test_residence_same_sequence_conflict_is_one_partial_record_with_issue() -> None:
    table = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["1", "地址甲一号", "--", "租房", "2025.01.02"],
                ["1", "地址乙二号", "--", "租房", "2025.01.02"],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_residence_records(result)

    assert len(records) == 1
    assert "address" not in records[0]
    assert records[0]["residence_status"] == "租房"
    assert any(
        issue["field_name"] == "address"
        and issue["target_record_id"] == records[0]["residence_record_id"]
        for issue in result._personal_detail_extraction_issues
    )


def test_residence_combined_status_date_header_recovers_exact_rows_and_providers() -> None:
    rows = [
        ["编号", "居住地址", "住宅电话", "信息更新日期 居住状况"],
        ["1", "福建省漳州市龙文区一号", "--", "自有 2025.07.11"],
        ["2", "福建省漳州市龙文区二号", "--", "2025.03.24"],
        ["编号", "数据发生机构名称", "", ""],
        ["1", "中国建设银行股份有限公司漳州龙文支行", "", ""],
        ["2", "中国工商银行股份有限公司福建省分行", "", ""],
    ]
    table = SimpleNamespace(
        table_id="residence-combined-status-date",
        metadata=_exact_grid_metadata(rows),
        headers=[],
        rows=[],
        bbox=[20.0, 20.0, 420.0, 80.0],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=2, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
    )

    records = _extract_residence_records(result)

    assert [record["sequence"] for record in records] == [1, 2]
    assert records[0]["residence_status"] == "自置"
    assert records[0]["information_updated_date"] == "2025-07-11"
    assert records[1]["information_updated_date"] == "2025-03-24"
    assert "residence_status" not in records[1]
    assert any(
        issue["target_record_id"] == records[1]["residence_record_id"]
        and issue["field_name"] == "residence_status"
        and issue["issue_code"] == "candidate_b_exact_slot_value_unreadable"
        for issue in result._personal_detail_extraction_issues
    )
    assert records[0]["data_provider"] == "中国建设银行股份有限公司漳州龙文支行"
    assert records[1]["data_provider"] == "中国工商银行股份有限公司福建省分行"


@pytest.mark.parametrize(
    "defect",
    ["wrong_header", "duplicate_header_evidence", "inexact_header", "extra_column"],
)
def test_residence_combined_status_date_header_fails_closed_without_exact_lattice(
    defect: str,
) -> None:
    rows = [
        ["编号", "居住地址", "住宅电话", "信息更新日期 居住状况"],
        ["1", "福建省漳州市龙文区一号", "--", "自有 2025.07.11"],
    ]
    if defect == "wrong_header":
        rows[0][3] = "信息更新日期 居住情况"
    elif defect == "extra_column":
        rows[0].append("噪声")
        rows[1].append("")
    metadata = _exact_grid_metadata(rows)
    geometry = metadata["geometry"]
    assert isinstance(geometry, dict)
    if defect == "duplicate_header_evidence":
        geometry["cell_evidence_ids"][0][3] = geometry["cell_evidence_ids"][0][2]
    elif defect == "inexact_header":
        geometry["cell_geometry_status"][0][3] = "derived"
    table = SimpleNamespace(
        table_id=f"residence-combined-{defect}",
        metadata=metadata,
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=2, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
    )

    assert _extract_residence_records(result) == []


def test_residence_three_role_combined_header_recovers_exact_token_components() -> None:
    rows = [
        ["编号", "居住地址", "住宅电话 居住状况 信息更新日期"],
        ["1", "福建省宁德市福鼎市一号", '2025.03.04 "'],
        ["2", "福建省宁德市福鼎市二号", "按揭 2025.01.18"],
        ["编号", "数据发生机构名称", ""],
        ["1", "样例银行一", ""],
        ["2", "样例银行二", ""],
    ]
    table = SimpleNamespace(
        table_id="residence-three-role-combined",
        metadata={
            **_exact_grid_metadata(rows),
            "source_logical_page": 2,
        },
        headers=[],
        rows=[],
    )
    geometry = table.metadata["geometry"]
    assert isinstance(geometry, dict)
    geometry["cell_evidence_ids"][1][2] = ["date-1", "noise-1"]
    geometry["cell_token_ids"][1][2] = ["date-1", "noise-1"]
    geometry["cell_evidence_ids"][2][2] = ["status-2", "date-2"]
    geometry["cell_token_ids"][2][2] = ["status-2", "date-2"]
    evidence_pages = [
        {
            "page": 2,
            "lines": [
                {"text": "2025.03.04", "bbox": [220, 30, 260, 39], "evidence_ids": ["date-1"]},
                {"text": '"', "bbox": [270, 30, 275, 39], "evidence_ids": ["noise-1"]},
                {"text": "按揭", "bbox": [220, 40, 240, 49], "evidence_ids": ["status-2"]},
                {"text": "2025.01.18", "bbox": [250, 40, 290, 49], "evidence_ids": ["date-2"]},
            ],
        }
    ]
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=2, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
        corrected_evidence_pages=lambda: evidence_pages,
    )

    records = _extract_residence_records(result)

    assert [record["sequence"] for record in records] == [1, 2]
    assert records[0]["information_updated_date"] == "2025-03-04"
    assert "residence_status" not in records[0]
    assert records[1]["residence_status"] == "按揭"
    assert records[1]["information_updated_date"] == "2025-01-18"
    assert all("residential_phone" not in record for record in records)
    issues = result._personal_detail_extraction_issues
    assert {
        (issue["target_record_id"], issue["field_name"])
        for issue in issues
        if issue["issue_code"] == "candidate_b_exact_slot_value_unreadable"
    } >= {
        (records[0]["residence_record_id"], "residential_phone"),
        (records[0]["residence_record_id"], "residence_status"),
        (records[1]["residence_record_id"], "residential_phone"),
    }


@pytest.mark.parametrize(
    "defect",
    ["wrong_header", "duplicate_header_evidence", "inexact_header", "extra_column"],
)
def test_residence_three_role_combined_header_fails_closed(defect: str) -> None:
    rows = [
        ["编号", "居住地址", "住宅电话 居住状况 信息更新日期"],
        ["1", "福建省宁德市福鼎市一号", "按揭 2025.01.18"],
    ]
    if defect == "wrong_header":
        rows[0][2] = "住宅电话 居住情况 信息更新日期"
    elif defect == "extra_column":
        rows[0].append("噪声")
        rows[1].append("")
    metadata = _exact_grid_metadata(rows)
    geometry = metadata["geometry"]
    assert isinstance(geometry, dict)
    if defect == "duplicate_header_evidence":
        geometry["cell_evidence_ids"][0][2] = geometry["cell_evidence_ids"][0][1]
    elif defect == "inexact_header":
        geometry["cell_geometry_status"][0][2] = "derived"
    table = SimpleNamespace(
        table_id=f"residence-three-role-{defect}",
        metadata=metadata,
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=2, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
        corrected_evidence_pages=lambda: [],
    )

    assert _extract_residence_records(result) == []


def test_mobile_and_spouse_conflicts_do_not_create_duplicate_business_records() -> None:
    mobile = SimpleNamespace(
        table_id="mobile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["编号", "手机号码", "信息更新日期", "数据发生机构名称"],
                ["1", "13800138000", "2025.01.02", "样例银行"],
                ["1", "13900139000", "2025.01.02", "样例银行"],
            ]
        },
        headers=[],
        rows=[],
    )
    spouse = SimpleNamespace(
        table_id="spouse",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["姓名", "证件类型", "证件号码", "工作单位", "联系电话"],
                ["张甲", "身份证", "110101199001010011", "单位甲", "13800138000"],
                ["张乙", "身份证", "110101199001010011", "单位甲", "13800138000"],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="report_header_and_identity",
                tables=[mobile, spouse],
            )
        ],
        tables_continue=lambda _left, _right: False,
    )

    details = _extract_profile_detail_records(result)

    assert len(details["mobile_phone_records"]) == 1
    assert "mobile_phone" not in details["mobile_phone_records"][0]
    assert len(details["spouse_records"]) == 1
    assert "name" not in details["spouse_records"][0]
    conflicts = {
        (issue["target_dataset"], issue["field_name"])
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_exact_slot_value_conflict"
    }
    assert ("mobile_phone_records", "mobile_phone") in conflicts
    assert ("spouse_records", "name") in conflicts


def test_mobile_row_without_canonical_owner_or_token_geometry_fails_closed() -> None:
    table = SimpleNamespace(
        table_id="mobile-collapsed",
        metadata={
            "raw_rows": [
                [
                    "手机号码 编号 15260467509 1",
                    "信息更新日期 2025.08.03",
                    "数据发生机构名称 深圳市乐信融资担保有限公司",
                    "",
                ]
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
        _personal_detail_extraction_issues=[],
    )

    details = _extract_profile_detail_records(result)

    assert details["mobile_phone_records"] == []
    assert result._personal_detail_extraction_issues == []


def _mobile_date_details(raw_date: str) -> tuple[dict[str, object], SimpleNamespace]:
    table = SimpleNamespace(
        table_id="mobile-date-contract",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["编号", "手机号码", "信息更新日期", "数据发生机构名称"],
                ["1", "13800138000", raw_date, "样例银行股份有限公司"],
            ]
        },
        headers=[],
        rows=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="report_header_and_identity",
                tables=[table],
            )
        ],
        tables_continue=lambda _left, _right: False,
    )
    record = _extract_profile_detail_records(result)["mobile_phone_records"][0]
    return record, result


@pytest.mark.parametrize(
    "raw",
    (
        "2022,09.15 A",
        "X2022.09.15Y",
        "???2022.09.15!!!",
        "2022.09.15?",
    ),
)
def test_mobile_date_withholds_one_valid_date_with_short_ascii_residue(raw: str) -> None:
    record, result = _mobile_date_details(raw)

    assert "information_updated_date" not in record
    assert record["canonical_raw"]["information_updated_date"] == [raw]
    assert any(
        item["issue_code"] == "candidate_b_exact_slot_value_invalid"
        and item["target_record_id"] == record["mobile_phone_record_id"]
        and item["field_name"] == "information_updated_date"
        and item["observed_value"] == [raw]
        and "normalized_value_withheld" in item.get("reason_codes", ())
        for item in result._personal_detail_extraction_issues
    )


def test_clean_mobile_date_stays_silent() -> None:
    record, result = _mobile_date_details("2022.09.15")

    assert record["information_updated_date"] == "2022-09-15"
    assert not hasattr(result, "_personal_detail_extraction_issues")


@pytest.mark.parametrize(
    "raw",
    (
        "2022.09.15 信息更新日期",
        "2022.09.15 2023.01.02",
        "DATE 2022.09.15",
    ),
)
def test_mobile_date_rejects_business_or_ambiguous_residue(raw: str) -> None:
    record, result = _mobile_date_details(raw)

    assert "information_updated_date" not in record
    assert record["canonical_raw"]["information_updated_date"] == [raw]
    assert any(
        item["issue_code"] == "candidate_b_exact_slot_value_invalid"
        and item["field_name"] == "information_updated_date"
        and item["observed_value"] == [raw]
        for item in result._personal_detail_extraction_issues
    )
    assert not any(
        item["issue_code"] == "candidate_b_date_ascii_residue_corrected"
        for item in result._personal_detail_extraction_issues
    )
