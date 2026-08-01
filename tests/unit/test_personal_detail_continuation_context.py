from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    PersonalDetailTransitionPolicy,
    build_personal_detail_extraction_context,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import _extract_accounts
from docmirror.plugins.credit_report.scanned_business import extract_scanned_credit_accounts
from docmirror.plugins.credit_report.shared.entity_decoder import CreditReportUnit


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
                texts=[SimpleNamespace(content="账户标识 A1", bbox=[30, 620, 200, 650])],
            ),
            SimpleNamespace(
                page_number=11,
                source_page_number=5,
                width=600,
                height=800,
                tables=[table_2],
                texts=[SimpleNamespace(content="账户状态 正常", bbox=[30, 40, 200, 70])],
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
    assert context.tables_continue("account-head", "account-tail") is True
    assert context.entity_context.entity_for_unit("personal_detail:table:p10:account-head").pages == (10, 11)
    assert context.allows_scanned_line_transition(10, evidence_line_1, 0, 11, evidence_line_2, 0) is True


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

    accounts, _repayments, _events = _extract_accounts(context)

    assert context.tables_continue("account", "employment") is False
    assert len(accounts) == 1
    assert accounts[0]["balance"] == 100


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
                    {"text": "非循环贷账户", "bbox": [20, 650, 580, 680]},
                    {"text": "账户 1 （示例贷款）", "bbox": [20, 690, 580, 720]},
                    {"text": "账户状态 正常 管理机构 示例银行", "bbox": [20, 740, 580, 780]},
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
                    {"text": "工作单位 示例公司 单位地址 示例地址", "bbox": [20, 20, 580, 60]},
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
