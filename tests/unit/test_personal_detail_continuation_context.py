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
    _extract_residence_records,
    _extract_summary_datasets,
)
from docmirror.plugins.credit_report.scanned_business import (
    _extract_residence_records as _extract_scanned_residence_records,
)
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

    # The closed-world registration layer excludes the unrelated unregistered
    # table before entity construction.  An unavailable relation is treated as
    # a split by the account extractor, never as permission to merge.
    assert context.tables_continue("account", "employment") is None
    assert len(accounts) == 1
    assert accounts[0]["balance"] == 100


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

    accounts, _repayments, _events = _extract_accounts(result)

    assert len(accounts) == 1
    assert accounts[0]["balance"] == 100


def test_summary_extraction_consumes_headerless_cross_page_fragment() -> None:
    head = SimpleNamespace(
        table_id="summary-head",
        metadata={
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
        metadata={"raw_rows": [["准贷记卡账户", "--", "--", "--", "--"]]},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 80],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[head], texts=[]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[tail], texts=[]),
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
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table], texts=[])],
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
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table], texts=[])],
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
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table], texts=[])],
        _personal_detail_extraction_issues=[],
    )

    _records, cells = _extract_summary_datasets(context)

    assert not any(cell["value"] in {"贷款", "信用卡"} for cell in cells)
    assert [cell["column_label"] for cell in cells if cell["column_index"] == 2] == [
        "业务类型",
        "业务类型",
    ]


def test_residence_provider_continuation_uses_entity_and_sequence_not_page_number() -> None:
    residence = SimpleNamespace(
        table_id="residence",
        metadata={
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
                ["7", "某市某区某路7号", "010-12345678", "租房", "2025.01.02"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 600, 580, 780],
    )
    provider = SimpleNamespace(
        table_id="provider",
        metadata={"raw_rows": [["7", "样例银行"]]},
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
                ["2", "工程技术人员", "信息技术业", "一般员工", "工程师", "2020", "2025.01.02"],
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
            SimpleNamespace(page_number=1, source_page_number=1, tables=[basic]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[detail]),
            SimpleNamespace(page_number=3, source_page_number=3, tables=[provider]),
        ],
        tables_continue=lambda left, right: (left, right) in continuations,
    )

    records = _extract_employment_records(result)

    assert len(records) == 1
    assert records[0]["sequence"] == 2
    assert records[0]["employer"] == "样例科技有限公司"
    assert records[0]["occupation"] == "工程技术人员"
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
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[basic])],
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


def test_residence_combined_date_and_unkeyed_provider_suffix_are_bound_conservatively() -> None:
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
                ["敬", "样例银行股份有限公司"],
                ["P", "样例消费金融有限公司"],
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
        "某市某区某路1号",
        "某市某区某路2号",
    ]
    assert [record["information_updated_date"] for record in records] == [
        "2025-01-02",
        "2024-12-03",
    ]
    assert [record["data_provider"] for record in records] == [
        "样例银行股份有限公司",
        "样例消费金融有限公司",
    ]
    assert not any(
        issue["issue_code"] == "candidate_b_residence_provider_missing"
        for issue in getattr(result, "_personal_detail_extraction_issues", [])
    )


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
            SimpleNamespace(page_number=1, source_page_number=1, tables=[basic]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[continuation]),
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
            SimpleNamespace(page_number=1, source_page_number=1, tables=[employment]),
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
    assert [page["page"] for page in context.corrected_evidence_pages()] == [1, 3, 4, 2]
    assert [account["account_id"] for account in accounts] == [
        "credit_account:credit_card:1",
        "credit_account:credit_card:2",
        "credit_account:credit_card:3",
    ]
    assert [account["source_refs"][0]["logical_page"] for account in accounts] == [1, 3, 4]


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
    profile = SimpleNamespace(
        table_id="profile",
        bbox=[20, 20, 580, 300],
        metadata={
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
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[profile, spouse]),
            SimpleNamespace(page_number=2, source_page_number=1, tables=[residence, employment]),
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
    assert employments == []
    assert details["mobile_phone_records"] == []
    assert details["spouse_records"][0]["name"] == "林航"
    assert details["spouse_records"][0]["phone"] == "13763822211"
    assert details["spouse_records"][0]["data_provider"] == "示例消费金融有限公司"
    issue_codes = {row["issue_code"] for row in result._personal_detail_extraction_issues}
    assert "candidate_b_canonical_header_graph_unresolved" in issue_codes
    assert "candidate_b_continuation_sequence_unresolved" in issue_codes


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

    accounts, _repayments, _events = _extract_accounts(result)

    assert len(accounts) == 1
    assert accounts[0]["management_institution"] == "样例银行"
    assert accounts[0]["open_date"] == "2024-01-02"
    assert accounts[0]["loan_amount"] == 5000
    assert "account_identifier" not in accounts[0]
    assert accounts[0]["source_refs_by_field"]["loan_amount"][0]["binding"] == "canonical_field_slot"
    assert accounts[0]["source_refs_by_field"]["loan_amount"][0]["geometry_scope"] == "cell"
    assert any(
        issue["field_name"] == "account_identifier"
        and issue["issue_code"] == "candidate_b_exact_slot_value_invalid"
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

    accounts, _repayments, _events = _extract_accounts(result)

    assert len(accounts) == 1
    assert "management_institution" not in accounts[0]
    assert accounts[0]["canonical_raw"]["management_institution"] == ["样例银行甲", "样例银行乙"]
    assert any(
        issue["field_name"] == "management_institution"
        and issue["issue_code"] == "candidate_b_exact_slot_value_conflict"
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


def test_mobile_and_spouse_conflicts_do_not_create_duplicate_business_records() -> None:
    mobile = SimpleNamespace(
        table_id="mobile",
        metadata={
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
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[mobile, spouse])],
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
