# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import CellValue, PageContent, TableBlock, TableRow, TextBlock
from docmirror.plugins.credit_report.business_assembly import _build_audit
from docmirror.plugins.credit_report.business_records import (
    _merge_enterprise_accounts,
    derive_overdue_records,
    extract_native_credit_business,
)
from docmirror.plugins.credit_report.currency_codes import (
    ISO_4217_CURRENT_CODES,
    normalize_currency_code,
)
from docmirror.plugins.credit_report.enterprise_native.extraction import (
    extract_enterprise_capital_summary,
    extract_enterprise_report_metadata,
    extract_enterprise_report_metadata_records,
    extract_enterprise_report_notes,
)
from docmirror.plugins.credit_report.personal_brief_native.extraction import (
    extract_personal_brief_section_content,
)


def _result(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text)])],
    )


def test_enterprise_capital_and_report_metadata_preserve_separate_business_meanings() -> None:
    capital_table = TableBlock(
        table_id="capital",
        metadata={
            "raw_rows": [
                ["类型", "出资方", "身份标识类型", "身份标识号码", "出资比例"],
                ["--", "--", "--", "--", "--"],
                ["信息来源机构：示例银行 更新日期：2022-10-08", "", "", "", ""],
            ]
        },
    )
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=4,
                texts=[TextBlock(content="注册资本折人民币合计 100万元")],
                tables=[capital_table],
            )
        ]
    )

    assert extract_enterprise_capital_summary(result) == [
        {
            "sequence": 1,
            "contributor_status": "no_records",
            "contributor_count": 0,
            "source_page": 4,
            "source": "canonical_enterprise_capital_table",
            "source_refs": [
                {
                    "source": "canonical_physical_table",
                    "page": 4,
                    "table_id": "capital",
                    "row": 0,
                }
            ],
            "confidence": 1.0,
            "source_institution": "示例银行",
            "update_date": "2022-10-08",
            "registered_capital_amount": 100,
            "currency": "CNY",
            "amount_unit": "CNY_10K",
        }
    ]
    assert extract_enterprise_report_metadata(
        result,
        "企业信用报告（自主查询版） 汇率（美元折人民币）：6.96 有效期：2023-01",
    ) == {
        "report_edition": "independent_query",
        "exchange_rate_usd_cny": 6.96,
        "exchange_rate_effective_period": "2023-01",
    }


def test_enterprise_report_cover_notes_and_exchange_keep_source_page_grain() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告\n（自主查询版）")],
            ),
            PageContent(
                page_number=2,
                texts=[
                    TextBlock(content="报告说明"),
                    TextBlock(
                        content=(
                            "1．第一条说明跨\n行续写。\n"
                            "2．第二条说明。\n"
                            "汇率（美元折人民币）：6.96　有效期：2023-01\n"
                            "第 2 页/共8 页"
                        )
                    ),
                ],
            ),
            PageContent(page_number=3, texts=[TextBlock(content="身份标识")]),
        ]
    )

    assert extract_enterprise_report_notes(result) == [
        {
            "note_id": extract_enterprise_report_notes(result)[0]["note_id"],
            "sequence": 1,
            "content": "第一条说明跨行续写。",
            "source_page": 2,
            "source": "enterprise_report_notes",
            "source_refs": [{"source": "native_text_report_note", "page": 2}],
            "confidence": 1.0,
        },
        {
            "note_id": extract_enterprise_report_notes(result)[1]["note_id"],
            "sequence": 2,
            "content": "第二条说明。",
            "source_page": 2,
            "source": "enterprise_report_notes",
            "source_refs": [{"source": "native_text_report_note", "page": 2}],
            "confidence": 1.0,
        },
    ]
    metadata = extract_enterprise_report_metadata_records(result)
    assert metadata["enterprise_report_metadata"][0]["source_page"] == 1
    assert metadata["enterprise_report_metadata"][0]["report_edition"] == "independent_query"
    assert metadata["enterprise_exchange_rates"][0]["source_page"] == 2
    assert metadata["enterprise_exchange_rates"][0]["exchange_rate_usd_cny"] == 6.96
    assert metadata["enterprise_exchange_rates"][0]["exchange_rate_effective_period"] == "2023-01"


def test_enterprise_balance_reconciliation_is_audit_only() -> None:
    accounts = [
        {
            "account_id": f"credit_account:{index}",
            "balance": balance,
            "currency": "CNY",
            "amount_unit": "CNY_10K",
            "source": "canonical_physical_table",
            "source_refs": [{"source": "canonical_physical_table", "page": 4 + index}],
        }
        for index, balance in enumerate((34.88, 4.67, 25.87), start=1)
    ]
    summary = {
        "reported_account_count": 3,
        "reported_account_balance": 65.41,
        "account_population_comparable": True,
    }
    audit = _build_audit(
        parse_result=SimpleNamespace(pages=[]),
        full_text="企业信用报告",
        report_subtype="enterprise",
        content_mode="native_text",
        collections={
            "credit_accounts": accounts,
            "credit_lines": [],
            "repayment_liability_records": [],
            "repayment_records": [],
            "overdue_records": [],
            "inquiry_records": [],
            "public_records": [],
        },
        conflicts=[],
        credit_summary=summary,
    )

    assert "detail_account_balance" not in summary
    assert "account_balance_difference" not in summary
    assert "account_balance_reconciliation_tolerance" not in summary
    assert "account_balance_reconciliation_status" not in summary
    reconciliation = next(
        item
        for item in audit["reconciliations"]
        if item["name"] == "credit_account_balance"
    )
    assert reconciliation == {
        "name": "credit_account_balance",
        "expected": 65.41,
        "actual": 65.42,
        "difference": 0.01,
        "tolerance": 0.02,
        "currency": "CNY",
        "amount_unit": "CNY_10K",
        "matched": True,
        "status": "within_rounding_tolerance",
    }


def test_personal_brief_extracts_narrative_accounts_and_inquiry_ledger() -> None:
    text = """
    个人信用报告 信贷记录
    2022年01月02日示例银行信用卡中心发放的贷记卡（人民币账户，卡片尾号：1234）。
    截至2024年11月，信用额度100,000，余额2,000，当前无逾期。
    最近5年内有1个月处于逾期状态，没有发生过90天以上逾期。
    2023年02月03日示例商业银行发放的300,000元（人民币）个人经营性贷款，
    2026年02月03日到期。截至2024年11月，余额200,000，从未发生过逾期。
    2024年03月04日示例商业银行为个人经营性贷款授信，额度有效期至2027年03月04日，
    可循环使用。截至2024年11月，信用额度500,000元（人民币），余额为120,000，当前无逾期。
    机构查询记录明细
    编号 查询日期 查询机构 查询原因
    1 2024年10月01日 示例商业银行 贷款审批
    2 2024年11月02日 示例商业银行 贷后管理
    个人查询记录明细
    1 2024年12月03日 本人查询（互联网个人信用信息服务平台）
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    assert len(business["credit_accounts"]) == 3
    assert len({item["account_id"] for item in business["credit_accounts"]}) == 3
    assert [item["sequence"] for item in business["credit_accounts"]] == [1, 1, 2]
    assert business["credit_accounts"][1]["loan_amount"] == 300000
    assert business["credit_lines"][0]["total_limit"] == 500000
    assert business["credit_lines"][0]["used_limit"] == 120000
    overdue = business["overdue_records"][0]
    assert overdue["sequence"] == 1
    assert overdue["management_institution"] == "示例银行信用卡中心"
    assert overdue["overdue_months"] == 1
    assert overdue["over_90_days_months"] == 0
    assert overdue["current_overdue_status"] == "not_overdue"
    assert len(business["inquiry_records"]) == 3
    personal_inquiry = business["inquiry_records"][-1]
    assert personal_inquiry["reason"] == "本人查询"
    assert personal_inquiry["source_reason"] == "本人查询（互联网个人信用信息服务平台）"
    assert personal_inquiry["query_channel"] == "互联网个人信用信息服务平台"
    assert business["credit_summary"]["institution_inquiry_count"] == 2
    assert business["credit_summary"]["personal_inquiry_count"] == 1
    assert business["credit_summary"]["activated_credit_card_account_count"] == 1
    assert business["credit_summary"]["inactive_credit_card_account_count"] == 0


def test_personal_brief_keeps_indistinguishable_masked_accounts() -> None:
    text = """
    个人信用报告 信贷记录
    2020年01月01日示例银行信用卡中心发放的贷记卡（人民币账户）。截至2024年11月，余额0。
    2020年01月01日示例银行信用卡中心发放的贷记卡（人民币账户）。截至2024年11月，余额0。
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    assert len(business["credit_accounts"]) == 2
    assert len({item["account_id"] for item in business["credit_accounts"]}) == 2


def test_personal_brief_card_type_uses_issued_product_not_later_heading() -> None:
    text = """
    个人信用报告 信贷记录
    2022年07月19日华夏银行信用卡中心发放的贷记卡（人民币账户）。
    最近5年内有3个月处于逾期状态，其中2个月逾期超过90天。
    从未逾期过的贷记卡及透支未超过60天的准贷记卡账户明细如下：
    2023年08月20日示例银行信用卡中心发放的准贷记卡（人民币账户）。
    最近5年内有1个月处于逾期状态。
    从未逾期过的贷记卡账户明细如下：
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    assert [row["business_type"] for row in business["credit_accounts"]] == [
        "贷记卡",
        "准贷记卡",
    ]
    assert [row["business_type"] for row in business["overdue_records"]] == [
        "贷记卡",
        "准贷记卡",
    ]


def test_personal_brief_preserves_complete_overdue_business_facts() -> None:
    text = """
    个人信用报告 信贷记录
    2024年01月02日示例商业银行发放的300,000元（人民币）个人经营性贷款，
    2026年02月03日到期。截至2025年11月，余额200,000，当前有逾期。
    最近5年内有6个月处于逾期状态，其中3个月逾期超过90天。
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    overdue = business["overdue_records"][0]
    assert overdue["sequence"] == 1
    assert overdue["account_type"] == "loan"
    assert overdue["management_institution"] == "示例商业银行"
    assert overdue["business_type"] == "个人经营性贷款"
    assert overdue["open_date"] == "2024-01-02"
    assert overdue["currency"] == "CNY"
    assert overdue["overdue_months"] == 6
    assert overdue["over_90_days_months"] == 3
    assert overdue["over_90_days"] is True
    assert overdue["current_overdue"] is True
    assert overdue["current_overdue_status"] == "overdue"


def test_personal_brief_normalizes_global_currency_names_without_false_cny() -> None:
    text = """
    个人信用报告 信贷记录
    2020年01月01日示例银行信用卡中心发放的贷记卡（香港元账户）。截至2024年11月，余额0。
    2020年01月02日示例银行信用卡中心发放的贷记卡（瑞士法郎账户）。截至2024年11月，余额0。
    2020年01月03日示例银行信用卡中心发放的贷记卡（未来测试币账户）。截至2024年11月，余额0。
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    assert [item["currency"] for item in business["credit_accounts"]] == [
        "HKD",
        "CHF",
        "未来测试币",
    ]


def test_currency_registry_accepts_every_current_iso_4217_list_one_code() -> None:
    assert len(ISO_4217_CURRENT_CODES) == 178
    assert {normalize_currency_code(code) for code in ISO_4217_CURRENT_CODES} == ISO_4217_CURRENT_CODES
    assert normalize_currency_code("加勒比盾") == "XCG"
    assert normalize_currency_code("津巴布韦金") == "ZWG"


def test_personal_brief_preserves_source_summary_nulls_and_unclosed_definition() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        headers=["", "信用卡", "贷款", "", "其他业务"],
        rows=[
            TableRow(
                cells=[
                    CellValue(text="账户数"),
                    CellValue(text="21"),
                    CellValue(text="2"),
                    CellValue(text="22"),
                    CellValue(text="--"),
                ]
            ),
            TableRow(
                cells=[
                    CellValue(text="未结清/未销户账户数"),
                    CellValue(text="18"),
                    CellValue(text="--"),
                    CellValue(text="11"),
                    CellValue(text="--"),
                ]
            ),
            TableRow(
                cells=[
                    CellValue(text="发生过逾期的账户数"),
                    CellValue(text="--"),
                    CellValue(text="--"),
                    CellValue(text="--"),
                    CellValue(text="--"),
                ]
            ),
            TableRow(
                cells=[
                    CellValue(text="发生过90天以上逾期的账户数"),
                    CellValue(text="--"),
                    CellValue(text="--"),
                    CellValue(text="--"),
                    CellValue(text="--"),
                ]
            ),
        ],
    )
    result = SimpleNamespace(pages=[PageContent(page_number=1, tables=[table])])

    business = extract_native_credit_business(
        result,
        "个人信用报告（个人版）",
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    summary = business["credit_summary"]
    assert summary["source_account_count"] == 45
    assert summary["source_unclosed_account_count"] == 29
    assert summary["source_unclosed_account_counts"] == {
        "credit_card": 18,
        "housing_loan": None,
        "other_loan": 11,
        "other_business": None,
    }
    assert summary["source_overdue_account_count"] is None
    assert summary["source_overdue_account_count_status"] == "not_reported"


def test_personal_brief_extracts_empty_sections_and_all_numbered_notes() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=3,
                texts=[
                    TextBlock(content="非信贷交易记录"),
                    TextBlock(content="系统中没有您最近5年内的非信贷交易记录。"),
                    TextBlock(content="公共记录"),
                    TextBlock(content="系统中没有您最近5年内的公共信息记录。"),
                ],
            ),
            PageContent(
                page_number=8,
                texts=[
                    TextBlock(content="说明"),
                    TextBlock(content="1.第一条说明。\n2.第二条说明。\n3.第三条说明。\n4.第四条说明。\n5.第五条说明。"),
                ],
            ),
        ]
    )

    content = extract_personal_brief_section_content(result, "")

    assert content["non_credit_transaction_summary"]["record_status"] == "no_records"
    assert content["public_record_summary"]["record_status"] == "no_records"
    assert [note["sequence"] for note in content["report_notes"]] == [1, 2, 3, 4, 5]


def test_personal_brief_keeps_identical_inquiry_occurrences_with_distinct_sequences() -> None:
    text = """
    个人信用报告 查询记录
    机构查询记录明细
    编号 查询日期 查询机构 查询原因
    25 2024年09月10日 中国银行股份有限公司福建省分行 贷后管理
    26 2024年09月10日 中国银行股份有限公司福建省分行 贷后管理
    个人查询记录明细
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    inquiries = business["inquiry_records"]
    assert [item["sequence"] for item in inquiries] == [25, 26]
    assert len({item["inquiry_id"] for item in inquiries}) == 2


@pytest.mark.parametrize("reason", ["资信审查", "融资审批"])
def test_personal_brief_recognizes_additional_institution_inquiry_reasons(reason: str) -> None:
    text = f"""
    个人信用报告 查询记录
    机构查询记录明细
    编号 查询日期 查询机构 查询原因
    8 2023年12月22日 福建永鸿兴融资担保有限公司 {reason}
    个人查询记录明细
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    assert business["inquiry_records"][0]["sequence"] == 8
    assert business["inquiry_records"][0]["reason"] == reason


def test_personal_brief_handles_wrapped_fields_liabilities_and_online_queries() -> None:
    text = """
    个人信用报告 信贷记录
    从未发生过逾期的账户明细如下：
    2024年01月02日示例银行信用卡中心发放的贷记卡（人民币账户，卡片尾号：1234）。
    截至2025年03月，信用额
    度10,000，已使用额
    度0，大额专项分期余
    额300，尚未激活。
    相关还款责任信息
    2023年04月05日，为张三（证件类型：身份证，证件号码：110101199001011234）
    在示例商业银行办理的个人经营性贷款承担相关还款责任，责任人类型为保证人，
    相关还款责任金额50,000（保证合同编号：HT-001）。
    截至2025年03月，余额20,000（人民币元）。
    查询记录
    机构查询记录明细
    个人查询记录明细
    1 2025年04月06日 本人查询（商业银行网上银行）
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="personal_brief",
        content_mode="native_text",
    )

    account = business["credit_accounts"][0]
    assert account["credit_limit"] == 10000
    assert account["used_amount"] == 0
    assert account["unbilled_installment_balance"] == 300
    assert account["account_status"] == "inactive"
    assert account["ever_overdue"] is False
    assert "balance" not in account

    liability = business["repayment_liability_records"][0]
    assert liability["related_party_name"] == "张三"
    assert liability["management_institution"] == "示例商业银行"
    assert liability["responsibility_amount"] == 50000
    assert liability["contract_number"] == "HT-001"
    assert liability["snapshot_date"] == "2025-03"
    assert liability["balance"] == 20000
    assert business["credit_summary"]["repayment_liability_count"] == 1
    assert business["credit_summary"]["personal_inquiry_count"] == 1
    personal_inquiry = business["inquiry_records"][0]
    assert personal_inquiry["reason"] == "本人查询"
    assert personal_inquiry["query_channel"] == "商业银行网上银行"
    assert business["credit_summary"]["activated_credit_card_account_count"] == 0
    assert business["credit_summary"]["inactive_credit_card_account_count"] == 1


def test_enterprise_extracts_summary_facilities_accounts_and_public_records() -> None:
    text = """
    企业信用报告 信息概要
    首次有信贷交易的年份 发生信贷交易的机构数 当前有未结清信贷交易的机构数
    首次有相关还款责任的年份
    2019 3 2 2024
    借贷交易 担保交易 余额 37311.68 余额 6000 其中：被追偿余额 0
    非信贷交易账户数 欠税记录条数 民事判决记录条数 强制执行记录条数 行政处罚记录条数
    0 0 0 0 0
    非循环信用额度 循环信用额度
    总额 已用额度 剩余可用额度 总额 已用额度 剩余可用额度
    3000 2500 500 4900 4000 900
    责任类型
    公共记录明细 获得许可记录
    许可部门 许可类型 许可日期 截止日期 许可内容
    示例市生态环境
    局 普通 2023-05-25 2028-07-21 排污
    许可
    认证部门 认证类型 认证日期 截止日期 认证内容
    国家税务总局 纳税信用A级纳税人 -- 2028-12-31 2022年度纳税信用A级纳税人
    附件1：信用记录补充信息
    1.未结清账户编号：G10323310H000123456789
    授信机构：示例商业银行股份有限公司
    业务种类：流动资金贷款
    信息报告日期 余额 五级分类
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="enterprise",
        content_mode="native_text",
    )

    assert business["credit_summary"]["first_credit_year"] == 2019
    assert business["credit_summary"]["active_credit_institution_count"] == 2
    assert business["credit_summary"]["credit_balance"] == 37311.68
    # Aggregate facility totals are summary facts, not source-grained records.
    assert business["credit_lines"] == []
    assert len(business["credit_accounts"]) == 1
    assert business["credit_accounts"][0]["account_status"] == "active"
    assert {item["record_type"] for item in business["public_records"]} == {
        "license",
        "certification",
    }
    license_record = next(item for item in business["public_records"] if item["record_type"] == "license")
    assert license_record["authority"] == "示例市生态环境局"
    assert license_record["content"] == "排污许可"


def test_enterprise_summary_uses_canonical_table_when_markdown_separates_headers() -> None:
    table = TableBlock(
        table_id="summary",
        headers=[
            "首次有信贷交易的年份",
            "发生信贷交易的机构数",
            "当前有未结清信贷\n交易的机构数",
            "首次有相关还款\n责任的年份",
        ],
        rows=[
            TableRow(
                cells=[
                    CellValue(text="2019"),
                    CellValue(text="3"),
                    CellValue(text="2"),
                    CellValue(text="2024"),
                ]
            )
        ],
    )
    balances = TableBlock(
        table_id="balances",
        metadata={
            "raw_rows": [
                ["借贷交易", "", "担保交易", ""],
                ["余额", "37311.68", "余额", "6000"],
                ["其中：被追偿余额", "0", "其中：关注类余额", "0"],
            ]
        },
    )
    result = SimpleNamespace(pages=[PageContent(page_number=1, tables=[table, balances])])

    business = extract_native_credit_business(
        result,
        "| headers |\n| --- |\n| 2019 | 3 | 2 | 2024 |",
        report_subtype="enterprise",
        content_mode="native_text",
    )

    assert {
        "first_credit_year": 2019,
        "credit_institution_count": 3,
        "active_credit_institution_count": 2,
        "first_repayment_responsibility_year": 2024,
    }.items() <= business["credit_summary"].items()
    assert business["credit_summary"]["credit_balance"] == 37311.68
    assert business["credit_summary"]["guarantee_balance"] == 6000
    assert business["credit_summary"]["recovered_debt_balance"] == 0


def test_enterprise_canonical_cards_join_page_continuation_and_preserve_facility_values() -> None:
    summary = TableBlock(
        table_id="pt_summary",
        metadata={
            "raw_rows": [
                ["", "正常类", "", "关注类", "", "不良类", "", "合计", ""],
                ["", "账户数", "余额", "账户数", "余额", "账户数", "余额", "账户数", "余额"],
                ["中长期借款", "1", "34.88", "0", "0", "0", "0", "1", "34.88"],
                ["循环透支", "1", "25.87", "0", "0", "0", "0", "1", "25.87"],
                ["合计", "2", "60.75", "0", "0", "0", "0", "2", "60.75"],
            ]
        },
    )
    facilities = TableBlock(
        table_id="pt_facilities",
        metadata={
            "raw_rows": [
                ["非循环信用额度", "", "", "循环信用额度", "", ""],
                ["总额", "已用额度", "剩余可用额度", "总额", "已用额度", "剩余可用额度"],
                ["0", "0", "0", "34.33", "25.87", "8.47"],
            ]
        },
    )
    first_page = TableBlock(
        table_id="pt_account_start",
        metadata={
            "raw_rows": [
                ["中长期借款", "", "", "", "共 1 笔", "", "", ""],
                ["账户编号", "授信机构", "业务种类", "开立日期", "到期日", "币种", "借款金额", "发放形式"],
                [
                    "Y10061000H00",
                    "示例汽车金融有限公司",
                    "固定资产贷款",
                    "2021-08-03",
                    "2024-08-03",
                    "人民币元",
                    "62.99",
                    "新增",
                ],
            ]
        },
    )
    continuation = TableBlock(
        table_id="pt_account_continuation",
        metadata={
            "raw_rows": [
                ["01EIP1967714", "组合", "34.88", "正常", "0", "0", "0", "2023-01-03"],
                ["", "1.94", "正常还款", "--", "--", "见附件", "2023-01-03", ""],
            ]
        },
    )
    revolving = TableBlock(
        table_id="pt_revolving",
        metadata={
            "raw_rows": [
                ["循环透支", "", "", "", "共 1 笔", "", "", ""],
                ["账户编号", "授信机构", "业务种类", "开立日期", "到期日", "币种", "信用额度", "发放形式"],
                [
                    "D10055840H0001LE20220228XS000007641",
                    "示例银行股份有限公司",
                    "流动资金贷款",
                    "2022-02-28",
                    "2023-02-28",
                    "人民币元",
                    "34.10",
                    "新增",
                ],
                ["", "保证", "25.87", "正常", "0", "0", "0", "2022-12-07"],
                [
                    "",
                    "2.10",
                    "正常还款",
                    "21",
                    "--",
                    "D10055840H0001CE20220228XS000007641",
                    "见附件",
                    "2022-12-31",
                ],
            ]
        },
    )
    facility_detail = TableBlock(
        table_id="pt_facility_detail",
        metadata={
            "raw_rows": [
                ["授信信息", "", "", "共 1 笔", "", "", ""],
                ["授信协议编号", "授信机构", "授信额度类型", "额度循环标志", "生效日期", "到期日", "信息报告日期"],
                ["", "币种", "授信额度", "已用额度", "授信限额", "授信限额编号", ""],
                [
                    "D10055840H0001CE20220228XS000007641",
                    "示例银行股份有限公司",
                    "贷款",
                    "是",
                    "2023-01-07",
                    "2023-02-28",
                    "2023-01-07",
                ],
                ["", "人民币元", "34.33", "25.87", "--", "--", ""],
            ]
        },
    )
    result = SimpleNamespace(
        pages=[
            PageContent(page_number=3, tables=[summary, facilities]),
            PageContent(page_number=4, tables=[first_page]),
            PageContent(page_number=5, tables=[continuation, revolving, facility_detail]),
        ]
    )

    business = extract_native_credit_business(
        result,
        "企业信用报告",
        report_subtype="enterprise",
        content_mode="native_text",
    )

    assert [account["account_identifier"] for account in business["credit_accounts"]] == [
        "Y10061000H0001EIP1967714",
        "D10055840H0001LE20220228XS000007641",
    ]
    first, second = business["credit_accounts"]
    assert first["balance"] == 34.88
    assert first["loan_amount"] == 62.99
    assert first["snapshot_date"] == "2023-01-03"
    assert second["balance"] == 25.87
    assert second["credit_limit"] == 34.1
    assert "loan_amount" not in second
    assert business["credit_summary"]["reported_account_count"] == 2
    assert "detail_account_balance" not in business["credit_summary"]
    assert "account_balance_reconciliation_tolerance" not in business["credit_summary"]
    assert len(business["credit_lines"]) == 1
    facility = business["credit_lines"][0]
    assert (facility["total_limit"], facility["used_limit"]) == (34.33, 25.87)
    assert "available_limit" not in facility
    assert facility["account_id"] == second["account_id"]
    assert business["credit_summary"]["facility_summary"] == {
        "non_revolving": {
            "total_limit": 0,
            "used_limit": 0,
            "available_limit": 0,
            "currency": "CNY",
            "amount_unit": "CNY_10K",
        },
        "revolving": {
            "total_limit": 34.33,
            "used_limit": 25.87,
            "available_limit": 8.47,
            "currency": "CNY",
            "amount_unit": "CNY_10K",
        },
    }


def test_enterprise_merge_rejects_truncated_canonical_account_prefix() -> None:
    canonical = [
        {
            "account_identifier": "D10123320H000170060110009",
            "account_id": "credit_account:D10123320H000170060110009",
            "source": "canonical_physical_table",
        }
    ]
    narrative = [
        {
            "account_identifier": "D10123320H000170060110009026522",
            "account_id": "credit_account:D10123320H000170060110009026522",
            "source": "enterprise_account_history",
        },
        {
            "account_identifier": "D10123320H000170060110009027778",
            "account_id": "credit_account:D10123320H000170060110009027778",
            "source": "enterprise_account_history",
        },
    ]

    merged = _merge_enterprise_accounts(canonical, narrative)

    assert {item["account_identifier"] for item in merged} == {
        "D10123320H000170060110009026522",
        "D10123320H000170060110009027778",
    }


def test_enterprise_settled_cards_keep_closed_status_and_do_not_absorb_later_tables() -> None:
    settled = TableBlock(
        table_id="settled",
        metadata={
            "raw_rows": [
                ["账户编号", "授信机构", "业务种类", "开立日期", "到期日", "", "币种", "借款金额"],
                ["", "关闭日期", "五级分类", "最后一次还款日期", "", "最后一次还款形式", "", "历史表现"],
                [
                    "N101W5810H00010123",
                    "示例银行",
                    "固定资产贷款",
                    "2012-07-01",
                    "2015-06-30",
                    "",
                    "人民币",
                    "100",
                ],
                ["", "2015-06-30", "次级", "2015-06-30", "", "担保代偿", "", "见附件"],
                [
                    "N101W5810H00012432",
                    "示例银行",
                    "固定资产贷款",
                    "2013-02-01",
                    "2015-07-31",
                    "",
                    "人民币",
                    "80",
                ],
                ["", "2015-07-31", "--", "2015-07-31", "", "正常还款", "", "见附件"],
            ]
        },
    )
    unrelated = TableBlock(
        table_id="liability",
        metadata={
            "raw_rows": [
                ["", "借款金额", "余额", "五级分类"],
                ["", "37.5", "37.5", "正常"],
            ]
        },
    )
    result = SimpleNamespace(pages=[PageContent(page_number=8, tables=[settled, unrelated])])

    business = extract_native_credit_business(
        result,
        "企业信用报告",
        report_subtype="enterprise",
        content_mode="native_text",
    )

    assert len(business["credit_accounts"]) == 2
    assert [record["account_status"] for record in business["credit_accounts"]] == ["settled", "settled"]
    assert [record["close_date"] for record in business["credit_accounts"]] == [
        "2015-06-30",
        "2015-07-31",
    ]
    assert all("balance" not in record for record in business["credit_accounts"])


def test_enterprise_public_records_support_one_cell_per_text_line() -> None:
    text = """
    企业信用报告（自主查询版）
    公共记录明细
    许可部门
    许可类型
    许可日期
    截止日期
    许可内容
    示例市生态环境局
    普通
    2024-01-02
    2025-01-02
    排污许可
    示例省示例市市场
    监督管理局
    普通
    2024-03-04
    2026-03-04
    热食类食品制售
    认证部门
    认证类型
    认证日期
    截止日期
    认证内容
    国家税务总局
    纳税信用A级纳税人
    --
    2025-12-31
    2024年度纳税信
    用A级纳税人
    附件1：信用记录补充信息
    """

    business = extract_native_credit_business(
        _result(text),
        text,
        report_subtype="enterprise",
        content_mode="native_text",
    )

    records = business["public_records"]
    assert len(records) == 3
    assert records[1]["authority"] == "示例省示例市市场监督管理局"
    assert records[2]["content"] == "2024年度纳税信用A级纳税人"


def test_derive_overdue_records_from_scanned_account_and_repayment_month() -> None:
    records = derive_overdue_records(
        [
            {
                "source_structure_id": "account-1",
                "account_status": {"normalized_value": "逾期"},
                "overdue_amount": {"normalized_value": "1,200"},
                "confidence": 0.91,
            }
        ],
        [
            {
                "grid_id": "grid-1",
                "year": 2024,
                "month": 8,
                "status": "2",
                "confidence": 0.88,
                "source_cell_refs": [{"cell_id": "status-8"}],
            }
        ],
    )

    assert len(records) == 2
    assert records[0]["period_scope"] == "account_snapshot"
    assert records[0]["overdue_amount"] == 1200
    assert records[1]["period_scope"] == "month"
    assert records[1]["overdue_level"] == 2
