# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private credit-report subtype coverage for canonical facts and Bundle v3."""

from __future__ import annotations

import asyncio
import csv
import json
import re
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.server.edition_outputs import write_outputs
from docmirror.server.output_builder import build_community_bundle
from scripts.validate.validate_community_artifacts import validate_community_artifacts

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]
_FIXTURE_DIR = Path("tests/fixtures-private/credit_report")
_DIGITAL_PERSONAL_BRIEF_DIR = _FIXTURE_DIR / "Digital Personal Brief"
_DIGITAL_ENTERPRISE_DIR = _FIXTURE_DIR / "Digital Enterprise"

_DIGITAL_PERSONAL_BRIEF_EXPECTED = {
    "人行征信报告-2025-04-14.pdf": (18, 0, 82, 10, 1),
    "人行征信报告-2026-06-24 08-52-53(1).pdf": (39, 0, 100, 13, 0),
    "征信报告_平安银行_20090811_1.pdf": (29, 4, 97, 1, 2),
    "汪婧妍征信.pdf": (39, 6, 170, 13, 0),
    "沈俊艺个人征信.pdf": (83, 3, 90, 6, 1),
    "赵思雯个人征信.pdf": (45, 4, 124, 9, 1),
    "陈是兴_征信报告_中国建设银行_20101012.pdf": (87, 5, 108, 4, 0),
    "陈是兴_征信报告_中国建设银行_20101012_1.pdf": (87, 5, 108, 4, 0),
}
_DIGITAL_PERSONAL_BRIEF_MARITAL_STATUS = {
    "人行征信报告-2025-04-14.pdf": ("divorced", "离婚"),
    "人行征信报告-2026-06-24 08-52-53(1).pdf": ("married", "已婚"),
    "征信报告_平安银行_20090811_1.pdf": ("married", "已婚"),
    "汪婧妍征信.pdf": ("married", "已婚"),
    "沈俊艺个人征信.pdf": ("divorced", "离婚"),
    "赵思雯个人征信.pdf": ("married", "已婚"),
    "陈是兴_征信报告_中国建设银行_20101012.pdf": ("married", "已婚"),
    "陈是兴_征信报告_中国建设银行_20101012_1.pdf": ("married", "已婚"),
}


def _cases(pattern: str, subtype: str, public_type: str) -> list[pytest.ParameterSet]:
    fixtures = sorted(_FIXTURE_DIR.glob(pattern))
    if not fixtures:
        return [pytest.param(Path("__missing__"), subtype, public_type, marks=pytest.mark.skip)]
    return [pytest.param(path, subtype, public_type, id=f"{subtype}-{index}") for index, path in enumerate(fixtures, 1)]


CASES = [
    *_cases("*_个人简版征信报告.pdf", "personal_brief", "personal_credit_report_brief"),
    *[
        pytest.param(path, "personal_brief", "personal_credit_report_brief", id=f"digital-personal-brief-{index}")
        for index, path in enumerate(sorted(_DIGITAL_PERSONAL_BRIEF_DIR.glob("*.pdf")), 1)
    ],
    *_cases("*_个人详版征信报告.pdf", "personal_detail", "personal_credit_report_detailed"),
    *[
        pytest.param(path, "enterprise", "enterprise_credit_report", id=f"digital-enterprise-{index}")
        for index, path in enumerate(sorted(_DIGITAL_ENTERPRISE_DIR.glob("*.pdf")), 1)
    ],
    *_cases("*_企业征信*.pdf", "enterprise", "enterprise_credit_report"),
]


def test_digital_enterprise_stacked_accounts_and_facilities_are_exact(tmp_path: Path) -> None:
    fixtures = sorted(_DIGITAL_ENTERPRISE_DIR.glob("*(1).pdf"))
    if not fixtures:
        pytest.skip("audited private digital-enterprise fixture is unavailable")
    fixture = fixtures[0]
    sealed = asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    accounts = [row["normalized"] for row in datasets["credit_accounts"]["rows"]]
    credit_lines = [row["normalized"] for row in datasets["credit_lines"]["rows"]]
    facts = semantic["domain"]["facts"]
    audit = facts["credit_extraction_audit"]
    facilities = [row["normalized"] for row in datasets["enterprise_facility_summary"]["rows"]]
    capital = [row["normalized"] for row in datasets["enterprise_capital_summary"]["rows"]]
    stakeholders = [row["normalized"] for row in datasets["enterprise_stakeholders"]["rows"]]
    relationships = [row["normalized"] for row in datasets["enterprise_relationships"]["rows"]]
    supplement = [row["normalized"] for row in datasets["enterprise_credit_supplement"]["rows"]]
    report_metadata = [row["normalized"] for row in datasets["enterprise_report_metadata"]["rows"]]
    exchange_rates = [row["normalized"] for row in datasets["enterprise_exchange_rates"]["rows"]]
    report_notes = [row["normalized"] for row in datasets["report_notes"]["rows"]]

    assert len(accounts) == 3
    assert len({account["account_identifier"] for account in accounts}) == 3
    assert [account["balance"] for account in accounts] == ["34.88", "4.67", "25.87"]
    assert [account["loan_amount"] for account in accounts] == ["62.99", "5.6", None]
    assert [account["credit_limit"] for account in accounts] == [None, None, "34.1"]
    assert len(credit_lines) == 1
    assert (credit_lines[0]["total_limit"], credit_lines[0]["used_limit"]) == ("34.33", "25.87")
    assert credit_lines[0]["available_limit"] is None
    assert credit_lines[0]["facility_product"] == "贷款"
    assert credit_lines[0]["revolving_flag"] is True
    assert datasets["credit_lines"]["rows"][0]["record_id"] == credit_lines[0]["credit_line_id"]
    assert [(row["facility_type"], row["total_limit"], row["used_limit"]) for row in facilities] == [
        ("non_revolving", "0", "0"),
        ("revolving", "34.33", "25.87"),
    ]
    assert len(supplement) == 34
    assert [
        sum(row["account_identifier"] == account["account_identifier"] for row in supplement)
        for account in accounts
    ] == [18, 5, 11]
    assert relationships[0]["relationship_type"] == "actual_controller"
    assert capital == [
        {
            "amount_unit": "CNY_10K",
            "contributor_count": 0,
            "contributor_source_page": None,
            "contributor_status": "no_records",
            "currency": "CNY",
            "registered_capital_amount": "100",
            "sequence": 1,
            "source_institution": "深圳前海微众银行股份有限公司",
            "source_page": 4,
            "update_date": "2022-10-08",
        }
    ]
    assert stakeholders[0]["source_institution"] == "深圳前海微众银行股份有限公司"
    assert stakeholders[0]["update_date"] == "2022-10-08"
    assert relationships[0]["source_institution"] == "中国工商银行股份有限公司"
    assert relationships[0]["update_date"] == "2021-01-18"
    assert report_metadata[0]["report_edition"] == "independent_query"
    assert report_metadata[0]["source_page"] == 1
    assert exchange_rates[0]["exchange_rate_usd_cny"] == "6.96"
    assert exchange_rates[0]["exchange_rate_effective_period"] == "2023-01"
    assert exchange_rates[0]["source_page"] == 2
    assert [note["sequence"] for note in report_notes] == list(range(1, 21))
    assert all(note["source_page"] == 2 for note in report_notes)
    assert [account["current_overdue_status"] for account in accounts] == [
        "not_overdue",
        "not_reported",
        "not_overdue",
    ]
    assert datasets["credit_accounts"]["status"] == "complete"
    assert datasets["credit_accounts"]["completeness"]["verified"] is True
    assert datasets["credit_lines"]["status"] == "complete"
    assert datasets["credit_lines"]["completeness"]["verified"] is True
    assert "id_type" not in facts
    assert "id_number" not in facts
    assert "subject_id" not in facts
    assert "report_edition" not in facts
    assert "exchange_rate_usd_cny" not in facts
    assert "exchange_rate_effective_period" not in facts
    assert semantic["domain"]["extensions"]["presentation_policy"]["classification"] == (
        "sensitive_enterprise_credit_data"
    )

    enhanced = bundle.render_enhanced_markdown(semantic)
    assert "**企业名称:**" in enhanced
    assert "**证件类型:**" not in enhanced
    assert "## 基本信息" in enhanced
    assert "### 企业基本信息" in enhanced
    assert "**Document type:**" not in enhanced
    assert facts["unified_social_credit_code"] in enhanced
    assert facts["organization_code"] in enhanced
    assert facts["national_tax_id"] in enhanced
    assert "**non revolving:**" not in enhanced
    assert "amount unit" not in enhanced
    assert '{"amount_unit"' not in enhanced
    assert "### 授信额度汇总" in enhanced
    assert "**额度使用率:** 75.36%" in enhanced
    assert "#### 账户 Y10061000H0001EIP1967714" in enhanced
    assert "### 信用记录补充信息" not in enhanced
    assert "##### 余额与风险" in enhanced
    assert "##### 还款表现" in enhanced
    assert "| 注册资本 | 币种 | 金额单位 | 主要出资人记录数 | 主要出资人信息状态 | 信息来源机构 | 更新日期 |" in enhanced
    assert "| 100 | 人民币 | 万元（人民币） | 0 | 无记录 | 深圳前海微众银行股份有限公司 | 2022-10-08 |" in enhanced
    assert "#### 100 · 万元（人民币）" not in enhanced
    assert "**信息来源机构:** 深圳前海微众银行股份有限公司" in enhanced
    assert "**更新日期:** 2022-10-08" in enhanced
    assert "**授信机构:** 梅赛德斯-奔驰汽车金融有限公司" in enhanced
    assert "**管理机构:**" not in enhanced
    assert "**兼容状态（已弃用）:**" not in enhanced
    account_table = enhanced.split("### 信贷账户", maxsplit=1)[1].split(
        "\n### ",
        maxsplit=1,
    )[0]
    assert (
        "| 组内序号 | 业务类别 | 账户标识 | 授信机构 | 业务类型 | 账户状态 | "
        "开立日期 | 到期日期 | 信息截至日期 |"
    ) in account_table
    assert (
        "| 1 | 中长期借款 | Y10061000H0001EIP1967714 | "
        "梅赛德斯-奔驰汽车金融有限公司 | 固定资产贷款 | 未结清 |"
    ) in account_table
    assert "| 2 | 短期借款 | JQ20220902XS0M00000460UN |" in account_table
    assert "| 3 | 循环透支 | D10055840H0001LE20220228XS000007641 |" in account_table
    assert "当前逾期报告状态" in account_table
    assert "剩余还款月数" in account_table
    assert "#### 1. 中长期借款" not in account_table
    assert "| 1 | -- | -- |" not in enhanced
    assert "#### 法定代表人/非法人组织负责人" in enhanced
    assert "法定代表人/非法人组织负责人 · 林岚挺" not in enhanced
    assert "**名称/姓名:** 林岚挺" in enhanced
    assert "#### 循环信用额度 · 贷款" not in enhanced
    assert "**授信额度类型:** 贷款" in enhanced
    report_position = enhanced.index("## 报告信息")
    notes_position = enhanced.index("## 说明")
    identity_position = enhanced.index("## 身份标识")
    assert report_position < notes_position < identity_position
    assert "| 自主查询版 |" in enhanced[report_position:notes_position]
    assert "| 20 | 更多咨询，请致电全国客户服务热线400-810-8866。 |" in enhanced[notes_position:identity_position]
    assert "| 6.96 | 2023-01 |" in enhanced[notes_position:identity_position]
    assert "detail_account_balance" not in facts["credit_summary"]
    assert "account_balance_difference" not in facts["credit_summary"]
    assert "account_balance_reconciliation_tolerance" not in facts["credit_summary"]
    assert "account_balance_reconciliation_status" not in facts["credit_summary"]
    balance_reconciliation = next(
        item for item in audit["reconciliations"] if item["name"] == "credit_account_balance"
    )
    assert balance_reconciliation == {
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
    assert "account_balance_reconciliation_tolerance" not in semantic["domain"]["data_dictionary"]["fields"]
    appendix_position = enhanced.index("## 附录：文档来源与审计信息")
    assert appendix_position > enhanced.index("## 信用记录补充信息")
    assert "企业提取完整性审计" not in enhanced[:appendix_position]
    assert "审计舍入容差" not in enhanced[:appendix_position]
    audit_appendix = enhanced[appendix_position:]
    assert "### 企业提取完整性审计" in audit_appendix
    assert "enterprise extraction audit" not in enhanced.lower()
    assert [line for line in enhanced.splitlines() if line.startswith("## ")][-1] == (
        "## 附录：文档来源与审计信息"
    )
    assert "### 账户余额可比性检查（审计信息）" in audit_appendix
    assert "**源报告账户余额合计:** 65.41" in audit_appendix
    assert "**账户明细余额计算合计:** 65.42" in audit_appendix
    assert "**两者差额:** 0.01" in audit_appendix
    assert "**审计舍入容差:** 0.02" in audit_appendix
    assert "**审计结论:** 存在差异，但在舍入容差内" in audit_appendix
    assert "审计计算不改写任何业务数据" in audit_appendix
    assert facts["credit_summary"]["first_credit_year"] == 2021
    assert facts["credit_summary"]["first_repayment_responsibility_year_status"] == "not_reported"
    assert facts["credit_summary"]["public_record_counts"]["administrative_penalties"] == 0

    _task_id, written = write_outputs(
        sealed,
        tmp_path,
        file_path=str(fixture),
        file_id="001",
        task_id="private-enterprise-reading",
        include_mirror=False,
        include_manifest=False,
    )
    assert validate_community_artifacts(written["community"]) == []
    persisted = json.loads(written["community"].read_text(encoding="utf-8"))
    persisted_semantic = json.loads(written["community_semantic"].read_text(encoding="utf-8"))
    persisted_reconciliation = next(
        item
        for item in persisted_semantic["domain"]["facts"]["credit_extraction_audit"]["reconciliations"]
        if item["name"] == "credit_account_balance"
    )
    assert persisted_reconciliation == balance_reconciliation
    assert "_audit_reconciliations" not in {
        dataset["name"] for dataset in persisted["datasets"]
    }
    audit_rows = list(
        csv.DictReader(
            (
                written["datasets"] / "_audit_cells.csv"
            ).read_text(encoding="utf-8-sig").splitlines()
        )
    )
    reconciliation_rows = [
        row
        for row in audit_rows
        if row["dataset_id"] == "_audit_reconciliations"
        and row["record_id"] == "audit:credit_account_balance"
    ]
    assert {row["field_key"]: row["value"] for row in reconciliation_rows} == {
        "expected": "65.41",
        "actual": "65.42",
        "difference": "0.01",
        "tolerance": "0.02",
        "currency": "CNY",
        "amount_unit": "CNY_10K",
        "matched": "True",
        "status": "within_rounding_tolerance",
    }
    assert {dataset["name"]: dataset["row_count"] for dataset in persisted["datasets"]} == {
        "credit_accounts": 3,
        "credit_lines": 1,
        "report_notes": 20,
        "enterprise_profile_fields": 9,
        "enterprise_report_metadata": 1,
        "enterprise_exchange_rates": 1,
        "enterprise_capital_summary": 1,
        "enterprise_stakeholders": 1,
        "enterprise_relationships": 1,
            "enterprise_facility_summary": 2,
            "enterprise_current_credit_summary": 4,
            "enterprise_extraction_audit": 5,
            "enterprise_attachment_accounts": 3,
        "enterprise_credit_supplement": 34,
    }


def test_digital_enterprise_long_attachment_is_complete_and_correctly_bound() -> None:
    fixtures = sorted(_DIGITAL_ENTERPRISE_DIR.glob("*20250710.pdf"))
    if not fixtures:
        pytest.skip("audited 77-page digital-enterprise fixture is unavailable")
    fixture = fixtures[0]
    sealed = asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    facts = semantic["domain"]["facts"]

    credit_lines = [row["normalized"] for row in datasets["credit_lines"]["rows"]]
    assert [row["account_identifier"] for row in credit_lines] == [
        "B11215800H0001N24044608",
        "B11215800H0001N25027358",
    ]
    assert [(row["total_limit"], row["used_limit"]) for row in credit_lines] == [
        ("500", "300"),
        ("400", "300"),
    ]
    assert datasets["credit_lines"]["completeness"]["verified"] is True
    assert datasets["credit_lines"]["completeness"]["expected_row_count"] == 2

    assert facts["business_registration_number"] == "913100005515731558"
    capital = datasets["enterprise_capital_summary"]["rows"][0]["normalized"]
    assert capital["registered_capital_amount"] == "6000"
    assert capital["currency"] == "CNY"
    assert capital["amount_unit"] == "CNY_10K"
    assert capital["source_page"] == 4
    assert capital["contributor_source_page"] == 5

    closed = [
        row["normalized"]
        for row in datasets["enterprise_closed_credit_summary"]["rows"]
        if not row["normalized"]["is_total"]
    ]
    assert {
        row["business_category"]: row["total_account_count"]
        for row in closed
    } == {
        "中长期借款": 13,
        "短期借款": 77,
        "贴现": 304,
        "银行承兑汇票": 1,
        "其他担保交易": 7,
    }
    responsibility = next(
        row["normalized"]
        for row in datasets["enterprise_repayment_responsibility_summary"]["rows"]
        if not row["normalized"]["is_total"]
    )
    assert responsibility["other_credit_responsibility_amount"] == "4055"
    assert responsibility["other_credit_account_count"] == 3
    assert responsibility["other_credit_balance"] == "2180"

    attachment_accounts = [
        row["normalized"] for row in datasets["enterprise_attachment_accounts"]["rows"]
    ]
    histories = [
        row["normalized"] for row in datasets["enterprise_credit_supplement"]["rows"]
    ]
    details = [
        row["normalized"]
        for row in datasets["enterprise_attachment_credit_details"]["rows"]
    ]
    transactions = [
        row["normalized"] for row in datasets["enterprise_special_transactions"]["rows"]
    ]
    assert len(attachment_accounts) == 201
    assert sum(
        row["attachment_record_type"] == "account"
        and row["account_status"] == "settled"
        for row in attachment_accounts
    ) == 190
    assert sum(
        row["attachment_record_type"] == "account"
        and row["account_status"] == "active"
        for row in attachment_accounts
    ) == 8
    assert sum(row["attachment_record_type"] == "business" for row in attachment_accounts) == 3
    assert len(histories) == 465
    assert max(row["source_page"] for row in histories) == 77
    assert histories[0]["account_identifier"] == "G10312900H000131055214010025006"
    assert histories[0]["institution"] == "上海农村商业银行股份有限公司宝山支行"
    assert len(details) == 109
    assert max(row["source_page"] for row in details) == 77
    assert len(transactions) == 23
    attachment_ids = {row["attachment_account_id"] for row in attachment_accounts}
    assert all(row["attachment_account_id"] in attachment_ids for row in histories)
    assert all(row["attachment_account_id"] in attachment_ids for row in details)
    assert all(row["attachment_account_id"] in attachment_ids for row in transactions)

    summary = facts["credit_summary"]
    assert summary["source_display_limited"] is True
    assert summary["attachment_account_count"] == 201
    assert summary["attachment_credit_detail_count"] == 109
    assert summary["attachment_special_transaction_count"] == 23
    assert datasets["credit_accounts"]["status"] == "partial"

    enhanced = bundle.render_enhanced_markdown(semantic)
    for expected in (
        "工商注册号",
        "6000",
        "已结清信贷信息概要",
        "相关还款责任信息概要",
        "附件账户及业务清单",
        "附件信贷明细",
        "特定交易提示",
        "信贷账户及附件说明",
        "账户余额可比性检查（审计信息）",
    ):
        assert expected in enhanced


def test_digital_enterprise_cross_page_business_sections_are_complete() -> None:
    fixture = _DIGITAL_ENTERPRISE_DIR / "安徽华英征信报告20250728.pdf"
    if not fixture.exists():
        pytest.skip("audited cross-page digital-enterprise fixture is unavailable")
    sealed = asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    enhanced = bundle.render_enhanced_markdown(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    facts = semantic["domain"]["facts"]

    facilities = [
        row["normalized"] for row in datasets["enterprise_facility_summary"]["rows"]
    ]
    assert [
        (row["facility_type"], row["total_limit"], row["used_limit"], row["available_limit"])
        for row in facilities
    ] == [
        ("non_revolving", "3000", "3000", "0"),
        ("revolving", "4900", "4900", "0"),
    ]

    current = [
        row["normalized"] for row in datasets["enterprise_current_credit_summary"]["rows"]
    ]
    assert len(current) == 6
    assert {
        row["business_category"]
        for row in current
        if row["transaction_group"] == "借贷交易"
    } == {"短期借款", "贴现", "合计"}
    guarantee_rows = {
        row["business_category"]: row
        for row in current
        if row["transaction_group"] == "担保交易"
    }
    assert guarantee_rows["银行承兑汇票"]["total_account_count"] == 1
    assert guarantee_rows["银行承兑汇票"]["total_balance"] == "2000"
    assert guarantee_rows["信用证"]["normal_account_count"] == 0
    assert guarantee_rows["信用证"]["total_account_count"] == 2
    assert guarantee_rows["信用证"]["total_balance"] == "4000"
    assert guarantee_rows["合计"]["total_account_count"] == 3
    assert guarantee_rows["合计"]["total_balance"] == "6000"

    responsibility = [
        row["normalized"]
        for row in datasets["enterprise_repayment_responsibility_summary"]["rows"]
        if not row["normalized"]["is_total"]
    ]
    assert len(responsibility) == 1
    assert responsibility[0]["responsibility_type"] == "保证人/反担保人"
    assert responsibility[0]["other_credit_responsibility_amount"] == "2200"
    assert responsibility[0]["other_credit_account_count"] == 1
    assert responsibility[0]["other_credit_balance"] == "1100"

    liabilities = [
        row["normalized"] for row in datasets["repayment_liability_records"]["rows"]
    ]
    assert len(liabilities) == 1
    assert liabilities[0]["account_identifier"] == "D10023330H00029030001124000265200"
    assert liabilities[0]["contract_number"] == "D10023330H0002DB2024111100000171"
    assert liabilities[0]["responsibility_amount"] == "2200"
    assert liabilities[0]["loan_or_credit_amount"] == "1100"
    assert liabilities[0]["balance"] == "1100"
    assert liabilities[0]["five_tier_class"] == "正常"
    assert liabilities[0]["snapshot_date"] == "2025-07-20"
    liability_preview = enhanced.split("### 相关还款责任信息\n", maxsplit=1)[1].split(
        "\n### ",
        maxsplit=1,
    )[0]
    assert "**保证合同编号:** D10023330H0002DB2024111100000171" in liability_preview
    assert "**还款责任金额:** 2200" in liability_preview
    assert "**借款金额/信用额度:** 1100" in liability_preview
    assert "**逾期月数/还款状态:** 0" in liability_preview
    assert "**信息报告日期:** 2025-07-20" in liability_preview

    summary = facts["credit_summary"]
    assert summary["public_record_counts"] == {
        "non_credit_accounts": 0,
        "tax_arrears": 0,
        "civil_judgments": 0,
        "enforcements": 0,
        "administrative_penalties": 0,
    }
    assert summary["public_record_type_counts"] == {
        "license": 2,
        "certification": 1,
    }
    assert "**许可记录:** 2" in enhanced
    assert "**认证记录:** 1" in enhanced
    assert "**license:**" not in enhanced
    assert "**certification:**" not in enhanced
    assert "public_record:" not in enhanced
    assert "pt_3_4" not in enhanced


def test_enterprise_self_query_keeps_account_facility_and_public_record_grains_separate() -> None:
    fixture = _FIXTURE_DIR / "企业信用报告（自主查询版）.pdf"
    if not fixture.exists():
        pytest.skip("audited enterprise self-query fixture is unavailable")
    sealed = asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    enhanced = bundle.render_enhanced_markdown(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    accounts = [row["normalized"] for row in datasets["credit_accounts"]["rows"]]
    credit_lines = [row["normalized"] for row in datasets["credit_lines"]["rows"]]
    public_records = [row["normalized"] for row in datasets["public_records"]["rows"]]
    facts = semantic["domain"]["facts"]

    assert len(accounts) == 9
    assert sum(account["status"] == "settled" for account in accounts) == 6
    assert accounts[0]["balance"] == "50"
    assert all(account["balance"] != "37.5" for account in accounts)
    assert all(account["business_type"] not in {"2015-08-01", "2015-09-01"} for account in accounts)
    assert len(credit_lines) == 1
    assert (
        credit_lines[0]["total_limit"],
        credit_lines[0]["used_limit"],
        credit_lines[0]["facility_limit"],
    ) == ("500", "100", "400")
    assert credit_lines[0]["limit_identifier"] == "N101W1100H0051239548"
    assert {
        "tax_arrears",
        "civil_judgment",
        "enforcement",
        "administrative_penalty",
        "license",
        "certification",
        "qualification",
        "patent",
        "subject_statement",
        "dispute_annotation",
    } <= {record["record_type"] for record in public_records}
    assert len(public_records) == 20
    assert facts["credit_summary"]["reported_account_count"] == 28
    assert facts["credit_summary"]["account_population_comparable"] is False
    assert datasets["credit_accounts"]["completeness"]["basis"] == "emitted_records_only"
    assert datasets["credit_accounts"]["completeness"]["expected_row_count"] == 9
    assert "id_number" not in facts
    assert "subject_id" not in facts
    assert datasets["enterprise_profile_fields"]["row_count"] == 9
    assert datasets["enterprise_stakeholders"]["row_count"] == 8
    assert datasets["enterprise_current_credit_summary"]["row_count"] == 11
    assert datasets["enterprise_closed_credit_summary"]["row_count"] == 11
    assert datasets["enterprise_repayment_responsibility_summary"]["row_count"] == 8
    assert datasets["repayment_liability_records"]["row_count"] == 3
    assert datasets["enterprise_attachment_accounts"]["row_count"] == 12
    assert datasets["enterprise_credit_supplement"]["row_count"] == 3
    assert datasets["enterprise_attachment_credit_details"]["row_count"] == 10
    assert datasets["enterprise_special_transactions"]["row_count"] == 2
    audit_rows = [
        row["normalized"]
        for row in datasets["enterprise_extraction_audit"]["rows"]
    ]
    assert all(row["reconciliation_status"] == "complete" for row in audit_rows)
    assert all(row["unresolved_record_count"] == 0 for row in audit_rows)
    public_record_tokens = {
        "utility_payment",
        "tax_arrears",
        "civil_judgment",
        "enforcement",
        "administrative_penalty",
        "social_security_payment",
        "license",
        "certification",
        "qualification",
        "award",
        "export_quality",
        "inspection_exemption",
        "regulatory_supervision",
        "patent",
        "financing_restriction",
        "data_provider_statement",
        "credit_bureau_statement",
        "subject_statement",
        "dispute_annotation",
    }
    assert not any(token in enhanced for token in public_record_tokens)
    public_record_table = enhanced.split("### 公共记录\n", maxsplit=1)[1].split(
        "\n## ",
        maxsplit=1,
    )[0]
    assert (
        "| 序号 | 记录类型 | 主管/发布机构 | 记录类别 | 生效/发生日期 | 截止日期 | 记录内容 |"
        in public_record_table
    )
    assert public_record_table.count("\n| ") == len(public_records) + 2
    assert "| 许可记录 |" in public_record_table
    assert "| 认证记录 |" in public_record_table
    assert "#### " not in public_record_table
    assert "#### 账户 借贷交易" not in enhanced
    assert "#### 账户 担保交易" not in enhanced


@pytest.mark.parametrize("fixture,subtype,public_type", CASES)
def test_credit_report_subtype_projects_complete_v3(
    fixture: Path,
    subtype: str,
    public_type: str,
    tmp_path: Path,
) -> None:
    sealed = asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    result = sealed.to_read_view()
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    assert "report_subtype" not in result.entities.domain_specific
    assert payload is not None
    assert payload["document"]["type"] == public_type
    assert semantic["classification"]["document_type"] == public_type
    assert semantic["schema"]["document_type"] == public_type
    assert semantic["source"]["fingerprint"] == sealed.integrity_fingerprint
    assert semantic["domain"]["facts"]["report_subtype"] == subtype
    assert semantic["structure"]["blocks"]
    assert len(semantic["bindings"]) == sum(dataset["row_count"] for dataset in semantic["datasets"])
    assert validate_projection_payload("community_semantic", semantic).valid
    assert validate_projection_payload("community", payload).valid
    assert payload["sections"]
    assert any(dataset["row_count"] > 0 for dataset in payload["datasets"])
    if subtype == "enterprise":
        enterprise_datasets = {
            dataset["name"]: dataset for dataset in payload["datasets"]
        }
        audit_dataset = enterprise_datasets["enterprise_extraction_audit"]
        audit_rows = [row["normalized"] for row in audit_dataset["rows"]]
        assert all(
            row["expected_record_count"] == row["extracted_record_count"]
            for row in audit_rows
        )
        assert all(
            row["reconciliation_status"] == "complete"
            and row["unresolved_record_count"] == 0
            for row in audit_rows
        )
    if fixture.name in _DIGITAL_PERSONAL_BRIEF_EXPECTED:
        expected_accounts, expected_liabilities, expected_inquiries, expected_personal, expected_inactive = (
            _DIGITAL_PERSONAL_BRIEF_EXPECTED[fixture.name]
        )
        datasets = {dataset["name"]: dataset["rows"] for dataset in payload["datasets"]}
        inquiry_dataset = next(dataset for dataset in payload["datasets"] if dataset["name"] == "inquiry_records")
        accounts = datasets["credit_accounts"]
        inquiries = datasets["inquiry_records"]
        liabilities = datasets.get("repayment_liability_records", [])
        assert len(accounts) == expected_accounts
        assert len(liabilities) == expected_liabilities
        assert len(inquiries) == expected_inquiries
        assert semantic["document"]["title"] == "个人信用报告"
        assert inquiry_dataset["completeness"]["verified"] is True
        assert sum(row["normalized"]["inquiry_type"] == "personal" for row in inquiries) == expected_personal
        personal_inquiries = [row["normalized"] for row in inquiries if row["normalized"]["inquiry_type"] == "personal"]
        assert all(row["reason"] == "本人查询" for row in personal_inquiries)
        assert all(row["source_reason"].startswith("本人查询") for row in personal_inquiries)
        assert sum(row["normalized"]["status"] == "inactive" for row in accounts) == expected_inactive
        enhanced_preview = bundle.render_enhanced_markdown(semantic)
        account_types = {row["normalized"]["account_type"] for row in accounts}
        if "credit_card" in account_types:
            assert "#### 信用卡账户" in enhanced_preview
            credit_card_preview = enhanced_preview.split("#### 信用卡账户", maxsplit=1)[1].split(
                "\n#### ",
                maxsplit=1,
            )[0]
            assert "信用额度" in credit_card_preview
            assert "贷款发放金额" not in credit_card_preview
        if "loan" in account_types:
            assert "#### 贷款账户" in enhanced_preview
            loan_preview = enhanced_preview.split("#### 贷款账户", maxsplit=1)[1].split(
                "\n#### ",
                maxsplit=1,
            )[0]
            assert "贷款发放金额" in loan_preview
            assert "信用额度" not in loan_preview
        if "credit_line" in account_types:
            assert "#### 贷款授信" in enhanced_preview
        if liabilities:
            liability_preview = enhanced_preview.split("### 相关还款责任信息", maxsplit=1)[1].split(
                "\n### ",
                maxsplit=1,
            )[0]
            assert (
                "| 组内序号 | 责任发生日期 | 相关方名称 | 管理机构 | 业务类型 | "
                "责任金额 | 余额 | 币种 |"
            ) in liability_preview
            assert "| sequence |" not in liability_preview
            assert "liability date" not in liability_preview
            assert "related party name" not in liability_preview
            assert "responsibility amount" not in liability_preview
        information_summary_preview = enhanced_preview.split("## 信息概要", maxsplit=1)[1].split(
            "\n## 信贷记录",
            maxsplit=1,
        )[0]
        assert "### 个人信息" in information_summary_preview
        assert "### 信用概览" in information_summary_preview
        assert "### 报告信息" in information_summary_preview
        assert "## 附录：文档来源与提取信息" in enhanced_preview
        assert semantic["domain"]["facts"]["id_number"] in information_summary_preview
        assert semantic["domain"]["facts"]["report_number"] in information_summary_preview
        expected_marital_status, expected_marital_label = _DIGITAL_PERSONAL_BRIEF_MARITAL_STATUS[fixture.name]
        assert semantic["domain"]["facts"]["marital_status"] == expected_marital_status
        assert semantic["domain"]["entity_fields"]["marital_status"] == expected_marital_status
        assert f"**婚姻状况:** {expected_marital_label}" in information_summary_preview
        assert not any(re.search(r"[A-Za-z_]", label) for label in re.findall(r"\*\*([^*]+):\*\*", enhanced_preview))
        if personal_inquiries:
            assert "#### 个人查询" in enhanced_preview
            assert "#### 本人查询" not in enhanced_preview
            assert "| 本人 | 本人查询 | 个人查询 |" in enhanced_preview
        if fixture.name == "人行征信报告-2026-06-24 08-52-53(1).pdf":
            overdue_rows = datasets["overdue_records"]
            normalized_overdue = [row["normalized"] for row in overdue_rows]
            huaxia_account = next(
                row["normalized"]
                for row in accounts
                if row["normalized"]["institution"] == "华夏银行股份有限公司信用卡中心"
                and row["normalized"]["open_date"] == "2022-07-19"
            )
            huaxia_overdue = next(
                row
                for row in normalized_overdue
                if row["institution"] == "华夏银行股份有限公司信用卡中心"
                and row["open_date"] == "2022-07-19"
            )
            assert huaxia_account["business_type"] == "贷记卡"
            assert huaxia_overdue["business_type"] == "贷记卡"
            assert len(normalized_overdue) == 9
            assert [row["over_90_days_months"] for row in normalized_overdue] == [1, 3, 2, 2, 2, 1, 2, 1, 0]
            assert all(row["current_overdue_status"] == "overdue" for row in normalized_overdue)
            assert sum(row["over_90_days"] is True for row in normalized_overdue) == 8
            overdue_markdown = enhanced_preview.split("### 逾期记录", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
            assert (
                "| 组内序号 | 账户类型 | 管理机构 | 业务类型 | 卡片尾号 | 开立日期 | "
                "最近5年逾期月数 | 其中超过90天月数 | 当前逾期状态 |"
            ) in overdue_markdown
            assert "account id" not in overdue_markdown
            assert "overdue id" not in overdue_markdown
            assert "last_5_years" not in overdue_markdown
            assert "当前有逾期" in overdue_markdown
        if (expected_accounts, expected_liabilities, expected_inquiries) == (45, 4, 124):
            semantic_datasets = {dataset["name"]: dataset for dataset in semantic["datasets"]}
            summary = semantic["domain"]["facts"]["credit_summary"]
            audit = semantic["domain"]["facts"]["credit_extraction_audit"]
            sections = semantic["structure"]["sections"]
            owned_blocks = [block_ref for section in sections for block_ref in section["block_refs"]]
            inquiry_bindings = [
                binding
                for binding in semantic["bindings"]
                if binding["dataset_id"] == semantic_datasets["inquiry_records"]["id"]
            ]
            logical_inquiry_table = next(
                table for table in semantic["structure"]["source_tables"] if table["id"] == "logical:ds_inquiry_records"
            )

            assert "credit_lines" not in semantic_datasets
            assert semantic_datasets["report_notes"]["row_count"] == 5
            assert semantic["domain"]["data_dictionary"]["fields"]
            assert semantic["domain"]["extensions"]["rendering_contract"]["do_not_union_representations"] is True
            assert summary["source_unclosed_account_count"] == 29
            assert summary["source_account_count"] == 45
            assert summary["source_overdue_account_count"] is None
            assert summary["source_overdue_account_count_status"] == "not_reported"
            assert audit["source_page_complete"] is True
            assert audit["evidence_complete"] is True
            assert len(sections) == 6
            assert len(owned_blocks) == len(set(owned_blocks))
            assert len(semantic["structure"]["outline"]) == 6
            assert len(semantic["structure"]["cross_page_flows"]) == 1
            assert len(logical_inquiry_table["rows"]) == 124
            assert len(logical_inquiry_table["segments"]) == 5
            assert len(inquiry_bindings) == 124
            assert all(binding["source_block_refs"] for binding in inquiry_bindings)
            assert all(binding["source_table_refs"] == ["logical:ds_inquiry_records"] for binding in inquiry_bindings)
            assert all(binding["evidence_refs"] for binding in inquiry_bindings)
            assert semantic["diagnostics"]["evidence_ids"]
        if fixture.name == "赵思雯个人征信.pdf":
            normalized_accounts = [row["normalized"] for row in accounts]
            credit_cards = [row for row in normalized_accounts if row["account_type"] == "credit_card"]
            loans = [row for row in normalized_accounts if row["account_type"] != "credit_card"]
            assert [row["sequence"] for row in credit_cards] == list(range(1, 22))
            assert [row["sequence"] for row in loans] == list(range(1, 25))
            assert next(row for row in credit_cards if row["sequence"] == 13)["currency"] == "HKD"
            assert next(row for row in credit_cards if row["sequence"] == 14)["currency"] == "CHF"
            assert any(block["text"].strip() == "说明" for block in semantic["structure"]["blocks"])
            assert any(
                row and row[0] == "账户数" for table in semantic["structure"]["source_tables"] for row in table["rows"]
            )
            institution_sequences = {
                int(row["normalized"]["sequence"])
                for row in inquiries
                if row["normalized"]["inquiry_type"] == "institution"
            }
            assert institution_sequences == set(range(1, 116))
            _task_id, written = write_outputs(
                sealed,
                tmp_path,
                file_path=str(fixture),
                file_id="001",
                task_id="private-reading-parity",
                include_mirror=False,
                include_manifest=False,
            )
            persisted = json.loads(written["community"].read_text(encoding="utf-8"))
            assert persisted["document"]["title"] == "个人信用报告"
            persisted_inquiries = next(
                dataset for dataset in persisted["datasets"] if dataset["name"] == "inquiry_records"
            )
            persisted_accounts = next(
                dataset for dataset in persisted["datasets"] if dataset["name"] == "credit_accounts"
            )
            with (written["community"].parent / persisted_accounts["csv"]).open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                account_csv_rows = list(csv.DictReader(stream))
            with (written["community"].parent / persisted_inquiries["csv"]).open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                csv_rows = list(csv.DictReader(stream))
            with (written["community"].parent / "001_datasets" / "_audit_cells.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                audit_rows = list(csv.DictReader(stream))
            reading_table = next(
                table for table in persisted["reading"]["tables"] if table["dataset_id"] == persisted_inquiries["id"]
            )
            enhanced = written["enhanced_reading"].read_text(encoding="utf-8")
            assert next(line for line in enhanced.splitlines() if line.startswith("# ")) == "# 个人信用报告"
            assert (
                next(
                    row for row in account_csv_rows if row["account_type"] == "credit_card" and row["sequence"] == "13"
                )["currency"]
                == "HKD"
            )
            assert (
                next(
                    row for row in account_csv_rows if row["account_type"] == "credit_card" and row["sequence"] == "14"
                )["currency"]
                == "CHF"
            )
            table_lines = enhanced.split(f"### {reading_table['title']}", maxsplit=1)[1].split(
                "\n## ",
                maxsplit=1,
            )[0]
            table_count = sum(line.startswith("| ---") for line in table_lines.splitlines())
            markdown_rows = sum(line.startswith("| ") for line in table_lines.splitlines()) - (2 * table_count)
            assert validate_community_artifacts(written["community"]) == []
            assert (
                len(persisted_inquiries["rows"]) == len(csv_rows) == reading_table["row_count"] == markdown_rows == 124
            )
            assert "#### 机构查询" in table_lines
            assert "#### 个人查询" in table_lines
            assert [row["record_id"] for row in persisted_inquiries["rows"]] == [row["record_id"] for row in csv_rows]
            assert all(row["bbox"] for row in audit_rows)
            assert all(row["confidence"] for row in audit_rows)
            assert all(row["evidence_ref"] for row in audit_rows)
            assert semantic["domain"]["facts"]["id_number"] in enhanced
            assert semantic["domain"]["facts"]["report_number"] in enhanced
            information_summary = enhanced.split("## 信息概要", maxsplit=1)[1].split("\n## 信贷记录", maxsplit=1)[0]
            appendix = enhanced.split("## 附录：文档来源与提取信息", maxsplit=1)[1]
            assert information_summary.index("### 个人信息") < information_summary.index("### 信用概览")
            assert information_summary.index("### 信用概览") < information_summary.index("### 报告信息")
            assert "**内容模式:**" not in information_summary
            assert "**数据来源:**" not in information_summary
            assert "**旧版派生有效状态口径:**" not in information_summary
            assert "**源概要表标识:**" not in information_summary
            assert "**内容模式:**" in appendix
            assert "**数据来源:**" in appendix
            assert "**源概要表标识:**" in appendix
            assert "**源概要表页码:**" in appendix
            bold_labels = re.findall(r"\*\*([^*]+):\*\*", enhanced)
            assert bold_labels
            assert not any(re.search(r"[A-Za-z_]", label) for label in bold_labels)
