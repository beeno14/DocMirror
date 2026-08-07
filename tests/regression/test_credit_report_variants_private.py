# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private credit-report subtype coverage for canonical facts and Bundle v3."""

from __future__ import annotations

import asyncio
import csv
import io
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
_KUNMING_YUXUAN_FIXTURE = _DIGITAL_ENTERPRISE_DIR / "昆明煜萱.pdf"
_PERSONAL_BRIEF_DISPLAY_SAMPLE = _FIXTURE_DIR / "个人信用报告（本人简版）展示样本.pdf"

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


def test_digital_enterprise_institution_credit_code_is_preserved() -> None:
    if not _KUNMING_YUXUAN_FIXTURE.exists():
        pytest.skip("昆明煜萱 digital-enterprise regression fixture is unavailable")

    sealed = asyncio.run(
        perceive_document(
            _KUNMING_YUXUAN_FIXTURE,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(_KUNMING_YUXUAN_FIXTURE))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)

    assert semantic["domain"]["facts"]["institution_credit_code"] == "G1053011404727700K"
    assert semantic["domain"]["data_dictionary"]["fields"]["institution_credit_code"] == {
        "label": "机构信用代码",
        "type": "string",
        "format": "long_id",
        "sensitive": True,
    }
    identity_section = next(section for section in payload["sections"] if section["title"] == "身份标识")
    identity_item = next(item for item in identity_section["items"] if item["key"] == "institution_credit_code")
    assert identity_item == {
        "key": "institution_credit_code",
        "label": "机构信用代码",
        "value": "G1053011404727700K",
        "raw": "G1053011404727700K",
        "type": "string",
        "sensitive": True,
    }


def test_personal_brief_display_sample_projects_complete_business_schema() -> None:
    if not _PERSONAL_BRIEF_DISPLAY_SAMPLE.exists():
        pytest.skip("personal-brief display sample is unavailable")
    sealed = asyncio.run(
        perceive_document(
            _PERSONAL_BRIEF_DISPLAY_SAMPLE,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(_PERSONAL_BRIEF_DISPLAY_SAMPLE))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    expected_counts = {
        "credit_accounts": 15,
        "repayment_liability_records": 4,
        "overdue_records": 8,
        "inquiry_records": 12,
        "identity_documents": 2,
        "personal_report_metadata": 1,
        "personal_credit_summary_records": 20,
        "asset_disposition_records": 1,
        "guarantor_compensation_records": 1,
        "postpaid_records": 3,
        "tax_arrears_records": 1,
        "civil_judgment_records": 2,
        "enforcement_records": 2,
        "administrative_penalty_records": 1,
        "public_records": 6,
        "report_notes": 5,
    }
    assert {name: datasets[name]["row_count"] for name in expected_counts} == expected_counts
    assert validate_projection_payload("community_semantic", semantic).valid
    assert validate_projection_payload("community", payload).valid

    accounts = [row["normalized"] for row in datasets["credit_accounts"]["rows"]]
    credit_line = next(account for account in accounts if account["account_type"] == "credit_line")
    transferred = next(account for account in accounts if account["account_lifecycle_state"] == "transferred_out")
    bad_debt = next(account for account in accounts if account["credit_quality_status"] == "bad_debt")
    unactivated = next(account for account in accounts if account["card_activation_state"] == "not_activated")
    quasi_card = next(account for account in accounts if account.get("credit_card_type") == "quasi_credit_card")
    assert credit_line["credit_line_validity_type"] == "perpetual"
    assert credit_line.get("credit_line_expiry_date") is None
    assert credit_line.get("due_date") is None
    assert transferred["termination_event_type"] == "transferred_out"
    assert transferred["transfer_out_date"] == "2023-11"
    assert bad_debt["business_type"] == "贷记卡"
    assert unactivated["account_currency"] == "USD"
    assert unactivated["reporting_amount_currency"] == "CNY"
    assert quasi_card["business_type"] == "准贷记卡"
    assert all(account["reporting_amount_currency"] == "CNY" for account in accounts)

    summary = semantic["domain"]["facts"]["credit_summary"]
    assert summary["source_account_count"] == summary["account_count"] == 15
    assert summary["source_unclosed_account_count"] == summary["unclosed_account_count"] == 9
    assert summary["source_personal_liability_count"] == 2
    assert summary["source_enterprise_liability_count"] == 2
    assert summary["activated_credit_card_account_count"] == 0
    assert summary["inactive_credit_card_account_count"] == 1
    assert summary["closed_credit_card_account_count"] == 1
    assert summary["settled_account_count"] == 4
    assert summary["transferred_out_account_count"] == 1

    inquiries = [row["normalized"] for row in datasets["inquiry_records"]["rows"]]
    assert sum(row["inquiry_type"] == "institution" for row in inquiries) == 9
    assert sum(row["inquiry_type"] == "personal" for row in inquiries) == 3
    identities = [row["normalized"] for row in datasets["identity_documents"]["rows"]]
    assert [(row["document_type"], row["is_primary"]) for row in identities] == [
        ("身份证", True),
        ("护照", False),
    ]
    postpaid = [row["normalized"] for row in datasets["postpaid_records"]["rows"]]
    assert [row["current_arrears_amount"] for row in postpaid] == ["500", "200", "0"]
    penalties = [row["normalized"] for row in datasets["administrative_penalty_records"]["rows"]]
    assert penalties[0]["effective_date"] == "2021-08"
    assert penalties[0]["end_date"] == "2024-07"


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
        sum(row["account_identifier"] == account["account_identifier"] for row in supplement) for account in accounts
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
    assert (
        "| 注册资本 | 币种 | 金额单位 | 主要出资人记录数 | 主要出资人信息状态 | 信息来源机构 | 更新日期 |" in enhanced
    )
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
        "| 序号 | 业务类别 | 账户标识 | 授信机构 | 业务类型 | 账户状态 | 开立日期 | 到期日期 | 信息截至日期 |"
    ) in account_table
    assert (
        "| 1 | 中长期借款 | Y10061000H0001EIP1967714 | 梅赛德斯-奔驰汽车金融有限公司 | 固定资产贷款 | 未结清 |"
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
    balance_reconciliation = next(item for item in audit["reconciliations"] if item["name"] == "credit_account_balance")
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
    assert [line for line in enhanced.splitlines() if line.startswith("## ")][-1] == ("## 附录：文档来源与审计信息")
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
    assert "community_semantic" not in written
    assert "semantic_json" not in persisted["files"]
    assert not (written["community"].parent / "001_community_semantic.json").exists()
    assert "_audit_reconciliations" not in {dataset["name"] for dataset in persisted["datasets"]}
    audit_rows = list(
        csv.DictReader((written["datasets"] / "_audit_cells.csv").read_text(encoding="utf-8-sig").splitlines())
    )
    reconciliation_rows = [
        row
        for row in audit_rows
        if row["dataset_id"] == "_audit_reconciliations" and row["record_id"] == "audit:credit_account_balance"
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
            "enterprise_report_identity": 1,
            "enterprise_credit_overview": 1,
            "enterprise_public_record_counts": 5,
            "enterprise_profile": 1,
            "enterprise_capital_summary": 1,
            "enterprise_key_personnel": 1,
            "enterprise_stakeholders": 1,
            "enterprise_relationships": 1,
            "enterprise_credit_accounts": 3,
            "enterprise_credit_facilities": 1,
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

    accounts = [row["normalized"] for row in datasets["credit_accounts"]["rows"]]
    credit_lines = [row["normalized"] for row in datasets["credit_lines"]["rows"]]
    account_identifiers = {row["account_identifier"] for row in accounts}
    assert len(accounts) == 97
    assert {
        "2142019656",
        "2142018846",
        "D10123320H000170060110009028463",
    }.issubset(account_identifiers)
    assert "D10123320H000170060110009" not in account_identifiers
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
    assert {row["business_category"]: row["total_account_count"] for row in closed} == {
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

    attachment_accounts = [row["normalized"] for row in datasets["enterprise_attachment_accounts"]["rows"]]
    histories = [row["normalized"] for row in datasets["enterprise_credit_supplement"]["rows"]]
    details = [row["normalized"] for row in datasets["enterprise_attachment_credit_details"]["rows"]]
    transactions = [row["normalized"] for row in datasets["enterprise_special_transactions"]["rows"]]
    assert len(attachment_accounts) == 201
    assert (
        sum(
            row["attachment_record_type"] == "account" and row["account_status"] == "settled"
            for row in attachment_accounts
        )
        == 190
    )
    assert (
        sum(
            row["attachment_record_type"] == "account" and row["account_status"] == "active"
            for row in attachment_accounts
        )
        == 8
    )
    assert sum(row["attachment_record_type"] == "business" for row in attachment_accounts) == 3
    assert len(histories) == 477
    expected_cross_page_history_counts = {
        "B11313900H0001216450100300036861": 1,
        "D10123320H000170060110008723060": 2,
        "D10123320H000170060110007838469": 1,
        "0231000272200408000100": 1,
        "F10233450H00012018060500002506": 1,
        "216160100300390110": 1,
        "D10123320H00012024010333324851": 1,
        "D10123320H00012023061260320545": 1,
        "D10123320H00012022110399785230": 2,
        "D10123320H00012022051680855235": 1,
    }
    assert {
        account_identifier: sum(row["account_identifier"] == account_identifier for row in histories)
        for account_identifier in expected_cross_page_history_counts
    } == expected_cross_page_history_counts
    assert max(row["source_page"] for row in histories) == 77
    assert histories[0]["account_identifier"] == "G10312900H000131055214010025006"
    assert histories[0]["institution"] == "上海农村商业银行股份有限公司宝山支行"
    assert len(details) == 109
    assert all(row["five_tier_class"] for row in details)
    assert sum(row["five_tier_class_source"] == "parent_attachment_heading" for row in details) == 8
    active_discount = next(row for row in details if row["account_identifier"] == "D10123320H00012025070160898220")
    assert active_discount["guarantee_type"] == "信用/无担保"
    assert active_discount["snapshot_date"] == "2025-07-01"
    assert active_discount["credit_agreement_identifier"] == ""
    assert active_discount["credit_agreement_status"] == "not_reported"
    assert max(row["source_page"] for row in details) == 77
    assert len(transactions) == 23
    attachment_ids = {row["attachment_account_id"] for row in attachment_accounts}
    assert all(row["attachment_account_id"] in attachment_ids for row in histories)
    assert all(row["attachment_account_id"] in attachment_ids for row in details)
    assert all(row["attachment_account_id"] in attachment_ids for row in transactions)

    rendered_csvs = bundle.render_dataset_csvs(semantic)
    account_csv = next(content for path, content in rendered_csvs.items() if path.endswith("/credit_accounts.csv"))
    history_csv = next(
        content for path, content in rendered_csvs.items() if path.endswith("/enterprise_credit_supplement.csv")
    )
    detail_csv = next(
        content for path, content in rendered_csvs.items() if path.endswith("/enterprise_attachment_credit_details.csv")
    )
    account_csv_rows = list(csv.DictReader(account_csv.splitlines()))
    history_csv_rows = list(csv.DictReader(history_csv.splitlines()))
    detail_csv_rows = list(csv.DictReader(detail_csv.splitlines()))
    csv_account_identifiers = {row["account_identifier"].lstrip("'") for row in account_csv_rows}
    assert len(account_csv_rows) == 97
    assert len(history_csv_rows) == 477
    assert {
        "2142019656",
        "2142018846",
        "D10123320H000170060110009028463",
    }.issubset(csv_account_identifiers)
    active_discount_csv = next(
        row for row in detail_csv_rows if row["account_identifier"].lstrip("'") == "D10123320H00012025070160898220"
    )
    assert active_discount_csv["guarantee_type"] == "信用/无担保"
    assert active_discount_csv["snapshot_date"] == "2025-07-01"
    assert active_discount_csv["credit_agreement_status"] == "not_reported"

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
    accounts = [row["normalized"] for row in datasets["credit_accounts"]["rows"]]
    details = [row["normalized"] for row in datasets["enterprise_attachment_credit_details"]["rows"]]

    assert sum(row["issuance_form"] == "新增" for row in accounts) == 5
    assert len(details) == 188
    assert {
        status: sum(row["credit_agreement_status"] == status for row in details)
        for status in ("reported", "not_reported", "not_applicable")
    } == {
        "reported": 1,
        "not_reported": 79,
        "not_applicable": 108,
    }
    guarantee_detail = next(
        row for row in details if row["account_identifier"] == "G10423771H00065402377100013202503281003069991742282"
    )
    assert guarantee_detail["guarantee_type"] == ""
    assert guarantee_detail["counter_guarantee_type"] == "信用/无担保/保证金"
    assert guarantee_detail["deposit_ratio"] == "0.5"
    assert guarantee_detail["balance"] == "2000"
    assert guarantee_detail["risk_exposure_amount"] == "1000"
    assert guarantee_detail["credit_agreement_identifier"] == "G10423771H00065879024720250329"
    assert guarantee_detail["credit_agreement_status"] == "reported"
    assert guarantee_detail["snapshot_date"] == "2025-03-29"
    for account_identifier, expected_snapshot in (
        ("G10323310H0001DLC8022025000052", "2025-05-13"),
        ("G10323310H0001DLC8022025000055", "2025-05-14"),
    ):
        detail = next(row for row in details if row["account_identifier"] == account_identifier)
        assert detail["counter_guarantee_type"] == "信用/无担保/保证金"
        assert detail["deposit_ratio"] == "0.5"
        assert detail["credit_agreement_identifier"] == ""
        assert detail["credit_agreement_status"] == "not_reported"
        assert detail["snapshot_date"] == expected_snapshot

    detail_csv = next(
        content
        for path, content in bundle.render_dataset_csvs(semantic).items()
        if path.endswith("/enterprise_attachment_credit_details.csv")
    )
    detail_csv_rows = list(csv.DictReader(detail_csv.splitlines()))
    guarantee_detail_csv = next(
        row
        for row in detail_csv_rows
        if row["account_identifier"].lstrip("'") == "G10423771H00065402377100013202503281003069991742282"
    )
    assert guarantee_detail_csv["counter_guarantee_type"] == "信用/无担保/保证金"
    assert guarantee_detail_csv["deposit_ratio"] == "0.5"
    assert guarantee_detail_csv["balance"] == "2000"
    assert guarantee_detail_csv["risk_exposure_amount"] == "1000"
    assert guarantee_detail_csv["credit_agreement_identifier"].lstrip("'") == "G10423771H00065879024720250329"
    assert guarantee_detail_csv["snapshot_date"] == "2025-03-29"

    facilities = [row["normalized"] for row in datasets["enterprise_facility_summary"]["rows"]]
    assert [
        (row["facility_type"], row["total_limit"], row["used_limit"], row["available_limit"]) for row in facilities
    ] == [
        ("non_revolving", "3000", "3000", "0"),
        ("revolving", "4900", "4900", "0"),
    ]

    current = [row["normalized"] for row in datasets["enterprise_current_credit_summary"]["rows"]]
    assert len(current) == 6
    assert {row["business_category"] for row in current if row["transaction_group"] == "借贷交易"} == {
        "短期借款",
        "贴现",
        "合计",
    }
    guarantee_rows = {row["business_category"]: row for row in current if row["transaction_group"] == "担保交易"}
    assert guarantee_rows["银行承兑汇票"]["total_account_count"] == 1
    assert guarantee_rows["银行承兑汇票"]["total_balance"] == "2000"
    assert guarantee_rows["信用证"]["normal_account_count"] == 0
    assert guarantee_rows["信用证"]["total_account_count"] == 2
    assert guarantee_rows["信用证"]["total_balance"] == "4000"
    assert guarantee_rows["合计"]["total_account_count"] == 3
    assert guarantee_rows["合计"]["total_balance"] == "6000"

    displayed = [
        row["normalized"]
        for row in datasets["enterprise_displayed_credit_summary"]["rows"]
    ]
    assert len(displayed) == 7
    settled_discount = next(
        row
        for row in displayed
        if row["settlement_status"] == "settled"
        and row["business_category"] == "贴现"
    )
    assert settled_discount["source_group_account_count"] == 100
    assert settled_discount["source_account_count"] == 100
    assert settled_discount["source_reported_amount"] == "3836.96"
    assert settled_discount["amount_kind"] == "discount_amount"
    active_discount = next(
        row
        for row in displayed
        if row["settlement_status"] == "active"
        and row["business_category"] == "贴现"
    )
    assert active_discount["source_group_account_count"] == 77
    assert active_discount["source_account_count"] == 77
    assert active_discount["source_reported_amount"] == "7511.68"
    assert active_discount["overdue_total"] == "0"
    assert active_discount["overdue_principal"] == "0"
    displayed_csv = next(
        content
        for path, content in bundle.render_dataset_csvs(semantic).items()
        if path.endswith("/enterprise_displayed_credit_summary.csv")
    )
    displayed_csv_rows = list(csv.DictReader(displayed_csv.splitlines()))
    assert len(displayed_csv_rows) == 7
    assert any(
        row["settlement_status"] == "settled"
        and row["business_category"] == "贴现"
        and row["source_reported_amount"] == "3836.96"
        for row in displayed_csv_rows
    )
    assert "### 信贷记录明细分组汇总" in enhanced
    assert "3836.96" in enhanced

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

    liabilities = [row["normalized"] for row in datasets["repayment_liability_records"]["rows"]]
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
    assert "**信息截至日期:** 2025-07-20" in liability_preview

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
    semantic_datasets = {dataset["name"]: dataset for dataset in semantic["datasets"]}
    data_dictionary = semantic["domain"]["data_dictionary"]
    accounts = [row["normalized"] for row in datasets["credit_accounts"]["rows"]]
    credit_lines = [row["normalized"] for row in datasets["credit_lines"]["rows"]]
    typed_public_datasets = {
        name: dataset
        for name, dataset in semantic_datasets.items()
        if name.startswith("enterprise_public_") and name.endswith("_records")
    }
    dataset_names = [dataset["name"] for dataset in payload["datasets"]]
    public_section = next(
        section for section in payload["sections"] if section["type"] == "public_records"
    )
    facts = semantic["domain"]["facts"]

    assert data_dictionary["schema_id"] == "enterprise_credit_report"
    assert data_dictionary["version"] == "2.0.0"
    assert "identity_documents" not in data_dictionary["datasets"]
    assert "enterprise_credit_accounts" in data_dictionary["datasets"]
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
    assert set(payload) == {
        "schema",
        "document",
        "sections",
        "datasets",
        "reading",
        "files",
        "warnings",
    }
    assert "public_records" not in semantic
    assert "public_records" not in datasets
    assert dataset_names == [
        name
        for name in semantic["domain"]["extensions"]["dataset_document_order"]
        if name in datasets
    ]
    assert dataset_names.index("enterprise_current_credit_summary") < dataset_names.index(
        "enterprise_facility_summary"
    ) < dataset_names.index("enterprise_closed_credit_summary")
    assert dataset_names.index("enterprise_profile_fields") < dataset_names.index(
        "enterprise_capital_summary"
    ) < dataset_names.index("enterprise_stakeholders") < dataset_names.index(
        "enterprise_relationships"
    )
    credit_detail_order = [
        name
        for name in (
            "credit_accounts",
            "enterprise_displayed_credit_summary",
            "credit_lines",
        )
        if name in datasets
    ]
    assert [name for name in dataset_names if name in credit_detail_order] == credit_detail_order
    assert dataset_names[-1] == "enterprise_extraction_audit"
    assert {
        f"enterprise_public_{record_type}_records"
        for record_type in (
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
        )
    } <= set(typed_public_datasets)
    assert sum(dataset["row_count"] for dataset in typed_public_datasets.values()) == 20
    assert typed_public_datasets["enterprise_public_license_records"]["label"] == "行政许可记录"
    assert typed_public_datasets["enterprise_public_certification_records"]["label"] == "认证记录"
    assert all(
        dataset["section_id"] == public_section["id"]
        for dataset in typed_public_datasets.values()
    )
    assert {
        dataset["id"] for dataset in typed_public_datasets.values()
    } <= set(public_section["dataset_refs"])
    assert public_section["items"] == []
    assert public_section["groups"] == []
    assert set(public_section["dataset_refs"]) == {
        dataset["id"] for dataset in typed_public_datasets.values()
    }
    assert all(
        "content" not in row["normalized"]
        for dataset in typed_public_datasets.values()
        for row in dataset["rows"]
    )
    assert all(
        set(row["normalized"])
        <= {column["key"] for column in dataset["columns"]}
        for dataset in typed_public_datasets.values()
        for row in dataset["rows"]
    )
    assert facts["credit_summary"]["reported_account_count"] == 28
    assert facts["credit_summary"]["account_population_comparable"] is False
    assert datasets["credit_accounts"]["completeness"]["basis"] == "emitted_records_only"
    assert datasets["credit_accounts"]["completeness"]["expected_row_count"] == 9
    assert "id_number" not in facts
    assert "subject_id" not in facts
    assert datasets["enterprise_profile_fields"]["row_count"] == 9
    assert datasets["enterprise_stakeholders"]["row_count"] == 8
    assert datasets["enterprise_report_identity"]["row_count"] == 1
    report_identity = datasets["enterprise_report_identity"]["rows"][0]["normalized"]
    assert report_identity["enterprise_name"] == "北京报告样本有限责任公司"
    assert report_identity["report_edition"] == "independent_query"
    assert report_identity["report_time"] == "2015-11-01T10:05:15"
    assert report_identity["unified_social_credit_code"] == "91110102183797313J"
    assert report_identity["organization_code"] == "18379731-3"
    assert report_identity["zhongzheng_code"] == "4103090000069511"
    assert report_identity["query_institution"] == "中国某某银行北京分行"
    assert datasets["enterprise_dispute_overview"]["rows"][0]["normalized"][
        "in_progress_dispute_count"
    ] == 3
    assert datasets["enterprise_public_record_counts"]["row_count"] == 5
    recovery_rows = [
        row["normalized"] for row in datasets["enterprise_recovery_summary"]["rows"]
    ]
    assert [
        (row["recovery_type"], row["account_count"], row["balance"])
        for row in recovery_rows
    ] == [
        ("asset_management_disposed_debt", 5, "63"),
        ("advance", 4, "67.6"),
    ]
    overdue_summary = datasets["enterprise_overdue_summary"]["rows"][0]["normalized"]
    assert (
        overdue_summary["overdue_principal"],
        overdue_summary["overdue_interest_and_other"],
        overdue_summary["overdue_total"],
    ) == ("155", "14.86", "169.86")
    assert datasets["enterprise_profile"]["row_count"] == 1
    assert datasets["enterprise_contributors"]["row_count"] == 2
    assert datasets["enterprise_key_personnel"]["row_count"] == 6
    assert datasets["enterprise_credit_accounts"]["row_count"] == 9
    assert datasets["enterprise_credit_facilities"]["row_count"] == 1
    assert datasets["enterprise_repayment_responsibility_accounts"]["row_count"] == 3
    assert [
        row["normalized"] for row in datasets["enterprise_credit_accounts"]["rows"]
    ] == accounts
    assert [
        row["normalized"] for row in datasets["enterprise_credit_facilities"]["rows"]
    ] == credit_lines
    assert [
        row["normalized"]
        for row in datasets["enterprise_repayment_responsibility_accounts"]["rows"]
    ] == [
        row["normalized"] for row in datasets["repayment_liability_records"]["rows"]
    ]
    assert datasets["enterprise_interest_arrears"]["row_count"] == 2
    assert datasets["enterprise_public_utility_payment_records"]["rows"][0]["normalized"][
        "cumulative_arrears"
    ] == "0.3"
    patent = datasets["enterprise_public_patent_records"]["rows"][0]["normalized"]
    assert (
        patent["patent_number"],
        patent["application_date"],
        patent["grant_date"],
        patent["validity_years"],
    ) == ("专20140088", "2014-03-01", "2014-10-01", 10)
    financing_rows = [
        row["normalized"]
        for row in datasets["enterprise_public_financing_restriction_records"]["rows"]
    ]
    assert [
        (row["catalog"], row["control_type"], row["year"], row["scale"])
        for row in financing_rows
    ] == [
        ("土地储备机构名录", "年度可融资规模", 2016, "待定"),
        ("土地储备机构名录", "--", 2015, "1500"),
    ]
    assert datasets["enterprise_public_data_provider_statement_records"]["rows"][0][
        "normalized"
    ]["added_date"] == "2013-10-18"
    assert datasets["enterprise_utility_payment_history"]["row_count"] == 2
    assert datasets["enterprise_housing_fund_history"]["row_count"] == 2
    assert datasets["enterprise_current_credit_summary"]["row_count"] == 11
    assert datasets["enterprise_closed_credit_summary"]["row_count"] == 11
    assert datasets["enterprise_repayment_responsibility_summary"]["row_count"] == 8
    assert datasets["repayment_liability_records"]["row_count"] == 3
    assert datasets["enterprise_attachment_accounts"]["row_count"] == 12
    assert datasets["enterprise_credit_supplement"]["row_count"] == 3
    assert datasets["enterprise_attachment_credit_details"]["row_count"] == 10
    assert datasets["enterprise_special_transactions"]["row_count"] == 2
    assert {
        column["key"]
        for column in semantic_datasets["enterprise_public_license_records"]["columns"]
    } == {
        "sequence",
        "public_record_id",
        "licensing_authority",
        "license_type",
        "license_date",
        "license_expiry_date",
        "license_content",
        "source_page",
        "source_table_id",
    }
    assert not any(
        column["key"].startswith("license_")
        for column in semantic_datasets["enterprise_public_certification_records"]["columns"]
    )
    audit_rows = [row["normalized"] for row in datasets["enterprise_extraction_audit"]["rows"]]
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
    public_heading = f"## {public_section['title']}\n"
    assert public_heading in enhanced
    before_public, public_and_after = enhanced.split(public_heading, maxsplit=1)
    public_preview = public_and_after.split("\n## ", maxsplit=1)[0]
    assert "### 行政许可记录" not in before_public
    assert "### 认证记录" not in before_public
    assert "### 行政许可记录" in public_preview
    assert "### 认证记录" in public_preview
    assert "| 序号 | 记录类型 | 记录内容 |" not in enhanced
    assert "| 序号 | 许可部门 | 许可类型 | 许可日期 | 截止日期 | 许可内容 |" in enhanced
    assert "| 序号 | 认证部门 | 认证类型 | 认证日期 | 截止日期 | 认证内容 |" in enhanced
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
    dataset_names = [dataset["name"] for dataset in payload["datasets"]]
    document_order = semantic["domain"]["extensions"].get("dataset_document_order")
    if document_order:
        assert dataset_names == [name for name in document_order if name in dataset_names]
    if subtype == "enterprise":
        enterprise_datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
        audit_dataset = enterprise_datasets["enterprise_extraction_audit"]
        audit_rows = [row["normalized"] for row in audit_dataset["rows"]]
        audit_cells = list(csv.DictReader(io.StringIO(bundle.render_audit_csv(semantic).lstrip("\ufeff"))))
        enterprise_audit_cells = [row for row in audit_cells if not row["dataset_id"].startswith("_audit")]
        assert enterprise_audit_cells
        assert all(row["evidence_ref"] for row in enterprise_audit_cells)
        assert all(json.loads(row["evidence_ref"]) for row in enterprise_audit_cells)
        assert all(row["expected_record_count"] == row["extracted_record_count"] for row in audit_rows)
        assert all(
            row["reconciliation_status"] == "complete" and row["unresolved_record_count"] == 0 for row in audit_rows
        )
        displayed_dataset = enterprise_datasets.get("enterprise_displayed_credit_summary")
        if displayed_dataset is not None:
            displayed_rows = [row["normalized"] for row in displayed_dataset["rows"]]
            assert displayed_rows
            assert all(row["summary_scope"] == "displayed_detail_section" for row in displayed_rows)
            assert all(row["source_group_account_count"] > 0 for row in displayed_rows)
            assert all(row["source_account_count"] > 0 for row in displayed_rows)
    else:
        assert "enterprise_displayed_credit_summary" not in {
            dataset["name"] for dataset in payload["datasets"]
        }
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
                "| 组内序号 | 责任发生日期 | 相关方名称 | 管理机构 | 业务类型 | 责任金额 | 余额 | 币种 |"
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
                if row["institution"] == "华夏银行股份有限公司信用卡中心" and row["open_date"] == "2022-07-19"
            )
            assert huaxia_account["business_type"] == "贷记卡"
            assert huaxia_overdue["business_type"] == "贷记卡"
            assert len(normalized_overdue) == 9
            assert [row["over_90_days_months"] for row in normalized_overdue] == [1, 3, 2, 2, 2, 1, 2, 1, 0]
            assert all(row["current_overdue_status"] == "overdue" for row in normalized_overdue)
            assert sum(row["over_90_days"] is True for row in normalized_overdue) == 8
            for overdue_type, heading in {
                "credit_card": "信用卡账户",
                "loan": "贷款账户",
                "credit_line": "贷款授信",
            }.items():
                if not any(row["account_type"] == overdue_type for row in normalized_overdue):
                    continue
                account_preview = enhanced_preview.split(f"#### {heading}", maxsplit=1)[1].split(
                    "\n#### ",
                    maxsplit=1,
                )[0]
                assert "##### 逾期记录" in account_preview
            assert (
                "| 组内序号 | 账户类型 | 管理机构 | 业务类型 | 卡片尾号 | 开立日期 | "
                "最近5年逾期月数 | 其中超过90天月数 | 当前逾期状态 |"
            ) in enhanced_preview
            assert "\n### 逾期记录\n" not in enhanced_preview
            assert "account id" not in enhanced_preview
            assert "overdue id" not in enhanced_preview
            assert "last_5_years" not in enhanced_preview
            assert "当前有逾期" in enhanced_preview
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
