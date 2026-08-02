# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.server.output_builder import build_community_bundle

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]
_FIXTURE = Path("tests/fixtures-private/credit_report/个人信用报告（本人版）展示样本.pdf")


def test_personal_detail_sample_uses_canonical_typed_datasets() -> None:
    if not _FIXTURE.exists():
        pytest.skip("personal detailed display sample is unavailable")

    sealed = asyncio.run(
        perceive_document(
            _FIXTURE,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(_FIXTURE))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}

    assert payload["document"]["type"] == "personal_credit_report_detailed"
    assert semantic["domain"]["data_dictionary"]["schema_id"] == "personal_credit_report_detailed"
    assert semantic["domain"]["data_dictionary"]["version"] == "1.2.0"
    assert validate_projection_payload("community", payload).valid
    assert validate_projection_payload("personal_credit_report_detailed", payload).valid

    malformed = deepcopy(payload)
    malformed_profile = next(
        dataset for dataset in malformed["datasets"] if dataset["name"] == "personal_profile"
    )
    malformed_profile["rows"][0]["normalized"]["birth_date"] = {"ocr": "1981.08.15"}
    assert not validate_projection_payload("personal_credit_report_detailed", malformed).valid

    malformed = deepcopy(payload)
    malformed_observations = next(
        dataset
        for dataset in malformed["datasets"]
        if dataset["name"] == "personal_detail_field_observations"
    )
    malformed_observations["rows"][0]["normalized"]["observation_status"] = "maybe"
    assert not validate_projection_payload("personal_credit_report_detailed", malformed).valid

    malformed = deepcopy(payload)
    malformed["datasets"].append(deepcopy(malformed["datasets"][0]))
    assert not validate_projection_payload("personal_credit_report_detailed", malformed).valid

    expected_counts = {
        "personal_report_metadata": 1,
        "identity_documents": 2,
        "mobile_phone_records": 5,
        "spouse_records": 1,
        "residence_records": 5,
        "employment_records": 5,
        "recovery_records": 2,
        "credit_accounts": 15,
        "credit_lines": 5,
        "repayment_liability_records": 4,
        "repayment_records": 449,
        "overdue_records": 72,
        "postpaid_records": 3,
        "postpaid_payment_history": 48,
        "personal_detail_account_events": 8,
        "personal_detail_summary_records": 13,
        "personal_detail_summary_cells": 127,
        "public_records": 10,
        "inquiry_records": 12,
        "statements": 2,
        "annotations": 3,
    }
    assert {name: datasets[name]["row_count"] for name in expected_counts} == expected_counts
    assert all(datasets[name]["completeness"]["verified"] for name in expected_counts)
    assert {name: datasets[name]["completeness"]["basis"] for name in expected_counts} == {
        "personal_report_metadata": "personal_detail_source_header_count",
        "identity_documents": "personal_detail_source_header_count",
        "mobile_phone_records": "personal_detail_source_sequence_count",
        "spouse_records": "personal_detail_source_structure_count",
        "residence_records": "personal_detail_source_sequence_count",
        "employment_records": "personal_detail_source_sequence_count",
        "recovery_records": "personal_detail_source_sequence_count",
        "credit_accounts": "source_report_summary",
        "credit_lines": "personal_detail_source_structure_count",
        "repayment_liability_records": "personal_detail_source_sequence_count",
        "repayment_records": "personal_detail_repayment_grid_cell_count",
        "overdue_records": "personal_detail_derived_overdue_count",
        "postpaid_records": "personal_detail_source_sequence_count",
        "postpaid_payment_history": "personal_detail_repayment_grid_cell_count",
        "personal_detail_account_events": "personal_detail_source_event_count",
        "personal_detail_summary_records": "personal_detail_source_table_count",
        "personal_detail_summary_cells": "personal_detail_source_cell_count",
        "public_records": "personal_detail_source_sequence_count",
        "inquiry_records": "source_sequence_ledger",
        "statements": "personal_detail_source_marker_count",
        "annotations": "personal_detail_source_marker_count",
    }

    for dataset in datasets.values():
        assert dataset["storage_role"] == "canonical"
        assert dataset["record_path"] == "rows"
        assert dataset["primary_key"] == "record_id"
        assert dataset["row_count"] == len(dataset["rows"])
        record_ids = [row["record_id"] for row in dataset["rows"]]
        assert len(record_ids) == len(set(record_ids))
        column_keys = {column["key"] for column in dataset["columns"]}
        assert set(dataset.get("reading_columns") or ()) <= column_keys
        assert all(re.search(r"[\u4e00-\u9fff]", column["label"]) for column in dataset["columns"])

    assert {name: datasets[name]["label"] for name in datasets} == {
        "personal_report_metadata": "个人信用报告信息",
        "personal_profile": "个人基本资料",
        "personal_detail_field_observations": "字段观测与不确定性",
        "identity_documents": "身份证件",
        "mobile_phone_records": "手机号码历史",
        "spouse_records": "配偶信息",
        "residence_records": "居住信息",
        "employment_records": "职业信息",
        "recovery_records": "被追偿信息",
        "credit_accounts": "信贷交易账户明细",
        "credit_lines": "授信协议信息",
        "repayment_liability_records": "相关还款责任信息",
        "repayment_records": "月度还款记录",
        "overdue_records": "逾期明细（派生）",
        "postpaid_records": "后付费记录",
        "postpaid_payment_history": "后付费月度缴费记录",
        "personal_detail_account_events": "账户补充事件",
        "personal_detail_summary_records": "信息概要表",
        "personal_detail_summary_cells": "信息概要单元格",
        "personal_detail_credit_summary_metrics": "信用概要业务指标",
        "public_records": "公共信息明细",
        "tax_arrears_records": "欠税记录",
        "civil_judgment_records": "民事判决记录",
        "enforcement_records": "强制执行记录",
        "administrative_penalty_records": "行政处罚记录",
        "personal_housing_fund_records": "住房公积金参缴记录",
        "professional_qualification_records": "执业资格记录",
        "award_records": "行政奖励记录",
        "inquiry_records": "查询记录",
        "statements": "机构说明与本人声明",
        "annotations": "异议标注",
        "personal_detail_dataset_status": "业务数据集存在状态",
    }

    sections = {section["id"]: section for section in payload["sections"]}
    assert {
        section_id: (section["title"], section["type"], section["page_range"])
        for section_id, section in sections.items()
    } == {
        "sec_personal_basic": ("个人基本信息", "basic_information", [1, 2]),
        "sec_credit_summary": ("信息概要", "credit_summary", [2, 4]),
        "sec_credit_details": ("信贷交易信息明细", "credit_details", [4, 12]),
        "sec_non_credit_transactions": ("非信贷交易信息明细", "non_credit_transactions", [12, 12]),
        "sec_public_records": ("公共信息明细", "public_records", [12, 13]),
        "sec_statements": ("机构说明与本人声明", "statements", [6, 6]),
        "sec_annotations": ("异议标注", "annotations", [2, 13]),
        "sec_inquiries": ("查询记录", "inquiries", [13, 13]),
        "sec_report_explanation": ("报告说明与编制说明", "report_explanation", [14, 15]),
    }
    assert sections["sec_credit_summary"]["dataset_refs"] == [
        "ds_personal_detail_summary_records",
        "ds_personal_detail_summary_cells",
        "ds_personal_detail_credit_summary_metrics",
    ]
    assert "ds_recovery_records" in sections["sec_credit_details"]["dataset_refs"]
    assert "ds_personal_detail_account_events" in sections["sec_credit_details"]["dataset_refs"]
    assert "ds_postpaid_payment_history" in sections["sec_non_credit_transactions"]["dataset_refs"]
    assert sections["sec_statements"]["dataset_refs"] == ["ds_statements"]
    assert sections["sec_annotations"]["dataset_refs"] == ["ds_annotations"]
    assert not any(
        str(item.get("key") or "").startswith("personal_detail_expected_")
        for section in payload["sections"]
        for item in section.get("items") or []
    )

    metadata = datasets["personal_report_metadata"]["rows"][0]["normalized"]
    assert metadata["subject_name"] == "信小达"
    assert metadata["primary_id_type"] == "身份证"
    assert metadata["primary_id_number"] == "622926198108151111"
    assert metadata["report_time"] == "2025-05-25T10:05:15+08:00"

    accounts = [row["normalized"] for row in datasets["credit_accounts"]["rows"]]
    assert Counter(account["account_type"] for account in accounts) == {
        "non_revolving_loan": 6,
        "revolving_loan_subaccount": 2,
        "revolving_loan_account": 2,
        "credit_card": 4,
        "quasi_credit_card": 1,
    }
    assert all(account.get("account_identifier") for account in accounts)
    accounts_by_id = {account["account_id"]: account for account in accounts}

    first_loan = accounts_by_id["credit_account:non_revolving_loan:1"]
    assert first_loan["institution"] == "样例住房公积金管理中心"
    assert first_loan["overdue_principal_91_180"] == "0"

    revolving_subaccount = accounts_by_id["credit_account:revolving_loan_subaccount:1"]
    assert revolving_subaccount["credit_agreement_identifier"] == "X10114560H0001BOC22223"
    assert revolving_subaccount["overdue_principal_91_180"] == "0"
    assert accounts_by_id["credit_account:revolving_loan_subaccount:2"].get("credit_agreement_identifier") is None

    revolving_account = accounts_by_id["credit_account:revolving_loan_account:1"]
    assert revolving_account["credit_agreement_identifier"] == "B11011122G0001BOC0255220"

    billed_card = accounts_by_id["credit_account:credit_card:2"]
    assert billed_card.get("scheduled_payment_date") is None
    assert billed_card["billing_date"] == "2025-05-12"
    assert billed_card["recent_6_month_average_used_amount"] == "5000"
    assert billed_card["maximum_used_amount"] == "10000"

    quasi_card = accounts_by_id["credit_account:quasi_credit_card:1"]
    assert quasi_card.get("scheduled_payment_date") is None
    assert quasi_card["billing_date"] == "2025-05-15"
    assert quasi_card["recent_6_month_average_overdraft_balance"] == "167"
    assert quasi_card["maximum_overdraft_balance"] == "2000"
    assert quasi_card.get("due_date") is None
    assert quasi_card.get("contract_maturity_date") is None

    card_tails = {
        account["account_id"]: account.get("card_tail")
        for account in accounts
        if account["account_type"] in {"credit_card", "quasi_credit_card"}
    }
    assert card_tails == {
        "credit_account:credit_card:1": "1365",
        "credit_account:credit_card:2": "1635",
        "credit_account:credit_card:3": "8383",
        "credit_account:credit_card:4": "5700",
        "credit_account:quasi_credit_card:1": "1465",
    }

    continued_account = next(
        account for account in accounts if account["account_id"] == "credit_account:revolving_loan_account:1"
    )
    assert {
        "status": continued_account["status"],
        "five_tier_class": continued_account["five_tier_class"],
        "balance": continued_account["balance"],
        "remaining_periods": continued_account["remaining_periods"],
        "scheduled_payment": continued_account["scheduled_payment"],
        "scheduled_payment_date": continued_account["scheduled_payment_date"],
        "actual_payment": continued_account["actual_payment"],
        "last_repayment_date": continued_account["last_repayment_date"],
    } == {
        "status": "active",
        "five_tier_class": "正常",
        "balance": "5000",
        "remaining_periods": 0,
        "scheduled_payment": "5000",
        "scheduled_payment_date": "2025-05-05",
        "actual_payment": "5000",
        "last_repayment_date": "2025-05-05",
    }

    account_ids = {account["account_id"] for account in accounts}
    repayment_rows = [row["normalized"] for row in datasets["repayment_records"]["rows"]]
    assert all(row["account_id"] in account_ids for row in repayment_rows)
    assert all(row["account_identifier"] for row in repayment_rows)
    assert all(1 <= row["month"] <= 12 for row in repayment_rows)
    overdue_rows = [row["normalized"] for row in datasets["overdue_records"]["rows"]]
    assert not any(row["account_id"] == quasi_card["account_id"] for row in overdue_rows)

    def repayment(account_id: str, year: int) -> list[tuple[int, str, str | None]]:
        return [
            (row["month"], row["status"], row.get("overdue_amount"))
            for row in repayment_rows
            if row["account_id"] == account_id and row["year"] == year
        ]

    assert repayment("credit_account:non_revolving_loan:4", 2020) == [(month, "N", "0") for month in range(6, 13)]
    assert repayment("credit_account:non_revolving_loan:4", 2023) == [
        *[(month, "N", "0") for month in range(1, 11)],
        (11, "#", None),
    ]
    card_2024 = repayment("credit_account:credit_card:1", 2024)
    assert len(card_2024) == 12
    assert card_2024[1] == (2, "7", "15727")
    assert repayment("credit_account:credit_card:1", 2022)[-2:] == [
        (11, "1", "10618"),
        (12, "2", "10840"),
    ]
    assert repayment("credit_account:credit_card:1", 2020) == [(month, "N", "0") for month in range(6, 13)]

    residences = [row["normalized"] for row in datasets["residence_records"]["rows"]]
    assert residences[0] == {
        "address": "某市某区某小区7 号楼C522 室",
        "data_provider": "样例银行1",
        "information_updated_date": "2025-05-01",
        "page": 1,
        "residence_record_id": residences[0]["residence_record_id"],
        "residence_status": "按揭",
        "residential_phone": "010—83234323",
        "sequence": 1,
        "source_page": 1,
    }
    assert "values" not in residences[0]

    employment = [row["normalized"] for row in datasets["employment_records"]["rows"]]
    fifth_employment = employment[4]
    assert fifth_employment == {
        "data_provider": "样例小额贷款公司",
        "employer": "某软件中心",
        "employer_address": "某市经开区北辰东路2 号",
        "employer_phone": "010—57888888",
        "employer_type": "外资企业",
        "employment_record_id": fifth_employment["employment_record_id"],
        "entry_year": 2022,
        "industry": "科学研究和技术服务业",
        "information_updated_date": "2022-07-12",
        "occupation": "专业技术人员",
        "page": 2,
        "position": "一般员工",
        "professional_title": "初级",
        "sequence": 5,
        "source_page": 2,
    }

    summaries = [row["normalized"] for row in datasets["personal_detail_summary_records"]["rows"]]
    assert all("columns" not in row and "rows" not in row for row in summaries)
    overdue_summary = next(row for row in summaries if row["title"] == "逾期（透支）信息汇总")
    assert overdue_summary["source_row_count"] == 5
    summary_cells = [row["normalized"] for row in datasets["personal_detail_summary_cells"]["rows"]]
    assert all(not isinstance(value, (dict, list)) for row in summary_cells for value in row.values())
    assert all(row.get("column_label") for row in summary_cells)
    assert not any(row["value"] in {"账户类型", "账户数", "月份数"} for row in summary_cells)
    assert any(row["value"] == "23,505" and row["title"] == "呆账信息汇总" for row in summary_cells)
    summary_metrics = [
        row["normalized"] for row in datasets["personal_detail_credit_summary_metrics"]["rows"]
    ]
    assert all(row["mapping_status"] == "mapped" for row in summary_metrics)
    assert all(row.get("text_value") != "--" for row in summary_metrics)
    assert all(
        row.get("currency") == "CNY" and row.get("amount_unit") == "yuan"
        for row in summary_metrics
        if row["value_type"] == "money"
    )
    quasi_summary_cells = [
        row
        for row in summary_cells
        if row["title"] == "逾期（透支）信息汇总" and row["row_index"] == 5
    ]
    assert [(row["column_label"], row["value"]) for row in quasi_summary_cells] == [
        ("账户类型", "准贷记卡账户"),
        ("账户数", "--"),
        ("月份数", "--"),
        ("单月最高逾期/透支总额", "--"),
        ("最长逾期/透支月数", "--"),
    ]

    liabilities = [row["normalized"] for row in datasets["repayment_liability_records"]["rows"]]
    assert [row["overdue_months_or_repayment_status"] for row in liabilities] == ["N", "N", "0", "2"]

    public_records = [row["normalized"] for row in datasets["public_records"]["rows"]]
    assert Counter(row["record_type"] for row in public_records) == {
        "tax_arrears": 1,
        "civil_judgment": 2,
        "enforcement": 2,
        "administrative_penalty": 1,
        "housing_fund": 1,
        "professional_qualification": 2,
        "award": 1,
    }
    housing_fund = next(row for row in public_records if row["record_type"] == "housing_fund")
    assert json.loads(housing_fund["content"])["monthly_contribution"] == 3000
    typed_public_counts = {
        "tax_arrears_records": 1,
        "civil_judgment_records": 2,
        "enforcement_records": 2,
        "administrative_penalty_records": 1,
        "personal_housing_fund_records": 1,
        "professional_qualification_records": 2,
        "award_records": 1,
    }
    assert {name: datasets[name]["row_count"] for name in typed_public_counts} == typed_public_counts
    for name in typed_public_counts:
        column_keys = {column["key"] for column in datasets[name]["columns"]}
        assert all(set(row["normalized"]) <= column_keys for row in datasets[name]["rows"])
    typed_housing_fund = datasets["personal_housing_fund_records"]["rows"][0]["normalized"]
    assert typed_housing_fund["public_record_id"] == housing_fund["public_record_id"]
    assert typed_housing_fund["monthly_contribution"] == "3000"
    assert typed_housing_fund["reporting_amount_currency"] == "CNY"
    assert typed_housing_fund["reporting_amount_unit"] == "yuan"

    statuses = {
        row["normalized"]["dataset_name"]: row["normalized"]
        for row in datasets["personal_detail_dataset_status"]["rows"]
    }
    assert statuses["credit_accounts"]["observed_row_count"] == 15
    assert statuses["credit_accounts"]["presence_status"] == "observed_nonempty"
    assert statuses["inquiry_records"]["observed_row_count"] == 12
    assert statuses["inquiry_records"]["presence_status"] == "observed_nonempty"

    annotations = [row["normalized"]["text"] for row in datasets["annotations"]["rows"]]
    assert annotations == [
        "职业信息正处在异议处理期。",
        "该笔业务正处于异议处理中。",
        "信用报告存在信息缺失。",
    ]
    statements = [row["normalized"]["text"] for row in datasets["statements"]["rows"]]
    assert statements == [
        "该笔业务后续由样例住房公积金管理中心自行管理。",
        "本人于2021 年2 月发生的逾期是因出差没有及时还款造成。",
    ]

    account_events = [row["normalized"] for row in datasets["personal_detail_account_events"]["rows"]]
    assert next(row for row in account_events if row.get("transaction_type") == "信用卡个性化分期")["amount"] == "25484"
    assert (
        next(row for row in account_events if row["event_type"] == "large_installment")["installment_limit"] == "120000"
    )

    facts = semantic["domain"]["facts"]
    assert facts["id_type"] == "身份证"
    assert "unified_social_credit_code" not in facts
    assert facts["gender"] == "男"
    assert facts["birth_date"] == "1981-08-15"

    field_items = {
        item["key"]: item for section in payload["sections"] for item in section.get("items", []) if item.get("key")
    }
    assert field_items["id_type"]["value"] == "身份证"
    assert "unified_social_credit_code" not in field_items
    assert field_items["gender"]["value"] == "男"
