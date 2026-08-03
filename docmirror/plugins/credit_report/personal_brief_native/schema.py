# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Personal-brief schema boundary.

The current personal schema remains behavior-compatible, but enterprise code
no longer derives from it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def personal_brief_data_dictionary() -> dict[str, Any]:
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_data_dictionary,
    )

    dictionary = deepcopy(credit_report_data_dictionary())
    dictionary["schema_id"] = "personal_brief_credit_report"
    enums = dictionary.setdefault("enums", {})

    currency_labels = {
        "CNY": "人民币",
        "USD": "美元",
        "EUR": "欧元",
        "HKD": "港币",
    }
    for field_name in (
        "currency",
        "account_currency",
        "reporting_currency",
        "reporting_amount_currency",
    ):
        enums[field_name] = dict(currency_labels)

    amount_unit_labels = {
        "yuan": "元",
        "CNY_1": "元",
        "CNY_10K": "万元（人民币）",
    }
    for field_name in ("amount_unit", "reporting_amount_unit"):
        enums[field_name] = dict(amount_unit_labels)

    dictionary["datasets"]["personal_report_metadata"]["columns"][
        "marital_status"
    ] = deepcopy(dictionary["fields"]["marital_status"])

    enums["is_primary"] = {"true": "是", "false": "否"}
    enums["summary_scope"] = {
        "source_reported": "源报告",
    }
    enums["metric"] = {
        "account_count": "账户数",
        "unclosed_account_count": "未结清/未销户账户数",
        "ever_overdue_account_count": "发生过逾期的账户数",
        "over_90_days_account_count": "发生过90天以上逾期的账户数",
        "personal_repayment_liability_count": "为个人承担相关还款责任的账户数",
        "enterprise_repayment_liability_count": "为企业承担相关还款责任的账户数",
    }
    enums["business_category"] = {
        "credit_card": "信用卡",
        "housing_loan": "购房贷款",
        "other_loan": "其他贷款",
        "other_business": "其他业务",
        "all": "全部",
    }
    return dictionary


def personal_brief_semantic_extensions() -> dict[str, Any]:
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_semantic_extensions,
    )

    semantic = credit_report_semantic_extensions(report_subtype="personal_brief")
    semantic["dataset_document_order"] = [
        "personal_report_metadata",
        "report_notes",
        "identity_documents",
        "personal_credit_summary_records",
        "asset_disposition_records",
        "guarantor_compensation_records",
        "credit_accounts",
        "repayment_liability_records",
        "repayment_records",
        "overdue_records",
        "postpaid_records",
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
        "public_records",
        "inquiry_records",
    ]
    return semantic


__all__ = [
    "personal_brief_data_dictionary",
    "personal_brief_semantic_extensions",
]
