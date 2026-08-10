# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Personal-brief schema boundary.

The current personal schema remains behavior-compatible, but enterprise code
no longer derives from it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from docmirror.plugins.credit_report.personal_brief_native.contracts import (
    PERSONAL_BRIEF_AMOUNT_UNIT_LABELS,
    PERSONAL_BRIEF_ENUM_CONTRACT,
    PERSONAL_BRIEF_MONEY_FIELDS,
    PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
)


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

    for field_name in ("amount_unit", "reporting_amount_unit"):
        enums[field_name] = dict(PERSONAL_BRIEF_AMOUNT_UNIT_LABELS)

    dictionary["datasets"]["personal_report_metadata"]["columns"][
        "marital_status"
    ] = deepcopy(dictionary["fields"]["marital_status"])
    dictionary["fields"]["marital_status_raw"] = {
        "label": "婚姻状况原文",
        "type": "string",
        "definition": "源报告中婚姻状况的原始标签，用于保留未来或未枚举值。",
    }
    dictionary["datasets"]["personal_report_metadata"]["columns"][
        "marital_status_raw"
    ] = deepcopy(dictionary["fields"]["marital_status_raw"])
    for field_name, descriptor in {
        "source_section": {
            "label": "规范来源章节",
            "type": "string",
            "definition": "账户在本人简版规范布局中的唯一来源章节。",
        },
        "source_sequence": {
            "label": "源报告组内序号",
            "type": "integer",
            "definition": "源报告对应章节中打印的账户序号。",
        },
        "business_category": {
            "label": "业务大类",
            "type": "enum",
            "definition": "信用卡、贷款或其他业务。",
        },
        "institution_statement_id": {"label": "机构说明记录ID", "type": "string"},
        "statement_content": {"label": "说明内容", "type": "string"},
        "added_date": {"label": "添加日期", "type": "date"},
    }.items():
        dictionary["fields"][field_name] = descriptor
    dictionary["datasets"]["credit_accounts"]["columns"].update(
        {
            field_name: deepcopy(dictionary["fields"][field_name])
            for field_name in ("source_section", "source_sequence", "business_category")
        }
    )
    dictionary["datasets"]["credit_accounts"]["columns"]["close_date"][
        "label"
    ] = "结清/销户日期"
    dictionary["datasets"]["institution_statement_records"] = {
        "definition": "一行对应机构针对整份信用报告作出的一项说明。",
        "columns": {
            "institution_statement_id": deepcopy(dictionary["fields"]["institution_statement_id"]),
            "sequence": {"label": "序号", "type": "integer"},
            "statement_content": deepcopy(dictionary["fields"]["statement_content"]),
            "added_date": deepcopy(dictionary["fields"]["added_date"]),
        },
    }
    enums["marital_status"] = {
        "unmarried": "未婚",
        "married": "已婚",
        "divorced": "离婚",
        "widowed": "丧偶",
        "other": "其他",
        "not_reported": "未说明",
    }

    enums["is_primary"] = {"true": "是", "false": "否"}
    for field_name in ("ever_overdue", "over_90_days", "current_overdue"):
        enums[field_name] = {"true": "是", "false": "否"}
    enums["settlement_state"] = {
        "settled": "已结清",
        "not_reported": "未报告",
    }
    enums["summary_scope"] = {
        "source_reported": "源报告",
    }
    enums["metric"] = {
        "account_count": "账户数",
        "unclosed_account_count": "未结清/未销户账户数",
        "ever_overdue_account_count": "发生过逾期的账户数",
        "over_90_days_account_count": "发生过90天以上逾期的账户数",
        "asset_disposition_count": "资产处置信息账户数",
        "guarantor_compensation_count": "垫款信息账户数",
        "personal_repayment_liability_count": "为个人承担相关还款责任的账户数",
        "enterprise_repayment_liability_count": "为企业承担相关还款责任的账户数",
    }
    enums["business_category"] = {
        "credit_card": "信用卡",
        "housing_loan": "购房贷款",
        "other_loan": "其他贷款",
        "other_business": "其他业务",
        "credit_cards": "信用卡明细",
        "loans": "贷款明细",
        "all": "全部",
    }
    enums.setdefault("account_type", {})["other_business"] = "其他业务"

    # Dataset-qualified enum domains are the public contract.  Keep the
    # top-level union for legacy display lookup, while publishing the precise
    # domain on each dataset column for downstream consumers.
    for (dataset_name, field_name), labels in PERSONAL_BRIEF_ENUM_CONTRACT.items():
        descriptor = dictionary["datasets"][dataset_name]["columns"][field_name]
        descriptor["enum"] = deepcopy(labels)
        enums.setdefault(field_name, {}).update(deepcopy(labels))

    for dataset_name, money_fields in PERSONAL_BRIEF_MONEY_FIELDS.items():
        columns = dictionary["datasets"][dataset_name]["columns"]
        columns["reporting_amount_unit"]["enum"] = deepcopy(
            PERSONAL_BRIEF_AMOUNT_UNIT_LABELS
        )
        for field_name in money_fields:
            columns[field_name]["unit"] = PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT
    dictionary["datasets"]["personal_report_metadata"]["columns"][
        "reporting_amount_unit"
    ]["enum"] = deepcopy(PERSONAL_BRIEF_AMOUNT_UNIT_LABELS)
    return dictionary


def personal_brief_semantic_extensions() -> dict[str, Any]:
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_semantic_extensions,
    )

    semantic = credit_report_semantic_extensions(report_subtype="personal_brief")
    semantic["dataset_document_order"] = [
        "personal_report_metadata",
        "identity_documents",
        "personal_credit_summary_records",
        "asset_disposition_records",
        "guarantor_compensation_records",
        "credit_accounts",
        "overdue_records",
        "repayment_liability_records",
        "repayment_records",
        "postpaid_records",
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
        "public_records",
        "institution_statement_records",
        "inquiry_records",
        "report_notes",
    ]
    semantic["dataset_reading_columns"] = {
        "identity_documents": [
            "sequence",
            "holder_name",
            "document_type",
            "document_number",
            "is_primary",
        ],
        "personal_credit_summary_records": [
            "metric",
            "business_category",
            "value",
            "reporting_status",
        ],
        "institution_statement_records": [
            "sequence",
            "statement_content",
            "added_date",
        ],
    }

    presentation = semantic.setdefault("enhanced_markdown", {})
    presentation.pop("appendix", None)
    presentation["suppress_empty_sections"] = True
    presentation["section_layouts"] = {
        "report_header": {
            "omit_unlisted": True,
            "groups": [
                {
                    "title": "报告信息",
                    "fields": ["report_number", "report_time"],
                },
                {
                    "title": "个人信息",
                    "fields": ["subject_name", "marital_status"],
                },
            ],
        },
        "credit_details": {"omit_unlisted": True},
        "non_credit_transactions": {
            "omit_unlisted": True,
            "groups": [
                {
                    "hide_title": True,
                    "fields": ["record_status", "lookback_years", "source_statement"],
                }
            ],
        },
        "public_records": {
            "omit_unlisted": True,
            "groups": [
                {
                    "hide_title": True,
                    "fields": ["record_status", "lookback_years", "source_statement"],
                }
            ],
        },
        "institution_statements": {"omit_unlisted": True},
        "inquiries": {"omit_unlisted": True},
        "notes": {"omit_unlisted": True},
    }
    dataset_layouts = presentation.setdefault("dataset_layouts", {})
    for dataset_name in (
        "personal_report_metadata",
        "repayment_records",
        "overdue_records",
        "public_records",
        "report_notes",
    ):
        dataset_layouts[dataset_name] = {"hidden": True}
    dataset_layouts["credit_accounts"] = {
        "mode": "partitioned_tables",
        "partition_by": "source_section",
        "partitions": [
            {
                "value": "credit_cards",
                "title": "信用卡",
                "prepend_partition": {
                    "dataset": "overdue_records",
                    "join_on": "account_id",
                    "title": "信用卡逾期记录",
                    "columns": [
                        "sequence",
                        "institution",
                        "business_type",
                        "card_tail",
                        "open_date",
                        "overdue_months",
                        "over_90_days_months",
                        "current_overdue_status",
                    ],
                },
                "columns": [
                    "sequence",
                    "institution",
                    "business_type",
                    "card_tail",
                    "open_date",
                    "snapshot_date",
                    "close_date",
                    "account_currency",
                    "credit_limit",
                    "used_amount",
                    "balance",
                    "account_lifecycle_state",
                    "card_activation_state",
                ],
            },
            {
                "value": "loans",
                "title": "贷款",
                "prepend_partition": {
                    "dataset": "overdue_records",
                    "join_on": "account_id",
                    "title": "贷款逾期记录",
                    "columns": [
                        "sequence",
                        "institution",
                        "business_type",
                        "open_date",
                        "overdue_months",
                        "over_90_days_months",
                        "current_overdue_status",
                    ],
                },
                "columns": [
                    "sequence",
                    "institution",
                    "business_type",
                    "open_date",
                    "snapshot_date",
                    "contract_maturity_date",
                    "credit_line_expiry_date",
                    "close_date",
                    "account_currency",
                    "loan_amount",
                    "credit_limit",
                    "balance",
                    "account_lifecycle_state",
                    "payoff_state",
                ],
            },
            {
                "value": "other_business",
                "title": "其他业务",
                "prepend_partition": {
                    "dataset": "overdue_records",
                    "join_on": "account_id",
                    "title": "其他业务逾期记录",
                    "columns": [
                        "sequence",
                        "institution",
                        "business_type",
                        "open_date",
                        "overdue_months",
                        "over_90_days_months",
                        "current_overdue_status",
                    ],
                },
                "columns": [
                    "sequence",
                    "institution",
                    "business_type",
                    "open_date",
                    "snapshot_date",
                    "close_date",
                    "account_currency",
                    "loan_amount",
                    "credit_limit",
                    "balance",
                    "account_lifecycle_state",
                ],
            },
        ],
    }

    overrides = semantic.setdefault("community_projection_overrides", {})
    overrides["section_markers"] = {
        "personal_report_metadata": ["report_header"],
        "identity_documents": ["report_header"],
        "personal_credit_summary_records": ["credit_details"],
        "asset_disposition_records": ["credit_details"],
        "guarantor_compensation_records": ["credit_details"],
        "credit_accounts": ["credit_details"],
        "repayment_liability_records": ["credit_details"],
        "repayment_records": ["credit_details"],
        "overdue_records": ["credit_details"],
        "postpaid_records": ["non_credit_transactions"],
        "tax_arrears_records": ["public_records"],
        "civil_judgment_records": ["public_records"],
        "enforcement_records": ["public_records"],
        "administrative_penalty_records": ["public_records"],
        "public_records": ["public_records"],
        "institution_statement_records": ["institution_statements"],
        "inquiry_records": ["inquiries"],
        "report_notes": ["notes"],
    }
    overrides["dataset_labels"] = {
        "personal_credit_summary_records": "信息概要",
    }
    return semantic


__all__ = [
    "personal_brief_data_dictionary",
    "personal_brief_semantic_extensions",
]
