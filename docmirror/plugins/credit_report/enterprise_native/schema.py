# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enterprise-only public schema and presentation-policy foundations."""

from __future__ import annotations

from typing import Any


def enterprise_credit_report_data_dictionary() -> dict[str, Any]:
    """Return an enterprise-owned dictionary with no personal-report fields."""
    return {
        "schema_id": "enterprise_credit_report",
        "version": "2.0.0",
        "definitions": {
            "authoritative_business_records": "datasets",
            "amount_storage": (
                "Decimal business amounts are stored as strings in Community JSON; "
                "currency and amount_unit are explicit on every monetary record."
            ),
            "missing_values": (
                "JSON null means no normalized value. Reporting-status fields distinguish "
                "not_reported, not_applicable, derived, and unresolved source values."
            ),
            "date_semantics": (
                "maturity_date is contractual 到期日; expiry_date is a validity/许可截止日期; "
                "close_date is settlement or account closure; snapshot_date is 信息报告日期."
            ),
        },
        "fields": {
            "note_id": {"label": "说明记录ID", "type": "string"},
            "content": {"label": "说明内容", "type": "string"},
            "sequence": {"label": "序号", "type": "integer"},
            "source_page": {"label": "源页码", "type": "integer"},
            "source_table_id": {"label": "源表标识", "type": "string"},
        },
        "datasets": {
            "report_notes": {
                "definition": "报告说明中的一个编号条目一行。",
                "columns": {
                    "note_id": {"label": "说明记录ID", "type": "string"},
                    "sequence": {"label": "序号", "type": "integer"},
                    "content": {"label": "说明内容", "type": "string"},
                    "source_page": {"label": "源页码", "type": "integer"},
                },
            },
            "credit_accounts": {
                "definition": "企业报告正文中的一个可识别信贷账户卡片一行。",
                "columns": {},
            },
            "credit_lines": {
                "definition": "企业报告中的一份授信协议明细一行。",
                "columns": {},
            },
            "repayment_liability_records": {
                "definition": "相关还款责任明细中的一个可识别账户一行。",
                "columns": {},
            },
            "repayment_records": {
                "definition": "企业账户历史中的一个还款月份一行。",
                "columns": {},
            },
            "overdue_records": {
                "definition": (
                    "仅存放源报告明确报告的逾期事实；不得由五级分类关注、次级、可疑或损失推断。"
                ),
                "columns": {},
            },
            "inquiry_records": {
                "definition": "企业报告明确列示的一次查询记录一行。",
                "columns": {},
            },
        },
        "enums": {
            "reporting_status": {
                "reported": "已报告",
                "not_reported": "未报告",
                "not_applicable": "不适用",
                "derived": "派生值",
                "unresolved": "未解析",
            },
            "document_type": {
                "enterprise_credit_report": "企业信用报告",
            },
            "report_subtype": {
                "enterprise": "企业信用报告",
            },
            "content_mode": {
                "native_text": "原生文本",
                "mixed": "混合文本与图像",
                "scanned_ocr": "扫描图像",
            },
        },
    }


def enterprise_credit_report_semantic_extensions() -> dict[str, Any]:
    """Return enterprise-only rendering policy without personal defaults."""
    return {
        "rendering_contract": {
            "authoritative_business_records": "datasets",
            "presentation_order": "reading.document_flow",
            "source_provenance": "structure",
            "summary_facts": "domain.facts",
            "do_not_union_representations": True,
        },
        "presentation_policy": {
            "classification": "sensitive_enterprise_credit_data",
            "default_display": "masked",
            "enhanced_markdown_display": "full",
            "mask_fields": [
                "report_number",
                "account_identifier",
                "contract_number",
                "limit_identifier",
                "identity_number",
                "business_registration_number",
            ],
            "source_structure_contains_verbatim_sensitive_text": True,
            "access_control_required": True,
        },
        "enhanced_markdown": {
            "privacy_mode": "full",
            "show_top_document_metadata": False,
            "section_layouts": {},
            "dataset_layouts": {},
        },
    }


__all__ = [
    "enterprise_credit_report_data_dictionary",
    "enterprise_credit_report_semantic_extensions",
]
