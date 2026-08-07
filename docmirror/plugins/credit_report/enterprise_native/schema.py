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
            "extraction_failure_protocol": (
                "The semantic JSON extraction object reports schema, field, and record "
                "failures. A zero-record source section is complete, not failed."
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
            "enterprise_credit_accounts": {
                "definition": "企业报告正文中的一个可识别信贷账户卡片一行。",
                "columns": {},
            },
            "enterprise_credit_facilities": {
                "definition": "企业报告中的一份授信协议明细一行。",
                "columns": {},
            },
            "enterprise_repayment_responsibility_accounts": {
                "definition": "相关还款责任明细中的一个可识别账户一行。",
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
            "extraction_status": {
                "complete": "All checked extraction contracts passed.",
                "partial": "Business data was emitted with reported extraction failures.",
                "failed": "Input integrity prevented reliable canonical extraction.",
            },
            "extraction_failure_code": {
                "INPUT_INTEGRITY_VIOLATION": "The ParseResult failed an input-integrity contract.",
                "EXPECTED_FIELD_NOT_EXTRACTED": "A populated source field is absent from canonical data.",
                "EXTRACTED_FIELD_VALUE_MISMATCH": "A canonical value conflicts with its source value.",
                "RECORD_RECONSTRUCTION_MISMATCH": "Expected and extracted record populations disagree.",
                "UNCONSUMED_BUSINESS_TEXT": "A source unit was not assigned to the component graph.",
                "UNSTRUCTURED_BUSINESS_CONTENT": "Multiple fields remain packed into one content value.",
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
