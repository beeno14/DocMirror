# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical dataset contract for personal detailed credit reports."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PERSONAL_DETAIL_DATASET_ORDER = (
    "personal_report_metadata",
    "identity_documents",
    "mobile_phone_records",
    "spouse_records",
    "residence_records",
    "employment_records",
    "recovery_records",
    "credit_accounts",
    "credit_lines",
    "repayment_liability_records",
    "repayment_records",
    "overdue_records",
    "postpaid_records",
    "postpaid_payment_history",
    "personal_detail_account_events",
    "personal_detail_summary_records",
    "personal_detail_summary_cells",
    "public_records",
    "inquiry_records",
    "statements",
    "annotations",
)


def personal_detail_data_dictionary() -> dict[str, Any]:
    """Return the strict, dataset-authoritative detailed-report dictionary."""
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_data_dictionary,
    )

    dictionary = deepcopy(credit_report_data_dictionary())
    dictionary.update(
        {
            "schema_id": "personal_credit_report_detailed",
            "version": "1.0.0",
            "definitions": {
                "authoritative_business_records": "datasets[*].rows",
                "canonical_record_identity": "record_id",
                "raw_value_policy": (
                    "normalized contains typed values; canonical_raw retains the source value; "
                    "raw retains the extractor observation; source retains evidence references."
                ),
                "missing_value_policy": (
                    "A missing normalized value is omitted or null. Source '--' remains in "
                    "canonical_raw and is never converted to numeric zero."
                ),
                "date_policy": "Dates use ISO 8601 day or month precision; long-term is a validity enum.",
                "amount_policy": (
                    "Amounts are stored without thousands separators and carry explicit account and "
                    "reporting currencies plus amount units."
                ),
            },
        }
    )
    datasets = dictionary.setdefault("datasets", {})
    datasets["personal_report_metadata"]["columns"].update(
        {
            "query_institution": {"label": "查询机构", "type": "string"},
            "query_reason": {"label": "查询原因", "type": "string"},
            "reporting_currency": {"label": "报告金额币种", "type": "enum"},
        }
    )
    datasets["recovery_records"] = {
        "definition": "One row per 被追偿信息 account.",
        "columns": {
            "recovery_record_id": {"label": "被追偿记录ID", "type": "string"},
            "sequence": {"label": "序号", "type": "integer"},
            "institution": {"label": "管理机构", "type": "string"},
            "business_type": {"label": "业务种类", "type": "string"},
            "debt_received_date": {"label": "债权接收日期", "type": "date"},
            "original_creditor": {"label": "原债权人", "type": "string"},
            "original_business_type": {"label": "原债务业务种类", "type": "string"},
            "debt_amount": {"label": "债权金额", "type": "money", "unit": "yuan"},
            "transfer_repayment_status": {"label": "债权转移时的还款状态", "type": "enum"},
            "snapshot_date": {"label": "信息截至日期", "type": "date"},
            "account_status": {"label": "账户状态", "type": "enum"},
            "balance": {"label": "余额", "type": "money", "unit": "yuan"},
            "last_repayment_date": {"label": "最近一次还款日期", "type": "date"},
            "close_date": {"label": "账户关闭日期", "type": "date"},
            "reporting_amount_currency": {"label": "报告金额币种", "type": "enum"},
            "reporting_amount_unit": {"label": "报告金额单位", "type": "enum"},
        },
    }
    datasets["mobile_phone_records"] = {
        "definition": "One row per reported mobile-phone history entry.",
        "columns": {
            "mobile_phone_record_id": {"label": "手机记录ID", "type": "string"},
            "sequence": {"label": "编号", "type": "integer"},
            "mobile_phone": {"label": "手机号码", "type": "string"},
            "information_updated_date": {"label": "信息更新日期", "type": "date"},
            "data_provider": {"label": "数据发生机构名称", "type": "string"},
        },
    }
    datasets["spouse_records"] = {
        "definition": "One row per reported spouse profile.",
        "columns": {
            "spouse_record_id": {"label": "配偶记录ID", "type": "string"},
            "name": {"label": "姓名", "type": "string"},
            "document_type": {"label": "证件类型", "type": "string"},
            "document_number": {"label": "证件号码", "type": "long_id", "sensitive": True},
            "employer": {"label": "工作单位", "type": "string"},
            "phone": {"label": "联系电话", "type": "string", "sensitive": True},
            "data_provider": {"label": "数据发生机构名称", "type": "string"},
        },
    }
    datasets["residence_records"] = {
        "definition": "One row per reported residence-history entry.",
        "columns": {
            "residence_record_id": {"label": "居住记录ID", "type": "string"},
            "sequence": {"label": "编号", "type": "integer"},
            "address": {"label": "居住地址", "type": "string"},
            "residential_phone": {"label": "住宅电话", "type": "string", "sensitive": True},
            "residence_status": {"label": "居住状况", "type": "string"},
            "information_updated_date": {"label": "信息更新日期", "type": "date"},
            "data_provider": {"label": "数据发生机构名称", "type": "string"},
            "page": {"label": "逻辑页码", "type": "integer"},
            "source_page": {"label": "源页码", "type": "integer"},
        },
    }
    datasets["employment_records"] = {
        "definition": "One row per reported employment-history entry.",
        "columns": {
            "employment_record_id": {"label": "职业记录ID", "type": "string"},
            "sequence": {"label": "编号", "type": "integer"},
            "employer": {"label": "工作单位", "type": "string"},
            "employer_type": {"label": "单位性质", "type": "string"},
            "employer_address": {"label": "单位地址", "type": "string"},
            "employer_phone": {"label": "单位电话", "type": "string", "sensitive": True},
            "occupation": {"label": "职业", "type": "string"},
            "industry": {"label": "行业", "type": "string"},
            "position": {"label": "职务", "type": "string"},
            "professional_title": {"label": "职称", "type": "string"},
            "entry_year": {"label": "进入本单位年份", "type": "integer"},
            "information_updated_date": {"label": "信息更新日期", "type": "date"},
            "data_provider": {"label": "数据发生机构名称", "type": "string"},
            "page": {"label": "逻辑页码", "type": "integer"},
            "source_page": {"label": "源页码", "type": "integer"},
        },
    }
    datasets["credit_accounts"]["definition"] = (
        "One row per detailed-report credit account card. account_type discriminates non-revolving "
        "loans, the two revolving-loan forms, credit cards, and quasi-credit cards."
    )
    datasets["credit_accounts"]["columns"].update(
        {
            "account_currency": {"label": "账户计价币种", "type": "enum"},
            "currency": {"label": "币种", "type": "enum"},
            "reporting_amount_currency": {"label": "报告金额币种", "type": "enum"},
            "actual_payment": {"label": "本月实还款", "type": "money", "unit": "yuan"},
            "billing_date": {"label": "账单日", "type": "date"},
            "co_borrower_flag": {"label": "共同借款标志", "type": "enum"},
            "credit_agreement_identifier": {"label": "授信协议标识", "type": "string"},
            "current_overdue_amount": {"label": "当前逾期总额", "type": "money", "unit": "yuan"},
            "current_overdue_periods": {"label": "当前逾期期数", "type": "integer"},
            "current_overdue_status": {"label": "当前逾期状态", "type": "enum"},
            "five_tier_class": {"label": "五级分类", "type": "enum"},
            "guarantee_type": {"label": "担保方式", "type": "enum"},
            "last_repayment_date": {"label": "最近一次还款日期", "type": "date"},
            "management_institution": {"label": "管理机构", "type": "string"},
            "maximum_overdraft_balance": {"label": "最大透支余额", "type": "money", "unit": "yuan"},
            "maximum_used_amount": {"label": "最大使用额度", "type": "money", "unit": "yuan"},
            "overdue_principal_31_60": {"label": "逾期31至60天未还本金", "type": "money", "unit": "yuan"},
            "overdue_principal_61_90": {"label": "逾期61至90天未还本金", "type": "money", "unit": "yuan"},
            "overdue_principal_91_180": {"label": "逾期91至180天未还本金", "type": "money", "unit": "yuan"},
            "overdue_principal_over_180": {"label": "逾期180天以上未还本金", "type": "money", "unit": "yuan"},
            "remaining_periods": {"label": "剩余还款期数", "type": "integer"},
            "recent_6_month_average_overdraft_balance": {
                "label": "最近6个月平均透支余额",
                "type": "money",
                "unit": "yuan",
            },
            "recent_6_month_average_used_amount": {
                "label": "最近6个月平均使用额度",
                "type": "money",
                "unit": "yuan",
            },
            "repayment_frequency": {"label": "还款频率", "type": "string"},
            "repayment_method": {"label": "还款方式", "type": "string"},
            "repayment_periods": {"label": "还款期数", "type": "integer"},
            "scheduled_payment": {"label": "本月应还款", "type": "money", "unit": "yuan"},
            "scheduled_payment_date": {"label": "应还款日", "type": "date"},
            "shared_credit_limit": {"label": "共享授信额度", "type": "money", "unit": "yuan"},
        }
    )
    datasets["credit_lines"]["definition"] = "One row per 授信协议信息 agreement."
    datasets["credit_lines"]["columns"].update(
        {
            "account_currency": {"label": "账户计价币种", "type": "enum"},
            "amount_unit": {"label": "金额单位", "type": "string"},
            "due_date": {"label": "到期日期", "type": "date"},
            "effective_date": {"label": "生效日期", "type": "date"},
            "reporting_amount_currency": {"label": "报告金额币种", "type": "enum"},
            "reporting_amount_unit": {"label": "报告金额单位", "type": "string"},
            "validity_type": {"label": "有效期类型", "type": "enum"},
        }
    )
    datasets["repayment_records"]["definition"] = (
        "One row per observed account calendar month; status preserves the report code and "
        "overdue_amount preserves the paired amount when reported."
    )
    datasets["repayment_records"]["columns"]["account_identifier"] = {
        "label": "账户标识",
        "type": "string",
    }
    datasets["repayment_liability_records"]["definition"] = (
        "One row per personal or enterprise related-repayment-responsibility account."
    )
    datasets["repayment_liability_records"]["columns"].update(
        {
            "currency": {"label": "币种", "type": "enum"},
            "reporting_amount_currency": {"label": "报告金额币种", "type": "enum"},
            "due_date": {"label": "到期日期", "type": "date"},
            "five_tier_class": {"label": "五级分类", "type": "enum"},
            "open_date": {"label": "开立日期", "type": "date"},
            "overdue_months_or_repayment_status": {"label": "逾期月数或还款状态", "type": "string"},
        }
    )
    datasets["postpaid_payment_history"] = {
        "definition": "One row per observed postpaid-service calendar month.",
        "columns": {
            "postpaid_payment_history_id": {"label": "后付费月度记录ID", "type": "string"},
            "postpaid_record_id": {"label": "后付费账户记录ID", "type": "string"},
            "institution": {"label": "机构名称", "type": "string"},
            "business_type": {"label": "业务类型", "type": "string"},
            "year": {"label": "年份", "type": "integer"},
            "month": {"label": "月份", "type": "integer"},
            "status": {"label": "缴费状态代码", "type": "enum"},
        },
    }
    datasets["personal_detail_account_events"] = {
        "definition": (
            "One row per detailed-account special transaction, latest repayment, "
            "large-installment block, or special-event statement."
        ),
        "columns": {
            "account_event_id": {"label": "账户补充事件ID", "type": "string"},
            "account_id": {"label": "账户ID", "type": "string"},
            "event_type": {"label": "事件类型", "type": "enum"},
            "transaction_type": {"label": "特殊交易类型", "type": "string"},
            "event_date": {"label": "发生日期", "type": "date"},
            "changed_months": {"label": "变更月数", "type": "integer"},
            "amount": {"label": "发生金额", "type": "money", "unit": "yuan"},
            "installment_limit": {"label": "大额专项分期额度", "type": "money", "unit": "yuan"},
            "effective_date": {"label": "生效日期", "type": "date"},
            "expiry_date": {"label": "到期日期", "type": "date"},
            "used_installment_amount": {"label": "已用分期金额", "type": "money", "unit": "yuan"},
            "five_tier_class": {"label": "五级分类", "type": "enum"},
            "balance": {"label": "余额", "type": "money", "unit": "yuan"},
            "repayment_date": {"label": "还款日期", "type": "date"},
            "repayment_amount": {"label": "还款金额", "type": "money", "unit": "yuan"},
            "repayment_status": {"label": "当前还款状态", "type": "enum"},
            "details": {"label": "明细或说明", "type": "text"},
        },
    }
    datasets["personal_detail_summary_records"] = {
        "definition": "One row per business-summary grid in the personal detailed report.",
        "columns": {
            "summary_record_id": {"label": "汇总记录ID", "type": "string"},
            "summary_type": {"label": "汇总类型", "type": "string"},
            "title": {"label": "汇总标题", "type": "string"},
            "source_table_id": {"label": "源表ID", "type": "string"},
            "source_column_count": {"label": "源表列数", "type": "integer"},
            "source_row_count": {"label": "源业务行数", "type": "integer"},
        },
    }
    datasets["personal_detail_summary_cells"] = {
        "definition": "One row per populated cell in a detailed-report business-summary grid.",
        "columns": {
            "summary_cell_id": {"label": "汇总单元格ID", "type": "string"},
            "summary_record_id": {"label": "汇总记录ID", "type": "string"},
            "summary_type": {"label": "汇总类型", "type": "string"},
            "title": {"label": "汇总标题", "type": "string"},
            "row_index": {"label": "业务行序号", "type": "integer"},
            "column_index": {"label": "列序号", "type": "integer"},
            "column_label": {"label": "列标题", "type": "string"},
            "value": {"label": "单元格值", "type": "string"},
        },
    }
    datasets["statements"] = {
        "definition": "One row per institution explanation or information-subject statement.",
        "columns": {
            "id": {"label": "说明或声明记录ID", "type": "string"},
            "note_type": {"label": "记录类型", "type": "enum"},
            "text": {"label": "说明或声明内容", "type": "text"},
            "added_date": {"label": "添加日期", "type": "date"},
            "logical_page": {"label": "逻辑页码", "type": "integer"},
            "source_page": {"label": "源页码", "type": "integer"},
        },
    }
    datasets["annotations"] = {
        "definition": "One row per dispute annotation in the personal detailed report.",
        "columns": {
            "id": {"label": "标注记录ID", "type": "string"},
            "note_type": {"label": "标注类型", "type": "enum"},
            "text": {"label": "标注内容", "type": "text"},
            "added_date": {"label": "添加日期", "type": "date"},
            "logical_page": {"label": "逻辑页码", "type": "integer"},
            "source_page": {"label": "源页码", "type": "integer"},
        },
    }
    datasets["overdue_records"]["columns"].update(
        {
            "currency": {"label": "币种", "type": "enum"},
            "five_tier_class": {"label": "五级分类", "type": "enum"},
            "month": {"label": "月份", "type": "integer"},
            "overdue_amount": {"label": "逾期金额", "type": "money", "unit": "yuan"},
            "overdue_level": {"label": "逾期等级", "type": "integer"},
            "year": {"label": "年份", "type": "integer"},
        }
    )
    datasets["postpaid_records"]["columns"].update(
        {
            "billing_month": {"label": "记账年月（YYYY-MM）", "type": "string"},
            "reporting_amount_currency": {"label": "报告金额币种", "type": "enum"},
        }
    )
    datasets["public_records"]["columns"]["content"] = {
        "label": "类别结构化明细（JSON）",
        "type": "text",
    }
    datasets["public_records"]["definition"] = (
        "One row per typed public record; record_type includes tax, judgment, enforcement, "
        "administrative penalty, housing fund, professional qualification, and award records."
    )
    return dictionary


def personal_detail_semantic_extensions() -> dict[str, Any]:
    """Return detailed-report canonical storage and presentation policy."""
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_semantic_extensions,
    )

    semantic = credit_report_semantic_extensions(report_subtype="personal_detail")
    semantic["dataset_document_order"] = list(PERSONAL_DETAIL_DATASET_ORDER)
    semantic["community_projection_overrides"] = {
        "dataset_labels": {
            "personal_report_metadata": "个人信用报告信息",
            "credit_accounts": "信贷交易账户明细",
            "credit_lines": "授信协议信息",
            "repayment_records": "月度还款记录",
            "overdue_records": "逾期明细（派生）",
            "public_records": "公共信息明细",
            "statements": "机构说明与本人声明",
        },
        "section_markers": {
            "statements": ["statements", "notes"],
            "annotations": ["annotations", "notes"],
        },
    }
    semantic["dataset_reading_columns"] = {
        "personal_report_metadata": [
            "report_number",
            "report_time",
            "subject_name",
            "primary_id_type",
            "primary_id_number",
            "query_institution",
            "query_reason",
        ],
        "identity_documents": ["sequence", "holder_name", "document_type", "document_number", "is_primary"],
        "mobile_phone_records": ["sequence", "mobile_phone", "information_updated_date", "data_provider"],
        "spouse_records": ["name", "document_type", "document_number", "employer", "phone", "data_provider"],
        "residence_records": [
            "sequence",
            "address",
            "residential_phone",
            "residence_status",
            "information_updated_date",
            "data_provider",
        ],
        "employment_records": [
            "sequence",
            "employer",
            "employer_type",
            "employer_address",
            "employer_phone",
            "occupation",
            "industry",
            "position",
            "professional_title",
            "entry_year",
            "information_updated_date",
            "data_provider",
        ],
        "recovery_records": [
            "sequence",
            "institution",
            "business_type",
            "debt_received_date",
            "original_creditor",
            "debt_amount",
            "account_status",
            "balance",
        ],
        "credit_accounts": [
            "sequence",
            "account_type",
            "institution",
            "business_type",
            "account_identifier",
            "card_tail",
            "open_date",
            "snapshot_date",
            "account_state",
            "credit_quality_status",
            "balance",
            "currency",
        ],
        "credit_lines": [
            "credit_line_id",
            "account_identifier",
            "facility_type",
            "institution",
            "effective_date",
            "due_date",
            "total_limit",
            "used_limit",
            "currency",
            "validity_type",
        ],
        "repayment_liability_records": [
            "sequence",
            "related_party_name",
            "institution",
            "underlying_business_type",
            "responsibility_type",
            "responsibility_amount",
            "balance",
            "currency",
        ],
        "repayment_records": ["account_identifier", "year", "month", "status", "overdue_amount"],
        "overdue_records": ["account_id", "period_scope", "year", "month", "overdue_level", "overdue_amount"],
        "postpaid_records": [
            "sequence",
            "institution",
            "business_type",
            "service_start_date",
            "payment_status",
            "current_arrears_amount",
            "billing_month",
        ],
        "postpaid_payment_history": ["institution", "business_type", "year", "month", "status"],
        "personal_detail_account_events": [
            "account_id",
            "event_type",
            "transaction_type",
            "event_date",
            "amount",
            "details",
        ],
        "personal_detail_summary_records": [
            "summary_type",
            "title",
            "source_row_count",
            "source_column_count",
        ],
        "personal_detail_summary_cells": [
            "summary_type",
            "row_index",
            "column_index",
            "column_label",
            "value",
        ],
        "public_records": ["sequence", "record_type", "authority", "start_date", "end_date", "content"],
        "inquiry_records": ["sequence", "inquiry_date", "institution", "reason", "inquiry_type"],
        "statements": ["note_type", "text", "added_date", "source_page"],
        "annotations": ["note_type", "text", "added_date", "source_page"],
    }
    semantic.setdefault("rendering_contract", {}).update(
        {
            "authoritative_business_records": "datasets[*].rows",
            "domain_specific_schema": "personal_credit_report_detailed.v1",
            "do_not_union_representations": True,
        }
    )
    return semantic


__all__ = [
    "PERSONAL_DETAIL_DATASET_ORDER",
    "personal_detail_data_dictionary",
    "personal_detail_semantic_extensions",
]
