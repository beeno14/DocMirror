# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical dataset contract for personal detailed credit reports."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PERSONAL_DETAIL_DATASET_ORDER = (
    "personal_report_metadata",
    "personal_profile",
    "personal_detail_field_observations",
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
    "personal_detail_credit_summary_metrics",
    "public_records",
    "tax_arrears_records",
    "civil_judgment_records",
    "enforcement_records",
    "administrative_penalty_records",
    "personal_housing_fund_records",
    "professional_qualification_records",
    "award_records",
    "inquiry_records",
    "statements",
    "annotations",
    "personal_detail_extraction_issues",
    "personal_detail_dataset_status",
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
            "version": "1.2.0",
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
                "absence_policy": (
                    "Empty business datasets are never proof of business absence. "
                    "personal_detail_dataset_status distinguishes explicitly_empty, "
                    "not_observed, extraction_failed, partial, and not_applicable."
                ),
                "uncertainty_policy": (
                    "personal_detail_field_observations contains only potentially flawed personal_profile fields "
                    "and typed failures from other assessed business datasets. Successful observations are omitted. "
                    "Nullable confidence means the source supplied no field confidence."
                ),
                "date_policy": "Dates use ISO 8601 day or month precision; long-term is a validity enum.",
                "amount_policy": (
                    "Amounts are decimal strings without thousands separators and carry explicit "
                    "currencies and units, or inherit them through declared dataset lineage."
                ),
            },
        }
    )
    fields = dictionary.setdefault("fields", {})
    fields.update(
        {
            "gender": {"label": "性别", "type": "string"},
            "birth_date": {"label": "出生日期", "type": "date"},
            "employment_status": {"label": "就业状况", "type": "string"},
            "education_level": {"label": "学历", "type": "string"},
            "degree": {"label": "学位", "type": "string"},
            "nationality": {"label": "国籍", "type": "string"},
            "mobile_phone": {"label": "手机号码", "type": "string", "sensitive": True},
            "work_phone": {"label": "单位电话", "type": "string", "sensitive": True},
            "residence_phone": {"label": "住宅电话", "type": "string", "sensitive": True},
            "email": {"label": "电子邮箱", "type": "string", "sensitive": True},
            "mailing_address": {"label": "通讯地址", "type": "string", "sensitive": True},
            "household_address": {"label": "户籍地址", "type": "string", "sensitive": True},
        }
    )
    datasets = dictionary.setdefault("datasets", {})
    datasets["personal_profile"] = {
        "definition": "One canonical row for the information subject's personal profile.",
        "columns": {
            "personal_profile_id": {"label": "个人资料记录ID", "type": "string"},
            "subject_name": {"label": "姓名", "type": "string"},
            "primary_id_type": {"label": "主证件类型", "type": "string"},
            "primary_id_number": {"label": "主证件号码", "type": "long_id", "sensitive": True},
            "gender": {"label": "性别", "type": "string"},
            "birth_date": {"label": "出生日期", "type": "date"},
            "marital_status": {"label": "婚姻状况", "type": "string"},
            "employment_status": {"label": "就业状况", "type": "string"},
            "education_level": {"label": "学历", "type": "string"},
            "degree": {"label": "学位", "type": "string"},
            "nationality": {"label": "国籍", "type": "string"},
            "mobile_phone": {"label": "手机号码", "type": "string", "sensitive": True},
            "work_phone": {"label": "单位电话", "type": "string", "sensitive": True},
            "residence_phone": {"label": "住宅电话", "type": "string", "sensitive": True},
            "email": {"label": "电子邮箱", "type": "string", "sensitive": True},
            "mailing_address": {"label": "通讯地址", "type": "string", "sensitive": True},
            "household_address": {"label": "户籍地址", "type": "string", "sensitive": True},
        },
    }
    datasets["personal_detail_field_observations"] = {
        "definition": (
            "One row per field-level observation when a business value is observed, normalized, "
            "corrected, inferred, ambiguous, unreadable, absent, or not observed."
        ),
        "columns": {
            "field_observation_id": {"label": "字段观测记录ID", "type": "string"},
            "dataset_name": {"label": "业务数据集", "type": "string"},
            "business_record_id": {"label": "业务记录ID", "type": "string"},
            "field_name": {"label": "字段名", "type": "string"},
            "raw_value": {"label": "源观测值", "type": "text"},
            "normalized_value": {"label": "规范值", "type": "text"},
            "observation_status": {"label": "观测状态", "type": "enum"},
            "confidence": {"label": "字段置信度", "type": "decimal"},
            "confidence_status": {"label": "置信度可用状态", "type": "enum"},
            "confidence_basis": {"label": "置信度依据", "type": "string"},
            "reason": {"label": "状态原因", "type": "text"},
        },
    }
    datasets["personal_detail_extraction_issues"] = {
        "definition": (
            "One row per unresolved or safely suppressed extraction condition. Decoding continues, "
            "the observed value is preserved, and a human may correct the cited field."
        ),
        "columns": {
            "extraction_issue_id": {"label": "提取问题记录ID", "type": "string"},
            "category": {"label": "问题类别", "type": "enum"},
            "issue_code": {"label": "问题代码", "type": "string"},
            "severity": {"label": "严重程度", "type": "enum"},
            "status": {"label": "处理状态", "type": "enum"},
            "parser_stage": {"label": "解析阶段", "type": "string"},
            "target_dataset": {"label": "受影响数据集", "type": "string"},
            "target_record_id": {"label": "受影响记录ID", "type": "string"},
            "field_name": {"label": "受影响字段", "type": "string"},
            "observed_value": {"label": "保留的观测值", "type": "text"},
            "candidate_value": {"label": "候选值", "type": "text"},
            "confidence": {"label": "候选置信度", "type": "decimal"},
            "reason_codes": {"label": "原因代码", "type": "array"},
            "message": {"label": "人工复核说明", "type": "text"},
        },
    }
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
            "currency": {"label": "币种", "type": "enum", "enum_ref": "currency_code"},
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
    datasets["repayment_records"]["columns"]["status"]["enum_ref"] = "repayment_status_code"
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
    datasets["postpaid_payment_history"]["columns"]["status"]["enum_ref"] = "postpaid_payment_status_code"
    datasets["personal_detail_credit_summary_metrics"] = {
        "definition": (
            "One typed metric per source summary cell. The row and column coordinates, business "
            "category, source value, numeric value, and reporting status remain distinct."
        ),
        "columns": {
            "credit_summary_metric_id": {"label": "信用概要指标ID", "type": "string"},
            "summary_record_id": {"label": "汇总记录ID", "type": "string"},
            "summary_type": {"label": "汇总类型", "type": "string"},
            "summary_code": {"label": "稳定汇总代码", "type": "enum"},
            "title": {"label": "汇总标题", "type": "string"},
            "source_table_id": {"label": "源表ID", "type": "string"},
            "row_index": {"label": "业务行序号", "type": "integer"},
            "column_index": {"label": "列序号", "type": "integer"},
            "metric_name": {"label": "指标名称", "type": "string"},
            "metric_code": {"label": "稳定指标代码", "type": "enum"},
            "mapping_status": {"label": "代码映射状态", "type": "enum"},
            "row_dimension_name": {"label": "行维度名称", "type": "string"},
            "row_dimension_value": {"label": "行维度值", "type": "text"},
            "business_category": {"label": "业务类别", "type": "string"},
            "source_value": {"label": "源报告值", "type": "text"},
            "value_type": {"label": "指标值类型", "type": "enum"},
            "numeric_value": {"label": "规范数值", "type": "decimal"},
            "text_value": {"label": "规范文本值", "type": "text"},
            "date_value": {"label": "规范日期值", "type": "date"},
            "reporting_status": {"label": "报告状态", "type": "enum"},
            "currency": {"label": "币种", "type": "enum", "enum_ref": "currency_code"},
            "amount_unit": {"label": "金额单位", "type": "enum", "enum_ref": "amount_unit"},
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
    for typed_public_dataset in (
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
    ):
        datasets[typed_public_dataset]["columns"]["public_record_id"] = {
            "label": "公共记录源ID",
            "type": "string",
        }
    datasets["personal_housing_fund_records"] = {
        "definition": "One row per personal housing-fund contribution record.",
        "columns": {
            "personal_housing_fund_id": {"label": "住房公积金记录ID", "type": "string"},
            "public_record_id": {"label": "公共记录源ID", "type": "string"},
            "sequence": {"label": "序号", "type": "integer"},
            "employer": {"label": "缴存单位", "type": "string"},
            "contribution_location": {"label": "缴存地", "type": "string"},
            "participation_date": {"label": "开户日期", "type": "date"},
            "first_contribution_month": {"label": "初缴月份", "type": "string"},
            "paid_through_month": {"label": "缴至月份", "type": "string"},
            "payment_status": {"label": "当前缴存状态", "type": "enum"},
            "monthly_contribution": {"label": "月缴存额", "type": "money", "unit": "yuan"},
            "personal_contribution_ratio": {"label": "个人缴存比例", "type": "string"},
            "employer_contribution_ratio": {"label": "单位缴存比例", "type": "string"},
            "information_updated_month": {"label": "信息更新月份", "type": "string"},
            "reporting_amount_currency": {"label": "报告金额币种", "type": "enum", "enum_ref": "currency_code"},
            "reporting_amount_unit": {"label": "报告金额单位", "type": "string"},
        },
    }
    datasets["professional_qualification_records"] = {
        "definition": "One row per reported professional qualification.",
        "columns": {
            "professional_qualification_id": {"label": "执业资格记录ID", "type": "string"},
            "public_record_id": {"label": "公共记录源ID", "type": "string"},
            "sequence": {"label": "序号", "type": "integer"},
            "qualification_name": {"label": "执业资格名称", "type": "string"},
            "level": {"label": "资格等级", "type": "string"},
            "issuing_authority": {"label": "颁发机构", "type": "string"},
            "authority_location": {"label": "机构所在地", "type": "string"},
            "obtained_date": {"label": "取得日期", "type": "date"},
            "expiry_date": {"label": "到期日期", "type": "date"},
            "revocation_date": {"label": "吊销日期", "type": "date"},
        },
    }
    datasets["award_records"] = {
        "definition": "One row per reported administrative or public award.",
        "columns": {
            "award_record_id": {"label": "行政奖励记录ID", "type": "string"},
            "public_record_id": {"label": "公共记录源ID", "type": "string"},
            "sequence": {"label": "序号", "type": "integer"},
            "authority": {"label": "奖励机构", "type": "string"},
            "award_content": {"label": "奖励内容", "type": "text"},
            "effective_date": {"label": "生效日期", "type": "date"},
            "end_date": {"label": "截止日期", "type": "date"},
        },
    }
    for typed_public_dataset in (
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
        "personal_housing_fund_records",
        "professional_qualification_records",
        "award_records",
    ):
        datasets[typed_public_dataset]["columns"]["unmapped_content"] = {
            "label": "未映射源内容",
            "type": "text",
        }
    datasets["personal_detail_dataset_status"] = {
        "definition": (
            "One row per potentially incomplete or uncertain business dataset. Successful observations and "
            "source-confirmed empty/not-applicable datasets are omitted."
        ),
        "columns": {
            "dataset_status_id": {"label": "数据集状态记录ID", "type": "string"},
            "dataset_name": {"label": "业务数据集", "type": "string"},
            "applicability": {"label": "适用性", "type": "enum"},
            "presence_status": {"label": "存在状态", "type": "enum"},
            "observed_row_count": {"label": "已观测行数", "type": "integer"},
            "source_statement": {"label": "源文状态声明", "type": "text"},
            "confidence": {"label": "状态置信度", "type": "decimal"},
            "reason": {"label": "状态原因", "type": "text"},
        },
    }
    dictionary.setdefault("enums", {}).update(
        {
            "observation_status": {
                "observed": "直接观测",
                "normalized": "规范化",
                "ocr_corrected": "OCR纠正",
                "inferred": "推断",
                "ambiguous": "存在歧义",
                "unreadable": "无法辨认",
                "not_observed": "未观测到",
                "explicitly_absent": "源文明确缺失",
                "not_applicable": "不适用",
            },
            "presence_status": {
                "observed_nonempty": "已观测且非空",
                "explicitly_empty": "源文明示无记录",
                "not_applicable": "不适用",
                "not_observed": "未观测到",
                "partial": "部分观测",
                "extraction_failed": "提取失败",
                "unknown": "未知",
            },
            "value_type": {
                "integer": "整数",
                "decimal": "小数",
                "money": "金额",
                "percentage": "百分比",
                "date": "日期",
                "text": "文本",
                "placeholder": "未报告占位符",
            },
            "confidence_status": {
                "available": "可用",
                "not_available": "源未提供字段置信度",
            },
            "mapping_status": {"mapped": "已映射", "unmapped": "未映射"},
            "currency_code": {"CNY": "人民币"},
            "amount_unit": {"yuan": "元"},
            "repayment_status_code": {
                "*": "本月没有还款历史",
                "N": "正常",
                "1": "逾期1至30天",
                "2": "逾期31至60天",
                "3": "逾期61至90天",
                "4": "逾期91至120天",
                "5": "逾期121至150天",
                "6": "逾期151至180天",
                "7": "逾期180天以上",
                "A": "信用卡因调整账单日本月不出单",
                "B": "呆账",
                "C": "结清或销户（依账户类型解释）",
                "D": "担保人代还",
                "G": "结束",
                "M": "约定还款日后月底前还款",
                "Z": "以资抵债",
                "#": "账户已开立但当月状态未知",
                "unknown": "源状态无法可靠识别（非报告代码）",
            },
            "postpaid_payment_status_code": {
                "*": "服务已开通但本月不需缴费",
                "N": "正常",
                "0": "欠费超过宽限期不足1个月",
                "1": "欠费超过宽限期1个月不足2个月",
                "2": "欠费超过宽限期2个月不足3个月",
                "3": "欠费超过宽限期3个月不足4个月",
                "4": "欠费超过宽限期4个月不足5个月",
                "5": "欠费超过宽限期5个月不足6个月",
                "6": "欠费超过宽限期6个月以上",
                "C": "正常销户（结清后的销户）",
                "G": "结束（非正常结清的销户）",
                "#": "未知：没有此期数据",
                "unknown": "源状态无法可靠识别（非报告代码）",
            },
        }
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
        "internal_fields": [
            f"personal_detail_expected_{name}_count"
            for name in (
                "personal_profile",
                "personal_detail_field_observations",
                "personal_detail_extraction_issues",
                "personal_detail_credit_summary_metrics",
                "tax_arrears_records",
                "civil_judgment_records",
                "enforcement_records",
                "administrative_penalty_records",
                "personal_housing_fund_records",
                "professional_qualification_records",
                "award_records",
                "personal_detail_dataset_status",
            )
        ],
        "dataset_labels": {
            "personal_report_metadata": "个人信用报告信息",
            "personal_profile": "个人基本资料",
            "personal_detail_field_observations": "字段观测与不确定性",
            "personal_detail_extraction_issues": "提取问题与人工复核队列",
            "credit_accounts": "信贷交易账户明细",
            "credit_lines": "授信协议信息",
            "repayment_records": "月度还款记录",
            "overdue_records": "逾期明细（派生）",
            "personal_detail_credit_summary_metrics": "信用概要业务指标",
            "public_records": "公共信息明细",
            "tax_arrears_records": "欠税记录",
            "civil_judgment_records": "民事判决记录",
            "enforcement_records": "强制执行记录",
            "administrative_penalty_records": "行政处罚记录",
            "personal_housing_fund_records": "住房公积金参缴记录",
            "professional_qualification_records": "执业资格记录",
            "award_records": "行政奖励记录",
            "statements": "机构说明与本人声明",
            "personal_detail_dataset_status": "业务数据集存在状态",
        },
        "section_markers": {
            "personal_profile": ["basic_information"],
            "personal_detail_field_observations": ["extraction_review"],
            "personal_detail_extraction_issues": ["extraction_review"],
            "personal_detail_credit_summary_metrics": ["credit_summary"],
            "tax_arrears_records": ["public_records"],
            "civil_judgment_records": ["public_records"],
            "enforcement_records": ["public_records"],
            "administrative_penalty_records": ["public_records"],
            "personal_housing_fund_records": ["public_records"],
            "professional_qualification_records": ["public_records"],
            "award_records": ["public_records"],
            "personal_detail_dataset_status": ["extraction_review"],
            "statements": ["statements", "notes"],
            "annotations": ["annotations", "notes"],
        },
        "dataset_representation_roles": {
            "personal_profile": "business_canonical",
            "personal_detail_credit_summary_metrics": "business_canonical",
            "tax_arrears_records": "business_canonical",
            "civil_judgment_records": "business_canonical",
            "enforcement_records": "business_canonical",
            "administrative_penalty_records": "business_canonical",
            "personal_housing_fund_records": "business_canonical",
            "professional_qualification_records": "business_canonical",
            "award_records": "business_canonical",
            "personal_detail_field_observations": "control",
            "personal_detail_extraction_issues": "control",
            "personal_detail_dataset_status": "control",
            "personal_detail_summary_cells": "source_canonical",
            "public_records": "source_canonical",
        },
        "dataset_grains": {
            "personal_profile": "one row per information subject in the report",
            "personal_detail_field_observations": (
                "one row per potentially flawed personal-profile field or assessed typed extraction failure"
            ),
            "personal_detail_extraction_issues": "one row per distinct extraction condition",
            "personal_detail_credit_summary_metrics": "one row per source summary-grid cell",
            "tax_arrears_records": "one row per tax-arrears public record",
            "civil_judgment_records": "one row per civil-judgment public record",
            "enforcement_records": "one row per enforcement public record",
            "administrative_penalty_records": "one row per administrative-penalty public record",
            "personal_housing_fund_records": "one row per housing-fund public record",
            "professional_qualification_records": "one row per professional-qualification public record",
            "award_records": "one row per award public record",
            "personal_detail_dataset_status": "one row per potentially incomplete or uncertain business dataset",
        },
        "dataset_derived_from": {
            "personal_profile": ["personal_report_metadata", "subject_profile_facts"],
            "personal_detail_field_observations": ["subject_profile_facts", "personal_detail_extraction_issues"],
            "personal_detail_extraction_issues": ["ocr_audit", "page_topology_audit", "native_parser"],
            "personal_detail_credit_summary_metrics": [
                "personal_detail_summary_records",
                "personal_detail_summary_cells",
            ],
            "tax_arrears_records": ["public_records"],
            "civil_judgment_records": ["public_records"],
            "enforcement_records": ["public_records"],
            "administrative_penalty_records": ["public_records"],
            "personal_housing_fund_records": ["public_records"],
            "professional_qualification_records": ["public_records"],
            "award_records": ["public_records"],
            "personal_detail_dataset_status": ["final_assembled_business_datasets"],
        },
        "dataset_foreign_keys": {
            "repayment_records": [
                {
                    "columns": ["account_id"],
                    "reference_dataset": "credit_accounts",
                    "reference_columns": ["record_id"],
                }
            ],
            "personal_detail_credit_summary_metrics": [
                {
                    "columns": ["summary_record_id"],
                    "reference_dataset": "personal_detail_summary_records",
                    "reference_columns": ["summary_record_id"],
                }
            ],
            **{
                name: [
                    {
                        "columns": ["public_record_id"],
                        "reference_dataset": "public_records",
                        "reference_columns": ["public_record_id"],
                    }
                ]
                for name in (
                    "tax_arrears_records",
                    "civil_judgment_records",
                    "enforcement_records",
                    "administrative_penalty_records",
                    "personal_housing_fund_records",
                    "professional_qualification_records",
                    "award_records",
                )
            },
        },
        "completeness": {
            name: {
                "basis": "domain_fact_count",
                "count_key": f"personal_detail_expected_{name}_count",
                "public_basis": "contract_projection_count",
            }
            for name in (
                "personal_profile",
                "personal_detail_field_observations",
                "personal_detail_extraction_issues",
                "personal_detail_credit_summary_metrics",
                "tax_arrears_records",
                "civil_judgment_records",
                "enforcement_records",
                "administrative_penalty_records",
                "personal_housing_fund_records",
                "professional_qualification_records",
                "award_records",
                "personal_detail_dataset_status",
            )
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
        "personal_profile": [
            "subject_name",
            "primary_id_type",
            "primary_id_number",
            "gender",
            "birth_date",
            "marital_status",
            "employment_status",
            "education_level",
            "degree",
            "nationality",
            "mobile_phone",
            "email",
            "mailing_address",
        ],
        "personal_detail_field_observations": [
            "dataset_name",
            "field_name",
            "observation_status",
            "normalized_value",
            "confidence",
            "confidence_status",
            "confidence_basis",
            "reason",
        ],
        "personal_detail_extraction_issues": [
            "category",
            "issue_code",
            "severity",
            "status",
            "target_dataset",
            "field_name",
            "observed_value",
            "candidate_value",
            "confidence",
            "message",
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
        "personal_detail_credit_summary_metrics": [
            "summary_type",
            "summary_code",
            "row_dimension_name",
            "row_dimension_value",
            "business_category",
            "metric_name",
            "metric_code",
            "mapping_status",
            "source_value",
            "numeric_value",
            "value_type",
            "reporting_status",
            "currency",
            "amount_unit",
        ],
        "public_records": ["sequence", "record_type", "authority", "start_date", "end_date", "content"],
        "tax_arrears_records": ["sequence", "tax_authority", "statistics_date", "arrears_amount"],
        "civil_judgment_records": [
            "sequence",
            "filing_court",
            "case_number",
            "cause",
            "filing_date",
            "claim_subject",
            "claim_amount",
            "judgment_result",
        ],
        "enforcement_records": [
            "sequence",
            "court",
            "case_number",
            "filing_date",
            "case_status",
            "requested_amount",
            "executed_amount",
        ],
        "administrative_penalty_records": [
            "sequence",
            "authority",
            "document_number",
            "penalty_content",
            "penalty_amount",
            "effective_date",
            "end_date",
        ],
        "personal_housing_fund_records": [
            "sequence",
            "employer",
            "contribution_location",
            "payment_status",
            "monthly_contribution",
            "paid_through_month",
        ],
        "professional_qualification_records": [
            "sequence",
            "qualification_name",
            "level",
            "issuing_authority",
            "obtained_date",
            "expiry_date",
        ],
        "award_records": ["sequence", "authority", "award_content", "effective_date", "end_date"],
        "inquiry_records": ["sequence", "inquiry_date", "institution", "reason", "inquiry_type"],
        "statements": ["note_type", "text", "added_date", "source_page"],
        "annotations": ["note_type", "text", "added_date", "source_page"],
        "personal_detail_dataset_status": [
            "dataset_name",
            "applicability",
            "presence_status",
            "observed_row_count",
            "reason",
        ],
    }
    semantic["personal_detail_contract"] = {
        "canonical_profile_dataset": "personal_profile",
        "canonical_credit_summary_dataset": "personal_detail_credit_summary_metrics",
        "canonical_public_record_datasets": [
            "tax_arrears_records",
            "civil_judgment_records",
            "enforcement_records",
            "administrative_penalty_records",
            "personal_housing_fund_records",
            "professional_qualification_records",
            "award_records",
        ],
        "absence_dataset": "personal_detail_dataset_status",
        "uncertainty_dataset": "personal_detail_field_observations",
        "extraction_issue_dataset": "personal_detail_extraction_issues",
        "absence_requires_explicit_source_evidence": True,
        "empty_dataset_means_absent": False,
        "uncertainty_coverage": {
            "mode": "potentially_flawed_only",
            "covered_dataset": "personal_profile",
            "covered_fields": [
                "gender",
                "birth_date",
                "marital_status",
                "employment_status",
                "education_level",
                "degree",
                "nationality",
                "mobile_phone",
                "work_phone",
                "residence_phone",
                "email",
                "mailing_address",
                "household_address",
            ],
            "unlisted_dataset_default": "not_assessed",
            "confidence_policy": "nullable_when_source_confidence_unavailable",
        },
        "codebooks": {
            "repayment_records.status": "repayment_status_code",
            "postpaid_payment_history.status": "postpaid_payment_status_code",
            "source_pages": [14, 15],
        },
    }
    semantic["domain_schema"] = {
        "id": "personal_credit_report_detailed",
        "version": "1.2.0",
        "contract_uri": (
            "https://valuemapglobal.github.io/DocMirror/schemas/personal_credit_report_detailed.schema.json"
        ),
        "compatibility": "additive-over-1.1; community-v3-envelope",
    }
    semantic.setdefault("rendering_contract", {}).update(
        {
            "authoritative_business_records": "datasets[*].rows",
            "domain_specific_schema": "personal_credit_report_detailed.v1.2",
            "do_not_union_representations": True,
        }
    )
    return semantic


__all__ = [
    "PERSONAL_DETAIL_DATASET_ORDER",
    "personal_detail_data_dictionary",
    "personal_detail_semantic_extensions",
]
