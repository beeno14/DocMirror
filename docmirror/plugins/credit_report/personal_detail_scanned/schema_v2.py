# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PBOC-native v2 contract for personal detailed credit reports.

The extraction layer still produces the stable v1.2 intermediate collections.
This module is the plugin-owned migration boundary that projects those
collections into the v2 business model before Community JSON is assembled.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

PBOC_SCHEMA_ID = "personal_credit_report_detailed"
PBOC_SCHEMA_VERSION = "2.0.0"
PBOC_CONTRACT_URI = (
    "https://valuemapglobal.github.io/DocMirror/schemas/"
    "personal_credit_report_detailed_v2.schema.json"
)
PBOC_SCHEMA_VERSION_ENV = "DOCMIRROR_PERSONAL_DETAIL_SCHEMA_VERSION"

PBOC_DATASET_ORDER = (
    "report_metadata",
    "report_query",
    "subject_profile",
    "subject_identity_documents",
    "fraud_warnings",
    "dispute_overview",
    "subject_mobile_phones",
    "subject_spouse",
    "subject_residences",
    "subject_employment",
    "credit_scores",
    "credit_score_reasons",
    "credit_business_overview",
    "bad_debt_summary",
    "recovery_summary",
    "delinquency_summary",
    "account_portfolio_summary",
    "repayment_responsibility_summary",
    "postpaid_arrears_summary",
    "public_negative_summary",
    "inquiry_summary",
    "credit_accounts",
    "credit_account_snapshots",
    "credit_account_history_windows",
    "credit_account_monthly_performance",
    "credit_account_latest_repayments",
    "credit_account_special_transactions",
    "credit_account_special_events",
    "credit_card_large_installments",
    "recovery_account_details",
    "credit_agreements",
    "account_credit_agreement_links",
    "repayment_responsibilities",
    "postpaid_accounts",
    "postpaid_history_windows",
    "postpaid_monthly_performance",
    "tax_arrears_records",
    "civil_judgment_records",
    "enforcement_records",
    "administrative_penalty_records",
    "housing_fund_records",
    "social_assistance_records",
    "professional_qualification_records",
    "administrative_award_records",
    "annotation_statement_groups",
    "annotation_statements",
    "inquiries",
    "pboc_extension_fields",
    "dataset_status",
)

_PBOC_DATASET_LABELS = {
    "report_metadata": "报告元数据",
    "report_query": "报告查询信息",
    "subject_profile": "个人基本资料",
    "subject_identity_documents": "身份信息",
    "fraud_warnings": "欺诈警示",
    "dispute_overview": "异议概要",
    "subject_mobile_phones": "手机号码历史",
    "subject_spouse": "配偶信息",
    "subject_residences": "居住信息",
    "subject_employment": "职业信息",
    "credit_scores": "信用评分",
    "credit_score_reasons": "评分原因",
    "credit_business_overview": "信贷业务概要",
    "bad_debt_summary": "呆账概要",
    "recovery_summary": "被追偿概要",
    "delinquency_summary": "逾期透支概要",
    "account_portfolio_summary": "账户构成概要",
    "repayment_responsibility_summary": "相关还款责任概要",
    "postpaid_arrears_summary": "后付费欠费概要",
    "public_negative_summary": "公共负面信息概要",
    "inquiry_summary": "查询概要",
    "credit_accounts": "信贷交易账户",
    "credit_account_snapshots": "账户时点信息",
    "credit_account_history_windows": "账户历史展示区间",
    "credit_account_monthly_performance": "账户月度表现",
    "credit_account_latest_repayments": "账户最近还款",
    "credit_account_special_transactions": "账户特殊交易",
    "credit_account_special_events": "账户特殊事件",
    "credit_card_large_installments": "信用卡大额专项分期",
    "recovery_account_details": "被追偿信息",
    "credit_agreements": "授信协议",
    "account_credit_agreement_links": "账户授信协议关联",
    "repayment_responsibilities": "相关还款责任",
    "postpaid_accounts": "后付费账户",
    "postpaid_history_windows": "后付费历史展示区间",
    "postpaid_monthly_performance": "后付费月度表现",
    "tax_arrears_records": "欠税记录",
    "civil_judgment_records": "民事判决记录",
    "enforcement_records": "强制执行记录",
    "administrative_penalty_records": "行政处罚记录",
    "housing_fund_records": "住房公积金参缴记录",
    "social_assistance_records": "社会救助记录",
    "professional_qualification_records": "执业资格记录",
    "administrative_award_records": "行政奖励记录",
    "annotation_statement_groups": "说明标注对象",
    "annotation_statements": "机构说明本人声明及异议标注",
    "inquiries": "查询记录",
    "pboc_extension_fields": "人行业务扩展字段",
    "dataset_status": "业务数据集状态",
}

_DIRECT_DATASET_RENAMES = {
    "personal_profile": "subject_profile",
    "identity_documents": "subject_identity_documents",
    "mobile_phone_records": "subject_mobile_phones",
    "spouse_records": "subject_spouse",
    "residence_records": "subject_residences",
    "employment_records": "subject_employment",
    "recovery_records": "recovery_account_details",
    "credit_lines": "credit_agreements",
    "postpaid_records": "postpaid_accounts",
    "postpaid_payment_history": "postpaid_monthly_performance",
    "tax_arrears_records": "tax_arrears_records",
    "civil_judgment_records": "civil_judgment_records",
    "enforcement_records": "enforcement_records",
    "administrative_penalty_records": "administrative_penalty_records",
    "personal_housing_fund_records": "housing_fund_records",
    "professional_qualification_records": "professional_qualification_records",
    "award_records": "administrative_award_records",
    "inquiry_records": "inquiries",
}

_ACCOUNT_TYPE_CODES = {
    "non_revolving_loan": ("D1", "非循环贷账户"),
    "revolving_loan": ("R1", "循环贷账户（一）"),
    "revolving_loan_subaccount": ("R1", "循环贷账户（一）"),
    "revolving_loan_account": ("R2", "循环贷账户（二）"),
    "credit_card": ("R3", "贷记卡账户"),
    "quasi_credit_card": ("R4", "准贷记卡账户"),
}

_PUBLIC_RECORD_TARGETS = {
    "tax_arrears": "tax_arrears_records",
    "civil_judgment": "civil_judgment_records",
    "enforcement": "enforcement_records",
    "administrative_penalty": "administrative_penalty_records",
    "housing_fund": "housing_fund_records",
    "social_assistance": "social_assistance_records",
    "low_income_assistance": "social_assistance_records",
    "professional_qualification": "professional_qualification_records",
    "award": "administrative_award_records",
    "administrative_award": "administrative_award_records",
}

_ACCOUNT_EVENT_TARGETS = {
    "latest_repayment": "credit_account_latest_repayments",
    "large_installment": "credit_card_large_installments",
    "special_event": "credit_account_special_events",
    "special_event_note": "credit_account_special_events",
    "special_transaction": "credit_account_special_transactions",
}

_PUBLIC_STATUS_TARGETS = (
    "tax_arrears_records",
    "civil_judgment_records",
    "enforcement_records",
    "administrative_penalty_records",
    "housing_fund_records",
    "social_assistance_records",
    "professional_qualification_records",
    "administrative_award_records",
)


def _normalized(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("normalized")
    return dict(value) if isinstance(value, dict) else {
        key: deepcopy(item)
        for key, item in record.items()
        if key
        not in {
            "record_id",
            "canonical_raw",
            "raw",
            "source",
            "source_refs",
            "source_cell_refs",
            "evidence_ids",
            "confidence",
            "review",
        }
    }


def _replace_normalized(record: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(record)
    if isinstance(projected.get("normalized"), dict):
        projected["normalized"] = normalized
    else:
        for key in list(projected):
            if key not in {
                "record_id",
                "canonical_raw",
                "raw",
                "source",
                "source_refs",
                "source_cell_refs",
                "evidence_ids",
                "confidence",
                "review",
            }:
                projected.pop(key, None)
        projected.update(normalized)
    return projected


def _project_records(
    records: list[dict[str, Any]],
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        normalized = _normalized(record)
        if transform is not None:
            normalized = transform(normalized)
        projected.append(_replace_normalized(record, normalized))
    return projected


def _rename_key(values: dict[str, Any], old: str, new: str) -> None:
    if old in values and new not in values:
        values[new] = values.pop(old)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    return int(text) if re.fullmatch(r"[+-]?\d+", text) else None


def _year_month(year: Any, month: Any) -> str | None:
    normalized_year = _integer(year)
    normalized_month = _integer(month)
    if normalized_year is None or normalized_month is None:
        return None
    if not (1 <= normalized_year <= 9999 and 1 <= normalized_month <= 12):
        return None
    return f"{normalized_year:04d}-{normalized_month:02d}"


def _month_value(value: Any) -> str | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})[-./](\d{1,2})(?:[-./]\d{1,2})?", text)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not (1 <= year <= 9999 and 1 <= month <= 12):
        return None
    return f"{year:04d}-{month:02d}"


def _decimal_string(value: Any) -> str | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    text = str(value).strip().replace(",", "")
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _report_metadata(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "personal_report_metadata_id", "report_metadata_id")
    for key in ("query_institution", "query_reason"):
        values.pop(key, None)
    return values


def _report_query(values: dict[str, Any]) -> dict[str, Any]:
    query = {
        key: deepcopy(values[key])
        for key in (
            "report_number",
            "report_time",
            "subject_name",
            "primary_id_type",
            "primary_id_number",
            "query_institution",
            "query_reason",
        )
        if values.get(key) not in (None, "")
    }
    source_id = values.get("personal_report_metadata_id") or values.get("report_metadata_id")
    query["report_query_id"] = f"report_query:{source_id or 'primary'}"
    return query


def _subject_profile(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "personal_profile_id", "subject_profile_id")
    return values


def _credit_account(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    account_type = str(values.get("account_type") or "")
    type_code, type_label = _ACCOUNT_TYPE_CODES.get(account_type, (None, None))
    if type_code:
        values["pboc_account_type_code"] = type_code
        values["pboc_account_type_label"] = type_label
    return values


def _credit_agreement(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "credit_line_id", "credit_agreement_id")
    _rename_key(values, "total_limit", "facility_limit")
    return values


def _monthly_performance(
    values: dict[str, Any], account_types: dict[str, str]
) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "repayment_id", "monthly_performance_id")
    year = values.pop("year", None)
    month = values.pop("month", None)
    performance_month = _year_month(year, month)
    if performance_month is not None:
        values["performance_month"] = performance_month
    else:
        if year not in (None, ""):
            values["source_year"] = year
        if month not in (None, ""):
            values["source_month"] = month
    _rename_key(values, "status", "status_code")
    if values.get("status_code") not in (None, ""):
        values["status_code"] = str(values["status_code"])
    amount = values.pop("overdue_amount", None)
    if amount not in (None, ""):
        normalized_amount = _decimal_string(amount)
        if normalized_amount is not None:
            values["status_amount"] = normalized_amount
            account_type = account_types.get(str(values.get("account_id") or ""), "")
            values["status_amount_semantics"] = (
                "overdraft_balance" if account_type == "quasi_credit_card" else "delinquent_amount"
            )
            values.setdefault("reporting_amount_currency", "CNY")
            values.setdefault("reporting_amount_unit", "yuan")
        else:
            values["source_status_amount"] = amount
    return values


def _responsible_party_category(values: dict[str, Any]) -> str:
    id_type = str(values.get("related_party_id_type") or "")
    id_number = str(values.get("related_party_id_number") or "")
    if "身份" in id_type:
        return "person"
    if "统一社会信用" in id_type or "组织机构" in id_type:
        return "organization"
    if re.fullmatch(r"\d{17}[0-9Xx]", id_number):
        return "person"
    if re.fullmatch(r"[0-9A-Z]{18}", id_number):
        return "organization"
    return "unknown"


def _repayment_responsibility(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "liability_id", "repayment_responsibility_id")
    category = _responsible_party_category(values)
    values["related_party_category"] = category
    combined = values.pop("overdue_months_or_repayment_status", None)
    if combined not in (None, ""):
        values["source_status_value"] = combined
        if category == "person" and str(combined).isdigit():
            values["overdue_months"] = int(str(combined))
        else:
            values["repayment_status_code"] = str(combined)
    return values


def _monthly_postpaid(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "postpaid_payment_history_id", "postpaid_monthly_performance_id")
    year = values.pop("year", None)
    month = values.pop("month", None)
    performance_month = _year_month(year, month)
    if performance_month is not None:
        values["performance_month"] = performance_month
    else:
        if year not in (None, ""):
            values["source_year"] = year
        if month not in (None, ""):
            values["source_month"] = month
    _rename_key(values, "status", "status_code")
    if values.get("status_code") not in (None, ""):
        values["status_code"] = str(values["status_code"])
    return values


def _summary_metric(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "credit_summary_metric_id", "credit_business_overview_id")
    dimension_name = str(values.pop("row_dimension_name", "") or "")
    dimension_value = values.pop("row_dimension_value", None)
    if dimension_value not in (None, ""):
        if "类别" in dimension_name or "category" in dimension_name.lower():
            values["business_category"] = dimension_value
        elif "类型" in dimension_name or "type" in dimension_name.lower():
            values["business_type"] = dimension_value
        else:
            values["business_dimension_name"] = dimension_name
            values["business_dimension_value"] = dimension_value
    return values


def _event_target(values: dict[str, Any]) -> str | None:
    event_type = str(values.get("event_type") or "")
    return _ACCOUNT_EVENT_TARGETS.get(event_type)


def _account_event(values: dict[str, Any], target: str) -> dict[str, Any]:
    values = dict(values)
    target_id = {
        "credit_account_latest_repayments": "latest_repayment_id",
        "credit_card_large_installments": "large_installment_id",
        "credit_account_special_events": "special_event_id",
        "credit_account_special_transactions": "special_transaction_id",
    }[target]
    _rename_key(values, "account_event_id", target_id)
    return values


def _ratio_percent(value: Any) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([+-]?\d+)\s*%\s*", value)
        if match:
            return int(match.group(1))
    return value


def _housing_fund(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "personal_housing_fund_id", "housing_fund_record_id")
    for old, new in (
        ("personal_contribution_ratio", "personal_contribution_ratio_percent"),
        ("employer_contribution_ratio", "employer_contribution_ratio_percent"),
    ):
        if old in values:
            values[new] = _ratio_percent(values.pop(old))
    return values


def _month_precision_public(values: dict[str, Any], renames: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    values = dict(values)
    for old, new in renames:
        if old in values:
            original = values.pop(old)
            month = _month_value(original)
            if month is not None:
                values[new] = month
            elif original not in (None, ""):
                values[f"source_{old}"] = original
    return values


def _annotation_records(
    statements: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for kind, records in (("statement", statements), ("annotation", annotations)):
        for index, record in enumerate(records, start=1):
            values = _normalized(record)
            legacy_id = values.get("statement_id") or values.get("annotation_id") or record.get("record_id")
            statement_id = f"annotation_statement:{legacy_id or f'{kind}:{index}'}"
            group_id = f"annotation_group:{legacy_id or f'{kind}:{index}'}"
            group_values = {
                "annotation_statement_group_id": group_id,
                "annotation_kind": kind,
                "target_dataset": values.get("target_dataset"),
                "target_record_id": values.get("target_record_id"),
            }
            group_record = _replace_normalized(record, group_values)
            group_record["record_id"] = group_id
            groups.append(group_record)
            projected_values = dict(values)
            projected_values.pop("statement_id", None)
            projected_values.pop("annotation_id", None)
            projected_values.pop("id", None)
            projected_values.update(
                {
                    "annotation_statement_id": statement_id,
                    "annotation_statement_group_id": group_id,
                    "annotation_kind": kind,
                }
            )
            projected = _replace_normalized(record, projected_values)
            projected["record_id"] = statement_id
            rows.append(projected)
    return groups, rows


def _extension_record(
    source_dataset: str,
    record: dict[str, Any],
    index: int,
    *,
    field_name: str = "source_record",
    value: Any | None = None,
) -> dict[str, Any]:
    values = _normalized(record)
    source_id = record.get("record_id") or next(
        (item for key, item in values.items() if key.endswith("_id") and item),
        f"row:{index}",
    )
    extension_id = f"pboc_extension:{source_dataset}:{source_id}:{field_name}"
    extension_value = value if value is not None else values
    if isinstance(extension_value, (dict, list)):
        extension_value = json.dumps(extension_value, ensure_ascii=False, sort_keys=True)
    normalized = {
        "pboc_extension_field_id": extension_id,
        "namespace": "docmirror.legacy.v1.2",
        "source_dataset": source_dataset,
        "source_record_id": str(source_id),
        "field_name": field_name,
        "value": extension_value,
    }
    projected = _replace_normalized(record, normalized)
    projected["record_id"] = extension_id
    return projected


def _generic_public_records(
    records: list[dict[str, Any]], existing: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    typed: dict[str, list[dict[str, Any]]] = {}
    extensions: list[dict[str, Any]] = []
    existing_source_ids = {
        str(_normalized(row).get("public_record_id") or "")
        for rows in existing.values()
        for row in rows
    }
    for index, record in enumerate(records, start=1):
        values = _normalized(record)
        record_type = str(values.get("record_type") or "")
        target = _PUBLIC_RECORD_TARGETS.get(record_type)
        public_id = str(values.get("public_record_id") or record.get("record_id") or "")
        if target and public_id not in existing_source_ids:
            content = values.get("content")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except (TypeError, ValueError):
                    parsed = {"unmapped_content": content}
            else:
                parsed = content if isinstance(content, dict) else {}
            typed_values = dict(parsed)
            typed_values.setdefault("public_record_id", public_id)
            typed_values.setdefault("sequence", values.get("sequence"))
            typed.setdefault(target, []).append(_replace_normalized(record, typed_values))
        elif not target:
            extensions.append(_extension_record("public_records", record, index))
    return typed, extensions


def _status_sources_by_target() -> dict[str, tuple[str, ...]]:
    sources = {new: (old,) for old, new in _DIRECT_DATASET_RENAMES.items()}
    sources.update(
        {
            "report_metadata": ("personal_report_metadata",),
            "report_query": ("personal_report_metadata",),
            "credit_accounts": ("credit_accounts",),
            "credit_account_monthly_performance": ("repayment_records",),
            "credit_business_overview": (
                "personal_detail_credit_summary_metrics",
                "personal_detail_summary_records",
                "personal_detail_summary_cells",
            ),
            "credit_account_latest_repayments": ("personal_detail_account_events",),
            "credit_account_special_transactions": ("personal_detail_account_events",),
            "credit_account_special_events": ("personal_detail_account_events",),
            "credit_card_large_installments": ("personal_detail_account_events",),
            "repayment_responsibilities": ("repayment_liability_records",),
            "annotation_statement_groups": ("statements", "annotations"),
            "annotation_statements": ("statements", "annotations"),
            "pboc_extension_fields": (
                "personal_detail_account_events",
                "personal_detail_summary_cells",
                "public_records",
            ),
        }
    )
    for target in _PUBLIC_STATUS_TARGETS:
        legacy_sources = sources.get(target, ())
        sources[target] = tuple(dict.fromkeys((*legacy_sources, "public_records")))
    return sources


def _project_dataset_status(
    status_rows: list[dict[str, Any]],
    projected: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build one schema-native status row per v2 business dataset."""
    indexed: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for record in status_rows:
        if not isinstance(record, dict):
            continue
        values = _normalized(record)
        source_name = str(values.get("dataset_name") or "")
        if source_name:
            indexed.setdefault(source_name, []).append((record, values))

    source_mapping = _status_sources_by_target()
    result: list[dict[str, Any]] = []
    for target in PBOC_DATASET_ORDER:
        if target == "dataset_status":
            continue
        source_names = source_mapping.get(target, ())
        source_rows = [item for name in source_names for item in indexed.get(name, ())]
        source_values = [values for _record, values in source_rows]
        observed_count = len(projected.get(target, ()))
        source_presence = {
            str(values.get("presence_status") or "") for values in source_values
        }
        if observed_count:
            presence_status = "observed_nonempty"
            reason = "records_projected"
        elif "extraction_failed" in source_presence:
            presence_status = "extraction_failed"
            reason = "source_extraction_failed"
        elif "partial" in source_presence:
            presence_status = "partial"
            reason = "source_partially_observed"
        elif source_presence and source_presence <= {"not_applicable"}:
            presence_status = "not_applicable"
            reason = "source_not_applicable"
        elif source_presence and source_presence <= {"explicitly_empty"}:
            presence_status = "explicitly_empty"
            reason = "source_explicitly_empty"
        elif "unknown" in source_presence:
            presence_status = "unknown"
            reason = "source_status_unknown"
        elif source_values:
            presence_status = "not_observed"
            reason = "no_records_for_projected_dataset"
        else:
            presence_status = "not_observed"
            reason = "no_source_status_mapping"

        status_id = f"dataset_status:{target}"
        normalized: dict[str, Any] = {
            "dataset_status_record_id": status_id,
            "dataset_name": target,
            "applicability": (
                "not_applicable" if presence_status == "not_applicable" else "applicable"
            ),
            "presence_status": presence_status,
            "observed_row_count": observed_count,
            "reason": reason,
        }
        if source_names:
            normalized["source_dataset_name"] = ",".join(source_names)
        source_statements = tuple(
            dict.fromkeys(
                str(values.get("source_statement") or "").strip()
                for values in source_values
                if str(values.get("source_statement") or "").strip()
            )
        )
        if source_statements:
            normalized["source_statement"] = " | ".join(source_statements)
        confidences = [
            float(values["confidence"])
            for values in source_values
            if isinstance(values.get("confidence"), (int, float))
            and not isinstance(values.get("confidence"), bool)
        ]
        if confidences:
            normalized["confidence"] = max(0.0, min(1.0, min(confidences)))

        template = source_rows[0][0] if source_rows else {}
        row = _replace_normalized(template, normalized)
        row["record_id"] = status_id
        result.append(row)
    return result


def project_personal_detail_v2_datasets(
    datasets: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Return only PBOC v2 datasets while retaining unmapped business values."""
    source = {name: list(rows or []) for name, rows in datasets.items()}
    projected: dict[str, list[dict[str, Any]]] = {}

    metadata = source.get("personal_report_metadata") or []
    if metadata:
        projected["report_metadata"] = _project_records(metadata, _report_metadata)
        projected["report_query"] = _project_records(metadata, _report_query)

    direct_transforms: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "personal_profile": _subject_profile,
        "credit_lines": _credit_agreement,
        "postpaid_payment_history": _monthly_postpaid,
        "personal_housing_fund_records": _housing_fund,
        "administrative_penalty_records": lambda values: _month_precision_public(
            values, (("effective_date", "effective_month"), ("end_date", "end_month"))
        ),
        "professional_qualification_records": lambda values: _month_precision_public(
            values,
            (
                ("obtained_date", "obtained_month"),
                ("expiry_date", "expiry_month"),
                ("revocation_date", "revocation_month"),
            ),
        ),
        "award_records": lambda values: _month_precision_public(
            values, (("effective_date", "effective_month"), ("end_date", "end_month"))
        ),
    }
    for old_name, new_name in _DIRECT_DATASET_RENAMES.items():
        rows = source.get(old_name) or []
        if rows:
            projected[new_name] = _project_records(rows, direct_transforms.get(old_name))

    accounts = source.get("credit_accounts") or []
    if accounts:
        projected["credit_accounts"] = _project_records(accounts, _credit_account)
    account_types = {
        str(_normalized(record).get("account_id") or ""): str(
            _normalized(record).get("account_type") or ""
        )
        for record in accounts
    }
    repayment = source.get("repayment_records") or []
    if repayment:
        projected["credit_account_monthly_performance"] = _project_records(
            repayment, lambda values: _monthly_performance(values, account_types)
        )

    responsibilities = source.get("repayment_liability_records") or []
    if responsibilities:
        projected["repayment_responsibilities"] = _project_records(
            responsibilities, _repayment_responsibility
        )

    metrics = source.get("personal_detail_credit_summary_metrics") or []
    if metrics:
        projected["credit_business_overview"] = _project_records(metrics, _summary_metric)

    extensions: list[dict[str, Any]] = []
    for index, record in enumerate(source.get("personal_detail_account_events") or [], start=1):
        target = _event_target(_normalized(record))
        if target is None:
            extensions.append(
                _extension_record(
                    "personal_detail_account_events",
                    record,
                    index,
                )
            )
            continue
        projected.setdefault(target, []).append(
            _replace_normalized(record, _account_event(_normalized(record), target))
        )

    groups, notes = _annotation_records(
        source.get("statements") or [], source.get("annotations") or []
    )
    if groups:
        projected["annotation_statement_groups"] = groups
        projected["annotation_statements"] = notes

    typed_public, public_extensions = _generic_public_records(
        source.get("public_records") or [], projected
    )
    for name, rows in typed_public.items():
        projected.setdefault(name, []).extend(rows)

    extensions.extend(public_extensions)
    mapped_metric_cells = {
        (
            _normalized(row).get("summary_record_id"),
            _normalized(row).get("row_index"),
            _normalized(row).get("column_index"),
        )
        for row in metrics
    }
    for index, record in enumerate(source.get("personal_detail_summary_cells") or [], start=1):
        values = _normalized(record)
        identity = (
            values.get("summary_record_id"),
            values.get("row_index"),
            values.get("column_index"),
        )
        if identity not in mapped_metric_cells:
            extensions.append(
                _extension_record(
                    "personal_detail_summary_cells",
                    record,
                    index,
                    field_name=str(values.get("column_label") or "value"),
                    value=values.get("value"),
                )
            )
    for index, record in enumerate(source.get("personal_detail_extraction_issues") or [], start=1):
        values = _normalized(record)
        extensions.append(
            _extension_record(
                "personal_detail_extraction_issues",
                record,
                index,
                field_name=str(values.get("issue_code") or "extraction_issue"),
                value=values,
            )
        )
    if extensions:
        projected["pboc_extension_fields"] = extensions

    status_rows = source.get("personal_detail_dataset_status") or []
    if status_rows:
        projected["dataset_status"] = _project_dataset_status(status_rows, projected)

    return {
        name: projected[name]
        for name in PBOC_DATASET_ORDER
        if name in projected and projected[name]
    }


def personal_detail_v2_enabled() -> bool:
    """Return whether this process opted into the shadow/canary v2 projection."""
    return os.environ.get(PBOC_SCHEMA_VERSION_ENV, "1.2.0").strip() == PBOC_SCHEMA_VERSION


def _descriptor(
    label: str,
    type_: str = "string",
    *,
    logical_type: str | None = None,
    enum_ref: str | None = None,
    unit: str | None = None,
    sensitive: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {"label": label, "type": type_}
    if logical_type:
        value["logical_type"] = logical_type
    if enum_ref:
        value["enum_ref"] = enum_ref
    if unit:
        value["unit"] = unit
    if sensitive:
        value["sensitive"] = True
    return value


def personal_detail_v2_data_dictionary() -> dict[str, Any]:
    """Return the PBOC-native dataset and logical-type dictionary."""
    from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
        personal_detail_data_dictionary,
    )

    legacy = personal_detail_data_dictionary()
    legacy_datasets = legacy.get("datasets") or {}
    datasets: dict[str, Any] = {}
    for old_name, new_name in _DIRECT_DATASET_RENAMES.items():
        if old_name in legacy_datasets:
            datasets[new_name] = deepcopy(legacy_datasets[old_name])
    for unchanged in ("credit_accounts",):
        datasets[unchanged] = deepcopy(legacy_datasets[unchanged])

    datasets["report_metadata"] = deepcopy(legacy_datasets["personal_report_metadata"])
    datasets["report_metadata"]["columns"].pop("personal_report_metadata_id", None)
    datasets["report_metadata"]["columns"].pop("query_institution", None)
    datasets["report_metadata"]["columns"].pop("query_reason", None)
    datasets["report_metadata"]["columns"]["report_metadata_id"] = _descriptor("报告元数据ID")
    datasets["report_query"] = {
        "definition": "One row for the query that produced the report.",
        "columns": {
            "report_query_id": _descriptor("报告查询ID"),
            "report_number": _descriptor("报告编号", "long_id", sensitive=True),
            "report_time": _descriptor("报告时间", "datetime"),
            "subject_name": _descriptor("姓名"),
            "primary_id_type": _descriptor("证件类型"),
            "primary_id_number": _descriptor("证件号码", "long_id", sensitive=True),
            "query_institution": _descriptor("查询机构"),
            "query_reason": _descriptor("查询原因"),
        },
    }
    datasets["subject_profile"]["columns"].pop("personal_profile_id", None)
    datasets["subject_profile"]["columns"]["subject_profile_id"] = _descriptor("主体资料ID")
    datasets["credit_accounts"]["columns"].update(
        {
            "pboc_account_type_code": _descriptor(
                "人行账户类型代码", "enum", enum_ref="pboc_account_type_code"
            ),
            "pboc_account_type_label": _descriptor("人行账户类型名称"),
        }
    )
    agreement_columns = datasets["credit_agreements"]["columns"]
    agreement_columns.pop("credit_line_id", None)
    agreement_columns.pop("total_limit", None)
    agreement_columns["credit_agreement_id"] = _descriptor("授信协议ID")
    agreement_columns["facility_limit"] = _descriptor(
        "授信额度", "money", unit="yuan"
    )
    datasets["credit_account_monthly_performance"] = deepcopy(legacy_datasets["repayment_records"])
    monthly_columns = datasets["credit_account_monthly_performance"]["columns"]
    for key in ("repayment_id", "year", "month", "status", "overdue_amount"):
        monthly_columns.pop(key, None)
    monthly_columns.update(
        {
            "monthly_performance_id": _descriptor("月度表现记录ID"),
            "performance_month": _descriptor("表现月份", logical_type="Month"),
            "status_code": _descriptor("还款状态代码", "enum", enum_ref="repayment_status_code"),
            "status_amount": _descriptor("状态对应金额", "money", unit="yuan"),
            "status_amount_semantics": _descriptor(
                "状态金额含义", "enum", enum_ref="status_amount_semantics"
            ),
            "reporting_amount_currency": _descriptor("报告金额币种", "enum", enum_ref="currency_code"),
            "reporting_amount_unit": _descriptor("报告金额单位"),
        }
    )
    datasets["repayment_responsibilities"] = deepcopy(legacy_datasets["repayment_liability_records"])
    responsibility_columns = datasets["repayment_responsibilities"]["columns"]
    responsibility_columns.pop("liability_id", None)
    responsibility_columns.pop("overdue_months_or_repayment_status", None)
    responsibility_columns.update(
        {
            "repayment_responsibility_id": _descriptor("相关还款责任ID"),
            "related_party_category": _descriptor(
                "主业务借款人类别", "enum", enum_ref="party_category"
            ),
            "overdue_months": _descriptor("逾期月数", "integer", logical_type="Short"),
            "repayment_status_code": _descriptor("还款状态代码", "enum", enum_ref="repayment_status_code"),
            "source_status_value": _descriptor("源逾期月数或还款状态"),
        }
    )
    datasets["postpaid_monthly_performance"] = deepcopy(legacy_datasets["postpaid_payment_history"])
    postpaid_columns = datasets["postpaid_monthly_performance"]["columns"]
    for key in ("postpaid_payment_history_id", "year", "month", "status"):
        postpaid_columns.pop(key, None)
    postpaid_columns.update(
        {
            "postpaid_monthly_performance_id": _descriptor("后付费月度表现ID"),
            "performance_month": _descriptor("表现月份", logical_type="Month"),
            "status_code": _descriptor("缴费状态代码", "enum", enum_ref="postpaid_payment_status_code"),
        }
    )
    datasets["credit_business_overview"] = deepcopy(
        legacy_datasets["personal_detail_credit_summary_metrics"]
    )
    for legacy_key in (
        "credit_summary_metric_id",
        "row_dimension_name",
        "row_dimension_value",
    ):
        datasets["credit_business_overview"]["columns"].pop(legacy_key, None)
    datasets["credit_business_overview"]["columns"].update(
        {
            "credit_business_overview_id": _descriptor("信用业务概要ID"),
            "business_category": _descriptor("业务类别"),
            "business_type": _descriptor("业务类型"),
            "business_dimension_name": _descriptor("业务维度名称"),
            "business_dimension_value": _descriptor("业务维度值"),
        }
    )

    event_source = legacy_datasets["personal_detail_account_events"]
    for name, label, id_name in (
        ("credit_account_latest_repayments", "最近还款记录", "latest_repayment_id"),
        ("credit_account_special_transactions", "特殊交易记录", "special_transaction_id"),
        ("credit_account_special_events", "特殊事件记录", "special_event_id"),
        ("credit_card_large_installments", "大额专项分期记录", "large_installment_id"),
    ):
        datasets[name] = deepcopy(event_source)
        datasets[name]["definition"] = f"One row per {label}."
        datasets[name]["columns"].pop("account_event_id", None)
        datasets[name]["columns"][id_name] = _descriptor(f"{label}ID")

    datasets["annotation_statement_groups"] = {
        "definition": "One row per statement/annotation object and its optional business target.",
        "columns": {
            "annotation_statement_group_id": _descriptor("说明标注组ID"),
            "annotation_kind": _descriptor("说明标注类别", "enum", enum_ref="annotation_kind"),
            "target_dataset": _descriptor("关联业务数据集"),
            "target_record_id": _descriptor("关联业务记录ID"),
        },
    }
    note_columns = deepcopy(legacy_datasets["statements"]["columns"])
    note_columns.pop("id", None)
    note_columns.update(
        {
            "annotation_statement_id": _descriptor("说明标注ID"),
            "annotation_statement_group_id": _descriptor("说明标注组ID"),
            "annotation_kind": _descriptor("说明标注类别", "enum", enum_ref="annotation_kind"),
        }
    )
    datasets["annotation_statements"] = {
        "definition": "One row per institution statement, subject statement, or dispute annotation.",
        "columns": note_columns,
    }
    datasets["pboc_extension_fields"] = {
        "definition": "Lossless scalar fallback for PBOC business fields not yet represented by a typed v2 column.",
        "columns": {
            "pboc_extension_field_id": _descriptor("人行业务扩展字段ID"),
            "namespace": _descriptor("扩展命名空间"),
            "source_dataset": _descriptor("源数据集"),
            "source_record_id": _descriptor("源记录ID"),
            "field_name": _descriptor("字段名称"),
            "value": _descriptor("字段值", "text"),
        },
    }
    datasets["dataset_status"] = deepcopy(legacy_datasets["personal_detail_dataset_status"])
    datasets["dataset_status"]["columns"].pop("dataset_status_id", None)
    datasets["dataset_status"]["columns"].update(
        {
            "dataset_status_record_id": _descriptor("数据集状态ID"),
            "source_dataset_name": _descriptor("迁移前数据集名称"),
        }
    )

    public_fixes = {
        "housing_fund_records": {
            "personal_contribution_ratio_percent": _descriptor(
                "个人缴存比例", "integer", logical_type="Short"
            ),
            "employer_contribution_ratio_percent": _descriptor(
                "单位缴存比例", "integer", logical_type="Short"
            ),
        },
        "administrative_penalty_records": {
            "effective_month": _descriptor("处罚生效月份", logical_type="Month"),
            "end_month": _descriptor("处罚截止月份", logical_type="Month"),
        },
        "professional_qualification_records": {
            "obtained_month": _descriptor("资格取得月份", logical_type="Month"),
            "expiry_month": _descriptor("资格到期月份", logical_type="Month"),
            "revocation_month": _descriptor("资格吊销月份", logical_type="Month"),
        },
        "administrative_award_records": {
            "effective_month": _descriptor("奖励生效月份", logical_type="Month"),
            "end_month": _descriptor("奖励截止月份", logical_type="Month"),
        },
    }
    for name, columns in public_fixes.items():
        datasets[name]["columns"].update(columns)
    for legacy_key in ("personal_housing_fund_id", "personal_contribution_ratio", "employer_contribution_ratio"):
        datasets["housing_fund_records"]["columns"].pop(legacy_key, None)
    for legacy_key in ("effective_date", "end_date"):
        datasets["administrative_penalty_records"]["columns"].pop(legacy_key, None)
        datasets["administrative_award_records"]["columns"].pop(legacy_key, None)
    for legacy_key in ("obtained_date", "expiry_date", "revocation_date"):
        datasets["professional_qualification_records"]["columns"].pop(legacy_key, None)

    catalog = {
        "fraud_warnings": "欺诈警示",
        "dispute_overview": "异议概要",
        "credit_scores": "信用评分",
        "credit_score_reasons": "评分原因",
        "bad_debt_summary": "呆账概要",
        "recovery_summary": "被追偿概要",
        "delinquency_summary": "逾期透支概要",
        "account_portfolio_summary": "账户构成概要",
        "repayment_responsibility_summary": "相关还款责任概要",
        "postpaid_arrears_summary": "后付费欠费概要",
        "public_negative_summary": "公共负面信息概要",
        "inquiry_summary": "查询概要",
        "credit_account_snapshots": "账户时点信息",
        "credit_account_history_windows": "账户历史展示区间",
        "account_credit_agreement_links": "账户授信协议关联",
        "postpaid_history_windows": "后付费历史展示区间",
        "social_assistance_records": "社会救助记录",
    }
    for name, label in catalog.items():
        datasets.setdefault(
            name,
            {
                "definition": f"PBOC-native {label} records.",
                "columns": {
                    f"{name.removesuffix('s')}_id": _descriptor(f"{label}ID"),
                },
            },
        )

    return {
        "schema_id": PBOC_SCHEMA_ID,
        "version": PBOC_SCHEMA_VERSION,
        "standard": {
            "issuer": "People's Bank of China Credit Reference Center",
            "reference": "Q/PBCCRC 2.1-2016 (2019 revision)",
        },
        "definitions": {
            "authoritative_business_records": "datasets[*].rows[*].normalized",
            "canonical_record_identity": "record_id",
            "missing_value_policy": "Use null or omit the field; retain source sentinels only in canonical_raw/raw.",
            "month_policy": "PBOC Month uses lexical YYYY-MM; Date uses YYYY-MM-DD.",
            "amount_policy": "Money uses decimal strings with per-field currency/unit context; report amounts default to CNY/yuan.",
            "integer_policy": "PBOC Long is serialized as a decimal string; Short is a JSON integer.",
            "enum_policy": "Code and label are separate fields; a code may be null when only a displayed label is available.",
            "extension_policy": "pboc_extension_fields is a lossless fallback, not a replacement for a known typed field.",
        },
        "logical_types": {
            "N": {"json_type": "string", "pattern": "^[0-9]+$"},
            "AN": {"json_type": "string"},
            "ANC": {"json_type": "string"},
            "Year": {"json_type": "string", "pattern": "^[0-9]{4}$"},
            "Month": {"json_type": "string", "pattern": "^[0-9]{4}-(0[1-9]|1[0-2])$"},
            "Date": {"json_type": "string", "pattern": "^[0-9]{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])$"},
            "Time": {"json_type": "string"},
            "Short": {"json_type": "integer"},
            "Long": {"json_type": "string", "pattern": "^-?[0-9]+$"},
            "Float": {"json_type": "string", "pattern": "^-?[0-9]+(?:\\.[0-9]+)?$"},
        },
        "fields": deepcopy(legacy.get("fields") or {}),
        "datasets": datasets,
        "enums": {
            **deepcopy(legacy.get("enums") or {}),
            "pboc_account_type_code": {
                "D1": "非循环贷账户",
                "R1": "循环贷账户（一）",
                "R2": "循环贷账户（二）",
                "R3": "贷记卡账户",
                "R4": "准贷记卡账户",
                "C1": "被追偿信息",
            },
            "status_amount_semantics": {
                "delinquent_amount": "逾期金额",
                "overdraft_balance": "透支余额",
            },
            "party_category": {
                "person": "自然人",
                "organization": "组织机构",
                "unknown": "未确定",
            },
            "annotation_kind": {
                "statement": "机构说明或本人声明",
                "annotation": "异议标注",
            },
        },
    }


def personal_detail_v2_semantic_extensions() -> dict[str, Any]:
    """Return v2 contract identity and Community presentation policy."""
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_semantic_extensions,
    )

    semantic = credit_report_semantic_extensions(report_subtype="personal_detail")
    semantic["domain_schema"] = {
        "id": PBOC_SCHEMA_ID,
        "version": PBOC_SCHEMA_VERSION,
        "contract_uri": PBOC_CONTRACT_URI,
        "compatibility": "breaking-from-1.2; community-v3-envelope; detailed-report-only",
    }
    semantic["dataset_document_order"] = list(PBOC_DATASET_ORDER)
    semantic["dataset_reading_columns"] = {
        "report_metadata": ["report_number", "report_time", "subject_name"],
        "report_query": ["query_institution", "query_reason"],
        "subject_profile": ["subject_name", "gender", "birth_date", "marital_status"],
        "credit_accounts": [
            "pboc_account_type_code",
            "account_identifier",
            "institution",
            "business_type",
            "status",
            "balance",
        ],
        "credit_account_monthly_performance": [
            "account_id",
            "performance_month",
            "status_code",
            "status_amount",
            "status_amount_semantics",
        ],
        "repayment_responsibilities": [
            "related_party_name",
            "related_party_category",
            "responsibility_type",
            "responsibility_amount",
            "overdue_months",
            "repayment_status_code",
        ],
        "annotation_statements": ["annotation_kind", "text", "added_date"],
        "inquiries": ["inquiry_date", "institution", "reason"],
    }
    dictionary = personal_detail_v2_data_dictionary()
    foreign_keys = {
        name: [
            {
                "columns": ["account_id"],
                "reference_dataset": "credit_accounts",
                "reference_columns": ["record_id"],
            }
        ]
        for name in (
            "credit_account_snapshots",
            "credit_account_history_windows",
            "credit_account_monthly_performance",
            "credit_account_latest_repayments",
            "credit_account_special_transactions",
            "credit_account_special_events",
            "credit_card_large_installments",
        )
    }
    foreign_keys["annotation_statements"] = [
        {
            "columns": ["annotation_statement_group_id"],
            "reference_dataset": "annotation_statement_groups",
            "reference_columns": ["annotation_statement_group_id"],
        }
    ]
    foreign_keys["account_credit_agreement_links"] = [
        {
            "columns": ["account_id"],
            "reference_dataset": "credit_accounts",
            "reference_columns": ["record_id"],
        },
        {
            "columns": ["credit_agreement_id"],
            "reference_dataset": "credit_agreements",
            "reference_columns": ["record_id"],
        },
    ]
    semantic["community_projection_overrides"] = {
        "dataset_labels": dict(_PBOC_DATASET_LABELS),
        "dataset_representation_roles": {
            name: ("control" if name == "dataset_status" else "business_canonical")
            for name in PBOC_DATASET_ORDER
        },
        "dataset_grains": {
            name: f"one row per {_PBOC_DATASET_LABELS[name]}"
            for name in PBOC_DATASET_ORDER
        },
        "dataset_foreign_keys": foreign_keys,
        "dataset_derived_from": {
            "report_query": ["report_metadata"],
            "credit_business_overview": ["personal_detail_credit_summary_metrics"],
            "pboc_extension_fields": ["unmapped_source_business_fields"],
            "dataset_status": ["final_v2_business_datasets"],
        },
        "section_markers": {
            **{
                name: ["basic_information"]
                for name in PBOC_DATASET_ORDER[:10]
            },
            **{
                name: ["credit_summary"]
                for name in PBOC_DATASET_ORDER[10:21]
            },
            **{
                name: ["credit_details"]
                for name in PBOC_DATASET_ORDER[21:34]
            },
            **{
                name: ["public_records"]
                for name in PBOC_DATASET_ORDER[34:42]
            },
            "annotation_statement_groups": ["statements", "annotations"],
            "annotation_statements": ["statements", "annotations"],
            "inquiries": ["inquiries"],
            "pboc_extension_fields": ["extraction_review"],
            "dataset_status": ["extraction_review"],
        },
    }
    assert set(semantic["dataset_reading_columns"]) <= set(dictionary["datasets"])
    return semantic


__all__ = [
    "PBOC_CONTRACT_URI",
    "PBOC_DATASET_ORDER",
    "PBOC_SCHEMA_ID",
    "PBOC_SCHEMA_VERSION",
    "PBOC_SCHEMA_VERSION_ENV",
    "personal_detail_v2_data_dictionary",
    "personal_detail_v2_enabled",
    "personal_detail_v2_semantic_extensions",
    "project_personal_detail_v2_datasets",
]
