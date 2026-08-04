# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract-only projections for personal detailed credit reports.

The functions in this module do not discover OCR text.  They turn facts and
rows already emitted by the personal-detail extractors into stable business
datasets, while preserving the distinction between an unobserved value and an
explicitly reported absence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

PERSONAL_PROFILE_FIELDS = (
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
)

PERSONAL_DETAIL_BUSINESS_DATASETS = (
    "personal_profile",
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

_PLACEHOLDERS = frozenset({"", "-", "--", "---", "—", "–", "未报告", "不详"})
_OBSERVATION_STATUSES = frozenset(
    {
        "observed",
        "normalized",
        "ocr_corrected",
        "inferred",
        "ambiguous",
        "unreadable",
        "not_observed",
        "explicitly_absent",
        "not_applicable",
    }
)
_DATASET_PRESENCE_STATUSES = frozenset(
    {
        "observed_nonempty",
        "explicitly_empty",
        "not_applicable",
        "not_observed",
        "partial",
        "extraction_failed",
        "unknown",
    }
)

_SUMMARY_CODES = {
    "信用业务概要": "credit_business_overview",
    "呆账": "bad_debt",
    "逾期（透支）": "delinquency_overdraft",
    "被追偿": "recovery",
    "非循环贷账户": "non_revolving_loan",
    "循环贷账户一": "revolving_loan_subaccount",
    "循环贷账户二": "revolving_loan_account",
    "贷记卡账户": "credit_card",
    "准贷记卡账户": "quasi_credit_card",
    "相关还款责任": "related_repayment_liability",
    "后付费业务欠费": "postpaid_arrears",
    "公共": "public_records",
    "查询记录概要": "inquiry_overview",
}

_METRIC_CODES = {
    "业务类型": "business_type",
    "账户类型": "account_type",
    "信息类型": "information_type",
    "账户数": "account_count",
    "记录数": "record_count",
    "管理机构数": "management_institution_count",
    "发卡机构数": "issuer_count",
    "月份数": "delinquent_month_count",
    "最长逾期/透支月数": "maximum_delinquency_months",
    "首笔业务发放月份": "first_business_issue_month",
    "余额": "balance",
    "授信总额": "total_credit_limit",
    "涉及金额": "involved_amount",
    "欠费金额": "arrears_amount",
    "单月最高逾期/透支总额": "maximum_monthly_delinquency_amount",
    "最近6个月平均应还款": "recent_6_month_average_payment",
    "单家机构最高授信额": "maximum_single_institution_limit",
    "单家机构最低授信额": "minimum_single_institution_limit",
    "已用额度": "used_limit",
    "最近6个月平均使用额度": "recent_6_month_average_used_limit",
    "透支余额": "overdraft_balance",
    "最近6个月平均透支余额": "recent_6_month_average_overdraft_balance",
    "为个人/担保责任/账户数": "personal_guarantee_account_count",
    "为个人/担保责任/担保金额": "personal_guarantee_amount",
    "为个人/担保责任/余额": "personal_guarantee_balance",
    "为个人/其他相关还款责任/账户数": "personal_other_liability_account_count",
    "为个人/其他相关还款责任/还款责任金额": "personal_other_liability_amount",
    "为个人/其他相关还款责任/余额": "personal_other_liability_balance",
    "为企业/担保责任/账户数": "enterprise_guarantee_account_count",
    "为企业/担保责任/担保金额": "enterprise_guarantee_amount",
    "为企业/担保责任/余额": "enterprise_guarantee_balance",
    "为企业/其他相关还款责任/账户数": "enterprise_other_liability_account_count",
    "为企业/其他相关还款责任/还款责任金额": "enterprise_other_liability_amount",
    "为企业/其他相关还款责任/余额": "enterprise_other_liability_balance",
    "最近1个月内的查询机构数/贷款审批": "recent_1m_institution_count_loan_approval",
    "最近1个月内的查询机构数/信用卡审批": "recent_1m_institution_count_credit_card_approval",
    "最近1个月内的查询次数/贷款审批": "recent_1m_inquiry_count_loan_approval",
    "最近1个月内的查询次数/信用卡审批": "recent_1m_inquiry_count_credit_card_approval",
    "最近1个月内的查询次数/本人查询": "recent_1m_inquiry_count_self",
    "最近2年内的查询次数/贷后管理": "recent_2y_inquiry_count_post_loan_management",
    "最近2年内的查询次数/担保资格审查": "recent_2y_inquiry_count_guarantee_review",
    "最近2年内的查询次数/特约商户实名审查": "recent_2y_inquiry_count_merchant_identity_review",
}

_MONEY_METRIC_CODES = frozenset(
    {
        "balance",
        "total_credit_limit",
        "involved_amount",
        "arrears_amount",
        "maximum_monthly_delinquency_amount",
        "recent_6_month_average_payment",
        "maximum_single_institution_limit",
        "minimum_single_institution_limit",
        "used_limit",
        "recent_6_month_average_used_limit",
        "overdraft_balance",
        "recent_6_month_average_overdraft_balance",
        "personal_guarantee_amount",
        "personal_guarantee_balance",
        "personal_other_liability_amount",
        "personal_other_liability_balance",
        "enterprise_guarantee_amount",
        "enterprise_guarantee_balance",
        "enterprise_other_liability_amount",
        "enterprise_other_liability_balance",
    }
)

_PUBLIC_RECORD_SPECS: dict[str, tuple[str, str, dict[str, str]]] = {
    "tax_arrears": (
        "tax_arrears_records",
        "tax_arrears_id",
        {"authority": "tax_authority"},
    ),
    "civil_judgment": (
        "civil_judgment_records",
        "civil_judgment_id",
        {
            "authority": "filing_court",
            "start_date": "filing_date",
            "effective_date": "judgment_effective_date",
        },
    ),
    "enforcement": (
        "enforcement_records",
        "enforcement_record_id",
        {"authority": "court", "start_date": "filing_date", "end_date": "closure_date"},
    ),
    "administrative_penalty": (
        "administrative_penalty_records",
        "administrative_penalty_id",
        {"authority": "authority", "start_date": "effective_date", "end_date": "end_date"},
    ),
    "housing_fund": (
        "personal_housing_fund_records",
        "personal_housing_fund_id",
        {},
    ),
    "professional_qualification": (
        "professional_qualification_records",
        "professional_qualification_id",
        {"authority": "issuing_authority"},
    ),
    "award": (
        "award_records",
        "award_record_id",
        {"authority": "authority", "start_date": "effective_date", "end_date": "end_date"},
    ),
}


def _record_values(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    normalized = record.get("normalized")
    return dict(normalized) if isinstance(normalized, Mapping) else dict(record)


def _source_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    return [dict(ref) for ref in (value.get("source_refs") or []) if isinstance(ref, Mapping)]


def _dedupe_source_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for ref in refs:
        marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(ref)
    return result


def _compact_label(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _iso_date(value: Any) -> Any:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match = re.fullmatch(r"(\d{4})[-./](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    return value


def _reporting_context(datasets: Mapping[str, Any]) -> tuple[str | None, str | None]:
    rows = datasets.get("personal_report_metadata") or []
    metadata = _record_values(rows[0]) if rows else {}
    # The personal detailed-report contract declares report-level converted
    # amounts in renminbi yuan. Metadata may be OCR-missed, but that document
    # rule remains valid for summary and public-record projections.
    currency = str(metadata.get("reporting_currency") or "CNY").strip() or "CNY"
    amount_unit = str(metadata.get("reporting_amount_unit") or "yuan").strip() or "yuan"
    return currency, amount_unit


def _profile_contract(
    facts: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_profile = facts.get("subject_profile")
    profile = dict(source_profile) if isinstance(source_profile, Mapping) else {}
    metadata_rows = datasets.get("personal_report_metadata") or []
    metadata = _record_values(metadata_rows[0]) if metadata_rows else {}
    profile_id = "personal_profile:1"
    row: dict[str, Any] = {
        "record_id": profile_id,
        "personal_profile_id": profile_id,
    }
    for target, candidates in {
        "subject_name": ("subject_name",),
        "primary_id_type": ("primary_id_type", "id_type"),
        "primary_id_number": ("primary_id_number", "id_number"),
    }.items():
        value = None
        for name in candidates:
            candidate = metadata.get(name) or facts.get(name) or profile.get(name)
            if isinstance(candidate, Mapping):
                candidate = candidate.get("normalized_value", candidate.get("value"))
            if candidate not in (None, ""):
                value = candidate
                break
        if value not in (None, ""):
            row[target] = value

    observations: list[dict[str, Any]] = []
    all_refs: list[dict[str, Any]] = []
    for field_name in PERSONAL_PROFILE_FIELDS:
        entry = profile.get(field_name)
        entry_map = dict(entry) if isinstance(entry, Mapping) else {}
        normalized = (
            entry_map.get("normalized_value", entry_map.get("value"))
            if entry_map
            else entry
        )
        raw = entry_map.get("canonical_raw", entry_map.get("raw", normalized)) if entry_map else entry
        if field_name == "birth_date" and normalized not in (None, ""):
            normalized = _iso_date(normalized)
        refs = _source_refs(entry_map)
        all_refs.extend(refs)
        explicit_status = str(entry_map.get("observation_status") or "")
        if explicit_status in _OBSERVATION_STATUSES:
            status = explicit_status
        elif normalized in (None, ""):
            status = "not_observed"
        elif entry_map.get("ocr_corrected") is True:
            status = "ocr_corrected"
        elif raw not in (None, "") and str(raw) != str(normalized):
            status = "normalized"
        else:
            status = "observed"
        if normalized not in (None, ""):
            row[field_name] = normalized
        observation: dict[str, Any] = {
            "record_id": f"personal_profile_field:{field_name}",
            "field_observation_id": f"personal_profile_field:{field_name}",
            "dataset_name": "personal_profile",
            "business_record_id": profile_id,
            "field_name": field_name,
            "observation_status": status,
        }
        if raw not in (None, ""):
            observation["raw_value"] = raw
        if normalized not in (None, ""):
            observation["normalized_value"] = normalized
        confidence = entry_map.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            observation["confidence"] = max(0.0, min(1.0, float(confidence)))
            observation["confidence_status"] = "available"
            observation["confidence_basis"] = str(
                entry_map.get("confidence_basis") or "source_field_confidence"
            )
        else:
            observation["confidence_status"] = "not_available"
            observation["confidence_basis"] = "source_did_not_report_field_confidence"
        if status == "not_observed":
            observation["reason"] = "no_field_observation_emitted"
        elif entry_map.get("reason"):
            observation["reason"] = str(entry_map["reason"])
        if refs:
            observation["source_refs"] = refs
        observations.append(observation)
    if all_refs:
        row["source_refs"] = _dedupe_source_refs(all_refs)
    return [row], observations


def _summary_value(value: Any) -> tuple[str, str | None, str]:
    text = str(value or "").strip()
    if text in _PLACEHOLDERS:
        return "placeholder", None, "not_reported"
    compact = text.replace(",", "")
    if re.fullmatch(r"[-+]?\d+", compact):
        return "integer", compact, "reported"
    if re.fullmatch(r"\d{4}[-./]\d{1,2}(?:[-./]\d{1,2})?", compact):
        return "date", None, "reported"
    if re.fullmatch(r"[-+]?\d+\.\d+", compact):
        return "decimal", compact, "reported"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%", compact):
        return "percentage", compact[:-1], "reported"
    return "text", None, "reported"


def _summary_metric_contract(datasets: Mapping[str, Any]) -> list[dict[str, Any]]:
    reporting_currency, reporting_amount_unit = _reporting_context(datasets)
    cells = [_record_values(row) for row in (datasets.get("personal_detail_summary_cells") or [])]
    summary_rows = {
        str(values.get("summary_record_id") or ""): values
        for row in (datasets.get("personal_detail_summary_records") or [])
        if (values := _record_values(row)).get("summary_record_id")
    }
    dimensions: dict[tuple[str, int], tuple[str, str]] = {}
    for cell in sorted(cells, key=lambda item: (str(item.get("summary_record_id") or ""), int(item.get("row_index") or 0), int(item.get("column_index") or 0))):
        key = (str(cell.get("summary_record_id") or ""), int(cell.get("row_index") or 0))
        if key not in dimensions and str(cell.get("value") or "").strip() not in _PLACEHOLDERS:
            dimensions[key] = (
                str(cell.get("column_label") or "").strip(),
                str(cell.get("value") or "").strip(),
            )

    metrics: list[dict[str, Any]] = []
    for index, source_row in enumerate(datasets.get("personal_detail_summary_cells") or [], start=1):
        cell = _record_values(source_row)
        summary_record_id = str(cell.get("summary_record_id") or "")
        row_index = int(cell.get("row_index") or 0)
        column_index = int(cell.get("column_index") or 0)
        source_value = cell.get("value")
        value_type, numeric_value, reporting_status = _summary_value(source_value)
        source_id = str(cell.get("summary_cell_id") or f"summary_cell:{index}")
        metric_id = f"credit_summary_metric:{source_id}"
        dimension_name, dimension_value = dimensions.get((summary_record_id, row_index), ("", ""))
        summary_type = str(cell.get("summary_type") or "").strip()
        metric_name = str(cell.get("column_label") or "").strip()
        summary_code = _SUMMARY_CODES.get(summary_type)
        metric_code = _METRIC_CODES.get(_compact_label(metric_name))
        if metric_code in _MONEY_METRIC_CODES and reporting_status == "reported":
            value_type = "money"
        metric: dict[str, Any] = {
            "record_id": metric_id,
            "credit_summary_metric_id": metric_id,
            "summary_record_id": summary_record_id,
            "summary_type": summary_type,
            "summary_code": summary_code,
            "title": cell.get("title"),
            "row_index": row_index,
            "column_index": column_index,
            "metric_name": metric_name,
            "metric_code": metric_code,
            "mapping_status": "mapped" if summary_code and metric_code else "unmapped",
            "row_dimension_name": dimension_name,
            "row_dimension_value": dimension_value,
            "source_value": source_value,
            "value_type": value_type,
            "reporting_status": reporting_status,
        }
        if any(marker in dimension_name for marker in ("账户类型", "业务类型", "业务类别", "责任类型")):
            metric["business_category"] = dimension_value
        parent = summary_rows.get(summary_record_id, {})
        if parent.get("source_table_id"):
            metric["source_table_id"] = parent["source_table_id"]
        if numeric_value is not None:
            metric["numeric_value"] = numeric_value
        elif value_type == "date" and source_value not in (None, ""):
            metric["date_value"] = _iso_date(source_value)
        elif reporting_status == "reported" and source_value not in (None, ""):
            metric["text_value"] = source_value
        if value_type == "money":
            metric["currency"] = reporting_currency
            metric["amount_unit"] = reporting_amount_unit
        refs = _source_refs(source_row)
        if refs:
            metric["source_refs"] = refs
        metrics.append({key: value for key, value in metric.items() if value not in (None, "")})
    return metrics


def _content_object(content: Any) -> dict[str, Any]:
    if isinstance(content, Mapping):
        return dict(content)
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        decoded = json.loads(content)
    except (TypeError, ValueError):
        return {"unmapped_content": content}
    return dict(decoded) if isinstance(decoded, Mapping) else {"unmapped_content": content}


def project_typed_public_records(
    public_records: Any,
    *,
    reporting_currency: str | None = None,
    reporting_amount_unit: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    typed: dict[str, list[dict[str, Any]]] = {spec[0]: [] for spec in _PUBLIC_RECORD_SPECS.values()}
    for index, source_row in enumerate(public_records or [], start=1):
        record = _record_values(source_row)
        record_type = str(record.get("record_type") or "")
        spec = _PUBLIC_RECORD_SPECS.get(record_type)
        if spec is None:
            continue
        dataset_name, id_field, aliases = spec
        public_record_id = str(record.get("public_record_id") or f"public_record:{index}")
        typed_id = f"{dataset_name}:{public_record_id}"
        projected: dict[str, Any] = {
            "record_id": typed_id,
            id_field: typed_id,
            "public_record_id": public_record_id,
            "sequence": record.get("sequence", index),
        }
        content = _content_object(record.get("content"))
        candidates = dict(content)
        for key in (*aliases, "reporting_amount_currency", "reporting_amount_unit"):
            target = aliases.get(key, key)
            if target not in candidates and record.get(key) not in (None, ""):
                candidates[key] = record[key]
        for key, value in candidates.items():
            target = aliases.get(key, key)
            if target in {"record_id", "content", "normalized", "canonical_raw", "raw", "source", "source_refs"}:
                continue
            if value not in (None, ""):
                projected[target] = value
        if any(key.endswith("_amount") or key == "monthly_contribution" for key in projected):
            projected.setdefault("reporting_amount_currency", reporting_currency)
            projected.setdefault("reporting_amount_unit", reporting_amount_unit)
        refs = _source_refs(source_row)
        if refs:
            projected["source_refs"] = refs
        typed[dataset_name].append(projected)
    return typed


def _dataset_status_contract(
    facts: Mapping[str, Any],
    datasets: Mapping[str, Any],
    auxiliary_business: Mapping[str, Any],
    final_dataset_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    explicit_states = facts.get("personal_detail_dataset_states")
    states = dict(explicit_states) if isinstance(explicit_states, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for dataset_name in PERSONAL_DETAIL_BUSINESS_DATASETS:
        local_rows = datasets.get(dataset_name)
        auxiliary_rows = auxiliary_business.get(dataset_name)
        final_count = final_dataset_counts.get(dataset_name)
        if isinstance(local_rows, (list, tuple)) and local_rows:
            observed_count = sum(isinstance(record, Mapping) for record in local_rows)
        elif isinstance(final_count, int) and not isinstance(final_count, bool) and final_count >= 0:
            observed_count = final_count
        elif isinstance(local_rows, (list, tuple)):
            observed_count = 0
        elif isinstance(auxiliary_rows, (list, tuple)):
            observed_count = sum(isinstance(record, Mapping) for record in auxiliary_rows)
        else:
            observed_count = 0
        extractor_count = facts.get(f"personal_detail_expected_{dataset_name}_count")
        if not observed_count and isinstance(extractor_count, int) and not isinstance(extractor_count, bool):
            observed_count = max(0, extractor_count)
        explicit = states.get(dataset_name)
        explicit_map = dict(explicit) if isinstance(explicit, Mapping) else {}
        explicit_status = str(explicit_map.get("presence_status") or "")
        if observed_count:
            presence_status = "observed_nonempty"
            reason = "records_projected"
        elif explicit_status in _DATASET_PRESENCE_STATUSES:
            presence_status = explicit_status
            reason = str(explicit_map.get("reason") or "source_state_reported")
        else:
            presence_status = "not_observed"
            reason = "no_explicit_absence_evidence"
        status_id = f"personal_detail_dataset_status:{dataset_name}"
        row: dict[str, Any] = {
            "record_id": status_id,
            "dataset_status_id": status_id,
            "dataset_name": dataset_name,
            "applicability": str(explicit_map.get("applicability") or "applicable"),
            "presence_status": presence_status,
            "observed_row_count": observed_count,
            "reason": reason,
        }
        if explicit_map.get("source_statement"):
            row["source_statement"] = str(explicit_map["source_statement"])
        confidence = explicit_map.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            row["confidence"] = max(0.0, min(1.0, float(confidence)))
        refs = _source_refs(explicit_map)
        if refs:
            row["source_refs"] = refs
        rows.append(row)
    return rows


def apply_personal_detail_contract(
    content: dict[str, Any],
    auxiliary_business: Mapping[str, Any] | None = None,
    *,
    final_dataset_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Add schema-owned views to already extracted personal-detail content."""
    auxiliary = auxiliary_business or {}
    facts = content.setdefault("facts", {})
    if not isinstance(facts, dict):
        facts = {}
        content["facts"] = facts
    datasets = content.setdefault("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}
        content["datasets"] = datasets

    profile_rows, field_observations = _profile_contract(facts, datasets)
    if profile_rows:
        datasets["personal_profile"] = profile_rows
    datasets["personal_detail_field_observations"] = field_observations

    summary_metrics = _summary_metric_contract(datasets)
    if summary_metrics:
        datasets["personal_detail_credit_summary_metrics"] = summary_metrics

    reporting_currency, reporting_amount_unit = _reporting_context(datasets)
    public_records = datasets.get("public_records") or auxiliary.get("public_records") or []
    for dataset_name, rows in project_typed_public_records(
        public_records,
        reporting_currency=reporting_currency,
        reporting_amount_unit=reporting_amount_unit,
    ).items():
        if rows:
            datasets[dataset_name] = rows

    datasets["personal_detail_dataset_status"] = _dataset_status_contract(
        facts,
        datasets,
        auxiliary,
        final_dataset_counts or {},
    )
    for dataset_name in (
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
    ):
        rows = datasets.get(dataset_name)
        if isinstance(rows, list):
            facts[f"personal_detail_expected_{dataset_name}_count"] = len(rows)
    facts["canonical_dataset_schema"] = "personal_credit_report_detailed.v1.2"
    return content


__all__ = [
    "PERSONAL_DETAIL_BUSINESS_DATASETS",
    "PERSONAL_PROFILE_FIELDS",
    "apply_personal_detail_contract",
    "project_typed_public_records",
]
