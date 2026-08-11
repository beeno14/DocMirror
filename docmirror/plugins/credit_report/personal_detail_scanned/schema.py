# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical PBOC v2 contract for personal detailed credit reports.

The extraction layer supplies private source collections. This module converts
them into the only public business model before Community JSON is assembled.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping

from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
    is_explicit_source_absence,
    validate_pboc_field,
)
from docmirror.plugins.credit_report.personal_detail_scanned.quality import (
    decode_mapping,
    header_field_valid,
    normalize_currency,
    valid_iso_date,
)

PBOC_SCHEMA_ID = "personal_credit_report_detailed"
PBOC_SCHEMA_VERSION = "2.0.0"
PBOC_CONTRACT_URI = (
    "https://valuemapglobal.github.io/DocMirror/schemas/"
    "personal_credit_report_detailed.schema.json"
)
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
    "field_observations",
    "extraction_issues",
    "extraction_issue_evidence",
    "pboc_extension_fields",
    "dataset_status",
)

_CONTROL_DATASETS = frozenset(
    {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
)

# Extension rows carry quarantined business values and therefore still need
# scalar/schema hygiene. Only diagnostic/control relations bypass the value
# gate itself.
_QUALITY_GATE_EXEMPT_DATASETS = _CONTROL_DATASETS - {"pboc_extension_fields"}

_CANONICAL_MONTHLY_STATUS_CODES = frozenset(
    {"*", "/", "#", "N", "1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "G", "M", "Z"}
)

_SPARSE_DATASET_STATUS_SEMANTICS = {
    "mode": "potentially_flawed_only",
    "present_dataset_without_status": "silently_trusted_complete",
    "absent_dataset_without_status": "silently_trusted_empty_or_not_applicable",
    "status_row_present": "partial_unknown_or_failed_extraction",
}

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
    "field_observations": "字段观测与不确定性",
    "extraction_issues": "提取问题与人工复核队列",
    "extraction_issue_evidence": "提取问题结构化证据",
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

_INTERNAL_PROJECTION_METADATA_FIELDS = frozenset(
    {
        "audit",
        "amount_bbox",
        "bbox",
        "canonical_raw",
        "column",
        "confidence",
        "evidence_ids",
        "geometry_scope",
        "normalized",
        "page",
        "raw_status",
        "raw",
        "recognition_source",
        "review",
        "row",
        "source",
        "source_cell_refs",
        "source_page",
        "source_refs",
        "source_refs_by_field",
        "status_bbox",
        "table_id",
    }
)

_SOURCE_SENTINELS = frozenset({"-", "--", "---"})

# These are canonical scalar slots for which a dash-only printed value means
# that the source explicitly supplied no value.  Keep this list finite: blank
# OCR and dash-like prose must remain uncertainty rather than being silently
# converted to absence.
_EXPLICIT_SOURCE_ABSENCE_FIELDS: dict[str, frozenset[str]] = {
    "subject_profile": frozenset({"degree", "household_address"}),
    "subject_spouse": frozenset(
        {"name", "document_type", "document_number", "employer", "phone"}
    ),
    "subject_residences": frozenset({"residential_phone"}),
    "subject_employment": frozenset(
        {
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
        }
    ),
    "credit_accounts": frozenset(
        {"due_date", "repayment_method", "shared_credit_limit"}
    ),
    "credit_agreements": frozenset(
        {
            "institution",
            "facility_type",
            "effective_date",
            "due_date",
            "facility_limit",
            "credit_limit",
            "used_limit",
            "limit_identifier",
            "currency",
        }
    ),
    "repayment_responsibilities": frozenset({"overdue_months"}),
}

# A dash in these dense tables is accepted only from the extractor's direct
# per-field ledger.  OCR ``**``, a blank cell, or a dash embedded in a collapsed
# cluster is not source absence.  The final gate additionally requires a
# dash-only raw snapshot and rejects every competing non-dash observation.
_STRICT_DIRECT_SOURCE_ABSENCE_DATASETS = frozenset(
    {"subject_employment", "credit_agreements"}
)

_MISSING_FIELD_ROW_CONTEXT_ISSUE_CODES = frozenset(
    {
        "candidate_b_employment_canonical_cell_unresolved",
        "candidate_b_employment_provider_cell_unresolved",
        "candidate_b_employment_recovered_header_cell_unresolved",
        "candidate_b_employment_required_cell_unresolved",
    }
)

_SOURCE_ABSENCE_SUPERSEDED_ISSUE_CODES = frozenset(
    {
        "candidate_b_account_cluster_field_unresolved",
        "candidate_b_credit_agreement_currency_unresolved",
        "candidate_b_credit_agreement_required_field_unresolved",
        "candidate_b_credit_limit_identifier_unresolved",
        "candidate_b_employment_canonical_cell_unresolved",
        "candidate_b_employment_cluster_field_unresolved",
        "candidate_b_employment_provider_cell_unresolved",
        "candidate_b_employment_recovered_header_cell_unresolved",
        "candidate_b_employment_required_cell_unresolved",
        "candidate_b_exact_slot_value_invalid",
        "candidate_b_profile_contract_unresolved",
        "candidate_b_repayment_responsibility_field_invalid",
        "candidate_b_repayment_responsibility_required_field_unresolved",
        "pboc_cell_contract_unresolved",
    }
)


def _only_explicit_source_absence(value: Any) -> bool:
    """Return whether evidence consists solely of nonblank dash sentinels."""

    if isinstance(value, str):
        return is_explicit_source_absence(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        return bool(items) and all(
            isinstance(item, str) and is_explicit_source_absence(item)
            for item in items
        )
    return False


def _has_competing_non_absence_observation(value: Any) -> bool:
    """Return whether retained evidence contains any nonblank non-dash value."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and not is_explicit_source_absence(value)
    if isinstance(value, Mapping):
        return any(
            _has_competing_non_absence_observation(item) for item in value.values()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_has_competing_non_absence_observation(item) for item in value)
    return True


def _issue_has_competing_non_absence_observation(values: Mapping[str, Any]) -> bool:
    """Inspect only evidence that can belong to the issue's target field.

    Missing employment-cell issues retain the entire keyed row as context.  A
    non-dash value in another column is not a competing observation of the
    target field.  Collapsed-cluster issues are intentionally excluded: their
    raw cluster is the target-field evidence and must veto source absence.
    """

    observed = values.get("observed_value")
    if (
        str(values.get("issue_code") or "")
        in _MISSING_FIELD_ROW_CONTEXT_ISSUE_CODES
        and isinstance(observed, Mapping)
        and "physical_cells" in observed
    ):
        observed_competes = False
    else:
        observed_competes = _has_competing_non_absence_observation(observed)
    return observed_competes or _has_competing_non_absence_observation(
        values.get("candidate_value")
    )

_BUSINESS_RECORD_ENVELOPE_FIELDS = frozenset(
    {
        "canonical_raw",
        "confidence",
        "evidence_ids",
        "normalized",
        "raw",
        "record_id",
        "review",
        "source",
        "source_cell_refs",
        "source_refs",
    }
)


def _without_internal_projection_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep Community business rows free of plugin-only OCR/topology state.

    Potentially flawed values remain machine-readable through typed status
    fields plus ``field_observations``, ``extraction_issues``, and
    ``extraction_issue_evidence``.  Ad-hoc audit JSON and geometry therefore do
    not belong in the normalized/raw business record itself.
    """

    return {
        key: deepcopy(value)
        for key, value in values.items()
        if key not in _INTERNAL_PROJECTION_METADATA_FIELDS and not str(key).startswith("_")
    }


def _snapshot_evidence(
    snapshot: Mapping[str, Any],
    normalized: Mapping[str, Any],
    *,
    review_fields: frozenset[str],
) -> dict[str, Any]:
    """Retain only scalar evidence that differs from the public business value.

    Structured OCR/provenance objects belong in extraction-issue evidence, not
    inside a business row.  Identical successful snapshots are redundant and
    make it too easy for a decoder to mistake an old source alias for the v2
    value, so they are dropped here.
    """

    evidence: dict[str, Any] = {}
    for key, value in _without_internal_projection_metadata(snapshot).items():
        if isinstance(value, (Mapping, list, tuple, set)):
            continue
        normalized_value = normalized.get(key)
        if (
            key in review_fields
            or key not in normalized
            or normalized_value in (None, "")
            or value != normalized_value
        ):
            evidence[key] = deepcopy(value)
    return evidence


def _decoded_internal_audit(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): deepcopy(item) for key, item in value.items()}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(decoded, Mapping):
            return {str(key): deepcopy(item) for key, item in decoded.items()}
    return {}


def _promote_internal_uncertainty(
    record: dict[str, Any], values: dict[str, Any]
) -> dict[str, Any]:
    """Move diagnostic-only uncertainty into the structured record envelope."""

    audit = _decoded_internal_audit(values.get("audit"))
    recognition_source = str(values.get("recognition_source") or "")
    raw_status = values.get("raw_status")
    extraction_status = str(values.get("extraction_status") or "").lower()
    status = str(values.get("status_code") or values.get("status") or "")
    audit_reason = str(audit.get("reason") or "")
    duplicate_candidates = audit.get("duplicate_month_candidates")
    requires_review = (
        extraction_status in {"failed", "incomplete", "review", "uncertain", "unknown"}
        or status == "unknown"
        or raw_status not in (None, "")
        or recognition_source == "row_neighbor_consensus"
        or bool(audit_reason)
        or (
            isinstance(duplicate_candidates, int)
            and not isinstance(duplicate_candidates, bool)
            and duplicate_candidates > 1
        )
    )
    if not requires_review:
        return values

    values["extraction_status"] = "review"
    existing_review = record.get("review")
    review = dict(existing_review) if isinstance(existing_review, Mapping) else {}
    review.setdefault("status", "requires_review")
    review["extraction_status"] = "review"
    if raw_status not in (None, ""):
        review["raw_status"] = deepcopy(raw_status)
    if recognition_source:
        review["recognition_source"] = recognition_source
    diagnostics = {
        key: deepcopy(value)
        for key, value in audit.items()
        if key not in {"amount_bbox", "bbox", "source_ref", "status_bbox"}
    }
    if diagnostics:
        review["diagnostics"] = diagnostics
    record["review"] = review
    return values


def _sanitize_projected_business_record(
    record: dict[str, Any],
    *,
    dataset_name: str = "",
    review_fields: frozenset[str] = frozenset(),
    record_has_actionable_issue: bool = False,
) -> None:
    """Quarantine OCR diagnostics without silencing the uncertainty they carry."""

    values = _normalized(record)
    source_extraction_status = str(values.get("extraction_status") or "").lower()
    if review_fields or (
        dataset_name == "credit_account_monthly_performance"
        and record_has_actionable_issue
    ):
        values["extraction_status"] = "review"
    values = _promote_internal_uncertainty(record, values)
    audit = _decoded_internal_audit(values.get("audit"))
    recognition_source = str(values.get("recognition_source") or "")
    duplicate_candidates = audit.get("duplicate_month_candidates")
    intrinsic_monthly_uncertainty = bool(
        source_extraction_status in {"failed", "incomplete", "unknown"}
        or values.get("raw_status") not in (None, "")
        or recognition_source == "row_neighbor_consensus"
        or recognition_source.endswith("_unresolved")
        or (
            isinstance(duplicate_candidates, int)
            and not isinstance(duplicate_candidates, bool)
            and duplicate_candidates > 1
        )
    )
    stale_success_review = bool(
        dataset_name == "credit_account_monthly_performance"
        and (
            values.get("extraction_status") == "review"
            or isinstance(record.get("review"), Mapping)
        )
        and str(values.get("status_code") or "").strip().upper()
        in _CANONICAL_MONTHLY_STATUS_CODES
        and _explicit_valid_monthly_amount(values.get("status_amount"))
        and not review_fields
        and not record_has_actionable_issue
        and not intrinsic_monthly_uncertainty
    )
    if stale_success_review:
        values.pop("extraction_status", None)
        record.pop("extraction_status", None)
        record.pop("review", None)
    has_open_review = bool(
        review_fields
        or (
            dataset_name == "credit_account_monthly_performance"
            and record_has_actionable_issue
        )
        or values.get("extraction_status") == "review"
        or isinstance(record.get("review"), Mapping)
    )
    values = _without_internal_projection_metadata(values)
    # Per-field geometry is consumed by correction and issue construction. It
    # is not a Community record-envelope field; keeping it here lets the generic
    # serializer backfill a JSON-encoded provenance blob into normalized data.
    record.pop("source_refs_by_field", None)
    if isinstance(record.get("normalized"), dict):
        record["normalized"] = values
    else:
        for key in _INTERNAL_PROJECTION_METADATA_FIELDS:
            if key not in _BUSINESS_RECORD_ENVELOPE_FIELDS:
                record.pop(key, None)
        for key in tuple(record):
            if str(key).startswith("_"):
                record.pop(key, None)
        for key, value in values.items():
            if key not in _BUSINESS_RECORD_ENVELOPE_FIELDS:
                record[key] = value
    for snapshot in ("canonical_raw", "raw"):
        value = record.get(snapshot)
        if isinstance(value, Mapping):
            evidence = _snapshot_evidence(value, values, review_fields=review_fields)
            if stale_success_review:
                evidence.pop("extraction_status", None)
            if evidence:
                record[snapshot] = evidence
            else:
                record.pop(snapshot, None)
    if not has_open_review:
        # Confidence is diagnostic metadata, not a business value. Successful
        # rows stay silent; potentially flawed rows retain it in their review
        # envelope when the source supplied one.
        record.pop("confidence", None)


def _normalized(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("normalized")
    if isinstance(value, dict):
        return {
            key: deepcopy(item)
            for key, item in value.items()
            if key != "source_refs_by_field"
        }
    return {
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
            "source_refs_by_field",
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
        for key in ("currency", "account_currency", "reporting_amount_currency"):
            if normalized.get(key) not in (None, ""):
                normalized[key] = normalize_currency(normalized[key])
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


def _project_report_queries(
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive query rows with the query identity on both record layers.

    Source metadata enters this projector with a top-level ``record_id``.  A
    query is a distinct canonical relation, so retaining that metadata identity
    would make field issues point partly at the metadata row and partly at the
    derived query row.  Canonicalize the envelope before the final quality gate
    builds or deduplicates any query-field diagnostics.
    """

    records = _project_records(metadata, _report_query)
    for record in records:
        query_id = str(_normalized(record).get("report_query_id") or "").strip()
        if query_id:
            record["record_id"] = query_id
    return records


def _subject_profile(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "personal_profile_id", "subject_profile_id")
    # Canonical detailed reports represent telephones in their historical,
    # residence, and employment relations.  Do not duplicate those values in
    # the one-row subject profile.
    for redundant_phone in ("mobile_phone", "work_phone", "residence_phone"):
        values.pop(redundant_phone, None)
    return values


def _credit_account(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "validity_type", "credit_line_validity_type")
    if values.get("sequence") in (None, "") and values.get("category_sequence") not in (
        None,
        "",
    ):
        values["sequence"] = values["category_sequence"]
    if values.get("management_institution") in (None, "") and values.get(
        "institution"
    ) not in (None, ""):
        values["management_institution"] = values["institution"]
    if values.get("account_currency") in (None, "") and values.get("currency") not in (
        None,
        "",
    ):
        values["account_currency"] = values["currency"]
    if values.get("card_activation_state") in (None, "") and values.get(
        "activation_state"
    ) not in (None, ""):
        values["card_activation_state"] = values["activation_state"]

    status_alias = values.get("account_status")
    if status_alias in (None, ""):
        status_alias = values.get("status")
    if values.get("account_lifecycle_state") in (None, "") and status_alias not in (
        None,
        "",
    ):
        status_key = str(status_alias).strip().lower()
        values["account_lifecycle_state"] = {
            "active": "open",
            "open": "open",
            "inactive": "open",
            "settled": "settled",
            "closed": "closed",
            "transferred_out": "transferred_out",
        }.get(status_key, status_alias)
    for internal_field in (
        "raw_detail_lines",
        "raw_detail_text",
        "account_identifier_candidates",
        "account_family_quality",
        "account_identifier_source",
        "category_sequence",
        "account_status",
        "account_status_resolution",
        "account_status_raw",
        "institution",
        "currency",
        "activation_state",
        "status",
        "_repayment_context",
    ):
        values.pop(internal_field, None)
    account_type = str(values.get("account_type") or "")
    type_code, type_label = _ACCOUNT_TYPE_CODES.get(account_type, (None, None))
    if type_code:
        values["pboc_account_type_code"] = type_code
        values["pboc_account_type_label"] = type_label
    return values


def _credit_agreement(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    if "credit_line_id" in values:
        values.setdefault("credit_agreement_id", values["credit_line_id"])
        values.pop("credit_line_id", None)
    if "total_limit" in values:
        values.setdefault("facility_limit", values["total_limit"])
        values.pop("total_limit", None)
    return values


def _project_credit_accounts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the public account aliases consistently to values and evidence."""
    projected = _project_records(records, _credit_account)
    for record in projected:
        for snapshot_name in ("canonical_raw", "raw"):
            snapshot = record.get(snapshot_name)
            if isinstance(snapshot, dict):
                record[snapshot_name] = _credit_account(snapshot)
    return projected


def _project_credit_agreements(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rename v1 source aliases in normalized and evidence pools together."""
    projected = _project_records(records, _credit_agreement)
    for record in projected:
        for snapshot_name in ("canonical_raw", "raw"):
            snapshot = record.get(snapshot_name)
            if isinstance(snapshot, dict):
                record[snapshot_name] = _credit_agreement(snapshot)
    return projected


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
    source_status = values.pop("status", None)
    projected_status = values.pop("status_code", None)
    status = projected_status if projected_status not in (None, "") else source_status
    status = str(status).strip() if status not in (None, "") else ""
    allowed_statuses = _CANONICAL_MONTHLY_STATUS_CODES | {"unknown"}
    if status in allowed_statuses:
        values["status_code"] = status
        if status == "unknown":
            values["extraction_status"] = "review"
    else:
        # v2 requires a canonical status token.  A missing, placeholder, or
        # unrecognized source cell is represented explicitly as ``unknown``;
        # the unusable OCR text stays in the review envelope, never in the
        # normalized business field.
        values["status_code"] = "unknown"
        values["extraction_status"] = "review"
        if status:
            values["raw_status"] = status
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
    explicit = str(values.get("related_party_category") or "")
    if explicit in {"person", "organization"}:
        return explicit
    id_type = str(values.get("related_party_id_type") or "")
    id_number = str(values.get("related_party_id_number") or "")
    if "身份" in id_type:
        return "person"
    if "统一社会信用" in id_type or "中征码" in id_type or "组织机构" in id_type:
        return "organization"
    if re.fullmatch(r"\d{17}[0-9Xx]", id_number):
        return "person"
    if re.fullmatch(r"[0-9A-Z]{18}", id_number):
        return "organization"
    return "unknown"


def _repayment_responsibility(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "liability_id", "repayment_responsibility_id")
    id_type = str(values.get("related_party_id_type") or "").strip()
    identifier = str(values.get("related_party_id_number") or "").strip()
    identifier_is_valid = True
    if identifier and id_type == "统一社会信用代码":
        identifier_is_valid = re.fullmatch(r"[0-9A-Z]{18}", identifier) is not None
    elif identifier and id_type == "中征码":
        identifier_is_valid = re.fullmatch(r"[0-9A-Za-z]{16}", identifier) is not None
    if identifier and not identifier_is_valid:
        values.pop("related_party_id_number", None)
        values["source_related_party_id_number"] = identifier
        values.setdefault("extraction_status", "review")
    category = _responsible_party_category(values)
    values["related_party_category"] = category
    combined = values.pop("overdue_months_or_repayment_status", None)
    if combined not in (None, ""):
        values["source_status_value"] = combined
        # The legacy field erased whether the source label was 逾期月数 or
        # 还款状态.  Party category cannot restore that business meaning, so an
        # unlabeled legacy value remains reviewable source evidence only.
        values.setdefault("extraction_status", "review")
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
    source_status = values.pop("status", None)
    projected_status = values.pop("status_code", None)
    status = projected_status if projected_status not in (None, "") else source_status
    status = str(status).strip() if status not in (None, "") else ""
    if status and status not in _SOURCE_SENTINELS:
        values["status_code"] = status
    else:
        values["status_code"] = "unknown"
        values["extraction_status"] = "review"
        if status:
            values["raw_status"] = status
    return values


def _project_repayment_responsibilities(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Separate the legacy combined status field in every evidence pool."""

    projected = _project_records(records, _repayment_responsibility)
    for record in projected:
        for snapshot_name in ("canonical_raw", "raw"):
            snapshot = record.get(snapshot_name)
            if isinstance(snapshot, dict):
                record[snapshot_name] = _repayment_responsibility(snapshot)
    return projected


def _summary_metric(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    _rename_key(values, "credit_summary_metric_id", "credit_business_overview_id")
    if "numeric_value" not in values:
        if "integer_value" in values:
            values["numeric_value"] = values.pop("integer_value")
        elif "number_value" in values:
            values["numeric_value"] = values.pop("number_value")
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
    source_id = values.pop("personal_housing_fund_id", None)
    if source_id not in (None, ""):
        values.setdefault("public_record_id", source_id)
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
            source_id = values.get("statement_id") or values.get("annotation_id") or record.get("record_id")
            statement_id = f"annotation_statement:{source_id or f'{kind}:{index}'}"
            group_id = f"annotation_group:{source_id or f'{kind}:{index}'}"
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


def _project_closed_world_public_records(
    records: list[dict[str, Any]], existing: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue

    typed: dict[str, list[dict[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
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
                    issues.append(
                        make_issue(
                            category="schema_incompleteness",
                            issue_code="canonical_public_content_unresolved",
                            message="A known public-record row had content that was not a structured canonical object; the content was withheld.",
                            parser_stage="v2_closed_world_projection",
                            target_dataset=target,
                            target_record_id=public_id or f"public_records:{index}",
                            field_name="content",
                            observed_value=content,
                            reason_codes=(
                                "closed_canonical_catalog",
                                "unstructured_public_content",
                                "normalized_value_withheld",
                            ),
                        )
                    )
                    continue
            else:
                parsed = content if isinstance(content, dict) else None
            if not isinstance(parsed, dict):
                issues.append(
                    make_issue(
                        category="schema_incompleteness",
                        issue_code="canonical_public_content_unresolved",
                        message="A known public-record row had no structured canonical content; the content was withheld.",
                        parser_stage="v2_closed_world_projection",
                        target_dataset=target,
                        target_record_id=public_id or f"public_records:{index}",
                        field_name="content",
                        observed_value=content,
                        reason_codes=(
                            "closed_canonical_catalog",
                            "structured_content_required",
                            "normalized_value_withheld",
                        ),
                    )
                )
                continue
            typed_values = dict(parsed)
            typed_values.setdefault("public_record_id", public_id)
            typed_values.setdefault("sequence", values.get("sequence"))
            typed.setdefault(target, []).append(_replace_normalized(record, typed_values))
        elif not target:
            unknown_content = values.get("content")
            if isinstance(unknown_content, str):
                try:
                    unknown_content = json.loads(unknown_content)
                except (TypeError, ValueError):
                    pass
            issues.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="canonical_public_record_type_unresolved",
                    message="A public-record row was outside the finite canonical record catalog and was withheld.",
                    parser_stage="v2_closed_world_projection",
                    target_record_id=public_id or f"public_records:{index}",
                    field_name="record_type",
                    observed_value={
                        "record_type": record_type or None,
                        "content": unknown_content,
                    },
                    reason_codes=(
                        "closed_canonical_catalog",
                        "unknown_public_record_type",
                        "record_withheld",
                    ),
                )
            )
    return typed, issues


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
        }
    )
    for target in _PUBLIC_STATUS_TARGETS:
        existing_sources = sources.get(target, ())
        sources[target] = tuple(dict.fromkeys((*existing_sources, "public_records")))
    return sources


def _project_dataset_status(
    status_rows: list[dict[str, Any]],
    projected: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build sparse schema-native status rows for potentially flawed datasets."""
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
        if target in _CONTROL_DATASETS:
            continue
        source_names = source_mapping.get(target, ())
        source_rows = [item for name in source_names for item in indexed.get(name, ())]
        source_values = [values for _record, values in source_rows]
        observed_count = len(projected.get(target, ()))
        source_presence = {
            str(values.get("presence_status") or "") for values in source_values
        }
        if "extraction_failed" in source_presence:
            presence_status = "extraction_failed"
            reason = "source_extraction_failed"
        elif "partial" in source_presence:
            presence_status = "partial"
            reason = "source_partially_observed"
        elif "unknown" in source_presence:
            presence_status = "unknown"
            reason = "source_status_unknown"
        elif observed_count:
            presence_status = "observed_nonempty"
            reason = "records_projected"
        elif source_presence and source_presence <= {"not_applicable"}:
            presence_status = "not_applicable"
            reason = "source_not_applicable"
        elif source_presence and source_presence <= {"explicitly_empty"}:
            presence_status = "explicitly_empty"
            reason = "source_explicitly_empty"
        elif source_values and source_presence <= {"not_observed"} and all(
            values.get("source_statement") or values.get("absence_evidence") for values in source_values
        ):
            presence_status = "not_observed"
            reason = "source_proved_absence"
        elif source_values:
            presence_status = "unknown"
            reason = "source_presence_not_established"
        else:
            presence_status = "unknown"
            reason = "no_source_status_mapping"

        if presence_status not in {"not_observed", "partial", "extraction_failed", "unknown"}:
            continue

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
        expected_counts = [
            int(values["expected_row_count"])
            for values in source_values
            if isinstance(values.get("expected_row_count"), int)
            and not isinstance(values.get("expected_row_count"), bool)
            and int(values["expected_row_count"]) >= 0
        ]
        if expected_counts:
            normalized["expected_row_count"] = max(expected_counts)
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


def _canonical_dataset_name(source_name: Any) -> str:
    """Map a private source collection name to its canonical PBOC dataset."""
    name = str(source_name or "")
    for target, source_names in _status_sources_by_target().items():
        if name in source_names:
            return target
    return name


def _canonical_field_name(dataset_name: str, field_name: Any) -> str:
    """Return the final-v2 field that owns one source/compatibility value."""

    name = str(field_name or "")
    aliases = {
        "subject_profile": {"personal_profile_id": "subject_profile_id"},
        "credit_accounts": {
            "category_sequence": "sequence",
            "institution": "management_institution",
            "currency": "account_currency",
            "activation_state": "card_activation_state",
            "account_status": "account_lifecycle_state",
            "status": "account_lifecycle_state",
            "validity_type": "credit_line_validity_type",
        },
        "credit_agreements": {
            "credit_line_id": "credit_agreement_id",
            "total_limit": "facility_limit",
        },
        "credit_account_monthly_performance": {
            "overdue_amount": "status_amount",
        },
    }
    return aliases.get(dataset_name, {}).get(name, name)


def _field_observation(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    if values.get("observation_status") == "not_observed" and not (
        values.get("source_statement") or values.get("absence_evidence")
    ):
        values["observation_status"] = "unreadable"
        values.setdefault("reason", "source_presence_not_established")
    source_name = str(values.get("dataset_name") or "")
    if source_name == "unknown" and values.get("source_dataset_name"):
        return values
    canonical_name = _canonical_dataset_name(source_name)
    if canonical_name in PBOC_DATASET_ORDER and canonical_name not in _CONTROL_DATASETS:
        values["dataset_name"] = canonical_name
        if values.get("field_name"):
            values["field_name"] = _canonical_field_name(
                canonical_name, values["field_name"]
            )
    else:
        values["dataset_name"] = "unknown"
        if source_name:
            values["source_dataset_name"] = source_name
    raw_value = values.get("raw_value")
    if isinstance(raw_value, (dict, list, tuple)):
        values.pop("raw_value", None)
        values["raw_value_type"] = "object" if isinstance(raw_value, dict) else "array"
    return values


_EMPLOYMENT_BLOB_FIELDS = {
    "编号": "sequence",
    "工作单位": "employer",
    "单位性质": "employer_type",
    "单位地址": "employer_address",
    "单位电话": "employer_phone",
    "职业": "occupation",
    "行业": "industry",
    "职务": "position",
    "职称": "professional_title",
    "进入本单位年份": "entry_year",
    "信息更新日期": "information_updated_date",
    "数据发生机构名称": "data_provider",
}


def _employment_source_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decode legacy multi-field JSON blobs before the canonical projection."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue

    repaired: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, source_record in enumerate(records, start=1):
        if not isinstance(source_record, dict):
            continue
        record = deepcopy(source_record)
        values = _normalized(record)
        canonical_raw = dict(record.get("canonical_raw") or {})
        source_raw = dict(record.get("raw") or {})
        blob = next(
            (
                candidate
                for candidate in (
                    values.get("values"),
                    canonical_raw.get("values"),
                    source_raw.get("values"),
                )
                if candidate not in (None, "")
            ),
            None,
        )
        decoded = decode_mapping(blob)
        raw_blob = next(
            (
                candidate
                for candidate in (
                    values.get("raw_values"),
                    canonical_raw.get("raw_values"),
                    source_raw.get("raw_values"),
                )
                if candidate not in (None, "")
            ),
            None,
        )
        if blob not in (None, "") or raw_blob not in (None, ""):
            review = dict(record.get("review") or {})
            if blob not in (None, ""):
                review["source_values_blob"] = deepcopy(blob)
            if raw_blob not in (None, ""):
                review["source_raw_values_blob"] = deepcopy(raw_blob)
            record["review"] = review
        for pool_name, pool in (("canonical_raw", canonical_raw), ("raw", source_raw)):
            pool.pop("values", None)
            pool.pop("raw_values", None)
            if pool:
                record[pool_name] = pool
            else:
                record.pop(pool_name, None)
        unknown_keys: list[str] = []
        if decoded is not None:
            for source_key, source_value in decoded.items():
                target = _EMPLOYMENT_BLOB_FIELDS.get(str(source_key))
                if target is None:
                    if source_value not in (None, "", "--"):
                        unknown_keys.append(str(source_key))
                    continue
                if values.get(target) in (None, "") and source_value not in (None, "", "--"):
                    values[target] = source_value
            try:
                if values.get("sequence") not in (None, ""):
                    values["sequence"] = int(str(values["sequence"]).strip())
            except ValueError:
                unknown_keys.append("编号")
                values.pop("sequence", None)
            try:
                if values.get("entry_year") not in (None, ""):
                    values["entry_year"] = int(str(values["entry_year"]).strip())
            except ValueError:
                unknown_keys.append("进入本单位年份")
                values.pop("entry_year", None)
        values.pop("values", None)
        values.pop("raw_values", None)
        values.setdefault(
            "employment_record_id",
            str(record.get("record_id") or f"employment_record:{values.get('sequence') or index}"),
        )
        record = _replace_normalized(record, values)
        repaired.append(record)
        if (blob not in (None, "") and decoded is None) or unknown_keys:
            issues.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="unstructured_multifield_blob",
                    message="A serialized multi-field value could not be mapped completely to typed employment fields.",
                    parser_stage="v2_pre_projection_gate",
                    target_dataset="employment_records",
                    target_record_id=str(record.get("record_id") or values["employment_record_id"]),
                    field_name="values",
                    observed_value=blob,
                    reason_codes=("multi_field_scalar_rejected", "raw_evidence_preserved", "normalized_value_withheld"),
                )
            )
    return repaired, issues


def _extraction_issue(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    if values.get("target_dataset"):
        values["target_dataset"] = _canonical_dataset_name(values["target_dataset"])
    else:
        values.pop("target_dataset", None)
    if values.get("field_name") and values.get("target_dataset"):
        values["field_name"] = _canonical_field_name(
            str(values["target_dataset"]), values["field_name"]
        )
    return values


_ISSUE_SOURCE_PAGE_KEY_PRIORITY = (
    "logical_page",
    "page",
    "page_id",
    "page_number",
    "source_page",
)
_ISSUE_SOURCE_PAGE_KEYS = frozenset(_ISSUE_SOURCE_PAGE_KEY_PRIORITY)


def _issue_source_page_number(value: Any) -> int | None:
    """Return one explicit positive, one-based provenance page number."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            number = int(text)
            return number if number > 0 else None
        if len(text) > 1 and text[0].casefold() == "p" and text[1:].isdigit():
            number = int(text[1:])
            return number if number > 0 else None
    return None


def _issue_source_page_range(record: Mapping[str, Any]) -> list[int]:
    """Reduce parent issue provenance to its logical page span.

    Each provenance mapping contributes at most one scalar page.  A logical
    page therefore wins over the physical source-page fallback carried by the
    same ref.  Evidence children inherit only this compact range; the rich refs
    remain on their parent extraction issue.
    """

    pages: list[int] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            normalized_items = {
                str(key).casefold(): item for key, item in value.items()
            }
            selected_page = next(
                (
                    page
                    for key in _ISSUE_SOURCE_PAGE_KEY_PRIORITY
                    if (
                        page := _issue_source_page_number(normalized_items.get(key))
                    )
                    is not None
                ),
                None,
            )
            if selected_page is not None:
                pages.append(selected_page)
            else:
                page_range = normalized_items.get("page_range")
                if isinstance(page_range, (list, tuple)):
                    pages.extend(
                        page
                        for candidate in page_range
                        if (page := _issue_source_page_number(candidate)) is not None
                    )

            for key, item in normalized_items.items():
                if (
                    key not in _ISSUE_SOURCE_PAGE_KEYS
                    and key != "page_range"
                    and isinstance(item, (Mapping, list, tuple))
                ):
                    collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (Mapping, list, tuple)):
                    collect(item)

    for key in (
        "source",
        "source_refs",
        "source_cell_refs",
        "source_anchor",
        "page_range",
        "logical_page",
        "page",
        "page_id",
        "page_number",
        "source_page",
        "normalized",
    ):
        if key in record:
            collect({key: record[key]})
    return [min(pages), max(pages)] if pages else []


def _issue_evidence_rows(issue_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize nested issue evidence into a compact typed child relation."""
    evidence: list[dict[str, Any]] = []
    compact_issues: list[dict[str, Any]] = []

    def emit_leaves(
        issue_id: str,
        kind: str,
        path: str,
        value: Any,
        page_range: list[int],
    ) -> None:
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                child = f"{path}.{key}" if path else str(key)
                emit_leaves(issue_id, kind, child, value[key], page_range)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                emit_leaves(issue_id, kind, f"{path}[{index}]", item, page_range)
            return
        if value is None:
            return
        evidence_id = f"extraction_issue_evidence:{issue_id}:{len(evidence) + 1}"
        row: dict[str, Any] = {
            "record_id": evidence_id,
            "extraction_issue_evidence_id": evidence_id,
            "extraction_issue_id": issue_id,
            "evidence_kind": kind,
            "evidence_path": path or "value",
        }
        if page_range:
            row["source"] = {"page_range": list(page_range)}
        if isinstance(value, bool):
            row.update({"value_type": "boolean", "boolean_value": value})
        elif isinstance(value, int):
            row.update({"value_type": "integer", "integer_value": value})
        elif isinstance(value, float):
            row.update({"value_type": "number", "number_value": value})
        else:
            row.update({"value_type": "string", "string_value": str(value)})
        evidence.append(row)

    for record in issue_records:
        values = _extraction_issue(_normalized(record))
        issue_id = str(values.get("extraction_issue_id") or record.get("record_id") or "unresolved_issue")
        page_range = _issue_source_page_range(record)
        for field_name, kind in (("observed_value", "observed"), ("candidate_value", "candidate")):
            value = values.get(field_name)
            if isinstance(value, (dict, list, tuple)):
                values.pop(field_name, None)
                values[f"{field_name}_type"] = "object" if isinstance(value, dict) else "array"
                emit_leaves(issue_id, kind, "", value, page_range)
        reason_codes = values.pop("reason_codes", ())
        if isinstance(reason_codes, (list, tuple, set)):
            codes = tuple(dict.fromkeys(str(code) for code in reason_codes if str(code)))
            values["reason_code_count"] = len(codes)
            for index, code in enumerate(codes):
                emit_leaves(
                    issue_id,
                    "reason",
                    f"reason_codes[{index}]",
                    code,
                    page_range,
                )
        compact_issues.append(_replace_normalized(record, values))
    return compact_issues, evidence


def _issue_role(field_name: str) -> str | None:
    if field_name in {"currency", "account_currency", "reporting_amount_currency"}:
        return "currency"
    if field_name in {
        "gender",
        "marital_status",
        "employment_status",
        "education_level",
        "degree",
        "responsibility_type",
        "residence_status",
    }:
        return field_name
    if field_name in {"query_reason", "reason"}:
        return "inquiry_reason"
    if field_name == "nationality":
        return "country_or_region_code"
    if field_name == "account_status":
        return "account_status_code"
    return None


def _actionable_issue(record: dict[str, Any]) -> bool:
    values = _normalized(record)
    if str(values.get("status") or "") in {"resolved", "suppressed_redundant", "informational"}:
        return False
    if str(values.get("severity") or "") == "info":
        return False
    if str(values.get("target_dataset") or "") in {"datasets", "facts"}:
        return False
    if values.get("issue_code") != "pboc_cell_contract_unresolved":
        return True
    field_name = str(values.get("field_name") or "")
    observed = values.get("observed_value")
    if field_name == "status_inferred_from_adjacent_months" and isinstance(observed, bool):
        return False
    role = _issue_role(field_name)
    if role is None:
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            _is_valid_for_role,
            _mapping_role,
        )

        role = _mapping_role({}, field_name)
        if role and observed not in (None, ""):
            return not _is_valid_for_role(str(observed), role)
    if role is None or observed in (None, "") or isinstance(observed, (dict, list)):
        return True
    return not validate_pboc_field(str(observed), role).valid


def _withhold(record: dict[str, Any], field_name: str) -> tuple[dict[str, Any], Any]:
    values = _normalized(record)
    original = values.get(field_name)
    if original in (None, ""):
        return record, original
    raw = deepcopy(record.get("canonical_raw"))
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault(field_name, deepcopy(original))
    record = deepcopy(record)
    record["canonical_raw"] = raw
    values[field_name] = None
    projected = _replace_normalized(record, values)
    # Live credit projection rows are usually flat mappings with a separate
    # canonical_raw pool. Community serialization otherwise treats that raw
    # pool as normalized and resurrects the withheld value. Preserve the flat
    # compatibility view while making the normalized override explicit.
    if not isinstance(record.get("normalized"), dict):
        projected["normalized"] = deepcopy(values)
    if field_name in projected:
        projected[field_name] = None
    return projected, original


def _remove_noncanonical_field(record: dict[str, Any], field_name: str) -> tuple[dict[str, Any], Any]:
    """Remove a field outside a closed catalog; issue evidence retains its value."""

    values = _normalized(record)
    original = values.pop(field_name, None)
    projected = _replace_normalized(record, values)
    projected.pop(field_name, None)
    for pool_name in ("canonical_raw", "raw"):
        pool = projected.get(pool_name)
        if isinstance(pool, dict):
            pool = deepcopy(pool)
            pool.pop(field_name, None)
            if pool:
                projected[pool_name] = pool
            else:
                projected.pop(pool_name, None)
    return projected, original


def _record_identity(record: dict[str, Any], dataset_name: str, index: int) -> str:
    values = _normalized(record)
    # ``grid_id`` precedes ``monthly_performance_id`` in real Candidate-B
    # records.  The generic ``*_id`` fallback would therefore collapse every
    # month in a grid onto the grid identifier, breaking exact issue linkage.
    # Prefer the canonical row identity for this dataset before considering
    # relationship IDs.
    if dataset_name == "credit_account_monthly_performance":
        monthly_performance_id = values.get("monthly_performance_id")
        if monthly_performance_id not in (None, ""):
            return str(record.get("record_id") or monthly_performance_id)
    return str(
        record.get("record_id")
        or next((value for key, value in values.items() if key.endswith("_id") and value), None)
        or f"{dataset_name}:row:{index}"
    )


def _link_unique_issue_records(
    projected: dict[str, list[dict[str, Any]]],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a row ID when an unlinked issue value identifies exactly one row."""
    linked: list[dict[str, Any]] = []
    for issue in issues:
        values = _normalized(issue)
        dataset_name = str(values.get("target_dataset") or "")
        field_name = str(values.get("field_name") or "")
        observed = values.get("observed_value")
        if (
            values.get("target_record_id")
            or not dataset_name
            or not field_name
            or observed in (None, "")
            or isinstance(observed, (dict, list))
        ):
            linked.append(issue)
            continue

        matching_ids: set[str] = set()
        for index, record in enumerate(projected.get(dataset_name) or [], start=1):
            candidates = (_normalized(record), record.get("canonical_raw"), record.get("raw"))
            if any(
                isinstance(candidate, dict)
                and candidate.get(field_name) not in (None, "")
                and str(candidate.get(field_name)) == str(observed)
                for candidate in candidates
            ):
                matching_ids.add(_record_identity(record, dataset_name, index))
        if len(matching_ids) == 1:
            values["target_record_id"] = next(iter(matching_ids))
            issue = _replace_normalized(issue, values)
        linked.append(issue)
    return linked


def _address_suspicious(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(
        len(re.sub(r"\s+", "", text)) < 4
        or re.search(r"[=<>]", text)
        or re.search(r"\s[\u3400-\u9fff]\s*(?:[#*=?]+)?$", text)
    )


def _money_field(field_name: str) -> bool:
    if field_name.startswith("source_") or field_name in {"amount_unit", "reporting_amount_precision"}:
        return False
    return field_name.endswith(
        ("_amount", "_balance", "_limit", "_principal", "_payment", "_contribution")
    ) or field_name in {"amount", "balance", "facility_limit", "credit_limit", "loan_amount"}


def _decimal_valid(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return Decimal(str(value).replace(",", "")).is_finite()
    except (InvalidOperation, ValueError):
        return False


def _explicit_valid_monthly_amount(value: Any) -> bool:
    """Return whether a monthly row carries an explicit usable amount."""

    if value in (None, "") or not _decimal_valid(value):
        return False
    return Decimal(str(value).replace(",", "")) >= 0


def _canonical_quality_gate(
    projected: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Withhold invalid v2 values and publish one deduplicated uncertainty."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        _is_valid_for_role,
    )

    issues = [
        record
        for record in projected.get("extraction_issues") or []
        if isinstance(record, dict) and _actionable_issue(record)
    ]
    generated: list[dict[str, Any]] = []

    # Extraction keeps a private ``_source_absent_fields`` ledger until this
    # final public boundary.  Reconcile that typed evidence with dash-only
    # normalized/raw values before issue deduplication, so a successful source
    # absence cannot survive as both a JSON placeholder and a false failure.
    source_absence_targets: set[tuple[str, str, str]] = set()
    declared_source_absence_targets: set[tuple[str, str, str]] = set()
    account_absence_without_competing_evidence: set[tuple[str, str, str]] = set()
    for dataset_name, allowed_fields in _EXPLICIT_SOURCE_ABSENCE_FIELDS.items():
        for index, record in enumerate(projected.get(dataset_name) or [], start=1):
            if not isinstance(record, dict):
                continue
            values = _normalized(record)
            record_id = _record_identity(record, dataset_name, index)
            declared_absent = {
                _canonical_field_name(dataset_name, field_name)
                for field_name in values.get("_source_absent_fields") or ()
                if isinstance(field_name, str)
            }
            for field_name in allowed_fields:
                normalized_value = values.get(field_name)
                snapshot_values = [
                    snapshot[field_name]
                    for snapshot_name in ("canonical_raw", "raw")
                    if isinstance(snapshot := record.get(snapshot_name), Mapping)
                    and field_name in snapshot
                ]
                snapshot_absent = any(
                    _only_explicit_source_absence(value)
                    for value in snapshot_values
                )
                strict_direct_proof = (
                    dataset_name in _STRICT_DIRECT_SOURCE_ABSENCE_DATASETS
                )
                competing_snapshot = any(
                    _has_competing_non_absence_observation(value)
                    for value in snapshot_values
                )
                competing_issue = any(
                    str(issue_values.get("target_dataset") or "") == dataset_name
                    and str(issue_values.get("target_record_id") or "") == record_id
                    and str(issue_values.get("field_name") or "") == field_name
                    and _issue_has_competing_non_absence_observation(issue_values)
                    for issue in issues
                    if isinstance(issue, dict)
                    and (issue_values := _normalized(issue))
                )
                target = (dataset_name, record_id, field_name)
                if (
                    dataset_name == "credit_accounts"
                    and (
                        normalized_value in (None, "")
                        or _only_explicit_source_absence(normalized_value)
                    )
                    and not competing_snapshot
                    and not competing_issue
                ):
                    # Account slots may be observed by both a dense cluster and
                    # an exact header cell.  A dash is source absence only when
                    # neither retained plane contains a substantive candidate.
                    # Keep this proof separately so the issue-only dash path
                    # below cannot reintroduce a target rejected here.
                    account_absence_without_competing_evidence.add(target)
                if strict_direct_proof:
                    source_absent = bool(
                        normalized_value in (None, "")
                        and field_name in declared_absent
                        and snapshot_absent
                        and not competing_snapshot
                        and not competing_issue
                    )
                elif dataset_name == "credit_accounts":
                    source_absent = bool(
                        (
                            _only_explicit_source_absence(normalized_value)
                            or (
                                normalized_value in (None, "")
                                and (
                                    field_name in declared_absent
                                    or snapshot_absent
                                )
                            )
                        )
                        and not competing_snapshot
                        and not competing_issue
                    )
                else:
                    source_absent = bool(
                        _only_explicit_source_absence(normalized_value)
                        or (
                            normalized_value in (None, "")
                            and (field_name in declared_absent or snapshot_absent)
                        )
                    )
                if source_absent:
                    source_absence_targets.add(target)
                    if field_name in declared_absent:
                        declared_source_absence_targets.add(target)

    # A stale field-level failure may itself retain the dash witness.  This is
    # also sufficient source-absence evidence, but only for the finite field
    # catalog above and only when every retained candidate is a dash sentinel.
    for record in issues:
        values = _normalized(record)
        dataset_name = str(values.get("target_dataset") or "")
        field_name = str(values.get("field_name") or "")
        record_id = str(values.get("target_record_id") or "")
        if (
            field_name in _EXPLICIT_SOURCE_ABSENCE_FIELDS.get(dataset_name, frozenset())
            and dataset_name not in _STRICT_DIRECT_SOURCE_ABSENCE_DATASETS
            and record_id
            and _only_explicit_source_absence(values.get("observed_value"))
            and (
                dataset_name != "credit_accounts"
                or (dataset_name, record_id, field_name)
                in account_absence_without_competing_evidence
            )
        ):
            source_absence_targets.add((dataset_name, record_id, field_name))

    # In the canonical spouse scalar, an explicitly absent name is the source
    # assertion that no spouse row exists when every other spouse business
    # slot is also empty.  The provider footer is provenance and may remain;
    # a merely blank name has no such meaning and must stay uncertain.
    spouse_business_fields = (
        "name",
        "document_type",
        "document_number",
        "employer",
        "phone",
    )
    reconciled_spouse_rows: list[dict[str, Any]] = []
    for index, record in enumerate(projected.get("subject_spouse") or [], start=1):
        if not isinstance(record, dict):
            continue
        values = _normalized(record)
        record_id = _record_identity(record, "subject_spouse", index)
        name_is_explicitly_absent = (
            "subject_spouse",
            record_id,
            "name",
        ) in source_absence_targets
        normalized_name_is_absent = values.get("name") in (None, "") or (
            _only_explicit_source_absence(values.get("name"))
        )
        other_business_values_absent = all(
            values.get(field_name) in (None, "")
            or _only_explicit_source_absence(values.get(field_name))
            for field_name in spouse_business_fields[1:]
        )
        if (
            name_is_explicitly_absent
            and normalized_name_is_absent
            and other_business_values_absent
        ):
            for field_name in spouse_business_fields:
                source_absence_targets.add(
                    ("subject_spouse", record_id, field_name)
                )
                values[field_name] = None
            record = _replace_normalized(record, values)
        reconciled_spouse_rows.append(record)
    if reconciled_spouse_rows:
        projected["subject_spouse"] = reconciled_spouse_rows

    # Dataset status is intentionally sparse: a fully observed dataset has no
    # status row.  Upstream extraction labels the spouse dataset ``partial``
    # before this final boundary has reconciled dash-only source absence.  Clear
    # that stale status only for the exact canonical pattern proved complete:
    # one row, five directly declared dash-absent scalars, a provider, and no
    # actionable spouse issue that will survive source-absence reconciliation.
    spouse_rows = projected.get("subject_spouse") or []
    spouse_source_absence_complete = False
    if len(spouse_rows) == 1 and isinstance(spouse_rows[0], dict):
        spouse_values = _normalized(spouse_rows[0])
        spouse_record_id = _record_identity(
            spouse_rows[0], "subject_spouse", 1
        )
        spouse_source_absence_complete = bool(
            str(spouse_values.get("data_provider") or "").strip()
            and all(
                spouse_values.get(field_name) in (None, "")
                and (
                    "subject_spouse",
                    spouse_record_id,
                    field_name,
                )
                in declared_source_absence_targets
                for field_name in spouse_business_fields
            )
        )
        spouse_source_absence_complete = spouse_source_absence_complete and not any(
            str(values.get("target_dataset") or "") == "subject_spouse"
            and not (
                (
                    "subject_spouse",
                    str(values.get("target_record_id") or ""),
                    str(values.get("field_name") or ""),
                )
                in source_absence_targets
                and str(values.get("issue_code") or "")
                in _SOURCE_ABSENCE_SUPERSEDED_ISSUE_CODES
            )
            for issue in issues
            if isinstance(issue, dict) and (values := _normalized(issue))
        )
    if spouse_source_absence_complete:
        statuses = list(projected.get("dataset_status") or [])
        spouse_statuses = [
            record
            for record in statuses
            if str(_normalized(record).get("dataset_name") or "")
            == "subject_spouse"
        ]
        if spouse_statuses and all(
            str(_normalized(record).get("presence_status") or "") == "partial"
            and str(_normalized(record).get("reason") or "")
            == "source_partially_observed"
            for record in spouse_statuses
        ):
            remaining_statuses = [
                record
                for record in statuses
                if str(_normalized(record).get("dataset_name") or "")
                != "subject_spouse"
            ]
            if remaining_statuses:
                projected["dataset_status"] = remaining_statuses
            else:
                projected.pop("dataset_status", None)

    issues = [
        record
        for record in issues
        if not (
            (
                str(_normalized(record).get("target_dataset") or ""),
                str(_normalized(record).get("target_record_id") or ""),
                str(_normalized(record).get("field_name") or ""),
            )
            in source_absence_targets
            and str(_normalized(record).get("issue_code") or "")
            in _SOURCE_ABSENCE_SUPERSEDED_ISSUE_CODES
        )
    ]

    def actionable_target_field_key(record: dict[str, Any]) -> tuple[str, str, str] | None:
        values = _normalized(record)
        key = (
            str(values.get("target_dataset") or ""),
            str(values.get("target_record_id") or ""),
            str(values.get("field_name") or ""),
        )
        return key if all(key) else None

    actionable_target_fields = {
        key
        for record in issues
        if (key := actionable_target_field_key(record)) is not None
    }
    native_source_status_conflict_record_ids = {
        str(values.get("target_record_id") or "").strip()
        for issue in issues
        if (
            values := _normalized(issue)
        ).get("issue_code")
        == "candidate_b_native_source_cell_repayment_status_conflict"
        and str(values.get("target_dataset") or "")
        == "credit_account_monthly_performance"
        and str(values.get("field_name") or "") == "status_code"
        and str(values.get("target_record_id") or "").strip()
    }

    # Monthly status candidates must survive relationship construction and the
    # professional correction overlay. Only this final public-schema boundary
    # may discard a still-unresolved candidate. This preserves every candidate
    # that page-level OCR can repair while preventing ``unknown`` from becoming
    # a typed business status in Community JSON.
    monthly_rows = list(projected.get("credit_account_monthly_performance") or [])
    if monthly_rows:
        field_resolved_status_record_ids = {
            _record_identity(
                record,
                "credit_account_monthly_performance",
                index,
            )
            for index, record in enumerate(monthly_rows, start=1)
            if str(_normalized(record).get("status_code") or "").strip().upper()
            in _CANONICAL_MONTHLY_STATUS_CODES
        }
        account_values_by_id = {
            str(values.get("account_id") or ""): values
            for record in projected.get("credit_accounts") or []
            if isinstance(record, dict)
            and (values := _normalized(record)).get("account_id") not in (None, "")
        }
        terminal_checked_rows: list[dict[str, Any]] = []
        for index, record in enumerate(monthly_rows, start=1):
            values = _normalized(record)
            account_id = str(values.get("account_id") or "").strip()
            parent = account_values_by_id.get(account_id)
            lifecycle = str((parent or {}).get("account_lifecycle_state") or "").lower()
            close_date = str((parent or {}).get("close_date") or "").strip()
            close_month = (
                close_date[:7]
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", close_date)
                else ""
            )
            performance_month = str(values.get("performance_month") or "").strip()
            status = str(values.get("status_code") or "").strip().upper()
            if (
                parent is not None
                and lifecycle in {"settled", "closed"}
                and close_month
                and performance_month == close_month
                and status == "N"
            ):
                record_id = _record_identity(
                    record, "credit_account_monthly_performance", index
                )
                record, observed = _withhold(record, "status_code")
                refs: list[dict[str, Any]] = []
                ref_markers: set[str] = set()
                for source_name in ("source_cell_refs", "source_refs"):
                    for ref in record.get(source_name) or ():
                        if not isinstance(ref, Mapping):
                            continue
                        marker = json.dumps(
                            ref, ensure_ascii=False, sort_keys=True, default=str
                        )
                        if marker in ref_markers:
                            continue
                        ref_markers.add(marker)
                        refs.append(dict(ref))
                generated.append(
                    make_issue(
                        category="schema_incompleteness",
                        issue_code="candidate_b_monthly_terminal_status_conflict",
                        message=(
                            "The linked account is settled or closed in this exact "
                            "month, but the monthly status is N; the status was "
                            "withheld without inferring C."
                        ),
                        parser_stage="candidate_b_final_monthly_gate",
                        target_dataset="credit_account_monthly_performance",
                        target_record_id=record_id,
                        field_name="status_code",
                        observed_value=observed,
                        candidate_value={
                            "account_id": account_id,
                            "performance_month": performance_month,
                            "account_lifecycle_state": lifecycle,
                            "close_date": close_date,
                        },
                        source_refs=refs,
                        reason_codes=(
                            "linked_parent_account",
                            "terminal_parent_lifecycle",
                            "performance_month_matches_close_date_month",
                            "normal_status_conflicts_with_terminal_month",
                            "raw_status_preserved",
                            "normalized_value_withheld",
                            "terminal_status_not_inferred",
                        ),
                    )
                )
                actionable_target_fields.add(
                    (
                        "credit_account_monthly_performance",
                        record_id,
                        "status_code",
                    )
                )
            terminal_checked_rows.append(record)
        monthly_rows = terminal_checked_rows

        # N, *, /, and C explicitly mean that this cell carries no overdue
        # amount.  A nonzero paired value therefore cannot be published as
        # business truth even when both scalars are individually well-formed.
        # Preserve the status, withhold only the contradictory amount, and do
        # not replace it with an inferred zero.
        zero_status_amount_checked_rows: list[dict[str, Any]] = []
        for index, record in enumerate(monthly_rows, start=1):
            values = _normalized(record)
            status = str(values.get("status_code") or "").strip().upper()
            amount = _decimal_string(values.get("status_amount"))
            if (
                status in {"N", "*", "/", "C"}
                and amount is not None
                and Decimal(amount) != 0
            ):
                record_id = _record_identity(
                    record, "credit_account_monthly_performance", index
                )
                record, observed = _withhold(record, "status_amount")
                normalized = dict(_normalized(record))
                normalized["status_amount_semantics"] = None
                record = _replace_normalized(record, normalized)
                if "status_amount_semantics" in record:
                    record["status_amount_semantics"] = None
                for pool_name in ("canonical_raw", "raw"):
                    pool = record.get(pool_name)
                    if not isinstance(pool, dict):
                        continue
                    pool = deepcopy(pool)
                    pool.pop("status_amount_semantics", None)
                    record[pool_name] = pool
                refs: list[dict[str, Any]] = []
                ref_markers: set[str] = set()
                for source_name in ("source_cell_refs", "source_refs"):
                    for ref in record.get(source_name) or ():
                        if not isinstance(ref, Mapping):
                            continue
                        marker = json.dumps(
                            ref, ensure_ascii=False, sort_keys=True, default=str
                        )
                        if marker in ref_markers:
                            continue
                        ref_markers.add(marker)
                        refs.append(dict(ref))
                generated.append(
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_monthly_zero_status_amount_conflict",
                        message=(
                            "The monthly status requires a zero overdue amount, but "
                            "the extracted paired amount was nonzero; the amount was "
                            "withheld without inferring zero."
                        ),
                        parser_stage="candidate_b_final_monthly_gate",
                        target_dataset="credit_account_monthly_performance",
                        target_record_id=record_id,
                        field_name="status_amount",
                        observed_value=observed,
                        candidate_value={
                            "performance_month": values.get("performance_month"),
                            "status_code": status,
                        },
                        source_refs=refs,
                        reason_codes=(
                            "canonical_zero_overdue_status",
                            "explicit_nonzero_paired_amount",
                            "status_amount_business_conflict",
                            "raw_amount_preserved",
                            "normalized_value_withheld",
                            "zero_inference_forbidden",
                        ),
                    )
                )
                actionable_target_fields.add(
                    (
                        "credit_account_monthly_performance",
                        record_id,
                        "status_amount",
                    )
                )
            zero_status_amount_checked_rows.append(record)
        monthly_rows = zero_status_amount_checked_rows

        resolved_monthly_rows: list[dict[str, Any]] = []
        unresolved_by_grid: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        emitted_by_grid: dict[str, int] = {}
        grid_order: dict[str, int] = {}
        withheld_record_ids: set[str] = set()

        def monthly_group(values: Mapping[str, Any]) -> str:
            grid_id = str(values.get("grid_id") or "")
            if grid_id:
                return f"grid:{grid_id}"
            account_id = str(values.get("account_id") or "")
            return f"account:{account_id or 'unresolved'}"

        for index, record in enumerate(monthly_rows, start=1):
            values = _normalized(record)
            group = monthly_group(values)
            grid_order.setdefault(group, len(grid_order))
            status = str(values.get("status_code") or "").strip().upper()
            account_id = str(values.get("account_id") or "").strip()
            if status in _CANONICAL_MONTHLY_STATUS_CODES and account_id:
                resolved_monthly_rows.append(record)
                emitted_by_grid[group] = emitted_by_grid.get(group, 0) + 1
                continue
            record_id = _record_identity(
                record, "credit_account_monthly_performance", index
            )
            stable_record_id = str(
                record.get("record_id")
                or values.get("monthly_performance_id")
                or ""
            ).strip()
            grid_id = str(values.get("grid_id") or "").strip()
            performance_month = str(values.get("performance_month") or "").strip()
            retain_native_source_conflict = bool(
                stable_record_id
                and account_id
                and grid_id
                and re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", performance_month)
                and _explicit_valid_monthly_amount(values.get("status_amount"))
                and stable_record_id in native_source_status_conflict_record_ids
            )
            if retain_native_source_conflict:
                # The exact same-cell conflict proves that the status is
                # unknown, not that the entire month is absent.  Keep the
                # independently explicit amount and stable month identity,
                # while ensuring no candidate status enters normalized data.
                values["status_code"] = None
                values["extraction_status"] = "review"
                record = _replace_normalized(record, values)
                record["record_id"] = stable_record_id
                resolved_monthly_rows.append(record)
                emitted_by_grid[group] = emitted_by_grid.get(group, 0) + 1
                continue
            unresolved_by_grid.setdefault(group, []).append((index, record))
            withheld_record_ids.add(
                record_id
            )

        resolved_monthly_rows.sort(
            key=lambda record: (
                grid_order[monthly_group(_normalized(record))],
                str(_normalized(record).get("performance_month") or ""),
                _record_identity(
                    record,
                    "credit_account_monthly_performance",
                    0,
                ),
            )
        )
        projected["credit_account_monthly_performance"] = resolved_monthly_rows

        # A valid status does not establish the paired amount.  Preserve every
        # emitted status row, but make a missing amount explicitly actionable
        # at that exact business record instead of silently treating it as
        # zero.  Existing field-local correction issues remain authoritative.
        for index, record in enumerate(resolved_monthly_rows, start=1):
            values = _normalized(record)
            if values.get("status_amount") not in (None, ""):
                continue
            record_id = _record_identity(
                record, "credit_account_monthly_performance", index
            )
            target_field_key = (
                "credit_account_monthly_performance",
                record_id,
                "status_amount",
            )
            if target_field_key in actionable_target_fields:
                continue
            refs: list[dict[str, Any]] = []
            for source_name in ("source_cell_refs", "source_refs"):
                refs.extend(
                    dict(ref)
                    for ref in record.get(source_name) or ()
                    if isinstance(ref, Mapping)
                )
            generated.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_monthly_status_amount_unresolved",
                    message=(
                        "The monthly status row was emitted, but its paired amount "
                        "was not extractable; no zero amount was inferred."
                    ),
                    parser_stage="candidate_b_final_monthly_gate",
                    target_dataset="credit_account_monthly_performance",
                    target_record_id=record_id,
                    field_name="status_amount",
                    candidate_value={
                        "performance_month": values.get("performance_month"),
                        "status_code": values.get("status_code"),
                    },
                    source_refs=refs,
                    reason_codes=(
                        "monthly_status_emitted",
                        "paired_amount_not_extracted",
                        "zero_inference_forbidden",
                        "field_local_review_required",
                    ),
                )
            )
            actionable_target_fields.add(target_field_key)

        # Keep the aggregate issue for population conservation, but do not let
        # it erase the field-local reason that an exact month was withheld.
        # Geometry/physical-month ownership failures arrive under the private
        # ``status`` alias; normalize only those still-withheld exact targets to
        # the public ``status_code`` field.  Conversely, an active-looking
        # source issue is stale once that exact row now has a canonical status.
        reconciled_monthly_status_issues: list[dict[str, Any]] = []
        for issue in issues:
            issue_values = _normalized(issue)
            is_monthly_status_contract_issue = bool(
                str(issue_values.get("target_dataset") or "")
                == "credit_account_monthly_performance"
                and str(issue_values.get("field_name") or "")
                in {"status", "status_code"}
                and str(issue_values.get("issue_code") or "")
                == "pboc_cell_contract_unresolved"
            )
            if not is_monthly_status_contract_issue:
                reconciled_monthly_status_issues.append(issue)
                continue
            target_record_id = str(
                issue_values.get("target_record_id") or ""
            ).strip()
            if target_record_id in field_resolved_status_record_ids:
                continue
            if target_record_id in withheld_record_ids:
                issue_values = dict(issue_values)
                issue_values["field_name"] = "status_code"
                issue = _replace_normalized(issue, issue_values)
            reconciled_monthly_status_issues.append(issue)
        issues = reconciled_monthly_status_issues

        for group, indexed_records in sorted(unresolved_by_grid.items()):
            records = [record for _index, record in indexed_records]
            values = [_normalized(record) for record in records]
            months = sorted(
                {
                    str(item.get("performance_month") or "")
                    for item in values
                    if re.fullmatch(r"\d{4}-\d{2}", str(item.get("performance_month") or ""))
                }
            )
            grid_ids = {str(item.get("grid_id") or "") for item in values}
            grid_ids.discard("")
            account_ids = {str(item.get("account_id") or "") for item in values}
            account_ids.discard("")
            refs: list[dict[str, Any]] = []
            ref_markers: set[str] = set()
            for record in records:
                for source_key in ("source_cell_refs", "source_refs"):
                    for ref in record.get(source_key) or ():
                        if not isinstance(ref, Mapping):
                            continue
                        marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                        if marker in ref_markers:
                            continue
                        ref_markers.add(marker)
                        refs.append(dict(ref))
                        if len(refs) >= 8:
                            break
                    if len(refs) >= 8:
                        break
                if len(refs) >= 8:
                    break
            generated.append(
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_monthly_status_grid_unresolved",
                    message=(
                        "One or more monthly status cells remained unresolved after the one-shot page repair and "
                        "professional correction; only those rows were withheld from the typed dataset."
                    ),
                    parser_stage="candidate_b_final_monthly_gate",
                    target_dataset="credit_account_monthly_performance",
                    field_name="status_code",
                    observed_value={
                        "grid_id": next(iter(grid_ids)) if len(grid_ids) == 1 else None,
                        "account_id": next(iter(account_ids)) if len(account_ids) == 1 else None,
                        "withheld_candidate_count": len(records),
                        "withheld_month_count": len(months) or len(records),
                        "withheld_months": months,
                        "first_withheld_month": months[0] if months else None,
                        "last_withheld_month": months[-1] if months else None,
                    },
                    candidate_value={
                        "emitted_month_count_for_grid": emitted_by_grid.get(group, 0),
                    },
                    source_refs=refs,
                    reason_codes=(
                        "monthly_status_contract_failed",
                        "final_corrected_candidate_unresolved",
                        "unresolved_status_rows_withheld",
                        "dataset_incomplete",
                    ),
                )
            )

    dictionary_datasets = personal_detail_data_dictionary()["datasets"]
    money_fields_by_dataset = {
        dataset_name: {
            field_name
            for field_name, descriptor in ((definition.get("columns") or {}).items())
            if isinstance(descriptor, dict) and descriptor.get("type") == "money"
        }
        for dataset_name, definition in dictionary_datasets.items()
    }
    closed_world_datasets = {
        *_PUBLIC_STATUS_TARGETS,
        *_ACCOUNT_EVENT_TARGETS.values(),
        "credit_business_overview",
    }
    allowed_fields_by_dataset = {
        dataset_name: set(dictionary_datasets.get(dataset_name, {}).get("columns") or {})
        for dataset_name in closed_world_datasets
    }

    account_ids = {
        str(_normalized(record).get("account_id") or "")
        for record in projected.get("credit_accounts") or []
    }
    account_ids.discard("")
    account_ids_by_identifier: dict[str, set[str]] = {}
    account_identifier_by_id: dict[str, str] = {}
    for record in projected.get("credit_accounts") or []:
        values = _normalized(record)
        identifier = str(values.get("account_identifier") or "")
        account_id = str(values.get("account_id") or "")
        if identifier and account_id:
            account_ids_by_identifier.setdefault(identifier, set()).add(account_id)
            account_identifier_by_id[account_id] = identifier
    account_by_identifier = {
        identifier: next(iter(identifiers))
        for identifier, identifiers in account_ids_by_identifier.items()
        if len(identifiers) == 1
    }

    postpaid_parents = {
        str(_normalized(record).get("postpaid_record_id") or ""): _normalized(record)
        for record in projected.get("postpaid_accounts") or []
        if str(_normalized(record).get("postpaid_record_id") or "")
    }
    checked_postpaid_months: list[dict[str, Any]] = []
    for index, record in enumerate(projected.get("postpaid_monthly_performance") or [], start=1):
        values = _normalized(record)
        record_id = _record_identity(record, "postpaid_monthly_performance", index)
        parent_id = str(values.get("postpaid_record_id") or "")
        parent = postpaid_parents.get(parent_id)
        if parent is None:
            record, observed = _withhold(record, "postpaid_record_id")
            generated.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="unresolved_postpaid_parent_identity",
                    message="A postpaid monthly row could not be linked to one canonical postpaid account; the parent identity was withheld.",
                    parser_stage="v2_post_projection_gate",
                    target_dataset="postpaid_monthly_performance",
                    target_record_id=record_id,
                    field_name="postpaid_record_id",
                    observed_value=observed,
                    reason_codes=(
                        "orphan_postpaid_month",
                        "canonical_parent_required",
                        "normalized_value_withheld",
                    ),
                )
            )
            checked_postpaid_months.append(record)
            continue
        missing_parent_fields = [
            field_name
            for field_name in ("institution", "business_type", "billing_month")
            if parent.get(field_name) in (None, "")
        ]
        if missing_parent_fields:
            generated.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="postpaid_parent_identity_incomplete",
                    message="A linked postpaid account lacks required identity fields; the monthly row remains explicitly linked and uncertain.",
                    parser_stage="v2_post_projection_gate",
                    target_dataset="postpaid_monthly_performance",
                    target_record_id=record_id,
                    field_name="postpaid_record_id",
                    observed_value={
                        "postpaid_record_id": parent_id,
                        "missing_parent_fields": missing_parent_fields,
                    },
                    reason_codes=(
                        "canonical_parent_link_present",
                        "parent_identity_incomplete",
                        "linked_uncertainty_reported",
                    ),
                )
            )
        for field_name in ("institution", "business_type"):
            child_value = values.get(field_name)
            parent_value = parent.get(field_name)
            if child_value in (None, "") and parent_value not in (None, ""):
                values[field_name] = parent_value
                record = _replace_normalized(record, values)
            elif (
                child_value not in (None, "")
                and parent_value not in (None, "")
                and str(child_value) != str(parent_value)
            ):
                record, observed = _withhold(record, field_name)
                generated.append(
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="postpaid_month_parent_identity_conflict",
                        message="A postpaid monthly identity disagreed with its canonical parent; the child value was withheld.",
                        parser_stage="v2_post_projection_gate",
                        target_dataset="postpaid_monthly_performance",
                        target_record_id=record_id,
                        field_name=field_name,
                        observed_value={"monthly": observed, "parent": parent_value},
                        reason_codes=(
                            "canonical_parent_link",
                            "identity_conflict",
                            "normalized_value_withheld",
                        ),
                    )
                )
        checked_postpaid_months.append(record)
    if checked_postpaid_months:
        projected["postpaid_monthly_performance"] = checked_postpaid_months

    linked_datasets = (
        "credit_account_latest_repayments",
        "credit_account_special_transactions",
        "credit_account_special_events",
        "credit_card_large_installments",
    )
    for dataset_name in linked_datasets:
        repaired_rows: list[dict[str, Any]] = []
        for index, record in enumerate(projected.get(dataset_name) or [], start=1):
            values = _normalized(record)
            account_id = str(values.get("account_id") or "")
            event_identifier = str(values.get("account_identifier") or "").strip()
            if account_id not in account_ids:
                relinked = account_by_identifier.get(event_identifier)
                if relinked:
                    values["account_id"] = relinked
                    record = _replace_normalized(record, values)
                    account_id = relinked
                else:
                    record, observed = _withhold(record, "account_id")
                    generated.append(
                        make_issue(
                            category="schema_incompleteness",
                            issue_code="unresolved_account_event_link",
                            message="The account event could not be linked to an emitted account; the foreign key was withheld.",
                            parser_stage="v2_post_projection_gate",
                            target_dataset=dataset_name,
                            target_record_id=_record_identity(record, dataset_name, index),
                            field_name="account_id",
                            observed_value={
                                "account_id": observed,
                                "account_identifier": event_identifier or None,
                            },
                            reason_codes=("orphan_foreign_key", "raw_evidence_preserved", "normalized_value_withheld"),
                        )
                    )
                    account_id = ""
            parent_identifier = account_identifier_by_id.get(account_id, "")
            if event_identifier and parent_identifier and event_identifier != parent_identifier:
                record, observed_account_id = _withhold(record, "account_id")
                generated.append(
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="account_event_parent_identifier_conflict",
                        message=(
                            "The account event's printed account identifier disagreed with its linked canonical "
                            "account; the foreign key was withheld."
                        ),
                        parser_stage="v2_post_projection_gate",
                        target_dataset=dataset_name,
                        target_record_id=_record_identity(record, dataset_name, index),
                        field_name="account_id",
                        observed_value={
                            "account_id": observed_account_id,
                            "event_account_identifier": event_identifier,
                            "parent_account_identifier": parent_identifier,
                        },
                        reason_codes=(
                            "canonical_parent_link",
                            "account_identifier_conflict",
                            "normalized_value_withheld",
                        ),
                    )
                )
            if event_identifier:
                # The event's printed account identifier is only linkage
                # evidence.  Once the canonical account relation has been
                # checked, do not expose or report this denormalized duplicate
                # as though it were a missing event business field.
                record, _discarded_identifier = _remove_noncanonical_field(
                    record, "account_identifier"
                )
            repaired_rows.append(record)
        if repaired_rows:
            projected[dataset_name] = repaired_rows

    required_query_fields = (
        "report_number",
        "report_time",
        "subject_name",
        "primary_id_type",
        "primary_id_number",
        "query_institution",
        "query_reason",
    )
    for dataset_name, rows in list(projected.items()):
        if dataset_name in _QUALITY_GATE_EXEMPT_DATASETS:
            continue
        checked_rows: list[dict[str, Any]] = []
        for index, record in enumerate(rows, start=1):
            values = _normalized(record)
            record_id = _record_identity(record, dataset_name, index)
            invalid: list[tuple[str, Any, str]] = []
            source_absent_fields = {
                field_name
                for field_name in _EXPLICIT_SOURCE_ABSENCE_FIELDS.get(
                    dataset_name, frozenset()
                )
                if (dataset_name, record_id, field_name) in source_absence_targets
                and (
                    values.get(field_name) in (None, "")
                    or _only_explicit_source_absence(values.get(field_name))
                )
            }
            for currency_fields, unit_field in (
                (("account_currency", "currency"), "amount_unit"),
                (("reporting_amount_currency", "account_currency", "currency"), "reporting_amount_unit"),
            ):
                currency = next(
                    (
                        normalize_currency(values[field_name])
                        for field_name in currency_fields
                        if values.get(field_name) not in (None, "")
                    ),
                    None,
                )
                unit = values.get(unit_field)
                if currency not in (None, "", "CNY") and str(unit or "").strip().lower() == "yuan":
                    invalid.append((unit_field, unit, "currency_amount_unit_conflict"))
            if dataset_name in {"report_metadata", "report_query"}:
                header_fields = (
                    required_query_fields
                    if dataset_name == "report_query"
                    else required_query_fields[:5]
                )
                for field_name in header_fields:
                    value = values.get(field_name)
                    id_type = values.get("primary_id_type")
                    valid = (
                        validate_pboc_field(str(value or ""), "inquiry_reason").valid
                        if field_name == "query_reason" and value not in (None, "")
                        else header_field_valid(field_name, value, id_type=id_type)
                    )
                    if not valid:
                        invalid.append((field_name, value, "required_header_field_unresolved"))
            if dataset_name == "subject_profile":
                for field_name in ("subject_name", "primary_id_type", "primary_id_number"):
                    if values.get(field_name) not in (None, "") and not header_field_valid(
                        field_name, values[field_name], id_type=values.get("primary_id_type")
                    ):
                        invalid.append((field_name, values[field_name], "canonical_field_contract_failed"))
            if dataset_name == "subject_identity_documents":
                for field_name in ("document_type", "document_number"):
                    if values.get(field_name) not in (None, "") and not header_field_valid(
                        field_name, values[field_name], id_type=values.get("document_type")
                    ):
                        invalid.append((field_name, values[field_name], "canonical_field_contract_failed"))
            for field_name, value in list(values.items()):
                if (
                    value in (None, "")
                    or field_name.startswith("source_")
                    or field_name in _INTERNAL_PROJECTION_METADATA_FIELDS
                    or field_name.startswith("_")
                ):
                    continue
                if (
                    dataset_name in closed_world_datasets
                    and field_name not in allowed_fields_by_dataset.get(dataset_name, set())
                ):
                    invalid.append((field_name, value, "canonical_field_outside_closed_catalog"))
                elif (
                    isinstance(value, str)
                    and is_explicit_source_absence(value)
                ):
                    # A printed dash is a canonical assertion that this scalar
                    # is absent, not an extraction failure.  Normalize it to
                    # JSON null and keep successful-absence evidence out of the
                    # public row; unresolved OCR lookalikes remain reportable.
                    values[field_name] = None
                    source_absent_fields.add(field_name)
                    continue
                elif isinstance(value, (Mapping, list, tuple, set)):
                    invalid.append((field_name, value, "canonical_unstructured_value_withheld"))
                elif field_name.endswith("_date") or field_name in {"birth_date", "inquiry_date"}:
                    if not valid_iso_date(value):
                        invalid.append((field_name, value, "canonical_date_invalid"))
                elif field_name.endswith("_month"):
                    if not valid_iso_date(value, month_precision=True):
                        invalid.append((field_name, value, "canonical_month_invalid"))
                elif (
                    _money_field(field_name)
                    or field_name in money_fields_by_dataset.get(dataset_name, set())
                ) and not _decimal_valid(value):
                    invalid.append((field_name, value, "canonical_money_invalid"))
                elif field_name in {"mailing_address", "household_address", "address", "employer_address"}:
                    if _address_suspicious(value):
                        invalid.append((field_name, value, "canonical_address_suspicious"))
                elif field_name == "account_lifecycle_state" and str(value) not in {
                    "open",
                    "settled",
                    "closed",
                    "transferred_out",
                    "unknown",
                }:
                    invalid.append(
                        (field_name, value, "canonical_field_contract_failed")
                    )
                # The post-projection gate owns canonical business enums.  Other
                # role-sensitive text has already passed through the field-aware
                # correction overlay; replaying its broad role inference here
                # duplicates a parent account failure on every monthly child.
                role = _issue_role(field_name)
                if role and not _is_valid_for_role(str(value), role):
                    invalid.append((field_name, value, "canonical_field_contract_failed"))

            if source_absent_fields:
                record = _replace_normalized(record, values)
                for snapshot_name in ("canonical_raw", "raw"):
                    snapshot = record.get(snapshot_name)
                    if isinstance(snapshot, dict):
                        for field_name in source_absent_fields:
                            snapshot.pop(field_name, None)

            seen_fields: set[str] = set()
            for field_name, observed, issue_code in invalid:
                if field_name in seen_fields:
                    continue
                seen_fields.add(field_name)
                if issue_code == "canonical_field_outside_closed_catalog":
                    record, retained = _remove_noncanonical_field(record, field_name)
                else:
                    record, retained = _withhold(record, field_name)
                target_field_key = (dataset_name, record_id, field_name)
                if target_field_key not in actionable_target_fields:
                    generated.append(
                        make_issue(
                            category=(
                                "schema_incompleteness"
                                if observed in (None, "")
                                or issue_code
                                in {
                                    "canonical_source_sentinel_withheld",
                                    "canonical_field_outside_closed_catalog",
                                    "currency_amount_unit_conflict",
                                }
                                else "ocr_cell_level_error"
                            ),
                            issue_code=issue_code,
                            message="A required or typed canonical value was not safely extractable; its normalized value was withheld.",
                            parser_stage="v2_post_projection_gate",
                            target_dataset=dataset_name,
                            target_record_id=record_id,
                            field_name=field_name,
                            observed_value=retained,
                            reason_codes=(
                                "canonical_schema_gate",
                                "raw_evidence_preserved",
                                "normalized_value_withheld",
                            ),
                        )
                    )
                    actionable_target_fields.add(target_field_key)
            checked_rows.append(record)
        projected[dataset_name] = checked_rows

    generated_signatures = {
        (
            str(_normalized(record).get("target_dataset") or ""),
            str(_normalized(record).get("field_name") or ""),
            json.dumps(
                _normalized(record).get("observed_value"),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )
        for record in generated
    }
    issues = [
        record
        for record in issues
        if _normalized(record).get("target_record_id")
        or (
            str(_normalized(record).get("target_dataset") or ""),
            str(_normalized(record).get("field_name") or ""),
            json.dumps(
                _normalized(record).get("observed_value"),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )
        not in generated_signatures
    ]
    issues.extend(generated)
    issues = _link_unique_issue_records(projected, issues)
    unique_issues: dict[str, dict[str, Any]] = {}
    for record in issues:
        values = _extraction_issue(_normalized(record))
        target_field_key = actionable_target_field_key(record)
        marker_payload = (
            {
                "target_dataset": target_field_key[0],
                "target_record_id": target_field_key[1],
                "field_name": target_field_key[2],
            }
            if target_field_key is not None
            else {
                "target_dataset": values.get("target_dataset"),
                "target_record_id": values.get("target_record_id"),
                "field_name": values.get("field_name"),
                "observed_value": values.get("observed_value"),
                "issue_code": values.get("issue_code"),
            }
        )
        marker = json.dumps(
            marker_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        unique_issues.setdefault(marker, _replace_normalized(record, values))
    if unique_issues:
        projected["extraction_issues"] = list(unique_issues.values())
    else:
        projected.pop("extraction_issues", None)

    observations = [
        record
        for record in projected.get("field_observations") or []
        if isinstance(record, dict)
    ]
    for record in unique_issues.values():
        values = _normalized(record)
        field_name = str(values.get("field_name") or "")
        target = str(values.get("target_dataset") or "")
        if not field_name or not target:
            continue
        issue_id = str(values.get("extraction_issue_id") or record.get("record_id") or len(observations) + 1)
        observed = values.get("observed_value")
        observation = {
            "record_id": f"field_observation:{issue_id}",
            "field_observation_id": f"field_observation:{issue_id}",
            "dataset_name": target,
            "business_record_id": str(values.get("target_record_id") or "unresolved_record"),
            "field_name": field_name,
            "observation_status": "unreadable",
            "reason": str(values.get("issue_code") or "extraction_uncertain"),
            **({"raw_value": observed} if observed not in (None, "", []) else {}),
        }
        observations.append(observation)
    unique_observations: dict[str, dict[str, Any]] = {}
    for record in observations:
        values = _field_observation(_normalized(record))
        observation_target = (
            str(values.get("dataset_name") or ""),
            str(values.get("business_record_id") or ""),
            str(values.get("field_name") or ""),
        )
        if observation_target in source_absence_targets and (
            str(values.get("observation_status") or "") == "explicitly_absent"
            or str(values.get("reason") or "")
            in _SOURCE_ABSENCE_SUPERSEDED_ISSUE_CODES
        ):
            continue
        raw_value = values.get("raw_value")
        role = _issue_role(str(values.get("field_name") or ""))
        if role and raw_value not in (None, "") and validate_pboc_field(str(raw_value), role).valid:
            continue
        marker = json.dumps(
            {
                "dataset_name": values.get("dataset_name"),
                "business_record_id": values.get("business_record_id"),
                "field_name": values.get("field_name"),
                "raw_value": raw_value,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        unique_observations.setdefault(marker, _replace_normalized(record, values))
    if unique_observations:
        projected["field_observations"] = list(unique_observations.values())
    else:
        projected.pop("field_observations", None)

    statuses = list(projected.get("dataset_status") or [])
    status_index = {
        str(_normalized(record).get("dataset_name") or ""): index
        for index, record in enumerate(statuses)
    }
    affected = {
        str(_normalized(record).get("target_dataset") or "")
        for record in unique_issues.values()
        if str(_normalized(record).get("target_dataset") or "") in PBOC_DATASET_ORDER
        and str(_normalized(record).get("target_dataset") or "") not in _CONTROL_DATASETS
    }
    for target in sorted(affected):
        existing_values = (
            _normalized(statuses[status_index[target]])
            if target in status_index
            else {}
        )
        observed_row_count = len(projected.get(target) or [])
        source_presence = str(existing_values.get("presence_status") or "")
        if source_presence == "extraction_failed":
            merged_presence = "extraction_failed"
        elif source_presence == "unknown" and observed_row_count == 0:
            merged_presence = "unknown"
        else:
            merged_presence = "partial"
        normalized = {
            **existing_values,
            "dataset_status_record_id": f"dataset_status:{target}",
            "dataset_name": target,
            "applicability": "applicable",
            "presence_status": merged_presence,
            "observed_row_count": observed_row_count,
            "reason": str(existing_values.get("reason") or "unresolved_extraction_issue"),
        }
        if target == "credit_account_monthly_performance":
            withheld_count = sum(
                int(observed.get("withheld_month_count") or 0)
                for issue in unique_issues.values()
                if str(_normalized(issue).get("issue_code") or "")
                == "candidate_b_monthly_status_grid_unresolved"
                and isinstance(
                    observed := _normalized(issue).get("observed_value"), Mapping
                )
                and isinstance(observed.get("withheld_month_count"), int)
                and not isinstance(observed.get("withheld_month_count"), bool)
            )
            materialized_population = normalized["observed_row_count"] + withheld_count
            structural_expected_counts: list[int] = []
            for issue in unique_issues.values():
                issue_values = _normalized(issue)
                issue_code = str(issue_values.get("issue_code") or "")
                if issue_code == "monthly_population_incomplete_from_account_gap":
                    observed = issue_values.get("observed_value")
                    canonical_grid_count = (
                        observed.get("canonical_grid_row_count")
                        if isinstance(observed, Mapping)
                        else None
                    )
                    if (
                        isinstance(canonical_grid_count, int)
                        and not isinstance(canonical_grid_count, bool)
                        and canonical_grid_count >= 0
                    ):
                        structural_expected_counts.append(canonical_grid_count)
                    continue
                if issue_code != "canonical_monthly_reconstruction_incomplete":
                    continue
                candidate = issue_values.get("candidate_value")
                observed = issue_values.get("observed_value")
                if not isinstance(candidate, Mapping):
                    continue
                structural_expected = candidate.get("structural_expected_row_count")
                if (
                    isinstance(structural_expected, int)
                    and not isinstance(structural_expected, bool)
                    and structural_expected >= 0
                ):
                    structural_expected_counts.append(structural_expected)
                missing = candidate.get("missing_month_count")
                if (
                    not isinstance(missing, int)
                    or isinstance(missing, bool)
                    or missing < 0
                ):
                    continue
                canonical_count = (
                    observed.get("canonical_row_count")
                    if isinstance(observed, Mapping)
                    else None
                )
                structural_expected_counts.append(
                    (
                        canonical_count
                        if isinstance(canonical_count, int)
                        and not isinstance(canonical_count, bool)
                        and canonical_count >= 0
                        else materialized_population
                    )
                    + missing
                )
            if withheld_count or structural_expected_counts:
                source_expected = existing_values.get("expected_row_count")
                normalized["expected_row_count"] = max(
                    int(source_expected)
                    if isinstance(source_expected, int)
                    and not isinstance(source_expected, bool)
                    and source_expected >= 0
                    else 0,
                    materialized_population,
                    *structural_expected_counts,
                )
        if target in status_index:
            index = status_index[target]
            statuses[index] = _replace_normalized(statuses[index], normalized)
        else:
            statuses.append({"record_id": normalized["dataset_status_record_id"], **normalized})
    if statuses:
        projected["dataset_status"] = statuses
    return projected


_OBSERVED_INQUIRY_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[./:-]?(\d{2})[./:-](\d{2})(?!\d)"
)


def _compact_inquiry_noise(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or ""), flags=re.UNICODE)


def _bounded_inquiry_edge_equivalent(
    observed: Any,
    canonical: Any,
    *,
    max_noise: int,
    max_han_noise: int,
) -> bool:
    """Match a canonical scalar surrounded only by a tiny OCR edge residue."""

    observed_text = re.sub(r"\s+", "", str(observed or ""))
    canonical_text = re.sub(r"\s+", "", str(canonical or ""))
    if not observed_text or not canonical_text:
        return False
    if observed_text == canonical_text:
        return True
    if observed_text.count(canonical_text) != 1:
        return False
    prefix, suffix = observed_text.split(canonical_text, 1)
    residue = _compact_inquiry_noise(f"{prefix}{suffix}")
    return bool(
        len(residue) <= max_noise
        and len(re.findall(r"[\u3400-\u9fff]", residue)) <= max_han_noise
    )


def _bounded_observed_inquiry_date(value: Any) -> str | None:
    """Recover one date token only when the remaining cell noise is bounded."""

    text = str(value or "")
    matches = list(_OBSERVED_INQUIRY_DATE_RE.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    candidate = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    if not valid_iso_date(candidate):
        return None
    residue = _compact_inquiry_noise(f"{text[:match.start()]}{text[match.end():]}")
    if len(residue) > 3 or len(re.findall(r"[\u3400-\u9fff]", residue)) > 1:
        return None
    return candidate


def _reconcile_resolved_targetless_inquiry_issues(
    projected: dict[str, list[dict[str, Any]]],
) -> None:
    """Drop only stale malformed-row issues proved by one finalized inquiry row."""

    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        normalize_institution_name,
        normalize_role_candidate,
    )

    issues = list(projected.get("extraction_issues") or [])
    if not issues:
        return

    institutional_by_sequence: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for index, record in enumerate(projected.get("inquiries") or [], start=1):
        if not isinstance(record, dict):
            continue
        values = _normalized(record)
        channel = str(values.get("query_channel") or values.get("inquiry_type") or "")
        sequence = values.get("sequence")
        inquiry_date = str(values.get("inquiry_date") or "")
        institution = str(values.get("institution") or "")
        reason = str(values.get("reason") or "")
        extraction_status = str(values.get("extraction_status") or "").lower()
        if (
            channel != "institution"
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence <= 0
            or extraction_status
            in {"review", "failed", "partial", "unknown", "incomplete"}
            or not valid_iso_date(inquiry_date)
            or not validate_pboc_field(institution, "institution_name").valid
            or not validate_pboc_field(reason, "inquiry_reason").valid
        ):
            continue
        record_id = _record_identity(record, "inquiries", index)
        institutional_by_sequence.setdefault(sequence, []).append((record_id, values))

    issue_blocked_record_ids = {
        str(values.get("target_record_id") or "")
        for issue in issues
        if isinstance(issue, dict)
        and (values := _normalized(issue))
        and str(values.get("target_dataset") or "") == "inquiries"
        and str(values.get("target_record_id") or "")
        and _actionable_issue(issue)
    }

    reconciled: list[dict[str, Any]] = []
    for issue in issues:
        values = _normalized(issue)
        if not (
            str(values.get("issue_code") or "")
            == "candidate_b_inquiry_row_cells_unresolved"
            and str(values.get("target_dataset") or "") == "inquiries"
            and not str(values.get("target_record_id") or "")
        ):
            reconciled.append(issue)
            continue
        observed = values.get("observed_value")
        row = observed.get("row") if isinstance(observed, Mapping) else None
        sequence = observed.get("sequence") if isinstance(observed, Mapping) else None
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not isinstance(row, (list, tuple))
            or len(row) != 4
        ):
            reconciled.append(issue)
            continue
        row_sequence = re.fullmatch(r"\D*(\d{1,4})\D*", str(row[0] or ""))
        candidates = institutional_by_sequence.get(sequence) or []
        if (
            row_sequence is None
            or int(row_sequence.group(1)) != sequence
            or len(candidates) != 1
        ):
            reconciled.append(issue)
            continue
        record_id, final = candidates[0]
        observed_date = _bounded_observed_inquiry_date(row[1])
        observed_institution = normalize_institution_name(str(row[2] or ""))
        observed_reason = normalize_role_candidate(row[3], "inquiry_reason")
        resolved = bool(
            record_id not in issue_blocked_record_ids
            and observed_date == final.get("inquiry_date")
            and _bounded_inquiry_edge_equivalent(
                observed_institution,
                final.get("institution"),
                max_noise=3,
                max_han_noise=1,
            )
            and _bounded_inquiry_edge_equivalent(
                observed_reason,
                final.get("reason"),
                max_noise=3,
                max_han_noise=2,
            )
        )
        if not resolved:
            reconciled.append(issue)

    if reconciled:
        projected["extraction_issues"] = reconciled
    else:
        projected.pop("extraction_issues", None)


def project_personal_detail_datasets(
    datasets: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Return only closed-catalog PBOC v2 datasets and explicit uncertainties."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue

    source = {name: list(rows or []) for name, rows in datasets.items()}
    employment_rows, blob_issues = _employment_source_records(source.get("employment_records") or [])
    if employment_rows:
        source["employment_records"] = employment_rows
    if blob_issues:
        source.setdefault("personal_detail_extraction_issues", []).extend(blob_issues)
    projected: dict[str, list[dict[str, Any]]] = {}

    metadata = source.get("personal_report_metadata") or []
    if metadata:
        projected["report_metadata"] = _project_records(metadata, _report_metadata)
        projected["report_query"] = _project_report_queries(metadata)

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
            projected[new_name] = (
                _project_credit_agreements(rows)
                if old_name == "credit_lines"
                else _project_records(rows, direct_transforms.get(old_name))
            )

    accounts = source.get("credit_accounts") or []
    if accounts:
        projected["credit_accounts"] = _project_credit_accounts(accounts)
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
        projected["repayment_responsibilities"] = _project_repayment_responsibilities(
            responsibilities
        )

    metrics = source.get("personal_detail_credit_summary_metrics") or []
    mapped_metrics: list[dict[str, Any]] = []
    closed_world_issues: list[dict[str, Any]] = []
    for index, record in enumerate(metrics, start=1):
        values = _normalized(record)
        mapping_status = str(values.get("mapping_status") or "")
        if mapping_status == "unmapped" or not values.get("metric_code"):
            source_id = str(
                values.get("credit_summary_metric_id")
                or record.get("record_id")
                or f"summary_metric:{index}"
            )
            closed_world_issues.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="canonical_summary_metric_unresolved",
                    message="A summary cell could not be assigned to the finite canonical metric catalog and was withheld.",
                    parser_stage="v2_closed_world_projection",
                    target_dataset="personal_detail_credit_summary_metrics",
                    target_record_id=source_id,
                    field_name="metric_code",
                    observed_value={
                        "summary_type": values.get("summary_type"),
                        "metric_name": values.get("metric_name"),
                        "source_value": values.get("source_value") or values.get("value"),
                    },
                    reason_codes=(
                        "closed_canonical_catalog",
                        "summary_metric_mapping_unresolved",
                        "record_withheld",
                    ),
                )
            )
            continue
        mapped_metrics.append(record)
    summary_target_ids: set[str] = set()
    summary_source_target_remap: dict[str, str] = {}
    summary_target_values: dict[str, dict[str, Any]] = {}
    if mapped_metrics:
        overview_rows = _project_records(mapped_metrics, _summary_metric)
        projected["credit_business_overview"] = overview_rows
        source_cells_by_identity = {
            (
                str(values.get("summary_record_id") or ""),
                values.get("row_index"),
                values.get("column_index"),
            ): str(values.get("summary_cell_id") or record.get("record_id") or "")
            for record in source.get("personal_detail_summary_cells") or ()
            if isinstance(record, dict)
            and (values := _normalized(record))
        }
        for source_metric, overview_row in zip(mapped_metrics, overview_rows, strict=True):
            metric_values = _normalized(source_metric)
            target_id = str(
                overview_row.get("record_id")
                or _normalized(overview_row).get("credit_business_overview_id")
                or ""
            )
            if not target_id:
                continue
            summary_target_ids.add(target_id)
            summary_target_values[target_id] = _normalized(overview_row)
            source_cell_id = source_cells_by_identity.get(
                (
                    str(metric_values.get("summary_record_id") or ""),
                    metric_values.get("row_index"),
                    metric_values.get("column_index"),
                ),
                "",
            )
            if source_cell_id:
                summary_source_target_remap[source_cell_id] = target_id

    def remap_summary_business_target(
        values: dict[str, Any], *, id_field: str
    ) -> dict[str, Any]:
        """Bind source summary cells only to actually emitted metric rows."""

        values = dict(values)
        if values.get("target_dataset", values.get("dataset_name")) != "credit_business_overview":
            return values
        source_id = str(values.get(id_field) or "")
        if not source_id:
            return values
        target_id = (
            source_id
            if source_id in summary_target_ids
            else summary_source_target_remap.get(source_id)
        )
        if target_id:
            values[id_field] = target_id
            # Source summary cells and cross-dataset checks may name a generic
            # or prematurely selected value slot.  The emitted metric owns one
            # typed slot; derive it from the closed metric metadata even when
            # the value itself was withheld and ``value_type`` became unknown.
            if values.get("field_name") in {
                "value",
                "text_value",
                "numeric_value",
                "date_value",
            }:
                target_values = summary_target_values.get(target_id, {})
                value_type = str(target_values.get("value_type") or "")
                metric_code = str(target_values.get("metric_code") or "")
                mapping_status = str(target_values.get("mapping_status") or "")
                if value_type in {"integer", "decimal", "number", "money"}:
                    values["field_name"] = "numeric_value"
                elif value_type == "date":
                    values["field_name"] = "date_value"
                elif value_type == "text":
                    values["field_name"] = "text_value"
                elif mapping_status == "mapped" and metric_code:
                    if metric_code in {
                        "business_type",
                        "account_type",
                        "information_type",
                    }:
                        values["field_name"] = "text_value"
                    elif metric_code == "first_business_issue_month":
                        values["field_name"] = "date_value"
                    else:
                        values["field_name"] = "numeric_value"
        else:
            # A summary/table anchor or an unmapped cell has no public
            # business row.  Keep its uncertainty dataset-scoped instead of
            # publishing a dangling source-private identifier.
            values.pop(id_field, None)
        return values

    def project_extraction_issue(values: dict[str, Any]) -> dict[str, Any]:
        return remap_summary_business_target(
            _extraction_issue(values), id_field="target_record_id"
        )

    def project_field_observation(values: dict[str, Any]) -> dict[str, Any]:
        return remap_summary_business_target(
            _field_observation(values), id_field="business_record_id"
        )

    for index, record in enumerate(source.get("personal_detail_account_events") or [], start=1):
        target = _event_target(_normalized(record))
        if target is None:
            values = _normalized(record)
            closed_world_issues.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="canonical_account_event_type_unresolved",
                    message="An account event was outside the finite canonical event catalog and was withheld.",
                    parser_stage="v2_closed_world_projection",
                    target_record_id=str(
                        values.get("account_event_id")
                        or record.get("record_id")
                        or f"personal_detail_account_events:{index}"
                    ),
                    field_name="event_type",
                    observed_value=values,
                    reason_codes=(
                        "closed_canonical_catalog",
                        "unknown_account_event_type",
                        "record_withheld",
                    ),
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

    field_observations = source.get("personal_detail_field_observations") or []
    if field_observations:
        projected["field_observations"] = _project_records(
            field_observations, project_field_observation
        )

    extraction_issues = source.get("personal_detail_extraction_issues") or []
    if extraction_issues or closed_world_issues:
        projected["extraction_issues"] = _project_records(
            [*extraction_issues, *closed_world_issues], project_extraction_issue
        )
        _reconcile_resolved_targetless_inquiry_issues(projected)

    typed_public, public_issues = _project_closed_world_public_records(
        source.get("public_records") or [], projected
    )
    for name, rows in typed_public.items():
        projected.setdefault(name, []).extend(rows)

    if public_issues:
        projected.setdefault("extraction_issues", []).extend(
            _project_records(public_issues, project_extraction_issue)
        )
    mapped_metric_cells = {
        (
            _normalized(row).get("summary_record_id"),
            _normalized(row).get("row_index"),
            _normalized(row).get("column_index"),
        )
        for row in mapped_metrics
    }
    for index, record in enumerate(source.get("personal_detail_summary_cells") or [], start=1):
        values = _normalized(record)
        identity = (
            values.get("summary_record_id"),
            values.get("row_index"),
            values.get("column_index"),
        )
        if identity not in mapped_metric_cells:
            closed_world_issues.append(
                make_issue(
                    category="schema_incompleteness",
                    issue_code="canonical_summary_cell_unmapped",
                    message="A recognized summary cell had no typed canonical metric and was withheld.",
                    parser_stage="v2_closed_world_projection",
                    target_dataset="personal_detail_summary_cells",
                    target_record_id=str(
                        values.get("summary_cell_id")
                        or record.get("record_id")
                        or f"personal_detail_summary_cells:{index}"
                    ),
                    field_name=str(values.get("column_label") or "value"),
                    observed_value=values.get("value"),
                    reason_codes=(
                        "closed_canonical_catalog",
                        "typed_summary_metric_missing",
                        "normalized_value_withheld",
                    ),
                )
            )
    if closed_world_issues:
        existing_issue_ids = {
            str(_normalized(issue).get("extraction_issue_id") or issue.get("record_id") or "")
            for issue in projected.get("extraction_issues") or []
        }
        projected.setdefault("extraction_issues", []).extend(
            issue
            for issue in _project_records(closed_world_issues, project_extraction_issue)
            if str(_normalized(issue).get("extraction_issue_id") or issue.get("record_id") or "")
            not in existing_issue_ids
        )

    status_rows = source.get("personal_detail_dataset_status") or []
    if status_rows:
        projected["dataset_status"] = _project_dataset_status(status_rows, projected)

    projected = _canonical_quality_gate(projected)
    if projected.get("extraction_issues"):
        compact_issues, evidence_rows = _issue_evidence_rows(projected["extraction_issues"])
        projected["extraction_issues"] = compact_issues
        if evidence_rows:
            projected["extraction_issue_evidence"] = evidence_rows
    review_fields_by_target: dict[tuple[str, str], set[str]] = {}
    actionable_review_targets: set[tuple[str, str]] = set()
    for issue in projected.get("extraction_issues") or ():
        if not isinstance(issue, dict) or not _actionable_issue(issue):
            continue
        values = _normalized(issue)
        dataset_name = str(values.get("target_dataset") or "")
        record_id = str(values.get("target_record_id") or "")
        field_name = str(values.get("field_name") or "")
        if dataset_name and record_id:
            target = (dataset_name, record_id)
            actionable_review_targets.add(target)
            if field_name:
                review_fields_by_target.setdefault(target, set()).add(field_name)
    for observation in projected.get("field_observations") or ():
        if not isinstance(observation, dict):
            continue
        values = _normalized(observation)
        if str(values.get("observation_status") or "") in {
            "ocr_corrected",
            "resolved",
            "dismissed",
            "explicitly_absent",
        }:
            continue
        dataset_name = str(values.get("dataset_name") or "")
        record_id = str(values.get("business_record_id") or "")
        field_name = str(values.get("field_name") or "")
        if dataset_name and record_id:
            target = (dataset_name, record_id)
            actionable_review_targets.add(target)
            if field_name:
                review_fields_by_target.setdefault(target, set()).add(field_name)
    for dataset_name, rows in projected.items():
        if dataset_name in _QUALITY_GATE_EXEMPT_DATASETS:
            continue
        for index, record in enumerate(rows, start=1):
            if isinstance(record, dict):
                record_id = _record_identity(record, dataset_name, index)
                review_target = (dataset_name, record_id)
                _sanitize_projected_business_record(
                    record,
                    dataset_name=dataset_name,
                    review_fields=frozenset(
                        review_fields_by_target.get(review_target, ())
                    ),
                    record_has_actionable_issue=(
                        review_target in actionable_review_targets
                    ),
                )
    # Candidate B rows may retain a canonical_raw evidence pool beside their
    # flat compatibility fields. Community serialization treats such a pool
    # as authoritative when no explicit normalized pool exists, which would
    # otherwise discard schema-derived identities and enums. Add the explicit
    # pool only after all internal linking/gating has finished so the flat
    # compatibility view remains usable during projection.
    for rows in projected.values():
        for record in rows:
            if (
                isinstance(record, dict)
                and not isinstance(record.get("normalized"), dict)
                and (
                    isinstance(record.get("canonical_raw"), dict)
                    or isinstance(record.get("raw"), dict)
                )
            ):
                record["normalized"] = _normalized(record)
    reportable_empty_datasets = {
        str(values.get("dataset_name") or "")
        for record in projected.get("dataset_status") or ()
        if isinstance(record, dict)
        and (values := _normalized(record))
        and str(values.get("applicability") or "") == "applicable"
        and str(values.get("presence_status") or "")
        in {"partial", "extraction_failed", "unknown"}
        and isinstance(values.get("expected_row_count"), int)
        and not isinstance(values.get("expected_row_count"), bool)
        and int(values["expected_row_count"]) > 0
        and int(values.get("observed_row_count") or 0) == 0
    }
    return {
        name: projected.get(name, [])
        for name in PBOC_DATASET_ORDER
        if (name in projected and projected[name]) or name in reportable_empty_datasets
    }


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


def personal_detail_data_dictionary() -> dict[str, Any]:
    """Return the PBOC-native dataset and logical-type dictionary."""
    from docmirror.plugins.credit_report.personal_detail_scanned.source_schema_components import (
        personal_detail_source_dictionary_components,
    )

    source_components = personal_detail_source_dictionary_components()
    source_datasets = source_components.get("datasets") or {}
    datasets: dict[str, Any] = {}
    for old_name, new_name in _DIRECT_DATASET_RENAMES.items():
        if old_name in source_datasets:
            datasets[new_name] = deepcopy(source_datasets[old_name])
    for unchanged in ("credit_accounts",):
        datasets[unchanged] = deepcopy(source_datasets[unchanged])

    datasets["report_metadata"] = deepcopy(source_datasets["personal_report_metadata"])
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
    for redundant_phone in ("mobile_phone", "work_phone", "residence_phone"):
        datasets["subject_profile"]["columns"].pop(redundant_phone, None)
    datasets["subject_profile"]["columns"]["subject_profile_id"] = _descriptor("主体资料ID")
    for dataset_name in ("subject_residences", "subject_employment"):
        datasets[dataset_name]["columns"].pop("page", None)
        datasets[dataset_name]["columns"].pop("source_page", None)
    account_columns = datasets["credit_accounts"]["columns"]
    for deprecated_alias in ("institution", "currency", "activation_state", "status"):
        account_columns.pop(deprecated_alias, None)
    if "validity_type" in account_columns:
        account_columns["credit_line_validity_type"] = account_columns.pop(
            "validity_type"
        )
    if "close_date" in account_columns:
        account_columns["close_date"] = _descriptor("结清或销户日期", "date")
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
    for deprecated_alias in ("account_id", "account_state", "payoff_state", "status"):
        agreement_columns.pop(deprecated_alias, None)
    agreement_columns["sequence"] = _descriptor("组内序号", "integer")
    agreement_columns["credit_agreement_id"] = _descriptor("授信协议ID")
    agreement_columns["limit_identifier"] = _descriptor("授信限额编号")
    agreement_columns["facility_limit"] = _descriptor(
        "授信额度", "money", unit="yuan"
    )
    agreement_columns["credit_limit"] = _descriptor(
        "授信限额", "money", unit="yuan"
    )
    datasets["credit_account_monthly_performance"] = deepcopy(source_datasets["repayment_records"])
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
    datasets["repayment_responsibilities"] = deepcopy(source_datasets["repayment_liability_records"])
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
    datasets["postpaid_monthly_performance"] = deepcopy(source_datasets["postpaid_payment_history"])
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
        source_datasets["personal_detail_credit_summary_metrics"]
    )
    for source_key in (
        "credit_summary_metric_id",
        "row_dimension_name",
        "row_dimension_value",
    ):
        datasets["credit_business_overview"]["columns"].pop(source_key, None)
    datasets["credit_business_overview"]["columns"].update(
        {
            "credit_business_overview_id": _descriptor("信用业务概要ID"),
            "business_category": _descriptor("业务类别"),
            "business_type": _descriptor("业务类型"),
            "business_dimension_name": _descriptor("业务维度名称"),
            "business_dimension_value": _descriptor("业务维度值"),
        }
    )

    event_source = source_datasets["personal_detail_account_events"]
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
    note_columns = deepcopy(source_datasets["statements"]["columns"])
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
    datasets["field_observations"] = deepcopy(
        source_datasets["personal_detail_field_observations"]
    )
    datasets["field_observations"]["columns"]["raw_value_type"] = _descriptor(
        "源观测值容器类型", "enum"
    )
    datasets["extraction_issues"] = deepcopy(
        source_datasets["personal_detail_extraction_issues"]
    )
    datasets["extraction_issues"]["columns"].update(
        {
            "observed_value_type": _descriptor("观测证据容器类型", "enum"),
            "candidate_value_type": _descriptor("候选证据容器类型", "enum"),
            "reason_code_count": _descriptor("原因代码数", "integer"),
        }
    )
    datasets["extraction_issue_evidence"] = {
        "definition": "Typed scalar leaves of nested extraction-issue evidence, keyed to one extraction issue.",
        "columns": {
            "extraction_issue_evidence_id": _descriptor("问题证据ID"),
            "extraction_issue_id": _descriptor("提取问题ID"),
            "evidence_kind": _descriptor("证据类别", "enum"),
            "evidence_path": _descriptor("证据路径"),
            "value_type": _descriptor("值类型", "enum"),
            "string_value": _descriptor("字符串值", "text"),
            "integer_value": _descriptor("整数值", "integer"),
            "number_value": _descriptor("数值", "number"),
            "boolean_value": _descriptor("布尔值", "boolean"),
        },
    }
    datasets["pboc_extension_fields"] = {
        "definition": "Reserved for explicitly cataloged, structured PBOC extensions; never a fallback for failed canonical decoding.",
        "columns": {
            "pboc_extension_field_id": _descriptor("人行业务扩展字段ID"),
            "namespace": _descriptor("扩展命名空间"),
            "source_dataset": _descriptor("源数据集"),
            "source_record_id": _descriptor("源记录ID"),
            "field_name": _descriptor("字段名称"),
            "value": _descriptor("字段值", "text"),
        },
    }
    datasets["dataset_status"] = deepcopy(source_datasets["personal_detail_dataset_status"])
    datasets["dataset_status"]["columns"].pop("dataset_status_id", None)
    datasets["dataset_status"]["columns"].update(
        {
            "dataset_status_record_id": _descriptor("数据集状态ID"),
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
    for name in _PUBLIC_STATUS_TARGETS:
        if name in datasets:
            datasets[name]["columns"].pop("unmapped_content", None)
    for source_key in ("personal_housing_fund_id", "personal_contribution_ratio", "employer_contribution_ratio"):
        datasets["housing_fund_records"]["columns"].pop(source_key, None)
    for source_key in ("effective_date", "end_date"):
        datasets["administrative_penalty_records"]["columns"].pop(source_key, None)
        datasets["administrative_award_records"]["columns"].pop(source_key, None)
    for source_key in ("obtained_date", "expiry_date", "revocation_date"):
        datasets["professional_qualification_records"]["columns"].pop(source_key, None)

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
            "extension_policy": "pboc_extension_fields accepts only explicitly cataloged structured extensions; unknown canonical content is withheld with an extraction issue.",
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
        "fields": deepcopy(source_components.get("fields") or {}),
        "datasets": datasets,
        "enums": {
            **deepcopy(source_components.get("enums") or {}),
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


def personal_detail_semantic_extensions() -> dict[str, Any]:
    """Return v2 contract identity and Community presentation policy."""
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_semantic_extensions,
    )

    semantic = credit_report_semantic_extensions(report_subtype="personal_detail")
    semantic["domain_schema"] = {
        "id": PBOC_SCHEMA_ID,
        "version": PBOC_SCHEMA_VERSION,
        "contract_uri": PBOC_CONTRACT_URI,
        "compatibility": "canonical-v2; community-v3-envelope; detailed-report-only",
        "dataset_status_semantics": deepcopy(_SPARSE_DATASET_STATUS_SEMANTICS),
    }
    semantic["dataset_document_order"] = list(PBOC_DATASET_ORDER)
    semantic["dataset_reading_columns"] = {
        "report_metadata": ["report_number", "report_time", "subject_name"],
        "report_query": ["query_institution", "query_reason"],
        "subject_profile": ["subject_name", "gender", "birth_date", "marital_status"],
        "credit_accounts": [
            "pboc_account_type_code",
            "account_identifier",
            "management_institution",
            "business_type",
            "account_lifecycle_state",
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
        "field_observations": [
            "dataset_name",
            "business_record_id",
            "field_name",
            "observation_status",
            "raw_value",
            "normalized_value",
            "confidence",
            "reason",
        ],
        "extraction_issues": [
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
        "extraction_issue_evidence": [
            "extraction_issue_id",
            "evidence_kind",
            "evidence_path",
            "value_type",
            "string_value",
            "integer_value",
            "number_value",
            "boolean_value",
        ],
    }
    dictionary = personal_detail_data_dictionary()
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
    foreign_keys["extraction_issue_evidence"] = [
        {
            "columns": ["extraction_issue_id"],
            "reference_dataset": "extraction_issues",
            "reference_columns": ["record_id"],
        }
    ]
    semantic["community_projection_overrides"] = {
        "dataset_labels": dict(_PBOC_DATASET_LABELS),
        "dataset_representation_roles": {
            name: ("control" if name in _CONTROL_DATASETS else "business_canonical")
            for name in PBOC_DATASET_ORDER
        },
        "dataset_grains": {
            name: f"one row per {_PBOC_DATASET_LABELS[name]}"
            for name in PBOC_DATASET_ORDER
        },
        # Only source-proven empty collections survive v2 projection.  Publish
        # those zero-row envelopes so their independently known omitted count
        # is visible instead of disappearing with the business rows.
        "publish_empty_datasets": list(PBOC_DATASET_ORDER),
        # Community's generic warning is only a projection-conservation check.
        # Source-document completeness is represented explicitly by dataset_status.
        "completeness": {
            name: {
                "basis": "domain_fact_count",
                "count_key": f"personal_detail_v2_expected_{name}_count",
                "public_basis": "personal_detail_v2_projection_row_conservation",
            }
            for name in PBOC_DATASET_ORDER
        },
        "internal_facts": [
            f"personal_detail_v2_expected_{name}_count"
            for name in PBOC_DATASET_ORDER
        ],
        "internal_fields": [
            f"personal_detail_v2_expected_{name}_count"
            for name in PBOC_DATASET_ORDER
        ],
        "dataset_foreign_keys": foreign_keys,
        "dataset_derived_from": {
            "report_query": ["report_metadata"],
            "credit_business_overview": ["personal_detail_credit_summary_metrics"],
            "pboc_extension_fields": ["explicit_cataloged_extensions"],
            "field_observations": ["assessed_source_fields"],
            "extraction_issues": ["ocr_audit", "page_topology_audit", "native_parser"],
            "extraction_issue_evidence": ["extraction_issues"],
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
            "field_observations": ["extraction_review"],
            "extraction_issues": ["extraction_review"],
            "extraction_issue_evidence": ["extraction_review"],
            "pboc_extension_fields": ["extraction_review"],
            "dataset_status": ["extraction_review"],
        },
    }
    semantic["personal_detail_contract"] = {
        "canonical_profile_dataset": "subject_profile",
        "canonical_credit_summary_dataset": "credit_business_overview",
        "canonical_public_record_datasets": [
            "tax_arrears_records",
            "civil_judgment_records",
            "enforcement_records",
            "administrative_penalty_records",
            "housing_fund_records",
            "social_assistance_records",
            "professional_qualification_records",
            "administrative_award_records",
        ],
        "absence_dataset": "dataset_status",
        "uncertainty_dataset": "field_observations",
        "extraction_issue_dataset": "extraction_issues",
        "absence_requires_explicit_source_evidence": True,
        "empty_dataset_means_absent": False,
        "uncertainty_coverage": {
            "mode": "potentially_flawed_only",
            "unlisted_dataset_default": "not_assessed",
            "confidence_policy": "nullable_when_source_confidence_unavailable",
        },
        "sparse_dataset_status_semantics": deepcopy(
            _SPARSE_DATASET_STATUS_SEMANTICS
        ),
    }
    assert set(semantic["dataset_reading_columns"]) <= set(dictionary["datasets"])
    return semantic


__all__ = [
    "PBOC_CONTRACT_URI",
    "PBOC_DATASET_ORDER",
    "PBOC_SCHEMA_ID",
    "PBOC_SCHEMA_VERSION",
    "personal_detail_data_dictionary",
    "personal_detail_semantic_extensions",
    "project_personal_detail_datasets",
]
