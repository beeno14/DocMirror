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

PERSONAL_DETAIL_SOURCE_BUSINESS_DATASETS = (
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
_PROFILE_OBSERVATION_STATUS_ALIASES = {"source_absent": "explicitly_absent"}
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
_POTENTIALLY_FLAWED_OBSERVATION_STATUSES = frozenset(
    {"ocr_corrected", "inferred", "ambiguous", "unreadable", "not_observed"}
)
_POTENTIALLY_FLAWED_DATASET_STATUSES = frozenset(
    {"not_observed", "partial", "extraction_failed", "unknown"}
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

# Closed catalog of the printed summary headings used by the canonical PBOC
# personal detailed report.  Matching below never edits a heading or applies
# an OCR-distance heuristic: it only recognizes one of these exact phrases
# inside otherwise contaminated title text.
_SUMMARY_TITLE_CATALOG: dict[str, tuple[str, ...]] = {
    "信用业务概要": ("信用业务概要",),
    "呆账": ("呆账信息汇总",),
    "逾期（透支）": ("逾期（透支）信息汇总",),
    "被追偿": ("被追偿信息汇总",),
    "非循环贷账户": ("非循环贷账户信息汇总",),
    "循环贷账户一": ("循环贷账户一信息汇总",),
    "循环贷账户二": ("循环贷账户二信息汇总",),
    "贷记卡账户": ("贷记卡账户信息汇总",),
    "准贷记卡账户": ("准贷记卡账户信息汇总",),
    "相关还款责任": ("相关还款责任信息汇总",),
    "后付费业务欠费": ("后付费业务欠费信息汇总",),
    "公共": ("公共信息概要", "公共信息汇总"),
    "查询记录概要": ("查询记录概要",),
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
_TEXT_METRIC_CODES = frozenset({"business_type", "account_type", "information_type"})
_DATE_METRIC_CODES = frozenset({"first_business_issue_month"})

# Text-valued summary cells are categorical business dimensions, not free
# text.  Keep the catalog scoped by both the canonical summary and the printed
# metric so that a valid value from one PBOC table cannot silently contaminate
# another table.  These are the categories printed by the canonical personal
# detailed report; no person- or document-specific value belongs here.
_SUMMARY_TEXT_VALUE_CATALOG: dict[tuple[str, str], tuple[str, ...]] = {
    (
        "credit_business_overview",
        "business_type",
    ): (
        "个人住房贷款",
        "个人商用房贷款（包括商住两用房）",
        "其他类贷款",
        "贷记卡",
        "准贷记卡",
        "合计",
    ),
    (
        "delinquency_overdraft",
        "account_type",
    ): (
        "非循环贷账户",
        "循环贷账户一",
        "循环贷账户二",
        "贷记卡账户",
        "准贷记卡账户",
    ),
    (
        "recovery",
        "business_type",
    ): (
        "资产处置业务",
        "垫款业务",
        "合计",
    ),
    (
        "postpaid_arrears",
        "business_type",
    ): (
        "电信业务",
        "水电气等公用事业",
    ),
    (
        "public_records",
        "information_type",
    ): (
        "欠税信息",
        "民事判决信息",
        "强制执行信息",
        "行政处罚信息",
    ),
}
_SUMMARY_EXACT_ONLY_TEXT_VALUES = frozenset({"合计"})

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


def _summary_catalog_text(value: Any) -> str:
    """Normalize typography only; do not guess or repair title glyphs."""
    return _compact_label(value).translate(
        str.maketrans({"(": "（", ")": "）", "﹙": "（", "﹚": "）"})
    )


def _summary_catalog_candidate(value: Any) -> tuple[int, str, str] | None:
    """Return the longest exact catalog phrase contained in one OCR title.

    Longest-match semantics are required because ``准贷记卡账户`` contains
    ``贷记卡账户``.  Equal-length matches for different summary types are
    intentionally rejected as ambiguous.
    """
    observed = _summary_catalog_text(value)
    if not observed:
        return None
    matches: list[tuple[int, int, str, str]] = []
    for summary_type, titles in _SUMMARY_TITLE_CATALOG.items():
        candidates = (summary_type, *titles)
        for rank, phrase in enumerate(candidates):
            token = _summary_catalog_text(phrase)
            # A bare type such as ``公共`` is too short to accept inside an
            # arbitrary heading. Only the complete printed title may carry
            # prefix/suffix OCR contamination; a bare type must be exact.
            matched = observed == token if rank == 0 else token in observed
            if token and matched:
                canonical_title = titles[0] if rank == 0 else phrase
                matches.append((len(token), rank, summary_type, canonical_title))
    if not matches:
        return None
    longest = max(match[0] for match in matches)
    winners = [match for match in matches if match[0] == longest]
    if len({match[2] for match in winners}) != 1:
        return None
    length, _rank, summary_type, canonical_title = min(
        winners,
        key=lambda match: match[1],
    )
    return length, summary_type, canonical_title


def _canonical_summary_identity(*values: Any) -> tuple[str, str] | None:
    """Resolve mutually consistent title evidence to one finite summary type."""
    candidates = [
        candidate
        for value in values
        if (candidate := _summary_catalog_candidate(value)) is not None
    ]
    if not candidates or len({candidate[1] for candidate in candidates}) != 1:
        return None
    _length, summary_type, canonical_title = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return summary_type, _summary_catalog_text(canonical_title)


def _canonical_summary_text_value(
    value: Any,
    *,
    summary_code: str | None,
    metric_code: str | None,
) -> tuple[str | None, str]:
    """Resolve one categorical summary value without OCR-distance guessing.

    An exact catalog value is trusted.  A catalog token surrounded only by
    ASCII OCR debris is normalized when it is the sole semantic match.  The
    latter safely covers cells such as ``2 贷记卡 n`` while rejecting Chinese
    prefix/suffix text, cross-category concatenation, and unknown values.
    """

    observed = _summary_catalog_text(value)
    if observed in _PLACEHOLDERS:
        return None, "placeholder"
    catalog = _SUMMARY_TEXT_VALUE_CATALOG.get(
        (str(summary_code or ""), str(metric_code or "")),
        (),
    )
    if not observed or not catalog:
        return None, "unknown"

    normalized_catalog = {
        _summary_catalog_text(candidate): candidate for candidate in catalog
    }
    if observed in normalized_catalog:
        return normalized_catalog[observed], "exact"

    contained = {
        token: canonical
        for token, canonical in normalized_catalog.items()
        if token
        and canonical not in _SUMMARY_EXACT_ONLY_TEXT_VALUES
        and token in observed
    }
    # Remove lexical submatches such as ``贷记卡`` inside ``准贷记卡``.  Two
    # unrelated surviving categories are real ambiguity, regardless of their
    # relative lengths.
    semantic_matches = {
        token: canonical
        for token, canonical in contained.items()
        if not any(token != other and token in other for other in contained)
    }
    if len(semantic_matches) != 1:
        return None, "ambiguous" if semantic_matches else "unknown"

    token, canonical = next(iter(semantic_matches.items()))
    if observed.count(token) != 1:
        return None, "ambiguous"
    prefix, suffix = observed.split(token, 1)
    residual = f"{prefix}{suffix}"
    if residual and not residual.isascii():
        return None, "unsafe_noise"
    return canonical, "normalized"


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
        if entry is None:
            continue
        entry_map = dict(entry) if isinstance(entry, Mapping) else {}
        normalized = entry_map.get("normalized_value", entry_map.get("value")) if entry_map else entry
        raw = entry_map.get("canonical_raw", entry_map.get("raw", normalized)) if entry_map else entry
        if field_name == "birth_date" and normalized not in (None, ""):
            normalized = _iso_date(normalized)
        refs = _source_refs(entry_map)
        all_refs.extend(refs)
        explicit_status = _PROFILE_OBSERVATION_STATUS_ALIASES.get(
            str(entry_map.get("observation_status") or ""),
            str(entry_map.get("observation_status") or ""),
        )
        absence_proven = bool(entry_map.get("source_statement") or entry_map.get("absence_evidence"))
        if explicit_status == "not_observed" and not absence_proven:
            if raw in (None, "") and not refs:
                continue
            status = "unreadable"
        elif explicit_status in _OBSERVATION_STATUSES:
            status = explicit_status
        elif normalized in (None, ""):
            if raw in (None, "") and not refs:
                continue
            status = "unreadable"
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
            observation["confidence_basis"] = str(entry_map.get("confidence_basis") or "source_field_confidence")
        else:
            observation["confidence_status"] = "not_available"
            observation["confidence_basis"] = "source_did_not_report_field_confidence"
        if status == "not_observed":
            observation["reason"] = str(entry_map.get("source_statement") or "source_proved_field_absence")
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


def _validate_summary_scalar_cells(datasets: dict[str, Any]) -> None:
    """Withhold mapped summary cells whose OCR text violates the metric type."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue

    issues = datasets.setdefault("personal_detail_extraction_issues", [])
    if not isinstance(issues, list):
        issues = []
        datasets["personal_detail_extraction_issues"] = issues
    existing = {
        str(issue.get("extraction_issue_id") or "")
        for issue in issues
        if isinstance(issue, Mapping)
    }
    summary_rows = {
        str(values.get("summary_record_id") or ""): values
        for row in (datasets.get("personal_detail_summary_records") or ())
        if (values := _record_values(row)).get("summary_record_id")
    }
    for index, source_row in enumerate(datasets.get("personal_detail_summary_cells") or (), start=1):
        if not isinstance(source_row, dict):
            continue
        cell = _record_values(source_row)
        metric_code = _METRIC_CODES.get(_compact_label(str(cell.get("column_label") or "")))
        if not metric_code:
            continue

        if metric_code in _TEXT_METRIC_CODES:
            if _summary_value(cell.get("value"))[2] == "not_reported":
                continue
            parent = summary_rows.get(str(cell.get("summary_record_id") or ""), {})
            identity = _canonical_summary_identity(
                cell.get("summary_type"),
                cell.get("title"),
                parent.get("summary_type"),
                parent.get("title"),
            )
            # Unmapped summary identities are quarantined and diagnosed by the
            # existing summary-mapping contract.  Avoid a redundant value
            # diagnostic until the table itself has a canonical identity.
            if identity is None:
                continue
            summary_code = _SUMMARY_CODES.get(identity[0])
            canonical, resolution = _canonical_summary_text_value(
                cell.get("value"),
                summary_code=summary_code,
                metric_code=metric_code,
            )
            if canonical is not None:
                continue
            cell["value_status"] = "unreadable"
            if isinstance(source_row.get("normalized"), dict):
                source_row["normalized"] = cell
            else:
                source_row.update(cell)
            issue = make_issue(
                category="ocr_cell_level_error",
                issue_code="candidate_b_summary_text_dimension_unresolved",
                message=(
                    "A mapped summary dimension was outside its finite canonical "
                    "business catalog; no normalized business value was emitted."
                ),
                parser_stage="candidate_b_summary_schema",
                target_dataset="personal_detail_summary_cells",
                target_record_id=str(cell.get("summary_cell_id") or f"summary_cell:{index}"),
                field_name="value",
                observed_value=cell.get("value"),
                source_refs=_source_refs(source_row),
                reason_codes=(
                    "mapped_text_dimension",
                    "finite_category_contract_failed",
                    f"text_value_{resolution}",
                    "normalized_value_withheld",
                ),
            )
            if issue["extraction_issue_id"] not in existing:
                issues.append(issue)
                existing.add(issue["extraction_issue_id"])
            continue

        value_type, _numeric, reporting_status = _summary_value(cell.get("value"))
        valid = reporting_status == "not_reported" or (
            value_type == "date" if metric_code in _DATE_METRIC_CODES else value_type in {"integer", "decimal"}
        )
        if valid:
            continue
        cell["value_status"] = "unreadable"
        if isinstance(source_row.get("normalized"), dict):
            source_row["normalized"] = cell
        else:
            source_row.update(cell)
        issue = make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_summary_scalar_unresolved",
            message="A mapped summary metric contained non-scalar OCR text; no normalized business value was emitted.",
            parser_stage="candidate_b_summary_schema",
            target_dataset="personal_detail_summary_cells",
            target_record_id=str(cell.get("summary_cell_id") or f"summary_cell:{index}"),
            field_name="value",
            observed_value=cell.get("value"),
            source_refs=_source_refs(source_row),
            reason_codes=("mapped_scalar_metric", "typed_cell_contract_failed", "normalized_value_withheld"),
        )
        if issue["extraction_issue_id"] not in existing:
            issues.append(issue)
            existing.add(issue["extraction_issue_id"])


def _summary_metric_contract(datasets: Mapping[str, Any]) -> list[dict[str, Any]]:
    reporting_currency, reporting_amount_unit = _reporting_context(datasets)
    cells = [_record_values(row) for row in (datasets.get("personal_detail_summary_cells") or [])]
    summary_rows = {
        str(values.get("summary_record_id") or ""): values
        for row in (datasets.get("personal_detail_summary_records") or [])
        if (values := _record_values(row)).get("summary_record_id")
    }
    dimensions: dict[tuple[str, int], tuple[str, str]] = {}
    for cell in sorted(
        cells,
        key=lambda item: (
            str(item.get("summary_record_id") or ""),
            int(item.get("row_index") or 0),
            int(item.get("column_index") or 0),
        ),
    ):
        key = (str(cell.get("summary_record_id") or ""), int(cell.get("row_index") or 0))
        label = str(cell.get("column_label") or "").strip()
        metric_code = _METRIC_CODES.get(_compact_label(label))
        if metric_code not in _TEXT_METRIC_CODES or cell.get("value_status") == "unreadable":
            continue
        parent = summary_rows.get(key[0], {})
        identity = _canonical_summary_identity(
            cell.get("summary_type"),
            cell.get("title"),
            parent.get("summary_type"),
            parent.get("title"),
        )
        summary_code = _SUMMARY_CODES.get(identity[0]) if identity is not None else None
        canonical_value, _resolution = _canonical_summary_text_value(
            cell.get("value"),
            summary_code=summary_code,
            metric_code=metric_code,
        )
        if canonical_value is not None:
            dimensions.setdefault(key, (label, canonical_value))

    metrics: list[dict[str, Any]] = []
    for index, source_row in enumerate(datasets.get("personal_detail_summary_cells") or [], start=1):
        cell = _record_values(source_row)
        summary_record_id = str(cell.get("summary_record_id") or "")
        row_index = int(cell.get("row_index") or 0)
        column_index = int(cell.get("column_index") or 0)
        source_id = str(cell.get("summary_cell_id") or f"summary_cell:{index}")
        metric_id = f"credit_summary_metric:{source_id}"
        dimension_name, dimension_value = dimensions.get((summary_record_id, row_index), ("", ""))
        parent = summary_rows.get(summary_record_id, {})
        identity = _canonical_summary_identity(
            cell.get("summary_type"),
            cell.get("title"),
            parent.get("summary_type"),
            parent.get("title"),
        )
        raw_summary_type = str(cell.get("summary_type") or "").strip()
        summary_type = identity[0] if identity is not None else raw_summary_type
        title = identity[1] if identity is not None else cell.get("title")
        metric_name = str(cell.get("column_label") or "").strip()
        summary_code = _SUMMARY_CODES.get(summary_type)
        metric_code = _METRIC_CODES.get(_compact_label(metric_name))
        source_value = cell.get("value")
        value_type, numeric_value, reporting_status = _summary_value(source_value)
        projected_text_value = source_value
        if metric_code in _TEXT_METRIC_CODES and reporting_status == "reported":
            canonical_value, _resolution = _canonical_summary_text_value(
                source_value,
                summary_code=summary_code,
                metric_code=metric_code,
            )
            if canonical_value is None:
                value_type, numeric_value, reporting_status = "unknown", None, "unknown"
            else:
                value_type, numeric_value = "text", None
                projected_text_value = canonical_value
        if cell.get("value_status") == "unreadable":
            value_type, numeric_value, reporting_status = "unknown", None, "unknown"
        if metric_code in _MONEY_METRIC_CODES and reporting_status == "reported":
            value_type = "money"
        metric: dict[str, Any] = {
            "record_id": metric_id,
            "credit_summary_metric_id": metric_id,
            "summary_record_id": summary_record_id,
            "summary_type": summary_type,
            "summary_code": summary_code,
            "title": title,
            "row_index": row_index,
            "column_index": column_index,
            "metric_name": metric_name,
            "metric_code": metric_code,
            "mapping_status": "mapped" if summary_code and metric_code else "unmapped",
            "row_dimension_name": dimension_name,
            "row_dimension_value": dimension_value,
            "value_type": value_type,
            "reporting_status": reporting_status,
        }
        if any(marker in dimension_name for marker in ("账户类型", "业务类型", "业务类别", "责任类型")):
            metric["business_category"] = dimension_value
        if parent.get("source_table_id"):
            metric["source_table_id"] = parent["source_table_id"]
        if numeric_value is not None:
            metric["numeric_value"] = numeric_value
        elif value_type == "date" and source_value not in (None, ""):
            metric["date_value"] = _iso_date(source_value)
        elif reporting_status == "reported" and projected_text_value not in (None, ""):
            metric["text_value"] = projected_text_value
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
            elif candidates.get(f"{key}_status") == "not_reported":
                # Preserve the canonical distinction between an explicitly
                # printed "--" and a field that silently disappeared.
                projected[target] = None
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
    for dataset_name in PERSONAL_DETAIL_SOURCE_BUSINESS_DATASETS:
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
        if explicit_status in {"partial", "extraction_failed", "unknown"}:
            presence_status = explicit_status
            reason = str(explicit_map.get("reason") or "source_state_reported")
        elif observed_count:
            presence_status = "observed_nonempty"
            reason = "records_projected"
        elif explicit_status == "not_observed" and not (
            explicit_map.get("source_statement") or explicit_map.get("absence_evidence")
        ):
            presence_status = "unknown"
            reason = "source_presence_not_established"
        elif explicit_status in _DATASET_PRESENCE_STATUSES:
            presence_status = explicit_status
            reason = str(explicit_map.get("reason") or "source_state_reported")
        else:
            presence_status = "unknown"
            reason = "source_presence_not_established"
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
        expected_count = explicit_map.get("expected_row_count")
        if isinstance(expected_count, int) and not isinstance(expected_count, bool) and expected_count >= 0:
            row["expected_row_count"] = expected_count
        confidence = explicit_map.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            row["confidence"] = max(0.0, min(1.0, float(confidence)))
        refs = _source_refs(explicit_map)
        if refs:
            row["source_refs"] = refs
        if presence_status in _POTENTIALLY_FLAWED_DATASET_STATUSES:
            rows.append(row)
    return rows


def _issue_field_observations(datasets: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project typed extraction failures into the field uncertainty ledger."""
    rows: list[dict[str, Any]] = []
    for issue in datasets.get("personal_detail_extraction_issues") or ():
        if not isinstance(issue, Mapping):
            continue
        reason_codes = {str(value) for value in issue.get("reason_codes") or ()}
        if issue.get("issue_code") != "pboc_cell_contract_unresolved" and not (
            "normalized_value_withheld" in reason_codes or issue.get("observed_value") in (None, "", [])
        ):
            continue
        dataset_name = str(issue.get("target_dataset") or "")
        field_name = str(issue.get("field_name") or "")
        if not dataset_name or not field_name:
            continue
        issue_id = str(issue.get("extraction_issue_id") or issue.get("record_id") or len(rows) + 1)
        withheld = "normalized_value_withheld" in reason_codes
        observation_id = f"field_extraction:{issue_id}"
        row: dict[str, Any] = {
            "record_id": observation_id,
            "field_observation_id": observation_id,
            "dataset_name": dataset_name,
            "business_record_id": str(issue.get("target_record_id") or "unresolved_record"),
            "field_name": field_name,
            "observation_status": "unreadable" if withheld else "ambiguous",
            "confidence_status": "not_available",
            "confidence_basis": "typed_field_contract_failure",
            "reason": str(issue.get("issue_code") or "pboc_cell_contract_unresolved"),
        }
        if issue.get("observed_value") not in (None, ""):
            row["raw_value"] = issue["observed_value"]
        confidence = issue.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            row["confidence"] = max(0.0, min(1.0, float(confidence)))
            row["confidence_status"] = "available"
            row["confidence_basis"] = "ocr_field_contract_failure"
        refs = _source_refs(issue)
        if refs:
            row["source_refs"] = refs
        rows.append(row)
    return rows


def _sequence_evidence(rows: Any, *, grouped: bool = False) -> tuple[int, bool]:
    sequences: dict[str, set[int]] = {}
    for row in rows or ():
        values = _record_values(row)
        try:
            sequence = int(values.get("sequence") or 0)
        except (TypeError, ValueError):
            sequence = 0
        if sequence <= 0:
            continue
        group = str(values.get("inquiry_type") or "unknown") if grouped else "all"
        sequences.setdefault(group, set()).add(sequence)
    if not sequences:
        return 0, True
    expected = sum(max(values) for values in sequences.values())
    contiguous = all(values == set(range(1, max(values) + 1)) for values in sequences.values())
    return expected, contiguous


def _apply_source_completeness_ledger(
    facts: dict[str, Any],
    datasets: dict[str, Any],
    final_dataset_counts: Mapping[str, int],
) -> None:
    """Turn independent sequence/count evidence into explicit partial states."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue

    ledger = facts.get("personal_detail_source_completeness_ledger")
    ledger_map = dict(ledger) if isinstance(ledger, Mapping) else {}
    endpoints = ledger_map.get("sequence_endpoints")
    endpoint_map = dict(endpoints) if isinstance(endpoints, Mapping) else {}
    states = facts.setdefault("personal_detail_dataset_states", {})
    if not isinstance(states, dict):
        states = {}
        facts["personal_detail_dataset_states"] = states
    issues = datasets.setdefault("personal_detail_extraction_issues", [])
    if not isinstance(issues, list):
        issues = []
        datasets["personal_detail_extraction_issues"] = issues

    checks: dict[str, tuple[int, int, bool]] = {}
    for dataset_name in ("residence_records", "employment_records"):
        local_rows = datasets.get(dataset_name) or ()
        observed = sum(isinstance(row, Mapping) for row in local_rows)
        sequence_expected, contiguous = _sequence_evidence(local_rows)
        source_endpoint = endpoint_map.get(dataset_name)
        expected = max(
            sequence_expected,
            int(source_endpoint) if isinstance(source_endpoint, int) and not isinstance(source_endpoint, bool) else 0,
        )
        checks[dataset_name] = (observed, expected, contiguous)

    inquiry_rows = datasets.get("inquiry_records") or ()
    inquiry_observed = sum(isinstance(row, Mapping) for row in inquiry_rows)
    emitted_inquiry_expected, inquiry_contiguous = _sequence_evidence(inquiry_rows, grouped=True)
    source_inquiry_expected = ledger_map.get("inquiry_records")
    inquiry_expected = max(
        emitted_inquiry_expected,
        int(source_inquiry_expected)
        if isinstance(source_inquiry_expected, int) and not isinstance(source_inquiry_expected, bool)
        else 0,
    )
    checks["inquiry_records"] = (inquiry_observed, inquiry_expected, inquiry_contiguous)

    account_observed = int(final_dataset_counts.get("credit_accounts") or 0)
    account_expected = ledger_map.get("credit_accounts")
    if isinstance(account_expected, int) and not isinstance(account_expected, bool):
        checks["credit_accounts"] = (account_observed, account_expected, True)

    agreement_rows = datasets.get("credit_lines") or ()
    agreement_observed = sum(isinstance(row, Mapping) for row in agreement_rows)
    agreement_expected = ledger_map.get("credit_agreements")
    if isinstance(agreement_expected, int) and not isinstance(agreement_expected, bool):
        checks["credit_lines"] = (agreement_observed, agreement_expected, True)

    existing_ids = {
        str(row.get("extraction_issue_id") or "")
        for row in issues
        if isinstance(row, Mapping)
    }
    for dataset_name, (observed, expected, contiguous) in checks.items():
        # A source endpoint can prove missing population only when it exceeds
        # the projected population. Non-contiguous ordinals with the same row
        # count are a row-identity concern, not evidence that rows are absent.
        if expected <= 0 or observed >= expected:
            continue
        issue = make_issue(
            category="page_continuation",
            issue_code="source_sequence_or_count_gap",
            message="Printed sequence/count evidence exceeds the records projected for this dataset.",
            parser_stage="source_completeness_ledger",
            target_dataset=dataset_name,
            observed_value={"observed_row_count": observed},
            candidate_value={
                "source_expected_row_count": expected,
                **(
                    {
                        "source_sequence_endpoints": ledger_map.get(
                            "inquiry_sequence_endpoints", {}
                        ),
                        "unclassified_sequence_endpoints": ledger_map.get(
                            "inquiry_unclassified_sequence_endpoints", []
                        ),
                    }
                    if dataset_name == "inquiry_records"
                    else {}
                ),
            },
            source_refs=(
                (ledger_map.get("source_refs") or {}).get(dataset_name) or ()
                if isinstance(ledger_map.get("source_refs"), Mapping)
                else ()
            ),
            reason_codes=("independent_source_ledger", "dataset_incomplete", "no_missing_row_invented"),
        )
        if issue["extraction_issue_id"] not in existing_ids:
            issues.append(issue)
            existing_ids.add(issue["extraction_issue_id"])
        states[dataset_name] = {
            "presence_status": "partial",
            "reason": "source_sequence_or_count_gap",
            "expected_row_count": expected,
        }


def prepare_personal_detail_source_collections(
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
    final_counts = final_dataset_counts or {}

    # Counts captured before the canonical business merge can be zero when a
    # scanned repayment grid is available only through the auxiliary path.
    # Reconcile that impossible zero/positive contradiction without replacing
    # any nonzero source count that could still expose a real omission.
    for dataset_name, final_count in final_counts.items():
        count_key = f"personal_detail_expected_{dataset_name}_count"
        if facts.get(count_key) == 0 and isinstance(final_count, int) and final_count > 0:
            facts[count_key] = final_count

    _apply_source_completeness_ledger(facts, datasets, final_counts)
    _validate_summary_scalar_cells(datasets)

    # The generic scanned summary count is not authoritative for a detailed
    # report (category numbering restarts).  v2 uses its source ledger through
    # dataset_status and keeps Community completeness for row conservation.
    credit_summary = dict(auxiliary.get("credit_summary") or facts.get("credit_summary") or {})
    credit_summary["account_population_comparable"] = False
    credit_summary["projected_account_count"] = int(final_counts.get("credit_accounts") or 0)
    facts["credit_summary"] = credit_summary

    profile_rows, field_observations = _profile_contract(facts, datasets)
    if profile_rows:
        datasets["personal_profile"] = profile_rows
    observations = [
        *field_observations,
        *_issue_field_observations(datasets),
    ]
    observation_rows: list[dict[str, Any]] = []
    seen_observations: set[str] = set()
    for row in observations:
        if str(row.get("observation_status") or "") not in _POTENTIALLY_FLAWED_OBSERVATION_STATUSES:
            continue
        marker = json.dumps(
            {
                "dataset_name": row.get("dataset_name"),
                "business_record_id": row.get("business_record_id"),
                "field_name": row.get("field_name"),
                "observation_status": row.get("observation_status"),
                "raw_value": row.get("raw_value"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if marker in seen_observations:
            continue
        seen_observations.add(marker)
        observation_rows.append(row)
    datasets["personal_detail_field_observations"] = observation_rows

    all_summary_metrics = _summary_metric_contract(datasets)
    summary_metrics = [
        row
        for row in all_summary_metrics
        if str(row.get("mapping_status") or "") == "mapped"
    ]
    if summary_metrics:
        datasets["personal_detail_credit_summary_metrics"] = summary_metrics
    else:
        datasets.pop("personal_detail_credit_summary_metrics", None)
    if len(summary_metrics) < len(all_summary_metrics):
        states = facts.setdefault("personal_detail_dataset_states", {})
        if isinstance(states, dict):
            states["personal_detail_credit_summary_metrics"] = {
                "presence_status": "partial",
                "reason": "unmapped_summary_cells_quarantined",
                "expected_row_count": len(all_summary_metrics),
                "observed_row_count": len(summary_metrics),
            }

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
    facts["canonical_dataset_schema"] = "personal_credit_report_detailed.v2"
    # Community dataset completeness now means lossless v2 row projection.
    # Source-document completeness remains explicit in dataset_status, avoiding
    # a duplicate generic warning for the same uncertainty.
    from docmirror.plugins.credit_report.personal_detail_scanned.schema import project_personal_detail_datasets

    preview = project_personal_detail_datasets(datasets)
    for dataset_name, rows in preview.items():
        facts[f"personal_detail_v2_expected_{dataset_name}_count"] = len(rows)
    return content


__all__ = [
    "PERSONAL_DETAIL_SOURCE_BUSINESS_DATASETS",
    "PERSONAL_PROFILE_FIELDS",
    "prepare_personal_detail_source_collections",
    "project_typed_public_records",
]
