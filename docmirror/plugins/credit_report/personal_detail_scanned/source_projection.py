# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract-only projections for personal detailed credit reports.

The functions in this module do not discover OCR text.  They turn facts and
rows already emitted by the personal-detail extractors into stable business
datasets, while preserving the distinction between an unobserved value and an
explicitly reported absence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable

from docmirror.plugins.credit_report.value_utils import stable_record_id

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
    if re.fullmatch(r"\d{4}[-./]\d{1,2}(?:[-./]\d{1,2})?", text):
        return "date", None, "reported"
    integer = re.fullmatch(
        r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)",
        text,
    )
    if integer is not None:
        return "integer", text.replace(",", ""), "reported"
    decimal = re.fullmatch(
        r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)\.\d+",
        text,
    )
    if decimal is not None:
        return "decimal", text.replace(",", ""), "reported"
    percentage = re.fullmatch(
        r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?%",
        text,
    )
    if percentage is not None:
        return "percentage", text[:-1].replace(",", ""), "reported"
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


_ACCOUNT_POPULATION_FAMILIES = frozenset(
    {
        "non_revolving_loan",
        "revolving_loan_subaccount",
        "revolving_loan_account",
        "credit_card",
        "quasi_credit_card",
    }
)
_ACCOUNT_POPULATION_PRINTED_FIELDS = frozenset(
    {
        "management_institution",
        "account_identifier",
        "open_date",
        "due_date",
        "loan_amount",
        "credit_limit",
        "shared_credit_limit",
        "account_currency",
        "business_type",
        "guarantee_type",
        "repayment_periods",
        "repayment_frequency",
        "repayment_method",
        "co_borrower_flag",
        "snapshot_date",
        "account_state",
    }
)
_INQUIRY_POPULATION_TYPES = frozenset({"institution", "personal"})
_SEQUENCE_POPULATION_DATASETS = {
    "mobile_phone_records": {
        "id_field": "mobile_phone_record_id",
        "id_prefix": "personal_mobile_phone",
        "primary_component": "mobile",
        "components": {
            "mobile": (
                "sequence",
                "mobile_phone",
                "information_updated_date",
                "data_provider",
            ),
        },
        "printed_fields": frozenset(
            {"mobile_phone", "information_updated_date", "data_provider"}
        ),
    },
    "residence_records": {
        "id_field": "residence_record_id",
        "id_prefix": "credit_residence",
        "primary_component": "residence",
        "components": {
            "residence": (
                "sequence",
                "address",
                "residential_phone",
                "residence_status",
                "information_updated_date",
            ),
            "provider": ("sequence", "data_provider"),
        },
        "printed_fields": frozenset(
            {
                "address",
                "residential_phone",
                "residence_status",
                "information_updated_date",
                "data_provider",
            }
        ),
    },
    "employment_records": {
        "id_field": "employment_record_id",
        "id_prefix": "credit_employment",
        "primary_component": "basic",
        "components": {
            "basic": (
                "sequence",
                "employer",
                "employer_type",
                "employer_address",
                "employer_phone",
            ),
            "detail": (
                "sequence",
                "occupation",
                "industry",
                "position",
                "professional_title",
                "entry_year",
                "information_updated_date",
            ),
            "provider": ("sequence", "data_provider"),
        },
        "printed_fields": frozenset(
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
    },
}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _canonical_account_ordinal(values: Mapping[str, Any]) -> tuple[str, int] | None:
    family = str(values.get("account_type") or "")
    ordinal = _positive_int(values.get("category_sequence"))
    if family not in _ACCOUNT_POPULATION_FAMILIES or ordinal is None:
        return None
    expected = f"credit_account:{family}:{ordinal}"
    account_id = str(values.get("account_id") or values.get("record_id") or "")
    return (family, ordinal) if account_id == expected else None


def _canonical_agreement_ordinal(values: Mapping[str, Any]) -> int | None:
    ordinal = _positive_int(values.get("_printed_sequence"))
    key = str(values.get("_canonical_card_key") or "")
    if ordinal is not None and key == f"credit_agreement:{ordinal}":
        return ordinal
    match = re.fullmatch(r"credit_agreement:(\d+)", key)
    if match:
        return _positive_int(match.group(1))
    for identity_field in ("credit_line_id", "record_id"):
        identity_match = re.fullmatch(
            r"(?:credit_agreement|credit_line):(\d+)",
            str(values.get(identity_field) or ""),
        )
        if identity_match:
            return _positive_int(identity_match.group(1))
    return None


def _exact_emitted_agreement_identity_ref(raw_ref: Any) -> dict[str, Any] | None:
    if not isinstance(raw_ref, Mapping):
        return None
    ref = dict(raw_ref)
    bbox = ref.get("bbox")
    evidence_ids = ref.get("evidence_ids")
    logical_page = ref.get("logical_page")
    source_page = ref.get("source_page")
    if (
        str(ref.get("binding") or "") != "canonical_card_anchor"
        or not isinstance(logical_page, int)
        or isinstance(logical_page, bool)
        or logical_page <= 0
        or not isinstance(source_page, int)
        or isinstance(source_page, bool)
        or source_page <= 0
        or not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in bbox
        )
        or float(bbox[2]) <= float(bbox[0])
        or float(bbox[3]) <= float(bbox[1])
    ):
        return None
    if evidence_ids is not None and not isinstance(evidence_ids, (list, tuple)):
        return None
    return ref


def _emitted_agreement_ordinals(
    ledger: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> set[int]:
    """Resolve emitted agreements only through exact reciprocal source identity."""

    matches: Counter[int] = Counter()
    for row in datasets.get("credit_lines") or ():
        if not isinstance(row, Mapping):
            continue
        values = _record_values(row)
        ordinal = _canonical_agreement_ordinal(values)
        if ordinal is not None:
            matches[ordinal] += 1
            continue
        identity = values.get("_source_agreement_identity")
        if not isinstance(identity, Mapping):
            continue
        ordinal = _positive_int(identity.get("sequence"))
        if ordinal is None or _positive_int(values.get("sequence")) != ordinal:
            continue
        observation = _flat_ordinal_observation(
            ledger,
            "credit_agreement_ordinal_observations",
            ordinal,
        )
        ledger_refs = _exact_agreement_ordinal_source_refs(observation, ordinal)
        emitted_refs = [
            ref
            for raw_ref in identity.get("source_refs") or ()
            if (ref := _exact_emitted_agreement_identity_ref(raw_ref)) is not None
        ]
        if not ledger_refs or not emitted_refs:
            continue
        reciprocal = any(
            emitted_ref.get("logical_page") == ledger_ref.get("logical_page")
            and emitted_ref.get("source_page") == ledger_ref.get("source_page")
            and min(
                float(emitted_ref["bbox"][2]),
                float(ledger_ref["bbox"][2]),
            )
            > max(
                float(emitted_ref["bbox"][0]),
                float(ledger_ref["bbox"][0]),
            )
            and min(
                float(emitted_ref["bbox"][3]),
                float(ledger_ref["bbox"][3]),
            )
            > max(
                float(emitted_ref["bbox"][1]),
                float(ledger_ref["bbox"][1]),
            )
            for emitted_ref in emitted_refs
            for ledger_ref in ledger_refs
        )
        if reciprocal:
            matches[ordinal] += 1
    return {ordinal for ordinal, count in matches.items() if count == 1}


def _canonical_inquiry_ordinal(values: Mapping[str, Any]) -> tuple[str, int] | None:
    inquiry_type = str(values.get("inquiry_type") or "")
    ordinal = _positive_int(values.get("sequence"))
    if inquiry_type not in _INQUIRY_POPULATION_TYPES or ordinal is None:
        return None
    expected = stable_record_id("credit_inquiry", inquiry_type, ordinal)
    identity = str(values.get("inquiry_id") or values.get("record_id") or "")
    aliases = {
        expected,
        f"credit_inquiry:{inquiry_type}:{ordinal}",
        f"inquiry:{inquiry_type}:{ordinal}",
    }
    return (inquiry_type, ordinal) if identity in aliases else None


def _nested_ordinal_observation(
    ledger: Mapping[str, Any],
    key: str,
    group: str,
    ordinal: int,
) -> dict[str, Any]:
    groups = ledger.get(key)
    if not isinstance(groups, Mapping):
        return {}
    observations = groups.get(group)
    if not isinstance(observations, Mapping):
        return {}
    observation = observations.get(str(ordinal), observations.get(ordinal))
    return dict(observation) if isinstance(observation, Mapping) else {}


def _flat_ordinal_observation(
    ledger: Mapping[str, Any],
    key: str,
    ordinal: int,
) -> dict[str, Any]:
    observations = ledger.get(key)
    if not isinstance(observations, Mapping):
        return {}
    observation = observations.get(str(ordinal), observations.get(ordinal))
    return dict(observation) if isinstance(observation, Mapping) else {}


def _local_ordinal_source_refs(
    observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _dedupe_source_refs(_source_refs(observation))


def _strict_raw_profile_evidence_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in value
    ):
        return None
    normalized = tuple(value)
    return normalized if len(normalized) == len(set(normalized)) else None


def _strict_raw_profile_positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _strict_raw_profile_nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _strict_raw_profile_bbox(
    value: Any,
) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in value
        )
    ):
        return None
    bbox = tuple(float(item) for item in value)
    return (
        bbox
        if all(math.isfinite(item) for item in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
        else None
    )


def _exact_raw_profile_ref(
    raw_ref: Any,
    *,
    dataset: str,
    ordinal: int,
    field_name: str,
    sequence_ref: bool,
) -> dict[str, Any] | None:
    contract = _SEQUENCE_POPULATION_DATASETS.get(dataset)
    if not isinstance(raw_ref, Mapping) or contract is None:
        return None
    ref = dict(raw_ref)
    component = ref.get("component")
    components = contract["components"]
    expected_source = (
        "candidate_b_raw_profile_sequence_cell"
        if sequence_ref
        else "candidate_b_raw_profile_field_cell"
    )
    expected_binding = (
        "printed_profile_sequence" if sequence_ref else "printed_profile_field"
    )
    if (
        not isinstance(component, str)
        or component not in components
        or field_name not in components[component]
        or (
            sequence_ref
            and component != contract["primary_component"]
        )
        or ref.get("source") != expected_source
        or ref.get("geometry_scope") != "cell"
        or ref.get("binding") != expected_binding
        or ref.get("binding_quality") != expected_binding
        or ref.get("canonical_template_id") != "report_header_and_identity"
        or ref.get("dataset_name") != dataset
        or ref.get("field_name") != field_name
        or _strict_raw_profile_positive_int(ref.get("sequence")) != ordinal
        or _strict_raw_profile_positive_int(ref.get("logical_page")) is None
        or _strict_raw_profile_positive_int(ref.get("source_page")) is None
        or _strict_raw_profile_nonnegative_int(ref.get("row")) is None
        or _strict_raw_profile_nonnegative_int(ref.get("column")) is None
        or _strict_raw_profile_bbox(ref.get("bbox")) is None
        or _strict_raw_profile_evidence_ids(ref.get("evidence_ids")) is None
        or not isinstance(ref.get("table_id"), str)
        or not ref["table_id"]
        or ref["table_id"] != ref["table_id"].strip()
    ):
        return None
    return ref


def _exact_raw_profile_observations(
    ledger: Mapping[str, Any],
    dataset: str,
) -> tuple[int, dict[int, dict[str, Any]]] | None:
    """Validate a complete dense producer plane; never trust an endpoint alone."""

    contract = _SEQUENCE_POPULATION_DATASETS.get(dataset)
    endpoints = ledger.get("sequence_endpoints")
    observed_map = ledger.get("sequence_observed_sequences")
    observation_map = ledger.get("sequence_ordinal_observations")
    if (
        contract is None
        or not isinstance(endpoints, Mapping)
        or not isinstance(observed_map, Mapping)
        or not isinstance(observation_map, Mapping)
    ):
        return None
    endpoint = _strict_raw_profile_positive_int(endpoints.get(dataset))
    raw_sequences = observed_map.get(dataset)
    raw_observations = observation_map.get(dataset)
    if (
        endpoint is None
        or not isinstance(raw_sequences, (list, tuple))
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_sequences
        )
        or list(raw_sequences) != list(range(1, endpoint + 1))
        or not isinstance(raw_observations, Mapping)
    ):
        return None

    normalized: dict[int, dict[str, Any]] = {}
    evidence_owners: dict[str, set[tuple[Any, ...]]] = {}
    slot_roles: dict[tuple[Any, ...], tuple[str, int, str]] = {}

    def register_ref(
        ref: Mapping[str, Any], role: tuple[str, int, str]
    ) -> bool:
        signature = (
            ref.get("logical_page"),
            ref.get("source_page"),
            ref.get("table_id"),
            ref.get("row"),
            ref.get("column"),
            tuple(float(value) for value in ref["bbox"]),
        )
        existing_role = slot_roles.setdefault(signature, role)
        if existing_role != role:
            return False
        for evidence_id in ref["evidence_ids"]:
            evidence_owners.setdefault(evidence_id, set()).add(signature)
            if len(evidence_owners[evidence_id]) != 1:
                return False
        return True

    for raw_ordinal, raw_observation in raw_observations.items():
        if (
            isinstance(raw_ordinal, int)
            and not isinstance(raw_ordinal, bool)
            and raw_ordinal > 0
        ):
            ordinal = raw_ordinal
        elif isinstance(raw_ordinal, str) and re.fullmatch(r"[1-9]\d*", raw_ordinal):
            ordinal = int(raw_ordinal)
        else:
            return None
        if ordinal in normalized or not isinstance(raw_observation, Mapping):
            return None
        observation = dict(raw_observation)
        id_field = str(contract["id_field"])
        expected_id = stable_record_id(str(contract["id_prefix"]), ordinal)
        if (
            _strict_raw_profile_positive_int(observation.get("sequence")) != ordinal
            or observation.get(id_field) != expected_id
            or observation.get("canonical_template_id")
            != "report_header_and_identity"
        ):
            return None
        headers = observation.get("canonical_header_fields_by_component")
        if not isinstance(headers, Mapping) or set(headers) - set(
            contract["components"]
        ):
            return None
        primary = str(contract["primary_component"])
        if set(headers) == set() or primary not in headers:
            return None
        for component, header_fields in headers.items():
            if (
                not isinstance(header_fields, (list, tuple))
                or tuple(header_fields) != tuple(contract["components"][component])
            ):
                return None

        raw_refs = observation.get("source_refs")
        if not isinstance(raw_refs, (list, tuple)) or len(raw_refs) != 1:
            return None
        sequence_ref = _exact_raw_profile_ref(
            raw_refs[0],
            dataset=dataset,
            ordinal=ordinal,
            field_name="sequence",
            sequence_ref=True,
        )
        if sequence_ref is None:
            return None
        if not register_ref(sequence_ref, (dataset, ordinal, "sequence")):
            return None

        raw_printed_fields = observation.get("printed_fields")
        refs_by_field = observation.get("field_source_refs")
        if not isinstance(raw_printed_fields, (list, tuple)) or not isinstance(
            refs_by_field, Mapping
        ):
            return None
        if any(not isinstance(field_name, str) for field_name in raw_printed_fields):
            return None
        printed_fields = tuple(raw_printed_fields)
        if (
            len(printed_fields) != len(set(printed_fields))
            or set(printed_fields) - set(contract["printed_fields"])
            or set(refs_by_field) != set(printed_fields)
        ):
            return None
        exact_field_refs: dict[str, list[dict[str, Any]]] = {}
        for field_name in printed_fields:
            raw_field_refs = refs_by_field.get(field_name)
            if not isinstance(raw_field_refs, (list, tuple)) or len(raw_field_refs) != 1:
                return None
            field_ref = _exact_raw_profile_ref(
                raw_field_refs[0],
                dataset=dataset,
                ordinal=ordinal,
                field_name=field_name,
                sequence_ref=False,
            )
            if field_ref is None or field_ref["component"] not in headers:
                return None
            if not register_ref(field_ref, (dataset, ordinal, field_name)):
                return None
            exact_field_refs[field_name] = [field_ref]
        observation["source_refs"] = [sequence_ref]
        observation["field_source_refs"] = exact_field_refs
        observation["printed_fields"] = list(printed_fields)
        normalized[ordinal] = observation
    if set(normalized) != set(range(1, endpoint + 1)):
        return None
    return endpoint, normalized


def _exact_account_ordinal_source_refs(
    observation: Mapping[str, Any],
    family: str,
    ordinal: int,
) -> list[dict[str, Any]]:
    """Keep only the sealed printed anchor for one exact account identity."""

    expected_id = f"credit_account:{family}:{ordinal}"
    if str(observation.get("account_id") or "") != expected_id:
        return []
    refs: list[dict[str, Any]] = []
    for ref in _source_refs(observation):
        bbox = ref.get("bbox")
        evidence_ids = ref.get("evidence_ids")
        logical_page = ref.get("logical_page")
        source_page = ref.get("source_page")
        ref_ordinal = ref.get("category_sequence")
        if (
            str(ref.get("source") or "") != "candidate_b_account_anchor"
            or str(ref.get("geometry_scope") or "") != "line"
            or str(ref.get("binding") or "") != "printed_account_ordinal"
            or str(ref.get("binding_quality") or "")
            != "printed_account_ordinal"
            or str(ref.get("account_type") or "") != family
            or not isinstance(logical_page, int)
            or isinstance(logical_page, bool)
            or logical_page <= 0
            or not isinstance(source_page, int)
            or isinstance(source_page, bool)
            or source_page <= 0
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                and math.isfinite(float(coordinate))
                for coordinate in bbox
            )
            or float(bbox[2]) <= float(bbox[0])
            or float(bbox[3]) <= float(bbox[1])
            or not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(
                not isinstance(value, str) or not value.strip()
                for value in evidence_ids
            )
            or len(evidence_ids) != len({value.strip() for value in evidence_ids})
            or not isinstance(ref_ordinal, int)
            or isinstance(ref_ordinal, bool)
            or ref_ordinal != ordinal
        ):
            continue
        refs.append(ref)
    return _dedupe_source_refs(refs) if len(refs) == 1 else []


def _exact_agreement_ordinal_source_refs(
    observation: Mapping[str, Any],
    ordinal: int,
) -> list[dict[str, Any]]:
    """Keep only exact, source-bound refs for one printed agreement ordinal."""

    observation_sequence = observation.get("sequence")
    if (
        not isinstance(observation_sequence, int)
        or isinstance(observation_sequence, bool)
        or observation_sequence != ordinal
    ):
        return []
    refs: list[dict[str, Any]] = []
    for ref in _source_refs(observation):
        bbox = ref.get("bbox")
        evidence_ids = ref.get("evidence_ids")
        logical_page = ref.get("logical_page")
        source_page = ref.get("source_page")
        ref_sequence = ref.get("sequence")
        if (
            str(ref.get("source") or "")
            != "candidate_b_source_coverage_ledger"
            or str(ref.get("geometry_scope") or "") != "line"
            or str(ref.get("binding") or "")
            != "printed_credit_agreement_ordinal"
            or str(ref.get("binding_quality") or "")
            != "printed_credit_agreement_ordinal"
            or not isinstance(logical_page, int)
            or isinstance(logical_page, bool)
            or logical_page <= 0
            or not isinstance(source_page, int)
            or isinstance(source_page, bool)
            or source_page <= 0
            or not isinstance(ref_sequence, int)
            or isinstance(ref_sequence, bool)
            or ref_sequence != ordinal
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                and math.isfinite(float(coordinate))
                for coordinate in bbox
            )
            or float(bbox[2]) <= float(bbox[0])
            or float(bbox[3]) <= float(bbox[1])
            or not isinstance(evidence_ids, (list, tuple))
            or not any(str(value or "").strip() for value in evidence_ids)
        ):
            continue
        refs.append(ref)
    return _dedupe_source_refs(refs)


_AGREEMENT_POPULATION_PRINTED_FIELDS = frozenset(
    {
        "account_identifier",
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "currency",
    }
)


def _exact_agreement_field_source_refs(
    observation: Mapping[str, Any],
    field_name: str,
) -> list[dict[str, Any]]:
    refs_by_field = observation.get("field_source_refs")
    if not isinstance(refs_by_field, Mapping):
        return []
    refs: list[dict[str, Any]] = []
    for raw_ref in refs_by_field.get(field_name) or ():
        if not isinstance(raw_ref, Mapping):
            continue
        ref = dict(raw_ref)
        bbox = ref.get("bbox")
        evidence_ids = ref.get("evidence_ids")
        logical_page = ref.get("logical_page")
        source_page = ref.get("source_page")
        source = str(ref.get("source") or "")
        binding = str(ref.get("binding") or "")
        if (
            source
            not in {
                "native_detail_tolerant_table_cell",
                "personal_detail_corrected_page_cell",
            }
            or str(ref.get("geometry_scope") or "") != "cell"
            or binding not in {"label_column", "canonical_label_slot"}
            or str(ref.get("field_name") or "") != field_name
            or not isinstance(logical_page, int)
            or isinstance(logical_page, bool)
            or logical_page <= 0
            or not isinstance(source_page, int)
            or isinstance(source_page, bool)
            or source_page <= 0
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                and math.isfinite(float(coordinate))
                for coordinate in bbox
            )
            or float(bbox[2]) <= float(bbox[0])
            or float(bbox[3]) <= float(bbox[1])
            or not isinstance(evidence_ids, (list, tuple))
            or not any(str(value or "").strip() for value in evidence_ids)
        ):
            continue
        if source == "native_detail_tolerant_table_cell":
            row = ref.get("row")
            column = ref.get("column")
            if (
                not str(ref.get("table_id") or "").strip()
                or not isinstance(row, int)
                or isinstance(row, bool)
                or row < 0
                or not isinstance(column, int)
                or isinstance(column, bool)
                or column < 0
            ):
                continue
        refs.append(ref)
    return _dedupe_source_refs(refs)


def _dense_endpoint_is_credible(
    endpoint: int,
    observed_values: Any,
    outlier_values: Any,
) -> bool:
    """Fail closed on an isolated high OCR ordinal before expanding a gap."""

    observed = {
        value
        for item in observed_values or ()
        if (value := _positive_int(item)) is not None
    }
    outliers = {
        value
        for item in outlier_values or ()
        if (value := _positive_int(item)) is not None
    }
    observed.difference_update(outliers)
    if endpoint in outliers or endpoint not in observed:
        return False
    ceiling = max(3, len(observed) + max(2, len(observed) // 4))
    credible = {value for value in observed if value <= ceiling}
    high_values = sorted(value for value in observed if value > ceiling)
    tail: list[int] = []
    if high_values:
        tail = [high_values[-1]]
        for value in reversed(high_values[:-1]):
            if value != tail[0] - 1:
                break
            tail.insert(0, value)
    if len(tail) >= 3:
        credible.update(tail)
    return bool(credible) and max(credible) == endpoint


def _append_population_issue(
    issues: list[dict[str, Any]],
    existing_ids: set[str],
    **kwargs: Any,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue

    issue = make_issue(**kwargs)
    issue_id = str(issue["extraction_issue_id"])
    if issue_id in existing_ids:
        return
    issues.append(issue)
    existing_ids.add(issue_id)


def _append_exact_account_population_issues(
    ledger: Mapping[str, Any],
    datasets: Mapping[str, Any],
    issues: list[dict[str, Any]],
    existing_ids: set[str],
) -> None:
    endpoints = ledger.get("account_family_endpoints")
    if not isinstance(endpoints, Mapping):
        return
    emitted = {
        identity
        for row in datasets.get("credit_accounts") or ()
        if isinstance(row, Mapping)
        and (identity := _canonical_account_ordinal(_record_values(row))) is not None
    }
    for family in sorted(_ACCOUNT_POPULATION_FAMILIES):
        endpoint = _positive_int(endpoints.get(family))
        if endpoint is None:
            continue
        for ordinal in range(1, endpoint + 1):
            if (family, ordinal) in emitted:
                continue
            target = f"credit_account:{family}:{ordinal}"
            observation = _nested_ordinal_observation(
                ledger,
                "account_family_ordinal_observations",
                family,
                ordinal,
            )
            refs = _exact_account_ordinal_source_refs(observation, family, ordinal)
            if not refs:
                # The family endpoint proves an aggregate count, not the
                # source location of every ordinal below it.
                continue
            _append_population_issue(
                issues,
                existing_ids,
                category="schema_incompleteness",
                issue_code="source_account_record_omitted",
                message="A source-proven account family ordinal was not projected as a canonical account record.",
                parser_stage="source_completeness_ledger",
                target_dataset="credit_accounts",
                target_record_id=target,
                field_name="account_id",
                observed_value={"account_type": family, "category_sequence": ordinal},
                candidate_value={"source_family_endpoint": endpoint},
                source_refs=refs,
                reason_codes=(
                    "independent_source_ledger",
                    "exact_family_ordinal",
                    "missing_business_record",
                    "normalized_value_withheld",
                ),
            )
            if str(observation.get("account_id") or target) != target:
                continue
            field_refs = observation.get("field_source_refs")
            field_ref_map = dict(field_refs) if isinstance(field_refs, Mapping) else {}
            printed_fields = {
                str(field_name)
                for field_name in observation.get("printed_fields") or ()
                if str(field_name) in _ACCOUNT_POPULATION_PRINTED_FIELDS
            }
            for field_name in sorted(printed_fields):
                local_refs = [
                    dict(ref)
                    for ref in field_ref_map.get(field_name) or ()
                    if isinstance(ref, Mapping)
                ]
                if not local_refs:
                    continue
                _append_population_issue(
                    issues,
                    existing_ids,
                    category="schema_incompleteness",
                    issue_code="source_account_field_omitted",
                    message="A source-observed printed account field belongs to a source-proven record that was not projected.",
                    parser_stage="source_completeness_ledger",
                    target_dataset="credit_accounts",
                    target_record_id=target,
                    field_name=field_name,
                    observed_value={"source_field_observed": True},
                    candidate_value={"account_type": family, "category_sequence": ordinal},
                    source_refs=_dedupe_source_refs(local_refs),
                    reason_codes=(
                        "independent_source_ledger",
                        "exact_family_ordinal",
                        "printed_field_observed",
                        "missing_business_record",
                        "normalized_value_withheld",
                    ),
                )


def _append_exact_agreement_population_issues(
    ledger: Mapping[str, Any],
    datasets: Mapping[str, Any],
    issues: list[dict[str, Any]],
    existing_ids: set[str],
) -> None:
    endpoint = _positive_int(ledger.get("credit_agreement_sequence_endpoint"))
    if endpoint is None:
        return
    observed_sequences = ledger.get("credit_agreement_observed_sequences")
    outliers = {
        value
        for item in ledger.get("credit_agreement_sequence_outliers") or ()
        if (value := _positive_int(item)) is not None
    }
    if not _dense_endpoint_is_credible(endpoint, observed_sequences, outliers):
        return
    emitted = _emitted_agreement_ordinals(ledger, datasets)
    for ordinal in range(1, endpoint + 1):
        if ordinal in emitted:
            continue
        observation = _flat_ordinal_observation(
            ledger,
            "credit_agreement_ordinal_observations",
            ordinal,
        )
        refs = _exact_agreement_ordinal_source_refs(observation, ordinal)
        if not refs:
            continue
        _append_population_issue(
            issues,
            existing_ids,
            category="schema_incompleteness",
            issue_code="source_credit_agreement_record_omitted",
            message="A source-proven credit-agreement ordinal was not projected as a canonical agreement record.",
            parser_stage="source_completeness_ledger",
            target_dataset="credit_lines",
            target_record_id=f"credit_agreement:{ordinal}",
            field_name="credit_line_id",
            observed_value={"credit_agreement_sequence": ordinal},
            candidate_value={"source_sequence_endpoint": endpoint},
            source_refs=refs,
            reason_codes=(
                "independent_source_ledger",
                "exact_agreement_ordinal",
                "missing_business_record",
                "normalized_value_withheld",
            ),
        )
        printed_fields = {
            str(field_name)
            for field_name in observation.get("printed_fields") or ()
            if str(field_name) in _AGREEMENT_POPULATION_PRINTED_FIELDS
        }
        for field_name in sorted(printed_fields):
            field_refs = _exact_agreement_field_source_refs(
                observation,
                field_name,
            )
            if not field_refs:
                continue
            _append_population_issue(
                issues,
                existing_ids,
                category="schema_incompleteness",
                issue_code="source_credit_agreement_field_omitted",
                message="A source-observed agreement cell belongs to an exact source agreement that was not projected.",
                parser_stage="source_completeness_ledger",
                target_dataset="credit_lines",
                target_record_id=f"credit_agreement:{ordinal}",
                field_name=field_name,
                observed_value={"source_field_observed": True},
                candidate_value={"credit_agreement_sequence": ordinal},
                source_refs=field_refs,
                reason_codes=(
                    "independent_source_ledger",
                    "exact_agreement_ordinal",
                    "exact_agreement_field_cell",
                    "printed_field_observed",
                    "missing_business_record",
                    "normalized_value_withheld",
                ),
            )


def _finite_source_bbox(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            and math.isfinite(float(coordinate))
            for coordinate in value
        )
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _unique_source_evidence_ids(value: Any) -> frozenset[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if not value or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        return None
    evidence_ids = tuple(value)
    if len(evidence_ids) != len(set(evidence_ids)):
        return None
    return frozenset(evidence_ids)


def _exact_inquiry_source_signature(
    ref: Any,
) -> tuple[int, int, str, int] | None:
    """Return one exact page/table/row owner without scalar coercion."""

    if not isinstance(ref, Mapping):
        return None
    logical_page = ref.get("logical_page")
    source_page = ref.get("source_page")
    table_id = ref.get("table_id")
    row = ref.get("row")
    if (
        not isinstance(logical_page, int)
        or isinstance(logical_page, bool)
        or logical_page <= 0
        or not isinstance(source_page, int)
        or isinstance(source_page, bool)
        or source_page <= 0
        or not isinstance(table_id, str)
        or not table_id.strip()
        or table_id != table_id.strip()
        or not isinstance(row, int)
        or isinstance(row, bool)
        or row < 0
    ):
        return None
    return logical_page, source_page, table_id, row


def _exact_inquiry_physical_field_ref(
    ref: Any,
    *,
    field_name: str,
) -> frozenset[str] | None:
    if not isinstance(ref, Mapping):
        return None
    logical_page = ref.get("logical_page")
    source_page = ref.get("source_page")
    row = ref.get("row")
    column = ref.get("column")
    evidence_ids = _unique_source_evidence_ids(ref.get("evidence_ids"))
    source_contract = (
        ref.get("source"),
        ref.get("geometry_scope"),
        ref.get("binding"),
        ref.get("binding_quality"),
    )
    allowed_contracts = {
        (
            "native_detail_inquiry_physical_field",
            "token_y_band",
            "canonical_inquiry_column_y_band",
            "exact_tokens_uniquely_owned_by_date_band",
        ),
        (
            "native_detail_inquiry_raw_physical_field",
            "token_y_band",
            "sealed_raw_inquiry_role_y_band",
            "exact_tokens_uniquely_partitioned_in_registered_lattice",
        ),
    }
    if (
        field_name not in {"inquiry_date", "institution", "reason"}
        or source_contract not in allowed_contracts
        or ref.get("field_name") != field_name
        or not isinstance(logical_page, int)
        or isinstance(logical_page, bool)
        or logical_page <= 0
        or not isinstance(source_page, int)
        or isinstance(source_page, bool)
        or source_page <= 0
        or not str(ref.get("table_id") or "")
        or not isinstance(row, int)
        or isinstance(row, bool)
        or row < 0
        or not isinstance(column, int)
        or isinstance(column, bool)
        or column < 0
        or not _finite_source_bbox(ref.get("bbox"))
        or evidence_ids is None
    ):
        return None
    return evidence_ids


def _exact_typed_inquiry_field_ref(
    ref: Any,
    *,
    field_name: str,
) -> frozenset[str] | None:
    physical_ids = _exact_inquiry_physical_field_ref(
        ref,
        field_name=field_name,
    )
    if physical_ids is not None:
        return physical_ids
    if not isinstance(ref, Mapping):
        return None
    logical_page = ref.get("logical_page")
    source_page = ref.get("source_page")
    row = ref.get("row")
    column = ref.get("column")
    evidence_ids = _unique_source_evidence_ids(ref.get("evidence_ids"))
    if (
        field_name not in {"inquiry_date", "institution", "reason"}
        or ref.get("source") != "native_detail_inquiry_token"
        or ref.get("geometry_scope") != "token"
        or ref.get("binding") != "canonical_header_column_token"
        or ref.get("binding_quality") != "exact_token_in_canonical_cell"
        or ref.get("field_name") != field_name
        or not isinstance(logical_page, int)
        or isinstance(logical_page, bool)
        or logical_page <= 0
        or not isinstance(source_page, int)
        or isinstance(source_page, bool)
        or source_page <= 0
        or not str(ref.get("table_id") or "")
        or not isinstance(row, int)
        or isinstance(row, bool)
        or row < 0
        or not isinstance(column, int)
        or isinstance(column, bool)
        or column < 0
        or not _finite_source_bbox(ref.get("bbox"))
        or evidence_ids is None
    ):
        return None
    return evidence_ids


def _exact_inquiry_physical_observation(
    observation: Any,
) -> tuple[
    str,
    dict[str, list[dict[str, Any]]],
    frozenset[str],
    tuple[int, int, str, int],
] | None:
    fields = ("inquiry_date", "institution", "reason")
    if not isinstance(observation, Mapping) or any(
        key in observation
        for key in (
            "inquiry_type",
            "sequence",
            "value",
            "raw_value",
            "normalized_value",
        )
    ):
        return None
    physical_row_id = str(observation.get("source_physical_row_id") or "")
    if not physical_row_id.startswith(
        ("source_inquiry_physical_row:", "source_inquiry_raw_physical_position:")
    ):
        return None
    printed_fields = tuple(str(value) for value in observation.get("printed_fields") or ())
    refs_by_field = observation.get("field_source_refs")
    source_refs = observation.get("source_refs")
    if (
        len(printed_fields) != len(fields)
        or set(printed_fields) != set(fields)
        or not isinstance(refs_by_field, Mapping)
        or not isinstance(source_refs, list)
        or len(source_refs) != 1
        or not isinstance(source_refs[0], Mapping)
    ):
        return None
    owner_ref = source_refs[0]
    owner_signature = _exact_inquiry_source_signature(owner_ref)
    owner_ids = _unique_source_evidence_ids(owner_ref.get("evidence_ids"))
    owner_contract = (
        owner_ref.get("source"),
        owner_ref.get("geometry_scope"),
        owner_ref.get("binding"),
        owner_ref.get("binding_quality"),
    )
    if owner_contract == (
        "native_detail_inquiry_physical_row",
        "token_y_band",
        "canonical_inquiry_four_role_y_band",
        "all_exact_tokens_uniquely_consumed",
    ):
        id_namespace = "source_inquiry_physical_row"
    elif owner_contract in {
        (
            "native_detail_inquiry_raw_physical_row",
            "token_y_band",
            "sealed_raw_inquiry_registered_lattice_band",
            "all_exact_tokens_uniquely_partitioned",
        ),
        (
            "native_detail_inquiry_raw_physical_row",
            "exact_source_row_band",
            "sealed_raw_inquiry_registered_lattice_band",
            "sealed_exact_physical_position",
        ),
    }:
        id_namespace = "source_inquiry_raw_physical_position"
    else:
        return None
    expected_physical_row_id = (
        stable_record_id(
            id_namespace,
            *owner_signature,
            *sorted(owner_ids),
        )
        if owner_ids is not None and owner_signature is not None
        else ""
    )
    if (
        owner_ids is None
        or owner_signature is None
        or not _finite_source_bbox(owner_ref.get("bbox"))
        or physical_row_id != expected_physical_row_id
    ):
        return None

    exact_refs: dict[str, list[dict[str, Any]]] = {}
    field_ids: set[str] = set()
    for field_name in fields:
        raw_refs = refs_by_field.get(field_name)
        if (
            not isinstance(raw_refs, list)
            or len(raw_refs) != 1
            or not isinstance(raw_refs[0], Mapping)
        ):
            return None
        ref = dict(raw_refs[0])
        evidence_ids = _exact_inquiry_physical_field_ref(
            ref,
            field_name=field_name,
        )
        if (
            evidence_ids is None
            or field_ids.intersection(evidence_ids)
            or _exact_inquiry_source_signature(ref) != owner_signature
        ):
            return None
        field_ids.update(evidence_ids)
        exact_refs[field_name] = [ref]
    if not field_ids or not field_ids.issubset(owner_ids):
        return None
    return physical_row_id, exact_refs, frozenset(field_ids), owner_signature


def _exact_inquiry_raw_physical_position(
    observation: Any,
) -> tuple[str, dict[str, Any], frozenset[str]] | None:
    """Validate one anonymous physical source identity and nothing more."""

    if not isinstance(observation, Mapping) or any(
        key in observation
        for key in (
            "inquiry_type",
            "sequence",
            "value",
            "raw_value",
            "normalized_value",
            "printed_fields",
            "field_source_refs",
        )
    ):
        return None
    physical_id = str(observation.get("source_physical_row_id") or "")
    refs = observation.get("source_refs")
    if (
        not physical_id.startswith("source_inquiry_raw_physical_position:")
        or not isinstance(refs, list)
        or len(refs) != 1
        or not isinstance(refs[0], Mapping)
    ):
        return None
    ref = dict(refs[0])
    signature = _exact_inquiry_source_signature(ref)
    evidence_ids = _unique_source_evidence_ids(ref.get("evidence_ids"))
    contract = (
        ref.get("source"),
        ref.get("geometry_scope"),
        ref.get("binding"),
        ref.get("binding_quality"),
    )
    if (
        contract
        not in {
            (
                "native_detail_inquiry_raw_physical_row",
                "token_y_band",
                "sealed_raw_inquiry_registered_lattice_band",
                "all_exact_tokens_uniquely_partitioned",
            ),
            (
                "native_detail_inquiry_raw_physical_row",
                "exact_source_row_band",
                "sealed_raw_inquiry_registered_lattice_band",
                "sealed_exact_physical_position",
            ),
            (
                "native_detail_inquiry_raw_physical_row",
                "exact_source_row_band",
                "sealed_raw_inquiry_anonymous_exact_grid_row",
                "exact_grid_below_non_authoritative_registered_header",
            ),
        }
        or signature is None
        or not _finite_source_bbox(ref.get("bbox"))
        or evidence_ids is None
        or physical_id
        != stable_record_id(
            "source_inquiry_raw_physical_position",
            *signature,
            *sorted(evidence_ids),
        )
    ):
        return None
    return physical_id, ref, evidence_ids


def _inquiry_row_local_evidence_ids(row: Mapping[str, Any]) -> frozenset[str]:
    refs: list[Mapping[str, Any]] = []
    owners = [row]
    normalized = row.get("normalized")
    if isinstance(normalized, Mapping):
        owners.append(normalized)
    for owner in owners:
        refs.extend(
            ref
            for ref in owner.get("source_refs") or ()
            if isinstance(ref, Mapping)
        )
        refs_by_field = owner.get("source_refs_by_field")
        if isinstance(refs_by_field, Mapping):
            refs.extend(
                ref
                for field_refs in refs_by_field.values()
                for ref in (field_refs or ())
                if isinstance(ref, Mapping)
            )
    evidence_ids: set[str] = set()
    for ref in refs:
        if str(ref.get("geometry_scope") or "") in {
            "document",
            "page",
            "table",
        }:
            continue
        exact_ids = _unique_source_evidence_ids(ref.get("evidence_ids"))
        if exact_ids is not None:
            evidence_ids.update(exact_ids)
    return frozenset(evidence_ids)


def _inquiry_row_local_evidence_owners(
    row: Mapping[str, Any],
) -> list[tuple[tuple[int, int, str, int], frozenset[str]]]:
    """Return row-local emitted evidence grouped by one exact physical owner."""

    refs: list[Mapping[str, Any]] = []
    owners = [row]
    normalized = row.get("normalized")
    if isinstance(normalized, Mapping):
        owners.append(normalized)
    for owner in owners:
        refs.extend(
            ref
            for ref in owner.get("source_refs") or ()
            if isinstance(ref, Mapping)
        )
        refs_by_field = owner.get("source_refs_by_field")
        if isinstance(refs_by_field, Mapping):
            refs.extend(
                ref
                for field_refs in refs_by_field.values()
                if isinstance(field_refs, (list, tuple))
                for ref in field_refs
                if isinstance(ref, Mapping)
            )

    evidence_by_signature: dict[tuple[int, int, str, int], set[str]] = {}
    for ref in refs:
        if ref.get("geometry_scope") in {"document", "page", "table"}:
            continue
        signature = _exact_inquiry_source_signature(ref)
        evidence_ids = _unique_source_evidence_ids(ref.get("evidence_ids"))
        if (
            signature is None
            or evidence_ids is None
            or not _finite_source_bbox(ref.get("bbox"))
        ):
            continue
        evidence_by_signature.setdefault(signature, set()).update(evidence_ids)
    return [
        (signature, frozenset(evidence_ids))
        for signature, evidence_ids in sorted(evidence_by_signature.items())
        if evidence_ids
    ]


def _typed_inquiry_field_evidence_sets(
    ledger: Mapping[str, Any],
) -> set[frozenset[str]]:
    result: set[frozenset[str]] = set()
    physical_by_id: dict[str, frozenset[str]] = {}
    duplicate_physical_ids: set[str] = set()
    for raw_physical in ledger.get("inquiry_physical_field_observations") or ():
        exact_physical = _exact_inquiry_physical_observation(raw_physical)
        if exact_physical is None:
            continue
        physical_row_id, _physical_refs, physical_ids, _signature = exact_physical
        if physical_row_id in physical_by_id:
            physical_by_id.pop(physical_row_id, None)
            duplicate_physical_ids.add(physical_row_id)
        elif physical_row_id not in duplicate_physical_ids:
            physical_by_id[physical_row_id] = physical_ids
    by_type = ledger.get("inquiry_ordinal_observations")
    endpoints = ledger.get("inquiry_sequence_endpoints")
    observed_map = ledger.get("inquiry_observed_sequences")
    outlier_map = ledger.get("inquiry_sequence_outliers")
    if (
        not isinstance(by_type, Mapping)
        or not isinstance(endpoints, Mapping)
        or not isinstance(observed_map, Mapping)
        or not physical_by_id
    ):
        return result
    for inquiry_type, observations in by_type.items():
        inquiry_type = str(inquiry_type)
        endpoint = _positive_int(endpoints.get(inquiry_type))
        outliers = (
            outlier_map.get(inquiry_type)
            if isinstance(outlier_map, Mapping)
            else ()
        )
        if (
            inquiry_type not in _INQUIRY_POPULATION_TYPES
            or endpoint is None
            or not _dense_endpoint_is_credible(
                endpoint,
                observed_map.get(inquiry_type),
                outliers,
            )
        ):
            continue
        if not isinstance(observations, Mapping):
            continue
        for raw_ordinal, observation in observations.items():
            if not isinstance(observation, Mapping):
                continue
            ordinal = _positive_int(observation.get("sequence"))
            try:
                key_ordinal = int(str(raw_ordinal))
            except (TypeError, ValueError):
                continue
            if (
                ordinal is None
                or ordinal != key_ordinal
                or ordinal > endpoint
                or observation.get("inquiry_type") != inquiry_type
            ):
                continue
            refs_by_field = observation.get("field_source_refs")
            physical_bindings = observation.get("physical_field_observation_ids")
            printed_fields = {
                str(field_name)
                for field_name in observation.get("printed_fields") or ()
            }
            if (
                not isinstance(refs_by_field, Mapping)
                or printed_fields != {"inquiry_date", "institution", "reason"}
                or not isinstance(physical_bindings, list)
                or len(physical_bindings) != 1
                or str(physical_bindings[0]) not in physical_by_id
            ):
                continue
            evidence_ids: set[str] = set()
            valid = True
            owner_signature: tuple[Any, Any, Any, Any] | None = None
            for field_name in ("inquiry_date", "institution", "reason"):
                raw_refs = refs_by_field.get(field_name)
                if (
                    not isinstance(raw_refs, list)
                    or len(raw_refs) != 1
                    or not isinstance(raw_refs[0], Mapping)
                ):
                    valid = False
                    break
                ref = raw_refs[0]
                exact_ids = _exact_typed_inquiry_field_ref(
                    ref,
                    field_name=field_name,
                )
                signature = (
                    ref.get("logical_page"),
                    ref.get("source_page"),
                    ref.get("table_id"),
                    ref.get("row"),
                )
                if (
                    exact_ids is None
                    or evidence_ids.intersection(exact_ids)
                    or (
                        owner_signature is not None
                        and signature != owner_signature
                    )
                ):
                    valid = False
                    break
                owner_signature = signature
                evidence_ids.update(exact_ids)
            if (
                valid
                and evidence_ids
                and frozenset(evidence_ids)
                == physical_by_id[str(physical_bindings[0])]
            ):
                result.add(frozenset(evidence_ids))
    return result


def _exact_typed_inquiry_identity_owners(
    ledger: Mapping[str, Any],
) -> list[tuple[tuple[int, int, str, int], str, frozenset[str]]]:
    """Return unique typed ordinals bound to one exact physical source row."""

    endpoints = ledger.get("inquiry_sequence_endpoints")
    observed_map = ledger.get("inquiry_observed_sequences")
    outlier_map = ledger.get("inquiry_sequence_outliers")
    by_type = ledger.get("inquiry_ordinal_observations")
    if not all(isinstance(value, Mapping) for value in (endpoints, observed_map, by_type)):
        return []
    candidates: list[
        tuple[tuple[int, int, str, int], str, frozenset[str]]
    ] = []
    for inquiry_type, observations in by_type.items():
        inquiry_type = str(inquiry_type)
        endpoint = _positive_int(endpoints.get(inquiry_type))
        outliers = outlier_map.get(inquiry_type) if isinstance(outlier_map, Mapping) else ()
        if (
            inquiry_type not in _INQUIRY_POPULATION_TYPES
            or endpoint is None
            or not _dense_endpoint_is_credible(
                endpoint,
                observed_map.get(inquiry_type),
                outliers,
            )
            or not isinstance(observations, Mapping)
        ):
            continue
        for raw_ordinal, observation in observations.items():
            if not isinstance(observation, Mapping):
                continue
            ordinal = observation.get("sequence")
            if isinstance(raw_ordinal, int) and not isinstance(raw_ordinal, bool):
                key_ordinal = raw_ordinal
            elif isinstance(raw_ordinal, str) and re.fullmatch(
                r"[1-9]\d*", raw_ordinal
            ):
                key_ordinal = int(raw_ordinal)
            else:
                continue
            refs = observation.get("source_refs")
            if (
                not isinstance(ordinal, int)
                or isinstance(ordinal, bool)
                or ordinal <= 0
                or ordinal != key_ordinal
                or ordinal > endpoint
                or observation.get("inquiry_type") != inquiry_type
                or not isinstance(refs, list)
                or len(refs) != 1
                or not isinstance(refs[0], Mapping)
            ):
                continue
            ref = refs[0]
            signature = _exact_inquiry_source_signature(ref)
            evidence_ids = _unique_source_evidence_ids(ref.get("evidence_ids"))
            source_contract = (
                ref.get("source"),
                ref.get("geometry_scope"),
                ref.get("binding"),
                ref.get("binding_quality"),
            )
            ordinal_contract = (
                "native_detail_inquiry_token_ordinal",
                "token",
                "printed_inquiry_ordinal_token",
                "exact_token_in_sequence_cell",
            )
            token_row_contract = (
                "native_detail_inquiry_token_row",
                "token_row",
                "canonical_header_token_row",
                "exact_token_row",
            )
            native_row_contract = (
                "native_detail_table",
                "row",
                "canonical_header_row",
                None,
            )
            if source_contract == ordinal_contract:
                column = ref.get("column")
                if (
                    not isinstance(column, int)
                    or isinstance(column, bool)
                    or column < 0
                    or ref.get("sequence") != ordinal
                    or evidence_ids is None
                    or len(evidence_ids) != 1
                ):
                    continue
                coverage_kind = "ordinal_token"
            elif source_contract == token_row_contract:
                if ref.get("sequence") != ordinal or ref.get("column") is not None:
                    continue
                coverage_kind = "complete_token_row"
            elif source_contract == native_row_contract:
                ref_sequence = ref.get("sequence")
                if (
                    ref_sequence is not None
                    and (
                        not isinstance(ref_sequence, int)
                        or isinstance(ref_sequence, bool)
                        or ref_sequence != ordinal
                    )
                ) or ref.get("column") is not None:
                    continue
                coverage_kind = "complete_native_row"
            else:
                continue
            if (
                evidence_ids is None
                or signature is None
                or not _finite_source_bbox(ref.get("bbox"))
            ):
                continue
            candidates.append((signature, coverage_kind, evidence_ids))
    owner_counts = Counter(candidates)
    atom_counts: Counter[str] = Counter(
        evidence_id
        for _signature, _coverage_kind, evidence_ids in candidates
        for evidence_id in evidence_ids
    )
    return [
        owner
        for owner in candidates
        if owner_counts[owner] == 1
        and all(atom_counts[evidence_id] == 1 for evidence_id in owner[2])
    ]


def _complete_inquiry_physical_observation_owners(
    ledger: Mapping[str, Any],
) -> list[tuple[str, tuple[int, int, str, int], frozenset[str]]]:
    """Return unique complete three-field owners for raw-row suppression."""

    candidates: list[
        tuple[str, tuple[int, int, str, int], frozenset[str]]
    ] = []
    for raw_observation in ledger.get("inquiry_physical_field_observations") or ():
        exact = _exact_inquiry_physical_observation(raw_observation)
        if exact is not None:
            physical_id, _refs, field_ids, signature = exact
            candidates.append((physical_id, signature, field_ids))
    id_counts = Counter(physical_id for physical_id, _signature, _ids in candidates)
    atom_counts: Counter[str] = Counter(
        evidence_id
        for _physical_id, _signature, ids in candidates
        for evidence_id in ids
    )
    return [
        (physical_id, signature, ids)
        for physical_id, signature, ids in candidates
        if id_counts[physical_id] == 1
        and all(atom_counts[evidence_id] == 1 for evidence_id in ids)
    ]


def _append_exact_inquiry_raw_position_issues(
    ledger: Mapping[str, Any],
    datasets: Mapping[str, Any],
    issues: list[dict[str, Any]],
    existing_ids: set[str],
) -> None:
    """Report one unconsumed anonymous row/band without inventing identity."""

    raw_positions = ledger.get("inquiry_raw_physical_positions")
    if not isinstance(raw_positions, list):
        return
    candidates = [
        exact
        for raw_position in raw_positions
        if (exact := _exact_inquiry_raw_physical_position(raw_position)) is not None
    ]
    id_counts = Counter(physical_id for physical_id, _ref, _ids in candidates)
    atom_counts: Counter[str] = Counter(
        evidence_id for _physical_id, _ref, ids in candidates for evidence_id in ids
    )
    unique_positions = [
        candidate
        for candidate in candidates
        if id_counts[candidate[0]] == 1
        and all(atom_counts[evidence_id] == 1 for evidence_id in candidate[2])
    ]
    if not unique_positions:
        return
    emitted_owners = [
        owner
        for row in datasets.get("inquiry_records") or ()
        if isinstance(row, Mapping)
        for owner in _inquiry_row_local_evidence_owners(row)
    ]
    typed_owners = _exact_typed_inquiry_identity_owners(ledger)
    complete_fields = _complete_inquiry_physical_observation_owners(ledger)
    for physical_id, ref, evidence_ids in unique_positions:
        signature = _exact_inquiry_source_signature(ref)
        if signature is None:
            continue
        typed_matches = [
            owner
            for owner in typed_owners
            if owner[0] == signature
            and (
                owner[2].issubset(evidence_ids)
                if owner[1] == "ordinal_token"
                else owner[2] == evidence_ids
            )
        ]
        complete_matches = [
            owner
            for owner in complete_fields
            if owner[1] == signature and owner[2].issubset(evidence_ids)
        ]
        if (
            any(
                owner_signature == signature
                and evidence_ids.issubset(owner_ids)
                for owner_signature, owner_ids in emitted_owners
            )
            or len(typed_matches) == 1
            or len(complete_matches) == 1
        ):
            continue
        _append_population_issue(
            issues,
            existing_ids,
            category="schema_incompleteness",
            issue_code="source_inquiry_physical_record_omitted",
            message=(
                "A sealed inquiry source row or physical band was present but "
                "was not consumed by a canonical or field-local projection."
            ),
            parser_stage="source_completeness_ledger",
            target_dataset="inquiry_records",
            target_record_id=physical_id,
            field_name="inquiry_id",
            observed_value={"source_row_observed": True},
            candidate_value={"source_physical_row_id": physical_id},
            source_refs=[ref],
            reason_codes=(
                "independent_source_ledger",
                "sealed_raw_inquiry_physical_position",
                "anonymous_source_identity",
                "inquiry_type_withheld",
                "ordinal_withheld",
                "normalized_value_withheld",
            ),
        )


def _append_exact_inquiry_physical_field_issues(
    ledger: Mapping[str, Any],
    datasets: Mapping[str, Any],
    issues: list[dict[str, Any]],
    existing_ids: set[str],
) -> None:
    observations = ledger.get("inquiry_physical_field_observations")
    if not isinstance(observations, list):
        return
    emitted_evidence = [
        evidence_ids
        for row in datasets.get("inquiry_records") or ()
        if isinstance(row, Mapping)
        and (evidence_ids := _inquiry_row_local_evidence_ids(row))
    ]
    typed_evidence = _typed_inquiry_field_evidence_sets(ledger)
    seen_physical_ids: set[str] = set()
    for raw_observation in observations:
        exact = _exact_inquiry_physical_observation(raw_observation)
        if exact is None:
            continue
        physical_row_id, refs_by_field, field_evidence, _signature = exact
        if physical_row_id in seen_physical_ids:
            continue
        seen_physical_ids.add(physical_row_id)
        if (
            field_evidence in typed_evidence
            or any(field_evidence.issubset(evidence_ids) for evidence_ids in emitted_evidence)
        ):
            continue
        for field_name in ("inquiry_date", "institution", "reason"):
            _append_population_issue(
                issues,
                existing_ids,
                category="schema_incompleteness",
                issue_code="source_inquiry_field_omitted",
                message=(
                    "A source-observed inquiry field belongs to an exact "
                    "physical row band that was not projected."
                ),
                parser_stage="source_completeness_ledger",
                target_dataset="inquiry_records",
                target_record_id=physical_row_id,
                field_name=field_name,
                observed_value={"source_field_observed": True},
                candidate_value={"source_physical_row_id": physical_row_id},
                source_refs=refs_by_field[field_name],
                reason_codes=(
                    "independent_source_ledger",
                    "exact_inquiry_physical_y_band",
                    "ordinal_identity_withheld",
                    "printed_field_observed",
                    "missing_business_field_projection",
                    "normalized_value_withheld",
                ),
            )


def _append_exact_inquiry_population_issues(
    ledger: Mapping[str, Any],
    datasets: Mapping[str, Any],
    issues: list[dict[str, Any]],
    existing_ids: set[str],
) -> None:
    _append_exact_inquiry_raw_position_issues(
        ledger,
        datasets,
        issues,
        existing_ids,
    )
    _append_exact_inquiry_physical_field_issues(
        ledger,
        datasets,
        issues,
        existing_ids,
    )
    typed_physical_evidence = _typed_inquiry_field_evidence_sets(ledger)
    endpoints = ledger.get("inquiry_sequence_endpoints")
    observed_map = ledger.get("inquiry_observed_sequences")
    outlier_map = ledger.get("inquiry_sequence_outliers")
    if not isinstance(endpoints, Mapping) or not isinstance(observed_map, Mapping):
        return
    emitted_by_identity: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in datasets.get("inquiry_records") or ():
        if not isinstance(row, Mapping):
            continue
        identity = _canonical_inquiry_ordinal(_record_values(row))
        if identity is not None:
            emitted_by_identity.setdefault(identity, []).append(row)
    for inquiry_type in sorted(_INQUIRY_POPULATION_TYPES):
        endpoint = _positive_int(endpoints.get(inquiry_type))
        outliers = outlier_map.get(inquiry_type) if isinstance(outlier_map, Mapping) else ()
        if endpoint is None or not _dense_endpoint_is_credible(
            endpoint,
            observed_map.get(inquiry_type),
            outliers,
        ):
            continue
        for ordinal in range(1, endpoint + 1):
            identity = (inquiry_type, ordinal)
            emitted_rows = emitted_by_identity.get(identity, [])
            observation = _nested_ordinal_observation(
                ledger,
                "inquiry_ordinal_observations",
                inquiry_type,
                ordinal,
            )
            refs = _local_ordinal_source_refs(observation)
            if not refs:
                continue
            if not emitted_rows:
                _append_population_issue(
                    issues,
                    existing_ids,
                    category="schema_incompleteness",
                    issue_code="source_inquiry_record_omitted",
                    message="A source-proven inquiry type/ordinal was not projected as a canonical inquiry record.",
                    parser_stage="source_completeness_ledger",
                    target_dataset="inquiry_records",
                    target_record_id=f"credit_inquiry:{inquiry_type}:{ordinal}",
                    field_name="inquiry_id",
                    observed_value={"inquiry_type": inquiry_type, "sequence": ordinal},
                    candidate_value={"source_sequence_endpoint": endpoint},
                    source_refs=refs,
                    reason_codes=(
                        "independent_source_ledger",
                        "exact_inquiry_type_ordinal",
                        "dense_sequence_endpoint",
                        "missing_business_record",
                        "normalized_value_withheld",
                    ),
                )
            field_refs = observation.get("field_source_refs")
            field_ref_map = (
                dict(field_refs) if isinstance(field_refs, Mapping) else {}
            )
            printed_fields = {
                str(field_name)
                for field_name in observation.get("printed_fields") or ()
                if str(field_name) in {"inquiry_date", "institution", "reason"}
            }
            observation_field_ids: set[str] = set()
            exact_ids_by_field: dict[str, frozenset[str]] = {}
            for physical_field_name in (
                "inquiry_date",
                "institution",
                "reason",
            ):
                raw_field_refs = field_ref_map.get(physical_field_name)
                if (
                    not isinstance(raw_field_refs, list)
                    or len(raw_field_refs) != 1
                ):
                    continue
                exact_ids = _exact_typed_inquiry_field_ref(
                    raw_field_refs[0],
                    field_name=physical_field_name,
                )
                if (
                    exact_ids is None
                    or observation_field_ids.intersection(exact_ids)
                ):
                    continue
                observation_field_ids.update(exact_ids)
                exact_ids_by_field[physical_field_name] = exact_ids
            bound_physical_evidence = frozenset(observation_field_ids)
            physical_binding_is_exact = (
                len(exact_ids_by_field) == 3
                and bound_physical_evidence in typed_physical_evidence
            )
            for field_name in sorted(printed_fields):
                local_refs = [
                    dict(ref)
                    for ref in field_ref_map.get(field_name) or ()
                    if isinstance(ref, Mapping)
                ]
                if not local_refs:
                    continue
                if emitted_rows:
                    # Ordinary typed rows retain their historical identity
                    # consumption.  A y-band merge is stricter: its exact
                    # field atoms must occur in one emitted row, otherwise the
                    # typed field remains locally unconsumed.
                    if not physical_binding_is_exact:
                        continue
                    if any(
                        exact_ids_by_field[field_name].issubset(
                            _inquiry_row_local_evidence_ids(row)
                        )
                        for row in emitted_rows
                    ):
                        continue
                _append_population_issue(
                    issues,
                    existing_ids,
                    category="schema_incompleteness",
                    issue_code="source_inquiry_field_omitted",
                    message=(
                        "A source-observed inquiry field was not consumed by "
                        "the uniquely typed projected row."
                        if emitted_rows
                        else "A source-observed inquiry cell belongs to an exact row that was not projected."
                    ),
                    parser_stage="source_completeness_ledger",
                    target_dataset="inquiry_records",
                    target_record_id=f"credit_inquiry:{inquiry_type}:{ordinal}",
                    field_name=field_name,
                    observed_value={"source_field_observed": True},
                    candidate_value={
                        "inquiry_type": inquiry_type,
                        "sequence": ordinal,
                    },
                    source_refs=local_refs,
                    reason_codes=(
                        "independent_source_ledger",
                        "exact_inquiry_type_ordinal",
                        "printed_field_observed",
                        (
                            "missing_business_field_projection"
                            if emitted_rows
                            else "missing_business_record"
                        ),
                        "normalized_value_withheld",
                    ),
                )


def _active_profile_population_issue_keys(
    issues: Iterable[Any], dataset: str
) -> set[tuple[str, str]]:
    inactive_statuses = {"resolved", "closed", "dismissed", "superseded"}
    keys: set[tuple[str, str]] = set()
    for issue in issues:
        values = _record_values(issue)
        if str(values.get("target_dataset") or "") != dataset:
            continue
        status = str(values.get("status") or "active").strip().lower()
        if status in inactive_statuses:
            continue
        target_id = values.get("target_record_id")
        field_name = values.get("field_name")
        if isinstance(target_id, str) and target_id and isinstance(field_name, str):
            keys.add((target_id, field_name))
    return keys


def _exact_emitted_profile_population_rows(
    datasets: Mapping[str, Any],
    dataset: str,
    contract: Mapping[str, Any],
) -> dict[int, dict[str, Any]] | None:
    emitted: dict[int, dict[str, Any]] = {}
    id_field = str(contract["id_field"])
    id_prefix = str(contract["id_prefix"])
    for row in datasets.get(dataset) or ():
        if not isinstance(row, Mapping):
            continue
        values = _record_values(row)
        ordinal = _strict_raw_profile_positive_int(values.get("sequence"))
        if ordinal is None:
            continue
        expected_id = stable_record_id(id_prefix, ordinal)
        identities = {
            value
            for value in (
                row.get("record_id"),
                values.get("record_id"),
                values.get(id_field),
            )
            if isinstance(value, str) and value
        }
        if identities != {expected_id}:
            return None
        if ordinal in emitted:
            return None
        emitted[ordinal] = values
    return emitted


def _append_exact_sequence_population_issues(
    ledger: Mapping[str, Any],
    datasets: Mapping[str, Any],
    issues: list[dict[str, Any]],
    existing_ids: set[str],
) -> None:
    for dataset_name, contract in _SEQUENCE_POPULATION_DATASETS.items():
        validated = _exact_raw_profile_observations(ledger, dataset_name)
        if validated is None:
            continue
        endpoint, observations = validated
        id_field = str(contract["id_field"])
        id_prefix = str(contract["id_prefix"])
        emitted = _exact_emitted_profile_population_rows(
            datasets, dataset_name, contract
        )
        if emitted is None:
            continue
        active_keys = _active_profile_population_issue_keys(issues, dataset_name)
        issue_stem = {
            "mobile_phone_records": "mobile",
            "residence_records": "residence",
            "employment_records": "employment",
        }[dataset_name]
        for ordinal in range(1, endpoint + 1):
            observation = observations[ordinal]
            refs = [dict(observation["source_refs"][0])]
            target = stable_record_id(id_prefix, ordinal)
            emitted_values = emitted.get(ordinal)
            row_missing = emitted_values is None
            if row_missing and (target, id_field) not in active_keys:
                _append_population_issue(
                    issues,
                    existing_ids,
                    category="schema_incompleteness",
                    issue_code=f"source_{issue_stem}_record_omitted",
                    message=(
                        "A source-observed immutable PBOC ordinal row was not "
                        "projected as a business record."
                    ),
                    parser_stage="source_completeness_ledger",
                    target_dataset=dataset_name,
                    target_record_id=target,
                    field_name=id_field,
                    observed_value={
                        "dataset_name": dataset_name,
                        "sequence": ordinal,
                    },
                    candidate_value={"source_sequence_endpoint": endpoint},
                    source_refs=refs,
                    reason_codes=(
                        "independent_source_ledger",
                        "exact_sequence_ordinal",
                        "ordinal_local_source_evidence",
                        "missing_business_record",
                        "normalized_value_withheld",
                    ),
                )
                active_keys.add((target, id_field))
            field_ref_map = dict(observation["field_source_refs"])
            printed_fields = set(observation["printed_fields"])
            for field_name in sorted(printed_fields):
                source_absent = (
                    set(emitted_values.get("_source_absent_fields") or ())
                    if emitted_values is not None
                    and isinstance(
                        emitted_values.get("_source_absent_fields"),
                        (list, tuple, set, frozenset),
                    )
                    else set()
                )
                if (
                    not row_missing
                    and emitted_values.get(field_name) not in (None, "")
                ) or field_name in source_absent or (target, field_name) in active_keys:
                    continue
                local_refs = [dict(field_ref_map[field_name][0])]
                _append_population_issue(
                    issues,
                    existing_ids,
                    category="schema_incompleteness",
                    issue_code=f"source_{issue_stem}_field_omitted",
                    message=(
                        "A source-observed exact PBOC field cell was withheld "
                        "without a field-local issue."
                        if not row_missing
                        else "A source-observed printed field belongs to an exact row that was not projected."
                    ),
                    parser_stage="source_completeness_ledger",
                    target_dataset=dataset_name,
                    target_record_id=target,
                    field_name=field_name,
                    observed_value={"source_field_observed": True},
                    candidate_value={"sequence": ordinal},
                    source_refs=local_refs,
                    reason_codes=(
                        "independent_source_ledger",
                        "exact_sequence_ordinal",
                        "printed_field_observed",
                        "exact_source_cell",
                        (
                            "missing_business_record"
                            if row_missing
                            else "missing_business_field_projection"
                        ),
                        "normalized_value_withheld",
                    ),
                )
                active_keys.add((target, field_name))


def _apply_source_completeness_ledger(
    facts: dict[str, Any],
    datasets: dict[str, Any],
    final_dataset_counts: Mapping[str, int],
) -> None:
    """Turn independent sequence/count evidence into explicit partial states."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue

    ledger = facts.get("personal_detail_source_completeness_ledger")
    ledger_map = dict(ledger) if isinstance(ledger, Mapping) else {}
    states = facts.setdefault("personal_detail_dataset_states", {})
    if not isinstance(states, dict):
        states = {}
        facts["personal_detail_dataset_states"] = states
    issues = datasets.setdefault("personal_detail_extraction_issues", [])
    if not isinstance(issues, list):
        issues = []
        datasets["personal_detail_extraction_issues"] = issues

    checks: dict[str, tuple[int, int, bool]] = {}
    for dataset_name in (
        "mobile_phone_records",
        "residence_records",
        "employment_records",
    ):
        local_rows = datasets.get(dataset_name) or ()
        observed = sum(isinstance(row, Mapping) for row in local_rows)
        sequence_expected, contiguous = _sequence_evidence(local_rows)
        validated_profile = _exact_raw_profile_observations(
            ledger_map, dataset_name
        )
        source_expected = validated_profile[0] if validated_profile is not None else 0
        expected = max(
            sequence_expected,
            source_expected,
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

    _append_exact_account_population_issues(
        ledger_map,
        datasets,
        issues,
        existing_ids,
    )
    _append_exact_agreement_population_issues(
        ledger_map,
        datasets,
        issues,
        existing_ids,
    )
    _append_exact_inquiry_population_issues(
        ledger_map,
        datasets,
        issues,
        existing_ids,
    )
    _append_exact_sequence_population_issues(
        ledger_map,
        datasets,
        issues,
        existing_ids,
    )


def _apply_account_month_closure_ledger(
    facts: dict[str, Any],
    datasets: dict[str, Any],
) -> None:
    """Publish canonical identity closure and a conserved source-position ledger."""

    closure_proof_dataset = "_personal_detail_account_month_closure_proof"
    # ``content`` can be prepared more than once during projection.  Never let
    # an earlier closure proof survive a later empty or invalid source plane.
    datasets.pop(closure_proof_dataset, None)
    month_pattern = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")
    alias_issue_code = "candidate_b_monthly_source_position_alias_reconciled"
    detached_alias_sibling_issue_code = (
        "canonical_monthly_source_structure_missing_field"
    )
    owned_grid_issue_code = "candidate_b_monthly_owned_grid_missing_field"
    owned_grid_ref_source = "candidate_b_monthly_owned_grid_cell"
    allowed_account_month_owner_bases = frozenset(
        {
            "canonical_account_segment",
            "explicit_account_id_confirmed_by_canonical_segment",
            "exact_source_table_account_owner",
        }
    )
    detached_alias_reason_codes = frozenset(
        {
            "exact_account_month_identity",
            "distinct_source_position_alias",
            "canonical_identity_not_double_counted",
            "source_position_audit_preserved",
        }
    )
    detached_sibling_reason_codes = frozenset(
        {
            "detached_source_structure_exact_key",
            "canonical_deduplicated_key_missing",
            "source_structure_is_audit_only",
            "account_month_owner_reconciliation_pending",
            "dataset_incomplete",
            "exact_grid_month_source_position",
            "normalized_value_withheld",
            "owner_or_status_value_not_invented",
        }
    )
    detached_sibling_target_fields = frozenset(
        {"performance_month", "status_code", "status_amount"}
    )

    def exact_month(values: Mapping[str, Any]) -> str | None:
        performance_month = str(values.get("performance_month") or "").strip()
        if month_pattern.fullmatch(performance_month):
            return performance_month
        year = values.get("year")
        month = values.get("month")
        if (
            isinstance(year, int)
            and not isinstance(year, bool)
            and isinstance(month, int)
            and not isinstance(month, bool)
            and 1900 <= year <= 9999
            and 1 <= month <= 12
        ):
            return f"{year:04d}-{month:02d}"
        return None

    def raw_issue_ref_containers(
        issue: Any,
        values: Mapping[str, Any],
    ) -> list[Any]:
        """Return every source-ref container exactly where the row supplied it."""

        if not isinstance(issue, Mapping):
            return [values["source_refs"]] if "source_refs" in values else []
        normalized = issue.get("normalized")
        owners: list[Mapping[str, Any]] = []
        if isinstance(normalized, Mapping):
            owners.append(normalized)
            owners.append(issue)
        else:
            # ``values`` is a shallow copy of a direct issue in this branch;
            # inspect the actual row only so one container is not counted twice.
            owners.append(issue)
        source = issue.get("source")
        if isinstance(source, Mapping):
            owners.append(source)
        return [owner["source_refs"] for owner in owners if "source_refs" in owner]

    def issue_refs(issue: Any, values: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        for refs in raw_issue_ref_containers(issue, values):
            if isinstance(refs, (list, tuple)):
                return [ref for ref in refs if isinstance(ref, Mapping)]
        return []

    def exact_issue_refs(
        issue: Any,
        values: Mapping[str, Any],
    ) -> list[Mapping[str, Any]] | None:
        """Retain raw container cardinality for strict producer claims."""

        containers = raw_issue_ref_containers(issue, values)
        if len(containers) != 1:
            return None
        refs = containers[0]
        if (
            not isinstance(refs, (list, tuple))
            or len(refs) != 2
            or any(not isinstance(ref, Mapping) for ref in refs)
        ):
            return None
        return list(refs)

    def record_refs(record: Any, values: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        refs: list[Mapping[str, Any]] = []
        owners = [values]
        if isinstance(record, Mapping) and record is not values:
            owners.append(record)
        for owner in owners:
            for key in ("source_cell_refs", "source_refs"):
                refs.extend(
                    ref
                    for ref in owner.get(key) or ()
                    if isinstance(ref, Mapping)
                )
        return refs

    def physical_position_fingerprints(
        refs: Iterable[Mapping[str, Any]],
        *,
        account_id: str,
        performance_month: str,
    ) -> tuple[
        set[tuple[Any, ...]],
        dict[tuple[Any, ...], list[dict[str, Any]]],
    ]:
        """Return immutable, grid-alias-independent identities for one month.

        A source diagnostic can carry a detector-local ``grid_id`` that differs
        from the canonical or alias inventory even though both point at the
        same sealed cell.  Exact evidence atoms and exact physical geometry are
        independent identity planes: a detector may refine a box without
        changing its sealed evidence, or add evidence without changing the
        physical box.  Requiring both at once turns either harmless refinement
        into a second physical position.  Parser, table, and grid aliases
        deliberately do not participate in either identity.  The exact source-
        table month lattice also bridges an early grid-band alias ref to the
        final source-table-calibrated cell refs produced later in the lifecycle.
        """

        def exact_native_index(value: Any, *, positive: bool) -> int | None:
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            if (positive and value <= 0) or (not positive and value < 0):
                return None
            return value

        def exact_bbox(value: Any) -> tuple[float, float, float, float] | None:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 4
                or any(
                    not isinstance(coordinate, (int, float))
                    or isinstance(coordinate, bool)
                    or not math.isfinite(float(coordinate))
                    for coordinate in value
                )
            ):
                return None
            bbox = tuple(float(coordinate) for coordinate in value)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return None
            return bbox

        def exact_source_table_cell_identity(
            ref: Mapping[str, Any],
            provenance: Any,
            *,
            page: int,
            bbox: tuple[float, float, float, float] | None,
        ) -> tuple[int, int | None, str] | None:
            """Mirror the closed source-table month-cell identity contract."""

            if not (
                isinstance(provenance, Mapping)
                and provenance.get("source") == "source_table_geometry"
                and ref.get("coordinate_system") == "pdf_points_top_left"
                and provenance.get("coordinate_system")
                == "pdf_points_top_left"
                and ref.get("geometry_scope") == "cell"
                and "page" in ref
                and "logical_page" in ref
                and "logical_page" in provenance
                and isinstance(provenance.get("table_id"), str)
                and bool(provenance["table_id"].strip())
                and provenance.get("calibrated_from_source_table_geometry")
                is True
                and provenance.get("active_cell_geometry_exact") is True
                and provenance.get("value_inputs_used") is False
                and bbox is not None
            ):
                return None

            page_aliases = [
                (ref, "page"),
                (ref, "logical_page"),
                (provenance, "logical_page"),
            ]
            if "page" in provenance:
                page_aliases.append((provenance, "page"))
            page_values: list[int] = []
            for owner, key in page_aliases:
                value = exact_native_index(owner.get(key), positive=True)
                if value is None:
                    return None
                page_values.append(value)
            if set(page_values) != {page}:
                return None

            table_ids = {provenance["table_id"].strip()}
            if "table_id" in ref:
                raw_table_id = ref.get("table_id")
                if not isinstance(raw_table_id, str) or not raw_table_id.strip():
                    return None
                table_ids.add(raw_table_id.strip())
            if len(table_ids) != 1:
                return None

            source_pages: set[int] = set()
            for owner in (ref, provenance):
                if "source_page" not in owner:
                    continue
                source_page = exact_native_index(
                    owner.get("source_page"),
                    positive=True,
                )
                if source_page is None:
                    return None
                source_pages.add(source_page)
            if len(source_pages) > 1:
                return None

            if "bbox" in provenance:
                provenance_bbox = exact_bbox(provenance.get("bbox"))
                if provenance_bbox is None or provenance_bbox != bbox:
                    return None
            return page, next(iter(source_pages), None), next(iter(table_ids))

        fingerprints: set[tuple[Any, ...]] = set()
        physical_claim_refs: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        expected_month = int(performance_month[5:7])
        for ref in refs:
            raw_ref_month = ref.get("performance_month")
            if raw_ref_month in (None, ""):
                ref_month = ""
            elif isinstance(raw_ref_month, str):
                ref_month = raw_ref_month.strip()
            else:
                continue
            if ref_month and ref_month != performance_month:
                continue

            column_values: set[int] = set()
            malformed_coordinate = False
            for key in ("column", "col"):
                if key not in ref:
                    continue
                column = exact_native_index(ref.get(key), positive=False)
                if column is None:
                    malformed_coordinate = True
                    break
                column_values.add(column)
            if malformed_coordinate or len(column_values) > 1:
                continue
            if "row" in ref and exact_native_index(
                ref.get("row"), positive=False
            ) is None:
                continue
            raw_column = next(iter(column_values), None)
            if ref_month:
                if raw_column is not None and raw_column != expected_month:
                    continue
            elif raw_column != expected_month:
                continue

            page_values: set[int] = set()
            malformed_page = False
            for key in ("logical_page", "page"):
                if key not in ref:
                    continue
                page_value = exact_native_index(ref.get(key), positive=True)
                if page_value is None:
                    malformed_page = True
                    break
                page_values.add(page_value)
            if malformed_page or len(page_values) != 1:
                continue
            page = next(iter(page_values))
            source_page: int | None = None
            if "source_page" in ref:
                source_page = exact_native_index(
                    ref.get("source_page"), positive=True
                )
                if source_page is None:
                    continue

            geometry_scope = ref.get("geometry_scope")
            if geometry_scope not in {"cell", "grid"}:
                continue
            coordinate_system = ref.get("coordinate_system")
            bbox = exact_bbox(ref.get("bbox"))
            table_id = ""
            if (
                geometry_scope == "grid"
                and coordinate_system == "pdf_points_top_left"
                and ref.get("source") == "candidate_b_monthly_grid_omission"
                and ref.get("binding") == "source_account_month_alias"
                and ref.get("binding_quality") == "source_account_month_alias"
                and "page" in ref
                and "logical_page" in ref
                and bbox is not None
                and isinstance(ref.get("table_id"), str)
                and bool(ref["table_id"].strip())
            ):
                table_id = ref["table_id"].strip()
            elif geometry_scope == "cell" and coordinate_system == "pdf_points_top_left":
                provenance = ref.get("geometry_provenance")
                identity = exact_source_table_cell_identity(
                    ref,
                    provenance,
                    page=page,
                    bbox=bbox,
                )
                if identity is not None:
                    source_page = identity[1]
                    table_id = identity[2]
            if account_id and table_id:
                fingerprints.add(
                    (
                        "physical_account_month_source_table_lattice",
                        account_id,
                        performance_month,
                        page,
                        source_page,
                        table_id,
                    )
                )
            raw_evidence_ids = ref.get("evidence_ids")
            evidence_ids: tuple[str, ...] = ()
            if isinstance(raw_evidence_ids, (list, tuple)) and raw_evidence_ids:
                normalized_evidence_ids = tuple(
                    value.strip()
                    for value in raw_evidence_ids
                    if isinstance(value, str) and value.strip()
                )
                if (
                    len(normalized_evidence_ids) == len(raw_evidence_ids)
                    and len(set(normalized_evidence_ids))
                    == len(normalized_evidence_ids)
                ):
                    evidence_ids = tuple(sorted(normalized_evidence_ids))
            for evidence_id in evidence_ids:
                physical_claim = (
                    "physical_account_month_evidence",
                    performance_month,
                    page,
                    source_page,
                    evidence_id,
                )
                fingerprints.add(
                    (
                        "physical_account_month_evidence",
                        account_id,
                        performance_month,
                        page,
                        source_page,
                        evidence_id,
                    )
                )
                physical_claim_refs.setdefault(physical_claim, []).append(dict(ref))
            if bbox is not None:
                physical_claim = (
                    "physical_account_month_geometry",
                    performance_month,
                    page,
                    source_page,
                    geometry_scope,
                    tuple(round(value, 6) for value in bbox),
                )
                fingerprints.add(
                    (
                        "physical_account_month_geometry",
                        account_id,
                        performance_month,
                        page,
                        source_page,
                        geometry_scope,
                        tuple(round(value, 6) for value in bbox),
                    )
                )
                physical_claim_refs.setdefault(physical_claim, []).append(dict(ref))
        return fingerprints, physical_claim_refs

    def unique_physical_position_count(
        positions: set[tuple[str, str]],
        fingerprints_by_position: Mapping[
            tuple[str, str], set[tuple[Any, ...]]
        ],
    ) -> int:
        """Count exact physical cells while retaining every alias observation."""

        parents = {position: position for position in positions}

        def find(position: tuple[str, str]) -> tuple[str, str]:
            parent = parents[position]
            while parent != parents[parent]:
                parent = parents[parent]
            while position != parent:
                next_position = parents[position]
                parents[position] = parent
                position = next_position
            return parent

        def union(left: tuple[str, str], right: tuple[str, str]) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        owners_by_fingerprint: dict[tuple[Any, ...], tuple[str, str]] = {}
        for position in sorted(positions):
            for fingerprint in fingerprints_by_position.get(position, ()):
                prior = owners_by_fingerprint.setdefault(fingerprint, position)
                union(position, prior)
        return len({find(position) for position in positions})

    def has_source_local_month_ref(
        refs: Iterable[Mapping[str, Any]],
        *,
        grid_id: str,
        performance_month: str,
    ) -> bool:
        expected_month = int(performance_month[5:7])
        for ref in refs:
            if str(ref.get("grid_id") or "").strip() != grid_id:
                continue
            ref_month = str(ref.get("performance_month") or "").strip()
            raw_col = ref.get("col")
            if ref_month == performance_month or (
                isinstance(raw_col, int)
                and not isinstance(raw_col, bool)
                and raw_col == expected_month
            ):
                return True
        return False

    def exact_owned_grid_ref(
        ref: Mapping[str, Any],
        *,
        account_id: str,
        performance_month: str,
    ) -> bool:
        raw_row = ref.get("row")
        raw_column = ref.get("column")
        bbox = ref.get("bbox")
        evidence_ids = [
            value
            for value in ref.get("evidence_ids") or ()
            if isinstance(value, str) and value.strip()
        ]
        return bool(
            str(ref.get("source") or "") == owned_grid_ref_source
            and str(ref.get("binding") or "") == "source_account_month_identity"
            and str(ref.get("binding_quality") or "")
            == "source_account_month_identity"
            and str(ref.get("account_id") or "") == account_id
            and str(ref.get("grid_id") or "").strip()
            and str(ref.get("performance_month") or "") == performance_month
            and str(ref.get("field_name") or "") == "performance_month"
            and str(ref.get("geometry_scope") or "") == "cell"
            and isinstance(ref.get("logical_page") or ref.get("page"), int)
            and not isinstance(ref.get("logical_page") or ref.get("page"), bool)
            and int(ref.get("logical_page") or ref.get("page")) > 0
            and isinstance(ref.get("source_page"), int)
            and not isinstance(ref.get("source_page"), bool)
            and int(ref["source_page"]) > 0
            and str(ref.get("table_id") or "").strip()
            and isinstance(raw_row, int)
            and not isinstance(raw_row, bool)
            and raw_row >= 0
            and isinstance(raw_column, int)
            and not isinstance(raw_column, bool)
            and raw_column >= 0
            and isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in bbox
            )
            and float(bbox[2]) > float(bbox[0])
            and float(bbox[3]) > float(bbox[1])
            and evidence_ids
        )

    def canonical_account_month_target(
        account_id: str,
        performance_month: str,
    ) -> str:
        owner_key = stable_record_id(
            "source_account_month_owner", account_id
        ).split(":", 1)[-1]
        return f"source_account_month:{owner_key}:{performance_month}"

    def exact_reason_code_set(
        values: Mapping[str, Any],
        expected: frozenset[str],
    ) -> bool:
        reason_codes = values.get("reason_codes")
        return bool(
            isinstance(reason_codes, (list, tuple))
            and len(reason_codes) == len(expected)
            and all(isinstance(value, str) and value for value in reason_codes)
            and frozenset(reason_codes) == expected
        )

    def exact_detached_cell_ref(
        ref: Mapping[str, Any],
        *,
        account_id: str,
        grid_id: str,
        performance_month: str,
        alias: bool,
    ) -> tuple[
        tuple[Any, ...],
        str,
        int,
        int,
        tuple[float, float, float, float],
    ] | None:
        """Validate one exact detached-grid status/amount cell ref."""

        common_keys = {
            "page",
            "logical_page",
            "geometry_scope",
            "coordinate_system",
            "grid_id",
            "row",
            "col",
            "field_name",
            "source_field_name",
            "performance_month",
            "bbox",
        }
        alias_keys = {
            "account_id",
            "binding",
            "binding_quality",
        }
        expected_keys = common_keys | (alias_keys if alias else set())
        if set(ref) not in {frozenset(expected_keys), frozenset(expected_keys | {"source_page"})}:
            return None

        page = ref.get("page")
        logical_page = ref.get("logical_page")
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page <= 0
            or isinstance(logical_page, bool)
            or not isinstance(logical_page, int)
            or logical_page != page
        ):
            return None
        if "source_page" in ref:
            source_page = ref.get("source_page")
            if (
                isinstance(source_page, bool)
                or not isinstance(source_page, int)
                or source_page <= 0
            ):
                return None
            source_page_state: tuple[str, int | None] = ("present", source_page)
        else:
            source_page_state = ("absent", None)

        raw_row = ref.get("row")
        raw_column = ref.get("col")
        bbox = ref.get("bbox")
        source_field_name = str(ref.get("source_field_name") or "")
        if not (
            str(ref.get("source") or "") == ""
            and str(ref.get("role") or "") == ""
            and str(ref.get("grid_id") or "") == grid_id
            and str(ref.get("performance_month") or "") == performance_month
            and str(ref.get("field_name") or "") == "performance_month"
            and source_field_name in {"status", "overdue_amount"}
            and str(ref.get("geometry_scope") or "") == "cell"
            and str(ref.get("coordinate_system") or "")
            == "pdf_points_top_left"
            and isinstance(raw_row, int)
            and not isinstance(raw_row, bool)
            and raw_row >= 0
            and isinstance(raw_column, int)
            and not isinstance(raw_column, bool)
            and raw_column == int(performance_month[5:7])
            and isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in bbox
            )
            and float(bbox[2]) > float(bbox[0])
            and float(bbox[3]) > float(bbox[1])
        ):
            return None
        if alias:
            if not (
                str(ref.get("binding") or "") == "source_account_month_alias"
                and str(ref.get("binding_quality") or "")
                == "source_account_month_alias"
                and str(ref.get("account_id") or "") == account_id
            ):
                return None
        return (
            (grid_id, performance_month, logical_page, source_page_state),
            source_field_name,
            raw_row,
            raw_column,
            tuple(float(value) for value in bbox),
        )

    def exact_detached_ref_locator(
        refs: list[Mapping[str, Any]],
        *,
        account_id: str,
        grid_id: str,
        performance_month: str,
        alias: bool,
    ) -> tuple[Any, ...] | None:
        """Require the exact two-cell producer grammar for one locator."""

        if len(refs) != 2:
            return None
        by_role: dict[
            str,
            tuple[
                tuple[Any, ...],
                int,
                int,
                tuple[float, float, float, float],
            ],
        ] = {}
        for ref in refs:
            validated = exact_detached_cell_ref(
                ref,
                account_id=account_id,
                grid_id=grid_id,
                performance_month=performance_month,
                alias=alias,
            )
            if validated is None:
                return None
            locator, source_field_name, row, column, bbox = validated
            if source_field_name in by_role:
                return None
            by_role[source_field_name] = (locator, row, column, bbox)
        if set(by_role) != {"status", "overdue_amount"}:
            return None
        status_locator, status_row, status_column, status_bbox = by_role["status"]
        amount_locator, amount_row, amount_column, amount_bbox = by_role[
            "overdue_amount"
        ]
        if not (
            status_locator == amount_locator
            and amount_row == status_row + 1
            and status_column == amount_column
            and status_bbox[0] == amount_bbox[0]
            and status_bbox[2] == amount_bbox[2]
            and status_bbox[1] < amount_bbox[1]
            and status_bbox[3] < amount_bbox[3]
        ):
            return None
        role_fingerprint = tuple(
            sorted(
                (
                    role,
                    row,
                    column,
                    tuple(round(value, 6) for value in bbox),
                )
                for role, (_locator, row, column, bbox) in by_role.items()
            )
        )
        return (*status_locator, role_fingerprint)

    def exact_detached_ref_address(
        issue: Any,
        values: Mapping[str, Any],
        *,
        alias: bool,
    ) -> tuple[
        tuple[str, str],
        tuple[str, str] | None,
        tuple[Any, ...],
    ] | None:
        """Recover an address only from a complete target/ref identity plane."""

        refs = exact_issue_refs(issue, values)
        if refs is None:
            return None
        grid_ids = {
            str(ref.get("grid_id") or "").strip()
            for ref in refs
            if str(ref.get("grid_id") or "").strip()
        }
        performance_months = {
            str(ref.get("performance_month") or "").strip()
            for ref in refs
            if month_pattern.fullmatch(
                str(ref.get("performance_month") or "").strip()
            )
        }
        if len(grid_ids) != 1 or len(performance_months) != 1:
            return None
        grid_id = next(iter(grid_ids))
        performance_month = next(iter(performance_months))
        account_id = ""
        identity: tuple[str, str] | None = None
        if alias:
            account_ids = {
                str(ref.get("account_id") or "").strip()
                for ref in refs
                if str(ref.get("account_id") or "").strip()
            }
            if len(account_ids) != 1:
                return None
            account_id = next(iter(account_ids))
            identity = (account_id, performance_month)
            expected_target = canonical_account_month_target(
                account_id,
                performance_month,
            )
        else:
            expected_target = f"{grid_id}:{performance_month}"
        if str(values.get("target_record_id") or "") != expected_target:
            return None
        locator = exact_detached_ref_locator(
            refs,
            account_id=account_id,
            grid_id=grid_id,
            performance_month=performance_month,
            alias=alias,
        )
        if locator is None:
            return None
        return (grid_id, performance_month), identity, locator

    source_issues = list(datasets.get("personal_detail_extraction_issues") or ())
    detached_alias_target_counts: Counter[str] = Counter()
    detached_alias_sibling_target_field_counts: dict[str, Counter[str]] = {}
    detached_alias_sibling_field_observation_counts: Counter[
        tuple[tuple[str, str], str]
    ] = Counter()
    detached_alias_sibling_counts: Counter[
        tuple[tuple[str, str], tuple[Any, ...]]
    ] = Counter()
    for issue in source_issues:
        values = _record_values(issue)
        observed = values.get("observed_value")
        observed = observed if isinstance(observed, Mapping) else {}
        grid_id = str(observed.get("grid_id") or "").strip()
        performance_month = str(observed.get("performance_month") or "").strip()
        issue_code = str(values.get("issue_code") or "")
        target_record_id = str(values.get("target_record_id") or "")
        field_name = str(values.get("field_name") or "")
        if issue_code == alias_issue_code and target_record_id:
            detached_alias_target_counts[target_record_id] += 1
        if issue_code == detached_alias_sibling_issue_code:
            if target_record_id:
                detached_alias_sibling_target_field_counts.setdefault(
                    target_record_id,
                    Counter(),
                )[field_name] += 1
            addressed_positions: set[tuple[str, str]] = set()
            if grid_id and month_pattern.fullmatch(performance_month):
                addressed_positions.add((grid_id, performance_month))
            recovered_address = exact_detached_ref_address(
                issue,
                values,
                alias=False,
            )
            if recovered_address is not None:
                addressed_positions.add(recovered_address[0])
            for addressed_position in addressed_positions:
                detached_alias_sibling_field_observation_counts[
                    (addressed_position, field_name)
                ] += 1
        if not (
            str(values.get("category") or "") == "ocr_structure_correction"
            and str(values.get("issue_code") or "")
            == detached_alias_sibling_issue_code
            and str(values.get("status") or "") == "requires_review"
            and str(values.get("parser_stage") or "")
            == "canonical_monthly_grid_materialization"
            and str(values.get("target_dataset") or "") == "repayment_records"
            and str(values.get("field_name") or "") == "performance_month"
        ):
            continue
        candidate = values.get("candidate_value")
        source_structure_key_count = observed.get("source_structure_key_count")
        required_observed_keys = {
            "grid_id",
            "performance_month",
            "field_state",
        }
        allowed_observed_keys = required_observed_keys | {
            "source_observations",
            "source_structure_key_count",
        }
        if not (
            grid_id
            and month_pattern.fullmatch(performance_month)
            and required_observed_keys <= set(observed) <= allowed_observed_keys
            and str(observed.get("field_state") or "")
            == "source_position_withheld"
            and (
                "source_observations" not in observed
                or observed.get("source_observations") == [performance_month]
            )
            and (
                "source_structure_key_count" not in observed
                or (
                    isinstance(source_structure_key_count, int)
                    and not isinstance(source_structure_key_count, bool)
                    and source_structure_key_count > 0
                )
            )
            and isinstance(candidate, Mapping)
            and set(candidate) == {"resolution"}
            and str(candidate.get("resolution") or "")
            == "withheld_pending_review"
            and exact_reason_code_set(values, detached_sibling_reason_codes)
        ):
            continue
        if str(values.get("target_record_id") or "") != (
            f"{grid_id}:{performance_month}"
        ):
            continue
        refs = exact_issue_refs(issue, values)
        if refs is None:
            continue
        locator = exact_detached_ref_locator(
            refs,
            account_id="",
            grid_id=grid_id,
            performance_month=performance_month,
            alias=False,
        )
        if locator is None:
            continue
        detached_alias_sibling_counts[((grid_id, performance_month), locator)] += 1

    def exact_detached_alias_claim(
        values: Mapping[str, Any],
        observed: Mapping[str, Any],
        refs: list[Mapping[str, Any]],
        *,
        account_id: str,
        grid_id: str,
        performance_month: str,
    ) -> tuple[tuple[str, str], tuple[str, str]] | None:
        """Return one closed detached-alias claim or fail closed."""

        owner_basis = str(
            observed.get("account_month_owner_basis") or ""
        ).strip()
        candidate = values.get("candidate_value")
        if not (
            str(values.get("category") or "") == "ocr_structure_correction"
            and str(values.get("issue_code") or "") == alias_issue_code
            and str(values.get("severity") or "") == "info"
            and str(values.get("status") or "") in {"informational", "resolved"}
            and str(values.get("parser_stage") or "")
            == "candidate_b_relationship_schema"
            and str(values.get("target_dataset") or "") == "repayment_records"
            and str(values.get("field_name") or "") == "performance_month"
            and str(values.get("target_record_id") or "")
            == canonical_account_month_target(account_id, performance_month)
            and set(observed)
            == {
                "account_id",
                "grid_id",
                "performance_month",
                "source_position_state",
                "account_month_owner_basis",
            }
            and str(observed.get("source_position_state") or "")
            == "owner_bound_alias"
            and owner_basis in allowed_account_month_owner_bases
            and isinstance(candidate, Mapping)
            and set(candidate) == {"resolution"}
            and str(candidate.get("resolution") or "")
            == "reconciled_to_existing_account_month_identity"
            and refs
            and exact_reason_code_set(values, detached_alias_reason_codes)
        ):
            return None
        locator = exact_detached_ref_locator(
            refs,
            account_id=account_id,
            grid_id=grid_id,
            performance_month=performance_month,
            alias=True,
        )
        position = (grid_id, performance_month)
        sibling_target = f"{grid_id}:{performance_month}"
        sibling_field_counts = detached_alias_sibling_target_field_counts.get(
            sibling_target,
            Counter(),
        )
        alias_target = canonical_account_month_target(account_id, performance_month)
        if (
            locator is None
            or detached_alias_target_counts[alias_target] != 1
            or sibling_field_counts["performance_month"] != 1
            or any(
                field not in detached_sibling_target_fields or count != 1
                for field, count in sibling_field_counts.items()
            )
            or detached_alias_sibling_field_observation_counts[
                (position, "performance_month")
            ]
            != 1
            or detached_alias_sibling_counts[(position, locator)] != 1
        ):
            return None
        return position, (account_id, performance_month)

    candidate_identities: set[tuple[str, str]] = set()
    bound_positions_by_identity: dict[
        tuple[str, str], set[tuple[str, str]]
    ] = {}
    fingerprints_by_source_position: dict[
        tuple[str, str], set[tuple[Any, ...]]
    ] = {}
    raw_physical_fingerprints_by_source_position: dict[
        tuple[str, str], set[tuple[Any, ...]]
    ] = {}
    physical_claims_by_source_position: dict[
        tuple[str, str],
        list[
            tuple[
                tuple[Any, ...],
                tuple[str, str],
                list[dict[str, Any]],
            ]
        ],
    ] = {}
    for record in datasets.get("repayment_records") or ():
        values = _record_values(record)
        account_id = str(values.get("account_id") or "").strip()
        performance_month = exact_month(values)
        proof = values.get("_account_month_identity_proof")
        proof_grid_id = (
            str(proof.get("grid_id") or "").strip()
            if isinstance(proof, Mapping)
            else ""
        )
        record_grid_id = str(values.get("grid_id") or "").strip()
        proof_status = str(
            values.get("_account_month_identity_proof_status") or ""
        ).strip()
        if not (
            account_id
            and performance_month
            and proof_grid_id
            and not proof_status.startswith("unproven")
            and isinstance(proof, Mapping)
            and str(proof.get("account_id") or "").strip() == account_id
            and str(proof.get("performance_month") or "").strip()
            == performance_month
            and proof.get("account_anchor_exact") is True
            and proof.get("printed_month_range_exact") is True
            and proof.get("grid_geometry_exact") is True
            and proof.get("unique_owner") is True
            and str(proof.get("owner_basis") or "")
            in {
                "canonical_account_segment",
                "explicit_account_id_confirmed_by_canonical_segment",
                "exact_source_table_account_owner",
            }
            and (not record_grid_id or record_grid_id == proof_grid_id)
        ):
            continue
        identity = (account_id, performance_month)
        candidate_identities.add(identity)
        grid_id = proof_grid_id
        if grid_id:
            source_position = (grid_id, performance_month)
            bound_positions_by_identity.setdefault(identity, set()).add(source_position)
            fingerprints, physical_claim_refs = physical_position_fingerprints(
                record_refs(record, values),
                account_id=account_id,
                performance_month=performance_month,
            )
            fingerprints_by_source_position.setdefault(source_position, set()).update(
                fingerprints
            )
            raw_physical_fingerprints_by_source_position.setdefault(
                source_position, set()
            ).update(physical_claim_refs)
            for physical_claim, claim_refs in physical_claim_refs.items():
                physical_claims_by_source_position.setdefault(source_position, []).append(
                    (physical_claim, identity, claim_refs)
                )

    localized_identities: set[tuple[str, str]] = set()
    raw_unresolved_source_positions: set[tuple[str, str]] = set()
    unlocalized_raw_positions: set[tuple[str, str]] = set()
    unlocalized_bound_identity_issues: set[tuple[str, str]] = set()
    explicit_alias_positions: set[tuple[str, str]] = set()
    detached_alias_observation_counts: Counter[
        tuple[tuple[str, str], tuple[str, str]]
    ] = Counter()
    for issue in source_issues:
        values = _record_values(issue)
        if str(values.get("issue_code") or "") != alias_issue_code:
            continue
        observed = values.get("observed_value")
        observed = observed if isinstance(observed, Mapping) else {}
        account_id = str(observed.get("account_id") or "").strip()
        grid_id = str(observed.get("grid_id") or "").strip()
        performance_month = str(observed.get("performance_month") or "").strip()
        addressed_claims: set[
            tuple[tuple[str, str], tuple[str, str]]
        ] = set()
        if account_id and grid_id and month_pattern.fullmatch(performance_month):
            addressed_claims.add(
                ((grid_id, performance_month), (account_id, performance_month))
            )
        recovered_address = exact_detached_ref_address(
            issue,
            values,
            alias=True,
        )
        if recovered_address is not None and recovered_address[1] is not None:
            addressed_claims.add((recovered_address[0], recovered_address[1]))
        for addressed_claim in addressed_claims:
            detached_alias_observation_counts[addressed_claim] += 1
    detached_alias_claims: Counter[
        tuple[tuple[str, str], tuple[str, str]]
    ] = Counter()
    for issue in source_issues:
        values = _record_values(issue)
        observed = values.get("observed_value")
        observed = observed if isinstance(observed, Mapping) else {}
        account_id = str(observed.get("account_id") or "").strip()
        performance_month = str(observed.get("performance_month") or "").strip()
        grid_id = str(observed.get("grid_id") or "").strip()
        if not month_pattern.fullmatch(performance_month):
            continue
        issue_code = str(values.get("issue_code") or "")
        if str(values.get("target_dataset") or "") not in {
            "repayment_records",
            "credit_account_monthly_performance",
        }:
            continue
        refs = issue_refs(issue, values)
        ref_grid_ids = {
            str(ref.get("grid_id") or "").strip()
            for ref in refs
            if str(ref.get("grid_id") or "").strip()
        }
        if not grid_id and len(ref_grid_ids) == 1:
            grid_id = next(iter(ref_grid_ids))
        source_position = (grid_id, performance_month) if grid_id else None
        source_position_is_local = bool(
            source_position
            and has_source_local_month_ref(
                refs,
                grid_id=grid_id,
                performance_month=performance_month,
            )
        )
        if source_position:
            fingerprints, physical_claim_refs = physical_position_fingerprints(
                refs,
                account_id=account_id,
                performance_month=performance_month,
            )
            fingerprints_by_source_position.setdefault(source_position, set()).update(
                fingerprints
            )
            raw_physical_fingerprints_by_source_position.setdefault(
                source_position, set()
            ).update(physical_claim_refs)
            if account_id:
                identity = (account_id, performance_month)
                for physical_claim, claim_refs in physical_claim_refs.items():
                    physical_claims_by_source_position.setdefault(
                        source_position, []
                    ).append((physical_claim, identity, claim_refs))
        if issue_code == alias_issue_code:
            if source_position_is_local and account_id:
                explicit_alias_positions.add(source_position)
                identity = (account_id, performance_month)
                bound_positions_by_identity.setdefault(identity, set()).add(
                    source_position
                )
                exact_refs = exact_issue_refs(issue, values)
                detached_alias_claim = (
                    exact_detached_alias_claim(
                        values,
                        observed,
                        exact_refs,
                        account_id=account_id,
                        grid_id=grid_id,
                        performance_month=performance_month,
                    )
                    if exact_refs is not None
                    else None
                )
                if detached_alias_claim is not None:
                    claimed_position, claimed_identity = detached_alias_claim
                    detached_alias_claims[(claimed_position, claimed_identity)] += 1
            elif source_position:
                # An alias without an exact account owner is still a raw
                # printed position, but it cannot be reconciled as bound.
                raw_unresolved_source_positions.add(source_position)
                if not source_position_is_local:
                    unlocalized_raw_positions.add(source_position)
            continue
        target_record_id = str(values.get("target_record_id") or "").strip()
        field_name = str(values.get("field_name") or "").strip()
        source_identity_type = str(
            observed.get("source_identity_type") or ""
        ).strip()
        owner_key = (
            stable_record_id("source_account_month_owner", account_id).split(":", 1)[-1]
            if account_id
            else ""
        )
        expected_target_id = (
            f"source_account_month:{owner_key}:{performance_month}"
            if owner_key
            else ""
        )
        exact_owned_grid_identity = bool(
            issue_code == owned_grid_issue_code
            and field_name == "performance_month"
            and set(observed) == {"account_id", "performance_month"}
            and target_record_id == expected_target_id
            and len(refs) == 1
            and exact_owned_grid_ref(
                refs[0],
                account_id=account_id,
                performance_month=performance_month,
            )
        )
        exact_range_identity = bool(
            issue_code == "candidate_b_monthly_account_range_missing_month"
            and field_name == "performance_month"
            and source_identity_type == "account_month_from_printed_repayment_range"
            and target_record_id == expected_target_id
            and len(refs) == 1
            and str(refs[0].get("account_id") or "") == account_id
            and str(refs[0].get("performance_month") or "") == performance_month
            and str(refs[0].get("binding") or "") == "source_account_month_range"
        )
        exact_localized_identity = exact_owned_grid_identity or exact_range_identity
        if issue_code == owned_grid_issue_code and not exact_owned_grid_identity:
            if source_position:
                raw_unresolved_source_positions.add(source_position)
                if not source_position_is_local:
                    unlocalized_raw_positions.add(source_position)
            continue
        if account_id and exact_localized_identity:
            identity = (account_id, performance_month)
            if refs:
                localized_identities.add(identity)
                if source_position_is_local:
                    bound_positions_by_identity.setdefault(identity, set()).add(
                        source_position
                    )
            else:
                # Do not promote an unlocalized diagnostic into the canonical
                # denominator.  Keep it visible in the audit status instead.
                unlocalized_bound_identity_issues.add(identity)
                if source_position:
                    raw_unresolved_source_positions.add(source_position)
                    unlocalized_raw_positions.add(source_position)
            continue
        if account_id:
            # An issue merely mentioning an emitted account/month (for
            # example, a duplicate-value conflict) is not a missing identity
            # and must not change either side of the closure equation.
            continue
        if source_position:
            raw_unresolved_source_positions.add(source_position)
            if not source_position_is_local:
                unlocalized_raw_positions.add(source_position)

    closure = candidate_identities | localized_identities
    all_bound_source_positions = {
        position
        for positions in bound_positions_by_identity.values()
        for position in positions
    }
    identities_by_bound_position: dict[
        tuple[str, str], set[tuple[str, str]]
    ] = {}
    for identity, positions in bound_positions_by_identity.items():
        for position in positions:
            identities_by_bound_position.setdefault(position, set()).add(identity)
    detached_audit_only_alias_positions: set[tuple[str, str]] = set()
    for (position, identity), claim_count in detached_alias_claims.items():
        if (
            claim_count != 1
            or detached_alias_observation_counts[(position, identity)] != 1
        ):
            continue
        if identities_by_bound_position.get(position) != {identity}:
            # Never hide a source position claimed by more than one owner.
            continue
        non_alias_positions = (
            bound_positions_by_identity.get(identity, set())
            - explicit_alias_positions
        )
        if non_alias_positions:
            detached_audit_only_alias_positions.add(position)
    claims_by_physical_fingerprint: dict[
        tuple[Any, ...],
        list[tuple[tuple[str, str], tuple[str, str], list[dict[str, Any]]]],
    ] = {}
    for source_position in all_bound_source_positions:
        for physical_claim, identity, claim_refs in physical_claims_by_source_position.get(
            source_position, ()
        ):
            claims_by_physical_fingerprint.setdefault(physical_claim, []).append(
                (identity, source_position, claim_refs)
            )
    physical_owner_conflict_groups: dict[
        tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
        dict[str, Any],
    ] = {}
    for physical_fingerprint, claims in claims_by_physical_fingerprint.items():
        identities = tuple(sorted({identity for identity, _position, _refs in claims}))
        if len(identities) <= 1:
            continue
        source_positions = tuple(
            sorted({position for _identity, position, _refs in claims})
        )
        conflict = physical_owner_conflict_groups.setdefault(
            (identities, source_positions),
            {"physical_fingerprints": set(), "source_refs": []},
        )
        conflict["physical_fingerprints"].add(physical_fingerprint)
        conflict["source_refs"].extend(
            ref
            for _identity, _position, refs in claims
            for ref in refs
            if isinstance(ref, Mapping)
        )
    alias_source_positions = set(explicit_alias_positions)
    for source_positions in bound_positions_by_identity.values():
        primary_candidates = sorted(source_positions - explicit_alias_positions)
        if len(primary_candidates) > 1:
            alias_source_positions.update(primary_candidates[1:])
    inventoried_physical_fingerprints = {
        fingerprint
        for position in all_bound_source_positions | alias_source_positions
        for fingerprint in raw_physical_fingerprints_by_source_position.get(position, ())
    }
    reconciled_detached_diagnostic_positions = {
        position
        for position in raw_unresolved_source_positions
        if position not in all_bound_source_positions
        and position not in alias_source_positions
        and raw_physical_fingerprints_by_source_position.get(position, set())
        & inventoried_physical_fingerprints
    }
    unresolved_source_positions = (
        raw_unresolved_source_positions
        - all_bound_source_positions
        - alias_source_positions
        - reconciled_detached_diagnostic_positions
    )
    unlocalized_unresolved_positions = (
        unlocalized_raw_positions & unresolved_source_positions
    )
    if not (
        closure
        or unresolved_source_positions
        or alias_source_positions
        or unlocalized_bound_identity_issues
    ):
        return
    digest_input = "".join(
        f"{account_id}\t{performance_month}\n"
        for account_id, performance_month in sorted(closure)
    ).encode("utf-8")
    identity_sha256 = hashlib.sha256(digest_input).hexdigest()
    # Source observations and canonical identities are deliberately separate
    # ledgers.  Multiple parser/grid aliases can observe one exact physical
    # cell; raw and owner-bound counts conserve that cell once while alias
    # observations remain independently auditable.
    source_month_position_observations = len(
        all_bound_source_positions | raw_unresolved_source_positions
    )
    owner_bound_account_months = unique_physical_position_count(
        all_bound_source_positions - detached_audit_only_alias_positions,
        fingerprints_by_source_position,
    )
    owner_unresolved_positions = unique_physical_position_count(
        unresolved_source_positions,
        fingerprints_by_source_position,
    )
    raw_source_month_positions = (
        owner_bound_account_months + owner_unresolved_positions
    )
    physical_alias_source_month_observations = (
        source_month_position_observations - raw_source_month_positions
    )
    balance_valid = raw_source_month_positions == (
        owner_bound_account_months + owner_unresolved_positions
    )
    if physical_owner_conflict_groups:
        ledger_status = "physical_owner_conflict"
    elif unlocalized_bound_identity_issues or unlocalized_unresolved_positions:
        ledger_status = "source_localization_invalid"
    elif unresolved_source_positions:
        ledger_status = "partial_owner_unresolved"
    else:
        ledger_status = "identity_closed"
    facts["personal_detail_account_month_closure"] = {
        "schema": "docmirror.pboc.account_month_closure.v2",
        "identity_fields": ["account_id", "performance_month"],
        "candidate_identity_count": len(candidate_identities),
        "localized_missing_identity_count": len(
            localized_identities - candidate_identities
        ),
        "expected_identity_count": len(closure),
        "canonical_account_month_identity_count": len(closure),
        "source_month_position_observations": source_month_position_observations,
        "raw_source_month_positions": raw_source_month_positions,
        "unique_physical_source_month_positions": raw_source_month_positions,
        "physical_alias_source_month_observations": (
            physical_alias_source_month_observations
        ),
        "cross_owner_physical_conflict_count": len(
            physical_owner_conflict_groups
        ),
        "physical_owner_conflict_free": not physical_owner_conflict_groups,
        "owner_bound_account_months": owner_bound_account_months,
        "owner_unresolved_positions": owner_unresolved_positions,
        "owner_unresolved_position_observations": len(
            unresolved_source_positions
        ),
        "alias_source_month_positions": len(alias_source_positions),
        "reconciled_detached_diagnostic_positions": len(
            reconciled_detached_diagnostic_positions
        ),
        "source_localized_owner_unresolved_positions": (
            len(unresolved_source_positions) - len(unlocalized_unresolved_positions)
        ),
        "unlocalized_owner_unresolved_positions": len(
            unlocalized_unresolved_positions
        ),
        "unlocalized_owner_bound_identity_issues": len(
            unlocalized_bound_identity_issues
        ),
        "source_position_balance": (
            "raw_source_month_positions = owner_bound_account_months + "
            "owner_unresolved_positions"
        ),
        "source_position_balance_valid": balance_valid,
        "unresolved_source_position_count": len(unresolved_source_positions),
        "identity_sha256": identity_sha256,
        "status": ledger_status,
    }
    if not physical_owner_conflict_groups:
        datasets[closure_proof_dataset] = [
            {
                "record_id": "personal_detail_account_month_closure_proof:1",
                "schema": "docmirror.pboc.account_month_closure_proof.v1",
                "identity_fields": ["account_id", "performance_month"],
                "proof_basis": "exact_source_account_month_identity_set",
                "expected_identity_count": len(closure),
                "identity_sha256": identity_sha256,
            }
        ]
    else:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            make_issue,
        )

        issues = datasets.setdefault("personal_detail_extraction_issues", [])
        if not isinstance(issues, list):
            issues = []
            datasets["personal_detail_extraction_issues"] = issues
        existing_issue_ids = {
            str(issue.get("extraction_issue_id") or "")
            for issue in issues
            if isinstance(issue, Mapping)
        }
        for (identities, source_positions), conflict in sorted(
            physical_owner_conflict_groups.items()
        ):
            unique_refs: list[dict[str, Any]] = []
            seen_refs: set[str] = set()
            for ref in conflict["source_refs"]:
                marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen_refs:
                    seen_refs.add(marker)
                    unique_refs.append(dict(ref))
            target_record_id = stable_record_id(
                "source_account_month_physical_owner_conflict",
                identities,
                source_positions,
            )
            issue = make_issue(
                category="schema_incompleteness",
                issue_code="account_month_physical_owner_conflict",
                message=(
                    "One exact monthly source cell was claimed by multiple canonical "
                    "account-month owners; the closure proof was withheld."
                ),
                parser_stage="account_month_closure_ledger",
                target_dataset="repayment_records",
                target_record_id=target_record_id,
                field_name="account_id",
                observed_value={
                    "claimed_account_month_identities": [
                        {
                            "account_id": account_id,
                            "performance_month": performance_month,
                        }
                        for account_id, performance_month in identities
                    ],
                    "source_positions": [
                        {"grid_id": grid_id, "performance_month": performance_month}
                        for grid_id, performance_month in source_positions
                    ],
                    "shared_physical_fingerprint_count": len(
                        conflict["physical_fingerprints"]
                    ),
                },
                source_refs=unique_refs,
                reason_codes=(
                    "exact_physical_source_cell",
                    "multiple_canonical_account_month_owners",
                    "cross_owner_alias_rejected",
                    "closure_proof_withheld",
                ),
            )
            if issue["extraction_issue_id"] not in existing_issue_ids:
                issues.append(issue)
                existing_issue_ids.add(issue["extraction_issue_id"])
    states = facts.setdefault("personal_detail_dataset_states", {})
    if not isinstance(states, dict):
        states = {}
        facts["personal_detail_dataset_states"] = states
    states["repayment_records"] = {
        "presence_status": (
            "partial"
            if localized_identities - candidate_identities
            or unresolved_source_positions
            or unlocalized_bound_identity_issues
            or unlocalized_unresolved_positions
            or physical_owner_conflict_groups
            else "present"
        ),
        "reason": (
            "account_month_physical_owner_conflict"
            if physical_owner_conflict_groups
            else (
                "account_month_source_localization_invalid"
                if unlocalized_bound_identity_issues or unlocalized_unresolved_positions
                else (
                    "account_month_identity_partial_owner_unresolved"
                    if unresolved_source_positions
                    else "account_month_identity_closure"
                )
            )
        ),
        "observed_row_count": len(candidate_identities),
        **({"expected_row_count": len(closure)} if closure else {}),
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
    _apply_account_month_closure_ledger(facts, datasets)
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

    from docmirror.plugins.credit_report.personal_detail_scanned.fail_closed_field_reporting import (
        append_fail_closed_field_issues,
    )

    # Profile aliases and typed summary identities do not exist until their
    # canonical contracts have been materialized.  Conserving source-bound
    # failures here also keeps them visible to the field-observation and
    # Community projections.
    append_fail_closed_field_issues(facts, datasets)
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
