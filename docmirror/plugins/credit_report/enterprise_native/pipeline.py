# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-pass canonical pipeline for native enterprise credit reports.

The page-aware parser output is first copied into a pagination-independent
connected-component IR.  Every enterprise business field is then decoded once
from that IR and projected through the one canonical enterprise schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from docmirror.plugins.credit_report.enterprise_native.extraction import (
    _reported_account_summary,
    _reported_credit_line_count,
    extract_enterprise_accounts_from_tables,
    extract_enterprise_attachment_datasets,
    extract_enterprise_capital_summary,
    extract_enterprise_continuation_audit,
    extract_enterprise_credit_lines_from_tables,
    extract_enterprise_facility_summary,
    extract_enterprise_identity_facts,
    extract_enterprise_interest_arrears,
    extract_enterprise_non_credit_history_datasets,
    extract_enterprise_overview,
    extract_enterprise_overview_datasets,
    extract_enterprise_profile_datasets,
    extract_enterprise_public_records_from_tables,
    extract_enterprise_repayment_liability_records,
    extract_enterprise_report_identity_records,
    extract_enterprise_report_metadata_records,
    extract_enterprise_report_notes,
    extract_enterprise_summary_datasets,
    project_enterprise_public_record_datasets,
)
from docmirror.plugins.credit_report.enterprise_native.extraction_validation import (
    build_enterprise_extraction_report,
)
from docmirror.plugins.credit_report.enterprise_native.ir import (
    CanonicalEnterpriseDocumentIR,
    build_canonical_enterprise_document,
)
from docmirror.plugins.credit_report.enterprise_native.source_quality import (
    assess_enterprise_source_information,
)
from docmirror.plugins.credit_report.value_utils import parse_number

_META_FIELDS = frozenset(
    {
        "normalized",
        "source_refs",
        "source_cell_refs",
        "source",
        "confidence",
        "extraction_status",
        "audit",
        "bbox",
        "page",
        "record_id",
    }
)
_ACCOUNT_FIELDS = (
    "account_id",
    "sequence",
    "account_type",
    "business_category",
    "business_type",
    "account_identifier",
    "repayment_method",
    "issuance_form",
    "guarantee_type",
    "counter_guarantee_type",
    "special_transaction",
    "credit_agreement_identifier",
    "history_status",
    "open_date",
    "close_date",
    "currency",
    "amount_unit",
    "credit_limit",
    "loan_amount",
    "discount_amount",
    "guarantee_amount",
    "balance",
    "risk_exposure_amount",
    "deposit_ratio",
    "snapshot_date",
    "remaining_periods",
    "scheduled_payment",
    "actual_payment",
    "scheduled_payment_date",
    "last_repayment_date",
    "current_overdue_periods",
    "current_overdue_amount",
    "overdue_principal",
    "overdue_total",
    "five_tier_class",
    "current_overdue",
    "current_overdue_status",
    "overdue_months",
    "original_creditor_name",
    "original_claim_type",
    "receive_date",
)
_LINE_FIELDS = (
    "credit_line_id",
    "account_id",
    "account_identifier",
    "facility_type",
    "facility_product",
    "revolving_flag",
    "effective_date",
    "snapshot_date",
    "total_limit",
    "used_limit",
    "available_limit",
    "facility_limit",
    "limit_identifier",
    "currency",
    "amount_unit",
)
_LIABILITY_FIELDS = (
    "liability_id",
    "sequence",
    "account_identifier",
    "liability_date",
    "open_date",
    "related_party_name",
    "related_party_id_type",
    "related_party_id_number",
    "business_type",
    "responsibility_type",
    "responsibility_amount",
    "responsibility_amount_reported",
    "responsibility_amount_status",
    "contract_number",
    "contract_number_status",
    "loan_or_credit_amount",
    "snapshot_date",
    "balance",
    "five_tier_class",
    "overdue_total",
    "overdue_principal",
    "overdue_months_or_repayment_status",
    "remaining_periods",
    "continuation_complete",
    "currency",
    "amount_unit",
)
_NUMBER_FIELDS = frozenset(
    {
        "sequence",
        "credit_limit",
        "loan_amount",
        "discount_amount",
        "guarantee_amount",
        "balance",
        "risk_exposure_amount",
        "responsibility_amount",
        "loan_or_credit_amount",
        "overdue_total",
        "overdue_principal",
        "total_limit",
        "used_limit",
        "available_limit",
        "facility_limit",
        "remaining_periods",
        "scheduled_payment",
        "actual_payment",
        "current_overdue_periods",
        "current_overdue_amount",
        "overdue_months",
    }
)
_MISSING_MARKERS = frozenset({"", "-", "--", "—", "－", "/", "不详", "未知"})
_SECTION_SPECS = (
    ("sec_enterprise_report_metadata", "报告信息", "report_metadata", ("企业信用报告（自主查询版）", "企业信用报告(自主查询版)")),
    ("sec_enterprise_report_notes", "说明", "notes", ("报告说明",)),
    ("sec_enterprise_identity", "身份标识", "identity", ("身份标识",)),
    ("sec_enterprise_summary", "信息概要", "credit_summary", ("信息概要",)),
    ("sec_enterprise_profile", "基本信息", "basic_information", ("基本信息",)),
    ("sec_enterprise_credit", "信贷记录明细", "credit_details", ("信贷记录明细",)),
    ("sec_enterprise_public", "公共记录明细", "public_records", ("公共记录明细",)),
    (
        "sec_enterprise_supplement",
        "信用记录补充信息",
        "credit_supplement",
        ("信用记录补充信息", "附件1：信用记录补充信息", "附件1:信用记录补充信息"),
    ),
)


@dataclass(frozen=True)
class EnterpriseSemanticDocument:
    facts: dict[str, Any]
    datasets: dict[str, list[dict[str, Any]]]
    sections: tuple[dict[str, Any], ...]
    credit_summary: dict[str, Any]
    credit_extraction_audit: dict[str, Any]
    continuation_audit: tuple[dict[str, Any], ...]
    quality_flags: tuple[dict[str, Any], ...]
    dataset_completeness: dict[str, dict[str, Any]]
    extraction_report: dict[str, Any]

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "schema": {"id": "enterprise_credit_report", "version": "2.0.0"},
            "facts": self.facts,
            "credit_summary": self.credit_summary,
            "credit_extraction_audit": self.credit_extraction_audit,
            "continuation_audit": list(self.continuation_audit),
            "quality_flags": list(self.quality_flags),
            "dataset_completeness": self.dataset_completeness,
            "extraction": self.extraction_report,
            "datasets": self.datasets,
            "sections": list(self.sections),
        }

    def to_debug_json(self) -> str:
        return json.dumps(self.to_debug_payload(), ensure_ascii=False, sort_keys=True, default=str)


@dataclass(frozen=True)
class EnterprisePipelineArtifacts:
    document_ir: CanonicalEnterpriseDocumentIR
    semantic_document: EnterpriseSemanticDocument
    ir_debug_json: str
    semantic_debug_json: str


def _value(record: dict[str, Any], field: str) -> Any:
    aliases = {
        "institution": ("management_institution", "institution"),
        "status": ("account_status", "status"),
    }.get(field, (field,))
    return next(
        (record.get(alias) for alias in aliases if record.get(alias) not in (None, "")),
        None,
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped in _MISSING_MARKERS else stripped
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _cohere_field_statuses(normalized: dict[str, Any]) -> None:
    for field, value in tuple(normalized.items()):
        if field.endswith("_status"):
            continue
        status_key = f"{field}_status"
        explicit = str(normalized.get(status_key) or "")
        if value is None:
            status = explicit if explicit and explicit != "reported" else "not_reported"
        elif explicit in {"not_reported", "unresolved"}:
            status = "derived"
            normalized[status_key] = status
        else:
            status = explicit or "reported"
        if value is not None and not explicit:
            continue
        normalized[status_key] = status


def _merge_explicit_statuses(source: dict[str, Any], normalized: dict[str, Any]) -> None:
    for key, value in source.items():
        if not key.endswith("_status"):
            continue
        target = "maturity_date_status" if key == "due_date_status" else key
        if target in normalized:
            normalized[target] = _canonical_value(value)


def _normalized_value(field: str, value: Any) -> Any:
    value = _canonical_value(value)
    if field not in _NUMBER_FIELDS:
        return value
    number = parse_number(value)
    if number is None:
        return None
    if field in {"sequence", "remaining_periods", "current_overdue_periods", "overdue_months"}:
        return int(number)
    return number


def _base_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out.setdefault("confidence", 1.0 if out.get("source_refs") else 0.8)
    return out


def _normalize_account(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized = {field: _normalized_value(field, _value(out, field)) for field in _ACCOUNT_FIELDS}
    institution = _value(out, "institution")
    if institution:
        normalized["institution"] = institution
    status = str(_value(out, "status") or "").lower()
    if status:
        normalized["status"] = status
    normalized["maturity_date"] = _canonical_value(_value(out, "due_date"))
    normalized["account_state"] = (
        "open" if status in {"active", "inactive"} else "closed" if status in {"closed", "settled"} else "unknown"
    )
    normalized["payoff_state"] = (
        "settled" if status in {"closed", "settled"} else "outstanding" if status == "active" else "unknown"
    )
    normalized["activation_state"] = "not_applicable"
    normalized.setdefault("account_type", str(normalized.get("account_type") or "enterprise_credit"))
    for field in ("credit_limit", "loan_amount", "balance"):
        normalized[f"{field}_status"] = "reported" if normalized.get(field) not in (None, "") else "not_reported"
    _merge_explicit_statuses(out, normalized)
    _cohere_field_statuses(normalized)
    out["normalized"] = normalized
    return out


def _normalize_line(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized = {field: _normalized_value(field, _value(out, field)) for field in _LINE_FIELDS}
    institution = _value(out, "institution")
    if institution:
        normalized["institution"] = institution
    status = str(_value(out, "status") or "").lower()
    if status:
        normalized["status"] = status
    normalized["maturity_date"] = _canonical_value(_value(out, "due_date"))
    normalized["account_state"] = (
        "open" if status in {"active", "inactive"} else "closed" if status in {"closed", "settled"} else "unknown"
    )
    normalized["payoff_state"] = (
        "settled" if status in {"closed", "settled"} else "outstanding" if status == "active" else "unknown"
    )
    for field in ("total_limit", "used_limit", "available_limit"):
        normalized[f"{field}_status"] = "reported" if normalized.get(field) not in (None, "") else "not_reported"
    _merge_explicit_statuses(out, normalized)
    _cohere_field_statuses(normalized)
    out["normalized"] = normalized
    return out


def _normalize_liability(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized = {field: _normalized_value(field, _value(out, field)) for field in _LIABILITY_FIELDS}
    institution = _value(out, "institution")
    if institution:
        normalized["institution"] = institution
    normalized["maturity_date"] = _canonical_value(_value(out, "due_date"))
    _merge_explicit_statuses(out, normalized)
    _cohere_field_statuses(normalized)
    out["normalized"] = normalized
    return out


def _normalize_passthrough(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized: dict[str, Any] = {}
    aliases = {"due_date": "maturity_date", "due_date_status": "maturity_date_status"}
    for key, value in out.items():
        if key in _META_FIELDS:
            continue
        target = aliases.get(key, key)
        canonical = _canonical_value(value)
        if target in normalized and normalized[target] is not None:
            continue
        normalized[target] = canonical
    _cohere_field_statuses(normalized)
    out["normalized"] = normalized
    return out


def _identity(collection: str, record: dict[str, Any], index: int) -> str:
    keys = {
        "enterprise_credit_accounts": ("account_id", "account_identifier"),
        "enterprise_credit_facilities": ("credit_line_id", "account_identifier"),
        "enterprise_repayment_responsibility_accounts": ("liability_id", "account_identifier"),
    }[collection]
    normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
    return str(
        next(
            (record.get(key) or normalized.get(key) for key in keys if record.get(key) or normalized.get(key)),
            f"{collection}:r{index:06d}",
        )
    )


def _normalize_collection(collection: str, records: list[Any]) -> list[dict[str, Any]]:
    normalizer = {
        "enterprise_credit_accounts": _normalize_account,
        "enterprise_credit_facilities": _normalize_line,
        "enterprise_repayment_responsibility_accounts": _normalize_liability,
    }.get(collection, _normalize_passthrough)
    ordered: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for source_index, candidate in enumerate(records, start=1):
        if not isinstance(candidate, dict):
            continue
        normalized = normalizer(candidate)
        identity = _identity(collection, normalized, source_index)
        if identity in positions:
            ordered[positions[identity]] = normalized
        else:
            positions[identity] = len(ordered)
            ordered.append(normalized)
    return ordered


def _normalize_all_datasets(
    datasets: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    canonical = {
        "enterprise_credit_accounts",
        "enterprise_credit_facilities",
        "enterprise_repayment_responsibility_accounts",
    }
    normalized: dict[str, list[dict[str, Any]]] = {}
    for name, rows in datasets.items():
        if not rows:
            continue
        if name in canonical:
            normalized[name] = _normalize_collection(name, rows)
        else:
            normalized[name] = [_normalize_passthrough(row) for row in rows if isinstance(row, dict)]
    return {name: rows for name, rows in normalized.items() if rows}


def _heading_lines(document: CanonicalEnterpriseDocumentIR) -> list[tuple[int, int, str]]:
    lines: list[tuple[int, int, str]] = []
    order = 0
    for component in sorted(document.components, key=lambda item: item.global_order):
        for segment in component.segments:
            if str(segment.get("kind") or "") == "table":
                continue
            page = int(segment.get("source_page") or 0)
            for raw_line in str(segment.get("text") or "").splitlines():
                line = "".join(raw_line.split()).strip()
                if not line:
                    continue
                order += 1
                lines.append((order, page, line))
    return lines


def _is_heading(line: str, aliases: tuple[str, ...]) -> bool:
    candidate = re.sub(
        r"^(?:(?:附件\d+[:：])|(?:[一二三四五六七八九十]+[、.．])|(?:[（(][一二三四五六七八九十0-9]+[）)]))",
        "",
        line,
    )
    return line in aliases or candidate in aliases


def _sections(document: CanonicalEnterpriseDocumentIR) -> tuple[dict[str, Any], ...]:
    headings = _heading_lines(document)
    anchored: list[tuple[int, int, str, str, str]] = []
    for section_id, title, section_type, markers in _SECTION_SPECS:
        if section_type == "report_metadata":
            match = next(((order, page) for order, page, line in headings if _is_heading(line, markers)), None)
            if match is None:
                match = next(
                    (
                        (order, page)
                        for index, (order, page, line) in enumerate(headings[:-1])
                        if line == "企业信用报告" and headings[index + 1][2] == "（自主查询版）"
                    ),
                    None,
                )
        else:
            match = next(((order, page) for order, page, line in headings if _is_heading(line, markers)), None)
        if match is not None:
            anchored.append((match[0], match[1], section_id, title, section_type))
    anchored.sort(key=lambda item: item[0])
    sections: list[dict[str, Any]] = []
    last_page = max((page.page_number for page in document.pages), default=document.source_page_count)
    for index, (_order, source_page, section_id, title, section_type) in enumerate(anchored):
        next_page = anchored[index + 1][1] if index + 1 < len(anchored) else last_page
        source_page_end = max(source_page, next_page if next_page == source_page else next_page - 1)
        sections.append(
            {
                "id": section_id,
                "title": title,
                "name": title,
                "type": section_type,
                "page_start": source_page,
                "source_page_start": source_page,
                "source_page_end": source_page_end,
            }
        )
    return tuple(sections)


def _dataset_completeness(
    datasets: dict[str, list[dict[str, Any]]],
    continuation_audit: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    quality_flags: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reconcile like-for-like canonical source records with public rows."""
    audit_dataset = {
        "current_credit_summary": "enterprise_current_credit_summary",
        "closed_credit_summary": "enterprise_closed_credit_summary",
        "repayment_responsibility_summary": "enterprise_repayment_responsibility_summary",
        "repayment_liability": "enterprise_repayment_responsibility_accounts",
        "attachment_account": "enterprise_attachment_accounts",
        "attachment_credit_detail": "enterprise_attachment_credit_details",
    }
    audited: dict[str, dict[str, int]] = {}
    for row in continuation_audit:
        dataset = audit_dataset.get(str(row.get("continuation_family") or ""))
        if not dataset:
            continue
        current = audited.setdefault(dataset, {"expected": 0, "extracted": 0, "unresolved": 0, "unexpected": 0})
        for source_key, target_key in (
            ("expected_record_count", "expected"),
            ("extracted_record_count", "extracted"),
            ("unresolved_record_count", "unresolved"),
            ("unexpected_record_count", "unexpected"),
        ):
            current[target_key] += int(row.get(source_key) or 0)
    bad_input = any(
        flag.get("category") == "parseresult_input" and flag.get("severity") == "error"
        for flag in quality_flags
    )
    output: dict[str, dict[str, Any]] = {}
    for dataset, rows in datasets.items():
        if dataset in audited and audited[dataset]["extracted"] == len(rows):
            source = audited[dataset]
            expected = source["expected"]
            verified = not bad_input and not source["unresolved"] and not source["unexpected"]
            basis = "canonical_continuation_contract"
        else:
            expected = len(rows)
            verified = not bad_input and all(row.get("source_refs") or row.get("source_page") for row in rows)
            basis = "canonical_source_component_count"
        output[dataset] = {
            "expected_row_count": expected,
            "emitted_row_count": len(rows),
            "omitted_row_count": max(0, expected - len(rows)),
            "verified": bool(verified and expected == len(rows)),
            "basis": basis,
            "status": (
                "bad_parseresult_input"
                if bad_input
                else "complete"
                if verified and expected == len(rows)
                else "incomplete"
            ),
        }
    return output


def _apply_quality_statuses(
    datasets: dict[str, list[dict[str, Any]]],
    quality_flags: tuple[dict[str, Any], ...],
) -> None:
    for flag in quality_flags:
        dataset = str(flag.get("dataset") or "")
        field = str(flag.get("field") or "")
        status = str(flag.get("status") or "")
        if status not in {"estimated", "source_truncated"}:
            continue
        targets = datasets.get(dataset) or []
        if not targets and field == "available_limit":
            targets = [
                *(datasets.get("enterprise_credit_facilities") or []),
                *(datasets.get("enterprise_facility_summary") or []),
            ]
        for row in targets:
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            if not field or field not in normalized:
                continue
            normalized[f"{field}_status"] = status


def extract_enterprise_semantic_document(
    document: CanonicalEnterpriseDocumentIR,
    *,
    content_mode: str = "native_text",
) -> EnterpriseSemanticDocument:
    """Apply the complete enterprise schema exactly once to the canonical IR."""
    if not isinstance(document, CanonicalEnterpriseDocumentIR):
        raise TypeError("enterprise schema extraction requires CanonicalEnterpriseDocumentIR")

    attachment_datasets = extract_enterprise_attachment_datasets(document)
    facility_summary = extract_enterprise_facility_summary(document)
    overview = extract_enterprise_overview(document)
    reported = _reported_account_summary(document)
    accounts = extract_enterprise_accounts_from_tables(document)
    credit_lines = extract_enterprise_credit_lines_from_tables(document, accounts)
    liabilities = extract_enterprise_repayment_liability_records(document)
    public_records = extract_enterprise_public_records_from_tables(document)

    summary = {**overview, **reported}
    summary["reported_account_count_basis"] = "information_summary_current_account_population"
    summary["displayed_credit_account_card_count"] = len(accounts)
    reported_credit_lines = _reported_credit_line_count(document)
    if reported_credit_lines is not None:
        summary["reported_credit_line_count"] = reported_credit_lines
    if public_records:
        summary["public_record_count"] = len(public_records)
        public_counts: dict[str, int] = {}
        for record in public_records:
            record_type = str(record.get("record_type") or "unknown")
            public_counts[record_type] = public_counts.get(record_type, 0) + 1
        summary["public_record_type_counts"] = public_counts
    summary.update(
        {
            "account_dataset_scope": "main_report_account_cards",
            "account_dataset_scope_note": "信贷账户数据集对应报告正文展示的账户卡片；附件业务字段在独立规范数据集中列示。",
            "attachment_account_count": len(attachment_datasets["enterprise_attachment_accounts"]),
            "attachment_credit_detail_count": len(attachment_datasets["enterprise_attachment_credit_details"]),
            "attachment_special_transaction_count": len(attachment_datasets["enterprise_special_transactions"]),
            "displayed_credit_facility_count": len(credit_lines),
        }
    )

    datasets: dict[str, list[dict[str, Any]]] = {
        "enterprise_credit_accounts": accounts,
        "enterprise_credit_facilities": credit_lines,
        "enterprise_repayment_responsibility_accounts": liabilities,
    }
    datasets.update(extract_enterprise_profile_datasets(document))
    datasets.update(project_enterprise_public_record_datasets(public_records))
    datasets.update(extract_enterprise_report_metadata_records(document, document.full_text))
    datasets["enterprise_report_identity"] = extract_enterprise_report_identity_records(document, document.full_text)
    datasets.update(extract_enterprise_overview_datasets(document))
    datasets.update(extract_enterprise_summary_datasets(document))
    datasets["enterprise_interest_arrears"] = extract_enterprise_interest_arrears(document)
    datasets.update(attachment_datasets)
    datasets.update(extract_enterprise_non_credit_history_datasets(document))
    datasets["enterprise_capital_summary"] = extract_enterprise_capital_summary(document)
    datasets["enterprise_facility_summary"] = facility_summary
    datasets["report_notes"] = extract_enterprise_report_notes(document)

    continuation_audit = extract_enterprise_continuation_audit(document, datasets=datasets)
    datasets = _normalize_all_datasets(datasets)
    source_flags = tuple(flag.to_payload() for flag in assess_enterprise_source_information(document, datasets))
    quality_flags = tuple([*document.input_quality_flags, *source_flags])
    _apply_quality_statuses(datasets, quality_flags)
    source_limit_scopes = sorted(
        {
            scope
            for flag in quality_flags
            if flag.get("status") == "source_limited"
            for scope in (flag.get("scope") or ())
        }
    )
    summary["source_scope_status"] = "limited" if source_limit_scopes else "complete_as_reported"
    summary["source_limited_scopes"] = source_limit_scopes
    if any(flag.get("field") == "available_limit" and flag.get("status") == "estimated" for flag in quality_flags):
        summary["available_limit_status"] = "estimated"

    facts = extract_enterprise_identity_facts(document)
    identity_records = datasets.get("enterprise_report_identity") or []
    if identity_records:
        facts.update(
            {
                key: value
                for key, value in identity_records[0].items()
                if key
                not in frozenset(
                    {
                        "enterprise_identity_id",
                        "sequence",
                        "source_page",
                        "source_page_end",
                        "source",
                        "source_refs",
                        "confidence",
                    }
                )
                and value not in (None, "")
            }
        )
    facts.pop("company_name", None)
    facts.update({"report_subtype": "enterprise", "content_mode": content_mode})

    from docmirror.plugins.credit_report.business_assembly import _build_audit

    credit_audit = _build_audit(
        parse_result=document,
        full_text=document.full_text,
        report_subtype="enterprise",
        content_mode=content_mode,
        collections={
            "credit_accounts": datasets.get("enterprise_credit_accounts", []),
            "credit_lines": datasets.get("enterprise_credit_facilities", []),
            "repayment_liability_records": datasets.get("enterprise_repayment_responsibility_accounts", []),
            "repayment_records": [],
            "overdue_records": [],
            "inquiry_records": [],
        },
        conflicts=[],
        credit_summary=summary,
    )
    completeness = _dataset_completeness(datasets, continuation_audit, quality_flags)
    from docmirror.plugins.credit_report.enterprise_native.variant import variant

    extraction_report = build_enterprise_extraction_report(
        document,
        datasets,
        continuation_audit=continuation_audit,
        dataset_completeness=completeness,
        data_dictionary=variant.data_dictionary(),
    ).to_payload()
    return EnterpriseSemanticDocument(
        facts=facts,
        datasets=datasets,
        sections=_sections(document),
        credit_summary=summary,
        credit_extraction_audit=credit_audit,
        continuation_audit=tuple(continuation_audit),
        quality_flags=quality_flags,
        dataset_completeness=completeness,
        extraction_report=extraction_report,
    )


def run_enterprise_pipeline(
    parse_result: Any,
    *,
    content_mode: str = "native_text",
) -> EnterprisePipelineArtifacts:
    """Reconstruct the canonical IR, decode once, and retain private debug JSON."""
    document = build_canonical_enterprise_document(parse_result)
    semantic_document = extract_enterprise_semantic_document(document, content_mode=content_mode)
    return EnterprisePipelineArtifacts(
        document_ir=document,
        semantic_document=semantic_document,
        ir_debug_json=document.to_debug_json(),
        semantic_debug_json=semantic_document.to_debug_json(),
    )


__all__ = [
    "EnterprisePipelineArtifacts",
    "EnterpriseSemanticDocument",
    "extract_enterprise_semantic_document",
    "run_enterprise_pipeline",
]
