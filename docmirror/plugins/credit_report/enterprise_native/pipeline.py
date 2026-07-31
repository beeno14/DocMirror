# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enterprise-only business assembly.

The page parser and sealed evidence model remain shared infrastructure.  All
enterprise candidate selection, normalization, account semantics, and dataset
population decisions live in the enterprise package.
"""

from __future__ import annotations

from typing import Any

from docmirror.plugins.credit_report.value_utils import parse_number

_COLLECTIONS = (
    "credit_accounts",
    "credit_lines",
    "repayment_liability_records",
    "repayment_records",
    "overdue_records",
    "inquiry_records",
    "public_records",
)
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
    "due_date",
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
    "due_date",
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
    "due_date",
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
    "due_date_status",
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


def _value(record: dict[str, Any], field: str) -> Any:
    aliases = {
        "institution": ("management_institution", "institution"),
        "status": ("account_status", "status"),
    }.get(field, (field,))
    return next(
        (
            record.get(alias)
            for alias in aliases
            if record.get(alias) not in (None, "")
        ),
        None,
    )


def _normalized_value(field: str, value: Any) -> Any:
    if field in _NUMBER_FIELDS:
        number = parse_number(value)
        if number is None:
            return None
        if field in {"sequence", "remaining_periods", "current_overdue_periods", "overdue_months"}:
            return int(number)
        return number
    return value


def _base_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out.setdefault("confidence", 1.0 if out.get("source_refs") else 0.8)
    return out


def _normalize_account(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized: dict[str, Any] = {}
    for field in _ACCOUNT_FIELDS:
        value = _normalized_value(field, _value(out, field))
        if value not in (None, ""):
            normalized[field] = value
    institution = _value(out, "institution")
    if institution:
        normalized["institution"] = institution
    status = str(_value(out, "status") or "").lower()
    if status:
        normalized["status"] = status
    normalized["maturity_date"] = normalized.get("due_date")
    normalized["account_state"] = (
        "open"
        if status in {"active", "inactive"}
        else "closed"
        if status in {"closed", "settled"}
        else "unknown"
    )
    normalized["payoff_state"] = (
        "settled"
        if status in {"closed", "settled"}
        else "outstanding"
        if status == "active"
        else "unknown"
    )
    normalized["activation_state"] = "not_applicable"
    account_type = str(normalized.get("account_type") or "enterprise_credit")
    normalized.setdefault("account_type", account_type)
    for field in ("credit_limit", "loan_amount", "balance"):
        normalized[f"{field}_status"] = (
            "reported" if normalized.get(field) not in (None, "") else "not_reported"
        )
    out["normalized"] = normalized
    return out


def _normalize_line(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized: dict[str, Any] = {}
    for field in _LINE_FIELDS:
        value = _normalized_value(field, _value(out, field))
        if value not in (None, ""):
            normalized[field] = value
    institution = _value(out, "institution")
    if institution:
        normalized["institution"] = institution
    status = str(_value(out, "status") or "").lower()
    if status:
        normalized["status"] = status
    normalized["maturity_date"] = normalized.get("due_date")
    normalized["account_state"] = (
        "open"
        if status in {"active", "inactive"}
        else "closed"
        if status in {"closed", "settled"}
        else "unknown"
    )
    normalized["payoff_state"] = (
        "settled"
        if status in {"closed", "settled"}
        else "outstanding"
        if status == "active"
        else "unknown"
    )
    for field in ("total_limit", "used_limit", "available_limit"):
        normalized[f"{field}_status"] = (
            "reported" if normalized.get(field) not in (None, "") else "not_reported"
        )
    out["normalized"] = normalized
    return out


def _normalize_liability(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized: dict[str, Any] = {}
    for field in _LIABILITY_FIELDS:
        value = _normalized_value(field, _value(out, field))
        if value not in (None, ""):
            normalized[field] = value
    institution = _value(out, "institution")
    if institution:
        normalized["institution"] = institution
    normalized["maturity_date"] = normalized.get("due_date")
    out["normalized"] = normalized
    return out


def _normalize_passthrough(record: dict[str, Any]) -> dict[str, Any]:
    out = _base_record(record)
    normalized = {
        key: value
        for key, value in out.items()
        if key not in _META_FIELDS and value not in (None, "")
    }
    out["normalized"] = normalized
    return out


def _identity(collection: str, record: dict[str, Any], index: int) -> str:
    keys = {
        "credit_accounts": ("account_id", "account_identifier"),
        "credit_lines": ("credit_line_id", "account_identifier"),
        "repayment_liability_records": ("liability_id", "account_identifier"),
        "repayment_records": ("repayment_id",),
        "overdue_records": ("overdue_id",),
        "inquiry_records": ("inquiry_id",),
        "public_records": ("public_record_id",),
    }[collection]
    normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
    return str(
        next(
            (
                record.get(key) or normalized.get(key)
                for key in keys
                if record.get(key) or normalized.get(key)
            ),
            f"{collection}:r{index:06d}",
        )
    )


def _normalize_collection(
    collection: str,
    records: list[Any],
) -> list[dict[str, Any]]:
    normalizer = {
        "credit_accounts": _normalize_account,
        "credit_lines": _normalize_line,
        "repayment_liability_records": _normalize_liability,
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


def assemble_enterprise_business(
    parse_result: Any,
    full_text: str,
    *,
    content_mode: str,
    existing_collections: dict[str, list[Any]] | None,
    existing_summary: dict[str, Any] | None,
    variant: Any,
    variant_input: Any,
) -> dict[str, Any]:
    """Assemble enterprise records without entering personal/shared normalizers."""
    native = variant.extract_native_business(
        variant_input,
        full_text,
        content_mode=content_mode,
    )
    existing_collections = existing_collections or {}
    collections: dict[str, list[dict[str, Any]]] = {}
    for collection in _COLLECTIONS:
        # Enterprise native tables are authoritative. Generic candidates are
        # retained only when the enterprise extractor did not emit that grain.
        native_rows = list(native.get(collection) or [])
        source_rows = native_rows or list(existing_collections.get(collection) or [])
        collections[collection] = _normalize_collection(collection, source_rows)

    from docmirror.plugins.credit_report.semantic_enrichment import (
        enrich_credit_report_record_evidence,
    )

    enrich_credit_report_record_evidence(parse_result, collections)
    raw_summary = native.get("credit_summary")
    summary = {
        **dict(existing_summary or {}),
        **(dict(raw_summary) if isinstance(raw_summary, dict) else {}),
        "projected_account_count": len(collections["credit_accounts"]),
    }

    # Auditing is infrastructure rather than business normalization. Reuse its
    # stable output contract so existing artifact validators remain compatible.
    from docmirror.plugins.credit_report.business_assembly import _build_audit

    audit = _build_audit(
        parse_result=parse_result,
        full_text=full_text,
        report_subtype="enterprise",
        content_mode=content_mode,
        collections=collections,
        conflicts=[],
        credit_summary=summary,
    )
    return {
        **collections,
        "credit_summary": summary,
        "credit_extraction_audit": audit,
    }


__all__ = ["assemble_enterprise_business"]
