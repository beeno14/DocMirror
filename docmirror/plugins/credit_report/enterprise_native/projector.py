# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-seal projection for the canonical digital-enterprise pipeline."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.credit_report.enterprise_native.pipeline import run_enterprise_pipeline
from docmirror.plugins.credit_report.enterprise_native.quality import quality_warning
from docmirror.plugins.credit_report.projection import (
    _account_structure_warnings,
    _records,
)
from docmirror.plugins.credit_report.report_profile import (
    detect_credit_report_content_mode,
    recover_credit_report_header_fields,
)
from docmirror.plugins.credit_report.semantic_enrichment import enrich_credit_report_record_evidence

_OMITTED_PUBLIC_DATASETS = frozenset(
    {
        "report_notes",
        "enterprise_section_presence",
    }
)

_PUBLIC_BUSINESS_FIELDS: dict[str, tuple[str, ...]] = {
    "enterprise_report_metadata": (
        "report_edition",
        "report_number",
        "query_institution",
        "report_time",
    ),
    "enterprise_exchange_rates": (
        "exchange_rate_usd_cny",
        "exchange_rate_effective_period",
    ),
    "enterprise_report_identity": (
        "subject_name",
        "identity_subject_name",
        "cover_subject_name",
        "zhongzheng_code",
        "cover_zhongzheng_code",
        "unified_social_credit_code",
        "cover_unified_social_credit_code",
        "organization_code",
        "business_registration_number",
        "institution_credit_code",
        "national_tax_id",
        "local_tax_id",
    ),
    "enterprise_dispute_overview": (
        "in_progress_dispute_count",
        "dispute_status",
    ),
    "enterprise_credit_overview": (
        "first_credit_year",
        "credit_institution_count",
        "active_credit_institution_count",
        "first_repayment_responsibility_year",
        "credit_balance",
        "credit_attention_balance",
        "credit_adverse_balance",
        "guarantee_balance",
        "guarantee_attention_balance",
        "guarantee_adverse_balance",
        "recovered_debt_balance",
        "currency",
        "amount_unit",
    ),
    "enterprise_public_record_counts": (
        "record_type",
        "record_count",
    ),
    "enterprise_current_credit_summary": (
        "transaction_group",
        "business_category",
        "is_total",
        "normal_account_count",
        "normal_balance",
        "attention_account_count",
        "attention_balance",
        "adverse_account_count",
        "adverse_balance",
        "total_account_count",
        "total_balance",
        "currency",
        "amount_unit",
    ),
    "enterprise_facility_summary": (
        "facility_type",
        "total_limit",
        "used_limit",
        "available_limit",
        "utilization_rate",
        "currency",
        "amount_unit",
    ),
    "enterprise_repayment_responsibility_summary": (
        "transaction_group",
        "responsibility_type",
        "is_total",
        "recovered_responsibility_amount",
        "recovered_account_count",
        "recovered_balance",
        "other_credit_responsibility_amount",
        "other_credit_account_count",
        "other_credit_balance",
        "other_credit_attention_balance",
        "other_credit_adverse_balance",
        "guarantee_responsibility_amount",
        "guarantee_account_count",
        "guarantee_balance",
        "guarantee_attention_balance",
        "guarantee_adverse_balance",
        "currency",
        "amount_unit",
    ),
    "enterprise_closed_credit_summary": (
        "transaction_group",
        "business_category",
        "is_total",
        "normal_account_count",
        "attention_account_count",
        "adverse_account_count",
        "total_account_count",
    ),
    "enterprise_recovery_summary": (
        "settlement_status",
        "recovery_type",
        "account_count",
        "balance",
        "amount",
        "latest_disposal_date",
        "latest_repayment_date",
        "disposal_completion_date",
        "settlement_date",
        "currency",
        "amount_unit",
    ),
    "enterprise_overdue_summary": (
        "overdue_principal",
        "overdue_interest_and_other",
        "overdue_total",
        "currency",
        "amount_unit",
    ),
    "enterprise_profile": (
        "economic_type",
        "economic_type_source_institution",
        "organization_type",
        "organization_type_source_institution",
        "enterprise_scale",
        "enterprise_scale_source_institution",
        "industry",
        "industry_source_institution",
        "establishment_year",
        "establishment_year_source_institution",
        "registration_certificate_valid_through",
        "registration_certificate_valid_through_source_institution",
        "registered_address",
        "registered_address_source_institution",
        "operating_address",
        "operating_address_source_institution",
        "operating_status",
        "operating_status_source_institution",
    ),
    "enterprise_capital_summary": (
        "registered_capital_amount",
        "currency",
        "amount_unit",
        "contributor_count",
        "source_institution",
        "update_date",
    ),
    "enterprise_contributors": (
        "role",
        "name",
        "identity_type",
        "identity_number",
        "ownership_percentage",
        "source_institution",
        "update_date",
    ),
    "enterprise_key_personnel": (
        "role",
        "name",
        "identity_type",
        "identity_number",
        "ownership_percentage",
        "source_institution",
        "update_date",
    ),
    "enterprise_relationships": (
        "relationship_type",
        "name",
        "identity_type",
        "identity_number",
        "source_institution",
        "update_date",
    ),
    "enterprise_credit_detail_groups": (
        "group_phase",
        "group_kind",
        "business_category",
        "reported_record_count",
    ),
    "enterprise_displayed_credit_summary": (
        "transaction_group",
        "settlement_status",
        "business_category",
        "institution",
        "business_type",
        "five_tier_class",
        "account_count",
        "balance",
        "discount_amount",
        "overdue_total",
        "overdue_principal",
        "advance_flag",
        "currency",
        "amount_unit",
    ),
    "enterprise_credit_accounts": (
        "account_id",
        "business_category",
        "account_identifier",
        "institution",
        "creditor_institution",
        "business_type",
        "status",
        "open_date",
        "maturity_date",
        "close_date",
        "receive_date",
        "snapshot_date",
        "currency",
        "amount_unit",
        "loan_amount",
        "credit_limit",
        "discount_amount",
        "guarantee_amount",
        "balance",
        "risk_exposure_amount",
        "deposit_ratio",
        "issuance_form",
        "guarantee_type",
        "counter_guarantee_type",
        "five_tier_class",
        "current_overdue",
        "overdue_total",
        "overdue_principal",
        "overdue_months",
        "scheduled_payment_date",
        "scheduled_payment",
        "last_repayment_date",
        "last_repayment_amount",
        "repayment_method",
        "remaining_periods",
        "special_transaction",
        "credit_agreement_identifier",
        "original_creditor_name",
        "original_claim_type",
        "original_debt_type",
    ),
    "enterprise_account_annotations": (
        "account_id",
        "account_identifier",
        "annotation_type",
        "annotation_type_label",
        "issuer",
        "annotation_date",
        "annotation_content",
        "dispute_status",
    ),
    "enterprise_interest_arrears": (
        "institution",
        "arrears_type",
        "arrears_balance",
        "balance_change_date",
        "snapshot_date",
        "currency",
        "amount_unit",
    ),
    "enterprise_credit_facilities": (
        "account_id",
        "facility_type",
        "facility_product",
        "credit_agreement_identifier",
        "limit_identifier",
        "institution",
        "revolving_flag",
        "effective_date",
        "maturity_date",
        "snapshot_date",
        "total_limit",
        "used_limit",
        "available_limit",
        "facility_limit",
        "currency",
        "amount_unit",
        "status",
    ),
    "enterprise_repayment_responsibility_group_details": (
        "transaction_group",
        "responsibility_type",
        "contract_number",
        "responsibility_amount",
        "institution",
        "business_type",
        "five_tier_class",
        "account_count",
        "loan_amount",
        "guarantee_amount",
        "balance",
        "overdue_total",
        "overdue_principal",
        "currency",
        "amount_unit",
    ),
    "enterprise_repayment_responsibility_accounts": (
        "responsibility_type",
        "related_party_name",
        "related_party_id_type",
        "related_party_id_number",
        "liability_date",
        "account_identifier",
        "contract_number",
        "guarantee_contract_identifier",
        "institution",
        "business_type",
        "open_date",
        "open_or_receive_date",
        "maturity_date",
        "responsibility_amount",
        "responsibility_currency",
        "responsibility_amount_unit",
        "loan_or_credit_amount",
        "obligation_currency",
        "obligation_amount_unit",
        "balance",
        "five_tier_class",
        "overdue_total",
        "overdue_principal",
        "overdue_months",
        "repayment_status",
        "remaining_periods",
        "snapshot_date",
        "currency",
        "amount_unit",
    ),
    "enterprise_attachment_accounts": (
        "attachment_account_id",
        "attachment_record_type",
        "account_id",
        "account_identifier",
        "account_status",
        "business_category",
        "institution",
        "business_type",
        "five_tier_class",
    ),
    "enterprise_credit_supplement": (
        "attachment_account_id",
        "account_id",
        "account_identifier",
        "account_status",
        "business_category",
        "institution",
        "business_type",
        "report_date",
        "balance",
        "balance_change_date",
        "five_tier_class",
        "classification_date",
        "overdue_total",
        "overdue_principal",
        "overdue_months",
        "scheduled_repayment_date",
        "scheduled_repayment_amount",
        "actual_repayment_date",
        "actual_repayment_amount",
        "repayment_method",
        "remaining_periods",
        "currency",
        "amount_unit",
    ),
    "enterprise_attachment_credit_details": (
        "attachment_account_id",
        "account_id",
        "account_identifier",
        "account_status",
        "business_category",
        "institution",
        "business_type",
        "open_date",
        "maturity_date",
        "close_date",
        "snapshot_date",
        "last_repayment_date",
        "currency",
        "amount_unit",
        "credit_limit",
        "loan_amount",
        "discount_amount",
        "instrument_amount",
        "guarantee_amount",
        "balance",
        "risk_exposure_amount",
        "guarantee_type",
        "counter_guarantee_type",
        "deposit_ratio",
        "five_tier_class",
        "credit_agreement_identifier",
        "repayment_method",
        "advance_flag",
    ),
    "enterprise_special_transactions": (
        "attachment_account_id",
        "account_id",
        "account_identifier",
        "business_category",
        "institution",
        "business_type",
        "transaction_type",
        "transaction_date",
        "transaction_amount",
        "due_date_change_months",
        "transaction_detail",
        "currency",
        "amount_unit",
    ),
    "enterprise_utility_payment_history": (
        "statistics_month",
        "payment_status",
        "amount_due",
        "amount_paid",
        "cumulative_arrears",
        "currency",
        "amount_unit",
    ),
    "enterprise_housing_fund_history": (
        "statistics_month",
        "payment_status",
        "amount_due",
        "amount_paid",
        "cumulative_arrears",
        "currency",
        "amount_unit",
    ),
}

_NON_BUSINESS_FIELDS = frozenset(
    {
        "sequence",
        "source_page",
        "source_page_end",
        "source_table_id",
        "source_table_id_end",
        "source_sequence",
        "page",
        "contributor_source_page",
        "enterprise_identity_id",
        "enterprise_dispute_overview_id",
        "enterprise_credit_overview_id",
        "enterprise_public_record_count_id",
        "enterprise_recovery_summary_id",
        "enterprise_overdue_summary_id",
        "enterprise_profile_id",
        "current_summary_id",
        "closed_summary_id",
        "displayed_summary_id",
        "responsibility_summary_id",
        "credit_detail_group_id",
        "account_annotation_id",
        "interest_arrears_id",
        "credit_line_id",
        "responsibility_group_detail_id",
        "liability_id",
        "public_record_id",
        "supplement_id",
        "attachment_detail_id",
        "special_transaction_id",
        "utility_history_id",
        "housing_fund_history_id",
        "account_type",
        "account_state",
        "activation_state",
        "payoff_state",
        "count_scope",
        "summary_scope",
        "represented_dataset",
        "reported_record_count_conflicts",
        "five_tier_class_source",
        "continuation_complete",
        "responsibility_amount_reported",
        "source_group_account_count",
        "subject_name_assertion_status",
        "contributor_status",
        "history_status",
        "credit_agreement_status",
        "amount_kind",
        "amount",
        "source_account_count",
        "source_reported_amount",
        "overdue_months_or_repayment_status",
    }
)

_DISPLAYED_AMOUNT_FIELDS = {
    "balance": "balance",
    "discount_amount": "discount_amount",
}
_ATTACHMENT_AMOUNT_FIELDS = frozenset(
    {
        "credit_limit",
        "loan_amount",
        "discount_amount",
        "instrument_amount",
        "guarantee_amount",
    }
)
_DERIVED_COLUMN_SOURCES = {
    "account_count": "source_account_count",
    "balance": "source_reported_amount",
    "discount_amount": "source_reported_amount",
    "overdue_months": "overdue_months_or_repayment_status",
    "repayment_status": "overdue_months_or_repayment_status",
}
_DERIVED_COLUMN_LABELS = {
    "account_count": "账户数",
    "balance": "余额",
    "discount_amount": "贴现金额",
    "overdue_months": "逾期月数",
    "repayment_status": "还款状态",
}
_DERIVED_COLUMN_TYPES = {
    "account_count": "integer",
    "balance": "money",
    "discount_amount": "money",
    "overdue_months": "integer",
    "repayment_status": "string",
}

_SECTION_IDENTITY_DIRECT_FIELDS = frozenset(
    {
        "subject_name",
        "zhongzheng_code",
        "unified_social_credit_code",
        "organization_code",
        "business_registration_number",
        "institution_credit_code",
        "national_tax_id",
        "local_tax_id",
    }
)
_SECTION_IDENTITY_SOURCE_ALIASES = {
    "identity_subject_name": "subject_name",
    "cover_subject_name": "subject_name",
    "cover_zhongzheng_code": "zhongzheng_code",
    "cover_unified_social_credit_code": "unified_social_credit_code",
}
_SECTION_OVERVIEW_DIRECT_FIELDS = frozenset(
    {
        "first_credit_year",
        "credit_institution_count",
        "active_credit_institution_count",
        "first_repayment_responsibility_year",
        "credit_balance",
        "credit_attention_balance",
        "credit_adverse_balance",
        "guarantee_balance",
        "guarantee_attention_balance",
        "guarantee_adverse_balance",
        "recovered_debt_balance",
        "amount_unit",
    }
)
_SECTION_TECHNICAL_ITEMS = frozenset(
    {
        "subject_name_assertion_status",
        "first_credit_year_status",
        "first_repayment_responsibility_year_status",
        "available_limit_status",
        "public_record_overview_count_scope",
        "reported_account_count_basis",
        "source_account_summary_table_id",
        "source_account_summary_page",
        "displayed_credit_account_card_count",
        "displayed_credit_facility_count",
        "extracted_public_record_count",
        "extracted_public_record_count_scope",
        "account_dataset_scope",
        "account_dataset_scope_note",
        "attachment_account_count",
        "attachment_history_row_count",
        "attachment_detail_card_count",
        "attachment_special_transaction_count",
        "source_scope_status",
    }
)
_SECTION_REPORTED_TOTAL_FIELDS = {
    "reported_account_count": "total_account_count",
    "reported_account_balance": "total_balance",
}
_SECTION_REPORTED_GROUP_FIELDS = {
    "reported_account_counts": "total_account_count",
    "reported_account_balances": "total_balance",
}
_SECTION_PUBLIC_COUNT_ALIASES = {
    "non_credit_accounts": "non_credit_accounts",
    "tax_arrears": "tax_arrears",
    "civil_judgments": "civil_judgment",
    "enforcements": "enforcement",
    "administrative_penalties": "administrative_penalty",
}
_SECTION_TECHNICAL_GROUPS = frozenset({"extracted_public_record_type_counts"})
_BORROWING_TRANSACTION_GROUP = "借贷交易"


def enterprise_public_dataset_policy() -> dict[str, tuple[str, ...] | None]:
    """Return the closed-world enterprise Community JSON dataset contract."""

    from docmirror.plugins.credit_report.enterprise_native.extraction import (
        enterprise_public_record_dataset_specs,
    )
    from docmirror.plugins.credit_report.enterprise_native.variant import variant

    policy: dict[str, tuple[str, ...] | None] = {
        name: fields for name, fields in _PUBLIC_BUSINESS_FIELDS.items()
    }
    policy.update({name: None for name in _OMITTED_PUBLIC_DATASETS})
    for spec in enterprise_public_record_dataset_specs().values():
        policy[str(spec["dataset_id"])] = tuple(str(key) for key in spec["columns"])

    missing = [name for name in variant.dataset_names() if name not in policy]
    extra = [name for name in policy if name not in variant.dataset_names()]
    if missing or extra:
        raise RuntimeError(
            "enterprise public dataset policy is out of sync: "
            f"missing={missing}, extra={extra}"
        )
    return {name: policy[name] for name in variant.dataset_names()}


def _has_business_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _same_value(left: Any, right: Any) -> bool:
    if left == right:
        return True
    return str(left) == str(right)


def _normalize_public_business_row(
    dataset_name: str,
    normalized: dict[str, Any],
) -> dict[str, Any]:
    values = dict(normalized)

    if dataset_name == "enterprise_report_identity":
        for assertion_key, canonical_key in _SECTION_IDENTITY_SOURCE_ALIASES.items():
            assertion = values.get(assertion_key)
            canonical = values.get(canonical_key)
            if _has_business_value(assertion) and _same_value(assertion, canonical):
                values.pop(assertion_key, None)

    if dataset_name == "enterprise_displayed_credit_summary":
        account_count = values.pop("source_account_count", None)
        if _has_business_value(account_count):
            values["account_count"] = account_count
        amount = values.pop("source_reported_amount", None)
        amount_kind = str(values.pop("amount_kind", "") or "")
        if _has_business_value(amount):
            target = _DISPLAYED_AMOUNT_FIELDS.get(amount_kind)
            if target is None:
                raise ValueError(
                    "unsupported enterprise displayed-summary amount kind: "
                    f"{amount_kind!r}"
                )
            existing = values.get(target)
            if _has_business_value(existing) and not _same_value(existing, amount):
                raise ValueError(
                    f"conflicting enterprise displayed-summary {target}: "
                    f"{existing!r} != {amount!r}"
                )
            values[target] = amount

    if dataset_name == "enterprise_attachment_credit_details":
        amount = values.pop("amount", None)
        amount_kind = str(values.pop("amount_kind", "") or "")
        if _has_business_value(amount):
            if amount_kind not in _ATTACHMENT_AMOUNT_FIELDS:
                raise ValueError(
                    "unsupported enterprise attachment amount kind: "
                    f"{amount_kind!r}"
                )
            existing = values.get(amount_kind)
            if _has_business_value(existing) and not _same_value(existing, amount):
                raise ValueError(
                    f"conflicting enterprise attachment {amount_kind}: "
                    f"{existing!r} != {amount!r}"
                )
            values[amount_kind] = amount

    if dataset_name == "enterprise_repayment_responsibility_accounts":
        packed = values.pop("overdue_months_or_repayment_status", None)
        if _has_business_value(packed):
            text = str(packed).strip()
            if re.fullmatch(r"[+]?[0-9]+", text):
                values["overdue_months"] = int(text)
            elif text not in {"-", "--"}:
                values["repayment_status"] = packed

    return values


def _known_non_business_field(key: str, allowed: set[str]) -> bool:
    if key in _NON_BUSINESS_FIELDS:
        return True
    if key.endswith("_source_state"):
        return True
    if not key.endswith("_status"):
        return False
    base = key.removesuffix("_status")
    return (
        base in allowed
        or base in _NON_BUSINESS_FIELDS
        or base.endswith("_source_institution")
    )


def _project_business_columns(
    dataset_name: str,
    dataset: dict[str, Any],
    field_order: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    originals = {
        str(column.get("key") or ""): column
        for column in dataset.get("columns") or []
        if isinstance(column, dict) and column.get("key")
    }
    present = {
        key
        for row in rows
        for key, value in (row.get("normalized") or {}).items()
        if _has_business_value(value)
    }
    columns: list[dict[str, Any]] = []
    for key in field_order:
        if key not in present:
            continue
        source_key = _DERIVED_COLUMN_SOURCES.get(key, key)
        column = deepcopy(originals.get(key) or originals.get(source_key) or {})
        column.update(
            {
                "key": key,
                "label": str(column.get("label") or _DERIVED_COLUMN_LABELS.get(key) or key),
                "type": str(column.get("type") or _DERIVED_COLUMN_TYPES.get(key) or "string"),
                "nullable": any(
                    not _has_business_value((row.get("normalized") or {}).get(key))
                    for row in rows
                ),
                "raw_available": any(
                    _has_business_value((row.get("canonical_raw") or {}).get(key))
                    or _has_business_value((row.get("raw") or {}).get(key))
                    for row in rows
                ),
                "evidence_available": False,
            }
        )
        columns.append(column)
    return columns


def _project_business_dataset(
    dataset: dict[str, Any],
    field_order: tuple[str, ...],
) -> dict[str, Any]:
    projected = deepcopy(dataset)
    dataset_name = str(projected.get("name") or "")
    allowed = set(field_order)
    rows: list[dict[str, Any]] = []
    for row in projected.get("rows") or []:
        if not isinstance(row, dict):
            continue
        source_normalized = (
            row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
        )
        transformed = _normalize_public_business_row(dataset_name, source_normalized)
        unknown = sorted(
            key
            for key in source_normalized
            if key not in allowed and not _known_non_business_field(key, allowed)
        )
        if unknown:
            raise ValueError(
                f"unclassified enterprise Community fields in {dataset_name}: {unknown}"
            )
        normalized = {
            key: deepcopy(transformed[key])
            for key in field_order
            if key in transformed and _has_business_value(transformed[key])
        }
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        page_range = source.get("page_range")
        compact_source = (
            {"page_range": deepcopy(page_range)}
            if isinstance(page_range, list) and len(page_range) == 2
            else {}
        )
        canonical_raw = _project_business_raw_pool(
            dataset_name,
            row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {},
            normalized,
            source_normalized,
        )
        raw = _project_business_raw_pool(
            dataset_name,
            row.get("raw") if isinstance(row.get("raw"), dict) else {},
            normalized,
            source_normalized,
        )
        rows.append(
            {
                "record_id": str(row.get("record_id") or ""),
                "normalized": normalized,
                "canonical_raw": canonical_raw,
                "raw": raw,
                "source": compact_source,
            }
        )

    projected["rows"] = rows
    projected["row_count"] = len(rows)
    projected["columns"] = _project_business_columns(
        dataset_name,
        projected,
        field_order,
        rows,
    )
    present_columns = [str(column["key"]) for column in projected["columns"]]
    projected["reading_columns"] = present_columns
    return projected


def _normalized_dataset_rows(
    datasets: dict[str, dict[str, Any]],
    dataset_name: str,
) -> list[dict[str, Any]]:
    dataset = datasets.get(dataset_name) or {}
    return [
        row["normalized"]
        for row in dataset.get("rows") or []
        if isinstance(row, dict) and isinstance(row.get("normalized"), dict)
    ]


def _require_section_value(
    *,
    section_key: str,
    value: Any,
    rows: list[dict[str, Any]],
    dataset_key: str,
) -> None:
    matches = [row for row in rows if _same_value(row.get(dataset_key), value)]
    if len(matches) != 1:
        raise ValueError(
            "enterprise section business value is not uniquely materialized: "
            f"{section_key}={value!r} -> {dataset_key} (matches={len(matches)})"
        )


def _validate_section_business_conservation(
    sections: list[Any],
    public_datasets: list[dict[str, Any]],
) -> None:
    """Prove section business values exist in canonical datasets before pruning."""

    public_by_name = {
        str(dataset.get("name") or ""): dataset
        for dataset in public_datasets
        if isinstance(dataset, dict)
    }
    public_identity = _normalized_dataset_rows(
        public_by_name,
        "enterprise_report_identity",
    )
    overview = _normalized_dataset_rows(
        public_by_name,
        "enterprise_credit_overview",
    )
    current_summary = _normalized_dataset_rows(
        public_by_name,
        "enterprise_current_credit_summary",
    )
    public_counts = _normalized_dataset_rows(
        public_by_name,
        "enterprise_public_record_counts",
    )

    for section in sections:
        if not isinstance(section, dict):
            continue
        items = [item for item in section.get("items") or [] if isinstance(item, dict)]
        item_keys = [str(item.get("key") or "") for item in items]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("duplicate enterprise Community section item key")
        item_by_key = {str(item.get("key") or ""): item for item in items}
        for key, item in item_by_key.items():
            value = item.get("value")
            if key in _SECTION_TECHNICAL_ITEMS or key in _SECTION_REPORTED_TOTAL_FIELDS:
                continue
            if key in _SECTION_OVERVIEW_DIRECT_FIELDS:
                _require_section_value(
                    section_key=key,
                    value=value,
                    rows=overview,
                    dataset_key=key,
                )
                continue
            if key in _SECTION_IDENTITY_DIRECT_FIELDS:
                identity_keys = (
                    ("subject_name", "identity_subject_name", "cover_subject_name")
                    if key == "subject_name"
                    else (key,)
                )
                matches = [
                    row
                    for row in public_identity
                    if any(_same_value(row.get(identity_key), value) for identity_key in identity_keys)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "enterprise identity section value is not uniquely materialized: "
                        f"{key}={value!r} (matches={len(matches)})"
                    )
                continue
            if key in _SECTION_IDENTITY_SOURCE_ALIASES:
                canonical_key = _SECTION_IDENTITY_SOURCE_ALIASES[key]
                matches = [
                    row
                    for row in public_identity
                    if _same_value(row.get(key), value)
                    or _same_value(row.get(canonical_key), value)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "enterprise identity source assertion is not uniquely materialized: "
                        f"{key}={value!r} (matches={len(matches)})"
                    )
                continue
            raise ValueError(f"unclassified enterprise Community section item: {key}")

        reported_total_items = {
            key: item_by_key[key].get("value")
            for key in _SECTION_REPORTED_TOTAL_FIELDS
            if key in item_by_key
        }
        if reported_total_items:
            total_matches = [
                row
                for row in current_summary
                if row.get("is_total") is True
                and row.get("transaction_group") == _BORROWING_TRANSACTION_GROUP
                and all(
                    _same_value(row.get(_SECTION_REPORTED_TOTAL_FIELDS[key]), value)
                    for key, value in reported_total_items.items()
                )
            ]
            if len(total_matches) != 1:
                raise ValueError(
                    "enterprise reported account totals are not uniquely materialized in "
                    f"enterprise_current_credit_summary (matches={len(total_matches)})"
                )

        groups = [group for group in section.get("groups") or [] if isinstance(group, dict)]
        group_keys = [str(group.get("key") or "") for group in groups]
        if len(group_keys) != len(set(group_keys)):
            raise ValueError("duplicate enterprise Community section group key")
        for group in groups:
            group_key = str(group.get("key") or "")
            group_items = [
                item for item in group.get("items") or [] if isinstance(item, dict)
            ]
            group_item_keys = [str(item.get("key") or "") for item in group_items]
            if len(group_item_keys) != len(set(group_item_keys)):
                raise ValueError(
                    f"duplicate enterprise Community section group item key: {group_key}"
                )
            if group_key in _SECTION_TECHNICAL_GROUPS:
                continue
            if group_key == "public_record_overview_counts":
                for item in group_items:
                    item_key = str(item.get("key") or "")
                    record_type = _SECTION_PUBLIC_COUNT_ALIASES.get(item_key)
                    if record_type is None:
                        raise ValueError(
                            "unclassified enterprise public-record count section item: "
                            f"{item_key}"
                        )
                    matches = [
                        row
                        for row in public_counts
                        if row.get("record_type") == record_type
                        and _same_value(row.get("record_count"), item.get("value"))
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "enterprise public-record overview count is not uniquely "
                            f"materialized: {item_key} (matches={len(matches)})"
                        )
                continue
            if group_key in _SECTION_REPORTED_GROUP_FIELDS:
                value_key = _SECTION_REPORTED_GROUP_FIELDS[group_key]
                for item in group_items:
                    category = str(item.get("key") or "")
                    matches = [
                        row
                        for row in current_summary
                        if row.get("is_total") is False
                        and row.get("transaction_group") == _BORROWING_TRANSACTION_GROUP
                        and row.get("business_category") == category
                        and _same_value(row.get(value_key), item.get("value"))
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "enterprise reported account category is not uniquely materialized: "
                            f"{group_key}/{category} (matches={len(matches)})"
                        )
                continue
            raise ValueError(f"unclassified enterprise Community section group: {group_key}")


def _clean_public_navigation(payload: dict[str, Any]) -> None:
    dataset_ids = {
        str(dataset.get("id") or "")
        for dataset in payload.get("datasets") or []
        if isinstance(dataset, dict) and dataset.get("id")
    }
    columns_by_dataset = {
        str(dataset.get("id") or ""): [
            str(column.get("key") or "")
            for column in dataset.get("columns") or []
            if isinstance(column, dict) and column.get("key")
        ]
        for dataset in payload.get("datasets") or []
        if isinstance(dataset, dict)
    }
    sections: list[dict[str, Any]] = []
    for section in payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section["items"] = []
        section["groups"] = []
        section["dataset_refs"] = [
            str(ref) for ref in section.get("dataset_refs") or [] if str(ref) in dataset_ids
        ]
        if section["dataset_refs"]:
            sections.append(section)
    payload["sections"] = sections
    section_ids = {str(section.get("id") or "") for section in sections}

    reading = payload.get("reading") if isinstance(payload.get("reading"), dict) else {}
    reading["tables"] = [
        {
            **table,
            "column_keys": list(
                columns_by_dataset.get(str(table.get("dataset_id") or ""), [])
            ),
        }
        for table in reading.get("tables") or []
        if isinstance(table, dict) and str(table.get("dataset_id") or "") in dataset_ids
    ]
    flow = [
        item
        for item in reading.get("document_flow") or []
        if isinstance(item, dict)
        and (
            (
                str(item.get("kind") or "") != "dataset"
                or str(item.get("ref_id") or "") in dataset_ids
            )
            and (
                str(item.get("kind") or "") != "section"
                or str(item.get("ref_id") or "") in section_ids
            )
        )
    ]
    reading["document_flow"] = [
        {**item, "order": order} for order, item in enumerate(flow, start=1)
    ]
    payload["reading"] = reading


def project_enterprise_community_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Project rich enterprise semantic datasets to business-only Community JSON."""

    projected = deepcopy(payload)
    policy = enterprise_public_dataset_policy()
    datasets: list[dict[str, Any]] = []
    for dataset in projected.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        dataset_name = str(dataset.get("name") or "")
        if dataset_name not in policy:
            raise ValueError(f"unclassified enterprise Community dataset: {dataset_name}")
        fields = policy[dataset_name]
        if fields is None:
            continue
        datasets.append(_project_business_dataset(dataset, fields))
    projected["datasets"] = datasets
    _validate_section_business_conservation(
        list(projected.get("sections") or []),
        datasets,
    )
    _clean_public_navigation(projected)
    return projected


def _project_business_raw_pool(
    dataset_name: str,
    rich_pool: dict[str, Any],
    normalized: dict[str, Any],
    rich_normalized: dict[str, Any],
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    amount_kind = str(rich_normalized.get("amount_kind") or "")
    for key, value in normalized.items():
        source_key = key
        if dataset_name == "enterprise_displayed_credit_summary":
            if key == "account_count":
                source_key = "source_account_count"
            elif key == _DISPLAYED_AMOUNT_FIELDS.get(amount_kind):
                source_key = "source_reported_amount"
        elif (
            dataset_name == "enterprise_repayment_responsibility_accounts"
            and key in {"overdue_months", "repayment_status"}
        ):
            source_key = "overdue_months_or_repayment_status"
        raw_value = rich_pool.get(source_key, value)
        if _has_business_value(raw_value):
            projected[key] = deepcopy(raw_value)
    return projected


def project_enterprise_artifact_semantic(
    semantic: dict[str, Any],
    public_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a transient clean CSV/audit view while retaining rich evidence."""

    artifact = deepcopy(semantic)
    rich_datasets = {
        str(dataset.get("name") or ""): dataset
        for dataset in semantic.get("datasets") or []
        if isinstance(dataset, dict)
    }
    artifact_datasets = deepcopy(public_payload.get("datasets") or [])
    for dataset in artifact_datasets:
        dataset_name = str(dataset.get("name") or "")
        rich_dataset = rich_datasets.get(dataset_name, {})
        rich_rows = {
            str(row.get("record_id") or ""): row
            for row in rich_dataset.get("rows") or []
            if isinstance(row, dict)
        }
        for row in dataset.get("rows") or []:
            if not isinstance(row, dict):
                continue
            rich_row = rich_rows.get(str(row.get("record_id") or ""), {})
            rich_normalized = (
                rich_row.get("normalized")
                if isinstance(rich_row.get("normalized"), dict)
                else {}
            )
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            for pool_name in ("canonical_raw", "raw"):
                rich_pool = (
                    rich_row.get(pool_name)
                    if isinstance(rich_row.get(pool_name), dict)
                    else {}
                )
                row[pool_name] = _project_business_raw_pool(
                    dataset_name,
                    rich_pool,
                    normalized,
                    rich_normalized,
                )
            row["source"] = deepcopy(
                rich_row.get("source") if isinstance(rich_row.get("source"), dict) else {}
            )
            if rich_row.get("confidence") not in (None, ""):
                row["confidence"] = deepcopy(rich_row["confidence"])
    artifact["datasets"] = artifact_datasets
    structure = artifact.get("structure") if isinstance(artifact.get("structure"), dict) else {}
    structure["sections"] = deepcopy(public_payload.get("sections") or [])
    artifact["structure"] = structure
    artifact["reading"] = deepcopy(public_payload.get("reading") or {})
    return artifact


def derive_enterprise_projection(plugin: Any, parse_result: Any, full_text: str = "") -> ProjectionData:
    """Run ParseResult -> canonical IR -> one enterprise schema -> projection."""
    content_mode = detect_credit_report_content_mode(parse_result)
    artifacts = run_enterprise_pipeline(parse_result, content_mode=content_mode)
    document = artifacts.document_ir
    semantic_document = artifacts.semantic_document
    from docmirror.plugins.credit_report.enterprise_native.variant import variant

    recovered_header = recover_credit_report_header_fields(
        document,
        document.full_text,
        report_subtype="enterprise",
    )
    domain_facts = {**semantic_document.facts, **recovered_header}
    for field_name in (
        "company_name",
        "id_type",
        "id_number",
        "subject_id",
        "marital_status",
        "report_edition",
        "exchange_rate_usd_cny",
        "exchange_rate_effective_period",
    ):
        domain_facts.pop(field_name, None)
    domain_facts["report_subtype"] = "enterprise"
    domain_facts["content_mode"] = content_mode
    domain_facts["credit_summary"] = semantic_document.credit_summary
    domain_facts["extraction_report"] = semantic_document.extraction_report
    domain_facts["source_information_quality"] = {
        "status": (
            "bad_input"
            if any(flag.get("severity") == "error" for flag in semantic_document.quality_flags)
            else "limited"
            if any(flag.get("status") in {"source_limited", "source_truncated", "estimated"} for flag in semantic_document.quality_flags)
            else "complete_as_reported"
        ),
        "flags": list(semantic_document.quality_flags),
    }
    domain_facts["field_details"] = {
        key: {
            "source": "canonical_enterprise_document_ir",
            "confidence": document.confidence,
        }
        for key, value in domain_facts.items()
        if key not in {"credit_summary", "extraction_report", "source_information_quality"}
        and value not in (None, "")
    }
    domain_facts["data_dictionary"] = variant.data_dictionary()

    datasets = {
        name: _records(name, rows)
        for name, rows in semantic_document.datasets.items()
        if rows
    }
    evidence_ids = enrich_credit_report_record_evidence(parse_result, datasets)
    accounts = datasets.get("enterprise_credit_accounts") or []
    warnings = tuple(
        dict.fromkeys(
            [
                *list(getattr(getattr(parse_result, "parser_info", None), "warnings", None) or []),
                *_account_structure_warnings(accounts),
                *[
                    quality_warning(flag)
                    for flag in semantic_document.quality_flags
                    if flag.get("severity") in {"warning", "error"}
                ],
                *[
                    f"{failure.get('code', 'ENTERPRISE_EXTRACTION_FAILURE')}: "
                    f"{failure.get('message', '')}".strip()
                    for failure in semantic_document.extraction_report.get("failures") or []
                    if isinstance(failure, dict)
                ],
            ]
        )
    )
    entity_fields = {
        "subject_name": domain_facts["subject_name"]
    } if domain_facts.get("subject_name") else {}
    semantic = variant.semantic_extensions()
    overrides = semantic.setdefault("community_projection_overrides", {})
    completeness = overrides.setdefault("completeness", {})
    internal_fields = overrides.setdefault("internal_fields", [])
    internal_facts = overrides.setdefault("internal_facts", [])
    internal_fields.extend(
        (
            "report_number",
            "query_institution",
            "report_time",
            "report_subtype",
            "content_mode",
        )
    )
    for dataset_name, details in semantic_document.dataset_completeness.items():
        count_key = f"enterprise_expected_{dataset_name}_count"
        domain_facts[count_key] = int(details.get("expected_row_count") or 0)
        internal_fields.append(count_key)
        internal_facts.append(count_key)
        completeness[dataset_name] = {
            "basis": "domain_fact_count",
            "count_key": count_key,
            "public_basis": str(details.get("basis") or "canonical_source_component_count"),
        }
    semantic["enterprise_dataset_completeness"] = semantic_document.dataset_completeness
    return ProjectionData(
        projector_id=plugin.projector_id,
        document_type=plugin.domain_name,
        entity_fields=entity_fields,
        domain_facts=domain_facts,
        semantic=semantic,
        datasets=datasets,
        sections=semantic_document.sections,
        warnings=warnings,
        evidence_ids=tuple(evidence_ids),
        confidence=float(getattr(parse_result, "confidence", 1.0) or 0.0),
        reason="canonical enterprise IR schema projection",
    )


__all__ = [
    "derive_enterprise_projection",
    "enterprise_public_dataset_policy",
    "project_enterprise_artifact_semantic",
    "project_enterprise_community_json",
]
