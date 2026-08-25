# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-seal projection for the canonical personal-brief pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.credit_report.personal_brief_native.contracts import (
    PERSONAL_BRIEF_ENUM_CONTRACT,
    PERSONAL_BRIEF_MONEY_FIELDS,
    PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
    validate_personal_brief_public_record,
)
from docmirror.plugins.credit_report.personal_brief_native.pipeline import (
    run_personal_brief_pipeline,
)
from docmirror.plugins.credit_report.personal_brief_native.schema import (
    _personal_brief_dataset_descriptors,
    personal_brief_semantic_extensions,
)
from docmirror.plugins.credit_report.projection import (
    _account_structure_warnings,
    _records,
)
from docmirror.plugins.credit_report.report_profile import (
    detect_credit_report_content_mode,
)

_PUBLIC_BUSINESS_FIELDS: dict[str, tuple[str, ...]] = {
    "personal_report_metadata": (
        "report_number",
        "report_time",
        "subject_name",
        "primary_id_type",
        "primary_id_number",
        "marital_status",
        "marital_status_raw",
        "reporting_currency",
        "reporting_amount_unit",
        "reporting_amount_precision",
    ),
    "identity_documents": (
        "holder_name",
        "document_type",
        "document_number",
        "is_primary",
    ),
    "personal_credit_summary_records": (
        "metric",
        "business_category",
        "value",
        "reporting_status",
    ),
    "asset_disposition_records": (
        "disposition_date",
        "asset_management_company",
        "received_debt_amount",
        "snapshot_date",
        "balance",
        "last_repayment_date",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "guarantor_compensation_records": (
        "compensation_start_date",
        "guarantor",
        "cumulative_compensation_amount",
        "settlement_date",
        "settlement_state",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "credit_accounts": (
        "account_id",
        "account_identifier",
        "account_type",
        "business_category",
        "institution",
        "business_type",
        "credit_card_type",
        "card_tail",
        "open_date",
        "snapshot_date",
        "contract_maturity_date",
        "credit_line_expiry_date",
        "credit_line_validity_type",
        "termination_event_date",
        "termination_event_type",
        "account_currency",
        "reporting_amount_currency",
        "reporting_amount_unit",
        "credit_limit",
        "credit_limit_status",
        "used_amount",
        "used_amount_status",
        "loan_amount",
        "loan_amount_status",
        "balance",
        "balance_status",
        "account_lifecycle_state",
        "card_activation_state",
        "payoff_state",
        "credit_quality_status",
        "current_overdue",
        "ever_overdue",
        "overdue_months",
        "over_90_days",
        "unbilled_installment_balance",
    ),
    "overdue_records": (
        "account_id",
        "account_type",
        "institution",
        "business_type",
        "card_tail",
        "open_date",
        "currency",
        "period_scope",
        "overdue_months",
        "over_90_days_months",
        "current_overdue",
        "current_overdue_status",
        "over_90_days",
    ),
    "repayment_liability_records": (
        "liability_date",
        "related_party_name",
        "related_party_id_type",
        "related_party_id_number",
        "institution",
        "business_type",
        "underlying_business_type",
        "snapshot_balance_business_type",
        "responsibility_type",
        "responsibility_amount",
        "responsibility_amount_reported",
        "contract_number",
        "snapshot_date",
        "balance",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "postpaid_records": (
        "institution",
        "business_type",
        "billing_month",
        "service_start_date",
        "payment_status",
        "current_arrears_amount",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "tax_arrears_records": (
        "tax_authority",
        "statistics_date",
        "arrears_amount",
        "taxpayer_identifier",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "civil_judgment_records": (
        "filing_court",
        "case_number",
        "cause",
        "cause_status",
        "filing_date",
        "closure_method",
        "claim_subject",
        "claim_amount",
        "judgment_result",
        "judgment_effective_date",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "enforcement_records": (
        "court",
        "case_number",
        "cause",
        "cause_status",
        "filing_date",
        "case_status",
        "closure_method",
        "closure_date",
        "requested_subject",
        "requested_amount",
        "executed_subject",
        "executed_amount",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "administrative_penalty_records": (
        "authority",
        "document_number",
        "penalty_content",
        "penalty_amount",
        "effective_date",
        "end_date",
        "administrative_review_result",
        "administrative_review_result_status",
        "reporting_amount_currency",
        "reporting_amount_unit",
    ),
    "institution_statement_records": (
        "statement_content",
        "added_date",
    ),
    "inquiry_records": (
        "inquiry_date",
        "institution",
        "reason",
        "query_channel",
        "inquiry_type",
    ),
}

_OMITTED_PUBLIC_DATASETS = frozenset({"repayment_records", "public_records", "report_notes"})

_NON_BUSINESS_FIELDS: dict[str, frozenset[str]] = {
    "personal_report_metadata": frozenset(
        {
            "personal_report_metadata_id",
            "amount_policy_source",
        }
    ),
    "identity_documents": frozenset({"identity_document_id", "sequence"}),
    "personal_credit_summary_records": frozenset({"credit_summary_record_id", "sequence", "summary_scope"}),
    "asset_disposition_records": frozenset({"asset_disposition_id", "sequence"}),
    "guarantor_compensation_records": frozenset({"guarantor_compensation_id", "sequence"}),
    "credit_accounts": frozenset(
        {
            "sequence",
            "due_date",
            "close_date",
            "transfer_out_date",
            "currency",
            "amount_unit",
            "reporting_amount_precision",
            "account_state",
            "activation_state",
            "status",
            "source_section",
            "source_sequence",
        }
    ),
    "overdue_records": frozenset({"overdue_id", "sequence"}),
    "repayment_liability_records": frozenset(
        {
            "liability_id",
            "sequence",
            "currency",
            "amount_unit",
        }
    ),
    "postpaid_records": frozenset({"postpaid_record_id", "sequence"}),
    "tax_arrears_records": frozenset({"tax_arrears_id", "sequence"}),
    "civil_judgment_records": frozenset({"civil_judgment_id", "sequence"}),
    "enforcement_records": frozenset({"enforcement_record_id", "sequence"}),
    "administrative_penalty_records": frozenset(
        {
            "administrative_penalty_id",
            "sequence",
        }
    ),
    "institution_statement_records": frozenset({"institution_statement_id", "sequence"}),
    "inquiry_records": frozenset({"inquiry_id", "sequence", "source_reason"}),
}

_PUBLIC_RECORD_DATASETS = {
    "tax_arrears": "tax_arrears_records",
    "civil_judgment": "civil_judgment_records",
    "enforcement": "enforcement_records",
    "administrative_penalty": "administrative_penalty_records",
}

_SECTION_BUSINESS_ITEM_KEYS = frozenset({"record_status", "lookback_years", "source_statement"})

_COLUMN_OPTIONAL_BUSINESS_KEYS = (
    "unit",
    "format",
    "definition",
    "enum",
    "sensitive",
    "display",
)

_PUBLIC_COLUMN_TYPES = {
    "amount": "money",
    "currency": "string",
    "float": "decimal",
    "int": "integer",
    "long_id": "string",
    "number": "decimal",
}


@lru_cache(maxsize=1)
def _personal_brief_public_dataset_policy_template(
) -> Mapping[str, tuple[str, ...] | None]:
    """Validate and cache the package-private closed-world contract."""

    expected = list(personal_brief_semantic_extensions()["dataset_document_order"])
    policy: dict[str, tuple[str, ...] | None] = {
        **_PUBLIC_BUSINESS_FIELDS,
        **{name: None for name in _OMITTED_PUBLIC_DATASETS},
    }
    missing = [name for name in expected if name not in policy]
    extra = [name for name in policy if name not in expected]
    if missing or extra:
        raise RuntimeError(f"personal-brief public dataset policy is out of sync: missing={missing}, extra={extra}")

    dictionary = _personal_brief_dataset_descriptors()
    declared_enum_fields: set[tuple[str, str]] = set()
    declared_money_fields: set[tuple[str, str]] = set()
    for dataset_name, fields in policy.items():
        if fields is None:
            continue
        columns = (dictionary.get(dataset_name) or {}).get("columns") or {}
        schema_fields = set(columns)
        classified = set(fields) | set(_NON_BUSINESS_FIELDS.get(dataset_name, ()))
        unclassified = sorted(schema_fields - classified)
        unknown = sorted(classified - schema_fields)
        if unclassified or unknown:
            raise RuntimeError(
                f"personal-brief field policy is out of sync for {dataset_name}: "
                f"unclassified={unclassified}, unknown={unknown}"
            )
        declared_enum_fields.update(
            (dataset_name, field_name)
            for field_name in fields
            if str((columns.get(field_name) or {}).get("type") or "") == "enum"
        )
        declared_money_fields.update(
            (dataset_name, field_name)
            for field_name in fields
            if str((columns.get(field_name) or {}).get("type") or "")
            in {"amount", "money"}
        )
    contract_fields = set(PERSONAL_BRIEF_ENUM_CONTRACT)
    missing_enum_contracts = sorted(declared_enum_fields - contract_fields)
    extra_enum_contracts = sorted(contract_fields - declared_enum_fields)
    if missing_enum_contracts or extra_enum_contracts:
        raise RuntimeError(
            "personal-brief enum contract is out of sync: "
            f"missing={missing_enum_contracts}, extra={extra_enum_contracts}"
        )
    for dataset_name, field_name in sorted(declared_enum_fields):
        published = (dictionary[dataset_name]["columns"][field_name].get("enum") or {})
        expected_labels = PERSONAL_BRIEF_ENUM_CONTRACT[(dataset_name, field_name)]
        if published != expected_labels:
            raise RuntimeError(
                "personal-brief enum metadata is out of sync for "
                f"{dataset_name}.{field_name}: published={published!r}, "
                f"expected={expected_labels!r}"
            )
    contract_money_fields = {
        (dataset_name, field_name)
        for dataset_name, field_names in PERSONAL_BRIEF_MONEY_FIELDS.items()
        for field_name in field_names
    }
    missing_money_contracts = sorted(declared_money_fields - contract_money_fields)
    extra_money_contracts = sorted(contract_money_fields - declared_money_fields)
    if missing_money_contracts or extra_money_contracts:
        raise RuntimeError(
            "personal-brief money contract is out of sync: "
            f"missing={missing_money_contracts}, extra={extra_money_contracts}"
        )
    for dataset_name, field_name in sorted(declared_money_fields):
        unit = dictionary[dataset_name]["columns"][field_name].get("unit")
        if unit != PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT:
            raise RuntimeError(
                "personal-brief money-unit metadata is out of sync for "
                f"{dataset_name}.{field_name}: unit={unit!r}, "
                f"expected={PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT!r}"
            )
    return MappingProxyType({name: policy[name] for name in expected})


def personal_brief_public_dataset_policy() -> dict[str, tuple[str, ...] | None]:
    """Return an isolated copy of the validated Community JSON contract."""

    return dict(_personal_brief_public_dataset_policy_template())


def _has_business_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _require_equivalent_alias(
    dataset_name: str,
    values: dict[str, Any],
    alias: str,
    canonical: str,
) -> None:
    alias_value = values.get(alias)
    if not _has_business_value(alias_value):
        return
    canonical_value = values.get(canonical)
    if alias_value != canonical_value:
        raise ValueError(
            f"personal-brief field {dataset_name}.{alias} is not conserved by "
            f"{canonical}: {alias_value!r} != {canonical_value!r}"
        )


def _validate_discarded_aliases(dataset_name: str, values: dict[str, Any]) -> None:
    """Fail instead of silently dropping a non-equivalent compatibility value."""

    if dataset_name == "personal_report_metadata":
        amount_policy = values.get("amount_policy_source")
        if _has_business_value(amount_policy) and amount_policy != "personal_brief_standard":
            raise ValueError(f"unexpected personal-brief amount policy: {amount_policy!r}")
        return
    if dataset_name == "personal_credit_summary_records":
        summary_scope = values.get("summary_scope")
        if _has_business_value(summary_scope) and summary_scope != "source_reported":
            raise ValueError(f"unexpected personal-brief summary scope: {summary_scope!r}")
        return
    if dataset_name == "credit_accounts":
        for alias, canonical in (
            ("due_date", "contract_maturity_date"),
            ("close_date", "termination_event_date"),
            ("transfer_out_date", "termination_event_date"),
            ("currency", "account_currency"),
            ("amount_unit", "reporting_amount_unit"),
            ("source_section", "business_category"),
            ("source_sequence", "sequence"),
        ):
            _require_equivalent_alias(dataset_name, values, alias, canonical)
        precision = values.get("reporting_amount_precision")
        if _has_business_value(precision) and precision != 0:
            raise ValueError(f"unexpected personal-brief account amount precision: {precision!r}")
        lifecycle = str(values.get("account_lifecycle_state") or "")
        expected_account_state = (
            "open"
            if lifecycle == "open"
            else "closed"
            if lifecycle in {"closed", "settled", "transferred_out"}
            else "unknown"
        )
        account_state = values.get("account_state")
        if _has_business_value(account_state) and account_state != expected_account_state:
            raise ValueError(
                "personal-brief account_state is not conserved by account_lifecycle_state: "
                f"{account_state!r} != {expected_account_state!r}"
            )
        expected_activation = {
            "activated": "active",
            "not_activated": "inactive",
            "not_reported": "not_reported",
            "not_applicable": "not_applicable",
        }.get(str(values.get("card_activation_state") or ""), "not_reported")
        activation = values.get("activation_state")
        if _has_business_value(activation) and activation != expected_activation:
            raise ValueError(
                "personal-brief activation_state is not conserved by card_activation_state: "
                f"{activation!r} != {expected_activation!r}"
            )
        status = str(values.get("status") or "")
        expected_lifecycle = {
            "active": "open",
            "inactive": "open",
            "closed": "closed",
            "settled": "settled",
            "transferred_out": "transferred_out",
            "unknown": "unknown",
        }.get(status)
        if status and expected_lifecycle is None:
            raise ValueError(f"unclassified personal-brief deprecated account status: {status!r}")
        if status and lifecycle != expected_lifecycle:
            raise ValueError(
                "personal-brief deprecated status is not conserved by account_lifecycle_state: "
                f"{status!r} -> {expected_lifecycle!r}, got {lifecycle!r}"
            )
        return
    if dataset_name == "repayment_liability_records":
        _require_equivalent_alias(dataset_name, values, "currency", "reporting_amount_currency")
        _require_equivalent_alias(dataset_name, values, "amount_unit", "reporting_amount_unit")
        return
    if dataset_name == "inquiry_records":
        source_reason = values.get("source_reason")
        if not _has_business_value(source_reason):
            return
        reason = str(values.get("reason") or "")
        channel = str(values.get("query_channel") or "")
        equivalents = {reason}
        if channel:
            equivalents.update({f"{reason}（{channel}）", f"{reason}({channel})"})
        if source_reason not in equivalents:
            raise ValueError(
                "personal-brief source_reason is not conserved by reason/query_channel: "
                f"{source_reason!r} not in {sorted(equivalents)!r}"
            )


def _project_business_columns(
    dataset: dict[str, Any],
    field_order: tuple[str, ...],
    descriptors: Mapping[str, Any],
) -> list[dict[str, Any]]:
    originals = {
        str(column.get("key") or ""): column
        for column in dataset.get("columns") or []
        if isinstance(column, dict) and column.get("key")
    }
    columns: list[dict[str, Any]] = []
    for key in field_order:
        descriptor = descriptors.get(key) if isinstance(descriptors.get(key), dict) else {}
        original = originals.get(key) or descriptor
        declared_type = str(descriptor.get("type") or original.get("type") or "string")
        column = {
            "key": key,
            "label": str(descriptor.get("label") or original.get("label") or key),
            "type": _PUBLIC_COLUMN_TYPES.get(declared_type, declared_type),
            # The field list is the fixed canonical contract.  Personal-brief
            # reports are subsets, so any business field can be absent from a
            # particular source row without disappearing from the schema.
            "nullable": True,
            "raw_available": False,
            "evidence_available": False,
        }
        for optional_key in _COLUMN_OPTIONAL_BUSINESS_KEYS:
            if optional_key in descriptor:
                column[optional_key] = deepcopy(descriptor[optional_key])
            elif optional_key in original:
                column[optional_key] = deepcopy(original[optional_key])
        columns.append(column)
    return columns


def _project_business_dataset(
    dataset: dict[str, Any],
    field_order: tuple[str, ...],
    descriptors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = str(dataset.get("name") or "")
    if descriptors is None:
        dataset_descriptors = _personal_brief_dataset_descriptors().get(dataset_name) or {}
        descriptors = dataset_descriptors.get("columns") or {}
    allowed = set(field_order)
    classified = allowed | set(_NON_BUSINESS_FIELDS.get(dataset_name, ()))
    rows: list[dict[str, Any]] = []
    for row in dataset.get("rows") or []:
        if not isinstance(row, dict):
            continue
        source_normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
        unknown = sorted(set(source_normalized) - classified)
        if unknown:
            raise ValueError(f"unclassified personal-brief Community fields in {dataset_name}: {unknown}")
        _validate_discarded_aliases(dataset_name, source_normalized)
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"missing personal-brief record_id in {dataset_name}")
        validate_personal_brief_public_record(
            dataset_name,
            record_id,
            source_normalized,
        )
        rows.append(
            {
                "record_id": record_id,
                "normalized": {
                    key: deepcopy(source_normalized[key])
                    for key in field_order
                    if key in source_normalized and _has_business_value(source_normalized[key])
                },
                # Community v3 requires these record pools.  The rich semantic
                # payload remains the evidence/raw source; the downstream JSON
                # view deliberately avoids repeating it for every row.
                "canonical_raw": {},
                "raw": {},
                "source": {},
            }
        )

    completeness = deepcopy(dataset.get("completeness") if isinstance(dataset.get("completeness"), dict) else {})
    emitted = len(rows)
    raw_expected = completeness.get("expected_row_count")
    expected = (
        int(raw_expected) if isinstance(raw_expected, (int, float)) and not isinstance(raw_expected, bool) else emitted
    )
    completeness.update(
        {
            "expected_row_count": expected,
            "emitted_row_count": emitted,
            "omitted_row_count": max(expected - emitted, 0),
            "verified": bool(completeness.get("verified", expected == emitted)) and expected == emitted,
            "basis": str(completeness.get("basis") or "canonical_dataset"),
        }
    )
    projected = {
        key: deepcopy(dataset.get(key))
        for key in (
            "id",
            "name",
            "label",
            "type",
            "section_id",
            "csv",
            "grain",
            "schema_version",
        )
    }
    projected.update(
        {
            "row_count": emitted,
            "primary_key": "record_id",
            "status": str(dataset.get("status") or ("complete" if rows else "empty")),
            "columns": _project_business_columns(dataset, field_order, descriptors),
            "completeness": completeness,
            "rows": rows,
        }
    )
    return projected


def _validate_public_record_aggregate(payload: dict[str, Any]) -> None:
    datasets = {
        str(dataset.get("name") or ""): dataset
        for dataset in payload.get("datasets") or []
        if isinstance(dataset, dict)
    }
    if (datasets.get("repayment_records") or {}).get("rows"):
        raise ValueError("personal-brief Community JSON cannot contain monthly repayment records")
    aggregate = datasets.get("public_records")
    if not aggregate:
        return
    aggregate_counts: dict[str, int] = {}
    for row in aggregate.get("rows") or []:
        normalized = row.get("normalized") if isinstance(row, dict) else {}
        record_type = str((normalized or {}).get("record_type") or "")
        if record_type not in _PUBLIC_RECORD_DATASETS:
            raise ValueError(f"unclassified personal-brief aggregate public record type: {record_type!r}")
        aggregate_counts[record_type] = aggregate_counts.get(record_type, 0) + 1
    for record_type, dataset_name in _PUBLIC_RECORD_DATASETS.items():
        typed_count = len((datasets.get(dataset_name) or {}).get("rows") or [])
        if aggregate_counts.get(record_type, 0) != typed_count:
            raise ValueError(
                "personal-brief aggregate public records are not conserved by typed datasets: "
                f"{record_type} aggregate={aggregate_counts.get(record_type, 0)} "
                f"typed={typed_count}"
            )


def _clean_public_navigation(payload: dict[str, Any]) -> None:
    datasets = {
        str(dataset.get("id") or ""): dataset
        for dataset in payload.get("datasets") or []
        if isinstance(dataset, dict) and dataset.get("id")
    }
    dataset_ids = set(datasets)
    sections: list[dict[str, Any]] = []
    for source_section in payload.get("sections") or []:
        if not isinstance(source_section, dict):
            continue
        items = [
            {key: deepcopy(value) for key, value in item.items() if not str(key).startswith("_")}
            for item in source_section.get("items") or []
            if isinstance(item, dict) and str(item.get("key") or "") in _SECTION_BUSINESS_ITEM_KEYS
        ]
        refs = [str(ref) for ref in source_section.get("dataset_refs") or [] if str(ref) in dataset_ids]
        if not refs and not items:
            continue
        sections.append(
            {
                "id": str(source_section.get("id") or ""),
                "title": str(source_section.get("title") or ""),
                "type": str(source_section.get("type") or "generic"),
                "page_range": deepcopy(source_section.get("page_range") or []),
                "items": items,
                "groups": [],
                "dataset_refs": refs,
            }
        )
    payload["sections"] = sections
    section_ids = {str(section["id"]) for section in sections}

    reading = deepcopy(payload.get("reading") if isinstance(payload.get("reading"), dict) else {})
    tables: list[dict[str, Any]] = []
    for source_table in reading.get("tables") or []:
        if not isinstance(source_table, dict):
            continue
        dataset_id = str(source_table.get("dataset_id") or "")
        if dataset_id not in datasets:
            continue
        dataset = datasets[dataset_id]
        table = deepcopy(source_table)
        table["column_keys"] = [
            str(column.get("key") or "")
            for column in dataset.get("columns") or []
            if isinstance(column, dict) and column.get("key")
        ]
        table["row_count"] = int(dataset.get("row_count") or 0)
        tables.append(table)
    reading["tables"] = tables

    flow: list[dict[str, Any]] = []
    for item in reading.get("document_flow") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        ref_id = str(item.get("ref_id") or "")
        if kind == "dataset" and ref_id not in dataset_ids:
            continue
        if kind == "section" and ref_id not in section_ids:
            continue
        flow.append({**deepcopy(item), "order": len(flow) + 1})
    reading["document_flow"] = flow
    payload["reading"] = reading


def project_personal_brief_community_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Project rich personal-brief semantic datasets to lean business JSON."""

    projected = deepcopy(payload)
    _validate_public_record_aggregate(projected)
    policy = _personal_brief_public_dataset_policy_template()
    descriptors = _personal_brief_dataset_descriptors()
    datasets: list[dict[str, Any]] = []
    for dataset in projected.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        dataset_name = str(dataset.get("name") or "")
        if dataset_name not in policy:
            raise ValueError(f"unclassified personal-brief Community dataset: {dataset_name}")
        fields = policy[dataset_name]
        if fields is None:
            continue
        dataset_descriptors = descriptors.get(dataset_name) or {}
        columns = dataset_descriptors.get("columns") or {}
        datasets.append(_project_business_dataset(dataset, fields, columns))
    projected["datasets"] = datasets
    _clean_public_navigation(projected)
    return projected


def project_personal_brief_artifact_semantic(
    semantic: dict[str, Any],
    public_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a transient public-dataset audit view with rich provenance."""

    artifact = deepcopy(semantic)
    rich_datasets = {
        str(dataset.get("name") or ""): dataset
        for dataset in semantic.get("datasets") or []
        if isinstance(dataset, dict)
    }
    artifact_datasets = deepcopy(public_payload.get("datasets") or [])
    policy = _personal_brief_public_dataset_policy_template()
    for dataset in artifact_datasets:
        dataset_name = str(dataset.get("name") or "")
        field_order = policy.get(dataset_name) or ()
        rich_dataset = rich_datasets.get(dataset_name) or {}
        rich_rows = {
            str(row.get("record_id") or ""): row for row in rich_dataset.get("rows") or [] if isinstance(row, dict)
        }
        for row in dataset.get("rows") or []:
            if not isinstance(row, dict):
                continue
            rich_row = rich_rows.get(str(row.get("record_id") or "")) or {}
            for pool_name in ("canonical_raw", "raw"):
                rich_pool = rich_row.get(pool_name) if isinstance(rich_row.get(pool_name), dict) else {}
                row[pool_name] = {
                    key: deepcopy(rich_pool[key])
                    for key in field_order
                    if key in rich_pool and _has_business_value(rich_pool[key])
                }
            row["source"] = deepcopy(rich_row.get("source") if isinstance(rich_row.get("source"), dict) else {})
            if rich_row.get("confidence") not in (None, ""):
                row["confidence"] = deepcopy(rich_row["confidence"])

        for column in dataset.get("columns") or []:
            if not isinstance(column, dict):
                continue
            key = str(column.get("key") or "")
            column["raw_available"] = any(
                key in (row.get("canonical_raw") or {}) or key in (row.get("raw") or {})
                for row in dataset.get("rows") or []
                if isinstance(row, dict)
            )
            column["evidence_available"] = any(
                bool((row.get("source") or {}).get("source_refs"))
                or bool((row.get("source") or {}).get("evidence_ids"))
                for row in dataset.get("rows") or []
                if isinstance(row, dict)
            )
    artifact["datasets"] = artifact_datasets
    structure = artifact.get("structure") if isinstance(artifact.get("structure"), dict) else {}
    structure["sections"] = deepcopy(public_payload.get("sections") or [])
    artifact["structure"] = structure
    artifact["reading"] = deepcopy(public_payload.get("reading") or {})
    retained_rows = {
        (str(dataset.get("id") or ""), str(row.get("record_id") or ""))
        for dataset in artifact_datasets
        if isinstance(dataset, dict)
        for row in dataset.get("rows") or []
        if isinstance(row, dict)
    }
    artifact["bindings"] = [
        deepcopy(binding)
        for binding in semantic.get("bindings") or []
        if isinstance(binding, dict)
        and (str(binding.get("dataset_id") or ""), str(binding.get("record_id") or ""))
        in retained_rows
    ]
    return artifact


def derive_personal_brief_projection(
    plugin: Any,
    parse_result: Any,
    full_text: str = "",
) -> ProjectionData:
    """Run ParseResult -> canonical IR -> one rigid schema -> projection."""
    content_mode = detect_credit_report_content_mode(parse_result)
    artifacts = run_personal_brief_pipeline(parse_result, content_mode=content_mode)
    document = artifacts.document_ir
    semantic_document = artifacts.semantic_document
    from docmirror.plugins.credit_report.personal_brief_native.variant import variant

    raw_datasets = semantic_document.datasets
    # A source node may own one presentation section only.  Supplemental
    # datasets retain canonical unit/bbox/evidence provenance, while node
    # ownership is reserved for the authoritative presentation datasets.
    ownership_datasets = {
        "credit_accounts",
        "repayment_liability_records",
        "repayment_records",
        "overdue_records",
        "public_records",
        "inquiry_records",
        "report_notes",
    }
    for dataset_name, records in raw_datasets.items():
        if dataset_name in ownership_datasets:
            continue
        for record in records:
            for ref in record.get("source_refs") or []:
                if isinstance(ref, dict):
                    ref.pop("node_id", None)
                    ref.pop("node_ids", None)
    from docmirror.plugins.credit_report.business_assembly import (
        _NORMALIZED_FIELDS,
        _build_audit,
        _normalize_record,
    )

    datasets: dict[str, list[dict[str, Any]]] = {}
    for name, rows in raw_datasets.items():
        if not rows:
            continue
        if name in _NORMALIZED_FIELDS:
            normalized_rows: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                normalized = _normalize_record(name, row, index)
                normalized_payload = normalized.setdefault("normalized", {})
                if name == "public_records" and row.get("content") not in (None, ""):
                    # ``content`` is a typed nested business object.  The
                    # generic scalar normalizer intentionally unwraps dicts,
                    # so preserve this personal-brief contract explicitly.
                    normalized_payload["content"] = row["content"]
                if name == "credit_accounts":
                    for field in ("source_section", "source_sequence", "business_category"):
                        if row.get(field) not in (None, ""):
                            normalized_payload[field] = row[field]
                normalized_rows.append(normalized)
            datasets[name] = normalized_rows
        else:
            datasets[name] = _records(name, rows)
    audit = _build_audit(
        parse_result=document,
        full_text=document.full_text,
        report_subtype="personal_brief",
        content_mode=content_mode,
        collections={
            "credit_accounts": datasets.get("credit_accounts", []),
            "credit_lines": [],
            "repayment_liability_records": datasets.get("repayment_liability_records", []),
            "repayment_records": datasets.get("repayment_records", []),
            "overdue_records": datasets.get("overdue_records", []),
            "inquiry_records": datasets.get("inquiry_records", []),
            "public_records": datasets.get("public_records", []),
        },
        conflicts=[],
        credit_summary=semantic_document.credit_summary,
    )
    failures = semantic_document.extraction_report.get("failures") or []
    if failures:
        audit["issues"] = list(
            dict.fromkeys(
                [
                    *list(audit.get("issues") or []),
                    *[
                        f"canonical_extraction:{failure.get('code')}"
                        for failure in failures
                        if isinstance(failure, dict)
                    ],
                ]
            )
        )
        audit["status"] = "review"

    domain_facts = dict(semantic_document.facts)
    domain_facts["credit_summary"] = semantic_document.credit_summary
    domain_facts["credit_extraction_audit"] = audit
    domain_facts["personal_brief_extraction_report"] = semantic_document.extraction_report
    domain_facts["field_details"] = {
        key: {
            "source": "canonical_personal_brief_document_ir",
            "confidence": document.confidence,
        }
        for key, value in domain_facts.items()
        if key
        not in {
            "credit_summary",
            "credit_extraction_audit",
            "personal_brief_extraction_report",
            "canonical_section_presence",
        }
        and value not in (None, "")
    }
    domain_facts["data_dictionary"] = variant.data_dictionary()

    evidence_ids = tuple(
        dict.fromkeys(
            str(evidence_id)
            for rows in raw_datasets.values()
            for record in rows
            for evidence_id in (
                *list(record.get("evidence_ids") or []),
                *[
                    value
                    for ref in record.get("source_refs") or []
                    if isinstance(ref, dict)
                    for value in ref.get("evidence_ids") or []
                ],
            )
            if evidence_id
        )
    )
    accounts = datasets.get("credit_accounts") or []
    warnings = tuple(
        dict.fromkeys(
            [
                *list(getattr(getattr(parse_result, "parser_info", None), "warnings", None) or []),
                *_account_structure_warnings(accounts),
                *[
                    f"{failure.get('code', 'PERSONAL_BRIEF_EXTRACTION_FAILURE')}: {failure.get('message', '')}".strip()
                    for failure in failures
                    if isinstance(failure, dict)
                ],
            ]
        )
    )
    entity_fields: dict[str, Any] = {}
    for target, source in (
        ("subject_name", "subject_name"),
        ("subject_id", "id_number"),
        ("marital_status", "marital_status"),
    ):
        if domain_facts.get(source) not in (None, ""):
            entity_fields[target] = domain_facts[source]

    semantic = variant.semantic_extensions()
    overrides = semantic.setdefault("community_projection_overrides", {})
    completeness_policy = overrides.setdefault("completeness", {})
    internal_fields = overrides.setdefault("internal_fields", [])
    internal_facts = overrides.setdefault("internal_facts", [])
    for dataset_name, details in semantic_document.dataset_completeness.items():
        count_key = f"personal_brief_expected_{dataset_name}_count"
        domain_facts[count_key] = int(details.get("expected_row_count") or 0)
        if count_key not in internal_fields:
            internal_fields.append(count_key)
        if count_key not in internal_facts:
            internal_facts.append(count_key)
        completeness_policy[dataset_name] = {
            "basis": "domain_fact_count",
            "count_key": count_key,
            "public_basis": str(details.get("basis") or "canonical_source_component_count"),
        }
    semantic["personal_brief_dataset_completeness"] = semantic_document.dataset_completeness

    return ProjectionData(
        projector_id=plugin.projector_id,
        document_type=plugin.domain_name,
        entity_fields=entity_fields,
        domain_facts=domain_facts,
        semantic=semantic,
        datasets=datasets,
        sections=variant.build_sections(document, document.full_text),
        warnings=warnings,
        evidence_ids=tuple(evidence_ids),
        confidence=document.confidence,
        reason="canonical personal brief IR schema projection",
    )


__all__ = [
    "derive_personal_brief_projection",
    "personal_brief_public_dataset_policy",
    "project_personal_brief_artifact_semantic",
    "project_personal_brief_community_json",
]
