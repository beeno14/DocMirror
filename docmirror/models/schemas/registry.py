# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Projection schema registry for Mirror and Edition JSON contracts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "schemas"


@dataclass(frozen=True)
class ProjectionSchemaSpec:
    """Registered projection output schema."""

    name: str
    path: Path
    version: str
    description: str = ""
    compatibility: str = ""


@dataclass(frozen=True)
class ProjectionSchemaValidation:
    """Result of validating a projection payload against its schema."""

    name: str
    valid: bool
    errors: tuple[str, ...] = ()


def _builtin_specs() -> dict[str, ProjectionSchemaSpec]:
    specs = (
        ProjectionSchemaSpec(
            name="mirror",
            path=_SCHEMAS_DIR / "mirror.schema.json",
            version="1.1",
            description="Core Mirror JSON vNext",
        ),
        ProjectionSchemaSpec(
            name="community",
            path=_SCHEMAS_DIR / "community_bundle.schema.json",
            version="3.0.0",
            description="Self-contained Community JSON API with complete dataset records",
            compatibility="current-major; explicit-v2-exporter-required",
        ),
        ProjectionSchemaSpec(
            name="community_semantic",
            path=_SCHEMAS_DIR / "community_semantic.schema.json",
            version="1.0.0",
            description="Public post-seal semantic source for all Community renderers",
            compatibility="additive-community-companion",
        ),
        ProjectionSchemaSpec(
            name="personal_credit_report_detailed",
            path=_SCHEMAS_DIR / "personal_credit_report_detailed.schema.json",
            version="1.2.0",
            description="Canonical dataset contract for individual detailed credit reports",
            compatibility="community-v3-domain-profile",
        ),
        ProjectionSchemaSpec(
            name="personal_credit_report_detailed_v2",
            path=_SCHEMAS_DIR / "personal_credit_report_detailed_v2.schema.json",
            version="2.0.0",
            description="PBOC-native contract for individual detailed credit reports",
            compatibility="breaking-from-1.2; community-v3-domain-profile; detailed-report-only",
        ),
        ProjectionSchemaSpec(
            name="community_v2",
            path=_SCHEMAS_DIR / "edition_community.schema.json",
            version="2.2",
            description="Internal Community candidate retained for extended-edition fallback",
            compatibility="internal-only",
        ),
        ProjectionSchemaSpec(
            name="enterprise",
            path=_SCHEMAS_DIR / "edition_enterprise.schema.json",
            version="2.0",
            description="Enterprise edition envelope (DEC v2 + governance)",
        ),
        ProjectionSchemaSpec(
            name="finance",
            path=_SCHEMAS_DIR / "edition_finance.schema.json",
            version="3.0",
            description="Finance edition envelope (DEC v3)",
        ),
    )
    return {spec.name: spec for spec in specs}


_registry: dict[str, ProjectionSchemaSpec] = _builtin_specs()


def load_projection_registry() -> dict[str, ProjectionSchemaSpec]:
    """Return projection schema registry (built-in + registered extensions)."""
    return dict(_registry)


def get_projection_schema(name: str) -> ProjectionSchemaSpec | None:
    return load_projection_registry().get(name)


def load_projection_schema_json(name: str) -> dict[str, Any] | None:
    spec = get_projection_schema(name)
    if spec is None or not spec.path.is_file():
        return None
    with open(spec.path, encoding="utf-8") as f:
        return json.load(f)


def register_projection_schema(spec: ProjectionSchemaSpec) -> None:
    """Register a third-party or tenant-specific projection schema."""
    _registry[spec.name] = spec


def _personal_detail_invariant_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    """Check cross-dataset rules that JSON Schema cannot express safely."""
    datasets = [item for item in payload.get("datasets", []) if isinstance(item, dict)]
    names = [str(dataset.get("name") or "") for dataset in datasets]
    errors: list[str] = []
    if len(names) != len(set(names)):
        errors.append("datasets must have unique names")
    by_name = {str(dataset.get("name") or ""): dataset for dataset in datasets}
    for dataset_name, dataset in by_name.items():
        rows = [row for row in dataset.get("rows", []) if isinstance(row, dict)]
        if dataset.get("row_count") != len(rows):
            errors.append(f"{dataset_name}: row_count does not match rows length")
        record_ids = [str(row.get("record_id") or "") for row in rows]
        if len(record_ids) != len(set(record_ids)):
            errors.append(f"{dataset_name}: record_id values must be unique")

    contract = (payload.get("document") or {}).get("domain_schema") or {}
    if str(contract.get("id") or "") != "personal_credit_report_detailed":
        return tuple(errors)
    for required_name in (
        "personal_profile",
        "personal_detail_field_observations",
        "personal_detail_dataset_status",
    ):
        if required_name not in by_name:
            errors.append(f"missing personal-detail contract dataset: {required_name}")

    profile_rows = (by_name.get("personal_profile") or {}).get("rows") or []
    if profile_rows:
        birth_date = (profile_rows[0].get("normalized") or {}).get("birth_date")
        if birth_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(birth_date)):
            errors.append("personal_profile.birth_date must use ISO YYYY-MM-DD")

    observation_rows = (by_name.get("personal_detail_field_observations") or {}).get("rows") or []
    for row in observation_rows:
        normalized = row.get("normalized") or {}
        if normalized.get("confidence_status") not in {"available", "not_available"}:
            errors.append("personal_detail_field_observations: confidence_status is required")
        if not normalized.get("confidence_basis"):
            errors.append("personal_detail_field_observations: confidence_basis is required")

    metric_rows = (by_name.get("personal_detail_credit_summary_metrics") or {}).get("rows") or []
    for row in metric_rows:
        normalized = row.get("normalized") or {}
        if normalized.get("mapping_status") not in {"mapped", "unmapped"}:
            errors.append("personal_detail_credit_summary_metrics: mapping_status is required")
        if normalized.get("value_type") == "placeholder" and normalized.get("text_value") not in {
            None,
            "",
        }:
            errors.append("personal_detail_credit_summary_metrics: placeholder cannot be text_value")
        if normalized.get("value_type") == "money" and (
            normalized.get("currency") != "CNY" or normalized.get("amount_unit") != "yuan"
        ):
            errors.append("personal_detail_credit_summary_metrics: money requires CNY/yuan")

    statuses = {
        str((row.get("normalized") or {}).get("dataset_name") or ""): row.get("normalized") or {}
        for row in (by_name.get("personal_detail_dataset_status") or {}).get("rows") or []
        if isinstance(row, dict)
    }
    for dataset_name, status in statuses.items():
        target = by_name.get(dataset_name)
        actual_count = len((target or {}).get("rows") or [])
        if int(status.get("observed_row_count") or 0) != actual_count:
            errors.append(f"{dataset_name}: dataset-status observed_row_count mismatch")
        if actual_count and status.get("presence_status") != "observed_nonempty":
            errors.append(f"{dataset_name}: nonempty dataset must be observed_nonempty")

    for dataset_name, dataset in by_name.items():
        for foreign_key in dataset.get("foreign_keys") or []:
            columns = list(foreign_key.get("columns") or [])
            reference_columns = list(foreign_key.get("reference_columns") or [])
            reference_dataset = by_name.get(str(foreign_key.get("reference_dataset") or ""))
            if not columns or len(columns) != len(reference_columns) or reference_dataset is None:
                errors.append(f"{dataset_name}: invalid foreign-key declaration")
                continue
            reference_values = set()
            for reference_row in reference_dataset.get("rows") or []:
                normalized = reference_row.get("normalized") or {}
                reference_values.add(
                    tuple(
                        reference_row.get(column) if column == "record_id" else normalized.get(column)
                        for column in reference_columns
                    )
                )
            for row in dataset.get("rows") or []:
                normalized = row.get("normalized") or {}
                value = tuple(normalized.get(column) for column in columns)
                if any(part not in (None, "") for part in value) and value not in reference_values:
                    errors.append(f"{dataset_name}: unresolved foreign key {columns}={value}")
    return tuple(errors)


def _personal_detail_v2_invariant_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    """Check PBOC v2 relationships and semantic discriminators."""
    datasets = [item for item in payload.get("datasets", []) if isinstance(item, dict)]
    names = [str(dataset.get("name") or "") for dataset in datasets]
    errors: list[str] = []
    if len(names) != len(set(names)):
        errors.append("datasets must have unique names")
    by_name = {str(dataset.get("name") or ""): dataset for dataset in datasets}
    for required_name in ("report_metadata", "report_query", "subject_profile"):
        if required_name not in by_name:
            errors.append(f"missing PBOC v2 contract dataset: {required_name}")
    for dataset_name, dataset in by_name.items():
        rows = [row for row in dataset.get("rows", []) if isinstance(row, dict)]
        if dataset.get("row_count") != len(rows):
            errors.append(f"{dataset_name}: row_count does not match rows length")
        record_ids = [str(row.get("record_id") or "") for row in rows]
        if len(record_ids) != len(set(record_ids)):
            errors.append(f"{dataset_name}: record_id values must be unique")

    accounts = {
        str((row.get("normalized") or {}).get("account_id") or ""): row.get("normalized") or {}
        for row in (by_name.get("credit_accounts") or {}).get("rows") or []
        if isinstance(row, dict)
    }
    for row in (by_name.get("credit_account_monthly_performance") or {}).get("rows") or []:
        normalized = row.get("normalized") or {}
        account_id = str(normalized.get("account_id") or "")
        account = accounts.get(account_id)
        if account is None:
            errors.append(
                "credit_account_monthly_performance: unresolved account_id=" f"{account_id}"
            )
            continue
        expected_semantics = (
            "overdraft_balance"
            if account.get("pboc_account_type_code") == "R4"
            else "delinquent_amount"
        )
        if normalized.get("status_amount") not in (None, "") and normalized.get(
            "status_amount_semantics"
        ) != expected_semantics:
            errors.append(
                "credit_account_monthly_performance: status_amount_semantics conflicts "
                f"with {account.get('pboc_account_type_code')}"
            )

    for row in (by_name.get("repayment_responsibilities") or {}).get("rows") or []:
        normalized = row.get("normalized") or {}
        if "overdue_months_or_repayment_status" in normalized:
            errors.append(
                "repayment_responsibilities: combined overdue/status field is forbidden"
            )
        if normalized.get("related_party_category") == "person" and normalized.get(
            "source_status_value"
        ) not in (None, "") and normalized.get("overdue_months") is None and not normalized.get(
            "repayment_status_code"
        ):
            errors.append(
                "repayment_responsibilities: person row requires separated overdue months or status"
            )
        if normalized.get("related_party_category") == "organization" and not normalized.get(
            "repayment_status_code"
        ):
            errors.append(
                "repayment_responsibilities: organization row requires repayment_status_code"
            )

    for dataset_name, dataset in by_name.items():
        for foreign_key in dataset.get("foreign_keys") or []:
            columns = list(foreign_key.get("columns") or [])
            reference_columns = list(foreign_key.get("reference_columns") or [])
            reference_dataset = by_name.get(str(foreign_key.get("reference_dataset") or ""))
            if not columns or len(columns) != len(reference_columns) or reference_dataset is None:
                errors.append(f"{dataset_name}: invalid foreign-key declaration")
                continue
            reference_values = {
                tuple(
                    reference_row.get(column)
                    if column == "record_id"
                    else (reference_row.get("normalized") or {}).get(column)
                    for column in reference_columns
                )
                for reference_row in reference_dataset.get("rows") or []
            }
            for row in dataset.get("rows") or []:
                value = tuple((row.get("normalized") or {}).get(column) for column in columns)
                if any(part not in (None, "") for part in value) and value not in reference_values:
                    errors.append(f"{dataset_name}: unresolved foreign key {columns}={value}")
    return tuple(errors)


def validate_projection_payload(name: str, payload: dict[str, Any]) -> ProjectionSchemaValidation:
    """Validate a projection payload against the registered JSON schema.

    Full JSON Schema validation is used when ``jsonschema`` is installed. In
    minimal environments, this still verifies that the schema exists and all
    top-level required keys are present.
    """
    schema = load_projection_schema_json(name)
    if schema is None:
        return ProjectionSchemaValidation(name=name, valid=False, errors=(f"schema not found: {name}",))
    try:
        import jsonschema

        jsonschema.validate(instance=payload, schema=schema)
        if name == "personal_credit_report_detailed":
            errors = _personal_detail_invariant_errors(payload)
        elif name == "personal_credit_report_detailed_v2":
            errors = _personal_detail_v2_invariant_errors(payload)
        else:
            errors = ()
        return ProjectionSchemaValidation(name=name, valid=not errors, errors=errors)
    except ImportError:
        missing = [key for key in schema.get("required", []) if key not in payload]
        errors = list(f"missing required key: {key}" for key in missing)
        if not errors:
            if name == "personal_credit_report_detailed":
                errors.extend(_personal_detail_invariant_errors(payload))
            elif name == "personal_credit_report_detailed_v2":
                errors.extend(_personal_detail_v2_invariant_errors(payload))
        return ProjectionSchemaValidation(
            name=name,
            valid=not errors,
            errors=tuple(errors),
        )
    except Exception as exc:
        return ProjectionSchemaValidation(name=name, valid=False, errors=(str(exc),))


def projection_schema_manifest() -> dict[str, dict[str, str]]:
    """Return explicit schema identities for artifact/task manifests."""
    return {
        name: {
            "version": spec.version,
            "compatibility": spec.compatibility or "same-major",
        }
        for name, spec in sorted(load_projection_registry().items())
        if name != "community_v2"
    }


__all__ = [
    "ProjectionSchemaSpec",
    "ProjectionSchemaValidation",
    "get_projection_schema",
    "load_projection_registry",
    "load_projection_schema_json",
    "register_projection_schema",
    "projection_schema_manifest",
    "validate_projection_payload",
]
