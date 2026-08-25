# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conserve exact source-bound failures across the private-to-v2 projection.

This module never discovers source values.  It only turns already-published,
field-local evidence into a diagnostic for the canonical record that also owns
that field.  Ambiguous headers, table-level anchors, and endpoint-only counts
remain dataset-scoped diagnostics elsewhere in the pipeline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    _SEQUENCE_POPULATION_DATASETS,
    _exact_raw_profile_observations,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id

_PROFILE_METADATA_FIELDS = frozenset(
    {"subject_name", "primary_id_type", "primary_id_number"}
)
_PROFILE_ID = "personal_profile:1"
_ACTIVE_ISSUE_STATUSES = frozenset(
    {"active", "open", "requires_review", "review", "warning", "error"}
)
_NATIVE_HEADER_BINDING_PAIRS = frozenset(
    {
        ("canonical_header_column", "canonical_header_column"),
        ("canonical_field_slot", "canonical_header_column"),
        ("canonical_field_slot", "canonical_field_slot"),
    }
)
_CORRECTED_HEADER_BINDING_PAIR = (
    "canonical_field_slot",
    "canonical_cell_slot",
)
_PROFILE_TEMPLATE_ID = "report_header_and_identity"
_CANONICAL_AUDIT_KEYS = (
    "_personal_detail_canonical_layout_owner_census",
    "personal_detail_canonical_layout_audit",
    "canonical_layout",
)


def _values(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    normalized = record.get("normalized")
    return dict(normalized) if isinstance(normalized, Mapping) else dict(record)


def _source_refs(record: Any) -> list[dict[str, Any]]:
    if not isinstance(record, Mapping):
        return []
    direct = record.get("source_refs")
    if not isinstance(direct, (list, tuple)):
        source = record.get("source")
        direct = source.get("source_refs") if isinstance(source, Mapping) else ()
    return [deepcopy(dict(ref)) for ref in direct or () if isinstance(ref, Mapping)]


def _dedupe_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for ref in refs:
        marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
        unique.setdefault(marker, ref)
    return list(unique.values())


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _strict_positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _strict_nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _exact_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in value
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


def _strict_evidence_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        return None
    evidence_ids = tuple(value)
    return evidence_ids if len(evidence_ids) == len(set(evidence_ids)) else None


def _issue_is_active(values: Mapping[str, Any]) -> bool:
    status = str(values.get("status") or "requires_review").strip().lower()
    return status in _ACTIVE_ISSUE_STATUSES


def _field_missing(values: Mapping[str, Any], field_name: str) -> bool:
    return values.get(field_name) in (None, "")


def _metadata_identity(datasets: Mapping[str, Any]) -> str | None:
    rows = [row for row in datasets.get("personal_report_metadata") or () if isinstance(row, Mapping)]
    if len(rows) != 1:
        return None
    values = _values(rows[0])
    identities = {
        str(value)
        for value in (
            rows[0].get("record_id"),
            values.get("personal_report_metadata_id"),
            values.get("report_metadata_id"),
        )
        if value not in (None, "")
    }
    return next(iter(identities)) if len(identities) == 1 else None


def _profile_values(datasets: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in datasets.get("personal_profile") or () if isinstance(row, Mapping)]
    if len(rows) != 1:
        return {}
    values = _values(rows[0])
    identities = {
        str(value)
        for value in (
            rows[0].get("record_id"),
            values.get("personal_profile_id"),
            values.get("subject_profile_id"),
        )
        if value not in (None, "")
    }
    return values if identities == {_PROFILE_ID} else {}


def _canonical_layout_audit(
    facts: Mapping[str, Any], datasets: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Return one complete canonical-layout owner census, when published."""

    candidates: list[Mapping[str, Any]] = []
    for key in _CANONICAL_AUDIT_KEYS:
        candidate = facts.get(key)
        if isinstance(candidate, Mapping):
            candidates.append(candidate)
    extraction_audit = facts.get("credit_extraction_audit")
    if isinstance(extraction_audit, Mapping):
        candidate = extraction_audit.get("canonical_layout")
        if isinstance(candidate, Mapping):
            candidates.append(candidate)
    parser_audit = facts.get("personal_detail_parser_audit")
    if isinstance(parser_audit, Mapping):
        candidate = parser_audit.get("canonical_layout")
        if isinstance(candidate, Mapping):
            candidates.append(candidate)
    metadata_rows = [
        row
        for row in datasets.get("personal_report_metadata") or ()
        if isinstance(row, Mapping)
    ]
    if len(metadata_rows) == 1:
        for owner in (_values(metadata_rows[0]), metadata_rows[0]):
            candidate = owner.get("canonical_layout")
            if isinstance(candidate, Mapping):
                candidates.append(candidate)

    unique: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        # Both collections are required.  Fragment groups are the assembled
        # canonical source owners; registrations are the complete fragment-to-
        # template census.  A partial diagnostic is not ownership authority.
        if not isinstance(
            candidate.get("fragment_groups"), (list, tuple)
        ) or not isinstance(candidate.get("registrations"), (list, tuple)):
            return None
        marker = json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)
        unique.setdefault(marker, candidate)
    return next(iter(unique.values())) if len(unique) == 1 else None


def _positive_int_set(value: Any) -> frozenset[int]:
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(
        number
        for raw in value
        if (number := _positive_int(raw)) is not None
    )


def _profile_source_owners(
    facts: Mapping[str, Any], datasets: Mapping[str, Any]
) -> tuple[
    tuple[tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]], ...],
    tuple[Mapping[str, Any], ...],
] | None:
    """Resolve every exact canonical owner of the report-header template."""

    audit = _canonical_layout_audit(facts, datasets)
    if audit is None:
        return None
    raw_groups = audit.get("fragment_groups")
    raw_registrations = audit.get("registrations")
    assert isinstance(raw_groups, (list, tuple))
    assert isinstance(raw_registrations, (list, tuple))

    profile_groups: list[Mapping[str, Any]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            return None
        if str(raw_group.get("template_id") or "") != _PROFILE_TEMPLATE_ID:
            continue
        logical_pages = _positive_int_set(
            raw_group.get("fragment_logical_pages")
            or raw_group.get("logical_pages")
        )
        source_pages = _positive_int_set(raw_group.get("source_pages"))
        if not logical_pages or not source_pages:
            return None
        profile_groups.append(raw_group)
    if not profile_groups:
        return None

    registrations = tuple(
        raw_registration
        for raw_registration in raw_registrations
        if isinstance(raw_registration, Mapping)
        and str(raw_registration.get("template_id") or "") == _PROFILE_TEMPLATE_ID
    )
    if not registrations or any(
        str(registration.get("status") or "") != "registered"
        or _positive_int(registration.get("logical_page")) is None
        or _positive_int(registration.get("source_page")) is None
        for registration in registrations
    ):
        return None

    owners: list[
        tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]
    ] = []
    claimed_logical_pages: set[int] = set()
    for owner in profile_groups:
        owner_logical_pages = _positive_int_set(
            owner.get("fragment_logical_pages") or owner.get("logical_pages")
        )
        owner_source_pages = _positive_int_set(owner.get("source_pages"))
        if claimed_logical_pages & set(owner_logical_pages):
            return None
        claimed_logical_pages.update(owner_logical_pages)
        owner_registrations = tuple(
            registration
            for registration in registrations
            if _positive_int(registration.get("logical_page"))
            in owner_logical_pages
        )
        registered_pairs = tuple(
            (
                _positive_int(registration.get("logical_page")),
                _positive_int(registration.get("source_page")),
            )
            for registration in owner_registrations
        )
        if (
            not owner_registrations
            or len(set(registered_pairs)) != len(registered_pairs)
            or {logical for logical, _source in registered_pairs}
            != set(owner_logical_pages)
            or {source for _logical, source in registered_pairs}
            != set(owner_source_pages)
        ):
            return None
        owners.append((owner, owner_registrations))

    if claimed_logical_pages != {
        int(registration["logical_page"]) for registration in registrations
    }:
        return None
    for raw_registration in raw_registrations:
        if not isinstance(raw_registration, Mapping):
            return None
        logical_page = _positive_int(raw_registration.get("logical_page"))
        if (
            logical_page in claimed_logical_pages
            and str(raw_registration.get("template_id") or "")
            != _PROFILE_TEMPLATE_ID
        ):
            return None
    return tuple(owners), tuple(
        raw_registration
        for raw_registration in raw_registrations
        if isinstance(raw_registration, Mapping)
    )


def _profile_owner_for_ref(
    ref: Mapping[str, Any],
    owners: tuple[
        tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]], ...
    ],
) -> int | None:
    logical_page = _positive_int(ref.get("logical_page"))
    source_page = _positive_int(ref.get("source_page"))
    if logical_page is None or source_page is None:
        return None
    matches: list[int] = []
    for index, (owner, registrations) in enumerate(owners):
        owner_logical_pages = _positive_int_set(
            owner.get("fragment_logical_pages") or owner.get("logical_pages")
        )
        owner_source_pages = _positive_int_set(owner.get("source_pages"))
        if (
            logical_page not in owner_logical_pages
            or source_page not in owner_source_pages
        ):
            continue
        registrations_at_ref = [
            registration
            for registration in registrations
            if _positive_int(registration.get("logical_page")) == logical_page
            and _positive_int(registration.get("source_page")) == source_page
        ]
        if len(registrations_at_ref) == 1:
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


def _exact_metadata_field_refs(
    facts: Mapping[str, Any],
    datasets: Mapping[str, Any],
    issue: Mapping[str, Any],
    field_name: str,
) -> list[dict[str, Any]]:
    """Keep refs bound to one exact field slot on the sole header owner."""

    owner_proof = _profile_source_owners(facts, datasets)
    if owner_proof is None:
        return []
    owners, _registrations = owner_proof

    refs_with_owners: list[
        tuple[
            dict[str, Any],
            int,
            tuple[
                int,
                int,
                tuple[float, float, float, float],
                tuple[str, ...],
                str,
            ],
        ]
    ] = []
    for raw_ref in _source_refs(issue):
        source = str(raw_ref.get("source") or "")
        ref_field = str(raw_ref.get("field_name") or "")
        logical_page = _strict_positive_int(raw_ref.get("logical_page"))
        source_page = _strict_positive_int(raw_ref.get("source_page"))
        bbox = _exact_bbox(raw_ref.get("bbox"))
        evidence_ids = _strict_evidence_ids(raw_ref.get("evidence_ids"))
        owner_index = _profile_owner_for_ref(raw_ref, owners)
        common_exact = bool(
            logical_page is not None
            and source_page is not None
            and bbox is not None
            and evidence_ids is not None
            and str(raw_ref.get("geometry_scope") or "") == "cell"
            and ref_field == field_name
            and str(raw_ref.get("canonical_template_id") or "")
            in {"", _PROFILE_TEMPLATE_ID}
            and owner_index is not None
        )
        binding_pair = (
            str(raw_ref.get("binding") or ""),
            str(raw_ref.get("binding_quality") or ""),
        )
        native_exact = bool(
            common_exact
            and source == "native_detail_table_cell"
            and binding_pair in _NATIVE_HEADER_BINDING_PAIRS
            and _strict_nonnegative_int(raw_ref.get("row")) is not None
            and _strict_nonnegative_int(raw_ref.get("column")) is not None
            and isinstance(raw_ref.get("table_id"), str)
            and raw_ref["table_id"].strip()
        )
        corrected_exact = bool(
            common_exact
            and source == "personal_detail_corrected_page_cell"
            and binding_pair == _CORRECTED_HEADER_BINDING_PAIR
            and raw_ref.get("table_id") in (None, "")
            and raw_ref.get("row") is None
            and raw_ref.get("column") is None
        )
        if native_exact or corrected_exact:
            assert owner_index is not None
            assert logical_page is not None
            assert source_page is not None
            assert bbox is not None
            assert evidence_ids is not None
            refs_with_owners.append(
                (
                    raw_ref,
                    owner_index,
                    (
                        logical_page,
                        source_page,
                        bbox,
                        evidence_ids,
                        ref_field,
                    ),
                )
            )
    refs = _dedupe_refs([ref for ref, _owner, _slot in refs_with_owners])
    owner_indexes = {owner for _ref, owner, _slot in refs_with_owners}
    slots = {slot for _ref, _owner, slot in refs_with_owners}
    return refs if len(owner_indexes) == 1 and len(slots) == 1 else []


def _profile_issue_mirrors(
    facts: Mapping[str, Any], datasets: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Mirror one exact metadata-field failure to its missing profile alias."""

    metadata_id = _metadata_identity(datasets)
    profile = _profile_values(datasets)
    if not metadata_id or not profile:
        return []
    issues = [row for row in datasets.get("personal_detail_extraction_issues") or () if isinstance(row, Mapping)]
    existing = {
        (
            str(values.get("target_dataset") or ""),
            str(values.get("target_record_id") or ""),
            str(values.get("field_name") or ""),
        )
        for row in issues
        if (values := _values(row)) and _issue_is_active(values)
    }
    mirrors: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for source_issue in issues:
        values = _values(source_issue)
        field_name = str(values.get("field_name") or "")
        if (
            not _issue_is_active(values)
            or str(values.get("target_dataset") or "") != "personal_report_metadata"
            or str(values.get("target_record_id") or "") != metadata_id
            or field_name not in _PROFILE_METADATA_FIELDS
            or field_name in seen_fields
            or not _field_missing(profile, field_name)
            or ("personal_profile", _PROFILE_ID, field_name) in existing
        ):
            continue
        refs = _exact_metadata_field_refs(
            facts, datasets, source_issue, field_name
        )
        # Dataset/page-only diagnostics do not prove a field occurrence.  The
        # mirror is authorized only by an exact value cell on the unique
        # report-header canonical source owner carrying the same finite field.
        if not refs:
            continue
        mirrors.append(
            make_issue(
                category="schema_incompleteness",
                issue_code="source_bound_profile_field_omitted",
                message=(
                    "A source-bound report-header field also owned by the canonical "
                    "subject profile was withheld without a profile-local diagnostic."
                ),
                parser_stage="candidate_b_fail_closed_field_reporting",
                target_dataset="personal_profile",
                target_record_id=_PROFILE_ID,
                field_name=field_name,
                observed_value={
                    "source_issue_id": str(
                        values.get("extraction_issue_id")
                        or source_issue.get("record_id")
                        or ""
                    ),
                    "source_field_observed": True,
                },
                candidate_value={
                    "source_dataset": "personal_report_metadata",
                    "source_record_id": metadata_id,
                },
                source_refs=refs,
                reason_codes=(
                    "exact_source_record_identity",
                    "same_canonical_field_alias",
                    "unique_report_header_source_owner",
                    "exact_canonical_field_slot",
                    "normalized_value_withheld",
                    "profile_projection_conservation",
                ),
            )
        )
        seen_fields.add(field_name)
    return mirrors


def _emitted_raw_profile_rows(
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
        values = _values(row)
        sequence = _strict_positive_int(values.get("sequence"))
        if sequence is None:
            continue
        expected_id = stable_record_id(id_prefix, sequence)
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
        if sequence in emitted:
            return None
        emitted[sequence] = values
    return emitted


def _raw_profile_omission_issues(
    facts: Mapping[str, Any], datasets: Mapping[str, Any]
) -> list[dict[str, Any]]:
    ledger = facts.get("personal_detail_source_completeness_ledger")
    if not isinstance(ledger, Mapping):
        return []
    issues: list[dict[str, Any]] = []
    for dataset, contract in _SEQUENCE_POPULATION_DATASETS.items():
        validated = _exact_raw_profile_observations(ledger, dataset)
        if validated is None:
            continue
        endpoint, observations = validated
        emitted = _emitted_raw_profile_rows(datasets, dataset, contract)
        if emitted is None:
            continue
        existing = {
            (
                str(values.get("target_record_id") or ""),
                str(values.get("field_name") or ""),
            )
            for row in datasets.get("personal_detail_extraction_issues") or ()
            if isinstance(row, Mapping)
            and (values := _values(row))
            and str(values.get("target_dataset") or "") == dataset
            and _issue_is_active(values)
        }
        issue_stem = {
            "mobile_phone_records": "mobile",
            "residence_records": "residence",
            "employment_records": "employment",
        }[dataset]
        id_field = str(contract["id_field"])
        id_prefix = str(contract["id_prefix"])
        for ordinal in range(1, endpoint + 1):
            observation = observations[ordinal]
            target_id = stable_record_id(id_prefix, ordinal)
            emitted_values = emitted.get(ordinal)
            row_missing = emitted_values is None
            if row_missing and (target_id, id_field) not in existing:
                issues.append(
                    make_issue(
                        category="schema_incompleteness",
                        issue_code=f"source_{issue_stem}_record_omitted",
                        message=(
                            "An immutable source-observed PBOC profile row was "
                            "withheld without a row-local diagnostic."
                        ),
                        parser_stage="candidate_b_fail_closed_field_reporting",
                        target_dataset=dataset,
                        target_record_id=target_id,
                        field_name=id_field,
                        observed_value={
                            "sequence": ordinal,
                            "source_row_observed": True,
                        },
                        candidate_value={
                            "source_sequence_endpoint": endpoint,
                            "normalized_value_withheld": True,
                        },
                        source_refs=[dict(observation["source_refs"][0])],
                        reason_codes=(
                            "exact_canonical_header_graph",
                            "exact_source_cell",
                            "exact_profile_ordinal",
                            "missing_business_record",
                            "normalized_value_withheld",
                        ),
                    )
                )
                existing.add((target_id, id_field))
            source_absent = (
                set(emitted_values.get("_source_absent_fields") or ())
                if emitted_values is not None
                and isinstance(
                    emitted_values.get("_source_absent_fields"),
                    (list, tuple, set, frozenset),
                )
                else set()
            )
            for field_name in observation["printed_fields"]:
                if (
                    emitted_values is not None
                    and not _field_missing(emitted_values, field_name)
                ) or field_name in source_absent or (target_id, field_name) in existing:
                    continue
                refs = [dict(observation["field_source_refs"][field_name][0])]
                issues.append(
                    make_issue(
                        category="schema_incompleteness",
                        issue_code=f"source_{issue_stem}_field_omitted",
                        message=(
                            "An exact PBOC profile-table field cell was source-present "
                            "but its normalized canonical field was not emitted."
                        ),
                        parser_stage="candidate_b_fail_closed_field_reporting",
                        target_dataset=dataset,
                        target_record_id=target_id,
                        field_name=field_name,
                        observed_value={
                            "sequence": ordinal,
                            "source_field_observed": True,
                        },
                        candidate_value={
                            "canonical_header_fields_by_component": dict(
                                observation["canonical_header_fields_by_component"]
                            ),
                            "normalized_value_withheld": True,
                        },
                        source_refs=refs,
                        reason_codes=(
                            "exact_canonical_header_graph",
                            "exact_source_cell",
                            "exact_profile_ordinal",
                            (
                                "missing_business_record"
                                if row_missing
                                else "missing_business_field_projection"
                            ),
                            "normalized_value_withheld",
                        ),
                    )
                )
                existing.add((target_id, field_name))
    return issues


def append_fail_closed_field_issues(
    facts: Mapping[str, Any], datasets: dict[str, Any]
) -> None:
    """Append projection-local issues without replacing extractor diagnostics."""

    issues = datasets.setdefault("personal_detail_extraction_issues", [])
    if not isinstance(issues, list):
        issues = []
        datasets["personal_detail_extraction_issues"] = issues
    existing_ids = {
        str(_values(row).get("extraction_issue_id") or row.get("record_id") or "")
        for row in issues
        if isinstance(row, Mapping)
    }
    for issue in (
        *_profile_issue_mirrors(facts, datasets),
        *_raw_profile_omission_issues(facts, datasets),
    ):
        issue_id = str(issue["extraction_issue_id"])
        if issue_id not in existing_ids:
            issues.append(issue)
            existing_ids.add(issue_id)


__all__ = ["append_fail_closed_field_issues"]
