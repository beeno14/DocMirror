# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative Candidate B pipeline for scanned personal detailed reports.

There is deliberately one extraction result.  Compatibility entry points may
expose different slices of it to the generic credit-report projector, but they
must never invoke another OCR/business extractor or merge another population.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import isfinite
from typing import Any

_CORE_BUSINESS_DATASETS = (
    "credit_accounts",
    "credit_lines",
    "repayment_liability_records",
    "repayment_records",
    "overdue_records",
    "inquiry_records",
    "public_records",
)

_CROSS_PLANE_INDIVIDUALIZED_FIELDS = (
    ("residence_records", "residence_record_id", ("address",)),
    ("credit_accounts", "account_id", ("management_institution",)),
    ("credit_lines", "credit_line_id", ("institution",)),
)

_HEADER_OWNER_TEMPLATE = "report_header_and_identity"
_HEADER_METADATA_FIELDS = (
    "report_number",
    "report_time",
    "subject_name",
    "primary_id_type",
    "primary_id_number",
    "query_institution",
    "query_reason",
)
_HEADER_IDENTITY_FIELDS = ("document_type", "document_number")

_CANONICAL_REPAYMENT_STATUS_VALUES = frozenset(
    {
        "*",
        "/",
        "#",
        "N",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "A",
        "B",
        "C",
        "D",
        "G",
        "M",
        "Z",
    }
)

_FINAL_ACCOUNT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "account_currency": ("account_currency", "currency"),
    "business_type": ("business_type",),
    "guarantee_type": ("guarantee_type",),
    "account_identifier": ("account_identifier",),
    "management_institution": ("management_institution",),
}
_ACCOUNT_ISSUES_SUPERSEDED_BY_EXACT_FINAL = frozenset(
    {
        "candidate_b_account_cluster_field_unresolved",
        "candidate_b_account_cluster_residue_unresolved",
        "candidate_b_account_required_field_unresolved",
        "candidate_b_exact_slot_value_invalid",
        "candidate_b_exact_slot_value_row_missing",
        "candidate_b_exact_slot_value_unreadable",
        "candidate_b_institution_leading_boundary_ambiguous",
        "candidate_b_institution_branch_without_legal_root",
    }
)
_CURRENCY_ISSUES_REQUIRING_CORRECTED_CELL_EVIDENCE = frozenset(
    {
        "candidate_b_account_required_field_unresolved",
        "candidate_b_exact_slot_value_unreadable",
    }
)
_INACTIVE_ISSUE_STATUSES = frozenset(
    {"resolved", "suppressed_redundant", "informational"}
)


def _reconcile_repaired_inquiry_source_population(
    ledger: Mapping[str, Any],
    inquiry_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Close source population only after exact repaired-row reconciliation.

    The independent source pass can prove two dense typed ordinal populations
    while deliberately withholding some row-local ordinal refs. Once repair
    emits exactly one source-localized canonical row for every proven
    type/ordinal, the stale pre-repair scalar must follow those source
    endpoints. Neither row count nor encounter order can create an endpoint.
    """

    result = deepcopy(dict(ledger))
    endpoints = ledger.get("inquiry_sequence_endpoints")
    observed = ledger.get("inquiry_observed_sequences")
    outliers = ledger.get("inquiry_sequence_outliers")
    if not isinstance(endpoints, Mapping) or not isinstance(observed, Mapping):
        return result
    outlier_map = outliers if isinstance(outliers, Mapping) else {}
    allowed_types = {"institution", "personal"}
    expected_identities: set[tuple[str, int]] = set()
    for raw_type, raw_endpoint in endpoints.items():
        inquiry_type = str(raw_type or "")
        if (
            inquiry_type not in allowed_types
            or isinstance(raw_endpoint, bool)
            or not isinstance(raw_endpoint, int)
            or raw_endpoint <= 0
        ):
            return result
        raw_observed = observed.get(raw_type)
        if not isinstance(raw_observed, (list, tuple, set, frozenset)):
            return result
        observed_values = {
            value
            for value in raw_observed
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
        raw_outliers = outlier_map.get(raw_type, ())
        outlier_values = {
            value
            for value in raw_outliers
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
        if observed_values - outlier_values != set(range(1, raw_endpoint + 1)):
            return result
        expected_identities.update(
            (inquiry_type, sequence)
            for sequence in range(1, raw_endpoint + 1)
        )
    if not expected_identities:
        return result

    from docmirror.plugins.credit_report.value_utils import stable_record_id

    rows_by_identity: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    source_owner_by_identity: dict[tuple[str, int], set[tuple[Any, ...]]] = {}
    for row in inquiry_rows:
        if not isinstance(row, Mapping):
            return result
        inquiry_type = str(
            row.get("inquiry_type") or row.get("query_channel") or ""
        )
        sequence = row.get("sequence")
        if (
            inquiry_type not in allowed_types
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= 0
        ):
            return result
        identity = (inquiry_type, sequence)
        expected_id = stable_record_id("credit_inquiry", inquiry_type, sequence)
        if identity not in expected_identities or row.get("inquiry_id") != expected_id:
            return result
        localized_owners: set[tuple[Any, ...]] = set()
        for ref in row.get("source_refs") or ():
            if not isinstance(ref, Mapping):
                continue
            logical_page = ref.get("logical_page")
            source_page = ref.get("source_page")
            table_id = str(ref.get("table_id") or "").strip()
            source_row = ref.get("row")
            if (
                isinstance(logical_page, int)
                and not isinstance(logical_page, bool)
                and logical_page > 0
                and isinstance(source_page, int)
                and not isinstance(source_page, bool)
                and source_page > 0
                and table_id
                and isinstance(source_row, int)
                and not isinstance(source_row, bool)
                and source_row >= 0
            ):
                localized_owners.add(
                    (logical_page, source_page, table_id, source_row)
                )
        if not localized_owners:
            return result
        rows_by_identity.setdefault(identity, []).append(row)
        source_owner_by_identity.setdefault(identity, set()).update(
            localized_owners
        )
    if set(rows_by_identity) != expected_identities or any(
        len(rows) != 1 for rows in rows_by_identity.values()
    ):
        return result
    owner_claims: dict[tuple[Any, ...], set[tuple[str, int]]] = {}
    for identity, owners in source_owner_by_identity.items():
        for owner in owners:
            owner_claims.setdefault(owner, set()).add(identity)
    if any(len(identities) != 1 for identities in owner_claims.values()):
        return result

    current_population = result.get("inquiry_records")
    result["inquiry_records"] = max(
        current_population
        if isinstance(current_population, int)
        and not isinstance(current_population, bool)
        and current_population > 0
        else 0,
        len(expected_identities),
    )
    return result
_EXACT_ACCOUNT_FIELD_BINDINGS = frozenset(
    {
        "canonical_header_column",
        "canonical_account_header_geometry",
        "canonical_field_slot",
        "closed_canonical_account_cell_cluster",
    }
)
_ACCOUNT_INSTITUTION_LEGAL_ROOTS = (
    "银行股份有限公司",
    "银行有限责任公司",
    "小额贷款有限公司",
    "消费金融有限公司",
    "汽车金融有限公司",
    "金融租赁有限公司",
    "租赁有限公司",
    "信托有限公司",
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "农村商业银行",
    "农村合作银行",
    "信用合作联社",
    "农村信用合作社",
    "公积金管理中心",
)


def _individualized_scalar_key(value: Any) -> str:
    """Ignore layout whitespace, but preserve every individualized glyph."""

    return "".join(str(value or "").split())


def _field_raw(record: Mapping[str, Any], field_name: str, fallback: Any) -> Any:
    canonical_raw = record.get("canonical_raw")
    if isinstance(canonical_raw, Mapping) and canonical_raw.get(field_name) not in (None, ""):
        return canonical_raw[field_name]
    return fallback


def _plane_refs(
    record: Mapping[str, Any], field_name: str, evidence_plane: str
) -> list[dict[str, Any]]:
    refs_by_field = record.get("source_refs_by_field")
    refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else None
    if not refs:
        refs = record.get("source_refs")
    if not refs:
        cell_refs = record.get("source_cell_refs")
        if isinstance(cell_refs, (list, tuple)):
            field_aliases = {field_name}
            if field_name in {"status", "status_code"}:
                field_aliases.update({"status", "status_code"})
            field_refs = [
                ref
                for ref in cell_refs
                if isinstance(ref, Mapping)
                and str(ref.get("field_name") or "") in field_aliases
            ]
            refs = field_refs or cell_refs
    return [
        {**dict(ref), "evidence_plane": evidence_plane}
        for ref in refs or ()
        if isinstance(ref, Mapping)
    ]


def _exact_header_lifecycle_ref(
    ref: Any,
    *,
    field_name: str,
) -> tuple[dict[str, Any], tuple[int, int, str, int, int], tuple[str, ...]] | None:
    """Validate one immutable, canonical PBOC header-cell observation.

    Candidate B's repaired page plane may legitimately replace the table that
    exposed this cell.  The discovery value is nevertheless reusable only when
    its record retains the complete source-owned cell identity produced by
    ``_exact_report_header_cell_ref``.  Labels, row order, and page location by
    themselves are deliberately insufficient authority.
    """

    if not isinstance(ref, Mapping):
        return None
    if (
        str(ref.get("source") or "") != "native_detail_table_cell"
        or str(ref.get("geometry_scope") or "") != "cell"
        or str(ref.get("canonical_template_id") or "") != _HEADER_OWNER_TEMPLATE
        or str(ref.get("binding") or "") != "canonical_field_slot"
        or str(ref.get("binding_quality") or "") != "canonical_header_column"
        or str(ref.get("field_name") or "") != field_name
    ):
        return None
    for key in ("logical_page", "source_page", "row", "column", "canonical_row", "canonical_column"):
        if isinstance(ref.get(key), bool):
            return None
    try:
        logical_page = int(ref["logical_page"])
        source_page = int(ref["source_page"])
        row = int(ref["row"])
        column = int(ref["column"])
        canonical_row = int(ref["canonical_row"])
        canonical_column = int(ref["canonical_column"])
        bbox = tuple(float(value) for value in ref["bbox"])
    except (KeyError, TypeError, ValueError):
        return None
    table_id = str(ref.get("table_id") or "").strip()
    raw_evidence_ids = ref.get("evidence_ids")
    if not isinstance(raw_evidence_ids, (list, tuple)):
        return None
    evidence_ids = tuple(str(value).strip() for value in raw_evidence_ids)
    if (
        logical_page <= 0
        or source_page <= 0
        or row < 0
        or column < 0
        or canonical_row != row
        or canonical_column != column
        or not table_id
        or len(bbox) != 4
        or not all(isfinite(value) for value in bbox)
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
        or not evidence_ids
        or any(not value for value in evidence_ids)
        or len(set(evidence_ids)) != len(evidence_ids)
    ):
        return None
    return (
        dict(ref),
        (logical_page, source_page, table_id, row, column),
        evidence_ids,
    )


@dataclass(frozen=True)
class _HeaderFieldObservation:
    plane: str
    row_index: int
    field_name: str
    value: Any
    ref: dict[str, Any]
    slot: tuple[int, int, str, int, int]
    evidence_ids: tuple[str, ...]


def _header_field_observation(
    row: Mapping[str, Any],
    *,
    row_index: int,
    field_name: str,
    plane: str,
) -> _HeaderFieldObservation | None:
    """Return one field only when exactly one sealed cell owns its value."""

    value = row.get(field_name)
    if value in (None, "") or str(row.get("source") or "") != "native_detail_header":
        return None
    try:
        confidence = float(row.get("confidence"))
    except (TypeError, ValueError):
        return None
    if confidence != 1.0:
        return None
    refs = row.get("source_refs")
    if not isinstance(refs, (list, tuple)):
        return None
    matching = [
        ref
        for ref in refs
        if isinstance(ref, Mapping)
        and str(ref.get("field_name") or "") == field_name
    ]
    # Two cells claiming one field are not a consensus.  This also makes an
    # injected duplicate owner fail closed instead of being hidden by dedupe.
    if len(matching) != 1:
        return None
    validated = _exact_header_lifecycle_ref(matching[0], field_name=field_name)
    if validated is None:
        return None
    ref, slot, evidence_ids = validated
    return _HeaderFieldObservation(
        plane=plane,
        row_index=row_index,
        field_name=field_name,
        value=value,
        ref=ref,
        slot=slot,
        evidence_ids=evidence_ids,
    )


def _header_field_ref_state(
    row: Mapping[str, Any],
    *,
    field_name: str,
) -> str:
    """Classify exact field ownership without treating duplicates as absence."""

    if row.get(field_name) in (None, ""):
        return "absent"
    refs = row.get("source_refs")
    matching = [
        ref
        for ref in refs or ()
        if isinstance(ref, Mapping)
        and str(ref.get("field_name") or "") == field_name
        and _exact_header_lifecycle_ref(ref, field_name=field_name) is not None
    ]
    if len(matching) > 1:
        return "duplicate"
    return "exact" if len(matching) == 1 else "unproven"


def _record_header_lifecycle_conflict(
    context: Any,
    *,
    dataset: str,
    target_record_id: str,
    field_name: str,
    observations: list[_HeaderFieldObservation],
    reason_code: str,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    record_issue(
        context,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_header_lifecycle_conflict",
            message=(
                "Exact source-owned PBOC header observations conflicted across "
                "the Candidate B discovery and repaired-page lifecycle; the "
                "field or identity row was withheld without choosing a plane."
            ),
            parser_stage="candidate_b_header_lifecycle_reconciliation",
            target_dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            observed_value={
                observation.plane: observation.value
                for observation in observations
            },
            source_refs=(
                {
                    **observation.ref,
                    "evidence_plane": observation.plane,
                }
                for observation in observations
            ),
            reason_codes=(
                "registered_report_header_owner",
                "exact_canonical_header_cell",
                "immutable_header_cell_slot",
                reason_code,
                "normalized_value_withheld",
            ),
        ),
    )


def _reconcile_header_metadata_lifecycle(
    context: Any,
    discovery_rows: list[Mapping[str, Any]],
    repaired_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Conserve exact metadata fields lost solely through page replacement."""

    if len(repaired_rows) > 1:
        # Candidate B emits one report-metadata entity.  Multiple repaired
        # entities provide no generic basis for selecting a destination row.
        return [deepcopy(dict(row)) for row in repaired_rows]

    observations: dict[str, dict[str, list[_HeaderFieldObservation]]] = {
        "discovery_static": {field: [] for field in _HEADER_METADATA_FIELDS},
        "repaired_page": {field: [] for field in _HEADER_METADATA_FIELDS},
    }
    for plane, rows in (
        ("discovery_static", discovery_rows),
        ("repaired_page", repaired_rows),
    ):
        for row_index, row in enumerate(rows):
            for field_name in _HEADER_METADATA_FIELDS:
                observation = _header_field_observation(
                    row,
                    row_index=row_index,
                    field_name=field_name,
                    plane=plane,
                )
                if observation is not None:
                    observations[plane][field_name].append(observation)

    metadata_owner_conflicts: set[
        tuple[str, int, int, str, int, int]
    ] = set()
    metadata_evidence_owners: dict[
        tuple[str, str],
        set[tuple[int, int, str, int, int, str, str]],
    ] = {}
    for plane in ("discovery_static", "repaired_page"):
        for field_name in _HEADER_METADATA_FIELDS:
            for observation in observations[plane][field_name]:
                for evidence_id in observation.evidence_ids:
                    metadata_evidence_owners.setdefault(
                        (plane, evidence_id),
                        set(),
                    ).add(
                        (
                            *observation.slot,
                            field_name,
                            _individualized_scalar_key(observation.value),
                        )
                    )
    for (plane, _evidence_id), owners in metadata_evidence_owners.items():
        if len(owners) <= 1:
            continue
        for logical_page, source_page, table_id, row, column, _field, _value in owners:
            metadata_owner_conflicts.add(
                (plane, logical_page, source_page, table_id, row, column)
            )

    discovery_has_exact = any(
        observations["discovery_static"][field]
        for field in _HEADER_METADATA_FIELDS
    )
    if not repaired_rows and not discovery_has_exact:
        return []
    if repaired_rows:
        merged = deepcopy(dict(repaired_rows[0]))
    else:
        # Copy policy/default columns from the sole discovery entity, but only
        # exact cell-owned header values survive below.
        source_rows = {
            observation.row_index
            for field in _HEADER_METADATA_FIELDS
            for observation in observations["discovery_static"][field]
        }
        if len(source_rows) != 1:
            return []
        merged = deepcopy(dict(discovery_rows[next(iter(source_rows))]))
        for field_name in _HEADER_METADATA_FIELDS:
            merged[field_name] = None
    original_merged = deepcopy(merged)

    retained_refs: list[dict[str, Any]] = []
    for field_name in _HEADER_METADATA_FIELDS:
        discovery = observations["discovery_static"][field_name]
        repaired = observations["repaired_page"][field_name]
        global_owner_conflicts = [
            observation
            for observation in (*discovery, *repaired)
            if (observation.plane, *observation.slot) in metadata_owner_conflicts
        ]
        discovery_duplicate_rows = [
            (row_index, row)
            for row_index, row in enumerate(discovery_rows)
            if _header_field_ref_state(row, field_name=field_name) == "duplicate"
        ]
        repaired_duplicate_rows = [
            (row_index, row)
            for row_index, row in enumerate(repaired_rows)
            if _header_field_ref_state(row, field_name=field_name) == "duplicate"
        ]
        if (
            len(discovery) > 1
            or len(repaired) > 1
            or discovery_duplicate_rows
            or repaired_duplicate_rows
            or global_owner_conflicts
        ):
            merged[field_name] = None
            conflicting = [*discovery, *repaired]
            for plane, duplicate_rows in (
                ("discovery_static", discovery_duplicate_rows),
                ("repaired_page", repaired_duplicate_rows),
            ):
                for row_index, row in duplicate_rows:
                    for ref in row.get("source_refs") or ():
                        validated = _exact_header_lifecycle_ref(
                            ref,
                            field_name=field_name,
                        )
                        if validated is None:
                            continue
                        exact_ref, slot, evidence_ids = validated
                        conflicting.append(
                            _HeaderFieldObservation(
                                plane=plane,
                                row_index=row_index,
                                field_name=field_name,
                                value=row.get(field_name),
                                ref=exact_ref,
                                slot=slot,
                                evidence_ids=evidence_ids,
                            )
                        )
            _record_header_lifecycle_conflict(
                context,
                dataset="personal_report_metadata",
                target_record_id=str(
                    merged.get("personal_report_metadata_id")
                    or "personal_report_metadata:header_lifecycle"
                ),
                field_name=field_name,
                observations=[*conflicting, *global_owner_conflicts],
                reason_code=(
                    "conflicting_exact_header_evidence_owner"
                    if global_owner_conflicts
                    else "duplicate_exact_header_field_owner"
                ),
            )
            continue
        if discovery and repaired:
            native_value = _individualized_scalar_key(discovery[0].value)
            repaired_value = _individualized_scalar_key(repaired[0].value)
            if native_value != repaired_value:
                merged[field_name] = None
                merged.setdefault("canonical_raw", {})[field_name] = [
                    discovery[0].value,
                    repaired[0].value,
                ]
                unresolved = merged.setdefault("_unresolved_fields", [])
                if field_name not in unresolved:
                    unresolved.append(field_name)
                _record_header_lifecycle_conflict(
                    context,
                    dataset="personal_report_metadata",
                    target_record_id=str(
                        merged.get("personal_report_metadata_id")
                        or "personal_report_metadata:header_lifecycle"
                    ),
                    field_name=field_name,
                    observations=[discovery[0], repaired[0]],
                    reason_code="independent_exact_header_value_conflict",
                )
                continue
            merged[field_name] = repaired[0].value
            retained_refs.extend((discovery[0].ref, repaired[0].ref))
            continue
        observation = (repaired or discovery)
        if observation:
            merged[field_name] = observation[0].value
            retained_refs.append(observation[0].ref)
            continue
        # No exact lifecycle observation owns this field. Preserve the
        # repaired pass's ordinary fail-closed result; discovery values are
        # never resurrected through an unowned row.
        if repaired_rows:
            merged[field_name] = original_merged.get(field_name)

    # Header refs are field-specific and exact.  Do not carry a stale repaired
    # ref for a field that was just withheld.
    unique_refs: list[dict[str, Any]] = []
    seen_refs: set[tuple[Any, ...]] = set()
    for ref in retained_refs:
        marker = (
            ref.get("logical_page"),
            ref.get("source_page"),
            ref.get("table_id"),
            ref.get("row"),
            ref.get("column"),
            ref.get("field_name"),
            tuple(ref.get("evidence_ids") or ()),
        )
        if marker not in seen_refs:
            seen_refs.add(marker)
            unique_refs.append(deepcopy(ref))
    merged["source_refs"] = unique_refs

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        register_issue_target_remap,
    )
    from docmirror.plugins.credit_report.value_utils import stable_record_id

    old_ids = {
        str(row.get("personal_report_metadata_id") or "")
        for row in (*discovery_rows, *repaired_rows)
        if row.get("personal_report_metadata_id")
    }
    metadata_id = stable_record_id(
        "personal_report_metadata",
        merged.get("report_number"),
        merged.get("report_time"),
        merged.get("subject_name"),
    )
    merged["personal_report_metadata_id"] = metadata_id
    for old_id in old_ids:
        register_issue_target_remap(context, old_id, metadata_id)
    return [merged]


@dataclass(frozen=True)
class _ExactIdentityObservation:
    plane: str
    row_index: int
    row: dict[str, Any]
    fields: tuple[_HeaderFieldObservation, _HeaderFieldObservation]

    @property
    def is_primary(self) -> bool:
        return self.row.get("is_primary") is True

    @property
    def semantic_key(self) -> tuple[str, str, str]:
        return (
            "primary" if self.is_primary else "additional",
            _individualized_scalar_key(self.row.get("document_type")).upper(),
            _individualized_scalar_key(self.row.get("document_number")).upper(),
        )


def _exact_identity_observations(
    rows: list[Mapping[str, Any]],
    *,
    plane: str,
) -> list[_ExactIdentityObservation]:
    observations: list[_ExactIdentityObservation] = []
    for row_index, row in enumerate(rows):
        fields = tuple(
            observation
            for field_name in _HEADER_IDENTITY_FIELDS
            if (
                observation := _header_field_observation(
                    row,
                    row_index=row_index,
                    field_name=field_name,
                    plane=plane,
                )
            )
            is not None
        )
        if len(fields) != len(_HEADER_IDENTITY_FIELDS):
            continue
        observations.append(
            _ExactIdentityObservation(
                plane=plane,
                row_index=row_index,
                row=deepcopy(dict(row)),
                fields=(fields[0], fields[1]),
            )
        )
    return observations


def _reconcile_header_identity_lifecycle(
    context: Any,
    discovery_rows: list[Mapping[str, Any]],
    repaired_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Union exact identity rows while vetoing duplicate/conflicting owners."""

    discovery = _exact_identity_observations(
        discovery_rows,
        plane="discovery_static",
    )
    repaired = _exact_identity_observations(
        repaired_rows,
        plane="repaired_page",
    )
    candidates = [*discovery, *repaired]
    invalid: set[tuple[str, int]] = set()

    def invalidate(
        grouped: list[_ExactIdentityObservation],
        reason_code: str,
    ) -> None:
        if len(grouped) < 2:
            return
        observations = [field for candidate in grouped for field in candidate.fields]
        for candidate in grouped:
            invalid.add((candidate.plane, candidate.row_index))
        _record_header_lifecycle_conflict(
            context,
            dataset="identity_documents",
            target_record_id="identity_document:header_lifecycle",
            field_name="identity_document",
            observations=observations,
            reason_code=reason_code,
        )

    # More than one primary or the same additional document twice inside one
    # plane is an ambiguous entity population, even when the glyphs agree.
    for plane_candidates in (discovery, repaired):
        primaries = [candidate for candidate in plane_candidates if candidate.is_primary]
        if len(primaries) > 1:
            invalidate(primaries, "duplicate_exact_primary_identity_owner")
        by_semantic: dict[tuple[str, str, str], list[_ExactIdentityObservation]] = {}
        for candidate in plane_candidates:
            by_semantic.setdefault(candidate.semantic_key, []).append(candidate)
        for grouped in by_semantic.values():
            if len(grouped) > 1:
                invalidate(grouped, "duplicate_exact_identity_owner")

        by_slot: dict[
            tuple[int, int, str, int, int],
            list[tuple[_ExactIdentityObservation, _HeaderFieldObservation]],
        ] = {}
        by_evidence: dict[
            str,
            list[tuple[_ExactIdentityObservation, _HeaderFieldObservation]],
        ] = {}
        for candidate in plane_candidates:
            for field in candidate.fields:
                by_slot.setdefault(field.slot, []).append((candidate, field))
                for evidence_id in field.evidence_ids:
                    by_evidence.setdefault(evidence_id, []).append((candidate, field))
        for grouped in by_slot.values():
            if len({(field.field_name, field.value) for _candidate, field in grouped}) > 1:
                invalidate(
                    [candidate for candidate, _field in grouped],
                    "duplicate_exact_identity_cell_owner",
                )
        for grouped in by_evidence.values():
            signatures = {
                (field.field_name, field.slot)
                for _candidate, field in grouped
            }
            if len(signatures) > 1:
                invalidate(
                    [candidate for candidate, _field in grouped],
                    "conflicting_exact_identity_evidence_owner",
                )

    # The same immutable cell/evidence ID cannot publish two different fields
    # or values across the discovery and repaired planes.
    global_slots: dict[
        tuple[int, int, str, int, int],
        list[tuple[_ExactIdentityObservation, _HeaderFieldObservation]],
    ] = {}
    global_evidence: dict[
        str,
        list[tuple[_ExactIdentityObservation, _HeaderFieldObservation]],
    ] = {}
    for candidate in candidates:
        for field in candidate.fields:
            global_slots.setdefault(field.slot, []).append((candidate, field))
            for evidence_id in field.evidence_ids:
                global_evidence.setdefault(evidence_id, []).append((candidate, field))
    for grouped in global_slots.values():
        meanings = {
            (field.field_name, _individualized_scalar_key(field.value).upper())
            for _candidate, field in grouped
        }
        if len(meanings) > 1:
            invalidate(
                [candidate for candidate, _field in grouped],
                "independent_exact_identity_cell_conflict",
            )
    for grouped in global_evidence.values():
        owners = {
            (
                field.slot,
                field.field_name,
                _individualized_scalar_key(field.value).upper(),
            )
            for _candidate, field in grouped
        }
        if len(owners) > 1:
            invalidate(
                [candidate for candidate, _field in grouped],
                "independent_exact_identity_evidence_conflict",
            )

    valid_primaries = [
        candidate
        for candidate in candidates
        if candidate.is_primary
        and (candidate.plane, candidate.row_index) not in invalid
    ]
    if len({candidate.semantic_key for candidate in valid_primaries}) > 1:
        invalidate(valid_primaries, "independent_exact_primary_identity_conflict")

    accepted: dict[tuple[str, str, str], list[_ExactIdentityObservation]] = {}
    for candidate in candidates:
        if (candidate.plane, candidate.row_index) in invalid:
            continue
        accepted.setdefault(candidate.semantic_key, []).append(candidate)
    blocked_semantic_keys = {
        candidate.semantic_key
        for candidate in candidates
        if (candidate.plane, candidate.row_index) in invalid
    }

    accepted_rows: list[dict[str, Any]] = []
    for semantic_key, grouped in accepted.items():
        preferred = next(
            (candidate for candidate in grouped if candidate.plane == "repaired_page"),
            grouped[0],
        )
        merged = deepcopy(preferred.row)
        discovery_match = next(
            (candidate for candidate in grouped if candidate.plane == "discovery_static"),
            None,
        )
        if discovery_match is not None:
            for key, value in discovery_match.row.items():
                if merged.get(key) in (None, "") and value not in (None, ""):
                    merged[key] = deepcopy(value)
        refs: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for candidate in grouped:
            for field in candidate.fields:
                marker = (
                    field.slot,
                    field.field_name,
                    field.evidence_ids,
                )
                if marker in seen:
                    continue
                seen.add(marker)
                refs.append(deepcopy(field.ref))
        merged["source_refs"] = refs
        accepted_rows.append(merged)

    accepted_keys = set(accepted)
    exact_primary_candidate_seen = any(candidate.is_primary for candidate in candidates)
    primary_blocked = exact_primary_candidate_seen and not any(
        key[0] == "primary" for key in accepted_keys
    )
    repaired_exact_indices = {candidate.row_index for candidate in repaired}
    retained_unproven: list[dict[str, Any]] = []
    for row_index, row in enumerate(repaired_rows):
        if row_index in repaired_exact_indices:
            continue
        semantic_key = (
            "primary" if row.get("is_primary") is True else "additional",
            _individualized_scalar_key(row.get("document_type")).upper(),
            _individualized_scalar_key(row.get("document_number")).upper(),
        )
        if semantic_key in accepted_keys or semantic_key in blocked_semantic_keys:
            continue
        if row.get("source") == "native_detail_header" and all(
            row.get(field_name) not in (None, "")
            for field_name in _HEADER_IDENTITY_FIELDS
        ):
            # A header row whose essential fields lack the complete immutable
            # slot proof is not a lifecycle-preservable entity.  Leaving it in
            # the repaired pass would let duplicate/malformed owners bypass
            # the exact population gate above.
            continue
        if semantic_key[0] == "primary" and (
            primary_blocked or any(key[0] == "primary" for key in accepted_keys)
        ):
            continue
        retained_unproven.append(deepcopy(dict(row)))

    final_rows = [*accepted_rows, *retained_unproven]

    def sequence_key(row: Mapping[str, Any]) -> tuple[int, int]:
        primary_rank = 0 if row.get("is_primary") is True else 1
        try:
            sequence = int(row.get("sequence") or 10**9)
        except (TypeError, ValueError):
            sequence = 10**9
        return primary_rank, sequence

    final_rows.sort(key=sequence_key)
    return final_rows


def _reconcile_candidate_b_header_lifecycle(
    context: Any,
    discovery_datasets: Mapping[str, Any],
    repaired_datasets: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Preserve only sealed header observations across page-repair replacement."""

    reconciled = {
        name: [deepcopy(dict(row)) for row in rows if isinstance(row, Mapping)]
        for name, rows in repaired_datasets.items()
        if isinstance(rows, (list, tuple))
    }
    discovery_metadata = [
        row
        for row in discovery_datasets.get("personal_report_metadata") or ()
        if isinstance(row, Mapping)
    ]
    repaired_metadata = [
        row
        for row in repaired_datasets.get("personal_report_metadata") or ()
        if isinstance(row, Mapping)
    ]
    reconciled["personal_report_metadata"] = _reconcile_header_metadata_lifecycle(
        context,
        discovery_metadata,
        repaired_metadata,
    )
    discovery_identities = [
        row
        for row in discovery_datasets.get("identity_documents") or ()
        if isinstance(row, Mapping)
    ]
    repaired_identities = [
        row
        for row in repaired_datasets.get("identity_documents") or ()
        if isinstance(row, Mapping)
    ]
    reconciled["identity_documents"] = _reconcile_header_identity_lifecycle(
        context,
        discovery_identities,
        repaired_identities,
    )
    reconciled["credit_accounts"] = _reconcile_account_population_lifecycle(
        context,
        [
            row
            for row in discovery_datasets.get("credit_accounts") or ()
            if isinstance(row, Mapping)
        ],
        [
            row
            for row in repaired_datasets.get("credit_accounts") or ()
            if isinstance(row, Mapping)
        ],
    )
    return reconciled


_ACCOUNT_LIFECYCLE_ID_RE = re.compile(
    r"^credit_account:(?P<family>[a-z0-9_]+):(?P<ordinal>[1-9]\d*)$"
)
_ACCOUNT_LIFECYCLE_IDENTITY_FIELDS = (
    "account_id",
    "account_type",
    "category_sequence",
    "account_family_quality",
    "_printed_ordinal_status",
    "_canonical_segment",
    "page",
    "source_page",
    "bbox",
)
_ACCOUNT_LIFECYCLE_ALIAS_FIELDS = (
    "account_id",
    "record_id",
    "_table_observation_id",
    "_table_observation_instance_id",
)


def _sealed_discovery_account_id(row: Mapping[str, Any]) -> str | None:
    """Return an account id only when its printed ordinal owner is sealed."""

    account_id = str(row.get("account_id") or "").strip()
    match = _ACCOUNT_LIFECYCLE_ID_RE.fullmatch(account_id)
    if match is None:
        return None
    family = match.group("family")
    ordinal = int(match.group("ordinal"))
    try:
        category_sequence = int(row.get("category_sequence"))
    except (TypeError, ValueError):
        return None
    segment = row.get("_canonical_segment")
    if (
        str(row.get("account_type") or "") != family
        or category_sequence != ordinal
        or row.get("account_family_quality") != "exact"
        or row.get("_printed_ordinal_status") != "printed_unique"
        or not isinstance(segment, Mapping)
        or segment.get("ownership_basis") != "printed_anchor_to_next_anchor"
    ):
        return None

    for raw_ref in row.get("source_refs") or ():
        if not isinstance(raw_ref, Mapping):
            continue
        if raw_ref.get("source") != "candidate_b_account_anchor":
            continue
        binding = str(
            raw_ref.get("binding") or raw_ref.get("binding_quality") or ""
        )
        if binding and binding != "printed_account_ordinal":
            continue
        if raw_ref.get("account_type") not in (None, "", family):
            continue
        if raw_ref.get("category_sequence") not in (None, "", ordinal):
            continue
        evidence_ids = tuple(
            str(value).strip()
            for value in raw_ref.get("evidence_ids") or ()
            if str(value).strip()
        )
        bbox = raw_ref.get("bbox")
        try:
            valid_bbox = (
                isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and all(isfinite(float(value)) for value in bbox)
            )
        except (TypeError, ValueError):
            valid_bbox = False
        if (
            int(raw_ref.get("logical_page") or 0) > 0
            and int(raw_ref.get("source_page") or 0) > 0
            and valid_bbox
            and evidence_ids
        ):
            return account_id
    return None


def _account_lifecycle_identifier(row: Mapping[str, Any]) -> str | None:
    value = _account_field_value(row, "account_identifier")
    if not _final_account_field_is_valid("account_identifier", value):
        return None
    return _individualized_scalar_key(value).upper()


def _account_lifecycle_anchor_observations(
    row: Mapping[str, Any],
) -> tuple[tuple[int, int, tuple[float, float, float, float]], ...]:
    """Return only evidence-sealed printed-anchor geometry."""

    observations: list[
        tuple[int, int, tuple[float, float, float, float]]
    ] = []
    for raw_ref in row.get("source_refs") or ():
        if not isinstance(raw_ref, Mapping):
            continue
        if raw_ref.get("source") != "candidate_b_account_anchor":
            continue
        binding = str(
            raw_ref.get("binding") or raw_ref.get("binding_quality") or ""
        )
        if binding and binding != "printed_account_ordinal":
            continue
        try:
            logical_page = int(raw_ref.get("logical_page") or 0)
            source_page = int(raw_ref.get("source_page") or 0)
            raw_bbox = raw_ref.get("bbox")
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                continue
            bbox = tuple(float(value) for value in raw_bbox)
        except (TypeError, ValueError):
            continue
        evidence_ids = {
            str(value).strip()
            for value in raw_ref.get("evidence_ids") or ()
            if str(value).strip()
        }
        if (
            logical_page <= 0
            or source_page <= 0
            or not all(isfinite(value) for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
            or not evidence_ids
        ):
            continue
        observations.append((logical_page, source_page, bbox))
    return tuple(observations)


def _exact_source_census_account_id(row: Mapping[str, Any]) -> str | None:
    """Validate one identity-only observation from the independent census."""

    account_id = str(row.get("account_id") or "").strip()
    match = _ACCOUNT_LIFECYCLE_ID_RE.fullmatch(account_id)
    if match is None:
        return None
    family = match.group("family")
    ordinal = int(match.group("ordinal"))
    try:
        category_sequence = int(row.get("category_sequence"))
    except (TypeError, ValueError):
        return None
    if (
        str(row.get("account_type") or "") != family
        or category_sequence != ordinal
    ):
        return None
    for raw_ref in row.get("source_refs") or ():
        if not isinstance(raw_ref, Mapping):
            continue
        if raw_ref.get("source") != "candidate_b_account_anchor":
            continue
        binding = str(
            raw_ref.get("binding") or raw_ref.get("binding_quality") or ""
        )
        if binding != "printed_account_ordinal":
            continue
        if raw_ref.get("account_type") not in (None, "", family):
            continue
        if raw_ref.get("category_sequence") not in (None, "", ordinal):
            continue
        if _account_lifecycle_anchor_observations({"source_refs": [raw_ref]}):
            return account_id
    return None


def _exact_source_census_account_rows(context: Any) -> tuple[dict[str, Any], ...]:
    """Read exact account identities from the source census, never values."""

    from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
        _source_completeness_ledger,
    )

    ledger = _source_completeness_ledger(context)
    raw_families = ledger.get("account_family_ordinal_observations")
    if not isinstance(raw_families, Mapping):
        return ()
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_family, raw_observations in raw_families.items():
        family = str(raw_family or "").strip()
        if not isinstance(raw_observations, Mapping):
            continue
        for raw_ordinal, raw_observation in raw_observations.items():
            if not isinstance(raw_observation, Mapping):
                continue
            try:
                ordinal = int(raw_ordinal)
            except (TypeError, ValueError):
                continue
            if str(raw_ordinal).strip() != str(ordinal) or ordinal <= 0:
                continue
            row = deepcopy(dict(raw_observation))
            expected_id = f"credit_account:{family}:{ordinal}"
            if (
                _exact_source_census_account_id(row) != expected_id
                or expected_id in seen_ids
            ):
                continue
            seen_ids.add(expected_id)
            rows.append(row)
    return tuple(rows)


def _account_lifecycle_anchor_geometry_matches(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Match two OCR planes only when the same printed line strongly overlaps."""

    for left_page, left_source, left_box in _account_lifecycle_anchor_observations(
        left
    ):
        for (
            right_page,
            right_source,
            right_box,
        ) in _account_lifecycle_anchor_observations(right):
            if (left_page, left_source) != (right_page, right_source):
                continue
            intersection_width = max(
                0.0,
                min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]),
            )
            intersection_height = max(
                0.0,
                min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]),
            )
            intersection = intersection_width * intersection_height
            left_area = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])
            right_area = (right_box[2] - right_box[0]) * (
                right_box[3] - right_box[1]
            )
            if intersection / min(left_area, right_area) >= 0.80:
                return True
    return False


def _account_lifecycle_aliases(row: Mapping[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for owner in (
        row,
        *(
            nested
            for key in ("canonical_raw", "raw", "normalized")
            if isinstance((nested := row.get(key)), Mapping)
        ),
    ):
        aliases.update(
            str(owner.get(field_name) or "").strip()
            for field_name in _ACCOUNT_LIFECYCLE_ALIAS_FIELDS
            if str(owner.get(field_name) or "").strip()
        )
    return aliases


def _authoritative_discovery_account_rows(
    context: Any,
    discovery_rows: list[Mapping[str, Any]],
    *,
    source_census_rows: tuple[Mapping[str, Any], ...] = (),
) -> list[Mapping[str, Any]]:
    """Join discovery values to the separately sealed pre-repair anchors."""

    raw_inventory = getattr(
        context,
        "_candidate_b_pre_repair_account_anchor_inventory",
        (),
    )
    inventory_rows = [
        row
        for row in raw_inventory
        if isinstance(row, Mapping) and _sealed_discovery_account_id(row)
    ]
    census_rows = [
        row
        for row in source_census_rows
        if isinstance(row, Mapping) and _exact_source_census_account_id(row)
    ]
    if not inventory_rows and not census_rows:
        return list(discovery_rows)

    inventory_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in inventory_rows:
        account_id = _sealed_discovery_account_id(row)
        if account_id:
            inventory_groups.setdefault(account_id, []).append(row)
    blocked_inventory_ids = {
        account_id for account_id, rows in inventory_groups.items() if len(rows) != 1
    }
    inventory_by_id = {
        account_id: rows[0]
        for account_id, rows in inventory_groups.items()
        if len(rows) == 1
    }
    census_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in census_rows:
        account_id = _exact_source_census_account_id(row)
        if account_id:
            census_groups.setdefault(account_id, []).append(row)
    for account_id, rows in census_groups.items():
        if (
            len(rows) == 1
            and account_id not in inventory_by_id
            and account_id not in blocked_inventory_ids
        ):
            inventory_by_id[account_id] = rows[0]
    discovery_by_id: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(discovery_rows):
        account_id = str(row.get("account_id") or "").strip()
        if account_id:
            discovery_by_id.setdefault(account_id, []).append((index, row))

    selected_discovery_indices: set[int] = set()
    authoritative: dict[str, Mapping[str, Any]] = {}
    for account_id, inventory in inventory_by_id.items():
        same_id = discovery_by_id.get(account_id) or ()
        if len(same_id) == 1:
            index, discovery = same_id[0]
            selected_discovery_indices.add(index)
            authoritative[account_id] = _merged_account_lifecycle_row(
                inventory,
                discovery,
            )
        else:
            authoritative[account_id] = inventory

    unmatched_inventory_ids = [
        account_id
        for account_id in authoritative
        if not (discovery_by_id.get(account_id) or ())
    ]
    unmatched_discovery = {
        index: row
        for index, row in enumerate(discovery_rows)
        if index not in selected_discovery_indices
    }
    discovery_matches: dict[int, set[str]] = {}
    inventory_matches: dict[str, set[int]] = {}
    for account_id in unmatched_inventory_ids:
        inventory = authoritative[account_id]
        for index, discovery in unmatched_discovery.items():
            if _account_lifecycle_anchor_geometry_matches(inventory, discovery):
                inventory_matches.setdefault(account_id, set()).add(index)
                discovery_matches.setdefault(index, set()).add(account_id)
    for account_id, indices in inventory_matches.items():
        if len(indices) != 1:
            continue
        index = next(iter(indices))
        if len(discovery_matches.get(index) or ()) != 1:
            continue
        authoritative[account_id] = _merged_account_lifecycle_row(
            authoritative[account_id],
            unmatched_discovery[index],
        )
        selected_discovery_indices.add(index)

    # A sealed discovery identity that was not present in the captured anchor
    # inventory is still independently authoritative.  Non-sealed rows are
    # intentionally omitted from this owner plane; they remain repair inputs.
    for index, row in enumerate(discovery_rows):
        if index in selected_discovery_indices:
            continue
        account_id = _sealed_discovery_account_id(row)
        if account_id and account_id not in authoritative:
            authoritative[account_id] = row
    return list(authoritative.values())


def _merged_account_lifecycle_row(
    discovery: Mapping[str, Any], repaired: Mapping[str, Any]
) -> dict[str, Any]:
    """Use repaired business values while retaining the sealed entity owner."""

    merged = deepcopy(dict(discovery))
    merged.update(deepcopy(dict(repaired)))
    for nested_name in ("canonical_raw", "raw"):
        discovery_nested = discovery.get(nested_name)
        repaired_nested = repaired.get(nested_name)
        if isinstance(discovery_nested, Mapping) or isinstance(
            repaired_nested, Mapping
        ):
            nested = (
                deepcopy(dict(discovery_nested))
                if isinstance(discovery_nested, Mapping)
                else {}
            )
            if isinstance(repaired_nested, Mapping):
                nested.update(deepcopy(dict(repaired_nested)))
            merged[nested_name] = nested

    for field_name in _ACCOUNT_LIFECYCLE_IDENTITY_FIELDS:
        if field_name in discovery:
            merged[field_name] = deepcopy(discovery[field_name])
    for nested_name in ("canonical_raw", "raw"):
        nested = merged.get(nested_name)
        if not isinstance(nested, dict):
            continue
        for field_name in ("account_id", "account_type", "category_sequence"):
            if field_name in discovery:
                nested[field_name] = deepcopy(discovery[field_name])

    refs: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for owner in (discovery, repaired):
        for raw_ref in owner.get("source_refs") or ():
            if not isinstance(raw_ref, Mapping):
                continue
            ref = deepcopy(dict(raw_ref))
            marker = repr(sorted(ref.items(), key=lambda item: str(item[0])))
            if marker not in seen_refs:
                seen_refs.add(marker)
                refs.append(ref)
    merged["source_refs"] = refs
    return merged


def _record_account_lifecycle_ambiguity(
    context: Any,
    repaired_rows: list[Mapping[str, Any]],
    identifier: str,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    row = repaired_rows[0]
    record_issue(
        context,
        make_issue(
            category="schema_incompleteness",
            issue_code="candidate_b_account_lifecycle_identifier_ambiguous",
            message=(
                "Repair produced a non-unique full account identifier; the "
                "account identity was not guessed across extraction planes."
            ),
            parser_stage="candidate_b_account_lifecycle",
            target_dataset="credit_accounts",
            target_record_id=str(row.get("account_id") or "") or None,
            field_name="account_identifier",
            observed_value=identifier,
            source_refs=(
                ref
                for candidate in repaired_rows
                for ref in candidate.get("source_refs") or ()
                if isinstance(ref, Mapping)
            ),
            reason_codes=("non_unique_cross_plane_account_identifier",),
        ),
    )


def _reconcile_account_population_lifecycle(
    context: Any,
    discovery_rows: list[Mapping[str, Any]],
    repaired_rows: list[Mapping[str, Any]],
    *,
    source_census_rows: tuple[Mapping[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    """Prevent page repair from deleting or relabelling sealed accounts."""

    authoritative_discovery = _authoritative_discovery_account_rows(
        context,
        discovery_rows,
        source_census_rows=source_census_rows,
    )
    source_census_ids = {
        account_id
        for row in source_census_rows
        if (account_id := _exact_source_census_account_id(row))
    }
    sealed_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in authoritative_discovery:
        account_id = _sealed_discovery_account_id(row)
        if account_id is None:
            census_id = _exact_source_census_account_id(row)
            account_id = census_id if census_id in source_census_ids else None
        if account_id:
            sealed_groups.setdefault(account_id, []).append(row)
    sealed_by_id = {
        account_id: rows[0]
        for account_id, rows in sealed_groups.items()
        if len(rows) == 1
    }

    repaired_by_id: dict[str, list[Mapping[str, Any]]] = {}
    repaired_by_identifier: dict[str, list[Mapping[str, Any]]] = {}
    for row in repaired_rows:
        repaired_by_id.setdefault(str(row.get("account_id") or ""), []).append(row)
        if identifier := _account_lifecycle_identifier(row):
            repaired_by_identifier.setdefault(identifier, []).append(row)

    sealed_by_identifier: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for account_id, row in sealed_by_id.items():
        if identifier := _account_lifecycle_identifier(row):
            sealed_by_identifier.setdefault(identifier, []).append((account_id, row))

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        register_issue_target_remap,
    )

    geometry_candidates_by_repaired_index: dict[int, set[str]] = {}
    repaired_indices_by_sealed_id: dict[str, set[int]] = {}
    for repaired_index, repaired in enumerate(repaired_rows):
        for account_id, discovery in sealed_by_id.items():
            if _account_lifecycle_anchor_geometry_matches(discovery, repaired):
                geometry_candidates_by_repaired_index.setdefault(
                    repaired_index,
                    set(),
                ).add(account_id)
                repaired_indices_by_sealed_id.setdefault(account_id, set()).add(
                    repaired_index
                )

    consumed_sealed_ids: set[str] = set()
    reconciled: list[dict[str, Any]] = []
    reported_ambiguities: set[str] = set()
    for repaired_index, repaired in enumerate(repaired_rows):
        repaired_id = str(repaired.get("account_id") or "")
        discovery: Mapping[str, Any] | None = None
        discovery_id: str | None = None
        if (
            repaired_id in sealed_by_id
            and len(repaired_by_id.get(repaired_id) or ()) == 1
        ):
            discovery_id = repaired_id
            discovery = sealed_by_id[repaired_id]
        else:
            identifier = _account_lifecycle_identifier(repaired)
            discovery_matches = sealed_by_identifier.get(identifier or "") or ()
            repaired_matches = repaired_by_identifier.get(identifier or "") or ()
            if identifier and len(discovery_matches) == len(repaired_matches) == 1:
                discovery_id, discovery = discovery_matches[0]
            elif identifier and (len(discovery_matches) > 1 or len(repaired_matches) > 1):
                if identifier not in reported_ambiguities:
                    reported_ambiguities.add(identifier)
                    _record_account_lifecycle_ambiguity(
                        context,
                        list(repaired_matches) or [repaired],
                        identifier,
                    )

        if discovery is None:
            geometry_matches = geometry_candidates_by_repaired_index.get(
                repaired_index,
                set(),
            )
            if len(geometry_matches) == 1:
                geometry_id = next(iter(geometry_matches))
                if len(repaired_indices_by_sealed_id.get(geometry_id) or ()) == 1:
                    discovery_id = geometry_id
                    discovery = sealed_by_id[geometry_id]

        if (
            discovery is None
            or discovery_id is None
            or discovery_id in consumed_sealed_ids
        ):
            reconciled.append(deepcopy(dict(repaired)))
            continue
        consumed_sealed_ids.add(discovery_id)
        reconciled.append(_merged_account_lifecycle_row(discovery, repaired))
        for alias in _account_lifecycle_aliases(repaired) | _account_lifecycle_aliases(
            discovery
        ):
            register_issue_target_remap(context, alias, discovery_id)

    for account_id, row in sealed_by_id.items():
        if account_id in consumed_sealed_ids:
            continue
        reconciled.append(deepcopy(dict(row)))
        for alias in _account_lifecycle_aliases(row):
            register_issue_target_remap(context, alias, account_id)

    def sequence_key(row: Mapping[str, Any]) -> tuple[int, int]:
        try:
            return 0, int(row.get("sequence"))
        except (TypeError, ValueError):
            return 1, 0

    reconciled.sort(key=sequence_key)
    return reconciled


def _repayment_performance_month(record: Mapping[str, Any]) -> str | None:
    """Return an exact YYYY-MM key, never a fuzzy date interpretation."""

    explicit = str(record.get("performance_month") or "").strip()
    if re.fullmatch(r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])", explicit):
        return explicit
    year = record.get("year")
    month = record.get("month")
    if isinstance(year, bool) or isinstance(month, bool):
        return None
    try:
        year_number = int(str(year).strip())
        month_number = int(str(month).strip())
    except (TypeError, ValueError):
        return None
    if not 1900 <= year_number <= 2099 or not 1 <= month_number <= 12:
        return None
    return f"{year_number:04d}-{month_number:02d}"


def _repayment_grid_month_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    grid_id = str(record.get("grid_id") or "").strip()
    performance_month = _repayment_performance_month(record)
    if not grid_id or performance_month is None:
        return None
    return grid_id, performance_month


def _unique_rows_by_key(
    rows: list[Mapping[str, Any]], key_loader: Any
) -> dict[Any, Mapping[str, Any]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = key_loader(row)
        if key not in (None, ""):
            grouped.setdefault(key, []).append(row)
    return {key: matches[0] for key, matches in grouped.items() if len(matches) == 1}


def _repayment_status_field(record: Mapping[str, Any]) -> tuple[str, Any] | None:
    for field_name in ("status", "status_code"):
        value = record.get(field_name)
        normalized = _individualized_scalar_key(value).upper()
        if normalized not in _CANONICAL_REPAYMENT_STATUS_VALUES:
            continue
        raw_value = _field_raw(record, field_name, value)
        raw_normalized = _individualized_scalar_key(raw_value).upper()
        if raw_normalized in _CANONICAL_REPAYMENT_STATUS_VALUES:
            return field_name, raw_value
    return None


def _exact_repayment_status_refs(
    record: Mapping[str, Any],
    field_name: str,
    evidence_plane: str,
) -> list[dict[str, Any]]:
    """Return exact source-bound refs for one canonical monthly status cell."""

    grid_id = str(record.get("grid_id") or "").strip()
    performance_month = _repayment_performance_month(record)
    if not grid_id or performance_month is None:
        return []
    expected_month = int(performance_month[-2:])
    field_aliases = ("status", "status_code")
    refs: list[Any] = []
    refs_by_field = record.get("source_refs_by_field")
    if isinstance(refs_by_field, Mapping):
        for alias in field_aliases:
            values = refs_by_field.get(alias)
            if isinstance(values, (list, tuple)):
                refs.extend(values)
    if not refs:
        values = record.get("source_cell_refs")
        if isinstance(values, (list, tuple)):
            refs.extend(values)
    if not refs:
        values = record.get("source_refs")
        if isinstance(values, (list, tuple)):
            refs.extend(values)

    exact: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        ref_field = str(ref.get("field_name") or field_name).strip()
        bbox = ref.get("bbox")
        try:
            row = int(ref["row"])
            month = int(ref["col"])
            page = int(ref.get("logical_page") or ref.get("page") or 0)
            coordinates = tuple(float(value) for value in bbox)
        except (KeyError, TypeError, ValueError):
            continue
        if not (
            ref_field in field_aliases
            and str(ref.get("geometry_scope") or "") == "cell"
            and str(ref.get("geometry_status") or "") != "unresolved"
            and str(ref.get("grid_id") or "") == grid_id
            and row >= 0
            and month == expected_month
            and page > 0
            and len(coordinates) == 4
            and all(isfinite(value) for value in coordinates)
            and coordinates[2] > coordinates[0]
            and coordinates[3] > coordinates[1]
        ):
            continue
        exact.append({**dict(ref), "evidence_plane": evidence_plane})
    return exact


def _withhold_repayment_plane_conflicts(
    context: Any,
    native_datasets: Mapping[str, Any],
    corrected_datasets: Mapping[str, Any],
) -> None:
    """Withhold status when the two exact source-bound repayment planes disagree."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    native_rows = [
        row
        for row in native_datasets.get("repayment_records") or ()
        if isinstance(row, Mapping)
    ]
    corrected_rows = [
        row
        for row in corrected_datasets.get("repayment_records") or ()
        if isinstance(row, dict)
    ]
    if not native_rows or not corrected_rows:
        return

    def repayment_id(row: Mapping[str, Any]) -> str | None:
        value = str(row.get("repayment_id") or "").strip()
        return value or None

    native_by_id = _unique_rows_by_key(native_rows, repayment_id)
    corrected_by_id = _unique_rows_by_key(corrected_rows, repayment_id)
    native_by_grid_month = _unique_rows_by_key(native_rows, _repayment_grid_month_key)
    corrected_by_grid_month = _unique_rows_by_key(
        corrected_rows, _repayment_grid_month_key
    )

    for corrected in corrected_rows:
        native: Mapping[str, Any] | None = None
        corrected_id = repayment_id(corrected)
        if corrected_id and corrected_by_id.get(corrected_id) is corrected:
            native = native_by_id.get(corrected_id)
            if native is not None:
                native_month = _repayment_performance_month(native)
                corrected_month = _repayment_performance_month(corrected)
                if (
                    native_month is not None
                    and corrected_month is not None
                    and native_month != corrected_month
                ):
                    native = None
        if native is None:
            fallback_key = _repayment_grid_month_key(corrected)
            if (
                fallback_key is not None
                and corrected_by_grid_month.get(fallback_key) is corrected
            ):
                native = native_by_grid_month.get(fallback_key)
        if native is None:
            continue

        native_status = _repayment_status_field(native)
        corrected_status = _repayment_status_field(corrected)
        if native_status is None or corrected_status is None:
            continue
        native_field, native_value = native_status
        corrected_field, corrected_value = corrected_status
        native_refs = _exact_repayment_status_refs(
            native,
            native_field,
            "native_static",
        )
        corrected_refs = _exact_repayment_status_refs(
            corrected,
            corrected_field,
            "corrected_page",
        )
        if not native_refs or not corrected_refs:
            continue
        if _individualized_scalar_key(native_value).upper() == _individualized_scalar_key(
            corrected_value
        ).upper():
            continue

        native_raw = native_value
        corrected_raw = corrected_value
        refs = [*native_refs, *corrected_refs]
        for field_name in ("status", "status_code"):
            corrected.pop(field_name, None)
        unresolved = corrected.setdefault("_unresolved_fields", [])
        if "status" not in unresolved:
            unresolved.append("status")
        corrected["extraction_status"] = "review"
        corrected.setdefault("canonical_raw", {})["status"] = [
            native_raw,
            corrected_raw,
        ]
        grid_month = _repayment_grid_month_key(corrected)
        target_record_id = (
            corrected_id
            or repayment_id(native)
            or str(corrected.get("record_id") or "").strip()
            or (f"{grid_month[0]}:{grid_month[1]}" if grid_month else "")
        )
        record_issue(
            context,
            make_issue(
                category="ocr_cell_level_error",
                issue_code="candidate_b_independent_plane_repayment_status_conflict",
                message=(
                    "Independent source-bound OCR planes disagreed on a monthly "
                    "repayment status; the status was withheld without choosing a plane."
                ),
                parser_stage="candidate_b_cross_plane_repayment_reconciliation",
                target_dataset="repayment_records",
                target_record_id=target_record_id,
                field_name="status_code",
                observed_value={
                    "native_static": native_raw,
                    "corrected_page": corrected_raw,
                },
                source_refs=refs,
                reason_codes=(
                    "exact_repayment_identity",
                    "independent_source_bound_observations",
                    "monthly_status_conflict",
                    "normalized_value_withheld",
                ),
            ),
        )


def _withhold_independent_plane_conflicts(
    context: Any,
    native_datasets: Mapping[str, Any],
    corrected_datasets: Mapping[str, Any],
) -> None:
    """Withhold individualized scalars when two source-bound OCR planes disagree.

    This runs only after the existing one-shot page repair produced a second
    schema pass.  It never attempts fuzzy correction: both glyph sequences are
    retained as issue evidence and neither is published as business truth.
    """

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    for dataset, identity_field, field_names in _CROSS_PLANE_INDIVIDUALIZED_FIELDS:
        native_rows = [row for row in native_datasets.get(dataset) or () if isinstance(row, Mapping)]
        corrected_rows = [
            row for row in corrected_datasets.get(dataset) or () if isinstance(row, dict)
        ]
        native_by_id = {
            str(row.get(identity_field) or ""): row
            for row in native_rows
            if row.get(identity_field)
        }
        for corrected in corrected_rows:
            record_id = str(corrected.get(identity_field) or "")
            native = native_by_id.get(record_id)
            if native is None:
                continue
            for field_name in field_names:
                native_value = native.get(field_name)
                corrected_value = corrected.get(field_name)
                if native_value in (None, "") or corrected_value in (None, ""):
                    continue
                if _individualized_scalar_key(native_value) == _individualized_scalar_key(
                    corrected_value
                ):
                    continue
                native_raw = _field_raw(native, field_name, native_value)
                corrected_raw = _field_raw(corrected, field_name, corrected_value)
                corrected.pop(field_name, None)
                unresolved = corrected.setdefault("_unresolved_fields", [])
                if field_name not in unresolved:
                    unresolved.append(field_name)
                corrected["extraction_status"] = "review"
                corrected.setdefault("canonical_raw", {})[field_name] = [
                    native_raw,
                    corrected_raw,
                ]
                refs = [
                    *_plane_refs(native, field_name, "native_static"),
                    *_plane_refs(corrected, field_name, "corrected_page"),
                ]
                record_issue(
                    context,
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_independent_plane_field_conflict",
                        message=(
                            "Independent source-bound OCR planes disagreed on an "
                            "individualized field; the value was withheld without guessing."
                        ),
                        parser_stage="candidate_b_cross_plane_field_reconciliation",
                        target_dataset=dataset,
                        target_record_id=record_id,
                        field_name=field_name,
                        observed_value={
                            "native_static": native_raw,
                            "corrected_page": corrected_raw,
                        },
                        source_refs=refs,
                        reason_codes=(
                            "independent_source_bound_observations",
                            "individualized_glyph_conflict",
                            "normalized_value_withheld",
                        ),
                    ),
                )
    _withhold_repayment_plane_conflicts(
        context,
        native_datasets,
        corrected_datasets,
    )


def _overdue_view_input_basis(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    """Snapshot only the inputs consumed by the existing derived view."""

    return tuple(
        tuple(deepcopy({key: row.get(key) for key in keys}) for row in payload.get(dataset) or () if isinstance(row, Mapping))
        for dataset, keys in (
            ("credit_accounts", ("account_id", "account_type", "credit_card_type", "account_status", "account_state", "status", "five_tier_class", "overdue_amount", "current_overdue_amount", "source_refs", "confidence")),
            ("repayment_records", ("account_id", "status", "status_code", "year", "month", "performance_month", "overdue_amount", "status_amount", "source_cell_refs", "confidence")),
        )
    )


def _refresh_final_overdue_view(payload: dict[str, Any], previous_basis: tuple[Any, ...]) -> None:
    if _overdue_view_input_basis(payload) == previous_basis:
        return
    from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
        derive_candidate_b_overdue_records,
    )

    payload["overdue_records"] = derive_candidate_b_overdue_records(
        list(payload.get("credit_accounts") or ()), list(payload.get("repayment_records") or ()),
    )


def _canonical_account_issue_field(field_name: Any) -> str:
    field = str(field_name or "")
    return "account_currency" if field in {"currency", "account_currency"} else field


def _account_record_values(record: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = record.get("normalized")
    return normalized if isinstance(normalized, Mapping) else record


def _account_field_value(record: Mapping[str, Any], field_name: str) -> Any:
    values = _account_record_values(record)
    observed = [
        owner.get(alias)
        for owner in (values, record)
        for alias in _FINAL_ACCOUNT_FIELD_ALIASES[field_name]
        if isinstance(owner, Mapping) and owner.get(alias) not in (None, "")
    ]
    distinct = {"".join(str(value).split()) for value in observed}
    return observed[0] if observed and len(distinct) == 1 else None


def _account_field_refs(
    record: Mapping[str, Any], field_name: str
) -> list[dict[str, Any]]:
    values = _account_record_values(record)
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for owner in (values, record):
        if not isinstance(owner, Mapping):
            continue
        refs_by_field = owner.get("source_refs_by_field")
        if not isinstance(refs_by_field, Mapping):
            continue
        for alias in _FINAL_ACCOUNT_FIELD_ALIASES[field_name]:
            for ref in refs_by_field.get(alias) or ():
                if not isinstance(ref, Mapping):
                    continue
                normalized_ref = dict(ref)
                marker = repr(sorted(normalized_ref.items(), key=lambda item: str(item[0])))
                if marker in seen:
                    continue
                seen.add(marker)
                refs.append(normalized_ref)
    return refs


def _account_field_is_marked(
    record: Mapping[str, Any], marker_name: str, field_name: str
) -> bool:
    aliases = set(_FINAL_ACCOUNT_FIELD_ALIASES[field_name])
    return any(
        aliases.intersection(str(value) for value in owner.get(marker_name) or ())
        for owner in (_account_record_values(record), record)
        if isinstance(owner, Mapping)
    )


def _final_account_field_is_valid(field_name: str, value: Any) -> bool:
    from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
        validate_pboc_field,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
        _exact_source_account_institution,
        _typed_identifier,
    )

    text = str(value or "").strip()
    if not text:
        return False
    if field_name == "account_currency":
        return validate_pboc_field(text, "currency").valid
    if field_name == "business_type":
        return validate_pboc_field(text, "account_business_type").valid
    if field_name == "guarantee_type":
        return validate_pboc_field(text, "guarantee_type").valid
    if field_name == "account_identifier":
        return _typed_identifier(text) is not None
    if field_name == "management_institution":
        compact = "".join(text.split())
        normalized = _exact_source_account_institution(
            text,
            independently_corroborated=True,
        )
        return bool(
            compact
            and any(root in compact for root in _ACCOUNT_INSTITUTION_LEGAL_ROOTS)
            and normalized is not None
        )
    return False


def _source_ref_locator(ref: Mapping[str, Any]) -> tuple[Any, ...] | None:
    page = int(ref.get("source_page") or ref.get("logical_page") or 0)
    table_id = str(ref.get("table_id") or "")
    row = ref.get("row")
    column = ref.get("column")
    if table_id and isinstance(row, int):
        return ("table_cell", page, table_id, row, column)
    bbox = ref.get("bbox")
    if page and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return ("bbox", page, *(round(float(value), 4) for value in bbox))
        except (TypeError, ValueError):
            return None
    evidence_ids = tuple(str(value) for value in ref.get("evidence_ids") or () if value)
    if page and evidence_ids:
        return ("evidence", page, *evidence_ids)
    return None


def _account_table_provenance_keys(value: Mapping[str, Any]) -> set[tuple[str, int, str]]:
    """Return exact physical-table keys usable only for diagnostic linkage."""

    refs = [
        ref
        for ref in value.get("source_refs") or ()
        if isinstance(ref, Mapping)
    ]
    refs_by_field = value.get("source_refs_by_field")
    if isinstance(refs_by_field, Mapping):
        refs.extend(
            ref
            for field_refs in refs_by_field.values()
            for ref in field_refs or ()
            if isinstance(ref, Mapping)
        )
    keys: set[tuple[str, int, str]] = set()
    for ref in refs:
        if str(ref.get("source") or "") not in {
            "native_detail_table",
            "native_detail_table_cell",
        }:
            continue
        table_id = str(ref.get("table_id") or "").strip()
        if not table_id:
            continue
        for plane in ("logical_page", "source_page"):
            try:
                page = int(ref.get(plane) or 0)
            except (TypeError, ValueError):
                continue
            if page > 0:
                keys.add((plane, page, table_id))
    return keys


def _account_ref_is_exact_for_field(ref: Mapping[str, Any], field_name: str) -> bool:
    binding = str(ref.get("binding_quality") or ref.get("binding") or "")
    if binding not in _EXACT_ACCOUNT_FIELD_BINDINGS:
        if not (
            ref.get("geometry_scope") == "canonical_field_slot"
            and isinstance(ref.get("row"), int)
            and isinstance(ref.get("column"), int)
        ):
            return False
    ref_field = _canonical_account_issue_field(ref.get("field_name"))
    return not ref_field or ref_field == field_name


def _account_ref_is_corrected_currency_cell(
    ref: Mapping[str, Any], field_name: str
) -> bool:
    """Require value-associated corrected-cell proof for a blank currency slot."""

    if field_name != "account_currency":
        return False
    if str(ref.get("source") or "") != "personal_detail_corrected_page_cell":
        return False
    if str(ref.get("geometry_scope") or "") != "cell":
        return False
    binding = str(ref.get("binding_quality") or ref.get("binding") or "")
    if binding != "canonical_field_slot":
        return False
    if _canonical_account_issue_field(ref.get("field_name")) != field_name:
        return False
    if not any(str(value or "").strip() for value in ref.get("evidence_ids") or ()):
        return False
    bbox = ref.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        finite_bbox = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return all(isfinite(value) for value in finite_bbox)


def _clear_resolved_account_field_markers(
    record: dict[str, Any], field_name: str
) -> None:
    aliases = set(_FINAL_ACCOUNT_FIELD_ALIASES[field_name])
    for owner in (_account_record_values(record), record):
        if not isinstance(owner, dict):
            continue
        for marker_name in (
            "_unresolved_fields",
            "_invalid_observation_fields",
            "_reported_invalid_fields",
        ):
            retained = [
                value
                for value in owner.get(marker_name) or ()
                if str(value) not in aliases
            ]
            if retained:
                owner[marker_name] = retained
            else:
                owner.pop(marker_name, None)


def _reconcile_final_account_field_issues(
    context: Any, records: Any
) -> None:
    """Drop only bad-alternate diagnostics superseded by exact final evidence.

    Closed-cell and anchor-geometry decoders deliberately report their rejected
    observations immediately.  A later exact, field-valid observation can win
    the final account merge, so those early issues require one final lifecycle
    check.  Conflicts, source absence, same-cell retries, and unsupported fields
    are never closed here.
    """

    issues = getattr(context, "_personal_detail_extraction_issues", None)
    if not isinstance(issues, list):
        return
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        _resolve_issue_target,
        register_issue_target_remap,
    )

    records_by_id = {
        str(_account_record_values(record).get("account_id") or record.get("account_id") or ""): record
        for record in records or ()
        if isinstance(record, dict)
    }
    owners_by_table_key: dict[tuple[str, int, str], set[str]] = {}
    for record_id, record in records_by_id.items():
        if not record_id:
            continue
        for key in _account_table_provenance_keys(record):
            owners_by_table_key.setdefault(key, set()).add(record_id)
    for issue in issues:
        if (
            not isinstance(issue, Mapping)
            or str(issue.get("target_dataset") or "") != "credit_accounts"
        ):
            continue
        source_id = str(issue.get("target_record_id") or "").strip()
        if not source_id or source_id in records_by_id:
            continue
        owner_ids = {
            owner_id
            for key in _account_table_provenance_keys(issue)
            for owner_id in owners_by_table_key.get(key, ())
        }
        if len(owner_ids) == 1:
            register_issue_target_remap(
                context,
                source_id,
                next(iter(owner_ids)),
            )

    target_remaps = getattr(context, "_personal_detail_issue_target_remaps", None)

    def issue_pair(issue: Mapping[str, Any]) -> tuple[str, str] | None:
        field_name = _canonical_account_issue_field(issue.get("field_name"))
        if field_name not in _FINAL_ACCOUNT_FIELD_ALIASES:
            return None
        record_id = str(issue.get("target_record_id") or "")
        if record_id and isinstance(target_remaps, Mapping):
            remapped_id, ambiguous = _resolve_issue_target(target_remaps, record_id)
            if ambiguous:
                return None
            record_id = remapped_id or record_id
        return record_id, field_name

    active_by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        if str(issue.get("target_dataset") or "") != "credit_accounts":
            continue
        if str(issue.get("status") or "requires_review") in _INACTIVE_ISSUE_STATUSES:
            continue
        pair = issue_pair(issue)
        if pair is None:
            continue
        active_by_pair.setdefault(pair, []).append(issue)

    resolved_pairs: set[tuple[str, str]] = set()
    for (record_id, field_name), pair_issues in active_by_pair.items():
        record = records_by_id.get(record_id)
        if record is None:
            continue
        if any(
            str(issue.get("issue_code") or "")
            not in _ACCOUNT_ISSUES_SUPERSEDED_BY_EXACT_FINAL
            for issue in pair_issues
        ):
            continue
        if _account_field_is_marked(record, "_reported_field_conflicts", field_name):
            continue
        if _account_field_is_marked(record, "_source_absent_fields", field_name):
            continue
        value = _account_field_value(record, field_name)
        if not _final_account_field_is_valid(field_name, value):
            continue

        issue_locators: set[tuple[Any, ...]] = set()
        issue_sources_complete = True
        for issue in pair_issues:
            locators = {
                locator
                for ref in issue.get("source_refs") or ()
                if isinstance(ref, Mapping)
                if (locator := _source_ref_locator(ref)) is not None
            }
            if not locators:
                issue_sources_complete = False
                break
            issue_locators.update(locators)
        if not issue_sources_complete:
            continue

        final_refs = [
            ref
            for ref in _account_field_refs(record, field_name)
            if _account_ref_is_exact_for_field(ref, field_name)
        ]
        if (
            field_name == "account_currency"
            and any(
                str(issue.get("issue_code") or "")
                in _CURRENCY_ISSUES_REQUIRING_CORRECTED_CELL_EVIDENCE
                for issue in pair_issues
            )
        ):
            final_refs = [
                ref
                for ref in final_refs
                if _account_ref_is_corrected_currency_cell(ref, field_name)
            ]
        exact_final_locators = {
            locator
            for ref in final_refs
            if (locator := _source_ref_locator(ref)) is not None
        }
        if not exact_final_locators.difference(issue_locators):
            continue
        resolved_pairs.add((record_id, field_name))

    if not resolved_pairs:
        return
    retained_issues: list[Any] = []
    for issue in issues:
        pair = issue_pair(issue) if isinstance(issue, Mapping) else None
        if (
            isinstance(issue, Mapping)
            and str(issue.get("target_dataset") or "") == "credit_accounts"
            and str(issue.get("issue_code") or "")
            in _ACCOUNT_ISSUES_SUPERSEDED_BY_EXACT_FINAL
            and pair in resolved_pairs
        ):
            continue
        retained_issues.append(issue)
    context._personal_detail_extraction_issues = retained_issues
    for record_id, field_name in resolved_pairs:
        _clear_resolved_account_field_markers(records_by_id[record_id], field_name)


@dataclass(frozen=True)
class CandidateBExtraction:
    """One immutable-by-convention result shared by every variant adapter hook."""

    business: dict[str, Any]
    section_content: dict[str, Any]
    audit: dict[str, Any]


@dataclass
class _StagePayload:
    """Detached output and diagnostic effects of one source extraction stage."""

    business: dict[str, Any] = dataclass_field(default_factory=dict)
    datasets: dict[str, list[dict[str, Any]]] = dataclass_field(
        default_factory=dict
    )
    profile: dict[str, Any] | None = None
    status_glyph_observations: list[dict[str, Any]] = dataclass_field(
        default_factory=list
    )
    context_cache_entries: dict[str, Any] = dataclass_field(
        default_factory=dict
    )
    context_attributes: dict[str, Any] = dataclass_field(default_factory=dict)
    removed_issue_ids: tuple[str, ...] = ()
    upserted_issues: tuple[dict[str, Any], ...] = ()
    removed_remap_edges: tuple[tuple[str, str], ...] = ()
    added_remap_edges: tuple[tuple[str, str], ...] = ()


def _preserve_discovery_stage_outputs_on_empty_repair(
    stage: Any,
    discovery_payload: _StagePayload,
    repaired_payload: _StagePayload,
) -> bool:
    """Keep source-backed discovery output when a repair loses the whole stage.

    Repair is an evidence overlay, not permission to erase a section that was
    already materialized from the same immutable document.  This guard is
    intentionally stage-atomic and narrow: it acts only when every declared
    repaired output is empty and at least one discovery output is non-empty.
    Partial repaired populations are left untouched for their dataset-specific
    reconciliation policies.
    """

    def output_value(payload: _StagePayload, output_name: str) -> Any:
        if output_name == "status_glyph_observations":
            return payload.status_glyph_observations
        if output_name == "subject_profile":
            return payload.profile
        if output_name in payload.business:
            return payload.business[output_name]
        return payload.datasets.get(output_name)

    def has_output(value: Any) -> bool:
        if isinstance(value, Mapping):
            return bool(value)
        if isinstance(value, (list, tuple, set, frozenset)):
            return bool(value)
        return value is not None

    output_names = tuple(getattr(stage, "output_names", ()) or ())
    if not output_names:
        return False
    discovery_values = {
        output_name: output_value(discovery_payload, output_name)
        for output_name in output_names
    }
    repaired_values = {
        output_name: output_value(repaired_payload, output_name)
        for output_name in output_names
    }
    if any(has_output(value) for value in repaired_values.values()) or not any(
        has_output(value) for value in discovery_values.values()
    ):
        return False

    for output_name, value in discovery_values.items():
        if output_name == "status_glyph_observations":
            repaired_payload.status_glyph_observations = deepcopy(value or [])
        elif output_name == "subject_profile":
            repaired_payload.profile = deepcopy(value)
        elif output_name in discovery_payload.business:
            repaired_payload.business[output_name] = deepcopy(value)
        elif output_name in discovery_payload.datasets:
            repaired_payload.datasets[output_name] = deepcopy(value)
    return True


def _reconcile_repaired_account_stage_payload(
    context: Any,
    discovery_payload: _StagePayload,
    repaired_payload: _StagePayload,
) -> None:
    """Conserve account owners before any repaired dependent stage executes."""

    discovery_rows = [
        row
        for row in discovery_payload.business.get("credit_accounts") or ()
        if isinstance(row, Mapping)
    ]
    repaired_rows = [
        row
        for row in repaired_payload.business.get("credit_accounts") or ()
        if isinstance(row, Mapping)
    ]
    reconciled = _reconcile_account_population_lifecycle(
        context,
        discovery_rows,
        repaired_rows,
    )
    if len(reconciled) < len(discovery_rows):
        source_census_rows = _exact_source_census_account_rows(context)
        if source_census_rows:
            reconciled = _reconcile_account_population_lifecycle(
                context,
                discovery_rows,
                repaired_rows,
                source_census_rows=source_census_rows,
            )
    repaired_payload.business["credit_accounts"] = reconciled

    cached_collections = repaired_payload.context_cache_entries.get(
        "account_collections"
    )
    if not (
        isinstance(cached_collections, (list, tuple))
        and len(cached_collections) == 3
    ):
        context_cache = getattr(context, "_cache", None)
        cached_collections = (
            context_cache.get("account_collections")
            if isinstance(context_cache, Mapping)
            else None
        )
    if isinstance(cached_collections, (list, tuple)) and len(cached_collections) == 3:
        updated_collections = (
            deepcopy(reconciled),
            deepcopy(cached_collections[1]),
            deepcopy(cached_collections[2]),
        )
        repaired_payload.context_cache_entries[
            "account_collections"
        ] = updated_collections
        context_cache = getattr(context, "_cache", None)
        if isinstance(context_cache, dict):
            context_cache["account_collections"] = deepcopy(updated_collections)


@dataclass(frozen=True)
class _SourcePassResult:
    business: dict[str, Any]
    datasets: dict[str, list[dict[str, Any]]]
    profile: dict[str, Any]
    status_glyph_observations: list[dict[str, Any]]
    stage_payloads: dict[str, _StagePayload]
    stage_snapshots: dict[str, Any]
    derived_issue_ids: frozenset[str]


def _issue_id(issue: Mapping[str, Any]) -> str:
    return str(
        issue.get("extraction_issue_id")
        or issue.get("record_id")
        or ""
    ).strip()


def _issue_rows(context: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        deepcopy(dict(issue))
        for issue in getattr(context, "_personal_detail_extraction_issues", ())
        if isinstance(issue, Mapping)
    )


def _remap_edges(context: Any) -> frozenset[tuple[str, str]]:
    registry = getattr(context, "_personal_detail_issue_target_remaps", None)
    if not isinstance(registry, Mapping):
        return frozenset()
    edges: set[tuple[str, str]] = set()
    for source, raw_targets in registry.items():
        source_id = str(source or "").strip()
        targets = (raw_targets,) if isinstance(raw_targets, str) else raw_targets or ()
        edges.update(
            (source_id, str(target or "").strip())
            for target in targets
            if source_id and str(target or "").strip()
        )
    return frozenset(edges)


def _remap_registry_from_edges(
    edges: set[tuple[str, str]] | frozenset[tuple[str, str]],
) -> dict[str, set[str]]:
    registry: dict[str, set[str]] = {}
    for source, target in sorted(edges):
        registry.setdefault(source, set()).add(target)
    return registry


@contextmanager
def _candidate_b_extraction_stage(context: Any, stage_name: str):
    """Tag diagnostics without changing any extractor's call contract."""

    attribute = "_candidate_b_active_extraction_stage"
    marker = object()
    previous = getattr(context, attribute, marker)
    setattr(context, attribute, stage_name)
    try:
        yield
    finally:
        if previous is marker:
            try:
                delattr(context, attribute)
            except AttributeError:
                pass
        else:
            setattr(context, attribute, previous)


def _stage_diagnostic_delta(
    before_issues: tuple[dict[str, Any], ...],
    before_edges: frozenset[tuple[str, str]],
    context: Any,
) -> tuple[
    tuple[str, ...],
    tuple[dict[str, Any], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    after_issues = _issue_rows(context)
    before_by_id = {
        issue_id: issue
        for issue in before_issues
        if (issue_id := _issue_id(issue))
    }
    after_by_id = {
        issue_id: issue
        for issue in after_issues
        if (issue_id := _issue_id(issue))
    }
    removed = tuple(sorted(set(before_by_id).difference(after_by_id)))
    upserted = tuple(
        issue
        for issue in after_issues
        if (issue_id := _issue_id(issue))
        and before_by_id.get(issue_id) != issue
    )
    after_edges = _remap_edges(context)
    return (
        removed,
        upserted,
        tuple(sorted(before_edges.difference(after_edges))),
        tuple(sorted(after_edges.difference(before_edges))),
    )


def _mark_direct_stage_diagnostics(
    context: Any,
    stage_name: str,
    payload: _StagePayload,
) -> None:
    issue_owners = getattr(context, "_candidate_b_issue_stage_owners", None)
    if not isinstance(issue_owners, dict):
        issue_owners = {}
        context._candidate_b_issue_stage_owners = issue_owners
    for issue in payload.upserted_issues:
        if issue_id := _issue_id(issue):
            owners = issue_owners.setdefault(issue_id, set())
            if not isinstance(owners, set):
                owners = set(owners or ())
                issue_owners[issue_id] = owners
            owners.add(stage_name)

    remap_owners = getattr(context, "_candidate_b_remap_stage_owners", None)
    if not isinstance(remap_owners, dict):
        remap_owners = {}
        context._candidate_b_remap_stage_owners = remap_owners
    for edge in payload.added_remap_edges:
        owners = remap_owners.setdefault(edge, set())
        if not isinstance(owners, set):
            owners = set(owners or ())
            remap_owners[edge] = owners
        owners.add(stage_name)


def _merge_stage_diagnostic_delta(
    payload: _StagePayload,
    delta: tuple[
        tuple[str, ...],
        tuple[dict[str, Any], ...],
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
    ],
) -> None:
    """Merge a later callback's effects into its reusable stage snapshot."""

    removed_issue_ids = set(payload.removed_issue_ids)
    upserted_issues = {
        issue_id: issue
        for issue in payload.upserted_issues
        if (issue_id := _issue_id(issue))
    }
    for issue_id in delta[0]:
        removed_issue_ids.add(issue_id)
        upserted_issues.pop(issue_id, None)
    for issue in delta[1]:
        if issue_id := _issue_id(issue):
            removed_issue_ids.discard(issue_id)
            upserted_issues[issue_id] = issue
    payload.removed_issue_ids = tuple(sorted(removed_issue_ids))
    payload.upserted_issues = tuple(upserted_issues.values())

    removed_edges = set(payload.removed_remap_edges)
    added_edges = set(payload.added_remap_edges)
    for edge in delta[2]:
        removed_edges.add(edge)
        added_edges.discard(edge)
    for edge in delta[3]:
        removed_edges.discard(edge)
        added_edges.add(edge)
    payload.removed_remap_edges = tuple(sorted(removed_edges))
    payload.added_remap_edges = tuple(sorted(added_edges))


def _apply_reused_stage_diagnostics(
    context: Any,
    stage_name: str,
    payload: _StagePayload,
    *,
    discovery_issues: tuple[dict[str, Any], ...],
    discovery_issue_owners: Mapping[str, Any],
    discovery_remap_owners: Mapping[tuple[str, str], Any],
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        record_issue,
        register_issue_target_remap,
    )

    context_cache = getattr(context, "_cache", None)
    if isinstance(context_cache, dict):
        context_cache.update(deepcopy(payload.context_cache_entries))
    for attribute_name, value in payload.context_attributes.items():
        setattr(context, attribute_name, deepcopy(value))

    removed_ids = set(payload.removed_issue_ids)
    if removed_ids:
        context._personal_detail_extraction_issues = [
            issue
            for issue in getattr(context, "_personal_detail_extraction_issues", ())
            if not isinstance(issue, Mapping) or _issue_id(issue) not in removed_ids
        ]

    active_rows = getattr(context, "_personal_detail_extraction_issues", None)
    if not isinstance(active_rows, list):
        active_rows = []
        context._personal_detail_extraction_issues = active_rows
    index_by_id = {
        issue_id: index
        for index, issue in enumerate(active_rows)
        if isinstance(issue, Mapping) and (issue_id := _issue_id(issue))
    }
    with _candidate_b_extraction_stage(context, stage_name):
        for issue in payload.upserted_issues:
            issue_id = _issue_id(issue)
            if issue_id in index_by_id:
                active_rows[index_by_id[issue_id]] = deepcopy(issue)
            else:
                record_issue(context, issue)
                index_by_id[issue_id] = len(active_rows) - 1

        # Duplicate suppression can make a stage's local delta empty even
        # though it independently owns the diagnostic. Restore such rows from
        # the detached final discovery ledger.
        for issue in discovery_issues:
            issue_id = _issue_id(issue)
            owners = discovery_issue_owners.get(issue_id) or ()
            if issue_id and stage_name in owners:
                # ``record_issue`` records this stage as an owner before its
                # duplicate check, so call it even when an earlier stage has
                # already restored the shared row.
                record_issue(context, issue)

        edges = set(_remap_edges(context))
        edges.difference_update(payload.removed_remap_edges)
        context._personal_detail_issue_target_remaps = _remap_registry_from_edges(edges)
        for source, target in payload.added_remap_edges:
            register_issue_target_remap(context, source, target)
        edges = set(_remap_edges(context))
        for edge, owners in discovery_remap_owners.items():
            if stage_name in (owners or ()):
                # As with issues, the idempotent remap API also restores every
                # emitting stage's private ownership sidecar.
                register_issue_target_remap(context, *edge)
                edges.add(edge)


class CandidateBPipeline:
    """Extract registered canonical pages into the PBOC source schema once."""

    def __init__(
        self,
        context: Any,
        full_text: str,
        *,
        extraction_strategy: Any | None = None,
    ) -> None:
        self.context = context
        self.full_text = str(full_text or "")
        if extraction_strategy is None:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
                LazyExtractionStrategy,
            )

            extraction_strategy = LazyExtractionStrategy()
        self.extraction_strategy = extraction_strategy

    def run(self) -> CandidateBExtraction:
        from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
            CANDIDATE_B_STAGE_REGISTRY,
            candidate_b_repair_scope,
            plan_candidate_b_initial_extraction,
            section_census_from_canonical_audit,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger import (
            apply_document_consistency_ledger,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.document_glyph_bank import (
            apply_document_local_status_glyph_bank,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            collect_extraction_issues,
            dataset_states_from_issues,
            register_final_liability_issue_records,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
            StageSnapshot,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            _enforce_employment_record_contracts,
            _extract_credit_lines,
            _extract_employment_records,
            _extract_header_datasets,
            _extract_inquiries,
            _extract_liabilities,
            _extract_personal_notes,
            _extract_postpaid_payment_history,
            _extract_postpaid_records,
            _extract_profile_detail_records,
            _extract_public_records,
            _extract_recovery_records,
            _extract_residence_records,
            _extract_source_rows,
            _extract_summary_datasets,
            _record_pre_repair_source_gaps,
            _source_completeness_ledger,
            reconcile_candidate_b_account_sequence_issues,
            reconcile_candidate_b_credit_lines,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
            apply_candidate_b_native_status_conflict_guard,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
            extract_candidate_b_profile,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
            candidate_b_repayment_anchor_ledger,
            derive_candidate_b_overdue_records,
            link_candidate_b_repayments,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
            prepare_personal_detail_source_collections,
        )

        def status_glyph_observations() -> list[dict[str, Any]]:
            loader = getattr(
                self.context,
                "candidate_b_status_glyph_observations",
                None,
            )
            return list(loader() or ()) if callable(loader) else []

        def materialize_stage(
            stage_name: str,
            payloads: Mapping[str, _StagePayload],
        ) -> _StagePayload:
            """Invoke one unchanged extractor behind its strategy stage."""

            if stage_name == "account_inventory":
                accounts, discarded, account_events = (
                    self.context.account_collections()
                )
                account_context_attributes = {}
                if hasattr(
                    self.context,
                    "_candidate_b_account_anchor_skeleton_cache",
                ):
                    account_context_attributes[
                        "_candidate_b_account_anchor_skeleton_cache"
                    ] = getattr(
                        self.context,
                        "_candidate_b_account_anchor_skeleton_cache",
                    )
                return _StagePayload(
                    business={"credit_accounts": accounts},
                    datasets={
                        "personal_detail_account_events": account_events,
                    },
                    context_cache_entries={
                        "account_collections": (
                            accounts,
                            discarded,
                            account_events,
                        )
                    },
                    context_attributes=account_context_attributes,
                )
            if stage_name == "monthly_repayments":
                accounts = payloads["account_inventory"].business.get(
                    "credit_accounts"
                ) or []
                repayments = self.context.corrected_repayment_records()
                evidence_loader = getattr(
                    self.context,
                    "corrected_evidence_pages",
                    None,
                )
                repayment_anchors = candidate_b_repayment_anchor_ledger(
                    evidence_loader() if callable(evidence_loader) else [],
                    accounts,
                )
                self.context._candidate_b_repayment_anchor_ledger = (
                    repayment_anchors
                )
                repayments = link_candidate_b_repayments(
                    repayments,
                    accounts,
                    self.context.corrected_repayment_micro_grids(),
                    reading_order_by_logical=dict(
                        self.context.reading_order_by_logical
                    ),
                    issue_context=self.context,
                    repayment_anchors=repayment_anchors,
                )
                return _StagePayload(
                    business={"repayment_records": repayments},
                    context_attributes={
                        "_candidate_b_repayment_anchor_ledger": (
                            repayment_anchors
                        )
                    },
                )
            if stage_name == "overdue":
                accounts = payloads["account_inventory"].business.get(
                    "credit_accounts"
                ) or ()
                repayments = payloads["monthly_repayments"].business.get(
                    "repayment_records"
                ) or ()
                return _StagePayload(
                    business={
                        "overdue_records": derive_candidate_b_overdue_records(
                            accounts,
                            repayments,
                        )
                    }
                )
            if stage_name == "credit_agreements":
                # Reconciliation remains part of schema extraction.  Staging
                # changes when it runs, never how candidates are extracted.
                return _StagePayload(
                    business={
                        "credit_lines": reconcile_candidate_b_credit_lines(
                            self.context,
                            _extract_credit_lines(self.context),
                        )
                    }
                )
            if stage_name == "liabilities":
                return _StagePayload(
                    business={
                        "repayment_liability_records": _extract_liabilities(
                            self.context
                        )
                    }
                )
            if stage_name == "inquiries":
                return _StagePayload(
                    business={
                        "inquiry_records": _extract_inquiries(self.context)
                    }
                )
            if stage_name == "public":
                return _StagePayload(
                    business={
                        "public_records": _extract_public_records(self.context)
                    }
                )
            if stage_name == "notes":
                annotations, statements = _extract_personal_notes(self.context)
                return _StagePayload(
                    datasets={
                        "annotations": annotations,
                        "statements": statements,
                    }
                )
            if stage_name == "summary":
                summary_records, summary_cells = _extract_summary_datasets(
                    self.context
                )
                return _StagePayload(
                    datasets={
                        "personal_detail_summary_records": summary_records,
                        "personal_detail_summary_cells": summary_cells,
                    }
                )
            if stage_name == "header":
                return _StagePayload(
                    datasets=_extract_header_datasets(
                        self.context,
                        self.full_text,
                    )
                )
            if stage_name == "recovery":
                return _StagePayload(
                    datasets={
                        "recovery_records": _extract_recovery_records(
                            self.context
                        )
                    }
                )
            if stage_name == "postpaid_records":
                return _StagePayload(
                    datasets={
                        "postpaid_records": _extract_postpaid_records(
                            self.context
                        )
                    }
                )
            if stage_name == "postpaid_history":
                return _StagePayload(
                    datasets={
                        "postpaid_payment_history": (
                            _extract_postpaid_payment_history(self.context)
                        )
                    }
                )
            if stage_name == "residence":
                return _StagePayload(
                    datasets={
                        "residence_records": _extract_residence_records(
                            self.context
                        )
                    }
                )
            if stage_name == "employment":
                return _StagePayload(
                    datasets={
                        "employment_records": _extract_employment_records(
                            self.context
                        )
                    }
                )
            if stage_name == "source_rows":
                return _StagePayload(
                    datasets={
                        "personal_detail_source_rows": _extract_source_rows(
                            self.context
                        )
                    }
                )
            if stage_name == "profile_details":
                return _StagePayload(
                    datasets=_extract_profile_detail_records(self.context)
                )
            if stage_name == "profile":
                return _StagePayload(
                    profile=extract_candidate_b_profile(self.context)
                )
            raise RuntimeError(
                f"Candidate B stage has no extractor callback: {stage_name}"
            )

        def record_count(payload: _StagePayload, output_name: str) -> int:
            if output_name == "status_glyph_observations":
                return len(payload.status_glyph_observations)
            if output_name == "subject_profile":
                return int(payload.profile is not None)
            value = payload.business.get(output_name)
            if value is None:
                value = payload.datasets.get(output_name)
            if isinstance(value, Mapping):
                return 1
            if isinstance(value, (list, tuple, set, frozenset)):
                return len(value)
            return int(value is not None)

        def extract_source_pass(
            plan: Any,
            *,
            generation: int,
            reusable_stage_payloads: Mapping[str, _StagePayload] | None = None,
            reusable_stage_snapshots: Mapping[str, Any] | None = None,
            discovery_issues: tuple[dict[str, Any], ...] = (),
            discovery_issue_owners: Mapping[str, Any] | None = None,
            discovery_remap_owners: Mapping[tuple[str, str], Any]
            | None = None,
        ) -> _SourcePassResult:
            reusable_stage_payloads = reusable_stage_payloads or {}
            reusable_stage_snapshots = reusable_stage_snapshots or {}
            discovery_issue_owners = discovery_issue_owners or {}
            discovery_remap_owners = discovery_remap_owners or {}
            execute = set(plan.ordered_stage_names)
            reuse = set(plan.reused_stage_names)
            payloads: dict[str, _StagePayload] = {}

            # Walk the full stable registry order so reused diagnostic effects
            # are restored before any dependent dirty stage executes.
            for stage_name in CANDIDATE_B_STAGE_REGISTRY.ordered():
                if stage_name in reuse:
                    if stage_name not in reusable_stage_payloads:
                        raise RuntimeError(
                            "Candidate B plan requested an unavailable stage "
                            f"snapshot: {stage_name}"
                        )
                    payload = deepcopy(reusable_stage_payloads[stage_name])
                    _apply_reused_stage_diagnostics(
                        self.context,
                        stage_name,
                        payload,
                        discovery_issues=discovery_issues,
                        discovery_issue_owners=discovery_issue_owners,
                        discovery_remap_owners=discovery_remap_owners,
                    )
                    payloads[stage_name] = payload
                    continue
                if stage_name not in execute:
                    continue
                before_issues = _issue_rows(self.context)
                before_edges = _remap_edges(self.context)
                with _candidate_b_extraction_stage(
                    self.context,
                    stage_name,
                ):
                    payload = materialize_stage(stage_name, payloads)
                    if (
                        generation > 1
                        and stage_name == "account_inventory"
                        and isinstance(
                            reusable_stage_payloads.get("account_inventory"),
                            _StagePayload,
                        )
                    ):
                        _reconcile_repaired_account_stage_payload(
                            self.context,
                            reusable_stage_payloads["account_inventory"],
                            payload,
                        )
                    if (
                        generation > 1
                        and isinstance(
                            reusable_stage_payloads.get(stage_name),
                            _StagePayload,
                        )
                    ):
                        _preserve_discovery_stage_outputs_on_empty_repair(
                            CANDIDATE_B_STAGE_REGISTRY.stage(stage_name),
                            reusable_stage_payloads[stage_name],
                            payload,
                        )
                (
                    payload.removed_issue_ids,
                    payload.upserted_issues,
                    payload.removed_remap_edges,
                    payload.added_remap_edges,
                ) = _stage_diagnostic_delta(
                    before_issues,
                    before_edges,
                    self.context,
                )
                _mark_direct_stage_diagnostics(
                    self.context,
                    stage_name,
                    payload,
                )
                payloads[stage_name] = payload

            merged_business: dict[str, Any] = {}
            merged_datasets: dict[str, list[dict[str, Any]]] = {}
            profile: dict[str, Any] = {}
            for stage_name in CANDIDATE_B_STAGE_REGISTRY.ordered():
                payload = payloads.get(stage_name)
                if payload is None:
                    continue
                merged_business.update(deepcopy(payload.business))
                merged_datasets.update(deepcopy(payload.datasets))
                if payload.profile is not None:
                    profile = deepcopy(payload.profile)

            business: dict[str, Any] = {
                name: list(merged_business.get(name) or ())
                for name in _CORE_BUSINESS_DATASETS
            }
            business["credit_summary"] = {
                "source": "candidate_b_canonical_templates",
                "reported_account_count": len(business["credit_accounts"]),
                "projected_account_count": len(business["credit_accounts"]),
                "repayment_liability_count": len(
                    business["repayment_liability_records"]
                ),
                "inquiry_count": len(business["inquiry_records"]),
                "account_population_comparable": False,
            }

            datasets: dict[str, list[dict[str, Any]]] = {}
            header_payload = payloads.get("header")
            if header_payload is not None:
                datasets.update(deepcopy(header_payload.datasets))
            datasets.update(
                {
                    name: list(business.get(name) or ())
                    for name in _CORE_BUSINESS_DATASETS
                }
            )
            for name in (
                "recovery_records",
                "postpaid_records",
                "postpaid_payment_history",
                "personal_detail_account_events",
                "personal_detail_summary_records",
                "personal_detail_summary_cells",
                "residence_records",
                "employment_records",
                "annotations",
                "statements",
                "personal_detail_source_rows",
            ):
                datasets[name] = list(merged_datasets.get(name) or ())
            profile_details_payload = payloads.get("profile_details")
            if profile_details_payload is not None:
                datasets.update(deepcopy(profile_details_payload.datasets))

            issues_before_source_gaps = {
                _issue_id(issue)
                for issue in _issue_rows(self.context)
                if _issue_id(issue)
            }
            _record_pre_repair_source_gaps(self.context, datasets)
            issues_after_source_gaps = {
                _issue_id(issue)
                for issue in _issue_rows(self.context)
                if _issue_id(issue)
            }
            derived_issue_ids = frozenset(
                issues_after_source_gaps.difference(issues_before_source_gaps)
            )

            monthly_payload = payloads.get("monthly_repayments")
            if monthly_payload is not None and "monthly_repayments" in execute:
                before_status_issues = _issue_rows(self.context)
                before_status_edges = _remap_edges(self.context)
                with _candidate_b_extraction_stage(
                    self.context,
                    "monthly_repayments",
                ):
                    monthly_payload.status_glyph_observations = (
                        status_glyph_observations()
                    )
                _merge_stage_diagnostic_delta(
                    monthly_payload,
                    _stage_diagnostic_delta(
                        before_status_issues,
                        before_status_edges,
                        self.context,
                    ),
                )
                _mark_direct_stage_diagnostics(
                    self.context,
                    "monthly_repayments",
                    monthly_payload,
                )
            status_observations = (
                list(monthly_payload.status_glyph_observations)
                if monthly_payload is not None
                else []
            )

            snapshots: dict[str, Any] = {}
            for stage_name in CANDIDATE_B_STAGE_REGISTRY.ordered():
                payload = payloads.get(stage_name)
                if payload is None:
                    continue
                if stage_name in reuse and stage_name in reusable_stage_snapshots:
                    snapshots[stage_name] = deepcopy(
                        reusable_stage_snapshots[stage_name]
                    )
                    continue
                stage = CANDIDATE_B_STAGE_REGISTRY.stage(stage_name)
                snapshots[stage_name] = StageSnapshot(
                    stage_name=stage_name,
                    generation=generation,
                    dependency_generations=tuple(
                        (
                            dependency,
                            snapshots[dependency].generation,
                        )
                        for dependency in sorted(stage.dependencies)
                    ),
                    output_names=stage.output_names,
                    record_counts=tuple(
                        (
                            output_name,
                            record_count(payload, output_name),
                        )
                        for output_name in stage.output_names
                    ),
                )

            return _SourcePassResult(
                business=business,
                datasets=datasets,
                profile=profile,
                status_glyph_observations=status_observations,
                stage_payloads=payloads,
                stage_snapshots=snapshots,
                derived_issue_ids=derived_issue_ids,
            )

        baseline_issues = _issue_rows(self.context)
        baseline_issue_ids = {
            _issue_id(issue) for issue in baseline_issues if _issue_id(issue)
        }
        baseline_remap_edges = _remap_edges(self.context)
        self.context._candidate_b_issue_stage_owners = {}
        self.context._candidate_b_remap_stage_owners = {}

        # Registration and fragment joining consume static ParseResult evidence
        # only.  The detached audit is the section census for the lazy planner.
        with _candidate_b_extraction_stage(
            self.context,
            "canonical_census",
        ):
            canonical_pages = self.context.pages
            del canonical_pages
            discovery_canonical_audit = deepcopy(
                self.context.canonical_layout_audit()
            )
        discovery_census, discovery_plan = (
            plan_candidate_b_initial_extraction(
                discovery_canonical_audit,
                strategy=self.extraction_strategy,
            )
        )
        first_result = extract_source_pass(
            discovery_plan,
            generation=1,
        )
        first_business = first_result.business
        first_datasets = first_result.datasets
        first_profile = first_result.profile
        first_status_glyph_observations = (
            first_result.status_glyph_observations
        )
        discovery_issues = _issue_rows(self.context)
        discovery_issue_owners = deepcopy(
            getattr(self.context, "_candidate_b_issue_stage_owners", {})
        )
        discovery_remap_owners = deepcopy(
            getattr(self.context, "_candidate_b_remap_stage_owners", {})
        )
        discovery_issue_ids = {
            _issue_id(issue)
            for issue in discovery_issues
            if _issue_id(issue)
        }
        unowned_discovery_issue_ids = tuple(
            sorted(
                discovery_issue_ids.difference(baseline_issue_ids)
                .difference(discovery_issue_owners)
                .difference(first_result.derived_issue_ids)
            )
        )
        unowned_discovery_remap_edges = tuple(
            sorted(
                _remap_edges(self.context)
                .difference(baseline_remap_edges)
                .difference(discovery_remap_owners)
            )
        )
        repair_payload = {
            "credit_summary": dict(first_business.get("credit_summary") or {}),
            **first_datasets,
        }
        repair_applied = self.context.prepare_candidate_b_business_repair(repair_payload)
        repair_scope = None
        repair_plan = None
        repaired_census_audit = None
        if repair_applied:
            # ``prepare`` resets extraction caches and issues.  Remap edges are
            # reset here as well so only clean-stage edges are rehydrated.
            self.context._personal_detail_extraction_issues = list(
                deepcopy(baseline_issues)
            )
            self.context._personal_detail_issue_target_remaps = (
                _remap_registry_from_edges(baseline_remap_edges)
            )
            self.context._candidate_b_issue_stage_owners = {}
            self.context._candidate_b_remap_stage_owners = {}
            with _candidate_b_extraction_stage(
                self.context,
                "canonical_census",
            ):
                repaired_pages = self.context.pages
                del repaired_pages
                repaired_canonical_audit = deepcopy(
                    self.context.canonical_layout_audit()
                )
            repaired_census_audit = section_census_from_canonical_audit(
                repaired_canonical_audit
            ).audit()
            repair_scope = candidate_b_repair_scope(
                getattr(self.context, "_business_repair_plan", None),
                discovery_canonical_audit,
                repaired_canonical_audit,
            )
            repair_request = repair_scope.extraction_request(
                available_stage_names=first_result.stage_payloads,
            )
            if (
                unowned_discovery_issue_ids
                or unowned_discovery_remap_edges
            ):
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
                    ExtractionRequest,
                )

                repair_request = ExtractionRequest.repair(
                    repair_scope.census,
                    available_stage_names=first_result.stage_payloads,
                    dirty_stage_names=repair_scope.dirty_stage_names,
                    dirty_section_names=(
                        section_name
                        for section_name in repair_scope.dirty_sections
                        if section_name
                        in CANDIDATE_B_STAGE_REGISTRY.sections
                    ),
                    dependency_closure_known=False,
                    force_eager_reason=(
                        "issue_stage_ownership_unknown"
                        if unowned_discovery_issue_ids
                        else "remap_stage_ownership_unknown"
                    ),
                )
            repair_plan = self.extraction_strategy.plan(
                CANDIDATE_B_STAGE_REGISTRY,
                repair_request,
            )
            repaired_result = extract_source_pass(
                repair_plan,
                generation=2,
                reusable_stage_payloads=first_result.stage_payloads,
                reusable_stage_snapshots=first_result.stage_snapshots,
                discovery_issues=discovery_issues,
                discovery_issue_owners=discovery_issue_owners,
                discovery_remap_owners=discovery_remap_owners,
            )
            source_business = repaired_result.business
            source_datasets = repaired_result.datasets
            source_profile = repaired_result.profile
            source_status_glyph_observations = (
                repaired_result.status_glyph_observations
            )
            overdue_input_basis = _overdue_view_input_basis(source_datasets)
            source_datasets = _reconcile_candidate_b_header_lifecycle(
                self.context,
                first_datasets,
                source_datasets,
            )
            _withhold_independent_plane_conflicts(
                self.context,
                first_datasets,
                source_datasets,
            )
        else:
            repaired_result = None
            source_business, source_datasets, source_profile = (
                first_business,
                first_datasets,
                first_profile,
            )
            source_status_glyph_observations = first_status_glyph_observations
            overdue_input_basis = _overdue_view_input_basis(source_datasets)

        # The final correction plane covers every source dataset, including
        # monthly grids and profile/detail tables. It consumes only evidence
        # selected by the document-wide repair coordinator.
        from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
            apply_planned_monthly_field_repairs,
        )

        corrected_payload = self.context.correct_candidate_b_datasets(
            apply_planned_monthly_field_repairs(self.context, {
                "credit_summary": dict(source_business.get("credit_summary") or {}),
                **source_datasets,
            })
        )
        _enforce_employment_record_contracts(
            self.context,
            [
                row
                for row in corrected_payload.get("employment_records") or ()
                if isinstance(row, dict)
            ],
        )
        corrected_payload["credit_lines"] = reconcile_candidate_b_credit_lines(
            self.context,
            list(corrected_payload.get("credit_lines") or ()),
        )
        consistency_input = dict(corrected_payload)
        consistency_input["personal_profile"] = [source_profile]
        consistency_audit = apply_document_consistency_ledger(
            self.context,
            consistency_input,
        )
        _reconcile_final_account_field_issues(
            self.context,
            corrected_payload.get("credit_accounts") or (),
        )
        status_glyph_bank_audit = apply_document_local_status_glyph_bank(
            [
                row
                for row in corrected_payload.get("repayment_records") or ()
                if isinstance(row, dict)
            ],
            source_status_glyph_observations,
            accounts=(
                row
                for row in corrected_payload.get("credit_accounts") or ()
                if isinstance(row, Mapping)
            ),
            issues=(
                issue
                for issue in getattr(
                    self.context,
                    "_personal_detail_extraction_issues",
                    (),
                )
                if isinstance(issue, Mapping)
            ),
            native_plane_records=(
                row
                for row in first_datasets.get("repayment_records") or ()
                if isinstance(row, Mapping)
            ),
            corrected_plane_records=(
                row
                for row in source_datasets.get("repayment_records") or ()
                if isinstance(row, Mapping)
            ),
        )
        # Run the sealed-source check after every consistency and glyph stage;
        # no later operation may change a monthly status.  This keeps the value
        # audited here identical to the one handed to the final projection.
        native_status_conflict_audit = apply_candidate_b_native_status_conflict_guard(
            self.context,
            [
                row
                for row in corrected_payload.get("repayment_records") or ()
                if isinstance(row, dict)
            ],
            enabled=True,
        )
        # This view depends on the final status/amount, not on the discovery
        # pass. Field overlays and final guards may repair or withhold either
        # input after the lazy derived stage has already run.
        _refresh_final_overdue_view(corrected_payload, overdue_input_basis)
        all_datasets: dict[str, list[dict[str, Any]]] = {
            name: list(corrected_payload.get(name) or ())
            for name in dict.fromkeys((*source_datasets, "overdue_records"))
        }
        business: dict[str, Any] = {
            name: list(all_datasets.get(name) or ())
            for name in _CORE_BUSINESS_DATASETS
        }
        business["credit_summary"] = dict(corrected_payload.get("credit_summary") or {})
        all_datasets = {name: rows for name, rows in all_datasets.items() if rows}

        register_final_liability_issue_records(
            self.context,
            [
                row
                for row in all_datasets.get("repayment_liability_records") or ()
                if isinstance(row, Mapping)
            ],
        )
        canonical_layout_owner_census = deepcopy(
            self.context.canonical_layout_audit()
        )
        source_completeness_ledger = _source_completeness_ledger(self.context)
        source_completeness_ledger = (
            _reconcile_repaired_inquiry_source_population(
                source_completeness_ledger,
                tuple(
                    row
                    for row in all_datasets.get("inquiry_records") or ()
                    if isinstance(row, dict)
                ),
            )
        )
        reconcile_candidate_b_account_sequence_issues(
            self.context,
            source_completeness_ledger,
            all_datasets.get("credit_accounts") or (),
        )
        issues = collect_extraction_issues(self.context)
        if issues:
            all_datasets["personal_detail_extraction_issues"] = issues
        final_counts = {
            name: sum(isinstance(row, dict) for row in rows)
            for name, rows in all_datasets.items()
            if isinstance(rows, list)
        }
        facts: dict[str, Any] = {
            "subject_profile": source_profile,
            "credit_summary": dict(business.get("credit_summary") or {}),
            "canonical_dataset_schema": "personal_credit_report_detailed.v2",
            # Publish the immutable topology/registration plane before source
            # projection. Fail-closed field conservation must resolve exact
            # metadata cells against their canonical owner at this stage; the
            # public extraction audit is assembled only after projection.
            "_personal_detail_canonical_layout_owner_census": canonical_layout_owner_census,
            "personal_detail_source_completeness_ledger": source_completeness_ledger,
            "personal_detail_document_consistency_ledger": consistency_audit,
            "personal_detail_dataset_states": dataset_states_from_issues(issues),
            **{f"personal_detail_expected_{name}_count": count for name, count in final_counts.items()},
        }
        content = prepare_personal_detail_source_collections(
            {"facts": facts, "datasets": all_datasets},
            business,
            final_dataset_counts=final_counts,
        )
        projected_facts = content.get("facts")
        if isinstance(projected_facts, dict):
            projected_facts.pop(
                "_personal_detail_canonical_layout_owner_census", None
            )
        supplemental = {
            name: rows
            for name, rows in (content.get("datasets") or {}).items()
            if name not in _CORE_BUSINESS_DATASETS
        }
        section_content = {
            "facts": dict(content.get("facts") or {}),
            "datasets": supplemental,
        }
        audit = {
            "architecture": "candidate_b_clean",
            "source_of_truth": "static_canonical_pages_then_schema_triggered_repair",
            "candidate_population_count": 1,
            "schema_extraction_pass_count": 2
            if self.context.ocr_correction_audit().get("business_repair", {}).get("second_schema_pass_required")
            else 1,
            "parse_result_mutated": False,
            "canonical_layout": canonical_layout_owner_census,
            "page_topology": self.context.page_topology_audit(),
            "ocr_correction": self.context.ocr_correction_audit(),
            "source_extraction_strategy": {
                "architecture": "candidate_b_stage_materialization_v1",
                "canonical_census_mode": (
                    "always_recomputed_before_stage_planning"
                ),
                "shared_release_gates": {
                    "mode": "always_eager_once_after_source_materialization",
                    "stages": [
                        "final_dataset_correction",
                        "employment_contract_enforcement",
                        "credit_line_reconciliation",
                        "document_consistency",
                        "final_account_field_issue_reconciliation",
                        "document_local_status_glyph_bank",
                        "native_source_cell_status_guard",
                        "final_liability_issue_registration",
                        "source_completeness",
                        "account_sequence_issue_reconciliation",
                        "extraction_issue_collection",
                        "source_projection",
                    ],
                },
                "discovery": {
                    "census": discovery_census.audit(),
                    "plan": discovery_plan.audit().to_dict(),
                    "stage_snapshots": [
                        first_result.stage_snapshots[stage_name].to_audit_dict()
                        for stage_name in CANDIDATE_B_STAGE_REGISTRY.ordered()
                        if stage_name in first_result.stage_snapshots
                    ],
                },
                "repair": (
                    {
                        "scope": repair_scope.audit(),
                        "repaired_census": repaired_census_audit,
                        "plan": repair_plan.audit().to_dict(),
                        "stage_snapshots": [
                            repaired_result.stage_snapshots[
                                stage_name
                            ].to_audit_dict()
                            for stage_name in CANDIDATE_B_STAGE_REGISTRY.ordered()
                            if stage_name in repaired_result.stage_snapshots
                        ],
                    }
                    if repair_scope is not None
                    and repair_plan is not None
                    and repaired_result is not None
                    else None
                ),
                "diagnostic_ownership": {
                    "unowned_discovery_issue_ids": list(
                        unowned_discovery_issue_ids
                    ),
                    "unowned_discovery_remap_edges": [
                        list(edge) for edge in unowned_discovery_remap_edges
                    ],
                },
            },
            "document_consistency": consistency_audit,
            "native_source_cell_status_guard": native_status_conflict_audit,
            "document_local_status_glyph_bank": status_glyph_bank_audit,
            "source_dataset_counts": final_counts,
        }
        business["credit_extraction_audit"] = audit
        return CandidateBExtraction(
            business=business,
            section_content=section_content,
            audit=audit,
        )


__all__ = ["CandidateBExtraction", "CandidateBPipeline"]
