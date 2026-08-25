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
from copy import deepcopy
from dataclasses import dataclass
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

    records_by_id = {
        str(_account_record_values(record).get("account_id") or record.get("account_id") or ""): record
        for record in records or ()
        if isinstance(record, dict)
    }
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


class CandidateBPipeline:
    """Extract registered canonical pages into the PBOC source schema once."""

    def __init__(self, context: Any, full_text: str) -> None:
        self.context = context
        self.full_text = str(full_text or "")

    def run(self) -> CandidateBExtraction:
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

        def extract_source_pass() -> tuple[
            dict[str, Any],
            dict[str, list[dict[str, Any]]],
            dict[str, Any],
        ]:
            # Registration and fragment joining use static ParseResult evidence
            # only. No OCR can be started before business candidates exist.
            canonical_pages = self.context.pages
            del canonical_pages

            accounts, _discarded_parallel_monthly_rows, account_events = self.context.account_collections()
            repayments = self.context.corrected_repayment_records()
            evidence_loader = getattr(self.context, "corrected_evidence_pages", None)
            repayment_anchors = candidate_b_repayment_anchor_ledger(
                evidence_loader() if callable(evidence_loader) else [],
                accounts,
            )
            self.context._candidate_b_repayment_anchor_ledger = repayment_anchors
            repayments = link_candidate_b_repayments(
                repayments,
                accounts,
                self.context.corrected_repayment_micro_grids(),
                reading_order_by_logical=dict(self.context.reading_order_by_logical),
                issue_context=self.context,
                repayment_anchors=repayment_anchors,
            )
            business: dict[str, Any] = {
                "credit_accounts": accounts,
                # Reconciliation is part of schema extraction, not a release-
                # only cleanup.  Running it in the discovery pass exposes
                # field conflicts/missing slots early enough to select the
                # one permitted complete-page OCR repair.
                "credit_lines": reconcile_candidate_b_credit_lines(
                    self.context,
                    _extract_credit_lines(self.context),
                ),
                "repayment_liability_records": _extract_liabilities(self.context),
                "repayment_records": repayments,
                "overdue_records": derive_candidate_b_overdue_records(accounts, repayments),
                "inquiry_records": _extract_inquiries(self.context),
                "public_records": _extract_public_records(self.context),
            }
            business["credit_summary"] = {
                "source": "candidate_b_canonical_templates",
                "reported_account_count": len(business["credit_accounts"]),
                "projected_account_count": len(business["credit_accounts"]),
                "repayment_liability_count": len(business["repayment_liability_records"]),
                "inquiry_count": len(business["inquiry_records"]),
                "account_population_comparable": False,
            }
            annotations, statements = _extract_personal_notes(self.context)
            summary_records, summary_cells = _extract_summary_datasets(self.context)
            datasets: dict[str, list[dict[str, Any]]] = {
                **_extract_header_datasets(self.context, self.full_text),
                **{name: list(business.get(name) or ()) for name in _CORE_BUSINESS_DATASETS},
                "recovery_records": _extract_recovery_records(self.context),
                "postpaid_records": _extract_postpaid_records(self.context),
                "postpaid_payment_history": _extract_postpaid_payment_history(self.context),
                "personal_detail_account_events": account_events,
                "personal_detail_summary_records": summary_records,
                "personal_detail_summary_cells": summary_cells,
                "residence_records": _extract_residence_records(self.context),
                "employment_records": _extract_employment_records(self.context),
                "annotations": annotations,
                "statements": statements,
                "personal_detail_source_rows": _extract_source_rows(self.context),
                **_extract_profile_detail_records(self.context),
            }
            profile = extract_candidate_b_profile(self.context)
            _record_pre_repair_source_gaps(self.context, datasets)
            return business, datasets, profile

        def status_glyph_observations() -> list[dict[str, Any]]:
            loader = getattr(
                self.context,
                "candidate_b_status_glyph_observations",
                None,
            )
            return list(loader() or ()) if callable(loader) else []

        first_business, first_datasets, first_profile = extract_source_pass()
        first_status_glyph_observations = status_glyph_observations()
        repair_payload = {
            "credit_summary": dict(first_business.get("credit_summary") or {}),
            **first_datasets,
        }
        repair_applied = self.context.prepare_candidate_b_business_repair(repair_payload)
        if repair_applied:
            source_business, source_datasets, source_profile = extract_source_pass()
            source_status_glyph_observations = status_glyph_observations()
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
            source_business, source_datasets, source_profile = (
                first_business,
                first_datasets,
                first_profile,
            )
            source_status_glyph_observations = first_status_glyph_observations

        # The final correction plane covers every source dataset, including
        # monthly grids and profile/detail tables. It consumes only evidence
        # selected by the document-wide repair coordinator.
        corrected_payload = self.context.correct_candidate_b_datasets(
            {
                "credit_summary": dict(source_business.get("credit_summary") or {}),
                **source_datasets,
            }
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
        all_datasets: dict[str, list[dict[str, Any]]] = {
            name: list(corrected_payload.get(name) or ())
            for name in source_datasets
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
