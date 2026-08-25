# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-detected layout profiles for scanned PBOC detailed reports.

The PBOC report family has stable semantic roles, but its printed table
revisions are not interchangeable.  This module keeps those two facts
separate: an exact header can identify the PBOC inquiry schema and its physical
role columns, while revision-specific repair capabilities are enabled only by
a registered role layout.  Business values, fixture identities, page counts,
and OCR acquisition are intentionally outside this detector.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


class InquiryNumberingModel(str, Enum):
    """How the two printed inquiry subsections assign their ordinals."""

    INDEPENDENT_RESTARTS = "independent_restarts"
    COMBINED_CONTINUITY = "combined_continuity"
    UNKNOWN = "unknown"


class LayoutCapability(str, Enum):
    """Revision-specific inquiry repairs that require an explicit profile."""

    COLLAPSED_HEADER = "inquiry_collapsed_header"
    SPLIT_PERSONAL_HEADER = "inquiry_split_personal_header"
    MIXED_PAGE_HEADER = "inquiry_mixed_page_header"
    HEADERLESS_CONTINUATION = "inquiry_headerless_continuation"
    HEADERLESS_BOOTSTRAP = "inquiry_headerless_bootstrap"
    TWO_ROW_CELL_SPLIT = "inquiry_two_row_cell_split"
    MERGED_PERSONAL_ROW = "inquiry_merged_personal_row"
    TOKEN_BRIDGE = "inquiry_token_bridge"


@dataclass(frozen=True)
class InquiryLocalRepairProof:
    """Capability-specific authorization for one bounded inquiry repair.

    This value is deliberately produced by the native extractor *after* its
    local semantic, geometry, and source-owner checks have all succeeded.  A
    document-level header profile can veto the proof, but can never manufacture
    one.  Keeping this as a sealed value object prevents a role-map-shaped
    dictionary from becoming a cosmetic repair gate.
    """

    capability: LayoutCapability
    inquiry_role_columns: tuple[tuple[str, int], ...]
    evidence_ids: tuple[str, ...]
    geometry_bbox: tuple[float, float, float, float]
    local_trait: str
    section_owner_role: str | None = None

    @classmethod
    def create(
        cls,
        capability: LayoutCapability,
        *,
        inquiry_role_columns: Mapping[str, int],
        evidence_ids: Iterable[str],
        geometry_bbox: Iterable[float],
        local_trait: str,
        section_owner_role: str | None = None,
    ) -> InquiryLocalRepairProof | None:
        """Create one proof only from finite, unique, capability-local facts."""

        if not isinstance(capability, LayoutCapability):
            return None
        columns = {
            str(role): int(column)
            for role, column in inquiry_role_columns.items()
            if isinstance(column, int) and not isinstance(column, bool) and column >= 0
        }
        if set(columns) != set(_INQUIRY_ROLES) or len(set(columns.values())) != len(_INQUIRY_ROLES):
            return None
        owners = tuple(str(value) for value in evidence_ids if str(value or ""))
        bbox = _finite_bbox(tuple(geometry_bbox))
        trait = str(local_trait or "").strip()
        owner_role = str(section_owner_role or "").strip() or None
        expected_trait = _LOCAL_TRAIT_BY_CAPABILITY.get(capability)
        if (
            not owners
            or len(owners) != len(set(owners))
            or bbox is None
            or not trait
            or trait != expected_trait
        ):
            return None
        return cls(
            capability=capability,
            inquiry_role_columns=tuple(
                (role, columns[role]) for role in _INQUIRY_ROLES
            ),
            evidence_ids=owners,
            geometry_bbox=bbox,
            local_trait=trait,
            section_owner_role=owner_role,
        )

    def columns(self) -> dict[str, int]:
        return dict(self.inquiry_role_columns)


_LOCAL_TRAIT_BY_CAPABILITY: Mapping[LayoutCapability, str] = {
    LayoutCapability.COLLAPSED_HEADER: "exact_collapsed_header_lattice",
    LayoutCapability.SPLIT_PERSONAL_HEADER: "exact_split_personal_header_lattice",
    LayoutCapability.MIXED_PAGE_HEADER: "exact_mixed_page_heading_header_lattice",
    LayoutCapability.HEADERLESS_CONTINUATION: "exact_headerless_continuation_row",
    LayoutCapability.HEADERLESS_BOOTSTRAP: "sealed_headerless_population",
    LayoutCapability.TWO_ROW_CELL_SPLIT: "exact_two_row_token_lattice",
    LayoutCapability.MERGED_PERSONAL_ROW: "exact_merged_personal_token_lattice",
    LayoutCapability.TOKEN_BRIDGE: "exact_terminal_token_bridge",
}


_INQUIRY_ROLES = ("sequence", "inquiry_date", "institution", "reason")
_INQUIRY_LABELS: Mapping[str, tuple[str, ...]] = {
    "sequence": ("编号", "序号"),
    "inquiry_date": ("查询日期",),
    "institution": ("查询机构",),
    "reason": ("查询原因",),
}
_DATE_RE = re.compile(r"(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}")
_PERSONAL_REASON_RE = re.compile(r"本人查询(?:[（(][^）)]{1,32}[）)])?")


@dataclass(frozen=True)
class PBOCInquirySchemaSpec:
    """One registered inquiry-table schema, expressed only in semantic roles."""

    schema_id: str
    version: int
    inquiry_role_order: tuple[str, ...]
    capabilities: frozenset[LayoutCapability]


@dataclass(frozen=True)
class PBOCLayoutProfile:
    """Immutable result of document-local PBOC layout detection."""

    pboc_family: str
    layout_revision: str
    inquiry_schema_profile: str
    profile_version: int
    inquiry_role_columns: tuple[tuple[str, int], ...]
    inquiry_numbering_model: InquiryNumberingModel
    capabilities: frozenset[LayoutCapability]
    detection_reasons: tuple[str, ...]
    exact_header_count: int = 0
    exact_numbering_group_count: int = 0

    @property
    def profile_id(self) -> str:
        return f"{self.pboc_family}:{self.inquiry_schema_profile}:v{self.profile_version}"

    def inquiry_columns(self) -> dict[str, int]:
        return dict(self.inquiry_role_columns)

    def allows(self, capability: LayoutCapability) -> bool:
        return capability in self.capabilities

    def allows_local_proof(
        self,
        capability: LayoutCapability,
        *,
        proof: InquiryLocalRepairProof | None = None,
    ) -> bool:
        """Authorize one capability only after its complete local proof.

        Exact global headers contribute only negative or consistency evidence:
        they can veto a conflicting local role map, but an ordinary four-role
        header never authorizes collapsed, split, headerless, or token repair.
        """

        map_conflict = "conflicting_exact_inquiry_role_maps" in self.detection_reasons
        owner_conflict = "exact_header_owner_conflict" in self.detection_reasons
        if (
            not isinstance(capability, LayoutCapability)
            or not isinstance(proof, InquiryLocalRepairProof)
            or proof.capability is not capability
            or proof.local_trait != _LOCAL_TRAIT_BY_CAPABILITY.get(capability)
            or not proof.evidence_ids
            or len(proof.evidence_ids) != len(set(proof.evidence_ids))
            or _finite_bbox(proof.geometry_bbox) is None
            or owner_conflict
        ):
            return False
        candidate = proof.columns()
        profile_columns = self.inquiry_columns()
        if capability is LayoutCapability.MIXED_PAGE_HEADER:
            # Column order is table-local.  Multiple exact PBOC inquiry tables
            # may legitimately print the same four roles in different orders.
            # The mixed-page caller has already proved this table's canonical
            # section owner, exact heading boundary, and exact header lattice;
            # a different map elsewhere may not suppress that local evidence.
            return bool(
                self.pboc_family == "pboc_personal_detailed"
                and self.exact_header_count >= 1
                and proof.section_owner_role == "annotations_and_inquiries"
            )
        if (
            profile_columns
            and candidate != profile_columns
            and self.inquiry_schema_profile
            not in {"unknown", "unregistered_semantic_role_map"}
        ):
            return False
        if map_conflict:
            return False
        if self.inquiry_schema_profile == "unregistered_semantic_role_map":
            # A mixed-page header is not a column-shape repair: the table-local
            # PBOC section owner and exact header lattice have already proved
            # every semantic role.  Permit that one capability when its local
            # map exactly matches the independently detected document map.
            # Other revision-shaped repairs remain disabled for reordered or
            # otherwise unregistered physical layouts.
            # A reordered exact map normally vetoes revision-shaped repair. A
            # distinct, complete local topology may still prove the canonical
            # role map when the document has no global conflict; this is local
            # PBOC evidence, not inheritance from the reordered header.
            return candidate != profile_columns and candidate == {
                "sequence": 0,
                "inquiry_date": 1,
                "institution": 2,
                "reason": 3,
            }
        return self.pboc_family == "pboc_personal_detailed"

    def audit(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "pboc_family": self.pboc_family,
            "layout_revision": self.layout_revision,
            "inquiry_schema_profile": self.inquiry_schema_profile,
            "profile_version": self.profile_version,
            "inquiry_role_columns": self.inquiry_columns(),
            "inquiry_numbering_model": self.inquiry_numbering_model.value,
            "capabilities": sorted(capability.value for capability in self.capabilities),
            "detection_reasons": list(self.detection_reasons),
            "exact_header_count": self.exact_header_count,
            "exact_numbering_group_count": self.exact_numbering_group_count,
            "fixture_identity_used": False,
            "ocr_used": False,
            "section_graph_authority": False,
            "pagination_authority": False,
            "capabilities_require_local_proof": True,
            "local_repair_proof_vetoed": not self.allows_local_proof(
                LayoutCapability.COLLAPSED_HEADER
            ),
        }


_CANONICAL_FOUR_COLUMN = PBOCInquirySchemaSpec(
    schema_id="pboc_personal_detailed_inquiry_four_column",
    version=1,
    inquiry_role_order=_INQUIRY_ROLES,
    # Exact role order is stable semantic evidence only.  Repair capabilities
    # are activated by the corresponding complete local topology helper.
    capabilities=frozenset(),
)
_REGISTERED_INQUIRY_SCHEMAS = (_CANONICAL_FOUR_COLUMN,)


def _value(owner: Any, key: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
        return owner.get(key, default)
    return getattr(owner, key, default)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _table_rows(table: Any) -> list[list[str]]:
    metadata = _value(table, "metadata", {})
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, Mapping) else None
    if isinstance(raw_rows, list) and raw_rows:
        return [
            [str(cell or "") for cell in row]
            for row in raw_rows
            if isinstance(row, list)
        ]
    headers = [str(_value(cell, "text", cell) or "") for cell in _value(table, "headers", ()) or ()]
    rows = [
        [str(_value(cell, "text", "") or "") for cell in _value(row, "cells", ()) or ()]
        for row in _value(table, "rows", ()) or ()
    ]
    return ([headers] if headers else []) + rows


def _finite_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        bbox = tuple(float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


@dataclass(frozen=True)
class ExactInquiryHeaderOwner:
    """One immutable, table-local PBOC inquiry header owner.

    The owner records semantic roles rather than a physical column order.  It
    is intentionally unavailable when a header contains an unregistered label,
    residual business text, repeated roles, competing header candidates, or
    non-exact/replayed source ownership.
    """

    inquiry_role_columns: tuple[tuple[str, int], ...]
    header_rows: tuple[int, ...]
    header_labels: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    cell_bboxes: tuple[tuple[float, float, float, float], ...]
    body_start: int
    binding: str

    def columns(self) -> dict[str, int]:
        return dict(self.inquiry_role_columns)


def _exact_cell_owner(
    table: Any,
    *,
    row: int,
    column: int,
) -> tuple[tuple[float, float, float, float], tuple[str, ...]] | None:
    """Return one exact source cell with unique immutable evidence owners."""

    metadata = _value(table, "metadata", {})
    geometry = metadata.get("geometry") if isinstance(metadata, Mapping) else None
    if not isinstance(geometry, Mapping):
        return None
    statuses = geometry.get("cell_geometry_status")
    evidence_ids = geometry.get("cell_evidence_ids")
    bboxes = geometry.get("cell_bboxes")
    if not all(isinstance(grid, list) for grid in (statuses, evidence_ids, bboxes)):
        return None
    if any(
        row < 0
        or row >= len(grid)
        or not isinstance(grid[row], list)
        or column < 0
        or column >= len(grid[row])
        for grid in (statuses, evidence_ids, bboxes)
    ):
        return None
    if str(statuses[row][column] or "") != "exact":
        return None
    ids = evidence_ids[row][column]
    bbox = _finite_bbox(bboxes[row][column])
    if not isinstance(ids, list) or bbox is None:
        return None
    sealed = tuple(str(value) for value in ids if str(value or ""))
    if not sealed or len(sealed) != len(set(sealed)):
        return None
    return bbox, sealed


def _exact_cell_evidence(table: Any, *, row: int, column: int) -> tuple[str, ...] | None:
    owner = _exact_cell_owner(table, row=row, column=column)
    return owner[1] if owner is not None else None


def _exact_inquiry_label_role(value: Any) -> str | None:
    label = _compact(value)
    matches = [
        role for role, aliases in _INQUIRY_LABELS.items() if label in aliases
    ]
    return matches[0] if len(matches) == 1 else None


def _exact_header_map(
    table: Any,
    rows: list[list[str]],
    row_index: int,
) -> tuple[dict[str, int], frozenset[str]] | None:
    if row_index < 0 or row_index >= len(rows):
        return None
    row = rows[row_index]
    populated = [
        (column, _exact_inquiry_label_role(cell))
        for column, cell in enumerate(row)
        if _compact(cell)
    ]
    if (
        len(populated) != len(_INQUIRY_ROLES)
        or any(role is None for _column, role in populated)
        or {role for _column, role in populated if role is not None}
        != set(_INQUIRY_ROLES)
    ):
        return None
    columns: dict[str, int] = {}
    evidence_owners: set[str] = set()
    for role, aliases in _INQUIRY_LABELS.items():
        matches = [
            column
            for column, cell in enumerate(row)
            if _compact(cell) in aliases
        ]
        if len(matches) != 1:
            return None
        column = matches[0]
        ids = _exact_cell_evidence(table, row=row_index, column=column)
        if ids is None:
            return None
        if evidence_owners.intersection(ids):
            return None
        evidence_owners.update(ids)
        columns[role] = column
    if len(set(columns.values())) != len(_INQUIRY_ROLES):
        return None
    return columns, frozenset(evidence_owners)


def _exact_split_header_map(
    table: Any,
    rows: list[list[str]],
    row_index: int,
) -> tuple[dict[str, int], frozenset[str]] | None:
    """Map one two-row header without applying any business-row repair."""

    if row_index < 0 or row_index + 1 >= len(rows):
        return None
    first, second = rows[row_index], rows[row_index + 1]
    if len(first) != len(second) or len(first) != len(_INQUIRY_ROLES):
        return None
    columns: dict[str, int] = {}
    evidence_owners: set[str] = set()
    populated_on_first: list[int] = []
    for column, (first_cell, second_cell) in enumerate(zip(first, second, strict=True)):
        populated = [
            (row, _compact(cell))
            for row, cell in ((row_index, first_cell), (row_index + 1, second_cell))
            if _compact(cell)
        ]
        if len(populated) != 1:
            return None
        source_row, label = populated[0]
        matching_roles = [
            role for role, aliases in _INQUIRY_LABELS.items() if label in aliases
        ]
        if len(matching_roles) != 1 or matching_roles[0] in columns:
            return None
        ids = _exact_cell_evidence(table, row=source_row, column=column)
        if ids is None or evidence_owners.intersection(ids):
            return None
        evidence_owners.update(ids)
        columns[matching_roles[0]] = column
        if source_row == row_index:
            populated_on_first.append(column)
    if set(columns) != set(_INQUIRY_ROLES) or len(populated_on_first) != 1:
        return None

    metadata = _value(table, "metadata", {})
    geometry = metadata.get("geometry") if isinstance(metadata, Mapping) else None
    spans = geometry.get("cell_spans") if isinstance(geometry, Mapping) else None
    matching_spans = [
        span
        for span in spans or ()
        if isinstance(span, Mapping)
        and span.get("row") == row_index
        and span.get("col") == populated_on_first[0]
        and span.get("row_span") == 2
        and span.get("col_span") == 1
    ]
    if len(matching_spans) != 1:
        return None
    return columns, frozenset(evidence_owners)


def _exact_collapsed_header_map(
    table: Any,
    rows: list[list[str]],
    row_index: int,
) -> tuple[dict[str, int], frozenset[str]] | None:
    """Recognize one exact four-column PBOC header with one missed divider."""

    if row_index < 0 or row_index >= len(rows) or len(rows[row_index]) != 4:
        return None

    metadata = _value(table, "metadata", {})
    geometry = metadata.get("geometry") if isinstance(metadata, Mapping) else None
    if not isinstance(geometry, Mapping):
        return None
    statuses = geometry.get("cell_geometry_status")
    evidence_ids = geometry.get("cell_evidence_ids")
    bboxes = geometry.get("cell_bboxes")
    spans = geometry.get("cell_spans")
    if not all(isinstance(grid, list) and row_index < len(grid) for grid in (statuses, evidence_ids, bboxes)):
        return None
    if not all(isinstance(grid[row_index], list) and len(grid[row_index]) == 4 for grid in (statuses, evidence_ids, bboxes)):
        return None
    matching_spans = [
        span
        for span in spans or ()
        if isinstance(span, Mapping)
        and span.get("row") == row_index
        and isinstance(span.get("col"), int)
        and 0 <= int(span["col"]) < len(_INQUIRY_ROLES) - 1
        and span.get("row_span") == 1
        and span.get("col_span") == 2
    ]
    if len(matching_spans) != 1:
        return None
    owner_column = int(matching_spans[0]["col"])
    covered_column = owner_column + 1
    if (
        str(statuses[row_index][owner_column] or "") != "exact"
        or str(statuses[row_index][covered_column] or "") != "derived"
        or _finite_bbox(bboxes[row_index][owner_column]) is None
        or bboxes[row_index][covered_column] is not None
        or evidence_ids[row_index][covered_column] not in ([], None)
    ):
        return None

    # The merged cell must consist of exactly two registered adjacent role
    # labels.  Matching by exact concatenation leaves no room for an unknown
    # Han business label or OCR residue to become an ownership signal.
    owner_text = _compact(rows[row_index][owner_column])
    merged_role_pairs = [
        (left_role, right_role)
        for left_role, left_aliases in _INQUIRY_LABELS.items()
        for right_role, right_aliases in _INQUIRY_LABELS.items()
        if left_role != right_role
        if any(
            owner_text == left_label + right_label
            for left_label in left_aliases
            for right_label in right_aliases
        )
    ]
    if len(merged_role_pairs) != 1 or _compact(rows[row_index][covered_column]):
        return None

    columns: dict[str, int] = {
        merged_role_pairs[0][0]: owner_column,
        merged_role_pairs[0][1]: covered_column,
    }
    for column, cell in enumerate(rows[row_index]):
        if column in {owner_column, covered_column}:
            continue
        role = _exact_inquiry_label_role(cell)
        if role is None or role in columns:
            return None
        columns[role] = column
    if set(columns) != set(_INQUIRY_ROLES) or len(set(columns.values())) != len(
        _INQUIRY_ROLES
    ):
        return None

    owners: list[str] = []
    for column in range(4):
        if column == covered_column:
            continue
        if str(statuses[row_index][column] or "") != "exact" or _finite_bbox(bboxes[row_index][column]) is None:
            return None
        raw_ids = evidence_ids[row_index][column]
        if not isinstance(raw_ids, list) or not raw_ids:
            return None
        owners.extend(str(value) for value in raw_ids if str(value or ""))
    if not owners or len(owners) != len(set(owners)):
        return None
    return columns, frozenset(owners)


def _header_owner_cells_are_exact(
    table: Any,
    *,
    coordinates: Iterable[tuple[int, int]],
) -> tuple[
    tuple[tuple[float, float, float, float], ...],
    tuple[str, ...],
] | None:
    table_bbox = _finite_bbox(_value(table, "bbox"))
    unique_coordinates = tuple(dict.fromkeys(coordinates))
    if table_bbox is None or not unique_coordinates:
        return None
    owners = [
        _exact_cell_owner(table, row=row, column=column)
        for row, column in unique_coordinates
    ]
    if any(owner is None for owner in owners):
        return None
    exact = [owner for owner in owners if owner is not None]
    boxes = tuple(owner[0] for owner in exact)
    evidence_ids = tuple(
        evidence_id for _bbox, ids in exact for evidence_id in ids
    )
    if (
        len(evidence_ids) != len(set(evidence_ids))
        or any(
            left_index != right_index
            and max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
            * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
            > 1e-8
            for left_index, left in enumerate(boxes)
            for right_index, right in enumerate(boxes)
        )
        or any(
            box[0] < table_bbox[0] - 1e-6
            or box[1] < table_bbox[1] - 1e-6
            or box[2] > table_bbox[2] + 1e-6
            or box[3] > table_bbox[3] + 1e-6
            for box in boxes
        )
    ):
        return None
    return boxes, evidence_ids


def exact_inquiry_header_owner(table: Any) -> ExactInquiryHeaderOwner | None:
    """Return the one exact semantic inquiry header owned by ``table``.

    Standard, vertically split, and one-divider-collapsed PBOC headers share
    this single contract.  Column order and width are presentation details;
    the four registered roles, exact cell geometry, and immutable evidence are
    mandatory.  Multiple candidates are ambiguous and therefore fail closed.
    """

    rows = _table_rows(table)
    candidates: list[ExactInquiryHeaderOwner] = []
    detected_header_count = 0
    for row_index in range(len(rows)):
        detections = (
            (
                _exact_header_map(table, rows, row_index),
                (row_index,),
                row_index + 1,
                "exact_single_row_header_lattice",
            ),
            (
                _exact_split_header_map(table, rows, row_index),
                (row_index, row_index + 1),
                row_index + 2,
                "exact_complementary_header_lattice",
            ),
            (
                _exact_collapsed_header_map(table, rows, row_index),
                (row_index,),
                row_index + 1,
                "exact_collapsed_colspan_header_lattice",
            ),
        )
        for detected, header_rows, body_start, binding in detections:
            if detected is None:
                continue
            detected_header_count += 1
            if any(
                any(_compact(cell) for cell in rows[prior_row])
                for prior_row in range(row_index)
            ):
                # A query schema cannot begin beneath an unrelated business
                # header inside the same physical table.  Leading empty ruled
                # bands are harmless, but residual labels make ownership
                # ambiguous and must not be skipped to find a later match.
                continue
            columns, _detected_ids = detected
            coordinates: list[tuple[int, int]] = []
            for role, column in columns.items():
                matching_rows = [
                    header_row
                    for header_row in header_rows
                    if header_row < len(rows)
                    and column < len(rows[header_row])
                    and role
                    in {
                        candidate_role
                        for candidate_role, aliases in _INQUIRY_LABELS.items()
                        if _compact(rows[header_row][column]) in aliases
                    }
                ]
                if matching_rows:
                    coordinates.append((matching_rows[0], column))
            if binding == "exact_collapsed_colspan_header_lattice":
                metadata = _value(table, "metadata", {})
                geometry = (
                    metadata.get("geometry")
                    if isinstance(metadata, Mapping)
                    else None
                )
                spans = geometry.get("cell_spans") if isinstance(geometry, Mapping) else None
                matching = [
                    span
                    for span in spans or ()
                    if isinstance(span, Mapping)
                    and span.get("row") == row_index
                    and span.get("row_span") == 1
                    and span.get("col_span") == 2
                ]
                if len(matching) != 1:
                    continue
                owner_column = int(matching[0]["col"])
                coordinates = [
                    (row_index, column)
                    for column, cell in enumerate(rows[row_index])
                    if column != owner_column + 1 and _compact(cell)
                ]
            exact_owner = _header_owner_cells_are_exact(
                table,
                coordinates=coordinates,
            )
            if exact_owner is None:
                continue
            boxes, evidence_ids = exact_owner
            if frozenset(evidence_ids) != _detected_ids:
                continue
            candidates.append(
                ExactInquiryHeaderOwner(
                    inquiry_role_columns=tuple(
                        (role, columns[role]) for role in _INQUIRY_ROLES
                    ),
                    header_rows=header_rows,
                    header_labels=tuple(
                        _INQUIRY_LABELS[role][0]
                        for role, _column in sorted(
                            columns.items(), key=lambda item: item[1]
                        )
                    ),
                    evidence_ids=evidence_ids,
                    cell_bboxes=boxes,
                    body_start=body_start,
                    binding=binding,
                )
            )
    return (
        candidates[0]
        if detected_header_count == 1 and len(candidates) == 1
        else None
    )


@dataclass(frozen=True)
class _ExactNumberingGroup:
    inquiry_type: str
    sequences: tuple[int, ...]
    evidence_ids: frozenset[str]
    document_order: int


def _inquiry_type(institution: Any, reason: Any) -> str | None:
    institution_text = _compact(institution)
    reason_text = _compact(reason)
    if not institution_text or not reason_text:
        return None
    personal_institution = institution_text == "本人"
    personal_reason = _PERSONAL_REASON_RE.fullmatch(reason_text) is not None
    if personal_institution != personal_reason:
        return None
    return "personal" if personal_institution else "institution"


def _exact_numbering_group(
    table: Any,
    rows: list[list[str]],
    *,
    body_start: int,
    columns: Mapping[str, int],
    header_evidence_ids: frozenset[str],
    document_order: int,
) -> _ExactNumberingGroup | None:
    observations: list[tuple[int, str]] = []
    group_evidence_ids = set(header_evidence_ids)
    for row_index in range(body_start, len(rows)):
        row = rows[row_index]
        if not any(_compact(cell) for cell in row):
            break
        if _exact_header_map(table, rows, row_index) is not None or _exact_split_header_map(table, rows, row_index):
            break
        if any(column >= len(row) for column in columns.values()):
            return None
        exact_ids = [
            _exact_cell_evidence(table, row=row_index, column=columns[role])
            for role in _INQUIRY_ROLES
        ]
        if any(ids is None for ids in exact_ids):
            return None
        row_evidence_ids = {
            evidence_id
            for ids in exact_ids
            if ids is not None
            for evidence_id in ids
        }
        if (
            len(row_evidence_ids) != sum(len(ids or ()) for ids in exact_ids)
            or group_evidence_ids.intersection(row_evidence_ids)
        ):
            return None
        group_evidence_ids.update(row_evidence_ids)
        raw_sequence = _compact(row[columns["sequence"]])
        if re.fullmatch(r"[1-9]\d{0,3}", raw_sequence) is None:
            return None
        if _DATE_RE.fullmatch(_compact(row[columns["inquiry_date"]])) is None:
            return None
        inquiry_type = _inquiry_type(
            row[columns["institution"]],
            row[columns["reason"]],
        )
        if inquiry_type is None:
            return None
        observations.append((int(raw_sequence), inquiry_type))
    if not observations:
        return None
    inquiry_types = {inquiry_type for _sequence, inquiry_type in observations}
    sequences = tuple(sequence for sequence, _inquiry_type_value in observations)
    if len(inquiry_types) != 1 or any(
        right != left + 1
        for left, right in zip(sequences[:-1], sequences[1:], strict=True)
    ):
        return None
    return _ExactNumberingGroup(
        inquiry_type=next(iter(inquiry_types)),
        sequences=sequences,
        evidence_ids=frozenset(group_evidence_ids),
        document_order=document_order,
    )


def _numbering_model(
    groups: Iterable[_ExactNumberingGroup],
) -> tuple[InquiryNumberingModel, str, int]:
    exact_groups = sorted(groups, key=lambda group: group.document_order)
    if len(exact_groups) < 2:
        return InquiryNumberingModel.UNKNOWN, "numbering_requires_two_exact_groups", len(exact_groups)
    owners: set[str] = set()
    for group in exact_groups:
        if owners.intersection(group.evidence_ids):
            return InquiryNumberingModel.UNKNOWN, "numbering_group_owner_conflict", len(exact_groups)
        owners.update(group.evidence_ids)
        if len(set(group.sequences)) != len(group.sequences) or any(
            right <= left
            for left, right in zip(group.sequences[:-1], group.sequences[1:], strict=True)
        ):
            return InquiryNumberingModel.UNKNOWN, "numbering_group_order_conflict", len(exact_groups)

    ordered_types: list[str] = []
    sequences_by_type: dict[str, set[int]] = {}
    for group in exact_groups:
        if not ordered_types or ordered_types[-1] != group.inquiry_type:
            ordered_types.append(group.inquiry_type)
        sequences_by_type.setdefault(group.inquiry_type, set()).update(group.sequences)
    if len(sequences_by_type) != 2 or len(ordered_types) != 2:
        return InquiryNumberingModel.UNKNOWN, "numbering_subsection_boundary_conflict", len(exact_groups)
    first_type, second_type = ordered_types
    first = sorted(sequences_by_type[first_type])
    second = sorted(sequences_by_type[second_type])
    if first[0] == second[0] == 1:
        return InquiryNumberingModel.INDEPENDENT_RESTARTS, "two_exact_subsections_restart_at_one", len(exact_groups)
    if first[0] == 1 and second[0] == first[-1] + 1:
        combined = first + second
        if combined == list(range(1, combined[-1] + 1)):
            return InquiryNumberingModel.COMBINED_CONTINUITY, "two_exact_subsections_form_one_dense_sequence", len(
                exact_groups
            )
    return InquiryNumberingModel.UNKNOWN, "numbering_restart_and_continuity_not_uniquely_proven", len(exact_groups)


def infer_inquiry_numbering_model(
    exact_groups: Iterable[tuple[str, Iterable[int], Iterable[str], int]],
) -> tuple[InquiryNumberingModel, str, int]:
    """Public value-object seam for exact extraction/coverage observations."""

    groups: list[_ExactNumberingGroup] = []
    for inquiry_type, sequences, evidence_ids, document_order in exact_groups:
        raw_sequences = tuple(sequences)
        raw_evidence_ids = tuple(str(value) for value in evidence_ids if str(value or ""))
        if any(
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence <= 0
            for sequence in raw_sequences
        ) or len(raw_evidence_ids) != len(set(raw_evidence_ids)):
            return InquiryNumberingModel.UNKNOWN, "numbering_group_owner_or_value_conflict", len(groups) + 1
        normalized = tuple(int(sequence) for sequence in raw_sequences)
        owners = frozenset(raw_evidence_ids)
        if not inquiry_type or not normalized or not owners:
            continue
        groups.append(
            _ExactNumberingGroup(
                inquiry_type=str(inquiry_type),
                sequences=normalized,
                evidence_ids=owners,
                document_order=int(document_order),
            )
        )
    return _numbering_model(groups)


def _registered_inquiry_schema(columns: Mapping[str, int]) -> PBOCInquirySchemaSpec | None:
    physical_order = tuple(
        role for role, _column in sorted(columns.items(), key=lambda item: item[1])
    )
    physical_columns = {int(column) for column in columns.values()}
    return next(
        (
            spec
            for spec in _REGISTERED_INQUIRY_SCHEMAS
            if physical_order == spec.inquiry_role_order
            and physical_columns == set(range(len(spec.inquiry_role_order)))
        ),
        None,
    )


def detect_pboc_layout_profile(pages: Iterable[Any]) -> PBOCLayoutProfile:
    """Detect one profile from exact PBOC inquiry roles and their source cells."""

    exact_headers: list[tuple[dict[str, int], frozenset[str], int]] = []
    numbering_groups: list[_ExactNumberingGroup] = []
    document_order = 0
    for page in pages or ():
        for table in _value(page, "tables", ()) or ():
            rows = _table_rows(table)
            for row_index in range(len(rows)):
                detected = _exact_header_map(table, rows, row_index)
                body_start = row_index + 1
                if detected is None:
                    detected = _exact_split_header_map(table, rows, row_index)
                    body_start = row_index + 2
                if detected is None:
                    detected = _exact_collapsed_header_map(table, rows, row_index)
                    body_start = row_index + 1
                if detected is None:
                    continue
                columns, evidence_ids = detected
                exact_headers.append((columns, evidence_ids, document_order))
                group = _exact_numbering_group(
                    table,
                    rows,
                    body_start=body_start,
                    columns=columns,
                    header_evidence_ids=evidence_ids,
                    document_order=document_order,
                )
                if group is not None:
                    numbering_groups.append(group)
                document_order += 1
                if body_start == row_index + 2:
                    break

    numbering_model, numbering_reason, exact_group_count = _numbering_model(numbering_groups)
    if not exact_headers:
        return PBOCLayoutProfile(
            pboc_family="unknown",
            layout_revision="unknown",
            inquiry_schema_profile="unknown",
            profile_version=1,
            inquiry_role_columns=(),
            inquiry_numbering_model=InquiryNumberingModel.UNKNOWN,
            capabilities=frozenset(),
            detection_reasons=("no_unique_exact_pboc_inquiry_header", numbering_reason),
            exact_numbering_group_count=exact_group_count,
        )

    header_owners: set[str] = set()
    header_owner_conflict = False
    for _columns, evidence_ids, _order in exact_headers:
        if header_owners.intersection(evidence_ids):
            header_owner_conflict = True
            break
        header_owners.update(evidence_ids)
    if header_owner_conflict:
        return PBOCLayoutProfile(
            pboc_family="pboc_personal_detailed",
            layout_revision="unknown",
            inquiry_schema_profile="unknown",
            profile_version=1,
            inquiry_role_columns=(),
            inquiry_numbering_model=InquiryNumberingModel.UNKNOWN,
            capabilities=frozenset(),
            detection_reasons=("exact_header_owner_conflict", numbering_reason),
            exact_header_count=len(exact_headers),
            exact_numbering_group_count=exact_group_count,
        )

    mappings = {
        tuple((role, columns[role]) for role in _INQUIRY_ROLES)
        for columns, _ids, _order in exact_headers
    }
    if len(mappings) != 1:
        return PBOCLayoutProfile(
            pboc_family="pboc_personal_detailed",
            layout_revision="unknown",
            inquiry_schema_profile="unknown",
            profile_version=1,
            inquiry_role_columns=(),
            inquiry_numbering_model=InquiryNumberingModel.UNKNOWN,
            capabilities=frozenset(),
            detection_reasons=("conflicting_exact_inquiry_role_maps", numbering_reason),
            exact_header_count=len(exact_headers),
            exact_numbering_group_count=exact_group_count,
        )

    role_columns = next(iter(mappings))
    schema = _registered_inquiry_schema(dict(role_columns))
    if schema is None:
        return PBOCLayoutProfile(
            pboc_family="pboc_personal_detailed",
            layout_revision="unknown",
            inquiry_schema_profile="unregistered_semantic_role_map",
            profile_version=1,
            inquiry_role_columns=role_columns,
            inquiry_numbering_model=numbering_model,
            capabilities=frozenset(),
            detection_reasons=("exact_pboc_inquiry_roles_detected", "layout_revision_not_registered", numbering_reason),
            exact_header_count=len(exact_headers),
            exact_numbering_group_count=exact_group_count,
        )
    return PBOCLayoutProfile(
        pboc_family="pboc_personal_detailed",
        layout_revision="unknown",
        inquiry_schema_profile=schema.schema_id,
        profile_version=schema.version,
        inquiry_role_columns=role_columns,
        inquiry_numbering_model=numbering_model,
        capabilities=schema.capabilities,
        detection_reasons=(
            "exact_pboc_inquiry_roles_detected",
            "registered_inquiry_role_order_matched",
            "whole_report_layout_revision_not_inferred",
            numbering_reason,
        ),
        exact_header_count=len(exact_headers),
        exact_numbering_group_count=exact_group_count,
    )
