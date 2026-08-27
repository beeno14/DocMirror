# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-triggered page evidence repair for scanned detailed PBOC reports.

Topology and template registration are intentionally outside this module.  A
repair plan can only be created from first-pass business candidates that fail
typed PBOC field contracts (or from a canonical page whose business template
could not be registered).  Targets are grouped by the already-frozen logical
page.  Deterministic repairs do not start OCR.  Other field repairs may
acquire one context-rich page OCR view, but only an unresolved template is
allowed to replace a page in the second extraction plane; ordinary field
repairs consume that view through an exact-cell overlay.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from docmirror.plugins.credit_report.personal_detail_scanned.business_repair_policy import (
    bounded_inquiry_sequence_noise_candidate,
    deterministic_agreement_institution_candidate,
    deterministic_inquiry_date_candidate,
    deterministic_liability_business_type_candidate,
    separated_leading_han_company_boundary,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
    _exact_blank_account_currency_slot_identity,
    normalize_role_candidate,
    role_candidate_is_valid,
)


def _page_number(ref: Mapping[str, Any]) -> int:
    try:
        return int(ref.get("logical_page") or ref.get("page") or 0)
    except (TypeError, ValueError):
        return 0


def _bbox(value: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def _boxes_associate(
    target: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> bool:
    tx0, ty0, tx1, ty1 = target
    cx0, cy0, cx1, cy1 = candidate
    intersection = max(0.0, min(tx1, cx1) - max(tx0, cx0)) * max(
        0.0, min(ty1, cy1) - max(ty0, cy0)
    )
    target_area = max(1.0, (tx1 - tx0) * (ty1 - ty0))
    candidate_area = max(1.0, (cx1 - cx0) * (cy1 - cy0))
    if intersection / min(target_area, candidate_area) >= 0.25:
        return True
    halo = max(2.0, min(12.0, (ty1 - ty0) * 0.5))
    center_x = (cx0 + cx1) / 2.0
    center_y = (cy0 + cy1) / 2.0
    return tx0 - halo <= center_x <= tx1 + halo and ty0 - halo <= center_y <= ty1 + halo


def _field_ref_identity(ref: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return one immutable field-cell identity used by repair matching."""

    from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
        monthly_field_slot_identity,
    )

    monthly_owner = monthly_field_slot_identity(ref)
    if monthly_owner is not None:
        return monthly_owner

    blank_currency_owner = _exact_blank_account_currency_slot_identity(ref)
    if blank_currency_owner is not None:
        return blank_currency_owner

    logical_page = _page_number(ref)
    source_page = ref.get("source_page")
    table_id = str(ref.get("table_id") or "")
    row = ref.get("row")
    column = ref.get("column")
    raw_evidence_ids = ref.get("evidence_ids")
    evidence_ids = (
        tuple(raw_evidence_ids)
        if isinstance(raw_evidence_ids, (list, tuple))
        and raw_evidence_ids
        and all(
            isinstance(value, str) and value and value == value.strip()
            for value in raw_evidence_ids
        )
        and len(raw_evidence_ids) == len(set(raw_evidence_ids))
        else ()
    )
    box = _bbox(ref)
    if (
        logical_page <= 0
        or not isinstance(source_page, int)
        or isinstance(source_page, bool)
        or source_page <= 0
        or box is None
        or not evidence_ids
        or ref.get("geometry_scope") != "cell"
        or not table_id
        or not isinstance(row, int)
        or isinstance(row, bool)
        or row < 0
        or not isinstance(column, int)
        or isinstance(column, bool)
        or column < 0
    ):
        return None
    return logical_page, source_page, table_id, row, column, box, evidence_ids


def _record_id(record: Mapping[str, Any]) -> str:
    for key in (
        "record_id",
        "inquiry_id",
        "credit_line_id",
        "liability_id",
        "repayment_id",
        "account_id",
    ):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    for key, value in record.items():
        if str(key).endswith("_id") and value not in (None, "") and not isinstance(value, (dict, list)):
            return str(value)
    return ""


def _raw_payload_index(payload: Mapping[str, Any]) -> dict[tuple[str, str, str], Any]:
    index: dict[tuple[str, str, str], Any] = {}
    for dataset_name, rows in payload.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            values = row.get("normalized") if isinstance(row.get("normalized"), Mapping) else row
            identity = _record_id(values) or _record_id(row)
            raw = row.get("canonical_raw")
            if not isinstance(raw, Mapping) and values is not row:
                raw = values.get("canonical_raw")
            if not identity or not isinstance(raw, Mapping):
                continue
            for field_name, value in raw.items():
                index[(str(dataset_name), identity, str(field_name))] = deepcopy(value)
    return index


def _published_payload_index(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str, str], Any]:
    """Index the currently published scalar independently of preserved raw OCR."""

    index: dict[tuple[str, str, str], Any] = {}
    for dataset_name, rows in payload.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            values = (
                row.get("normalized")
                if isinstance(row.get("normalized"), Mapping)
                else row
            )
            identity = _record_id(values) or _record_id(row)
            if not identity:
                continue
            for field_name, value in values.items():
                index[(str(dataset_name), identity, str(field_name))] = deepcopy(
                    value
                )
    return index


def _observed_scalar(value: Any, *, field_name: str) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0] or "")
    if isinstance(value, Mapping):
        row = value.get("row")
        if isinstance(row, Mapping):
            return str(row.get(field_name) or row.get(f"raw_{field_name}") or "")
        if isinstance(row, (list, tuple)) and len(row) >= 4:
            column = {"sequence": 0, "inquiry_date": 1, "institution": 2, "reason": 3}.get(
                field_name
            )
            if column is not None:
                return str(row[column] or "")
        direct = value.get(field_name)
        if direct not in (None, ""):
            return str(direct)
    return str(value or "")


def _repair_role(dataset_name: str, field_name: str, fallback: str) -> str:
    if field_name in {"currency", "account_currency", "reporting_amount_currency"}:
        return "currency"
    if field_name == "inquiry_date":
        return "date"
    if field_name == "sequence" and dataset_name in {"inquiries", "inquiry_records"}:
        return "inquiry_sequence"
    if field_name == "institution":
        return "institution_name"
    if field_name == "reason" and dataset_name in {"inquiries", "inquiry_records"}:
        return "inquiry_reason"
    if dataset_name == "repayment_liability_records" and field_name == "business_type":
        return "liability_business_type"
    if dataset_name == "repayment_liability_records" and field_name == "related_party_name":
        return "liability_related_party_name"
    return fallback


@dataclass(frozen=True)
class BusinessUncertainty:
    """One first-pass field or template failure eligible for page repair."""

    uncertainty_id: str
    path: str
    role: str
    dataset_name: str
    record_id: str
    field_name: str
    observed_value: Any
    reason_codes: tuple[str, ...]
    source_refs: tuple[dict[str, Any], ...]
    logical_pages: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _uncertainty_requires_page_reconstruction(
    uncertainty: BusinessUncertainty,
) -> bool:
    """Reserve page replacement for structural/template uncertainty only."""

    if not uncertainty.field_name:
        return True
    role = uncertainty.role.lower()
    return bool(
        role in {code.lower() for code in uncertainty.reason_codes}
        and any(marker in role for marker in ("structure", "template"))
    )


@dataclass(frozen=True)
class BusinessFieldRepair:
    """One field-local repair directive selected before the second pass."""

    repair_id: str
    uncertainty_id: str
    mode: str
    dataset_name: str
    record_id: str
    field_name: str
    role: str
    observed_value: str
    candidate_value: str | None
    source_refs: tuple[dict[str, Any], ...]
    reason_codes: tuple[str, ...]
    published_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BusinessRepairPlan:
    """Detached plan and audit ledger for one document-wide repair pass."""

    uncertainties: tuple[BusinessUncertainty, ...] = ()
    unresolved_template_pages: tuple[int, ...] = ()
    affected_pages: tuple[int, ...] = ()
    field_repairs: tuple[BusinessFieldRepair, ...] = ()
    page_evidence: dict[int, dict[str, Any]] = field(default_factory=dict)
    reconstruction_evidence: dict[int, dict[str, Any]] = field(default_factory=dict)
    page_decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def requires_second_pass(self) -> bool:
        return bool(self.affected_pages)

    def audit(self) -> dict[str, Any]:
        scoped = sum(bool(item.logical_pages) for item in self.uncertainties)
        deterministic = [repair for repair in self.field_repairs if repair.mode == "deterministic"]
        context_rich = [repair for repair in self.field_repairs if repair.mode == "context_rich_reocr"]
        return {
            "architecture": "schema_triggered_field_local_repair_v2",
            "first_pass_uncertainty_count": len(self.uncertainties),
            "page_scoped_uncertainty_count": scoped,
            "unscoped_uncertainty_count": len(self.uncertainties) - scoped,
            "unresolved_template_pages": list(self.unresolved_template_pages),
            "affected_pages": list(self.affected_pages),
            "deterministic_field_repair_count": len(deterministic),
            "context_rich_reocr_field_count": len(context_rich),
            "field_repairs": [repair.to_dict() for repair in self.field_repairs],
            "reconstruction_pages": sorted(self.reconstruction_evidence),
            "second_schema_pass_required": self.requires_second_pass,
            "topology_ocr_requests": 0,
            "field_triggered_ocr_requests": sum(
                int(decision.get("ocr_invocations") or 0) for decision in self.page_decisions
            ),
            "page_decisions": deepcopy(self.page_decisions),
        }

    def field_repair_for(
        self,
        *,
        dataset_name: str,
        record_id: str,
        field_name: str,
        observed_value: Any,
        source_refs: Iterable[Mapping[str, Any]],
        mode: str | None = None,
    ) -> BusinessFieldRepair | None:
        """Resolve one directive by record identity or exact field-cell owner."""

        observed = str(observed_value or "")
        caller_owners = {
            owner
            for ref in source_refs
            if isinstance(ref, Mapping) and (owner := _field_ref_identity(ref)) is not None
        }
        eligible: list[tuple[BusinessFieldRepair, bool, bool]] = []
        for repair in self.field_repairs:
            repair_owners = {
                owner
                for ref in repair.source_refs
                if (owner := _field_ref_identity(ref)) is not None
            }
            identity_matches = bool(record_id and repair.record_id and record_id == repair.record_id)
            owner_matches = bool(caller_owners and repair_owners and caller_owners & repair_owners)
            if (
                repair.dataset_name != dataset_name
                or repair.field_name != field_name
                or (mode is not None and repair.mode != mode)
                or observed != repair.observed_value
            ):
                continue
            eligible.append((repair, owner_matches, identity_matches))
        owner_candidates = [repair for repair, owner, _identity in eligible if owner]
        if caller_owners:
            return owner_candidates[0] if len(owner_candidates) == 1 else None
        identity_candidates = [
            repair for repair, _owner, identity in eligible if identity
        ]
        return identity_candidates[0] if len(identity_candidates) == 1 else None


class BusinessUncertaintyRepairCoordinator:
    """Plan and resolve the only post-schema page repair stage."""

    def __init__(self, parse_result: Any, *, monthly_context: Any | None = None) -> None:
        self.parse_result = parse_result
        self.monthly_context = monthly_context if monthly_context is not None else parse_result

    def _monthly_uncertainties(self, payload: Mapping[str, Any]) -> list[BusinessUncertainty]:
        """Authorize only slots independently bound to the sealed month lattice."""

        from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
            MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM,
            _MonthlySourceEvidence,
            authenticated_monthly_field_slots,
        )

        evidence = _MonthlySourceEvidence(self.parse_result)
        uncertainties: list[BusinessUncertainty] = []
        physical_owners: dict[tuple[Any, ...], set[str]] = {}
        rows_and_slots: list[tuple[Mapping[str, Any], dict[str, dict[str, Any]]]] = []
        for record in payload.get("repayment_records") or ():
            if not isinstance(record, Mapping):
                continue
            slots = authenticated_monthly_field_slots(self.monthly_context, record, evidence=evidence)
            rows_and_slots.append((record, slots))
            for ref in slots.values():
                owner = _field_ref_identity(ref)
                if owner is not None:
                    physical_owners.setdefault(owner[:-1], set()).add(ref["monthly_slot_proof"]["account_id"])
        for record, slots in rows_and_slots:
            values = record.get("normalized") if isinstance(record.get("normalized"), Mapping) else record
            record_id = _record_id(values) or _record_id(record)
            if not record_id:
                continue
            for field_name, ref in slots.items():
                identity = _field_ref_identity(ref)
                if identity is None or len(physical_owners.get(identity[:-1], ())) != 1:
                    continue
                role = "repayment_status" if field_name in {"status", "status_code"} else "amount"
                published = values.get(field_name)
                valid = role_candidate_is_valid("" if published is None else str(published), role)
                score = ref.get("source_ocr_confidence")
                low_confidence = bool(
                    role == "repayment_status" and valid
                    and isinstance(score, (int, float)) and not isinstance(score, bool)
                    and score < MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM
                )
                if valid and not low_confidence:
                    continue
                observed = str(ref.get("observed_raw") or "")
                reason = "low_source_ocr_confidence" if low_confidence else "monthly_field_value_unresolved"
                marker = (record_id, field_name, _field_ref_identity(ref), observed, reason)
                digest = hashlib.sha256(repr(marker).encode("utf-8")).hexdigest()[:16]
                uncertainties.append(BusinessUncertainty(
                    uncertainty_id=f"personal_detail_business_uncertainty:{digest}",
                    path=f"repayment_records[{record_id}].{field_name}", role=role,
                    dataset_name="repayment_records", record_id=record_id, field_name=field_name,
                    observed_value=observed, reason_codes=(reason, "exact_monthly_source_slot", "field_local_overlay_only"),
                    source_refs=(deepcopy(ref),), logical_pages=(ref["logical_page"],),
                ))
        return uncertainties

    @staticmethod
    def _proactive_liability_uncertainties(
        payload: Mapping[str, Any],
    ) -> list[BusinessUncertainty]:
        """Target two bounded liability defects that broad text contracts admit."""

        uncertainties: list[BusinessUncertainty] = []
        for record in payload.get("repayment_liability_records") or ():
            if not isinstance(record, Mapping):
                continue
            values = record.get("normalized") if isinstance(record.get("normalized"), Mapping) else record
            record_identity = _record_id(values) or _record_id(record)
            raw_map = record.get("canonical_raw")
            if not isinstance(raw_map, Mapping) and values is not record:
                raw_map = values.get("canonical_raw")
            refs_by_field = record.get("source_refs_by_field")
            if not isinstance(refs_by_field, Mapping) and values is not record:
                refs_by_field = values.get("source_refs_by_field")
            if not record_identity or not isinstance(raw_map, Mapping) or not isinstance(refs_by_field, Mapping):
                continue
            candidates = {
                "business_type": deterministic_liability_business_type_candidate(
                    raw_map.get("business_type")
                ),
                "related_party_name": (
                    "context_rich_reocr"
                    if separated_leading_han_company_boundary(raw_map.get("related_party_name"))
                    else None
                ),
            }
            for field_name, candidate in candidates.items():
                if candidate is None:
                    continue
                refs = tuple(
                    dict(ref)
                    for ref in refs_by_field.get(field_name) or ()
                    if isinstance(ref, Mapping)
                )
                exact_refs = tuple(ref for ref in refs if _field_ref_identity(ref) is not None)
                if len(exact_refs) != 1:
                    continue
                observed = deepcopy(raw_map.get(field_name))
                marker = "\x1f".join(
                    (
                        "proactive_liability_field_repair",
                        record_identity,
                        field_name,
                        repr(observed),
                        repr(exact_refs),
                    )
                )
                digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()[:16]
                uncertainties.append(
                    BusinessUncertainty(
                        uncertainty_id=f"personal_detail_business_uncertainty:{digest}",
                        path=f"repayment_liability_records[{record_identity}].{field_name}",
                        role=_repair_role(
                            "repayment_liability_records",
                            field_name,
                            field_name,
                        ),
                        dataset_name="repayment_liability_records",
                        record_id=record_identity,
                        field_name=field_name,
                        observed_value=observed,
                        reason_codes=(
                            "proactive_bounded_field_policy",
                            (
                                "unique_closed_vocabulary_edge_noise"
                                if field_name == "business_type"
                                else "separated_leading_han_company_boundary"
                            ),
                        ),
                        source_refs=exact_refs,
                        logical_pages=tuple(
                            sorted({_page_number(ref) for ref in exact_refs if _page_number(ref) > 0})
                        ),
                    )
                )
        return uncertainties

    @staticmethod
    def _field_repairs(
        uncertainties: Iterable[BusinessUncertainty],
        payload: Mapping[str, Any],
    ) -> tuple[BusinessFieldRepair, ...]:
        raw_index = _raw_payload_index(payload)
        published_index = _published_payload_index(payload)
        repairs: list[BusinessFieldRepair] = []
        seen: set[tuple[Any, ...]] = set()
        for uncertainty in uncertainties:
            if not uncertainty.field_name or not uncertainty.logical_pages:
                continue
            exact_refs = tuple(
                dict(ref)
                for ref in uncertainty.source_refs
                if _field_ref_identity(ref) is not None
            )
            if len(exact_refs) != 1:
                # Page/table reconstruction may still run, but no scalar may be
                # overlaid without one exact source-owned field cell.
                continue
            observed_raw = raw_index.get(
                (
                    uncertainty.dataset_name,
                    uncertainty.record_id,
                    uncertainty.field_name,
                )
            )
            observed = _observed_scalar(
                uncertainty.observed_value
                if observed_raw is None or uncertainty.dataset_name == "repayment_records"
                else observed_raw,
                field_name=uncertainty.field_name,
            )
            role = _repair_role(
                uncertainty.dataset_name,
                uncertainty.field_name,
                uncertainty.role,
            )
            if not role or role in uncertainty.reason_codes:
                # A field-shaped source ref does not turn a structural issue
                # code into a scalar validation contract. Keep those findings
                # on the existing page-reconstruction fallback path.
                continue
            candidate: str | None = None
            mode = "context_rich_reocr"
            reason_codes: tuple[str, ...] = (
                "deterministic_evidence_insufficient",
                "independent_context_rich_reocr_required",
                "field_local_overlay_only",
            )
            if uncertainty.dataset_name == "repayment_records":
                reason_codes = tuple(dict.fromkeys((*reason_codes, *uncertainty.reason_codes)))
            if role == "date" and uncertainty.field_name == "inquiry_date":
                candidate = deterministic_inquiry_date_candidate(observed)
                if candidate is not None:
                    mode = "deterministic"
                    reason_codes = (
                        "one_valid_date_in_exact_owned_cell",
                        "short_nonnumeric_edge_residue",
                        "unique_deterministic_candidate",
                        "field_local_overlay_only",
                    )
            elif role == "inquiry_sequence":
                ordinal = bounded_inquiry_sequence_noise_candidate(observed)
                if ordinal is not None:
                    candidate = str(ordinal[0])
                    mode = "deterministic"
                    reason_codes = (
                        "one_bounded_sequence_edge_glyph",
                        "adjacent_outer_bracket_required_at_materialization",
                        "field_local_overlay_only",
                    )
            elif uncertainty.dataset_name == "credit_lines" and uncertainty.field_name == "institution":
                candidate = deterministic_agreement_institution_candidate(observed)
                if candidate is not None:
                    mode = "deterministic"
                    reason_codes = (
                        "one_complete_adjacent_agreement_label",
                        "one_legal_institution_name_span",
                        "unique_deterministic_candidate",
                        "field_local_overlay_only",
                    )
            elif role == "liability_business_type":
                candidate = deterministic_liability_business_type_candidate(observed)
                if candidate is not None:
                    mode = "deterministic"
                    reason_codes = (
                        "one_closed_liability_business_type",
                        "one_edge_noise_glyph",
                        "unique_deterministic_candidate",
                        "field_local_overlay_only",
                    )
            marker = (
                uncertainty.dataset_name,
                uncertainty.record_id if uncertainty.dataset_name == "repayment_records" else "",
                uncertainty.field_name,
                observed,
                candidate,
                mode,
                _field_ref_identity(exact_refs[0]),
            )
            if marker in seen:
                continue
            seen.add(marker)
            digest = hashlib.sha256(repr(marker).encode("utf-8")).hexdigest()[:16]
            repairs.append(
                BusinessFieldRepair(
                    repair_id=f"personal_detail_business_field_repair:{digest}",
                    uncertainty_id=uncertainty.uncertainty_id,
                    mode=mode,
                    dataset_name=uncertainty.dataset_name,
                    record_id=uncertainty.record_id,
                    field_name=uncertainty.field_name,
                    role=role,
                    observed_value=observed,
                    candidate_value=candidate,
                    source_refs=exact_refs,
                    reason_codes=reason_codes,
                    published_value=deepcopy(
                        published_index.get(
                            (
                                uncertainty.dataset_name,
                                uncertainty.record_id,
                                uncertainty.field_name,
                            )
                        )
                    ),
                )
            )
        return tuple(repairs)

    def plan(
        self,
        payload: Mapping[str, Any],
        *,
        canonical_audit: Mapping[str, Any],
        extraction_issues: Iterable[Mapping[str, Any]] = (),
    ) -> BusinessRepairPlan:
        detector = PersonalDetailOCRCorrectionOverlay(self.parse_result)
        detector.correct_business_candidates(payload, stage="candidate_b_first_pass_validation")
        uncertainties: list[BusinessUncertainty] = []
        for anomaly in detector.audit().get("cell_anomalies") or ():
            if not isinstance(anomaly, Mapping):
                continue
            refs = tuple(dict(ref) for ref in anomaly.get("source_refs") or () if isinstance(ref, Mapping))
            pages = tuple(sorted({_page_number(ref) for ref in refs if _page_number(ref) > 0}))
            marker = "\x1f".join(
                (
                    str(anomaly.get("path") or ""),
                    str(anomaly.get("role") or ""),
                    str(anomaly.get("value") or ""),
                    repr(refs),
                )
            )
            digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()[:16]
            uncertainties.append(
                BusinessUncertainty(
                    uncertainty_id=f"personal_detail_business_uncertainty:{digest}",
                    path=str(anomaly.get("path") or ""),
                    role=str(anomaly.get("role") or ""),
                    dataset_name=str(anomaly.get("dataset_name") or ""),
                    record_id=str(anomaly.get("record_id") or ""),
                    field_name=str(anomaly.get("field_name") or ""),
                    observed_value=deepcopy(anomaly.get("value")),
                    reason_codes=tuple(str(code) for code in anomaly.get("reason_codes") or ()),
                    source_refs=refs,
                    logical_pages=pages,
                )
            )
        known_ids = {item.uncertainty_id for item in uncertainties}
        for issue in extraction_issues:
            if not isinstance(issue, Mapping) or str(issue.get("status") or "open") == "resolved":
                continue
            refs = tuple(dict(ref) for ref in issue.get("source_refs") or () if isinstance(ref, Mapping))
            pages = tuple(sorted({_page_number(ref) for ref in refs if _page_number(ref) > 0}))
            issue_code = str(issue.get("issue_code") or "business_structure_uncertain")
            marker = "\x1f".join(
                (
                    issue_code,
                    str(issue.get("target_dataset") or ""),
                    str(issue.get("target_record_id") or ""),
                    str(issue.get("field_name") or ""),
                    repr(refs),
                )
            )
            digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()[:16]
            uncertainty_id = f"personal_detail_business_uncertainty:{digest}"
            if uncertainty_id in known_ids:
                continue
            known_ids.add(uncertainty_id)
            uncertainties.append(
                BusinessUncertainty(
                    uncertainty_id=uncertainty_id,
                    path=str(issue.get("target_dataset") or ""),
                    role=issue_code,
                    dataset_name=str(issue.get("target_dataset") or ""),
                    record_id=str(issue.get("target_record_id") or ""),
                    field_name=str(issue.get("field_name") or ""),
                    observed_value=deepcopy(issue.get("observed_value")),
                    reason_codes=tuple(
                        dict.fromkeys(
                            (issue_code, *(str(code) for code in issue.get("reason_codes") or ()))
                        )
                    ),
                    source_refs=refs,
                    logical_pages=pages,
                )
            )
        for uncertainty in self._proactive_liability_uncertainties(payload):
            if uncertainty.uncertainty_id in known_ids:
                continue
            known_ids.add(uncertainty.uncertainty_id)
            uncertainties.append(uncertainty)
        monthly_uncertainties = self._monthly_uncertainties(payload)
        for uncertainty in monthly_uncertainties:
            if uncertainty.uncertainty_id not in known_ids:
                known_ids.add(uncertainty.uncertainty_id)
                uncertainties.append(uncertainty)
        authenticated_monthly_ids = {item.uncertainty_id for item in monthly_uncertainties}
        # Keep unrepairable monthly uncertainties visible in the audit without
        # treating their caller-supplied grid refs as repair authorization.
        field_repairs = self._field_repairs(
            (item for item in uncertainties if item.dataset_name != "repayment_records"
             or item.uncertainty_id in authenticated_monthly_ids),
            payload,
        )
        unresolved = tuple(
            sorted(
                {
                    int(page)
                    for page in canonical_audit.get("unresolved_pages") or ()
                    if int(page) > 0
                }
            )
        )
        repair_pages = {
            _page_number(ref)
            for repair in field_repairs
            for ref in repair.source_refs
            if _page_number(ref) > 0
        }
        structural_pages = {
            page
            for uncertainty in uncertainties
            if _uncertainty_requires_page_reconstruction(uncertainty)
            for page in uncertainty.logical_pages
            if page > 0
        }
        pages = tuple(
            sorted(
                repair_pages
                | structural_pages
                | set(unresolved)
            )
        )
        return BusinessRepairPlan(
            uncertainties=tuple(uncertainties),
            unresolved_template_pages=unresolved,
            affected_pages=pages,
            field_repairs=field_repairs,
        )

    def resolve_page_evidence(
        self,
        plan: BusinessRepairPlan,
        *,
        source_pages: Iterable[Mapping[str, Any]],
        page_ocr_loader: Callable[..., list[dict[str, Any]]],
    ) -> BusinessRepairPlan:
        by_page = {
            int(page.get("page") or 0): deepcopy(dict(page))
            for page in source_pages
            if isinstance(page, Mapping) and int(page.get("page") or 0) > 0
        }
        targets_by_page: dict[int, list[BusinessUncertainty]] = {}
        for uncertainty in plan.uncertainties:
            for logical_page in uncertainty.logical_pages:
                targets_by_page.setdefault(logical_page, []).append(uncertainty)
        repairs_by_page: dict[int, list[BusinessFieldRepair]] = {}
        for repair in plan.field_repairs:
            for logical_page in {
                _page_number(ref) for ref in repair.source_refs if _page_number(ref) > 0
            }:
                repairs_by_page.setdefault(logical_page, []).append(repair)

        for logical_page in plan.affected_pages:
            source = by_page.get(logical_page)
            targets = targets_by_page.get(logical_page, [])
            field_repairs = repairs_by_page.get(logical_page, [])
            registration_unresolved = logical_page in plan.unresolved_template_pages
            deterministic_repairs = [
                repair for repair in field_repairs if repair.mode == "deterministic"
            ]
            context_rich_repairs = [
                repair
                for repair in field_repairs
                if repair.mode == "context_rich_reocr"
            ]
            def target_is_directed(target: BusinessUncertainty) -> bool:
                target_owners = {
                    owner
                    for ref in target.source_refs
                    if (owner := _field_ref_identity(ref)) is not None
                }
                return any(
                    repair.dataset_name == target.dataset_name
                    and repair.field_name == target.field_name
                    and bool(
                        target_owners
                        & {
                            owner
                            for ref in repair.source_refs
                            if (owner := _field_ref_identity(ref)) is not None
                        }
                    )
                    for repair in field_repairs
                )

            undirected_targets = [
                target
                for target in targets
                if not target_is_directed(target)
            ]
            structural_targets = [
                target
                for target in undirected_targets
                if _uncertainty_requires_page_reconstruction(target)
            ]
            unrepairable_field_targets = [
                target
                for target in undirected_targets
                if not _uncertainty_requires_page_reconstruction(target)
            ]
            requires_context_rich_reocr = bool(
                registration_unresolved
                or context_rich_repairs
                or structural_targets
            )
            requires_page_reconstruction = bool(
                registration_unresolved or structural_targets
            )

            if not requires_context_rich_reocr:
                plan.page_decisions.append(
                    {
                        "logical_page": logical_page,
                        "mode": "deterministic_field_repair_only",
                        "ocr_invocations": 0,
                        "target_count": len(targets),
                        "deterministic_field_count": len(deterministic_repairs),
                        "unrepairable_field_count": len(unrepairable_field_targets),
                        "acquisition_scope": "none",
                        "mutation_scope": "field",
                    }
                )
                continue

            reason = (
                "business_schema_template_unresolved"
                if registration_unresolved
                else "business_field_context_rich_reocr_required"
            )
            replay = page_ocr_loader({logical_page}, reason=reason)
            replacement = next(
                (dict(page) for page in replay or () if int(page.get("page") or 0) == logical_page),
                None,
            )
            if replacement is not None:
                plan.page_evidence[logical_page] = deepcopy(replacement)
                if requires_page_reconstruction:
                    plan.reconstruction_evidence[logical_page] = deepcopy(replacement)
                    mode = "one_shot_context_rich_page_ocr_with_reconstruction"
                else:
                    mode = "one_shot_context_rich_page_ocr_field_overlay_only"
            elif source is not None and requires_page_reconstruction:
                plan.page_evidence[logical_page] = source
                plan.reconstruction_evidence[logical_page] = deepcopy(source)
                mode = "page_ocr_failed_existing_evidence_retained"
            elif source is not None:
                mode = "page_ocr_failed_existing_evidence_not_used_for_field_repair"
            else:
                mode = "page_ocr_failed_no_evidence"
            plan.page_decisions.append(
                {
                    "logical_page": logical_page,
                    "mode": mode,
                    "ocr_invocations": 1,
                    "reason": reason,
                    "target_count": len(targets),
                    "deterministic_field_count": len(deterministic_repairs),
                    "context_rich_reocr_field_count": len(context_rich_repairs),
                    "unrepairable_field_count": len(unrepairable_field_targets),
                    "acquisition_scope": "logical_page",
                    "mutation_scope": "field",
                    "page_reconstruction": requires_page_reconstruction,
                }
            )
        return plan

    @staticmethod
    def _source_supports_target(
        page: Mapping[str, Any],
        uncertainty: BusinessUncertainty,
        *,
        logical_page: int,
    ) -> bool:
        """Require role-valid text, not merely an overlapping OCR rectangle."""
        # Structure/template uncertainties do not have a typed scalar contract;
        # existing first-pass text cannot prove that the missing structure was
        # recovered, so they always request the page's one allowed OCR pass.
        if not uncertainty.role or uncertainty.role in set(uncertainty.reason_codes):
            return False
        refs = [ref for ref in uncertainty.source_refs if _page_number(ref) == logical_page]
        if not refs:
            return False
        for ref in refs:
            if ref.get("geometry_scope") != "cell" and ref.get("binding") != "canonical_field_slot":
                # A table/page rectangle contains several values with the same
                # semantic role.  Finding one valid number or date somewhere
                # inside it cannot prove the requested field is supported.
                return False
            target = _bbox(ref)
            if target is None:
                return False
            associated = [
                str(line.get("text") or line.get("content") or "").strip()
                for line in page.get("lines") or ()
                if isinstance(line, Mapping)
                and (candidate := _bbox(line)) is not None
                and _boxes_associate(target, candidate)
            ]
            candidates = [*associated]
            if associated:
                candidates.append("".join(associated))
                candidates.append(" ".join(associated))
            if not any(
                role_candidate_is_valid(normalize_role_candidate(value, uncertainty.role), uncertainty.role)
                for value in candidates
                if value
            ):
                return False
        return True


def apply_planned_monthly_field_repairs(
    context: Any, payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize only approved monthly fields, including null source values.

    Reconstructing an affected page may produce a different grid-local ID. The
    immutable source slot, not that detector ID or a neighbouring value, binds
    this pass back to its directive. A failed attempt never promotes raw text.
    """

    from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
        _MonthlySourceEvidence,
        authenticated_monthly_field_slots,
        monthly_field_slot_identity,
    )

    corrected = deepcopy(dict(payload))
    plan = getattr(context, "_business_repair_plan", None)
    overlay = getattr(context, "_ocr_correction_overlay", None)
    if plan is None or overlay is None or not getattr(context, "_business_repair_active", False):
        return corrected
    directives = [
        repair for repair in getattr(plan, "field_repairs", ())
        if repair.dataset_name == "repayment_records" and repair.mode == "context_rich_reocr"
    ]
    if not directives:
        return corrected
    evidence = _MonthlySourceEvidence(context.parse_result)
    records_and_slots = []
    physical_records: dict[tuple[Any, ...], set[tuple[str, str]]] = {}
    for record in corrected.get("repayment_records") or ():
        if not isinstance(record, dict):
            continue
        values = record.get("normalized") if isinstance(record.get("normalized"), dict) else record
        slots = authenticated_monthly_field_slots(context, record, evidence=evidence)
        records_and_slots.append((record, values, slots))
        record_id = _record_id(values) or _record_id(record)
        for ref in slots.values():
            owner = monthly_field_slot_identity(ref)
            if owner is not None:
                physical_records.setdefault(owner[:-1], set()).add((owner[-1], record_id))
    for record, values, slots in records_and_slots:
        for field_name, ref in slots.items():
            owner = monthly_field_slot_identity(ref)
            if owner is None or len(physical_records.get(owner[:-1], ())) != 1:
                continue
            record_id = _record_id(values) or _record_id(record)

            def record_matches(repair: BusinessFieldRepair) -> bool:
                if record_id and record_id == repair.record_id:
                    return True
                # A page rebuild can change a detector-local grid name. Only
                # its generated grid:YYYY-MM alias is transferable; arbitrary
                # record IDs cannot borrow another record's directive.
                proof = ref["monthly_slot_proof"]
                suffix = f":{proof['year']:04d}-{proof['month']:02d}"
                old_grid = repair.source_refs[0].get("registered_source_ref", {}).get("grid_id")
                new_grid = values.get("grid_id") or record.get("grid_id")
                return bool(old_grid and new_grid and repair.record_id == f"{old_grid}{suffix}" and record_id == f"{new_grid}{suffix}")

            matches = [
                repair for repair in directives
                if repair.field_name == field_name and len(repair.source_refs) == 1
                and monthly_field_slot_identity(repair.source_refs[0]) == owner
                and record_matches(repair)
            ]
            if owner is None or len(matches) != 1:
                continue
            repair = matches[0]
            # Re-authentication above also verifies that the stored raw token
            # set still owns this slot. Never substitute an unrelated current
            # value for the original input used to authorize the repair.
            if str(ref.get("observed_raw") or "") != repair.observed_value:
                continue
            current = values.get(field_name)
            current_text = "" if current is None else str(current)
            published_text = "" if repair.published_value is None else str(repair.published_value)
            confirmation_only = bool(
                role_candidate_is_valid(current_text, repair.role)
                and current_text != published_text
            )
            updated, decision = overlay.repair_planned_text(
                repair.observed_value, repair=repair, source_refs=(ref,),
                confirmation_value=current_text if confirmation_only else None,
            )
            if decision is None or decision.action not in {"applied", "confirmed"}:
                continue
            if not role_candidate_is_valid(updated, repair.role):
                continue
            original = values.get(field_name)
            raw = record.setdefault("canonical_raw", {})
            if isinstance(raw, dict):
                raw.setdefault(field_name, repair.observed_value or original)
            values[field_name] = updated
            if values is not record and field_name in record:
                record[field_name] = updated
            aliases = {"status", "status_code"} if repair.role == "repayment_status" else {"overdue_amount", "status_amount"}
            for alias in aliases:
                if alias in values:
                    values[alias] = updated
                if values is not record and alias in record:
                    record[alias] = updated
            field_ref = deepcopy(ref["registered_source_ref"])
            field_ref["field_name"] = field_name
            old_refs = [dict(item) for item in record.get("source_cell_refs") or () if isinstance(item, Mapping)]
            record["source_cell_refs"] = [item for item in old_refs if item.get("field_name") not in aliases] + [field_ref]
            refs_by_field = record.setdefault("source_refs_by_field", {})
            if isinstance(refs_by_field, dict):
                for alias in aliases:
                    if alias in refs_by_field:
                        refs_by_field[alias] = [deepcopy(field_ref), deepcopy(decision.source_refs[-1])]
                refs_by_field[field_name] = [deepcopy(field_ref), deepcopy(decision.source_refs[-1])]
            audit = record.setdefault("audit", {})
            if isinstance(audit, dict):
                history = audit.setdefault("monthly_field_repairs", [])
                if isinstance(history, list):
                    history.append({
                        "field_name": field_name, "correction_id": decision.correction_id,
                        "action": decision.action, "original": original, "corrected": updated,
                        "selected_acquisition": decision.selected_acquisition,
                        "source_ocr_confidence": ref.get("source_ocr_confidence"),
                        "selected_ocr_confidence": decision.confidence,
                        "original_source_refs": [item for item in old_refs if item.get("field_name") in aliases],
                    })
                unresolved = audit.get("unresolved_fields")
                if isinstance(unresolved, list):
                    audit["unresolved_fields"] = [item for item in unresolved if item not in aliases]
            for container in (record, values):
                unresolved = container.get("_unresolved_fields")
                if isinstance(unresolved, list):
                    container["_unresolved_fields"] = [item for item in unresolved if item not in aliases]
            if repair.role == "amount":
                for container in (record, values):
                    pairing = container.get("_amount_pairing")
                    if isinstance(pairing, dict):
                        pairing["status"] = "exact"
                        pairing["reason"] = "independent_page_evidence_exact_month_slot"
            status = values.get("status_code", values.get("status"))
            amount = values.get("status_amount", values.get("overdue_amount"))
            if (
                isinstance(audit, dict)
                and audit.get("reason") in {"status_value_withheld", "corrected_status_planes_disagree"}
                and not audit.get("unresolved_fields") and not record.get("_unresolved_fields")
                and role_candidate_is_valid("" if status is None else str(status), "repayment_status")
                and role_candidate_is_valid("" if amount is None else str(amount), "amount")
            ):
                audit["resolved_source_reason"] = audit.pop("reason")
                record.pop("extraction_status", None)
                if values is not record:
                    values.pop("extraction_status", None)
    return corrected


__all__ = [
    "BusinessFieldRepair",
    "BusinessRepairPlan",
    "BusinessUncertainty",
    "BusinessUncertaintyRepairCoordinator",
    "apply_planned_monthly_field_repairs",
]
