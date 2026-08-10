# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-triggered page evidence repair for scanned detailed PBOC reports.

Topology and template registration are intentionally outside this module.  A
repair plan can only be created from first-pass business candidates that fail
typed PBOC field contracts (or from a canonical page whose business template
could not be registered).  Targets are grouped by the already-frozen logical
page.  Existing complete-page evidence is reused when it covers the target;
otherwise the coordinator requests the page's single allowed OCR pass.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
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


@dataclass(frozen=True)
class BusinessUncertainty:
    """One first-pass field or template failure eligible for page repair."""

    uncertainty_id: str
    path: str
    role: str
    dataset_name: str
    record_id: str
    field_name: str
    observed_value: str
    reason_codes: tuple[str, ...]
    source_refs: tuple[dict[str, Any], ...]
    logical_pages: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BusinessRepairPlan:
    """Detached plan and audit ledger for one document-wide repair pass."""

    uncertainties: tuple[BusinessUncertainty, ...] = ()
    unresolved_template_pages: tuple[int, ...] = ()
    affected_pages: tuple[int, ...] = ()
    page_evidence: dict[int, dict[str, Any]] = field(default_factory=dict)
    page_decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def requires_second_pass(self) -> bool:
        return bool(self.affected_pages)

    def audit(self) -> dict[str, Any]:
        scoped = sum(bool(item.logical_pages) for item in self.uncertainties)
        return {
            "architecture": "schema_triggered_page_repair_v1",
            "first_pass_uncertainty_count": len(self.uncertainties),
            "page_scoped_uncertainty_count": scoped,
            "unscoped_uncertainty_count": len(self.uncertainties) - scoped,
            "unresolved_template_pages": list(self.unresolved_template_pages),
            "affected_pages": list(self.affected_pages),
            "second_schema_pass_required": self.requires_second_pass,
            "topology_ocr_requests": 0,
            "field_triggered_ocr_requests": sum(
                int(decision.get("ocr_invocations") or 0) for decision in self.page_decisions
            ),
            "page_decisions": deepcopy(self.page_decisions),
        }


class BusinessUncertaintyRepairCoordinator:
    """Plan and resolve the only post-schema page repair stage."""

    def __init__(self, parse_result: Any) -> None:
        self.parse_result = parse_result

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
                    observed_value=str(anomaly.get("value") or ""),
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
                    observed_value=str(issue.get("observed_value") or ""),
                    reason_codes=tuple(
                        dict.fromkeys(
                            (issue_code, *(str(code) for code in issue.get("reason_codes") or ()))
                        )
                    ),
                    source_refs=refs,
                    logical_pages=pages,
                )
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
        pages = tuple(
            sorted(
                {
                    page
                    for item in uncertainties
                    for page in item.logical_pages
                }
                | set(unresolved)
            )
        )
        return BusinessRepairPlan(
            uncertainties=tuple(uncertainties),
            unresolved_template_pages=unresolved,
            affected_pages=pages,
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

        for logical_page in plan.affected_pages:
            source = by_page.get(logical_page)
            targets = targets_by_page.get(logical_page, [])
            registration_unresolved = logical_page in plan.unresolved_template_pages
            source_supports_targets = bool(source and source.get("lines")) and not registration_unresolved
            if source_supports_targets:
                source_supports_targets = bool(targets) and all(
                    self._source_supports_target(source, target, logical_page=logical_page)
                    for target in targets
                )

            if source_supports_targets and source is not None:
                plan.page_evidence[logical_page] = source
                plan.page_decisions.append(
                    {
                        "logical_page": logical_page,
                        "mode": "existing_complete_page_evidence",
                        "ocr_invocations": 0,
                        "target_count": len(targets),
                    }
                )
                continue

            reason = (
                "business_schema_template_unresolved"
                if registration_unresolved
                else "business_field_evidence_insufficient"
            )
            replay = page_ocr_loader({logical_page}, reason=reason)
            replacement = next(
                (dict(page) for page in replay or () if int(page.get("page") or 0) == logical_page),
                None,
            )
            if replacement is not None:
                plan.page_evidence[logical_page] = deepcopy(replacement)
                mode = "one_shot_complete_page_ocr"
            elif source is not None:
                plan.page_evidence[logical_page] = source
                mode = "page_ocr_failed_existing_evidence_retained"
            else:
                mode = "page_ocr_failed_no_evidence"
            plan.page_decisions.append(
                {
                    "logical_page": logical_page,
                    "mode": mode,
                    "ocr_invocations": 1,
                    "reason": reason,
                    "target_count": len(targets),
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


__all__ = [
    "BusinessRepairPlan",
    "BusinessUncertainty",
    "BusinessUncertaintyRepairCoordinator",
]
