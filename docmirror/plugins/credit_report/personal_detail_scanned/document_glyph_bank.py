# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict document-local N/asterisk repayment glyph corroboration.

The bank is deliberately ephemeral.  It consumes normalized bitmaps captured
while Candidate B materializes its final corrected repayment grids, but only
JSON-safe scores and source references leave this module in the semantic audit.
It never changes a status value; it can only remove one narrowly defined false
review after independent visual and business-contract gates all pass.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

_LABELS = ("N", "*")
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
_DECISIVE_SEED_BASIS = "decisive_static_classifier"
_EXACT_ROW_REVIEW_SEED_BASIS = "exact_row_label_review"
_MIN_SEEDS_PER_LABEL = 6
_MIN_GRIDS_PER_LABEL = 3
_MIN_PAGES_PER_LABEL = 2
_MIN_DECISIVE_ANCHORS_PER_LABEL = 2
_MIN_DECISIVE_ANCHOR_GRIDS_PER_LABEL = 2
_MIN_DECISIVE_ANCHOR_PAGES_PER_LABEL = 2
_MAX_SEEDS_PER_GRID = 2
_MIN_LOO_SIMILARITY = 0.92
_MIN_SEED_CROSS_LABEL_MARGIN = 0.08
_MAX_WITHIN_LABEL_MAD = 0.05
_MIN_CANDIDATE_SIMILARITY = 0.94
_MIN_CANDIDATE_MARGIN = 0.10
_NON_BLOCKING_ISSUE_STATUSES = {"resolved", "suppressed_redundant", "informational"}
_ALLOWED_REVIEW_REASON = "zero_status_static_corroboration_unavailable"
_LIFECYCLE_FIELDS = {
    "account_status",
    "account_status_code",
    "account_lifecycle_state",
    "lifecycle_state",
    "close_date",
    "account_close_date",
}


def _record_values(record: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = record.get("normalized")
    if isinstance(normalized, Mapping):
        return normalized
    values = record.get("values")
    if isinstance(values, Mapping):
        return values
    return record


def _record_id(record: Mapping[str, Any]) -> str:
    values = _record_values(record)
    return str(record.get("repayment_id") or values.get("repayment_id") or record.get("record_id") or "").strip()


def _record_grid_id(record: Mapping[str, Any]) -> str:
    values = _record_values(record)
    explicit = str(record.get("grid_id") or values.get("grid_id") or "").strip()
    if explicit:
        return explicit
    for ref in record.get("source_cell_refs") or values.get("source_cell_refs") or ():
        if isinstance(ref, Mapping) and ref.get("grid_id"):
            return str(ref["grid_id"]).strip()
    repayment_id = _record_id(record)
    return repayment_id.rsplit(":", 1)[0] if ":" in repayment_id else ""


def _record_month(record: Mapping[str, Any]) -> str:
    values = _record_values(record)
    explicit = str(values.get("performance_month") or "").strip()
    if re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", explicit):
        return explicit
    try:
        year = int(values.get("year") or record.get("year") or 0)
        month = int(values.get("month") or record.get("month") or 0)
    except (TypeError, ValueError):
        return ""
    return f"{year:04d}-{month:02d}" if 2000 <= year <= 2099 and 1 <= month <= 12 else ""


def _record_status(record: Mapping[str, Any]) -> str:
    values = _record_values(record)
    return str(values.get("status_code") or values.get("status") or "").strip().upper()


def _record_amount(record: Mapping[str, Any]) -> Any:
    values = _record_values(record)
    if "status_amount" in values:
        return values.get("status_amount")
    return values.get("overdue_amount")


def _record_account_id(record: Mapping[str, Any]) -> str:
    values = _record_values(record)
    return str(values.get("account_id") or record.get("account_id") or "").strip()


def _explicit_zero(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    raw = str(value).strip().replace(",", "")
    if not raw or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw):
        return False
    try:
        return Decimal(raw) == 0
    except InvalidOperation:
        return False


def _source_ref_is_exact(ref: Any, *, grid_id: str, month: int) -> bool:
    if not isinstance(ref, Mapping):
        return False
    bbox = ref.get("bbox")
    try:
        ref_month = int(ref.get("col") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        str(ref.get("geometry_scope") or "") == "cell"
        and str(ref.get("grid_id") or "") == grid_id
        and ref_month == month
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
    )


def _record_contains_source_ref(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    values = _record_values(record)
    expected_bbox = expected.get("bbox")
    try:
        expected_locator = (
            int(expected.get("logical_page") or expected.get("page") or 0),
            str(expected.get("grid_id") or ""),
            int(expected.get("row") or 0),
            int(expected.get("col") or 0),
            tuple(round(float(value), 4) for value in expected_bbox),
        )
    except (TypeError, ValueError):
        return False
    for ref in record.get("source_cell_refs") or values.get("source_cell_refs") or ():
        if not isinstance(ref, Mapping):
            continue
        bbox = ref.get("bbox")
        try:
            locator = (
                int(ref.get("logical_page") or ref.get("page") or 0),
                str(ref.get("grid_id") or ""),
                int(ref.get("row") or 0),
                int(ref.get("col") or 0),
                tuple(round(float(value), 4) for value in bbox),
            )
        except (TypeError, ValueError):
            continue
        if locator == expected_locator:
            return True
    return False


def _normalize_template(template: Any) -> Any | None:
    """Return a grid-line-suppressed 32x32 binary bitmap."""

    try:
        import cv2
        import numpy as np

        array = np.asarray(template)
        if array.ndim != 2 or not array.size:
            return None
        if float(array.max(initial=0.0)) <= 1.0:
            ink = array.astype(np.float32) >= 0.5
        else:
            ink = array.astype(np.float32) < 128.0
        if not bool(ink.any()):
            return None

        # Full-cell ruling survives some crops as a nearly solid row/column.
        # Remove only truly line-like spans so the two vertical N stems remain.
        ink = ink.copy()
        ink[ink.mean(axis=1) >= 0.86, :] = False
        ink[:, ink.mean(axis=0) >= 0.86] = False
        ys, xs = np.where(ink)
        if not len(xs):
            return None
        compact = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        height, width = compact.shape
        if height < 5 or width < 4:
            return None
        density = float(compact.mean())
        if not 0.025 <= density <= 0.62:
            return None
        scale = min(22.0 / width, 22.0 / height)
        resized = cv2.resize(
            compact.astype(np.uint8),
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST,
        )
        canvas = np.zeros((32, 32), dtype=np.float32)
        y0 = (32 - resized.shape[0]) // 2
        x0 = (32 - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = (resized >= 0.5).astype(np.float32)
        return canvas
    except Exception:
        return None


def _feature_vector(template: Any) -> Any | None:
    try:
        import cv2
        import numpy as np

        normalized = _normalize_template(template)
        if normalized is None:
            return None
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0.7)
        vector = cv2.resize(blurred, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None
    except Exception:
        return None


def _medoid(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select and retain one observed normalized 32x32 bitmap medoid."""

    try:
        import numpy as np

        vectors = np.stack([item["vector"] for item in items])
        similarities = vectors @ vectors.T
        return items[int(np.argmax(similarities.mean(axis=1)))]
    except Exception:
        return None


def _median_absolute_deviation(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    deviations = sorted(abs(value - median) for value in ordered)
    midpoint = len(deviations) // 2
    return deviations[midpoint] if len(deviations) % 2 else (deviations[midpoint - 1] + deviations[midpoint]) / 2.0


def _index_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    ids: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    positions: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        record_id = _record_id(record)
        if record_id:
            ids[record_id].append(record)
        grid_id, month = _record_grid_id(record), _record_month(record)
        if grid_id and month:
            positions[(grid_id, month)].append(record)
    return (
        {key: values[0] for key, values in ids.items() if len(values) == 1},
        {key: values[0] for key, values in positions.items() if len(values) == 1},
    )


def _find_record(
    observation: Mapping[str, Any],
    index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    by_id, by_position = index
    record_id = str(observation.get("repayment_id") or "").strip()
    if record_id and record_id in by_id:
        return by_id[record_id]
    try:
        month = f"{int(observation.get('year') or 0):04d}-{int(observation.get('month') or 0):02d}"
    except (TypeError, ValueError):
        return None
    return by_position.get((str(observation.get("grid_id") or ""), month))


def _account_values(account: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = account.get("normalized")
    return normalized if isinstance(normalized, Mapping) else account


def _account_id(account: Mapping[str, Any]) -> str:
    values = _account_values(account)
    return str(values.get("account_id") or account.get("record_id") or "").strip()


def _parse_year_month(value: Any) -> str:
    raw = str(value or "").strip().replace("/", "-").replace(".", "-")
    match = re.match(r"^(20\d{2})-(\d{1,2})", raw)
    if match is None:
        return ""
    month = int(match.group(2))
    return f"{match.group(1)}-{month:02d}" if 1 <= month <= 12 else ""


def _lifecycle_blocks(
    record: Mapping[str, Any],
    accounts: Mapping[str, Mapping[str, Any]],
) -> bool:
    account_id = _record_account_id(record)
    account = accounts.get(account_id)
    if not account_id or account is None:
        return True
    values = _account_values(account)
    unresolved = {str(field) for field in account.get("_unresolved_fields") or values.get("_unresolved_fields") or ()}
    if unresolved & _LIFECYCLE_FIELDS:
        return True
    close_month = _parse_year_month(values.get("close_date") or values.get("account_close_date"))
    lifecycle_state = (
        str(
            values.get("account_lifecycle_state")
            or values.get("lifecycle_state")
            or values.get("account_status_code")
            or ""
        )
        .strip()
        .lower()
    )
    if lifecycle_state in {"unknown", "unresolved"}:
        return True
    if lifecycle_state in {"settled", "closed", "结清", "关闭"} and not close_month:
        return True
    performance_month = _record_month(record)
    if close_month and performance_month and performance_month >= close_month:
        return True
    return False


def _issue_is_active(issue: Mapping[str, Any]) -> bool:
    return str(issue.get("status") or "requires_review") not in _NON_BLOCKING_ISSUE_STATUSES


def _source_ref_row(ref: Mapping[str, Any]) -> int | None:
    if "row" not in ref or isinstance(ref.get("row"), bool):
        return None
    try:
        row = int(ref["row"])
    except (TypeError, ValueError):
        return None
    return row if row >= 0 else None


def _source_ref_month(ref: Mapping[str, Any]) -> int | None:
    if "col" not in ref or isinstance(ref.get("col"), bool):
        return None
    try:
        month = int(ref["col"])
    except (TypeError, ValueError):
        return None
    return month if 1 <= month <= 12 else None


def _field_family(field_name: Any) -> str:
    normalized = str(field_name or "").strip().lower()
    if normalized in {"status", "status_code"}:
        return "status"
    if normalized in {"overdue_amount", "status_amount"}:
        return "amount"
    return ""


def _record_source_rows(
    record: Mapping[str, Any],
    *,
    grid_id: str,
    month: int | None,
    field_family: str,
) -> set[int]:
    values = _record_values(record)
    refs = record.get("source_cell_refs") or values.get("source_cell_refs") or ()
    rows: set[int] = set()
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        if str(ref.get("grid_id") or "") != grid_id:
            continue
        ref_month = _source_ref_month(ref)
        if month is not None and ref_month is not None and ref_month != month:
            continue
        if field_family and _field_family(ref.get("field_name")) not in {
            "",
            field_family,
        }:
            continue
        row = _source_ref_row(ref)
        if row is not None:
            rows.add(row)
    return rows


def _issue_blocks_record(issue: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    if not _issue_is_active(issue):
        return False
    record_id = _record_id(record)
    grid_id = _record_grid_id(record)
    month = _record_month(record)
    account_id = _record_account_id(record)
    target_id = str(issue.get("target_record_id") or "").strip()
    issue_code = str(issue.get("issue_code") or "").lower()
    target_dataset = str(issue.get("target_dataset") or "")
    field_name = str(issue.get("field_name") or "").lower()
    lifecycle_or_linkage = any(
        marker in issue_code
        for marker in (
            "link",
            "owner",
            "terminal",
            "lifecycle",
            "duplicate_conflict",
            "account_sequence_gap",
            "anchor_grid_missing",
        )
    )
    if target_id and target_id in {record_id, f"{grid_id}:{month}"}:
        return True
    if target_id and target_id in {account_id, grid_id} and lifecycle_or_linkage:
        return True
    for ref in issue.get("source_refs") or ():
        if not isinstance(ref, Mapping) or str(ref.get("grid_id") or "") != grid_id:
            continue
        record_month = int(month[-2:]) if month else None
        ref_month = _source_ref_month(ref)
        if record_month is not None and ref_month is not None and ref_month != record_month:
            continue
        ref_row = _source_ref_row(ref)
        if ref_row is None:
            return True
        record_rows = _record_source_rows(
            record,
            grid_id=grid_id,
            month=record_month,
            field_family=_field_family(ref.get("field_name") or field_name),
        )
        if not record_rows or ref_row in record_rows:
            return True
    return bool(
        target_dataset == "repayment_records"
        and target_id == record_id
        and field_name in {"status", "status_code", "overdue_amount", "status_amount", "performance_month"}
    )


def _review_reason(record: Mapping[str, Any]) -> str:
    audit = record.get("audit")
    if isinstance(audit, Mapping):
        return str(audit.get("reason") or "")
    return ""


def _review_can_be_cleared(record: Mapping[str, Any]) -> bool:
    if str(record.get("extraction_status") or "") != "review":
        return False
    if _review_reason(record) != _ALLOWED_REVIEW_REASON:
        return False
    if record.get("_amount_pairing"):
        return False
    unresolved = {str(field) for field in record.get("_unresolved_fields") or ()}
    return not unresolved or unresolved <= {"status", "status_code"}


def _plane_conflict(
    observation: Mapping[str, Any],
    expected_status: str,
    native_index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
    corrected_index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
) -> bool:
    # OCR sentinels such as ``unknown`` and ``unresolved`` describe the
    # absence of a substantive observation, not a competing repayment code.
    # Restrict this gate to the closed PBOC status alphabet so a missing plane
    # cannot veto otherwise exact document-local corroboration.
    statuses = [
        status
        for record in (
            _find_record(observation, native_index),
            _find_record(observation, corrected_index),
        )
        if record is not None
        and (status := _record_status(record)) in _CANONICAL_REPAYMENT_STATUS_VALUES
    ]
    return bool(any(status != expected_status for status in statuses) or len(set(statuses)) > 1)


def _exact_row_review_seed_label(
    observation: Mapping[str, Any],
    record: Mapping[str, Any],
) -> str:
    """Return a review-row label only when all row-bound labels agree."""

    if not _review_can_be_cleared(record):
        return ""
    labels = (
        _record_status(record),
        str(observation.get("observed_status") or "").strip().upper(),
        str(observation.get("resolved_status") or "").strip().upper(),
    )
    if len(set(labels)) != 1 or labels[0] not in _LABELS:
        return ""
    decisive_label = str(observation.get("decisive_label") or "").strip().upper()
    if decisive_label in _LABELS and decisive_label != labels[0]:
        return ""
    return labels[0]


def _observation_source_key(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    ref = observation.get("source_ref")
    bbox = tuple(ref.get("bbox") or ()) if isinstance(ref, Mapping) else ()
    return (
        int(observation.get("page") or 0),
        str(observation.get("grid_id") or ""),
        int(observation.get("year") or 0),
        int(observation.get("month") or 0),
        bbox,
    )


def _prepare_observations(
    observations: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    distinct: dict[tuple[Any, ...], dict[str, Any]] = {}
    rejection_counts: Counter[str] = Counter()
    input_count = 0
    for observation in observations:
        input_count += 1
        source_key = _observation_source_key(observation)
        if not source_key[1]:
            rejection_counts["missing_grid_id"] += 1
            continue
        if source_key in distinct:
            rejection_counts["duplicate_source_key"] += 1
            continue
        bitmap = _normalize_template(observation.get("template"))
        vector = _feature_vector(bitmap)
        if bitmap is None or vector is None:
            rejection_counts["glyph_normalization_failed"] += 1
            continue
        distinct[source_key] = {
            **dict(observation),
            "bitmap": bitmap,
            "vector": vector,
        }
    prepared = list(distinct.values())
    return prepared, {
        "input_observation_count": input_count,
        "prepared_observation_count": len(prepared),
        "rejected_observation_count": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def _business_contract_rejection_reasons(
    observation: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    label: str,
    accounts: Mapping[str, Mapping[str, Any]],
    issues: Sequence[Mapping[str, Any]],
    native_index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
    corrected_index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
) -> tuple[str, ...]:
    try:
        month = int(observation.get("month") or 0)
    except (TypeError, ValueError):
        return ("invalid_observation_month",)
    reasons: list[str] = []
    if label not in _LABELS:
        reasons.append("unsupported_label")
    if _record_status(record) != label:
        reasons.append("record_status_mismatch")
    if str(observation.get("observed_status") or "").upper() != label:
        reasons.append("observed_status_mismatch")
    if str(observation.get("resolved_status") or "").upper() != label:
        reasons.append("resolved_status_mismatch")
    if observation.get("alignment_exact") is not True:
        reasons.append("alignment_not_exact")
    if observation.get("exact_status_geometry") is not True:
        reasons.append("status_geometry_not_exact")
    if observation.get("geometry_reused_across_years"):
        reasons.append("geometry_reused_across_years")
    if observation.get("classifier_conflict"):
        reasons.append("classifier_conflict")
    decisive_label = str(observation.get("decisive_label") or "").strip().upper()
    if decisive_label in _LABELS and decisive_label != label:
        reasons.append("decisive_label_conflict")
    if observation.get("amount_pair_exact") is not True:
        reasons.append("amount_pair_not_exact")
    if observation.get("status_amount_conflict"):
        reasons.append("status_amount_conflict")
    if not _explicit_zero(observation.get("amount")):
        reasons.append("observation_amount_not_explicit_zero")
    if not _explicit_zero(_record_amount(record)):
        reasons.append("record_amount_not_explicit_zero")
    if not _source_ref_is_exact(
        observation.get("source_ref"),
        grid_id=str(observation.get("grid_id") or ""),
        month=month,
    ):
        reasons.append("source_ref_not_exact")
    if not _record_contains_source_ref(record, observation.get("source_ref") or {}):
        reasons.append("record_source_ref_not_bound")
    if _plane_conflict(
        observation,
        label,
        native_index,
        corrected_index,
    ):
        reasons.append("independent_plane_conflict")
    if _lifecycle_blocks(record, accounts):
        reasons.append("lifecycle_block")
    if any(_issue_blocks_record(issue, record) for issue in issues):
        reasons.append("active_issue_block")
    return tuple(reasons)


def _select_seeds(
    prepared: Sequence[Mapping[str, Any]],
    record_index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
    *,
    accounts: Mapping[str, Mapping[str, Any]],
    issues: Sequence[Mapping[str, Any]],
    native_index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
    corrected_index: tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    eligible: dict[str, list[dict[str, Any]]] = {label: [] for label in _LABELS}
    rejection_counts: Counter[str] = Counter()
    label_rejections: dict[str, Counter[str]] = {label: Counter() for label in _LABELS}
    decisive_counts: Counter[str] = Counter()
    exact_row_review_counts: Counter[str] = Counter()
    eligible_basis_counts: Counter[str] = Counter()
    eligible_label_basis_counts: dict[str, Counter[str]] = {
        label: Counter() for label in _LABELS
    }
    for observation in prepared:
        record = _find_record(observation, record_index)
        decisive_label = str(observation.get("decisive_label") or "").upper()
        if decisive_label in _LABELS:
            decisive_counts[decisive_label] += 1
        row_review_label = (
            _exact_row_review_seed_label(observation, record)
            if record is not None
            else ""
        )
        if row_review_label:
            exact_row_review_counts[row_review_label] += 1

        label = decisive_label
        seed_basis = _DECISIVE_SEED_BASIS
        reason = ""
        if record is None:
            reason = "record_not_uniquely_bound"
        elif row_review_label:
            # A reviewed row may seed the bank only through this exact-label
            # lane.  It still passes every geometry, amount, plane, lifecycle,
            # and active-issue gate below, and every selected seed participates
            # in whole-bank coherence validation.
            label = row_review_label
            seed_basis = _EXACT_ROW_REVIEW_SEED_BASIS
            contract_rejections = _business_contract_rejection_reasons(
                observation,
                record,
                label=label,
                accounts=accounts,
                issues=issues,
                native_index=native_index,
                corrected_index=corrected_index,
            )
            if contract_rejections:
                reason = contract_rejections[0]
        elif decisive_label not in _LABELS:
            reason = "missing_or_unsupported_decisive_label"
        else:
            try:
                confidence = float(observation.get("decisive_confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.95:
                reason = "decisive_confidence_below_threshold"
            elif str(record.get("extraction_status") or "") == "review":
                reason = "record_already_requires_review"
            elif bool(record.get("_unresolved_fields")):
                reason = "record_has_unresolved_fields"
            else:
                contract_rejections = _business_contract_rejection_reasons(
                    observation,
                    record,
                    label=label,
                    accounts=accounts,
                    issues=issues,
                    native_index=native_index,
                    corrected_index=corrected_index,
                )
                if contract_rejections:
                    reason = contract_rejections[0]
        if reason:
            rejection_counts[reason] += 1
            if label in _LABELS:
                label_rejections[label][reason] += 1
            continue
        seed = dict(observation)
        seed["seed_basis"] = seed_basis
        eligible[label].append(seed)
        eligible_basis_counts[seed_basis] += 1
        eligible_label_basis_counts[label][seed_basis] += 1

    selected: dict[str, list[dict[str, Any]]] = {label: [] for label in _LABELS}
    eligible_counts: dict[str, int] = {}
    for label in _LABELS:
        observations = eligible[label]
        eligible_counts[label] = len(observations)
        by_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            by_grid[str(observation.get("grid_id") or "")].append(observation)
        for grid_id in sorted(by_grid):
            ranked = sorted(
                by_grid[grid_id],
                key=lambda item: (
                    0
                    if str(item.get("seed_basis") or "")
                    == _DECISIVE_SEED_BASIS
                    else 1,
                    _observation_source_key(item),
                ),
            )
            selected[label].extend(ranked[:_MAX_SEEDS_PER_GRID])
            capped_count = max(0, len(ranked) - _MAX_SEEDS_PER_GRID)
            if capped_count:
                rejection_counts["per_grid_seed_cap"] += capped_count
                label_rejections[label]["per_grid_seed_cap"] += capped_count

    selected_count = sum(len(selected[label]) for label in _LABELS)
    selected_basis_counts = Counter(
        str(seed.get("seed_basis") or "")
        for label in _LABELS
        for seed in selected[label]
    )
    selection_audit = {
        "prepared_observation_count": len(prepared),
        "eligible_seed_count_before_grid_cap": sum(eligible_counts.values()),
        "selected_seed_count": selected_count,
        "rejected_observation_count": len(prepared) - selected_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "eligible_seed_basis_counts": dict(sorted(eligible_basis_counts.items())),
        "selected_seed_basis_counts": dict(sorted(selected_basis_counts.items())),
        "labels": {
            label: {
                "decisive_observation_count": decisive_counts[label],
                "exact_row_review_observation_count": exact_row_review_counts[label],
                "eligible_seed_count_before_grid_cap": eligible_counts[label],
                "selected_seed_count": len(selected[label]),
                "eligible_seed_basis_counts": dict(
                    sorted(eligible_label_basis_counts[label].items())
                ),
                "selected_seed_basis_counts": dict(
                    sorted(
                        Counter(
                            str(seed.get("seed_basis") or "")
                            for seed in selected[label]
                        ).items()
                    )
                ),
                "rejection_counts": dict(sorted(label_rejections[label].items())),
            }
            for label in _LABELS
        },
    }
    return selected, selection_audit


def _build_bank(seeds: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    audit: dict[str, Any] = {
        "enabled": False,
        "labels": {},
        "validation_mode": "all_selected_seeds_must_pass",
    }
    insufficient_labels: list[str] = []
    insufficient_anchor_labels: list[str] = []
    for label in _LABELS:
        values = list(seeds.get(label) or ())
        grids = {str(item.get("grid_id") or "") for item in values}
        pages = {int(item.get("page") or 0) for item in values}
        decisive_anchors = [
            seed
            for seed in values
            if str(seed.get("seed_basis") or "") == _DECISIVE_SEED_BASIS
        ]
        anchor_grids = {
            str(item.get("grid_id") or "") for item in decisive_anchors
        }
        anchor_pages = {int(item.get("page") or 0) for item in decisive_anchors}
        audit["labels"][label] = {
            "seed_count": len(values),
            "grid_count": len(grids),
            "page_count": len(pages),
            "seed_basis_counts": dict(
                sorted(
                    Counter(str(seed.get("seed_basis") or "") for seed in values).items()
                )
            ),
            "seed_record_ids": [_record_id(seed) for seed in values],
            "seed_source_refs": [
                deepcopy(seed.get("source_ref")) for seed in values if isinstance(seed.get("source_ref"), Mapping)
            ],
            "decisive_anchor_count": len(decisive_anchors),
            "decisive_anchor_grid_count": len(anchor_grids),
            "decisive_anchor_page_count": len(anchor_pages),
            "decisive_anchor_record_ids": [
                _record_id(seed) for seed in decisive_anchors
            ],
            "decisive_anchor_source_refs": [
                deepcopy(seed.get("source_ref"))
                for seed in decisive_anchors
                if isinstance(seed.get("source_ref"), Mapping)
            ],
        }
        if len(values) < _MIN_SEEDS_PER_LABEL or len(grids) < _MIN_GRIDS_PER_LABEL or len(pages) < _MIN_PAGES_PER_LABEL:
            insufficient_labels.append(label)
        if (
            len(decisive_anchors) < _MIN_DECISIVE_ANCHORS_PER_LABEL
            or len(anchor_grids) < _MIN_DECISIVE_ANCHOR_GRIDS_PER_LABEL
            or len(anchor_pages) < _MIN_DECISIVE_ANCHOR_PAGES_PER_LABEL
        ):
            insufficient_anchor_labels.append(label)
    if insufficient_labels:
        audit["disabled_reason"] = "insufficient_cross_grid_page_seed_coverage"
        audit["insufficient_labels"] = insufficient_labels
        return {}, audit
    if insufficient_anchor_labels:
        audit["disabled_reason"] = (
            "insufficient_independent_decisive_anchor_coverage"
        )
        audit["insufficient_anchor_labels"] = insufficient_anchor_labels
        return {}, audit

    prototypes = {label: _medoid(list(seeds[label])) for label in _LABELS}
    failed_prototypes = [label for label in _LABELS if prototypes.get(label) is None]
    if failed_prototypes:
        audit["disabled_reason"] = "prototype_construction_failed"
        audit["prototype_failed_labels"] = failed_prototypes
        return {}, audit

    unavailable_labels: list[str] = []
    validation_failures: dict[str, list[str]] = {label: [] for label in _LABELS}
    for label in _LABELS:
        other_label = "*" if label == "N" else "N"
        same_scores: list[float] = []
        loo_scores: list[float] = []
        seed_margins: list[float] = []
        for seed in seeds[label]:
            other_grids = [item for item in seeds[label] if item.get("grid_id") != seed.get("grid_id")]
            loo_prototype = _medoid(other_grids)
            cross_grid_values = [item for item in seeds[other_label] if item.get("grid_id") != seed.get("grid_id")]
            cross_prototype = _medoid(cross_grid_values)
            if loo_prototype is None or cross_prototype is None:
                unavailable_labels.append(label)
                break
            loo_score = float(seed["vector"] @ loo_prototype["vector"])
            same_score = float(seed["vector"] @ prototypes[label]["vector"])
            cross_score = float(seed["vector"] @ cross_prototype["vector"])
            loo_scores.append(loo_score)
            same_scores.append(same_score)
            seed_margins.append(loo_score - cross_score)
        if label in unavailable_labels:
            continue
        mad = _median_absolute_deviation(same_scores)
        label_audit = audit["labels"][label]
        label_audit.update(
            {
                "minimum_leave_one_grid_out_similarity": round(min(loo_scores), 4),
                "minimum_cross_label_margin": round(min(seed_margins), 4),
                "within_label_mad": round(mad, 4),
            }
        )
        if min(loo_scores) < _MIN_LOO_SIMILARITY:
            validation_failures[label].append("leave_one_grid_out_similarity_below_threshold")
        if min(seed_margins) < _MIN_SEED_CROSS_LABEL_MARGIN:
            validation_failures[label].append("cross_label_seed_margin_below_threshold")
        if mad > _MAX_WITHIN_LABEL_MAD:
            validation_failures[label].append("within_label_variation_above_threshold")

    if unavailable_labels:
        audit["disabled_reason"] = "leave_one_grid_out_prototype_unavailable"
        audit["prototype_unavailable_labels"] = unavailable_labels
        return {}, audit
    active_failures = {label: failures for label, failures in validation_failures.items() if failures}
    if active_failures:
        audit["validation_failures"] = active_failures
        for reason in (
            "leave_one_grid_out_similarity_below_threshold",
            "cross_label_seed_margin_below_threshold",
            "within_label_variation_above_threshold",
        ):
            if any(reason in failures for failures in active_failures.values()):
                audit["disabled_reason"] = reason
                break
        return {}, audit

    audit["enabled"] = True
    audit["disabled_reason"] = None
    return {**prototypes, "seeds": seeds}, audit


def _candidate_prototypes(
    bank: Mapping[str, Any],
    *,
    grid_id: str,
) -> tuple[Any, Any] | None:
    seeds = bank.get("seeds")
    if not isinstance(seeds, Mapping):
        return None
    prototypes: dict[str, Any] = {}
    for label in _LABELS:
        values = [item for item in seeds.get(label) or () if str(item.get("grid_id") or "") != grid_id]
        if (
            len({str(item.get("grid_id") or "") for item in values}) < 2
            or len({int(item.get("page") or 0) for item in values}) < 2
        ):
            return None
        prototype = _medoid(values)
        if prototype is None:
            return None
        prototypes[label] = prototype
    return prototypes["N"], prototypes["*"]


def _clear_static_review(record: MutableMapping[str, Any]) -> None:
    record.pop("extraction_status", None)
    record.pop("audit", None)
    record.pop("recognition_source", None)
    unresolved = [
        field for field in record.get("_unresolved_fields") or () if str(field) not in {"status", "status_code"}
    ]
    if unresolved:
        record["_unresolved_fields"] = unresolved
    else:
        record.pop("_unresolved_fields", None)


def apply_document_local_status_glyph_bank(
    records: list[dict[str, Any]],
    observations: Iterable[Mapping[str, Any]],
    *,
    accounts: Iterable[Mapping[str, Any]],
    issues: Iterable[Mapping[str, Any]],
    native_plane_records: Iterable[Mapping[str, Any]],
    corrected_plane_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Clear only false N/asterisk reviews proven by document-local glyphs."""

    prepared, preparation_audit = _prepare_observations(observations)
    record_index = _index_records(records)
    native_index = _index_records(native_plane_records)
    corrected_index = _index_records(corrected_plane_records)
    accounts_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for account in accounts:
        account_id = _account_id(account)
        if account_id:
            accounts_by_id[account_id].append(account)
    account_index = {account_id: matches[0] for account_id, matches in accounts_by_id.items() if len(matches) == 1}
    active_issues = [issue for issue in issues if _issue_is_active(issue)]
    seeds, seed_selection_audit = _select_seeds(
        prepared,
        record_index,
        accounts=account_index,
        issues=active_issues,
        native_index=native_index,
        corrected_index=corrected_index,
    )
    bank, audit = _build_bank(seeds)
    audit.update(
        {
            "observation_count": len(prepared),
            "observation_preparation": preparation_audit,
            "seed_selection": seed_selection_audit,
            "promoted_count": 0,
            "promotions": [],
            "veto_counts": {},
            "thresholds": {
                "minimum_seeds_per_label": _MIN_SEEDS_PER_LABEL,
                "minimum_grids_per_label": _MIN_GRIDS_PER_LABEL,
                "minimum_pages_per_label": _MIN_PAGES_PER_LABEL,
                "minimum_decisive_anchors_per_label": (
                    _MIN_DECISIVE_ANCHORS_PER_LABEL
                ),
                "minimum_decisive_anchor_grids_per_label": (
                    _MIN_DECISIVE_ANCHOR_GRIDS_PER_LABEL
                ),
                "minimum_decisive_anchor_pages_per_label": (
                    _MIN_DECISIVE_ANCHOR_PAGES_PER_LABEL
                ),
                "maximum_seeds_per_grid": _MAX_SEEDS_PER_GRID,
                "leave_one_grid_out_similarity": _MIN_LOO_SIMILARITY,
                "seed_cross_label_margin": _MIN_SEED_CROSS_LABEL_MARGIN,
                "within_label_mad": _MAX_WITHIN_LABEL_MAD,
                "candidate_similarity": _MIN_CANDIDATE_SIMILARITY,
                "candidate_margin": _MIN_CANDIDATE_MARGIN,
            },
        }
    )
    if not bank:
        return audit

    vetoes: Counter[str] = Counter()
    observations_by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in prepared:
        record = _find_record(observation, record_index)
        if record is not None:
            observations_by_record[_record_id(record)].append(observation)

    for record in records:
        label = _record_status(record)
        if label not in _LABELS or not _review_can_be_cleared(record):
            continue
        candidates = observations_by_record.get(_record_id(record), [])
        if len(candidates) != 1:
            vetoes["non_unique_source_bound_observation"] += 1
            continue
        observation = candidates[0]
        contract_rejections = _business_contract_rejection_reasons(
            observation,
            record,
            label=label,
            accounts=account_index,
            issues=active_issues,
            native_index=native_index,
            corrected_index=corrected_index,
        )
        if contract_rejections:
            vetoes[f"business_contract:{contract_rejections[0]}"] += 1
            continue
        candidate_prototypes = _candidate_prototypes(
            bank,
            grid_id=str(observation.get("grid_id") or ""),
        )
        if candidate_prototypes is None:
            vetoes["cross_grid_page_prototype_unavailable"] += 1
            continue
        n_prototype, star_prototype = candidate_prototypes
        same_prototype = n_prototype if label == "N" else star_prototype
        other_prototype = star_prototype if label == "N" else n_prototype
        similarity = float(observation["vector"] @ same_prototype["vector"])
        margin = similarity - float(observation["vector"] @ other_prototype["vector"])
        if similarity < _MIN_CANDIDATE_SIMILARITY:
            vetoes["candidate_similarity_below_threshold"] += 1
            continue
        if margin < _MIN_CANDIDATE_MARGIN:
            vetoes["candidate_cross_label_margin_below_threshold"] += 1
            continue
        _clear_static_review(record)
        audit["promotions"].append(
            {
                "repayment_id": _record_id(record),
                "status_code": label,
                "similarity": round(similarity, 4),
                "cross_label_margin": round(margin, 4),
                "source_refs": [deepcopy(observation.get("source_ref"))],
            }
        )

    audit["promoted_count"] = len(audit["promotions"])
    audit["veto_counts"] = dict(sorted(vetoes.items()))
    return audit


__all__ = ["apply_document_local_status_glyph_bank"]
