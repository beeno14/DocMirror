# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, plugin-owned extraction diagnostics for scanned detailed reports.

The report is deliberately non-fatal: uncertain source values remain available
for human correction and decoding continues.  This module never mutates sealed
OCR evidence or applies a guessed replacement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

ISSUE_DATASET = "personal_detail_extraction_issues"

ISSUE_CATEGORIES = frozenset(
    {
        "ocr_structure_correction",
        "page_continuation",
        "schema_incompleteness",
        "ocr_cell_level_error",
    }
)

_NON_DEGRADING_STATUSES = frozenset({"resolved", "suppressed_redundant", "informational"})
_ROW_BLOCKING_ISSUE_CODES = frozenset(
    {
        "recognized_native_section_missing_required_value",
    }
)
_PLACEHOLDERS = frozenset({"", "-", "--", "---", "未报告", "不详"})
_LIABILITY_FIELDS = frozenset(
    {
        "responsibility_type",
        "responsible_person_type",
        "liability_type",
        "guarantee_contract_number",
        "account_identifier",
        "management_institution",
        "institution",
        "responsibility_amount",
        "guarantee_amount",
        "balance",
        "effective_date",
        "expiration_date",
        "data_provider",
        "责任人类型",
        "保证合同编号",
        "管理机构",
        "责任金额",
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return str(value)


def _issue_id(payload: Mapping[str, Any]) -> str:
    identity = {
        key: _plain(payload.get(key))
        for key in (
            "category",
            "issue_code",
            "target_dataset",
            "target_record_id",
            "field_name",
            "observed_value",
        )
    }
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"personal_detail_extraction_issue:{digest}"


def make_issue(
    *,
    category: str,
    issue_code: str,
    message: str,
    severity: str = "warning",
    status: str = "requires_review",
    parser_stage: str | None = None,
    target_dataset: str | None = None,
    target_record_id: str | None = None,
    field_name: str | None = None,
    observed_value: Any | None = None,
    candidate_value: Any | None = None,
    confidence: float | None = None,
    source_refs: Iterable[Mapping[str, Any]] = (),
    reason_codes: Iterable[str] = (),
) -> dict[str, Any]:
    """Build one stable issue row without inventing source evidence."""
    if category not in ISSUE_CATEGORIES:
        raise ValueError(f"unsupported personal-detail issue category: {category}")
    row: dict[str, Any] = {
        "category": category,
        "issue_code": str(issue_code),
        "severity": str(severity),
        "status": str(status),
        "message": str(message),
    }
    optional = {
        "parser_stage": parser_stage,
        "target_dataset": target_dataset,
        "target_record_id": target_record_id,
        "field_name": field_name,
        "observed_value": _plain(observed_value),
        "candidate_value": _plain(candidate_value),
    }
    row.update({key: value for key, value in optional.items() if value not in (None, "")})
    if confidence is not None:
        row["confidence"] = max(0.0, min(1.0, float(confidence)))
    reasons = tuple(dict.fromkeys(str(value) for value in reason_codes if value))
    if reasons:
        row["reason_codes"] = list(reasons)
    refs = [deepcopy(dict(ref)) for ref in source_refs if isinstance(ref, Mapping)]
    if refs:
        row["source_refs"] = refs
    row["extraction_issue_id"] = _issue_id(row)
    row["record_id"] = row["extraction_issue_id"]
    return row


def record_issue(context: Any, issue: Mapping[str, Any]) -> None:
    """Attach a diagnostic to the plugin context for later publication."""
    rows = getattr(context, "_personal_detail_extraction_issues", None)
    if not isinstance(rows, list):
        rows = []
        setattr(context, "_personal_detail_extraction_issues", rows)
    issue_id = str(issue.get("extraction_issue_id") or "")
    if issue_id and any(str(row.get("extraction_issue_id") or "") == issue_id for row in rows):
        return
    rows.append(deepcopy(dict(issue)))


def liability_record_is_substantive(record: Mapping[str, Any]) -> bool:
    """Reject identifier-only compatibility rows, not valid empty-valued cells."""
    values = record.get("normalized")
    if not isinstance(values, Mapping):
        values = record
    nested_values = values.get("values") if isinstance(values, Mapping) else None
    candidates: list[Mapping[str, Any]] = [values]
    if isinstance(nested_values, Mapping):
        candidates.append(nested_values)
    for candidate in candidates:
        for key in _LIABILITY_FIELDS:
            value = candidate.get(key)
            if value is not None and str(value).strip() not in _PLACEHOLDERS:
                return True
    return False


def _dataset_from_path(path: str) -> str | None:
    for name in (
        "credit_accounts",
        "credit_lines",
        "repayment_liability_records",
        "repayment_records",
        "inquiry_records",
        "personal_report_metadata",
        "personal_profile",
        "residence_records",
        "employment_records",
        "postpaid_records",
    ):
        if name in path:
            return name
    return None


def _ocr_audit_issues(context: Any) -> list[dict[str, Any]]:
    loader = getattr(context, "ocr_correction_audit", None)
    audit = loader() if callable(loader) else {}
    result: list[dict[str, Any]] = []
    for anomaly in audit.get("cell_anomalies") or []:
        if not isinstance(anomaly, Mapping):
            continue
        path = str(anomaly.get("path") or "")
        withheld = bool(anomaly.get("normalized_value_withheld"))
        result.append(
            make_issue(
                category="ocr_cell_level_error",
                issue_code="pboc_cell_contract_unresolved",
                message=(
                    "The OCR value does not satisfy its PBOC field contract; the normalized value was withheld "
                    "and the raw observation was retained for review."
                    if withheld
                    else "The OCR value does not satisfy its PBOC field contract; the value was preserved for review."
                ),
                parser_stage=str(anomaly.get("stage") or "ocr_correction"),
                target_dataset=str(anomaly.get("dataset_name") or "") or _dataset_from_path(path),
                target_record_id=str(anomaly.get("record_id") or "") or None,
                field_name=str(anomaly.get("field_name") or anomaly.get("role") or "") or None,
                observed_value=anomaly.get("value"),
                source_refs=anomaly.get("source_refs") or (),
                reason_codes=anomaly.get("reason_codes") or (),
            )
        )
    for request in audit.get("page_reocr_failures") or []:
        if not isinstance(request, Mapping):
            continue
        result.append(
            make_issue(
                category="ocr_structure_correction",
                issue_code=f"page_reocr_{request.get('status') or 'failed'}",
                message="The page's single re-OCR attempt could not produce usable evidence; decoding continued without retry.",
                parser_stage="one_shot_page_reocr",
                observed_value=dict(request),
                reason_codes=("one_shot_terminal_result", "fallback_nonfatal", "human_review_available"),
            )
        )
    return result


def _topology_issues(context: Any) -> list[dict[str, Any]]:
    loader = getattr(context, "page_topology_audit", None)
    audit = loader() if callable(loader) else {}
    result: list[dict[str, Any]] = []
    for item in audit.get("issues") or []:
        if isinstance(item, Mapping):
            code = str(item.get("code") or item.get("issue_code") or "page_topology_uncertain")
            message = str(item.get("message") or "Logical-page topology is uncertain.")
            refs = item.get("source_refs") or ()
        else:
            code = str(item or "page_topology_uncertain")
            message = "Logical-page topology is uncertain."
            refs = ()
        category = (
            "page_continuation"
            if any(marker in code.lower() for marker in ("page", "spread", "footer", "continu", "pair"))
            else "ocr_structure_correction"
        )
        result.append(
            make_issue(
                category=category,
                issue_code=code,
                message=message,
                parser_stage="page_topology",
                source_refs=refs,
                reason_codes=("preserved_observed_pages", "no_page_position_invented"),
            )
        )
    for recovery in audit.get("static_split_recoveries") or []:
        if not isinstance(recovery, Mapping):
            continue
        result.append(
            make_issue(
                category="page_continuation",
                issue_code="static_split_subpage_constructed",
                message=(
                    "The static validator constructed a logical subpage from exact split geometry without OCR "
                    "or mutation of the sealed ParseResult."
                ),
                severity="info",
                status="resolved",
                parser_stage="static_topology_construction",
                observed_value=dict(recovery),
                reason_codes=(
                    "core_potential_split_signal",
                    "static_split_confirmed",
                    "exact_coordinate_transform",
                ),
            )
        )
    return result


def collect_extraction_issues(context: Any) -> list[dict[str, Any]]:
    """Return deduplicated plugin diagnostics accumulated by every stage."""
    rows = [
        *[dict(row) for row in getattr(context, "_personal_detail_extraction_issues", []) if isinstance(row, Mapping)],
        *_ocr_audit_issues(context),
        *_topology_issues(context),
    ]
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        issue_id = str(row.get("extraction_issue_id") or _issue_id(row))
        row["extraction_issue_id"] = issue_id
        row.setdefault("record_id", issue_id)
        existing = unique.get(issue_id)
        if existing is None:
            unique[issue_id] = row
            continue
        refs = [
            *[dict(value) for value in existing.get("source_refs") or () if isinstance(value, Mapping)],
            *[dict(value) for value in row.get("source_refs") or () if isinstance(value, Mapping)],
        ]
        if refs:
            seen: set[str] = set()
            existing["source_refs"] = [
                value
                for value in refs
                if not (
                    (marker := json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)) in seen
                    or seen.add(marker)
                )
            ]
        reasons = tuple(
            dict.fromkeys(
                str(value)
                for value in (*existing.get("reason_codes", ()), *row.get("reason_codes", ()))
                if value
            )
        )
        if reasons:
            existing["reason_codes"] = list(reasons)
        if existing.get("status") in _NON_DEGRADING_STATUSES and row.get("status") not in _NON_DEGRADING_STATUSES:
            existing["status"] = row["status"]
            existing["severity"] = row.get("severity", existing.get("severity"))
    return list(unique.values())


def dataset_states_from_issues(
    issues: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Project unresolved diagnostics into the existing absence ledger."""
    states: dict[str, dict[str, Any]] = {}
    for issue in issues:
        dataset_name = str(issue.get("target_dataset") or "")
        issue_code = str(issue.get("issue_code") or "")
        if not dataset_name or str(issue.get("status") or "") in _NON_DEGRADING_STATUSES:
            continue
        reason_codes = {str(value) for value in issue.get("reason_codes") or ()}
        row_blocking = issue_code in _ROW_BLOCKING_ISSUE_CODES
        value_withheld = "normalized_value_withheld" in reason_codes
        if not row_blocking and not value_withheld:
            continue
        current = states.get(dataset_name)
        status = "extraction_failed" if row_blocking else "partial"
        if current and current.get("presence_status") == "extraction_failed":
            continue
        states[dataset_name] = {
            "presence_status": status,
            "reason": issue_code,
            "confidence": issue.get("confidence"),
            "source_refs": deepcopy(issue.get("source_refs") or []),
        }
    return states


__all__ = [
    "ISSUE_CATEGORIES",
    "ISSUE_DATASET",
    "collect_extraction_issues",
    "dataset_states_from_issues",
    "liability_record_is_substantive",
    "make_issue",
    "record_issue",
]
