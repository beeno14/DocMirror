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
_LIABILITY_ISSUE_FIELD_ALIASES = {
    "管理机构": "institution",
    "业务种类": "business_type",
    "开立日期": "open_date",
    "到期日期": "due_date",
    "责任人类型": "responsibility_type",
    "还款责任金额": "responsibility_amount",
    "币种": "currency",
    "保证合同编号": "contract_number",
    "主业务借款人": "related_party_name",
    "主业务借款人证件类型": "related_party_id_type",
    "主业务借款人证件号码": "related_party_id_number",
    "余额": "balance",
    "五级分类": "five_tier_class",
    "逾期月数": "overdue_months",
    "还款状态": "repayment_status_code",
    "reporting_amount_currency": "currency",
    "status_code": "repayment_status_code",
}
_PACKED_LIABILITY_FIELDS = frozenset(
    {
        "institution",
        "business_type",
        "open_date",
        "due_date",
        "responsibility_type",
        "responsibility_amount",
        "currency",
        "contract_number",
    }
)
_LIABILITY_UNRESOLVED_VALUES = frozenset(
    {"", "-", "--", "---", "未报告", "不详", "未知", "unknown", "unreadable"}
)
_FINAL_LIABILITY_ISSUE_RECORDS_ATTR = "_personal_detail_final_liability_issue_records"


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


def retarget_issue_record(
    issue: Mapping[str, Any],
    target_record_id: Any,
) -> dict[str, Any]:
    """Return one issue linked to a stable final record identity."""

    row = deepcopy(dict(issue))
    target = str(target_record_id or "").strip()
    if target:
        row["target_record_id"] = target
    else:
        row.pop("target_record_id", None)
    issue_id = _issue_id(row)
    row["extraction_issue_id"] = issue_id
    row["record_id"] = issue_id
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


def register_issue_target_remap(context: Any, source_record_id: Any, target_record_id: Any) -> None:
    """Register a plugin-local identity change for later diagnostic linkage."""
    source = str(source_record_id or "").strip()
    target = str(target_record_id or "").strip()
    if not source or not target or source == target:
        return
    registry = getattr(context, "_personal_detail_issue_target_remaps", None)
    if not isinstance(registry, dict):
        registry = {}
        setattr(context, "_personal_detail_issue_target_remaps", registry)
    targets = registry.get(source)
    if not isinstance(targets, set):
        targets = {str(value) for value in targets or () if value} if targets else set()
        registry[source] = targets
    targets.add(target)


def _resolve_issue_target(
    registry: Mapping[str, Any], source_record_id: str
) -> tuple[str | None, bool]:
    """Return one terminal remap, or mark a branching/cyclic remap ambiguous."""
    frontier = [source_record_id]
    expanded: set[str] = set()
    terminals: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current in expanded:
            continue
        expanded.add(current)
        raw_targets = registry.get(current)
        if isinstance(raw_targets, str):
            targets = {raw_targets}
        else:
            targets = {str(value) for value in raw_targets or () if value}
        targets.discard(current)
        if not targets:
            terminals.add(current)
            continue
        frontier.extend(target for target in targets if target not in expanded)
    if len(terminals) == 1:
        terminal = next(iter(terminals))
        return (terminal if terminal != source_record_id else None), False
    return None, bool(expanded - {source_record_id})


def _remap_issue_target(context: Any, issue: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(issue))
    registry = getattr(context, "_personal_detail_issue_target_remaps", None)
    source = str(row.get("target_record_id") or "").strip()
    if source and isinstance(registry, Mapping):
        target, ambiguous = _resolve_issue_target(registry, source)
        if target:
            row["target_record_id"] = target
        elif ambiguous:
            row.pop("target_record_id", None)
            row["reason_codes"] = list(
                dict.fromkeys(
                    (
                        *(str(value) for value in row.get("reason_codes") or () if value),
                        "issue_target_identity_ambiguous",
                        "diagnostic_left_unlinked",
                    )
                )
            )
    issue_id = _issue_id(row)
    row["extraction_issue_id"] = issue_id
    row["record_id"] = issue_id
    return row


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
        reason_codes = tuple(str(code) for code in anomaly.get("reason_codes") or ())
        category = (
            "ocr_structure_correction"
            if "candidate_b_immediate_amount_pair_required" in reason_codes
            else "ocr_cell_level_error"
        )
        result.append(
            make_issue(
                category=category,
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
                reason_codes=reason_codes,
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


def register_final_liability_issue_records(
    context: Any,
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Register the final liability population used solely for issue linkage.

    The registry is plugin-local diagnostic state.  It does not alter the
    business rows or make an unresolved source observation authoritative.
    """

    setattr(
        context,
        _FINAL_LIABILITY_ISSUE_RECORDS_ATTR,
        [deepcopy(dict(record)) for record in records if isinstance(record, Mapping)],
    )


def _liability_issue_field(value: Any) -> str:
    field_name = str(value or "").strip()
    return _LIABILITY_ISSUE_FIELD_ALIASES.get(field_name, field_name)


def _liability_ref_keys(ref: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    """Return exact or bounded geometry identities for one source reference."""

    logical_page = str(ref.get("logical_page") or "")
    source_page = str(ref.get("source_page") or "")
    table_id = str(ref.get("table_id") or "")
    keys: set[tuple[Any, ...]] = set()
    evidence_ids = {
        str(value) for value in ref.get("evidence_ids") or () if value
    }
    keys.update(
        ("evidence", logical_page, source_page, evidence_id)
        for evidence_id in evidence_ids
    )
    row = ref.get("row")
    column = ref.get("column")
    if table_id and row is not None and column is not None:
        keys.add(("table_cell", logical_page, source_page, table_id, int(row), int(column)))
    bbox = ref.get("bbox")
    if (
        ref.get("geometry_scope") == "cell"
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
    ):
        keys.add(
            (
                "cell_bbox",
                logical_page,
                source_page,
                tuple(round(float(value), 3) for value in bbox),
            )
        )
    return keys


def _liability_refs_overlap(
    issue_ref: Mapping[str, Any],
    record_ref: Mapping[str, Any],
) -> bool:
    """Require compatible strong identities before accepting a locator match."""

    issue_table = str(issue_ref.get("table_id") or "")
    record_table = str(record_ref.get("table_id") or "")
    if issue_table and record_table and issue_table != record_table:
        return False
    issue_evidence = {
        str(value) for value in issue_ref.get("evidence_ids") or () if value
    }
    record_evidence = {
        str(value) for value in record_ref.get("evidence_ids") or () if value
    }
    if issue_evidence and record_evidence and issue_evidence.isdisjoint(record_evidence):
        return False
    return bool(_liability_ref_keys(issue_ref) & _liability_ref_keys(record_ref))


def _liability_record_refs(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    refs = [ref for ref in record.get("source_refs") or () if isinstance(ref, Mapping)]
    refs_by_field = record.get("source_refs_by_field")
    if isinstance(refs_by_field, Mapping):
        refs.extend(
            ref
            for field_refs in refs_by_field.values()
            for ref in field_refs or ()
            if isinstance(ref, Mapping)
        )
    return refs


def _packed_liability_issue_contract(issue: Mapping[str, Any]) -> str:
    observed = issue.get("observed_value")
    retained = observed.get("retained_typed_fields") if isinstance(observed, Mapping) else None
    if not isinstance(retained, Mapping):
        return ""
    value = retained.get("保证合同编号") or retained.get("contract_number")
    return "".join(str(value or "").split()).upper()


def _packed_liability_issue_ordinal(issue: Mapping[str, Any]) -> int | None:
    observed = issue.get("observed_value")
    value = observed.get("printed_sequence") if isinstance(observed, Mapping) else None
    return int(value) if str(value or "").isdigit() else None


def _unique_final_liability_for_issue(
    issue: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Resolve one issue only when independent source identities agree."""

    constraint_sets: list[set[int]] = []
    target_id = str(issue.get("target_record_id") or "")
    exact_targets = {
        index
        for index, record in enumerate(records)
        if target_id and str(record.get("liability_id") or "") == target_id
    }
    if exact_targets:
        constraint_sets.append(exact_targets)

    contract = _packed_liability_issue_contract(issue)
    contract_targets = {
        index
        for index, record in enumerate(records)
        if contract
        and "".join(str(record.get("contract_number") or "").split()).upper()
        == contract
    }
    if contract_targets:
        constraint_sets.append(contract_targets)

    ordinal = _packed_liability_issue_ordinal(issue)
    ordinal_targets = {
        index
        for index, record in enumerate(records)
        if ordinal is not None
        and str(record.get("_printed_sequence") or "").isdigit()
        and int(record["_printed_sequence"]) == ordinal
    }
    if ordinal_targets:
        constraint_sets.append(ordinal_targets)

    issue_refs = [
        ref for ref in issue.get("source_refs") or () if isinstance(ref, Mapping)
    ]
    provenance_targets = {
        index
        for index, record in enumerate(records)
        if issue_refs
        and any(
            _liability_refs_overlap(issue_ref, record_ref)
            for issue_ref in issue_refs
            for record_ref in _liability_record_refs(record)
        )
    }
    if provenance_targets:
        constraint_sets.append(provenance_targets)

    if not constraint_sets:
        return None
    candidates = set.intersection(*constraint_sets)
    return records[next(iter(candidates))] if len(candidates) == 1 else None


def _liability_issue_affected_fields(issue: Mapping[str, Any]) -> tuple[str, ...]:
    if issue.get("issue_code") == "pboc_cell_contract_unresolved":
        field_name = _liability_issue_field(issue.get("field_name"))
        return (field_name,) if field_name else ()
    observed = issue.get("observed_value")
    if not isinstance(observed, Mapping):
        return ()
    explicit = observed.get("affected_fields")
    if isinstance(explicit, (list, tuple, set, frozenset)):
        return tuple(
            dict.fromkeys(
                _liability_issue_field(value) for value in explicit if str(value or "").strip()
            )
        )
    retained = observed.get("retained_typed_fields")
    retained_fields = {
        _liability_issue_field(field_name)
        for field_name in retained
    } if isinstance(retained, Mapping) else set()
    return tuple(sorted(_PACKED_LIABILITY_FIELDS - retained_fields))


def _final_liability_field_resolved(record: Mapping[str, Any], field_name: str) -> bool:
    value = record.get(field_name)
    if value is None or str(value).strip().lower() in _LIABILITY_UNRESOLVED_VALUES:
        return False
    refs_by_field = record.get("source_refs_by_field")
    field_refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
    return bool(field_refs)


def _localize_final_liability_issue(
    issue: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    row = dict(issue)
    if (
        row.get("target_dataset") != "repayment_liability_records"
        or row.get("issue_code")
        not in {
            "candidate_b_packed_liability_row_unresolved",
            "pboc_cell_contract_unresolved",
        }
    ):
        return row
    target = _unique_final_liability_for_issue(row, records)
    if target is None:
        return row
    target_id = str(target.get("liability_id") or "")
    if not target_id:
        return row
    row["target_record_id"] = target_id
    affected_fields = _liability_issue_affected_fields(row)
    if affected_fields and all(
        _final_liability_field_resolved(target, field_name)
        for field_name in affected_fields
    ):
        return None
    if not affected_fields and row.get("issue_code") == "candidate_b_packed_liability_row_unresolved":
        return None
    row["reason_codes"] = list(
        dict.fromkeys(
            (
                *(str(value) for value in row.get("reason_codes") or () if value),
                "unique_final_liability_source_link",
            )
        )
    )
    issue_id = _issue_id(row)
    row["extraction_issue_id"] = issue_id
    row["record_id"] = issue_id
    return row


def collect_extraction_issues(context: Any) -> list[dict[str, Any]]:
    """Return deduplicated plugin diagnostics accumulated by every stage."""
    source_rows = [
        *[dict(row) for row in getattr(context, "_personal_detail_extraction_issues", []) if isinstance(row, Mapping)],
        *_ocr_audit_issues(context),
        *_topology_issues(context),
    ]
    rows = [_remap_issue_target(context, row) for row in source_rows]
    final_liabilities = [
        row
        for row in getattr(context, _FINAL_LIABILITY_ISSUE_RECORDS_ATTR, ()) or ()
        if isinstance(row, Mapping)
    ]
    if final_liabilities:
        rows = [
            localized
            for row in rows
            if (localized := _localize_final_liability_issue(row, final_liabilities))
            is not None
        ]
    precise_by_agreement_field: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        observed = row.get("observed_value")
        if (
            row.get("issue_code") == "pboc_cell_contract_unresolved"
            and observed not in (None, "", [], {})
        ):
            key = (
                str(row.get("target_dataset") or ""),
                str(row.get("target_record_id") or ""),
                str(row.get("field_name") or ""),
            )
            if all(key):
                precise_by_agreement_field[key] = row
    consolidated_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("target_dataset") or ""),
            str(row.get("target_record_id") or ""),
            str(row.get("field_name") or ""),
        )
        precise = precise_by_agreement_field.get(key)
        if (
            row.get("issue_code")
            == "candidate_b_credit_agreement_required_field_unresolved"
            and precise is not None
        ):
            refs = [
                *[dict(value) for value in precise.get("source_refs") or () if isinstance(value, Mapping)],
                *[dict(value) for value in row.get("source_refs") or () if isinstance(value, Mapping)],
            ]
            if refs:
                seen_refs: set[str] = set()
                precise["source_refs"] = [
                    value
                    for value in refs
                    if not (
                        (
                            marker := json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            )
                        )
                        in seen_refs
                        or seen_refs.add(marker)
                    )
                ]
            precise["reason_codes"] = list(
                dict.fromkeys(
                    str(value)
                    for value in (
                        *precise.get("reason_codes", ()),
                        *row.get("reason_codes", ()),
                    )
                    if value
                )
            )
            continue
        consolidated_rows.append(row)
    rows = consolidated_rows
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
    "register_final_liability_issue_records",
    "register_issue_target_remap",
    "retarget_issue_record",
]
