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
        "candidate_b_exact_slot_value_invalid",
        "candidate_b_institution_leading_boundary_ambiguous",
        "candidate_b_institution_branch_without_legal_root",
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
        _account_institution,
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
        normalized = _account_institution(text, independently_corroborated=True)
        return bool(
            compact
            and any(root in compact for root in _ACCOUNT_INSTITUTION_LEGAL_ROOTS)
            and (
                normalized is not None
                or any(
                    root in compact
                    for root in (
                        "银行股份有限公司",
                        "银行有限责任公司",
                        "小额贷款有限公司",
                    )
                )
            )
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


def _clear_resolved_account_field_markers(
    record: dict[str, Any], field_name: str
) -> None:
    aliases = set(_FINAL_ACCOUNT_FIELD_ALIASES[field_name])
    for owner in (_account_record_values(record), record):
        if not isinstance(owner, dict):
            continue
        for marker_name in ("_unresolved_fields", "_invalid_observation_fields"):
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
        field_name = _canonical_account_issue_field(issue.get("field_name"))
        if field_name not in _FINAL_ACCOUNT_FIELD_ALIASES:
            continue
        pair = (str(issue.get("target_record_id") or ""), field_name)
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

        exact_final_locators = {
            locator
            for ref in _account_field_refs(record, field_name)
            if _account_ref_is_exact_for_field(ref, field_name)
            if (locator := _source_ref_locator(ref)) is not None
        }
        if not exact_final_locators.difference(issue_locators):
            continue
        resolved_pairs.add((record_id, field_name))

    if not resolved_pairs:
        return
    context._personal_detail_extraction_issues = [
        issue
        for issue in issues
        if not (
            isinstance(issue, Mapping)
            and str(issue.get("target_dataset") or "") == "credit_accounts"
            and str(issue.get("issue_code") or "")
            in _ACCOUNT_ISSUES_SUPERSEDED_BY_EXACT_FINAL
            and (
                str(issue.get("target_record_id") or ""),
                _canonical_account_issue_field(issue.get("field_name")),
            )
            in resolved_pairs
        )
    ]
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
        native_status_conflict_audit = apply_candidate_b_native_status_conflict_guard(
            self.context,
            [
                row
                for row in corrected_payload.get("repayment_records") or ()
                if isinstance(row, dict)
            ],
            enabled=True,
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
            "personal_detail_source_completeness_ledger": _source_completeness_ledger(self.context),
            "personal_detail_document_consistency_ledger": consistency_audit,
            "personal_detail_dataset_states": dataset_states_from_issues(issues),
            **{f"personal_detail_expected_{name}_count": count for name, count in final_counts.items()},
        }
        content = prepare_personal_detail_source_collections(
            {"facts": facts, "datasets": all_datasets},
            business,
            final_dataset_counts=final_counts,
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
            "canonical_layout": self.context.canonical_layout_audit(),
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
