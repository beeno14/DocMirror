# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Credit report community domain plugin.

Premium community plugin for personal brief, personal detail, and enterprise
credit reports. Extracts identity fields, report subtype/content mode, optional
lightweight section hints, and table records via shared KV extract helpers.

Pipeline role: post-seal domain derivation and Community JSON projection.

Key exports: ``CreditReportPlugin``, ``plugin``.

Dependencies: ``ProjectionData`` and the credit-report projection orchestrator.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Any

from docmirror.output.community_bundle import CommunityBundle
from docmirror.plugins._base.projector import CommunityProjector, ProjectionData
from docmirror.plugins.credit_report.value_utils import stable_record_id

_PERSONAL_DETAIL_SOURCE_COMPLETE = frozenset(
    {"observed_nonempty", "explicitly_empty", "not_applicable"}
)
_PERSONAL_DETAIL_CONTROL_DATASETS = frozenset(
    {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
)
_PERSONAL_DETAIL_SPARSE_STATUS_SEMANTICS = {
    "mode": "potentially_flawed_only",
    "present_dataset_without_status": "silently_trusted_complete",
    "absent_dataset_without_status": "silently_trusted_empty_or_not_applicable",
    "status_row_present": "partial_unknown_or_failed_extraction",
}


def _merge_warning_page_range(
    warning: dict[str, Any],
    pages: Sequence[int],
) -> None:
    """Conserve all cited pages when compact findings share one warning row."""

    positive_pages = [
        int(page)
        for page in [*(warning.get("page_range") or []), *pages]
        if isinstance(page, int) and page > 0
    ]
    if positive_pages:
        warning["page_range"] = [min(positive_pages), max(positive_pages)]


def _personal_detail_source_rows(
    source_datasets: Sequence[Any],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for dataset in source_datasets:
        public = dataset if isinstance(dataset, Mapping) else getattr(dataset, "public", None)
        if not isinstance(public, Mapping):
            continue
        name = str(public.get("name") or "")
        if not name:
            continue
        source_rows = (
            public.get("rows")
            if isinstance(dataset, Mapping)
            else getattr(dataset, "rows", None)
        )
        rows_by_name[name] = [
            row for row in (source_rows or ()) if isinstance(row, dict)
        ]
    return rows_by_name


def _personal_detail_review_fields(
    datasets: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], set[str]]:
    affected: dict[tuple[str, str], set[str]] = {}
    by_name = {
        str(dataset.get("name") or ""): dataset
        for dataset in datasets
        if isinstance(dataset, dict)
    }
    issues = by_name.get("extraction_issues", {}).get("rows") or ()
    for wrapper in issues:
        if not isinstance(wrapper, Mapping):
            continue
        values = wrapper.get("normalized") if isinstance(wrapper.get("normalized"), Mapping) else wrapper
        status = str(values.get("status") or "")
        dataset_name = str(values.get("target_dataset") or "")
        record_id = str(values.get("target_record_id") or "")
        field_name = str(values.get("field_name") or "")
        if (
            status not in {"resolved", "dismissed"}
            and dataset_name
            and record_id
            and field_name
        ):
            affected.setdefault((dataset_name, record_id), set()).add(field_name)
    observations = by_name.get("field_observations", {}).get("rows") or ()
    for wrapper in observations:
        if not isinstance(wrapper, Mapping):
            continue
        values = wrapper.get("normalized") if isinstance(wrapper.get("normalized"), Mapping) else wrapper
        observation_status = str(values.get("observation_status") or "")
        dataset_name = str(values.get("dataset_name") or "")
        record_id = str(values.get("business_record_id") or "")
        field_name = str(values.get("field_name") or "")
        if (
            observation_status != "ocr_corrected"
            and dataset_name
            and record_id
            and field_name
        ):
            affected.setdefault((dataset_name, record_id), set()).add(field_name)
    return affected


def _review_metadata_requires_attention(review: Any) -> bool:
    """Accept only an explicit open-review contract, not any truthy mapping."""

    if not isinstance(review, Mapping):
        return False
    status = str(review.get("status") or review.get("extraction_status") or "").lower()
    return bool(
        review.get("required") is True
        or status
        in {
            "requires_review",
            "review",
            "unresolved",
            "failed",
            "partial",
            "uncertain",
            "unknown",
        }
        or review.get("reason")
        or review.get("reason_codes")
    )


_PAGE_REOCR_EVIDENCE_PREFIX = "personal_detail_page_reocr:"

_ACTIVE_EXACT_OMISSION_STATUSES = frozenset(
    {"active", "open", "requires_review", "review", "warning", "error"}
)
_ACCOUNT_OMISSION_ID = re.compile(
    r"^credit_account:(?P<family>non_revolving_loan|revolving_loan_subaccount|"
    r"revolving_loan_account|credit_card|quasi_credit_card):(?P<ordinal>[1-9]\d*)$"
)
_AGREEMENT_OMISSION_ID = re.compile(r"^credit_agreement:(?P<ordinal>[1-9]\d*)$")
_INQUIRY_OMISSION_ID = re.compile(
    r"^credit_inquiry:(?P<kind>institution|personal):(?P<ordinal>[1-9]\d*)$"
)
_INQUIRY_PHYSICAL_ROW_ID = re.compile(
    r"^source_inquiry_physical_row:(?P<digest>[0-9a-f]{16})$"
)
_INQUIRY_RAW_PHYSICAL_POSITION_ID = re.compile(
    r"^source_inquiry_raw_physical_position:(?P<digest>[0-9a-f]{16})$"
)
_MONTHLY_OMISSION_ID = re.compile(
    r"^(?P<grid>[A-Za-z0-9_.:-]{1,200}):"
    r"(?P<month>\d{4}-(?:0[1-9]|1[0-2]))$"
)
_SOURCE_ACCOUNT_MONTH_OMISSION_ID = re.compile(
    r"^source_account_month:(?P<owner_hash>[0-9a-f]{16}):"
    r"(?P<month>\d{4}-(?:0[1-9]|1[0-2]))$"
)
_OWNED_GRID_MONTHLY_OMISSION_CODE = "candidate_b_monthly_owned_grid_missing_field"
_OWNED_GRID_MONTHLY_REF_SOURCE = "candidate_b_monthly_owned_grid_cell"
_MONTHLY_OMISSION_CODES = frozenset(
    {
        "candidate_b_monthly_grid_owner_unresolved_field",
        "candidate_b_monthly_grid_contract_missing_field",
        "canonical_monthly_source_structure_missing_field",
    }
)
_EXACT_FIELD_FINDING_CODES = frozenset(
    {
        "candidate_b_account_cluster_field_unresolved",
        "candidate_b_credit_agreement_identifier_unresolved",
        "candidate_b_credit_limit_identifier_unresolved",
        "candidate_b_document_local_institution_glyph_conflict",
        "candidate_b_exact_slot_value_conflict",
        "candidate_b_exact_slot_value_invalid",
        "candidate_b_exact_slot_value_unreadable",
        "candidate_b_profile_contract_unresolved",
        "pboc_cell_contract_unresolved",
        "source_account_field_omitted",
        "source_bound_profile_field_omitted",
        "source_credit_agreement_field_omitted",
        "source_employment_field_omitted",
        "source_inquiry_field_omitted",
        "source_mobile_field_omitted",
        "source_residence_field_omitted",
    }
)
_EXACT_FIELD_REF_SOURCES = frozenset(
    {
        "candidate_b_canonical_table",
        "native_detail_table_cell",
        "native_detail_tolerant_table_cell",
        "personal_detail_corrected_page_cell",
    }
)
_EXACT_FIELD_REF_BINDINGS = frozenset(
    {
        "canonical_field_slot",
        "canonical_header_column",
        "canonical_label_slot",
        "closed_canonical_account_cell_cluster",
        "closed_canonical_account_merged_header_geometry",
        "label_column",
    }
)
_EXACT_MONTHLY_BINDINGS = frozenset(
    {
        "canonical_field_slot",
        "canonical_header_column",
        "canonical_label_slot",
        "grid_month_cell",
        "monthly_grid_cell",
        "source_monthly_field_cell",
    }
)


def _personal_detail_row_values(row: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = row.get("normalized")
    return normalized if isinstance(normalized, Mapping) else row


def _finite_exact_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        bbox = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        return False
    return bool(
        all(isfinite(coordinate) for coordinate in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _nonnegative_exact_index(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _exact_evidence_ids(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and value
        and all(
            isinstance(evidence_id, str)
            and evidence_id == evidence_id.strip()
            and bool(evidence_id)
            for evidence_id in value
        )
        and len(set(value)) == len(value)
    )


def _exact_issue_page_range(row: Mapping[str, Any]) -> tuple[int, int] | None:
    source = row.get("source")
    page_range = source.get("page_range") if isinstance(source, Mapping) else None
    if isinstance(page_range, (list, tuple)) and len(page_range) == 2:
        first = _positive_page_number(page_range[0])
        last = _positive_page_number(page_range[1])
        if first is not None and last is not None and last >= first:
            return first, last
    refs = row.get("source_refs")
    if not isinstance(refs, (list, tuple)) and isinstance(source, Mapping):
        refs = source.get("source_refs")
    pages = [
        page
        for ref in refs or ()
        if isinstance(ref, Mapping)
        and (page := _positive_page_number(ref.get("logical_page"))) is not None
    ]
    return (min(pages), max(pages)) if pages and len(pages) == len(refs or ()) else None


def _typed_issue_evidence_value(values: Mapping[str, Any]) -> Any:
    value_key = {
        "string": "string_value",
        "integer": "integer_value",
        "number": "number_value",
        "boolean": "boolean_value",
    }.get(str(values.get("value_type") or ""))
    return values.get(value_key) if value_key else None


def _issue_evidence_scalar_is_exact(
    values: Mapping[str, Any], expected: Any
) -> bool:
    value_type = str(values.get("value_type") or "")
    observed = _typed_issue_evidence_value(values)
    if type(expected) is bool:
        return value_type == "boolean" and type(observed) is bool and observed is expected
    if type(expected) is int:
        return value_type == "integer" and type(observed) is int and observed == expected
    if type(expected) is float:
        return (
            value_type == "number"
            and type(observed) in {int, float}
            and not isinstance(observed, bool)
            and float(observed) == expected
        )
    return value_type == "string" and isinstance(observed, str) and observed == expected


def _identity_evidence_is_exact(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    issue_id: str,
    page_range: tuple[int, int],
    expected: Mapping[str, Any],
) -> bool:
    observed: dict[str, list[Any]] = {}
    for row in evidence_rows:
        values = _personal_detail_row_values(row)
        if str(values.get("extraction_issue_id") or "") != issue_id:
            return False
        if _exact_issue_page_range(row) != page_range:
            return False
        if str(values.get("evidence_kind") or "") != "observed":
            continue
        path = str(values.get("evidence_path") or "")
        if path in expected:
            observed.setdefault(path, []).append(_typed_issue_evidence_value(values))
    return all(observed.get(path) == [value] for path, value in expected.items())


def _owned_grid_identity_evidence_is_exact(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    issue_id: str,
    page_range: tuple[int, int],
    expected: Mapping[str, Any],
) -> bool:
    """Require a closed observed account/month identity for an owned grid cell."""

    observed: dict[str, list[Any]] = {}
    if not evidence_rows:
        return False
    for row in evidence_rows:
        values = _personal_detail_row_values(row)
        if str(values.get("extraction_issue_id") or "") != issue_id:
            return False
        if _exact_issue_page_range(row) != page_range:
            return False
        kind = str(values.get("evidence_kind") or "")
        if kind == "reason":
            continue
        if kind != "observed":
            return False
        path = str(values.get("evidence_path") or "")
        if path not in expected:
            return False
        observed.setdefault(path, []).append(_typed_issue_evidence_value(values))
    return set(observed) == set(expected) and all(
        observed[path] == [value] for path, value in expected.items()
    )


def _inquiry_field_evidence_is_exact(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    issue_id: str,
    page_range: tuple[int, int],
    expected_candidate: Mapping[str, Any],
) -> bool:
    """Validate the typed child evidence left by compact issue projection."""

    expected = {
        ("observed", "source_field_observed"): True,
        **{
            ("candidate", str(path)): value
            for path, value in expected_candidate.items()
        },
    }
    observed: dict[tuple[str, str], list[Any]] = {}
    if not evidence_rows:
        return False
    for row in evidence_rows:
        values = _personal_detail_row_values(row)
        if str(values.get("extraction_issue_id") or "") != issue_id:
            return False
        if _exact_issue_page_range(row) != page_range:
            return False
        kind = str(values.get("evidence_kind") or "")
        if kind == "reason":
            continue
        if kind not in {"observed", "candidate"}:
            return False
        key = (kind, str(values.get("evidence_path") or ""))
        if key not in expected:
            return False
        if not _issue_evidence_scalar_is_exact(values, expected[key]):
            return False
        observed.setdefault(key, []).append(_typed_issue_evidence_value(values))
    return all(observed.get(key) == [value] for key, value in expected.items())


def _inquiry_raw_position_evidence_is_exact(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    issue_id: str,
    page_range: tuple[int, int],
    physical_position_id: str,
) -> bool:
    """Require the complete anonymous row-presence child-evidence contract."""

    expected = {
        ("observed", "source_row_observed"): True,
        ("candidate", "source_physical_row_id"): physical_position_id,
    }
    observed: dict[tuple[str, str], list[Any]] = {}
    if not evidence_rows:
        return False
    for row in evidence_rows:
        values = _personal_detail_row_values(row)
        if str(values.get("extraction_issue_id") or "") != issue_id:
            return False
        if _exact_issue_page_range(row) != page_range:
            return False
        kind = str(values.get("evidence_kind") or "")
        if kind == "reason":
            continue
        if kind not in {"observed", "candidate"}:
            return False
        key = (kind, str(values.get("evidence_path") or ""))
        if key not in expected:
            return False
        if not _issue_evidence_scalar_is_exact(values, expected[key]):
            return False
        observed.setdefault(key, []).append(_typed_issue_evidence_value(values))
    return set(observed) == set(expected) and all(
        observed[key] == [value] for key, value in expected.items()
    )


def _common_exact_omission_ref(
    ref: Mapping[str, Any], page_range: tuple[int, int]
) -> bool:
    logical_page = _positive_page_number(ref.get("logical_page"))
    source_page = _positive_page_number(ref.get("source_page"))
    return bool(
        logical_page is not None
        and source_page is not None
        and page_range[0] <= logical_page <= page_range[1]
        and _finite_exact_bbox(ref.get("bbox"))
        and _exact_evidence_ids(ref.get("evidence_ids"))
    )


def _exact_omission_ref(
    ref: Mapping[str, Any],
    *,
    kind: str,
    ordinal: int | None,
    grid_id: str | None,
    account_id: str | None,
    performance_month: str | None,
    physical_position_id: str | None,
    field_name: str,
    page_range: tuple[int, int],
) -> bool:
    if not _common_exact_omission_ref(ref, page_range):
        return False
    source = str(ref.get("source") or "")
    scope = str(ref.get("geometry_scope") or "")
    binding = str(ref.get("binding") or "")
    binding_quality = str(ref.get("binding_quality") or "")
    if kind == "account":
        return source == "candidate_b_account_anchor"
    if kind == "agreement":
        return bool(
            source == "candidate_b_source_coverage_ledger"
            and scope == "line"
            and binding == "printed_credit_agreement_ordinal"
            and binding_quality == "printed_credit_agreement_ordinal"
            and ref.get("sequence") == ordinal
        )
    if kind == "inquiry":
        exact_table_row = bool(
            source == "native_detail_table"
            and scope == "row"
            and binding == "canonical_header_row"
            and binding_quality in {"", "canonical_header_row"}
            and str(ref.get("table_id") or "").strip()
            and _nonnegative_exact_index(ref.get("row")) is not None
        )
        exact_ordinal_cell = bool(
            source in {
                "native_detail_table_cell",
                "native_detail_inquiry_token_ordinal",
            }
            and scope in {"cell", "token"}
            and binding
            in {"printed_inquiry_ordinal_cell", "printed_inquiry_ordinal_token"}
            and binding_quality
            in {"exact_cell_in_sequence_column", "exact_token_in_sequence_cell"}
            and ref.get("sequence") == ordinal
            and str(ref.get("table_id") or "").strip()
            and _nonnegative_exact_index(ref.get("row")) is not None
            and _nonnegative_exact_index(ref.get("column")) is not None
        )
        return exact_table_row or exact_ordinal_cell
    if kind == "inquiry_raw_physical_position":
        evidence_ids = tuple(str(value) for value in ref.get("evidence_ids") or ())
        exact_ref_keys = {
            "source",
            "logical_page",
            "source_page",
            "table_id",
            "row",
            "bbox",
            "geometry_scope",
            "evidence_ids",
            "binding",
            "binding_quality",
        }
        if (
            physical_position_id is None
            or _INQUIRY_RAW_PHYSICAL_POSITION_ID.fullmatch(physical_position_id)
            is None
            or source != "native_detail_inquiry_raw_physical_row"
            or set(ref) != exact_ref_keys
            or binding != "sealed_raw_inquiry_registered_lattice_band"
            or not isinstance(ref.get("table_id"), str)
            or ref.get("table_id") != str(ref.get("table_id") or "").strip()
            or not ref.get("table_id")
            or _nonnegative_exact_index(ref.get("row")) is None
            or _nonnegative_exact_index(ref.get("column")) is not None
            or any(isinstance(value, bool) for value in ref.get("bbox") or ())
            or any(
                key in ref
                for key in (
                    "inquiry_type",
                    "sequence",
                    "value",
                    "raw_value",
                    "normalized_value",
                    "field_name",
                )
            )
            or (
                scope,
                binding_quality,
            )
            not in {
                ("token_y_band", "all_exact_tokens_uniquely_partitioned"),
                ("exact_source_row_band", "sealed_exact_physical_position"),
            }
        ):
            return False
        expected_id = stable_record_id(
            "source_inquiry_raw_physical_position",
            ref.get("logical_page"),
            ref.get("source_page"),
            ref.get("table_id"),
            ref.get("row"),
            *sorted(evidence_ids),
        )
        return physical_position_id == expected_id
    if kind == "monthly":
        column = ref.get("column", ref.get("col"))
        return bool(
            scope == "cell"
            and str(ref.get("table_id") or "").strip()
            and _nonnegative_exact_index(ref.get("row")) is not None
            and _nonnegative_exact_index(column) is not None
            and str(ref.get("field_name") or "") == field_name
            and str(ref.get("grid_id") or "") == grid_id
            and str(ref.get("performance_month") or "") == performance_month
            and binding in _EXACT_MONTHLY_BINDINGS
            and (not binding_quality or binding_quality in _EXACT_MONTHLY_BINDINGS)
        )
    if kind == "source_account_month_range":
        return bool(
            source == "candidate_b_monthly_anchor_ledger"
            and scope == "line"
            and str(ref.get("account_id") or "") == account_id
            and str(ref.get("performance_month") or "") == performance_month
            and str(ref.get("field_name") or "") == "performance_month"
            and binding == "source_account_month_range"
            and binding_quality == "source_account_month_range"
        )
    if kind == "source_account_month_owned_grid":
        return bool(
            source == _OWNED_GRID_MONTHLY_REF_SOURCE
            and scope == "cell"
            and str(ref.get("table_id") or "").strip()
            and _nonnegative_exact_index(ref.get("row")) is not None
            and _nonnegative_exact_index(ref.get("column")) is not None
            and str(ref.get("account_id") or "") == account_id
            and str(ref.get("grid_id") or "").strip()
            and str(ref.get("performance_month") or "") == performance_month
            and str(ref.get("field_name") or "") == "performance_month"
            and binding == "source_account_month_identity"
            and binding_quality == "source_account_month_identity"
        )
    return False


def _trusted_exact_omission_refs(
    *,
    dataset_name: str,
    public_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Restore only source-local refs for a finite exact-omission contract."""

    if dataset_name != "extraction_issues" or not source_row:
        return []
    public = _personal_detail_row_values(public_row)
    rich = _personal_detail_row_values(source_row)
    identity_keys = (
        "extraction_issue_id",
        "issue_code",
        "status",
        "target_dataset",
        "target_record_id",
        "field_name",
    )
    if any(public.get(key) != rich.get(key) for key in identity_keys):
        return []
    issue_id = str(public.get("extraction_issue_id") or "")
    if (
        not issue_id
        or str(public_row.get("record_id") or "") != issue_id
        or str(source_row.get("record_id") or "") != issue_id
        or str(public.get("status") or "").strip().lower()
        not in _ACTIVE_EXACT_OMISSION_STATUSES
    ):
        return []

    issue_code = str(public.get("issue_code") or "")
    target_dataset = str(public.get("target_dataset") or "")
    target_record_id = str(public.get("target_record_id") or "")
    field_name = str(public.get("field_name") or "")
    kind = ""
    ordinal: int | None = None
    grid_id: str | None = None
    account_id: str | None = None
    performance_month: str | None = None
    physical_position_id: str | None = None
    expected_evidence: dict[str, Any]
    if issue_code == "source_account_record_omitted":
        match = _ACCOUNT_OMISSION_ID.fullmatch(target_record_id)
        if target_dataset != "credit_accounts" or field_name != "account_id" or match is None:
            return []
        kind = "account"
        ordinal = int(match.group("ordinal"))
        expected_evidence = {
            "account_type": match.group("family"),
            "category_sequence": ordinal,
        }
    elif issue_code == "source_credit_agreement_record_omitted":
        match = _AGREEMENT_OMISSION_ID.fullmatch(target_record_id)
        if (
            target_dataset != "credit_agreements"
            or field_name != "credit_agreement_id"
            or match is None
        ):
            return []
        kind = "agreement"
        ordinal = int(match.group("ordinal"))
        expected_evidence = {"credit_agreement_sequence": ordinal}
    elif issue_code == "source_inquiry_record_omitted":
        match = _INQUIRY_OMISSION_ID.fullmatch(target_record_id)
        if target_dataset != "inquiries" or field_name != "inquiry_id" or match is None:
            return []
        kind = "inquiry"
        ordinal = int(match.group("ordinal"))
        expected_evidence = {
            "inquiry_type": match.group("kind"),
            "sequence": ordinal,
        }
    elif issue_code == "source_inquiry_physical_record_omitted":
        match = _INQUIRY_RAW_PHYSICAL_POSITION_ID.fullmatch(target_record_id)
        if (
            target_dataset != "inquiries"
            or field_name != "inquiry_id"
            or match is None
            or public.get("observed_value_type") != "object"
            or rich.get("observed_value_type") != "object"
            or public.get("candidate_value_type") != "object"
            or rich.get("candidate_value_type") != "object"
        ):
            return []
        kind = "inquiry_raw_physical_position"
        physical_position_id = target_record_id
        expected_evidence = {}
    elif issue_code in {
        "candidate_b_monthly_account_range_missing_month",
        _OWNED_GRID_MONTHLY_OMISSION_CODE,
    }:
        match = _SOURCE_ACCOUNT_MONTH_OMISSION_ID.fullmatch(target_record_id)
        if (
            target_dataset != "credit_account_monthly_performance"
            or field_name != "performance_month"
            or match is None
        ):
            return []
        issue_evidence = [
            row
            for row in evidence_rows
            if str(_personal_detail_row_values(row).get("extraction_issue_id") or "")
            == issue_id
        ]
        observed: dict[str, list[Any]] = {}
        for row in issue_evidence:
            values = _personal_detail_row_values(row)
            if str(values.get("evidence_kind") or "") != "observed":
                continue
            path = str(values.get("evidence_path") or "")
            if path in {"account_id", "performance_month"}:
                observed.setdefault(path, []).append(
                    _typed_issue_evidence_value(values)
                )
        account_values = observed.get("account_id") or []
        month_values = observed.get("performance_month") or []
        performance_month = match.group("month")
        if len(account_values) != 1 or month_values != [performance_month]:
            return []
        account_id = str(account_values[0] or "")
        expected_owner_hash = stable_record_id(
            "source_account_month_owner", account_id
        ).split(":", 1)[-1]
        if not account_id or match.group("owner_hash") != expected_owner_hash:
            return []
        kind = (
            "source_account_month_owned_grid"
            if issue_code == _OWNED_GRID_MONTHLY_OMISSION_CODE
            else "source_account_month_range"
        )
        expected_evidence = {
            "account_id": account_id,
            "performance_month": performance_month,
        }
    elif issue_code in _MONTHLY_OMISSION_CODES:
        match = _MONTHLY_OMISSION_ID.fullmatch(target_record_id)
        if (
            target_dataset != "credit_account_monthly_performance"
            or field_name not in {"performance_month", "status_code"}
            or match is None
        ):
            return []
        kind = "monthly"
        grid_id = match.group("grid")
        performance_month = match.group("month")
        expected_evidence = {
            "grid_id": grid_id,
            "performance_month": performance_month,
        }
    else:
        return []

    page_range = _exact_issue_page_range(public_row)
    source_page_range = _exact_issue_page_range(source_row)
    if page_range is None or source_page_range != page_range:
        return []
    issue_evidence = [
        row
        for row in evidence_rows
        if str(_personal_detail_row_values(row).get("extraction_issue_id") or "")
        == issue_id
    ]
    if kind == "inquiry_raw_physical_position":
        if not _inquiry_raw_position_evidence_is_exact(
            issue_evidence,
            issue_id=issue_id,
            page_range=page_range,
            physical_position_id=str(physical_position_id or ""),
        ):
            return []
    else:
        evidence_is_exact = (
            _owned_grid_identity_evidence_is_exact
            if kind == "source_account_month_owned_grid"
            else _identity_evidence_is_exact
        )
        if not issue_evidence or not evidence_is_exact(
            issue_evidence,
            issue_id=issue_id,
            page_range=page_range,
            expected=expected_evidence,
        ):
            return []
    refs = source_row.get("source_refs")
    if not isinstance(refs, (list, tuple)):
        source = source_row.get("source")
        refs = source.get("source_refs") if isinstance(source, Mapping) else ()
    trusted = [
        deepcopy(dict(ref))
        for ref in refs or ()
        if isinstance(ref, Mapping)
        and _exact_omission_ref(
            ref,
            kind=kind,
            ordinal=ordinal,
            grid_id=grid_id,
            account_id=account_id,
            performance_month=performance_month,
            physical_position_id=physical_position_id,
            field_name=field_name,
            page_range=page_range,
        )
    ]
    if not trusted or len(trusted) != len(refs or ()):
        return []
    if kind in {
        "account",
        "source_account_month_range",
        "source_account_month_owned_grid",
        "inquiry_raw_physical_position",
    } and len(trusted) != 1:
        return []
    return trusted


def _trusted_exact_field_finding_refs(
    *,
    dataset_name: str,
    public_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Restore only evidence-sealed refs for one exact field finding.

    This intentionally excludes table/row/page diagnostics and every issue
    without one finite field-owned source cell.  Community can therefore
    distinguish a local, auditable field report from an aggregate warning
    without exposing the rich semantic projection wholesale.
    """

    if dataset_name != "extraction_issues" or not source_row:
        return []
    public = _personal_detail_row_values(public_row)
    rich = _personal_detail_row_values(source_row)
    identity_keys = (
        "extraction_issue_id",
        "issue_code",
        "status",
        "target_dataset",
        "target_record_id",
        "field_name",
    )
    if any(public.get(key) != rich.get(key) for key in identity_keys):
        return []
    issue_id = str(public.get("extraction_issue_id") or "")
    field_name = str(public.get("field_name") or "").strip()
    if (
        not issue_id
        or str(public_row.get("record_id") or "") != issue_id
        or str(source_row.get("record_id") or "") != issue_id
        or str(public.get("status") or "").strip().lower()
        not in _ACTIVE_EXACT_OMISSION_STATUSES
        or str(public.get("issue_code") or "") not in _EXACT_FIELD_FINDING_CODES
        or not str(public.get("target_dataset") or "").strip()
        or not str(public.get("target_record_id") or "").strip()
        or not field_name
    ):
        return []
    page_range = _exact_issue_page_range(public_row)
    if page_range is None or _exact_issue_page_range(source_row) != page_range:
        return []
    refs = source_row.get("source_refs")
    if not isinstance(refs, (list, tuple)):
        source = source_row.get("source")
        refs = source.get("source_refs") if isinstance(source, Mapping) else ()
    if str(public.get("issue_code") or "") == "source_inquiry_field_omitted":
        target_record_id = str(public.get("target_record_id") or "")
        typed_match = _INQUIRY_OMISSION_ID.fullmatch(target_record_id)
        canonical_physical_match = _INQUIRY_PHYSICAL_ROW_ID.fullmatch(
            target_record_id
        )
        raw_physical_match = _INQUIRY_RAW_PHYSICAL_POSITION_ID.fullmatch(
            target_record_id
        )
        if (
            str(public.get("target_dataset") or "") != "inquiries"
            or field_name not in {"inquiry_date", "institution", "reason"}
            or sum(
                match is not None
                for match in (
                    typed_match,
                    canonical_physical_match,
                    raw_physical_match,
                )
            )
            != 1
            or public.get("observed_value_type") != "object"
            or rich.get("observed_value_type") != "object"
            or public.get("candidate_value_type") != "object"
            or rich.get("candidate_value_type") != "object"
            or not isinstance(refs, (list, tuple))
            or len(refs) != 1
            or not isinstance(refs[0], Mapping)
        ):
            return []
        expected_candidate: dict[str, Any]
        if typed_match is not None:
            expected_candidate = {
                "inquiry_type": typed_match.group("kind"),
                "sequence": int(typed_match.group("ordinal")),
            }
        else:
            expected_candidate = {"source_physical_row_id": target_record_id}
        issue_evidence = [
            row
            for row in evidence_rows
            if str(_personal_detail_row_values(row).get("extraction_issue_id") or "")
            == issue_id
        ]
        if not _inquiry_field_evidence_is_exact(
            issue_evidence,
            issue_id=issue_id,
            page_range=page_range,
            expected_candidate=expected_candidate,
        ):
            return []
        ref = dict(refs[0])
        ref_contract = (
            str(ref.get("source") or ""),
            str(ref.get("geometry_scope") or ""),
            str(ref.get("binding") or ""),
            str(ref.get("binding_quality") or ""),
        )
        canonical_ref_contract = (
            "native_detail_inquiry_physical_field",
            "token_y_band",
            "canonical_inquiry_column_y_band",
            "exact_tokens_uniquely_owned_by_date_band",
        )
        raw_ref_contract = (
            "native_detail_inquiry_raw_physical_field",
            "token_y_band",
            "sealed_raw_inquiry_role_y_band",
            "exact_tokens_uniquely_partitioned_in_registered_lattice",
        )
        exact_field_ref_keys = {
            "source",
            "logical_page",
            "source_page",
            "table_id",
            "row",
            "column",
            "bbox",
            "geometry_scope",
            "evidence_ids",
            "field_name",
            "binding",
            "binding_quality",
        }
        if (
            not _common_exact_omission_ref(ref, page_range)
            or set(ref) != exact_field_ref_keys
            or ref_contract
            not in {canonical_ref_contract, raw_ref_contract}
            or (
                canonical_physical_match is not None
                and ref_contract != canonical_ref_contract
            )
            or (
                raw_physical_match is not None
                and ref_contract != raw_ref_contract
            )
            or str(ref.get("field_name") or "") != field_name
            or not isinstance(ref.get("table_id"), str)
            or ref.get("table_id") != str(ref.get("table_id") or "").strip()
            or not ref.get("table_id")
            or _nonnegative_exact_index(ref.get("row")) is None
            or _nonnegative_exact_index(ref.get("column")) is None
            or any(isinstance(value, bool) for value in ref.get("bbox") or ())
        ):
            return []
        return [deepcopy(ref)]
    trusted: list[dict[str, Any]] = []
    for raw_ref in refs or ():
        if not isinstance(raw_ref, Mapping):
            return []
        ref = dict(raw_ref)
        binding = str(ref.get("binding") or "")
        binding_quality = str(ref.get("binding_quality") or "")
        canonical_binding = str(ref.get("binding") or binding_quality)
        if (
            not _common_exact_omission_ref(ref, page_range)
            or str(ref.get("source") or "") not in _EXACT_FIELD_REF_SOURCES
            or str(ref.get("geometry_scope") or "") != "cell"
            or str(ref.get("field_name") or "") != field_name
            or not str(ref.get("table_id") or "").strip()
            or _nonnegative_exact_index(ref.get("row")) is None
            or _nonnegative_exact_index(ref.get("column")) is None
            or canonical_binding not in _EXACT_FIELD_REF_BINDINGS
            or (binding and binding not in _EXACT_FIELD_REF_BINDINGS)
            or (
                binding_quality
                and binding_quality not in _EXACT_FIELD_REF_BINDINGS
                and not binding_quality.startswith("closed_canonical_account_")
            )
        ):
            return []
        trusted.append(deepcopy(ref))
    return trusted if trusted else []


def _positive_page_number(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _valid_page_reocr_evidence_ids(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(
        isinstance(evidence_id, str)
        and evidence_id == evidence_id.strip()
        and evidence_id.startswith(_PAGE_REOCR_EVIDENCE_PREFIX)
        and len(evidence_id) > len(_PAGE_REOCR_EVIDENCE_PREFIX)
        for evidence_id in value
    )


def _corrected_account_currency_refs(
    *,
    dataset_name: str,
    source_row: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return evidence-bearing corrected currency refs from the rich row."""

    if dataset_name != "credit_accounts":
        return {}
    source_envelope = (
        source_row.get("source")
        if isinstance(source_row.get("source"), Mapping)
        else {}
    )
    candidates: list[Any] = [
        *(source_row.get("source_refs") or ()),
        *(source_envelope.get("source_refs") or ()),
    ]
    by_field: dict[str, list[dict[str, Any]]] = {}
    aliases = {
        "currency": "account_currency",
        "account_currency": "account_currency",
        "reporting_amount_currency": "reporting_amount_currency",
    }
    for raw_ref in candidates:
        if not isinstance(raw_ref, Mapping):
            continue
        ref = dict(raw_ref)
        field_name = aliases.get(str(ref.get("field_name") or ""))
        if field_name is None:
            continue
        if (
            str(ref.get("source") or "")
            != "personal_detail_corrected_page_cell"
            or str(ref.get("geometry_scope") or "") != "cell"
            or str(ref.get("binding") or "") != "canonical_field_slot"
            or str(ref.get("binding_quality") or "") != "canonical_field_slot"
            or str(ref.get("field_slot_role") or "") != "value"
            or str(ref.get("evidence_plane") or "") != "business_repair"
            or not _valid_page_reocr_evidence_ids(ref.get("evidence_ids"))
        ):
            continue
        bbox = ref.get("bbox")
        try:
            finite_bbox = (
                tuple(float(value) for value in bbox)
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4
                else ()
            )
        except (TypeError, ValueError):
            finite_bbox = ()
        logical_page = _positive_page_number(ref.get("logical_page"))
        source_page = _positive_page_number(ref.get("source_page"))
        if (
            len(finite_bbox) != 4
            or not all(isfinite(value) for value in finite_bbox)
            or finite_bbox[2] <= finite_bbox[0]
            or finite_bbox[3] <= finite_bbox[1]
            or logical_page is None
            or source_page is None
        ):
            continue
        ref["logical_page"] = logical_page
        ref["source_page"] = source_page
        ref["evidence_ids"] = list(ref["evidence_ids"])
        ref["field_name"] = field_name
        field_refs = by_field.setdefault(field_name, [])
        if ref not in field_refs:
            field_refs.append(ref)
    return by_field


def _trusted_account_currency_raw_fields(
    *,
    dataset_name: str,
    normalized: Mapping[str, Any],
    canonical_source: Mapping[str, Any],
    corrected_refs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> set[str]:
    """Trust only a canonical printed alias backed by its corrected cell."""

    if dataset_name != "credit_accounts":
        return set()
    from docmirror.plugins.credit_report.currency_codes import (
        CURRENCY_CODE_BY_ALIAS,
        ISO_4217_CURRENT_CODES,
    )

    retained: set[str] = set()
    for field_name in ("account_currency", "reporting_amount_currency"):
        if not corrected_refs.get(field_name):
            continue
        normalized_value = str(normalized.get(field_name) or "").strip().upper()
        raw_value = canonical_source.get(field_name)
        if not normalized_value or not isinstance(raw_value, str):
            continue
        compact_raw = "".join(raw_value.split()).strip(
            "()（）[]【】,，;；:："
        ).upper()
        if not compact_raw or compact_raw == normalized_value:
            continue
        source_code = (
            compact_raw
            if compact_raw in ISO_4217_CURRENT_CODES
            else CURRENCY_CODE_BY_ALIAS.get(compact_raw)
        )
        if source_code == normalized_value:
            retained.add(field_name)
    return retained


def _compact_personal_detail_public_projection(
    payload: dict[str, Any],
    *,
    source_datasets: Sequence[Any],
) -> None:
    """Close scanned-detail Community rows over the declared v2 contract.

    The rich semantic result keeps correction/provenance state.  The final
    Community JSON exposes only declared normalized fields and raw evidence for
    fields which still require review. Exact non-identical account-currency
    aliases remain as the source explanation for their trusted ISO value.
    """

    from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
        is_explicit_source_absence,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
        personal_detail_data_dictionary,
    )

    datasets = [item for item in payload.get("datasets") or () if isinstance(item, dict)]
    dictionary = personal_detail_data_dictionary().get("datasets") or {}
    source_rows_by_name = _personal_detail_source_rows(source_datasets)
    issue_evidence_rows = source_rows_by_name.get("extraction_issue_evidence", [])
    review_fields = _personal_detail_review_fields(datasets)

    for dataset in datasets:
        name = str(dataset.get("name") or "")
        definition = dictionary.get(name) if isinstance(dictionary.get(name), Mapping) else {}
        declared = definition.get("columns") if isinstance(definition.get("columns"), Mapping) else {}
        declared_keys = tuple(str(key) for key in declared)
        allowed = frozenset(declared_keys)
        public_rows = [row for row in dataset.get("rows") or () if isinstance(row, dict)]
        source_rows = source_rows_by_name.get(name, [])
        source_by_id = {
            str(row.get("record_id") or ""): row
            for row in source_rows
            if str(row.get("record_id") or "")
        }
        raw_available: set[str] = set()

        for index, row in enumerate(public_rows):
            source_normalized = (
                row.get("normalized")
                if isinstance(row.get("normalized"), Mapping)
                else {}
            )
            normalized = {
                key: source_normalized.get(key)
                for key in declared_keys
            }
            row["normalized"] = normalized
            record_id = str(row.get("record_id") or "")
            source_row = source_by_id.get(record_id)
            if source_row is None and not record_id and index < len(source_rows):
                positional_source = source_rows[index]
                if not str(positional_source.get("record_id") or ""):
                    source_row = positional_source
            source_row = source_row if isinstance(source_row, Mapping) else {}
            public_source = row.get("source")
            public_page_range = (
                tuple(public_source.get("page_range") or ())
                if isinstance(public_source, Mapping)
                else ()
            )
            canonical_source = (
                source_row.get("canonical_raw")
                if isinstance(source_row.get("canonical_raw"), Mapping)
                else {}
            )
            raw_source = (
                source_row.get("raw")
                if isinstance(source_row.get("raw"), Mapping)
                else {}
            )
            trusted_source_absence: dict[str, Any] = {}
            if name == "credit_accounts" and "repayment_periods" in allowed:
                field_name = "repayment_periods"
                canonical_absence = canonical_source.get(field_name)
                exact_absence = (
                    canonical_absence
                    if is_explicit_source_absence(canonical_absence)
                    else None
                )
                if is_explicit_source_absence(normalized.get(field_name)) or (
                    exact_absence is not None
                ):
                    normalized[field_name] = None
                    row.pop(field_name, None)
                if exact_absence is not None:
                    trusted_source_absence[field_name] = exact_absence
            affected = set(review_fields.get((name, record_id), ()))
            affected &= allowed
            if name in _PERSONAL_DETAIL_CONTROL_DATASETS:
                affected.clear()
            corrected_currency_refs = _corrected_account_currency_refs(
                dataset_name=name,
                source_row=source_row,
            )
            trusted_currency_raw = _trusted_account_currency_raw_fields(
                dataset_name=name,
                normalized=normalized,
                canonical_source=canonical_source,
                corrected_refs=corrected_currency_refs,
            )
            trusted_currency_raw &= allowed
            trusted_source_absence_fields = set(trusted_source_absence) & allowed
            retained_raw_fields = (
                affected | trusted_currency_raw | trusted_source_absence_fields
            )
            existing_review = row.get("review")
            keep_review_metadata = bool(
                name in _PERSONAL_DETAIL_CONTROL_DATASETS
                or affected
                or _review_metadata_requires_attention(existing_review)
            )

            public_canonical = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), Mapping) else {}
            public_raw = row.get("raw") if isinstance(row.get("raw"), Mapping) else {}
            canonical_evidence: dict[str, Any] = {}
            raw_evidence: dict[str, Any] = {}
            for key in declared_keys:
                if key not in retained_raw_fields:
                    continue
                if key in trusted_source_absence_fields:
                    exact_absence = trusted_source_absence[key]
                    canonical_evidence[key] = exact_absence
                    raw_evidence[key] = exact_absence
                    continue
                if key in trusted_currency_raw:
                    canonical_value = canonical_source[key]
                    canonical_evidence[key] = canonical_value
                    raw_evidence[key] = canonical_value
                    continue
                if key in canonical_source:
                    canonical_evidence[key] = canonical_source[key]
                elif key in raw_source:
                    canonical_evidence[key] = raw_source[key]
                elif key in public_canonical:
                    canonical_evidence[key] = public_canonical[key]
                if key in raw_source:
                    raw_evidence[key] = raw_source[key]
                elif key in canonical_source:
                    raw_evidence[key] = canonical_source[key]
                elif key in public_raw:
                    raw_evidence[key] = public_raw[key]
            row["canonical_raw"] = canonical_evidence
            row["raw"] = raw_evidence
            if not keep_review_metadata:
                # Community's record envelope requires a source object, but a
                # successfully trusted business row does not need to publish
                # page/provenance diagnostics.  The rich semantic payload is
                # untouched and still carries that evidence for internal use.
                row["source"] = {}
                row.pop("confidence", None)
                row.pop("review", None)
            trusted_refs = [
                ref
                for field_name in sorted(trusted_currency_raw)
                for ref in corrected_currency_refs.get(field_name) or ()
            ]
            trusted_refs.extend(
                _trusted_exact_omission_refs(
                    dataset_name=name,
                    public_row=row,
                    source_row=source_row,
                    evidence_rows=issue_evidence_rows,
                )
            )
            trusted_refs.extend(
                _trusted_exact_field_finding_refs(
                    dataset_name=name,
                    public_row=row,
                    source_row=source_row,
                    evidence_rows=issue_evidence_rows,
                )
            )
            if trusted_refs:
                source = row.get("source")
                public_source = dict(source) if isinstance(source, Mapping) else {}
                public_source["source_refs"] = trusted_refs
                pages = {
                    int(page)
                    for page in public_page_range
                    if isinstance(page, int)
                    and not isinstance(page, bool)
                    and page > 0
                }
                pages.update(
                    int(ref["logical_page"])
                    for ref in trusted_refs
                    if isinstance(ref.get("logical_page"), int)
                    and not isinstance(ref.get("logical_page"), bool)
                    and int(ref["logical_page"]) > 0
                )
                if pages:
                    public_source["page_range"] = [min(pages), max(pages)]
                row["source"] = public_source
            raw_available.update(
                key
                for key in retained_raw_fields
                if canonical_evidence.get(key) not in (None, "")
                or raw_evidence.get(key) not in (None, "")
            )

        columns = [
            column
            for column in dataset.get("columns") or ()
            if isinstance(column, dict) and str(column.get("key") or "") in allowed
        ]
        for column in columns:
            column["raw_available"] = str(column.get("key") or "") in raw_available
        dataset["columns"] = columns
        if isinstance(dataset.get("reading_columns"), list):
            dataset["reading_columns"] = [
                key for key in dataset["reading_columns"] if str(key) in allowed
            ]

    status_dataset = next(
        (dataset for dataset in datasets if str(dataset.get("name") or "") == "dataset_status"),
        None,
    )
    if status_dataset is not None:
        status_dataset["sparse_status_semantics"] = dict(
            _PERSONAL_DETAIL_SPARSE_STATUS_SEMANTICS
        )


def _apply_personal_detail_dataset_status(payload: dict[str, Any]) -> None:
    """Make public dataset envelopes agree with the v2 source-status ledger.

    Community's ordinary completeness calculation proves only that projected
    rows were conserved during serialization.  It must not turn a source-
    partial scanned dataset into ``complete`` merely because all rows that
    reached the projector were written successfully.
    """

    datasets = [item for item in payload.get("datasets") or () if isinstance(item, dict)]
    status_dataset = next(
        (item for item in datasets if str(item.get("name") or "") == "dataset_status"),
        None,
    )
    if status_dataset is None:
        return
    status_by_name: dict[str, dict[str, Any]] = {}
    for wrapper in status_dataset.get("rows") or ():
        if not isinstance(wrapper, dict):
            continue
        values = wrapper.get("normalized") if isinstance(wrapper.get("normalized"), dict) else wrapper
        name = str(values.get("dataset_name") or "")
        if name:
            status_by_name[name] = values

    for dataset in datasets:
        name = str(dataset.get("name") or "")
        control = status_by_name.get(name)
        if control is None:
            continue
        presence = str(control.get("presence_status") or "unknown")
        source_complete = presence in _PERSONAL_DETAIL_SOURCE_COMPLETE
        dataset["status"] = "complete" if source_complete else "partial"
        emitted = int(dataset.get("row_count") or len(dataset.get("rows") or ()))
        expected_raw = control.get("expected_row_count")
        expected = (
            int(expected_raw)
            if isinstance(expected_raw, (int, float)) and not isinstance(expected_raw, bool)
            else None
        )
        completeness = dict(dataset.get("completeness") or {})
        count_conflict = expected is not None and expected < emitted
        if expected is None:
            # The sparse status ledger intentionally omits a source expected
            # count when it only knows that extraction was partial/failed.
            # Keep Community's integer row-conservation count in that case;
            # ``verified`` and ``basis`` carry the source-completeness truth.
            projected_expected = completeness.get("expected_row_count")
            expected = (
                max(int(projected_expected), emitted)
                if isinstance(projected_expected, (int, float))
                and not isinstance(projected_expected, bool)
                else emitted
            )
        elif count_conflict:
            # The source ledger remains available in dataset_status, while the
            # public envelope must obey expected >= emitted.  More emitted
            # rows than the source expected is itself an unresolved population
            # conflict, never a verified complete dataset.
            expected = emitted
            source_complete = False
            dataset["status"] = "partial"
        completeness.update(
            {
                "expected_row_count": expected,
                "emitted_row_count": emitted,
                "omitted_row_count": max(expected - emitted, 0),
                "verified": bool(source_complete),
                "basis": (
                    f"personal_detail_dataset_status:{presence}:expected_less_than_emitted"
                    if count_conflict
                    else (
                        f"personal_detail_dataset_status:{presence}:observed_only:population_unverified"
                        if expected_raw is None
                        and control.get("reason") == "account_month_source_position_population_unverified"
                        else f"personal_detail_dataset_status:{presence}"
                    )
                ),
            }
        )
        dataset["completeness"] = completeness


class _CreditReportCommunityBundle(CommunityBundle):
    """Publish compact Community JSON without weakening rich semantic bindings."""

    @staticmethod
    def _uses_scanned_personal_detail_public_projection(facts: Mapping[str, Any]) -> bool:
        """Match the stable facts emitted by the personal-detail variant router."""

        return bool(
            facts.get("report_subtype") == "personal_detail"
            and facts.get("content_mode") in {"scanned_ocr", "mixed"}
        )

    @staticmethod
    def _is_enterprise_semantic(payload: dict[str, Any]) -> bool:
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        return facts.get("report_subtype") == "enterprise"

    @staticmethod
    def _is_personal_brief_semantic(payload: dict[str, Any]) -> bool:
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        return facts.get("report_subtype") == "personal_brief"

    def _cached_personal_brief_public_payload(
        self,
        semantic: dict[str, Any],
    ) -> dict[str, Any] | None:
        cached_source = getattr(self, "_personal_brief_cached_source", None)
        cached_payload = getattr(self, "_personal_brief_cached_public_payload", None)
        if cached_source == semantic and isinstance(cached_payload, dict):
            return deepcopy(cached_payload)
        return None

    def _cache_personal_brief_public_payload(
        self,
        semantic: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        self._personal_brief_cached_source = deepcopy(semantic)
        self._personal_brief_cached_public_payload = deepcopy(payload)
        self._personal_brief_cached_artifact_payload = None

    def semantic_payload(self) -> dict[str, Any]:
        payload = super().semantic_payload()
        if getattr(self, "_unvalidated_personal_detail", False):
            from docmirror.plugins.credit_report.personal_detail_scanned.unvalidated import (
                omit_validation,
            )

            return omit_validation(payload)
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        if facts.get("report_subtype") != "enterprise":
            return payload
        extraction = facts.pop("extraction_report", None)
        if isinstance(extraction, dict):
            payload["extraction"] = extraction
        audit = facts.pop("audit_report", None)
        if isinstance(audit, dict):
            payload["audit"] = audit
        for key in tuple(facts):
            if key.startswith("enterprise_expected_"):
                facts.pop(key, None)
        extensions = domain.get("extensions") if isinstance(domain.get("extensions"), dict) else {}
        overrides = (
            extensions.get("community_projection_overrides")
            if isinstance(extensions.get("community_projection_overrides"), dict)
            else {}
        )
        for key in ("internal_fields", "internal_facts"):
            values = overrides.get(key)
            if not isinstance(values, list):
                continue
            overrides[key] = [
                value for value in values if not str(value).startswith("enterprise_expected_")
            ]
        return payload

    def json_payload(self, semantic: dict[str, Any] | None = None) -> dict[str, Any]:
        semantic_payload = semantic or self.semantic_payload()
        if getattr(self, "_unvalidated_personal_detail", False):
            from docmirror.plugins.credit_report.personal_detail_scanned.unvalidated import (
                omit_validation,
            )
            from docmirror.plugins.credit_report.projection import _compact_public_datasets

            payload = super().json_payload(semantic_payload)
            for dataset in payload.get("datasets") or []:
                if isinstance(dataset, dict):
                    name = str(dataset.get("name") or dataset.get("id") or "dataset")
                    dataset["rows"] = _compact_public_datasets({name: dataset.get("rows") or []})[name]
            return omit_validation(payload)
        domain = (
            semantic_payload.get("domain")
            if isinstance(semantic_payload.get("domain"), dict)
            else {}
        )
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        enterprise = facts.get("report_subtype") == "enterprise"
        personal_brief = self._is_personal_brief_semantic(semantic_payload)
        scanned_personal_detail = self._uses_scanned_personal_detail_public_projection(facts)
        if personal_brief:
            cached_payload = self._cached_personal_brief_public_payload(semantic_payload)
            if cached_payload is not None:
                return cached_payload

        payload = super().json_payload(semantic_payload)
        if not personal_brief:
            from docmirror.plugins.credit_report.projection import _compact_public_datasets

            for dataset in payload.get("datasets") or []:
                if not isinstance(dataset, dict):
                    continue
                dataset_id = str(dataset.get("id") or dataset.get("name") or "dataset")
                records = [
                    record
                    for record in (dataset.get("rows") or [])
                    if isinstance(record, dict)
                ]
                dataset["rows"] = _compact_public_datasets({dataset_id: records})[
                    dataset_id
                ]
        if enterprise:
            from docmirror.plugins.credit_report.enterprise_native.projector import (
                project_enterprise_community_json,
            )

            payload = project_enterprise_community_json(payload)
            audit = (
                semantic_payload.get("audit")
                if isinstance(semantic_payload.get("audit"), dict)
                else {}
            )
            datasets_by_name = {
                str(dataset.get("name") or ""): str(dataset.get("id") or "")
                for dataset in payload.get("datasets") or []
                if isinstance(dataset, dict)
            }
            warning_rows = [
                warning
                for warning in (payload.get("warnings") or [])
                if isinstance(warning, dict)
            ]
            for finding in audit.get("findings") or []:
                if not isinstance(finding, dict) or finding.get("severity") not in {
                    "warning",
                    "error",
                }:
                    continue
                code = str(finding.get("code") or "ENTERPRISE_AUDIT_REVIEW")
                path = str(finding.get("path") or "")
                message = (
                    f"{code}: {str(finding.get('message') or '').strip()}"
                    f"{' Review ' + path + '.' if path else ''}"
                )
                warning = next(
                    (
                        row
                        for row in warning_rows
                        if row.get("code") == code and row.get("message") == message
                    ),
                    None,
                )
                if warning is None:
                    warning = {
                        "code": code,
                        "level": (
                            "error" if finding.get("severity") == "error" else "warning"
                        ),
                        "message": message,
                    }
                    warning_rows.append(warning)
                dataset_id = datasets_by_name.get(str(finding.get("dataset") or ""))
                if dataset_id:
                    warning["dataset_id"] = dataset_id
                pages = sorted(
                    {
                        int(page)
                        for page in (finding.get("source_pages") or [])
                        if str(page).isdigit() and int(page) > 0
                    }
                )
                if pages:
                    _merge_warning_page_range(warning, pages)
            payload["warnings"] = warning_rows
        elif personal_brief:
            from docmirror.plugins.credit_report.personal_brief_native.audit import (
                append_personal_brief_observational_warnings,
            )
            from docmirror.plugins.credit_report.personal_brief_native.projector import (
                project_personal_brief_community_json,
            )

            payload = project_personal_brief_community_json(payload)
            payload = append_personal_brief_observational_warnings(
                semantic_payload,
                payload,
            )
            self._cache_personal_brief_public_payload(semantic_payload, payload)
        elif scanned_personal_detail:
            semantic_datasets = semantic_payload.get("datasets")
            _compact_personal_detail_public_projection(
                payload,
                # The semantic payload is the post-repair evidence plane.
                # ``self.datasets`` can still hold the pre-repair rows used to
                # construct the bundle; consulting it here silently discards
                # valid field repairs during Community compaction.
                source_datasets=(
                    semantic_datasets
                    if isinstance(semantic_datasets, list)
                    else self.datasets
                ),
            )
            _apply_personal_detail_dataset_status(payload)
        return payload

    @staticmethod
    def _is_scanned_personal_detail_semantic(semantic: dict[str, Any]) -> bool:
        domain = semantic.get("domain") if isinstance(semantic.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        return bool(
            facts.get("report_subtype") == "personal_detail"
            and facts.get("content_mode") in {"scanned_ocr", "mixed"}
        )

    def render_markdown(self) -> str:
        semantic = self.semantic_payload()
        if not self._is_scanned_personal_detail_semantic(semantic):
            return super().render_markdown()
        from docmirror.plugins.credit_report.personal_detail_scanned.markdown import (
            render_personal_detail_business_markdown,
        )

        return render_personal_detail_business_markdown(semantic)

    def render_enhanced_markdown(self, semantic: dict[str, Any] | None = None) -> str:
        semantic_payload = semantic or self.semantic_payload()
        if not self._is_scanned_personal_detail_semantic(semantic_payload):
            return super().render_enhanced_markdown(semantic_payload)
        from docmirror.plugins.credit_report.personal_detail_scanned.markdown import (
            render_personal_detail_business_markdown,
        )

        return render_personal_detail_business_markdown(semantic_payload)

    def _enterprise_artifact_semantic(
        self,
        semantic: dict[str, Any],
    ) -> dict[str, Any]:
        from docmirror.plugins.credit_report.enterprise_native.projector import (
            project_enterprise_artifact_semantic,
        )

        return project_enterprise_artifact_semantic(
            semantic,
            self.json_payload(semantic),
        )

    def _personal_brief_artifact_semantic(
        self,
        semantic: dict[str, Any],
    ) -> dict[str, Any]:
        from docmirror.plugins.credit_report.personal_brief_native.projector import (
            project_personal_brief_artifact_semantic,
        )

        cached_source = getattr(self, "_personal_brief_cached_source", None)
        cached_payload = getattr(self, "_personal_brief_cached_artifact_payload", None)
        if cached_source == semantic and isinstance(cached_payload, dict):
            return deepcopy(cached_payload)

        artifact = project_personal_brief_artifact_semantic(
            semantic,
            self.json_payload(semantic),
        )
        self._personal_brief_cached_artifact_payload = deepcopy(artifact)
        return artifact

    def render_dataset_csvs(self, semantic: dict[str, Any] | None = None) -> dict[str, str]:
        semantic_payload = semantic or self.semantic_payload()
        if self._is_enterprise_semantic(semantic_payload):
            return super().render_dataset_csvs(
                self._enterprise_artifact_semantic(semantic_payload)
            )
        if self._is_personal_brief_semantic(semantic_payload):
            return super().render_dataset_csvs(
                self._personal_brief_artifact_semantic(semantic_payload)
            )
        return super().render_dataset_csvs(semantic_payload)

    def render_audit_csv(self, semantic: dict[str, Any] | None = None) -> str:
        semantic_payload = semantic or self.semantic_payload()
        if self._is_enterprise_semantic(semantic_payload):
            return super().render_audit_csv(
                self._enterprise_artifact_semantic(semantic_payload)
            )
        if self._is_personal_brief_semantic(semantic_payload):
            return super().render_audit_csv(
                self._personal_brief_artifact_semantic(semantic_payload)
            )
        return super().render_audit_csv(semantic_payload)


class CreditReportPlugin(CommunityProjector):
    """Community edition plugin for credit report document processing."""

    def __init__(self, *, unvalidated: bool = False) -> None:
        """Opt into extraction-only personal-detail output for this instance.

        The default and other report variants retain their existing behavior.
        Instance-local configuration avoids changing the registered singleton
        or another request's mode. This flag is deliberately not a JSON fact.
        """
        if not isinstance(unvalidated, bool):
            raise TypeError("unvalidated must be a boolean")
        self._unvalidated = unvalidated

    @property
    def unvalidated(self) -> bool:
        return self._unvalidated

    @property
    def domain_name(self) -> str:
        return "credit_report"

    @property
    def display_name(self) -> str:
        return "Credit Report (Community)"

    @property
    def projector_id(self) -> str:
        return self.domain_name

    @property
    def identity_fields(self) -> Sequence[tuple[str, Sequence[str]]]:
        return (
            ("subject_name", ("被查询者姓名", "企业名称", "姓名", "Name", "报告主体")),
            ("id_number", ("被查询者证件号码", "身份证号", "证件号码", "ID Number")),
            ("id_type", ("被查询者证件类型", "证件类型", "ID Type")),
            ("marital_status", ("婚姻状况",)),
            ("unified_social_credit_code", ("统一社会信用代码",)),
            ("zhongzheng_code", ("中征码", "贷款卡编码", "贷款卡号")),
            ("query_institution", ("查询机构",)),
            ("report_time", ("报告时间", "查询时间", "Report Time")),
            ("report_number", ("报告编号", "Report No", "NO.")),
        )

    def derive(self, parse_result, text: str = "") -> ProjectionData:
        from docmirror.plugins.credit_report.report_profile import (
            detect_credit_report_subtype,
        )

        report_subtype = detect_credit_report_subtype(parse_result, text)
        if report_subtype == "enterprise":
            from docmirror.plugins.credit_report.enterprise_native.projector import (
                derive_enterprise_projection,
            )

            return derive_enterprise_projection(self, parse_result, text)
        if report_subtype == "personal_brief":
            from docmirror.plugins.credit_report.personal_brief_native.projector import (
                derive_personal_brief_projection,
            )

            return derive_personal_brief_projection(self, parse_result, text)
        if self.unvalidated and report_subtype == "personal_detail":
            from docmirror.plugins.credit_report.personal_detail_scanned.unvalidated import (
                derive_unvalidated_personal_detail,
            )

            return derive_unvalidated_personal_detail(self, parse_result, text)
        from docmirror.plugins.credit_report.projection import derive_credit_report_projection

        return derive_credit_report_projection(self, parse_result, text)

    def project_bundle(
        self,
        sealed,
        *,
        file_path: str = "",
        file_id: str = "001",
        document_id: str = "",
    ):
        """Apply document-variant presentation overrides inside this plugin."""
        from docmirror.models.sealed import SealedParseResult
        from docmirror.output.community_bundle import project_community_bundle
        from docmirror.plugins._base.projector import load_projection_policy

        if not isinstance(sealed, SealedParseResult):
            raise TypeError(f"{type(self).__name__}.project expects SealedParseResult")
        if not self.supports(sealed):
            return None
        before = sealed.integrity_fingerprint
        read_view = sealed.to_read_view()
        derived = self.derive(
            read_view,
            str(read_view.full_text or read_view.raw_text or ""),
        )
        policy = load_projection_policy(type(self).__module__.rsplit(".", 1)[0])
        unvalidated_personal_detail = bool(
            self.unvalidated and derived.domain_facts.get("report_subtype") == "personal_detail"
        )
        overrides = derived.semantic.get("community_projection_overrides")
        if isinstance(overrides, dict):
            for key, values in overrides.items():
                if isinstance(values, dict):
                    policy[key] = {**dict(policy.get(key) or {}), **values}
                elif key in {
                    "internal_fields",
                    "internal_facts",
                    "publish_empty_datasets",
                } and isinstance(
                    values, (list, tuple)
                ):
                    policy[key] = list(
                        dict.fromkeys([*(policy.get(key) or ()), *map(str, values)])
                    )
        if unvalidated_personal_detail:
            policy.pop("completeness", None)
            policy["publish_empty_datasets"] = []
            # Never mix pre-existing domain output/control collections into
            # this fresh extraction. Initial OCR evidence remains read-only.
            source_keys = set(read_view.entities.domain_specific or {}) - set(derived.domain_facts) - set(derived.datasets)
            for key in ("internal_fields", "internal_facts"):
                policy[key] = list(dict.fromkeys([*(policy.get(key) or ()), *sorted(source_keys)]))
        projected = project_community_bundle(
            sealed,
            file_path=file_path,
            file_id=file_id,
            document_id=document_id,
            projection_data=derived.model_dump(mode="python"),
            projection_policy=policy,
        )
        bundle = _CreditReportCommunityBundle(
            schema=projected.schema,
            document=projected.document,
            sections=projected.sections,
            datasets=projected.datasets,
            files=projected.files,
            warnings=projected.warnings,
            result=projected.result,
            source_fingerprint=projected.source_fingerprint,
            parse_result_schema=projected.parse_result_schema,
            classification=projected.classification,
            domain=projected.domain,
            diagnostics=projected.diagnostics,
            content_markdown_override=projected.content_markdown_override,
        )
        if unvalidated_personal_detail:
            bundle._unvalidated_personal_detail = True
        bundle.render_markdown()
        if sealed.integrity_fingerprint != before or not sealed.verify_integrity():
            raise RuntimeError("Post-seal projector changed the sealed snapshot")
        return bundle

plugin = CreditReportPlugin()
