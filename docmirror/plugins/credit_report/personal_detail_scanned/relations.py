# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate B relationship materialization.

Account/month linking and overdue derivation live in the document-type branch
so the clean pipeline does not pass its rows through shared scanned-report
assembly code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from docmirror.plugins.credit_report.value_utils import stable_record_id

_REPAYMENT_ENDPOINT_RE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月")
_REPAYMENT_RANGE_SEPARATORS = frozenset({"", "-", "—", "–", "－", "至", "到", "一"})
_PERFORMANCE_MONTH_RE = re.compile(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])\Z")
_OWNED_GRID_MONTHLY_OMISSION_CODE = "candidate_b_monthly_owned_grid_missing_field"
_OWNED_GRID_MONTHLY_REF_SOURCE = "candidate_b_monthly_owned_grid_cell"
_OWNED_GRID_MONTHLY_INPUT_SOURCES = frozenset(
    {"native_detail_table_cell", "sealed_native_physical_table_cell"}
)
_OWNED_GRID_MONTHLY_INPUT_BINDINGS = frozenset(
    {
        "canonical_field_slot",
        "canonical_header_column",
        "canonical_label_slot",
        "grid_month_cell",
        "monthly_grid_cell",
        "source_monthly_field_cell",
    }
)


def _geometry_box(value: Any) -> list[float] | None:
    """Recover a usable grid box from its own box or constituent cells."""
    boxes: list[list[float]] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            raw = node.get("bbox")
            if isinstance(raw, list) and len(raw) == 4:
                try:
                    box = [float(item) for item in raw]
                except (TypeError, ValueError):
                    box = []
                if (
                    len(box) == 4
                    and all(math.isfinite(item) for item in box)
                    and box[2] > box[0]
                    and box[3] > box[1]
                ):
                    boxes.append(box)
            for key in ("cells", "rows", "col_bands", "row_bands"):
                collect(node.get(key))
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(value)
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _positive_native_int(value: Any) -> int | None:
    """Accept only positive JSON/Python integer identities, never coercions."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _grid_months(grid: dict[str, Any]) -> set[tuple[int, int]]:
    date_range = (grid.get("audit") or {}).get("date_range") or {}
    start_year = _positive_native_int(date_range.get("start_year"))
    start_month = _positive_native_int(date_range.get("start_month"))
    end_year = _positive_native_int(date_range.get("end_year"))
    end_month = _positive_native_int(date_range.get("end_month"))
    if None in (start_year, start_month, end_year, end_month):
        return set()
    if start_year < 1900 or end_year < start_year or not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
        return set()
    start = start_year * 12 + start_month - 1
    end = end_year * 12 + end_month - 1
    if end < start or end - start > 120:
        return set()
    return {(value // 12, value % 12 + 1) for value in range(start, end + 1)}


def _printed_repayment_range(text: Any) -> dict[str, int] | None:
    """Decode exactly two printed year/month endpoints from one repayment anchor.

    A finite separator catalog handles the normal dash forms first.  OCR may
    substitute that single separator with one short non-numeric token (for
    example ``一``) or omit it entirely.  That fallback remains fail-closed:
    the normalized line must contain the repayment-record suffix and exactly
    two complete ``YYYY年M月`` endpoints, with no third endpoint nearby.
    """

    compact = re.sub(r"\s+", "", str(text or ""))
    endpoints = list(_REPAYMENT_ENDPOINT_RE.finditer(compact))
    if len(endpoints) not in {1, 2}:
        return None
    prefix = compact[: endpoints[0].start()]
    suffix = compact[endpoints[-1].end() :]
    # OCR line grouping occasionally prepends one isolated sequence/glyph.  It
    # may not prepend a word or another date, and the repayment suffix must be
    # consumed completely rather than merely found somewhere in the line.
    if len(prefix) > 1 or suffix not in {"还款记录", "的还款记录"}:
        return None
    if len(endpoints) == 2:
        separator = compact[endpoints[0].end() : endpoints[1].start()]
        if separator not in _REPAYMENT_RANGE_SEPARATORS:
            return None
    raw_values = (
        int(endpoints[0].group(1)),
        int(endpoints[0].group(2)),
        int(endpoints[-1].group(1)),
        int(endpoints[-1].group(2)),
    )
    start_year, start_month, end_year, end_month = raw_values
    start = start_year * 12 + start_month - 1
    end = end_year * 12 + end_month - 1
    if (
        not 1 <= start_month <= 12
        or not 1 <= end_month <= 12
        or end < start
        or end - start > 120
    ):
        return None
    return {
        "start_year": start_year,
        "start_month": start_month,
        "end_year": end_year,
        "end_month": end_month,
    }


def _exact_performance_month(record: Any) -> tuple[int, int] | None:
    """Require an exact YYYY-MM identity and non-coerced matching aliases."""

    if not isinstance(record, dict):
        return None
    raw_performance_month = record.get("performance_month")
    if not isinstance(raw_performance_month, str):
        return None
    match = _PERFORMANCE_MONTH_RE.fullmatch(raw_performance_month)
    if match is None:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    for key, expected in (("year", year), ("month", month)):
        if key not in record or record.get(key) is None:
            continue
        alias = _positive_native_int(record.get(key))
        if alias != expected:
            return None
    return year, month


def _canonical_account_identifier(value: Any) -> str | None:
    """Accept only a role-valid identifier already in canonical form."""

    if not isinstance(value, str) or not value:
        return None
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        normalize_role_candidate,
        role_candidate_is_valid,
    )

    normalized = normalize_role_candidate(value, "account_identifier")
    if normalized != value or not role_candidate_is_valid(value, "account_identifier"):
        return None
    return value


def _source_table_identity(source_ref: Any) -> tuple[int, str] | None:
    """Return an exact logical-page/table key without inspecting cell values."""

    if not isinstance(source_ref, dict):
        return None
    provenance = (
        source_ref.get("geometry_provenance")
        if isinstance(source_ref.get("geometry_provenance"), dict)
        else {}
    )
    table_ids: set[str] = set()
    for owner in (source_ref, provenance):
        if "table_id" not in owner:
            continue
        raw_table_id = owner.get("table_id")
        if not isinstance(raw_table_id, str) or not raw_table_id.strip():
            return None
        table_ids.add(raw_table_id.strip())
    pages: set[int] = set()
    for owner, key in (
        (source_ref, "logical_page"),
        (source_ref, "page"),
        (provenance, "logical_page"),
    ):
        if key not in owner:
            continue
        page = _positive_native_int(owner.get(key))
        if page is None:
            return None
        pages.add(page)
    if len(pages) != 1 or len(table_ids) != 1:
        return None
    return next(iter(pages)), next(iter(table_ids))


def _exact_month_source_table_identity(source_ref: Any) -> tuple[int, str] | None:
    """Accept only value-free, exact source-table month-cell provenance."""

    if not isinstance(source_ref, dict):
        return None
    provenance = source_ref.get("geometry_provenance")
    if not (
        isinstance(provenance, dict)
        and provenance.get("source") == "source_table_geometry"
        and source_ref.get("coordinate_system") == "pdf_points_top_left"
        and provenance.get("coordinate_system") == "pdf_points_top_left"
        and "page" in source_ref
        and "logical_page" in source_ref
        and "logical_page" in provenance
        and isinstance(provenance.get("table_id"), str)
        and bool(str(provenance.get("table_id") or "").strip())
        and provenance.get("calibrated_from_source_table_geometry") is True
        and provenance.get("active_cell_geometry_exact") is True
        and provenance.get("value_inputs_used") is False
        and source_ref.get("geometry_scope") == "cell"
        and _geometry_box({"bbox": source_ref.get("bbox")}) is not None
    ):
        return None
    return _source_table_identity(source_ref)


def _exact_grid_source_table_identity(grid: Any) -> tuple[int, str] | None:
    """Read one exact source-table key from a grid's geometry-only audit."""

    if not isinstance(grid, dict):
        return None
    page = _positive_native_int(grid.get("page"))
    if page is None:
        return None
    audit = grid.get("audit") if isinstance(grid.get("audit"), dict) else {}
    by_page = (
        audit.get("visual_month_geometry_by_page")
        if isinstance(audit.get("visual_month_geometry_by_page"), dict)
        else {}
    )
    provenance_aliases = [
        by_page[key]
        for key in (str(page), page)
        if key in by_page
    ]
    if not provenance_aliases or any(
        candidate != provenance_aliases[0]
        for candidate in provenance_aliases[1:]
    ):
        return None
    provenance = provenance_aliases[0]
    if not (
        page > 0
        and grid.get("coordinate_system") == "pdf_points_top_left"
        and isinstance(provenance, dict)
        and provenance.get("source") == "source_table_geometry"
        and provenance.get("coordinate_system") == "pdf_points_top_left"
        and "logical_page" in provenance
        and isinstance(provenance.get("table_id"), str)
        and bool(str(provenance.get("table_id") or "").strip())
        and provenance.get("calibrated_from_source_table_geometry") is True
        and provenance.get("active_cell_geometry_exact") is True
        and provenance.get("value_inputs_used") is False
    ):
        return None
    return _source_table_identity(
        {
            "logical_page": page,
            "geometry_provenance": provenance,
        }
    )


def _box_contains(outer: list[float], inner: list[float], *, tolerance: float = 1.0) -> bool:
    return bool(
        inner[0] + tolerance >= outer[0]
        and inner[1] + tolerance >= outer[1]
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _exact_grid_month_x_bands(grid: Any) -> dict[int, tuple[float, float]] | None:
    """Require twelve exact, ordered physical month bands for table rescue."""

    if not isinstance(grid, dict) or grid.get("coordinate_system") != "pdf_points_top_left":
        return None
    grid_box = _geometry_box({"bbox": grid.get("bbox")})
    if grid_box is None:
        return None
    month_bands: dict[int, tuple[float, float]] = {}
    for band in grid.get("col_bands") or ():
        if not isinstance(band, dict) or band.get("role") != "month":
            continue
        raw_index = band.get("index")
        raw_header = str(band.get("header") or "").strip()
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or not raw_header.isdigit()
        ):
            return None
        month = raw_index
        if (
            month != int(raw_header)
            or month not in range(1, 13)
            or month in month_bands
            or band.get("geometry_status") != "exact"
            or band.get("geometry_source") != "source_table_geometry"
        ):
            return None
        bbox = _geometry_box({"bbox": band.get("bbox")})
        if (
            bbox is None
            or bbox[0] + 1.0 < grid_box[0]
            or bbox[2] > grid_box[2] + 1.0
        ):
            return None
        month_bands[month] = (bbox[0], bbox[2])
    if set(month_bands) != set(range(1, 13)):
        return None
    ordered = [month_bands[month] for month in range(1, 13)]
    if any(
        left[0] >= right[0] or left[1] > right[0] + 1e-6
        for left, right in zip(ordered, ordered[1:])
    ):
        return None
    return month_bands


def _account_has_exact_anchor_ownership(account: Any) -> bool:
    if not isinstance(account, dict) or account.get("_ownership_status"):
        return False
    segment = account.get("_canonical_segment")
    pages = segment.get("pages") if isinstance(segment, dict) else None
    if not (
        isinstance(segment, dict)
        and segment.get("ownership_basis") == "printed_anchor_to_next_anchor"
        and isinstance(pages, list)
        and pages
    ):
        return False
    anchor_page = _positive_native_int(segment.get("anchor_logical_page"))
    if anchor_page is None:
        return False
    anchor_box = _geometry_box({"bbox": segment.get("anchor_bbox")})
    if anchor_page <= 0 or anchor_box is None:
        return False
    anchor_owned = False
    for page_segment in pages:
        if not isinstance(page_segment, dict):
            return False
        page = _positive_native_int(page_segment.get("logical_page"))
        try:
            minimum = float(page_segment["min_y"])
            maximum = (
                float(page_segment["max_y"])
                if page_segment.get("max_y") is not None
                else None
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not (
            page is not None
            and math.isfinite(minimum)
            and (maximum is None or math.isfinite(maximum) and maximum > minimum)
        ):
            return False
        if (
            page == anchor_page
            and anchor_box[1] + 1.0 >= minimum
            and (maximum is None or anchor_box[1] < maximum)
        ):
            anchor_owned = True
    return anchor_owned


def _monthly_business_signature(record: dict[str, Any]) -> tuple[str, str | None]:
    status = str(record.get("status_code") or record.get("status") or "").strip().upper()
    raw_amount = record.get("overdue_amount")
    if raw_amount in (None, ""):
        raw_amount = record.get("status_amount")
    if raw_amount in (None, ""):
        return status, None
    amount = _number(raw_amount)
    if amount is None:
        amount = f"raw:{str(raw_amount).strip()}"
    elif Decimal(amount) == 0:
        amount = "0"
    return status, amount


def _monthly_record_id(record: dict[str, Any], grid_id: str) -> str | None:
    existing = str(record.get("repayment_id") or record.get("record_id") or "").strip()
    if existing:
        return existing
    calendar_identity = _exact_performance_month(record)
    if calendar_identity is not None:
        year, month = calendar_identity
    elif record.get("performance_month") not in (None, ""):
        return None
    else:
        year = _positive_native_int(record.get("year"))
        month = _positive_native_int(record.get("month"))
        if year is None or month is None:
            return None
    if not grid_id or year < 1900 or not 1 <= month <= 12:
        return None
    return f"{grid_id}:{year:04d}-{month:02d}"


def _merged_source_refs(*records: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    markers: set[str] = set()
    for record in records:
        for ref in record.get("source_cell_refs") or ():
            if not isinstance(ref, dict):
                continue
            marker = repr(sorted(ref.items()))
            if marker in markers:
                continue
            markers.add(marker)
            refs.append(dict(ref))
    return refs


_MONTHLY_CANONICAL_FIELD_ALIASES = {
    "performance_month": frozenset({"performance_month", "year", "month"}),
    "status_code": frozenset({"status", "status_code", "repayment_status_code"}),
    "status_amount": frozenset(
        {"overdue_amount", "status_amount", "source_status_amount"}
    ),
}


def _candidate_b_monthly_position(
    record: Mapping[str, Any],
) -> tuple[str, int, int] | None:
    """Return an exact grid/month identity without accepting coerced aliases."""

    refs = [
        ref
        for ref in record.get("source_cell_refs") or ()
        if isinstance(ref, Mapping)
    ]
    grid_ids = {
        str(value).strip()
        for value in (
            record.get("grid_id"),
            *(ref.get("grid_id") for ref in refs),
        )
        if str(value or "").strip()
    }
    calendar_identity = _exact_performance_month(dict(record))
    if calendar_identity is None or len(grid_ids) != 1:
        return None
    year, month = calendar_identity
    return next(iter(grid_ids)), year, month


def _candidate_b_monthly_observed_position(
    record: Mapping[str, Any],
) -> tuple[str, int, int] | None:
    """Accept an exact native year/month pair when performance_month is absent."""

    exact = _candidate_b_monthly_position(record)
    if exact is not None:
        return exact
    if record.get("performance_month") not in (None, ""):
        return None
    refs = [
        ref
        for ref in record.get("source_cell_refs") or ()
        if isinstance(ref, Mapping)
    ]
    grid_ids = {
        str(value).strip()
        for value in (
            record.get("grid_id"),
            *(ref.get("grid_id") for ref in refs),
        )
        if str(value or "").strip()
    }
    year = _positive_native_int(record.get("year"))
    month = _positive_native_int(record.get("month"))
    if len(grid_ids) != 1 or year is None or month is None:
        return None
    if year < 1900 or not 1 <= month <= 12:
        return None
    return next(iter(grid_ids)), year, month


def _monthly_source_observations(
    records: Iterable[Mapping[str, Any]],
    field_name: str,
) -> list[str]:
    """Return only explicit source observations for one canonical month field."""

    keys = {
        "status_code": ("raw_status", "status_code", "status"),
        "status_amount": (
            "raw_overdue_amount",
            "source_status_amount",
            "status_amount",
            "overdue_amount",
        ),
    }.get(field_name, ())
    observations: set[str] = set()
    for record in records:
        containers = [record]
        containers.extend(
            value
            for key in ("canonical_raw", "raw", "normalized")
            if isinstance((value := record.get(key)), Mapping)
        )
        for container in containers:
            for key in keys:
                value = container.get(key)
                text = str(value).strip() if value is not None else ""
                if text and text.lower() not in {"unknown", "unreadable"}:
                    observations.add(text)
    return sorted(observations)


def _has_observed_monthly_amount(records: Iterable[Mapping[str, Any]]) -> bool:
    """Distinguish a visible amount cell from inferred/default amount aliases."""

    amount_aliases = _MONTHLY_CANONICAL_FIELD_ALIASES["status_amount"]
    for record in records:
        for ref in record.get("source_cell_refs") or ():
            if (
                isinstance(ref, Mapping)
                and str(ref.get("field_name") or "").strip() in amount_aliases
            ):
                return True
        raw = record.get("canonical_raw")
        if isinstance(raw, Mapping) and any(
            raw.get(field_name) not in (None, "") for field_name in amount_aliases
        ):
            return True
    return False


def _monthly_grid_source_ref(
    grid: Mapping[str, Any] | None,
    *,
    grid_id: str,
    year: int,
    month: int,
    field_name: str,
) -> dict[str, Any]:
    """Build a value-free locator for one exact printed grid/month position."""

    grid = grid if isinstance(grid, Mapping) else {}
    page = _positive_native_int(grid.get("page"))
    bbox: list[float] | None = None
    for band in grid.get("col_bands") or ():
        if not isinstance(band, Mapping):
            continue
        header = str(band.get("header") or "").strip()
        index = band.get("index")
        if header == str(month) or (isinstance(index, int) and index == month):
            bbox = _geometry_box({"bbox": band.get("bbox")})
            if bbox is not None:
                break
    if bbox is None:
        bbox = _geometry_box({"bbox": grid.get("bbox")}) or _geometry_box(dict(grid))
    ref: dict[str, Any] = {
        "source": "candidate_b_monthly_grid_omission",
        "grid_id": grid_id,
        "performance_month": f"{year:04d}-{month:02d}",
        "field_name": field_name,
        "geometry_scope": "grid",
    }
    if page is not None:
        ref.update({"page": page, "logical_page": page})
    if bbox is not None:
        ref["bbox"] = bbox
    if grid.get("coordinate_system"):
        ref["coordinate_system"] = grid["coordinate_system"]
    identity = _exact_grid_source_table_identity(dict(grid))
    if identity is not None:
        ref["table_id"] = identity[1]
    return ref


def _localized_monthly_source_refs(
    records: Iterable[Mapping[str, Any]],
    grid: Mapping[str, Any] | None,
    *,
    grid_id: str,
    year: int,
    month: int,
    field_name: str,
) -> list[dict[str, Any]]:
    """Retarget exact source refs to one canonical monthly field."""

    records = list(records)
    aliases = _MONTHLY_CANONICAL_FIELD_ALIASES[field_name]
    refs: list[dict[str, Any]] = []
    fallback_refs: list[dict[str, Any]] = []
    for record in records:
        for raw_ref in record.get("source_cell_refs") or ():
            if not isinstance(raw_ref, Mapping):
                continue
            ref_month = raw_ref.get("performance_month")
            if ref_month not in (None, "", f"{year:04d}-{month:02d}"):
                continue
            raw_col = raw_ref.get("col")
            if (
                isinstance(raw_col, int)
                and not isinstance(raw_col, bool)
                and 1 <= raw_col <= 12
                and raw_col != month
            ):
                continue
            source_field = str(raw_ref.get("field_name") or "").strip()
            ref = dict(raw_ref)
            if source_field and source_field != field_name:
                ref["source_field_name"] = source_field
            ref.update(
                {
                    "grid_id": grid_id,
                    "performance_month": f"{year:04d}-{month:02d}",
                    "field_name": field_name,
                }
            )
            fallback_refs.append(ref)
            if source_field in aliases:
                refs.append(ref)
    if not refs and field_name in {"performance_month", "status_code"}:
        refs = fallback_refs
    if not refs:
        refs = [
            _monthly_grid_source_ref(
                grid,
                grid_id=grid_id,
                year=year,
                month=month,
                field_name=field_name,
            )
        ]
    unique: dict[str, dict[str, Any]] = {}
    for ref in refs:
        unique.setdefault(repr(sorted(ref.items())), ref)
    return list(unique.values())


def _has_exact_month_source_ref(
    records: Iterable[Mapping[str, Any]],
    *,
    grid_id: str,
    year: int,
    month: int,
) -> bool:
    """Require a bounded source cell for source-diff field localization."""

    month_label = f"{year:04d}-{month:02d}"
    for record in records:
        if _candidate_b_monthly_observed_position(record) != (grid_id, year, month):
            continue
        for ref in record.get("source_cell_refs") or ():
            if not isinstance(ref, Mapping):
                continue
            ref_grid_id = str(ref.get("grid_id") or "").strip()
            if ref_grid_id and ref_grid_id != grid_id:
                continue
            ref_month = str(ref.get("performance_month") or "").strip()
            raw_col = ref.get("col")
            month_matches = ref_month == month_label or (
                isinstance(raw_col, int)
                and not isinstance(raw_col, bool)
                and raw_col == month
            )
            bbox = _geometry_box({"bbox": ref.get("bbox")})
            if month_matches and bbox is not None and ref.get("geometry_scope") == "cell":
                return True
    return False


def _owned_grid_month_source_ref(
    refs: Iterable[Mapping[str, Any]],
    *,
    account_id: str,
    grid_id: str,
    year: int,
    month: int,
) -> dict[str, Any] | None:
    """Return one closed-vocabulary cell ref for an exact owned grid/month.

    The stable account/month target is stronger than a grid-local diagnostic.
    It therefore accepts only a physical native table cell with complete row,
    column, page, table, bbox, and evidence identity.  The output uses one
    dedicated derived ``source`` and binding so downstream Community checks do
    not need to trust the producer-specific source alias.
    """

    month_label = f"{year:04d}-{month:02d}"
    candidates: list[dict[str, Any]] = []
    for raw_ref in refs:
        if not isinstance(raw_ref, Mapping):
            continue
        source = str(raw_ref.get("source") or "").strip()
        binding = str(raw_ref.get("binding") or "").strip()
        binding_quality = str(raw_ref.get("binding_quality") or "").strip()
        ref_grid_id = str(raw_ref.get("grid_id") or "").strip()
        ref_month = str(raw_ref.get("performance_month") or "").strip()
        raw_column = raw_ref.get("column", raw_ref.get("col"))
        column = (
            raw_column
            if isinstance(raw_column, int)
            and not isinstance(raw_column, bool)
            and raw_column >= 0
            else None
        )
        raw_row = raw_ref.get("row")
        row = (
            raw_row
            if isinstance(raw_row, int)
            and not isinstance(raw_row, bool)
            and raw_row >= 0
            else None
        )
        raw_evidence_ids = raw_ref.get("evidence_ids")
        evidence_ids = (
            [value.strip() for value in raw_evidence_ids]
            if isinstance(raw_evidence_ids, (list, tuple))
            and raw_evidence_ids
            and all(
                isinstance(value, str) and value.strip()
                for value in raw_evidence_ids
            )
            and len(set(raw_evidence_ids)) == len(raw_evidence_ids)
            else []
        )
        logical_page = _positive_native_int(
            raw_ref.get("logical_page") or raw_ref.get("page")
        )
        source_page = _positive_native_int(raw_ref.get("source_page"))
        if not (
            source in _OWNED_GRID_MONTHLY_INPUT_SOURCES
            and binding in _OWNED_GRID_MONTHLY_INPUT_BINDINGS
            and binding_quality in _OWNED_GRID_MONTHLY_INPUT_BINDINGS
            and ref_grid_id == grid_id
            and (ref_month == month_label or (not ref_month and column == month))
            and str(raw_ref.get("geometry_scope") or "") == "cell"
            and logical_page is not None
            and source_page is not None
            and str(raw_ref.get("table_id") or "").strip()
            and row is not None
            and column is not None
            and _geometry_box({"bbox": raw_ref.get("bbox")}) is not None
            and evidence_ids
        ):
            continue
        source_field = str(raw_ref.get("field_name") or "").strip()
        ref = dict(raw_ref)
        ref.update(
            {
                "source": _OWNED_GRID_MONTHLY_REF_SOURCE,
                "source_origin": source,
                "account_id": account_id,
                "grid_id": grid_id,
                "performance_month": month_label,
                "field_name": "performance_month",
                "page": logical_page,
                "logical_page": logical_page,
                "source_page": source_page,
                "row": row,
                "column": column,
                "evidence_ids": evidence_ids,
                "binding": "source_account_month_identity",
                "binding_quality": "source_account_month_identity",
            }
        )
        if source_field and source_field != "performance_month":
            ref["source_field_name"] = source_field
        candidates.append(ref)
    if not candidates:
        return None
    unique_cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    for ref in candidates:
        bbox = tuple(float(value) for value in ref["bbox"])
        key = (
            str(ref.get("source_origin") or ""),
            int(ref["logical_page"]),
            int(ref["source_page"]),
            str(ref["table_id"]),
            int(ref["row"]),
            int(ref["column"]),
            bbox,
            tuple(sorted(ref["evidence_ids"])),
        )
        unique_cells.setdefault(key, ref)
    if len(unique_cells) != 1:
        return None
    return next(iter(unique_cells.values()))


def report_localized_monthly_omissions(
    issue_context: Any,
    *,
    issue_code: str,
    message: str,
    parser_stage: str,
    grid_id: str,
    months: Iterable[tuple[int, int]],
    account_id: str | None = None,
    account_month_identity_proven: bool = False,
    source_records: Iterable[Mapping[str, Any]] = (),
    grid: Mapping[str, Any] | None = None,
    reason_codes: Iterable[str] = (),
    observed_context: Mapping[str, Any] | None = None,
    require_exact_cell_ref: bool = False,
) -> int:
    """Report one exact issue per withheld canonical month field.

    This is diagnostic-only.  It never materializes a monthly row, owner, or
    status value, and it deliberately emits ``status_amount`` only when an
    explicit source amount observation belongs to that exact grid/month.  A
    caller may promote the diagnostic to canonical ``(account_id, month)``
    identity only after proving the printed account anchor, month range, grid
    geometry, and unique owner.  Otherwise the issue deliberately remains a
    grid-local, non-denominator source-position diagnostic.
    """

    if issue_context is None or not grid_id:
        return 0
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    records_by_month: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for record in source_records:
        if not isinstance(record, Mapping):
            continue
        position = _candidate_b_monthly_observed_position(record)
        if position is None or position[0] != grid_id:
            continue
        records_by_month.setdefault(position[1:], []).append(record)

    existing_targets = {
        (
            str(issue.get("target_record_id") or ""),
            {
                "status": "status_code",
                "repayment_status_code": "status_code",
                "overdue_amount": "status_amount",
            }.get(str(issue.get("field_name") or ""), str(issue.get("field_name") or "")),
        )
        for issue in getattr(issue_context, "_personal_detail_extraction_issues", ())
        if isinstance(issue, Mapping)
        and str(issue.get("target_dataset") or "")
        in {"repayment_records", "credit_account_monthly_performance"}
    }
    exact_account_id = str(account_id or "").strip()
    if not account_month_identity_proven:
        exact_account_id = ""
    emitted = 0
    for year, month in sorted(set(months)):
        if year < 1900 or not 1 <= month <= 12:
            continue
        month_label = f"{year:04d}-{month:02d}"
        month_records = records_by_month.get((year, month), [])
        if require_exact_cell_ref and not _has_exact_month_source_ref(
            month_records,
            grid_id=grid_id,
            year=year,
            month=month,
        ):
            continue
        owned_ref = None
        if exact_account_id:
            owned_ref = _owned_grid_month_source_ref(
                (
                    ref
                    for record in month_records
                    for ref in record.get("source_cell_refs") or ()
                    if isinstance(ref, Mapping)
                ),
                account_id=exact_account_id,
                grid_id=grid_id,
                year=year,
                month=month,
            )
        month_account_id = exact_account_id if owned_ref is not None else ""
        target_record_id = (
            _source_account_month_record_id(month_account_id, year, month)
            if month_account_id
            else f"{grid_id}:{month_label}"
        )
        fields = ["performance_month"] if month_account_id else [
            "performance_month",
            "status_code",
        ]
        if not month_account_id and _has_observed_monthly_amount(month_records):
            fields.append("status_amount")
        for field_name in fields:
            target = (target_record_id, field_name)
            if target in existing_targets:
                continue
            observations = (
                [month_label]
                if field_name == "performance_month"
                else _monthly_source_observations(month_records, field_name)
            )
            if month_account_id:
                observed_value = {
                    "account_id": month_account_id,
                    "performance_month": month_label,
                }
                localized_refs = [owned_ref]
                effective_issue_code = _OWNED_GRID_MONTHLY_OMISSION_CODE
            else:
                observed_value = {
                    "grid_id": grid_id,
                    "performance_month": month_label,
                    "field_state": "source_position_withheld",
                }
                if observations:
                    observed_value["source_observations"] = observations
                if observed_context:
                    observed_value.update(dict(observed_context))
                localized_refs = _localized_monthly_source_refs(
                    month_records,
                    grid,
                    grid_id=grid_id,
                    year=year,
                    month=month,
                    field_name=field_name,
                )
                effective_issue_code = issue_code
            record_issue(
                issue_context,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code=effective_issue_code,
                    message=message,
                    parser_stage=parser_stage,
                    target_dataset="repayment_records",
                    target_record_id=target_record_id,
                    field_name=field_name,
                    observed_value=observed_value,
                    candidate_value={"resolution": "withheld_pending_review"},
                    source_refs=localized_refs,
                    reason_codes=(
                        *reason_codes,
                        (
                            "exact_account_month_identity"
                            if month_account_id
                            else "exact_grid_month_source_position"
                        ),
                        "normalized_value_withheld",
                        "owner_or_status_value_not_invented",
                    ),
                ),
            )
            existing_targets.add(target)
            emitted += 1
    return emitted


def _resolve_reconciled_monthly_source_diagnostics(
    issue_context: Any,
    *,
    grid_id: str,
    month_label: str,
    alias_refs: Iterable[Mapping[str, Any]],
) -> None:
    """Close only the exact detached field trio covered by one alias proof."""

    issues = getattr(issue_context, "_personal_detail_extraction_issues", None)
    if not isinstance(issues, list):
        return

    def physical_signature(ref: Mapping[str, Any]) -> tuple[Any, ...] | None:
        role = str(ref.get("source_field_name") or ref.get("field_name") or "")
        bbox = ref.get("bbox")
        page = ref.get("page")
        logical_page = ref.get("logical_page")
        row = ref.get("row")
        column = ref.get("col")
        if not (
            role in {"status", "overdue_amount"}
            and str(ref.get("grid_id") or "") == grid_id
            and str(ref.get("performance_month") or "") == month_label
            and isinstance(page, int)
            and not isinstance(page, bool)
            and page > 0
            and logical_page == page
            and isinstance(row, int)
            and not isinstance(row, bool)
            and row >= 0
            and isinstance(column, int)
            and not isinstance(column, bool)
            and column == int(month_label[5:7])
            and isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in bbox
            )
            and float(bbox[2]) > float(bbox[0])
            and float(bbox[3]) > float(bbox[1])
        ):
            return None
        return (
            role,
            page,
            logical_page,
            row,
            column,
            tuple(round(float(value), 6) for value in bbox),
        )

    raw_alias_refs = list(alias_refs)
    parsed_alias_signatures = [
        physical_signature(ref) if isinstance(ref, Mapping) else None
        for ref in raw_alias_refs
    ]
    if len(raw_alias_refs) != 2 or any(
        signature is None for signature in parsed_alias_signatures
    ):
        return
    alias_signatures = {
        signature
        for signature in parsed_alias_signatures
        if signature is not None
    }
    if len(alias_signatures) != 2 or {
        signature[0] for signature in alias_signatures
    } != {"status", "overdue_amount"}:
        return

    target_record_id = f"{grid_id}:{month_label}"
    expected_roles = {
        "performance_month": {"status", "overdue_amount"},
        "status_code": {"status"},
        "status_amount": {"overdue_amount"},
    }
    expected_reason_codes = frozenset(
        {
            "detached_source_structure_exact_key",
            "canonical_deduplicated_key_missing",
            "source_structure_is_audit_only",
            "account_month_owner_reconciliation_pending",
            "dataset_incomplete",
            "exact_grid_month_source_position",
            "normalized_value_withheld",
            "owner_or_status_value_not_invented",
        }
    )
    matching: dict[str, dict[str, Any]] = {}
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        observed = issue.get("observed_value")
        candidate = issue.get("candidate_value")
        field_name = str(issue.get("field_name") or "")
        reason_codes = issue.get("reason_codes")
        source_refs = issue.get("source_refs")
        expected_ref_count = 2 if field_name == "performance_month" else 1
        if not (
            str(issue.get("category") or "") == "ocr_structure_correction"
            and issue.get("issue_code")
            == "canonical_monthly_source_structure_missing_field"
            and str(issue.get("status") or "") == "requires_review"
            and str(issue.get("parser_stage") or "")
            == "canonical_monthly_grid_materialization"
            and str(issue.get("target_dataset") or "") == "repayment_records"
            and str(issue.get("target_record_id") or "") == target_record_id
            and field_name in expected_roles
            and field_name not in matching
            and isinstance(observed, Mapping)
            and str(observed.get("grid_id") or "") == grid_id
            and str(observed.get("performance_month") or "") == month_label
            and str(observed.get("field_state") or "")
            == "source_position_withheld"
            and isinstance(candidate, Mapping)
            and candidate == {"resolution": "withheld_pending_review"}
            and isinstance(reason_codes, (list, tuple))
            and len(reason_codes) == len(expected_reason_codes)
            and frozenset(reason_codes) == expected_reason_codes
            and isinstance(source_refs, (list, tuple))
            and len(source_refs) == expected_ref_count
        ):
            continue
        parsed_signatures = [
            physical_signature(ref) if isinstance(ref, Mapping) else None
            for ref in source_refs
        ]
        if any(signature is None for signature in parsed_signatures):
            continue
        signatures = {
            signature for signature in parsed_signatures if signature is not None
        }
        if (
            signatures
            != {
                signature
                for signature in alias_signatures
                if signature[0] in expected_roles[field_name]
            }
        ):
            continue
        matching[field_name] = issue

    if set(matching) != set(expected_roles):
        return
    for issue in matching.values():
        issue["status"] = "resolved"


def _report_reconciled_monthly_source_alias(
    issue_context: Any,
    *,
    account_id: str,
    year: int,
    month: int,
    grid_id: str,
    source_records: Iterable[Mapping[str, Any]],
    grid: Mapping[str, Any] | None,
    linkage_basis: str,
) -> None:
    """Keep one source-position alias visible without double-counting it.

    A second exact printed position may resolve to an account/month identity
    already represented by another grid.  It is source-audit evidence, not a
    second canonical month.  Publish an informational, source-localized issue
    so the raw plane remains conserved while canonical closure stays set based.
    """

    if issue_context is None or not account_id or not grid_id:
        return
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    month_label = f"{year:04d}-{month:02d}"
    localized_refs = _localized_monthly_source_refs(
        source_records,
        grid,
        grid_id=grid_id,
        year=year,
        month=month,
        field_name="performance_month",
    )
    if not localized_refs:
        return
    localized_refs = [
        {
            **ref,
            "account_id": account_id,
            "performance_month": month_label,
            "field_name": "performance_month",
            "binding": str(ref.get("binding") or "source_account_month_alias"),
            "binding_quality": str(
                ref.get("binding_quality") or "source_account_month_alias"
            ),
        }
        for ref in localized_refs
    ]
    record_issue(
        issue_context,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_monthly_source_position_alias_reconciled",
            message=(
                "An exact printed grid/month position resolved to an already represented "
                "account-month identity and was retained as an audit-only alias."
            ),
            severity="info",
            status="informational",
            parser_stage="candidate_b_relationship_schema",
            target_dataset="repayment_records",
            target_record_id=_source_account_month_record_id(
                account_id, year, month
            ),
            field_name="performance_month",
            observed_value={
                "account_id": account_id,
                "grid_id": grid_id,
                "performance_month": month_label,
                "source_position_state": "owner_bound_alias",
                "account_month_owner_basis": linkage_basis or None,
            },
            candidate_value={
                "resolution": "reconciled_to_existing_account_month_identity"
            },
            source_refs=localized_refs,
            reason_codes=(
                "exact_account_month_identity",
                "distinct_source_position_alias",
                "canonical_identity_not_double_counted",
                "source_position_audit_preserved",
            ),
        ),
    )
    _resolve_reconciled_monthly_source_diagnostics(
        issue_context,
        grid_id=grid_id,
        month_label=month_label,
        alias_refs=localized_refs,
    )


def _account_segments(accounts: list[dict[str, Any]]) -> dict[str, list[tuple[int, float, float | None]]]:
    segments: dict[str, list[tuple[int, float, float | None]]] = {}
    fallback: dict[int, list[tuple[float, str]]] = {}
    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_id = str(account.get("account_id") or "")
        if not account_id:
            continue
        canonical = account.get("_canonical_segment")
        pages = canonical.get("pages") if isinstance(canonical, dict) else None
        exact: list[tuple[int, float, float | None]] = []
        if isinstance(pages, list):
            for item in pages:
                if not isinstance(item, dict):
                    continue
                try:
                    page = int(item.get("logical_page") or 0)
                    minimum = float(item["min_y"])
                    maximum = float(item["max_y"]) if item.get("max_y") is not None else None
                except (KeyError, TypeError, ValueError):
                    continue
                if page > 0:
                    exact.append((page, minimum, maximum))
        if exact:
            segments[account_id] = exact
            continue
        refs = [ref for ref in account.get("source_refs") or () if isinstance(ref, dict)]
        first = refs[0] if refs else {}
        bbox = account.get("bbox") or first.get("bbox")
        try:
            page = int(account.get("page") or first.get("logical_page") or first.get("page") or 0)
            minimum = float(bbox[1])
        except (TypeError, ValueError, IndexError):
            continue
        if page > 0:
            fallback.setdefault(page, []).append((minimum, account_id))
    for page, values in fallback.items():
        ordered = sorted(values)
        for index, (minimum, account_id) in enumerate(ordered):
            maximum = ordered[index + 1][0] if index + 1 < len(ordered) else None
            segments[account_id] = [(page, minimum, maximum)]
    return segments


def candidate_b_repayment_anchor_ledger(
    evidence_pages: list[dict[str, Any]],
    accounts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ledger visible monthly anchors without relying on grid materialization."""

    account_segments = _account_segments(accounts)
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for page_data in evidence_pages:
        if not isinstance(page_data, dict):
            continue
        page = int(page_data.get("page") or 0)
        source_page = int(page_data.get("source_page") or page)
        lines = [line for line in page_data.get("lines") or () if isinstance(line, dict)]
        for index, line in enumerate(lines):
            text = str(line.get("text") or line.get("content") or "").strip()
            if "还款记录" not in re.sub(r"\s+", "", text):
                continue
            bbox = line.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                box = [float(value) for value in bbox]
                top = box[1]
            except (TypeError, ValueError):
                continue
            source_lines = [line]
            date_range = _printed_repayment_range(text)
            if date_range is None:
                for neighbor_index in (index - 1, index + 1):
                    if not 0 <= neighbor_index < len(lines):
                        continue
                    neighbor = lines[neighbor_index]
                    neighbor_text = str(neighbor.get("text") or neighbor.get("content") or "").strip()
                    neighbor_box = neighbor.get("bbox")
                    if not isinstance(neighbor_box, list) or len(neighbor_box) != 4:
                        continue
                    try:
                        nearby = abs(float(neighbor_box[1]) - top) <= 32.0
                    except (TypeError, ValueError):
                        nearby = False
                    combined = (
                        f"{neighbor_text} {text}"
                        if neighbor_index < index
                        else f"{text} {neighbor_text}"
                    )
                    combined_range = _printed_repayment_range(combined)
                    if nearby and combined_range is not None:
                        text = combined
                        date_range = combined_range
                        source_lines.append(neighbor)
                        box = [
                            min(box[0], float(neighbor_box[0])),
                            min(box[1], float(neighbor_box[1])),
                            max(box[2], float(neighbor_box[2])),
                            max(box[3], float(neighbor_box[3])),
                        ]
                        top = box[1]
                        break
            owner_segments = [
                (account_id, minimum, maximum)
                for account_id, segments in account_segments.items()
                for segment_page, minimum, maximum in segments
                if segment_page == page
                    and top + 8.0 >= minimum
                    and (maximum is None or top < maximum)
            ]
            if len(owner_segments) != 1:
                continue
            account_id, segment_minimum, segment_maximum = owner_segments[0]
            marker = (account_id, page, round(top), re.sub(r"\s+", "", text))
            if marker in seen:
                continue
            seen.add(marker)
            evidence_ids = {
                str(value).strip()
                for value in (
                    *(
                        evidence_id
                        for source_line in source_lines
                        for evidence_id in source_line.get("evidence_ids") or ()
                    ),
                    *(
                        token_id
                        for source_line in source_lines
                        for token_id in source_line.get("token_ids") or ()
                    ),
                )
                if str(value or "").strip()
            }
            anchors.append(
                {
                    "anchor_id": stable_record_id("monthly_repayment_anchor", account_id, page, round(top), text),
                    "account_id": account_id,
                    "page": page,
                    "source_page": source_page,
                    "bbox": box,
                    "anchor_text": text,
                    "date_range": date_range,
                    "segment_min_y": segment_minimum,
                    "segment_max_y": segment_maximum,
                    "source_refs": [
                        {
                            "source": "candidate_b_monthly_anchor_ledger",
                            "logical_page": page,
                            "source_page": source_page,
                            "bbox": box,
                            "geometry_scope": "line",
                            **(
                                {"evidence_ids": sorted(evidence_ids)}
                                if evidence_ids
                                else {}
                            ),
                            **(
                                {
                                    "line_ids": sorted(
                                        {
                                            str(source_line["line_id"])
                                            for source_line in source_lines
                                            if source_line.get("line_id")
                                        }
                                    )
                                }
                                if any(source_line.get("line_id") for source_line in source_lines)
                                else {}
                            ),
                        }
                    ],
                }
            )
    return anchors


def _exact_repayment_anchor_source_ref(anchor: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the single immutable line ref that proves an account-bound range."""

    account_id = str(anchor.get("account_id") or "").strip()
    page = _positive_native_int(anchor.get("page"))
    source_page = _positive_native_int(anchor.get("source_page"))
    refs = [ref for ref in anchor.get("source_refs") or () if isinstance(ref, Mapping)]
    if not account_id or page is None or source_page is None or len(refs) != 1:
        return None
    raw_ref = refs[0]
    ref_page = _positive_native_int(raw_ref.get("logical_page"))
    ref_source_page = _positive_native_int(raw_ref.get("source_page"))
    bbox = _geometry_box({"bbox": raw_ref.get("bbox")})
    evidence_ids = {
        str(value).strip()
        for value in raw_ref.get("evidence_ids") or ()
        if str(value or "").strip()
    }
    if (
        raw_ref.get("source") != "candidate_b_monthly_anchor_ledger"
        or raw_ref.get("geometry_scope") != "line"
        or ref_page != page
        or ref_source_page != source_page
        or bbox is None
        or not evidence_ids
    ):
        return None
    return {**dict(raw_ref), "evidence_ids": sorted(evidence_ids), "bbox": bbox}


def _source_account_month_record_id(account_id: str, year: int, month: int) -> str:
    """Stable typed identity for a printed account/month with no parser grid."""

    account_key = stable_record_id("source_account_month_owner", account_id).split(":", 1)[-1]
    return f"source_account_month:{account_key}:{year:04d}-{month:02d}"


def _account_month_identity_proof(
    account: Mapping[str, Any] | None,
    grid: Mapping[str, Any] | None,
    *,
    linkage_basis: str,
    year: int,
    month: int,
) -> dict[str, Any] | None:
    """Prove one canonical account/month without relying on parser aliases."""

    if not (
        isinstance(account, dict)
        and isinstance(grid, Mapping)
        and _account_has_exact_anchor_ownership(account)
    ):
        return None
    account_id = str(account.get("account_id") or "").strip()
    grid_id = str(grid.get("grid_id") or "").strip()
    grid_page = _positive_native_int(grid.get("page"))
    grid_box = _geometry_box(grid)
    printed_months = _grid_months(dict(grid))
    if (
        not account_id
        or not grid_id
        or grid_page is None
        or grid_box is None
        or (year, month) not in printed_months
        or linkage_basis
        not in {
            "canonical_account_segment",
            "explicit_account_id_confirmed_by_canonical_segment",
            "exact_source_table_account_owner",
        }
    ):
        return None
    segments = _account_segments([dict(account)]).get(account_id, [])
    segment_contains_grid = any(
        page == grid_page
        and grid_box[1] + 8.0 >= minimum
        and (maximum is None or grid_box[1] < maximum)
        for page, minimum, maximum in segments
    )
    if not segment_contains_grid and linkage_basis != "exact_source_table_account_owner":
        return None
    return {
        "account_id": account_id,
        "performance_month": f"{year:04d}-{month:02d}",
        "grid_id": grid_id,
        "owner_basis": linkage_basis,
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }


def _report_account_range_monthly_omissions(
    issue_context: Any,
    *,
    anchor: Mapping[str, Any],
    missing_months: Iterable[tuple[int, int]],
) -> int:
    """Report exact range-derived identities while keeping status unknown."""

    if issue_context is None:
        return 0
    account_id = str(anchor.get("account_id") or "").strip()
    anchor_id = str(anchor.get("anchor_id") or "").strip()
    exact_ref = _exact_repayment_anchor_source_ref(anchor)
    if not account_id or not anchor_id or exact_ref is None:
        return 0
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    existing_targets = {
        (str(issue.get("target_record_id") or ""), str(issue.get("field_name") or ""))
        for issue in getattr(issue_context, "_personal_detail_extraction_issues", ())
        if isinstance(issue, Mapping)
    }
    emitted = 0
    for year, month in sorted(set(missing_months)):
        if year < 1900 or not 1 <= month <= 12:
            continue
        month_label = f"{year:04d}-{month:02d}"
        target_record_id = _source_account_month_record_id(account_id, year, month)
        common_observed = {
            "source_identity_type": "account_month_from_printed_repayment_range",
            "account_id": account_id,
            "performance_month": month_label,
            "anchor_id": anchor_id,
            "anchor_text": anchor.get("anchor_text"),
        }
        field_contracts = (
            (
                "performance_month",
                "candidate_b_monthly_account_range_missing_month",
                "An exact account-bound printed repayment range proves this month identity, but no canonical monthly row was emitted.",
                "source_account_month_range",
                "printed_range_proves_performance_month",
            ),
            (
                "status_code",
                "candidate_b_monthly_account_range_status_grid_unavailable",
                "The account/month identity is proven by a printed repayment range, but its status grid/cell was not reconstructed; no status was invented.",
                "source_account_month_identity",
                "status_source_grid_unavailable",
            ),
        )
        for field_name, issue_code, message, binding, reason in field_contracts:
            target = (target_record_id, field_name)
            if target in existing_targets:
                continue
            ref = {
                **exact_ref,
                "account_id": account_id,
                "performance_month": month_label,
                "field_name": field_name,
                "binding": binding,
                "binding_quality": binding,
            }
            record_issue(
                issue_context,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code=issue_code,
                    message=message,
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    target_record_id=target_record_id,
                    field_name=field_name,
                    observed_value={
                        **common_observed,
                        "field_state": "source_position_withheld",
                    },
                    candidate_value={"resolution": "withheld_pending_review"},
                    source_refs=(ref,),
                    reason_codes=(
                        "independent_visible_repayment_anchor",
                        "exact_account_segment_owner",
                        "exact_printed_date_range",
                        "source_account_month_identity",
                        reason,
                        "normalized_value_withheld",
                        "monthly_status_value_not_invented",
                    ),
                ),
            )
            existing_targets.add(target)
            emitted += 1
    return emitted


def link_candidate_b_repayments(
    repayments: list[dict[str, Any]],
    accounts: list[dict[str, Any]],
    micro_grids: list[dict[str, Any]],
    *,
    reading_order_by_logical: dict[int, int] | None = None,
    issue_context: Any | None = None,
    repayment_anchors: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Attach monthly grids only to their exact canonical account segment."""
    del reading_order_by_logical
    grids: dict[str, dict[str, Any]] = {}
    duplicate_grid_ids: set[str] = set()
    for grid in micro_grids:
        if not isinstance(grid, dict):
            continue
        grid_id = str(grid.get("grid_id") or "")
        if not grid_id:
            continue
        if grid_id in grids:
            duplicate_grid_ids.add(grid_id)
            continue
        grids[grid_id] = grid
    source_structure_grids: dict[str, dict[str, Any]] = {}
    duplicate_source_structure_grid_ids: set[str] = set()
    for grid in getattr(
        issue_context,
        "_candidate_b_monthly_source_structure_grids",
        (),
    ):
        if not isinstance(grid, dict):
            continue
        grid_id = str(grid.get("grid_id") or "").strip()
        if not grid_id:
            continue
        if grid_id in source_structure_grids:
            duplicate_source_structure_grid_ids.add(grid_id)
            continue
        source_structure_grids[grid_id] = grid
    source_structure_records = [
        record
        for record in getattr(
            issue_context,
            "_candidate_b_monthly_source_structure_records",
            (),
        )
        if isinstance(record, dict)
    ]
    valid_ids: set[str] = set()
    account_segments: dict[str, list[tuple[int, float, float | None]]] = {}
    fallback_anchors_by_page: dict[int, list[tuple[float, str]]] = {}
    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_id = str(account.get("account_id") or "")
        if account_id:
            valid_ids.add(account_id)
        if account.get("_ownership_status"):
            continue
        canonical_segment = account.get("_canonical_segment")
        segment_pages = canonical_segment.get("pages") if isinstance(canonical_segment, dict) else None
        exact_segments: list[tuple[int, float, float | None]] = []
        if isinstance(segment_pages, list):
            for page_segment in segment_pages:
                if not isinstance(page_segment, dict):
                    continue
                page = int(page_segment.get("logical_page") or 0)
                minimum = page_segment.get("min_y")
                maximum = page_segment.get("max_y")
                if page <= 0 or minimum is None:
                    continue
                try:
                    exact_segments.append(
                        (
                            page,
                            float(minimum),
                            float(maximum) if maximum is not None else None,
                        )
                    )
                except (TypeError, ValueError):
                    continue
        if account_id and exact_segments:
            account_segments[account_id] = exact_segments
            continue

        # Synthetic callers and old serialized candidates may lack the private
        # segment descriptor.  A same-page printed-anchor interval can still be
        # reconstructed, but table-only partial observations are not anchors.
        if account.get("_ownership_status"):
            continue
        refs = [ref for ref in account.get("source_refs") or () if isinstance(ref, dict)]
        anchor_ref = next(
            (ref for ref in refs if ref.get("source") == "candidate_b_account_anchor"),
            refs[0] if refs else {},
        )
        page = int(account.get("page") or anchor_ref.get("logical_page") or anchor_ref.get("page") or 0)
        bbox = account.get("bbox") or anchor_ref.get("bbox")
        if page <= 0 or not isinstance(bbox, list) or len(bbox) != 4 or not account_id:
            continue
        try:
            top = float(bbox[1])
        except (TypeError, ValueError):
            continue
        fallback_anchors_by_page.setdefault(page, []).append((top, account_id))

    for page, anchors in fallback_anchors_by_page.items():
        ordered = sorted(anchors)
        for index, (top, account_id) in enumerate(ordered):
            upper = ordered[index + 1][0] if index + 1 < len(ordered) else None
            account_segments[account_id] = [(page, top, upper)]

    linked: list[dict[str, Any]] = []
    reported_unresolved_grids: set[str] = set()
    owner_by_grid: dict[str, tuple[dict[str, Any] | None, str]] = {}
    accounts_by_id = {
        str(account.get("account_id") or ""): account
        for account in accounts
        if isinstance(account, dict) and account.get("account_id")
    }
    account_table_owners: dict[
        tuple[int, str],
        list[tuple[dict[str, Any], list[list[float]]]],
    ] = {}
    conflicting_account_table_geometry: set[tuple[int, str]] = set()
    invalid_account_table_geometry: set[tuple[int, str]] = set()
    for account in accounts:
        if not (
            isinstance(account, dict)
            and account.get("account_id")
            and _account_has_exact_anchor_ownership(account)
        ):
            continue
        boxes_by_table: dict[tuple[int, str], list[list[float]]] = {}
        invalid_table_ref = False
        for source_ref in account.get("source_refs") or ():
            if not (
                isinstance(source_ref, dict)
                and source_ref.get("source") == "native_detail_table"
                and source_ref.get("geometry_scope") == "table"
            ):
                continue
            identity = _source_table_identity(source_ref)
            bbox = _geometry_box({"bbox": source_ref.get("bbox")})
            if (
                identity is None
                or bbox is None
                or source_ref.get("coordinate_system") != "pdf_points_top_left"
            ):
                if identity is not None:
                    invalid_account_table_geometry.add(identity)
                invalid_table_ref = True
                break
            boxes_by_table.setdefault(identity, []).append(bbox)
        if invalid_table_ref:
            continue
        for identity, boxes in boxes_by_table.items():
            first_box = boxes[0]
            if any(
                any(abs(left - right) > 1e-6 for left, right in zip(first_box, box, strict=True))
                for box in boxes[1:]
            ):
                conflicting_account_table_geometry.add(identity)
                continue
            account_table_owners.setdefault(identity, []).append((account, boxes))
    explicit_ids_by_grid: dict[str, set[str]] = {}
    observed_explicit_ids_by_grid: dict[str, set[str]] = {}
    source_candidates_by_grid: dict[str, list[dict[str, Any]]] = {}
    for record in repayments:
        refs = record.get("source_cell_refs") if isinstance(record.get("source_cell_refs"), list) else []
        first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
        grid_id = str(record.get("grid_id") or first_ref.get("grid_id") or "")
        if grid_id:
            source_candidates_by_grid.setdefault(grid_id, []).append(record)
        explicit_id = str(record.get("account_id") or "")
        if grid_id and explicit_id:
            observed_explicit_ids_by_grid.setdefault(grid_id, set()).add(explicit_id)
        if grid_id and explicit_id in valid_ids:
            explicit_ids_by_grid.setdefault(grid_id, set()).add(explicit_id)
    source_structure_records_by_grid: dict[str, list[dict[str, Any]]] = {}
    for record in source_structure_records:
        position = _candidate_b_monthly_observed_position(record)
        if position is None:
            continue
        source_structure_records_by_grid.setdefault(position[0], []).append(record)
    for grid_id, grid in grids.items():
        explicit_id = str(grid.get("account_id") or "")
        if explicit_id:
            observed_explicit_ids_by_grid.setdefault(grid_id, set()).add(explicit_id)
            if explicit_id in valid_ids:
                explicit_ids_by_grid.setdefault(grid_id, set()).add(explicit_id)

    def segment_candidates(page: int, top: float, *, geometry_known: bool) -> list[dict[str, Any]]:
        if not geometry_known or page <= 0:
            return []
        candidates: list[dict[str, Any]] = []
        for account_id, segments in account_segments.items():
            if any(
                segment_page == page
                and top + 8.0 >= minimum
                and (maximum is None or top < maximum)
                for segment_page, minimum, maximum in segments
            ):
                account = accounts_by_id.get(account_id)
                if account is not None:
                    candidates.append(account)
        return candidates

    def exact_segment_owner(
        grid_id: str,
        grid: Mapping[str, Any],
        *,
        duplicate_ids: set[str],
    ) -> tuple[dict[str, Any] | None, str]:
        """Resolve one grid from scale-invariant segment containment only."""

        if grid_id in duplicate_ids:
            return None, "duplicate_grid_id"
        page = _positive_native_int(grid.get("page"))
        box = _geometry_box(grid)
        if page is None or box is None:
            return None, "account_segment_geometry_unresolved"
        candidates = segment_candidates(page, float(box[1]), geometry_known=True)
        exact_candidates = [
            candidate
            for candidate in candidates
            if _account_has_exact_anchor_ownership(candidate)
        ]
        explicit_id = str(grid.get("account_id") or "").strip()
        if explicit_id:
            exact_candidates = [
                candidate
                for candidate in exact_candidates
                if str(candidate.get("account_id") or "") == explicit_id
            ]
        if len(exact_candidates) == 1:
            return exact_candidates[0], "canonical_account_segment"
        if len(exact_candidates) > 1:
            return None, "ambiguous_account_segments"
        return None, "account_segment_not_observed"

    def exact_source_table_owner(
        grid_id: str,
    ) -> tuple[dict[str, Any] | None, str]:
        if grid_id in duplicate_grid_ids:
            return None, "duplicate_grid_id"
        grid = grids.get(grid_id)
        source_candidates = source_candidates_by_grid.get(grid_id, [])
        if not isinstance(grid, dict) or not source_candidates:
            return None, "source_table_grid_not_observed"
        grid_page = _positive_native_int(grid.get("page"))
        if grid_page is None:
            return None, "source_table_page_conflict"
        grid_box = _geometry_box({"bbox": grid.get("bbox")})
        expected_months = _grid_months(grid)
        if grid_page <= 0 or grid_box is None:
            return None, "source_table_grid_geometry_unresolved"
        if not expected_months:
            return None, "source_table_grid_month_contract_unresolved"
        month_x_bands = _exact_grid_month_x_bands(grid)
        if month_x_bands is None:
            return None, "source_table_grid_month_geometry_unresolved"

        observed_months: list[tuple[int, int]] = []
        record_table_identities: set[tuple[int, str]] = set()
        source_ref_boxes: list[list[float]] = []
        for candidate in source_candidates:
            refs = (
                candidate.get("source_cell_refs")
                if isinstance(candidate.get("source_cell_refs"), list)
                else []
            )
            grid_id_aliases = {
                str(value)
                for value in (
                    candidate.get("grid_id"),
                    *(
                        source_ref.get("grid_id")
                        for source_ref in refs
                        if isinstance(source_ref, dict)
                    ),
                )
                if value not in (None, "")
            }
            calendar_identity = _exact_performance_month(candidate)
            if calendar_identity is None:
                return None, "source_table_grid_month_contract_unresolved"
            year, month = calendar_identity
            if grid_id_aliases != {grid_id}:
                return None, "source_table_grid_identity_conflict"
            if (
                year < 1900
                or not 1 <= month <= 12
                or _monthly_record_id(candidate, grid_id) != f"{grid_id}:{year:04d}-{month:02d}"
                or not refs
            ):
                return None, "source_table_grid_month_contract_unresolved"
            observed_months.append((year, month))
            for source_ref in refs:
                raw_identity = _source_table_identity(source_ref)
                if raw_identity is not None and raw_identity[0] != grid_page:
                    return None, "source_table_page_conflict"
                identity = _exact_month_source_table_identity(source_ref)
                if identity is None:
                    return None, "source_table_exact_provenance_unresolved"
                if (
                    identity[0] != grid_page
                    or str(source_ref.get("grid_id") or "") != grid_id
                ):
                    return None, "source_table_page_conflict"
                raw_source_month = source_ref.get("col")
                if (
                    isinstance(raw_source_month, bool)
                    or not isinstance(raw_source_month, int)
                ):
                    return None, "source_table_grid_month_contract_unresolved"
                source_month = raw_source_month
                if source_month != month:
                    return None, "source_table_grid_month_contract_unresolved"
                record_table_identities.add(identity)
                ref_box = _geometry_box({"bbox": source_ref.get("bbox")})
                if ref_box is None:
                    return None, "source_table_exact_provenance_unresolved"
                expected_x_band = month_x_bands.get(month)
                if (
                    not _box_contains(grid_box, ref_box, tolerance=0.0)
                    or expected_x_band is None
                    or abs(ref_box[0] - expected_x_band[0]) > 1e-6
                    or abs(ref_box[2] - expected_x_band[1]) > 1e-6
                ):
                    return None, "source_table_grid_geometry_conflict"
                source_ref_boxes.append(ref_box)

        if len(observed_months) != len(expected_months) or set(observed_months) != expected_months:
            return None, "source_table_grid_month_contract_unresolved"
        grid_identity = _exact_grid_source_table_identity(grid)
        if grid_identity is None:
            return None, "source_table_exact_provenance_unresolved"
        if grid_identity[0] != grid_page:
            return None, "source_table_page_conflict"
        if record_table_identities != {grid_identity}:
            return None, "source_table_grid_provenance_conflict"

        if grid_identity in (
            conflicting_account_table_geometry | invalid_account_table_geometry
        ):
            return None, "source_table_account_geometry_conflict"
        owners = account_table_owners.get(grid_identity, [])
        if not owners:
            return None, "source_table_account_owner_not_observed"
        if len(owners) != 1:
            return None, "ambiguous_source_table_account_owners"
        owner, table_boxes = owners[0]
        matching_table_boxes = [
            table_box
            for table_box in table_boxes
            if _box_contains(table_box, grid_box)
            and all(_box_contains(table_box, source_ref_box) for source_ref_box in source_ref_boxes)
        ]
        if not matching_table_boxes:
            return None, "source_table_grid_geometry_conflict"

        owner_id = str(owner.get("account_id") or "")
        explicit_ids = observed_explicit_ids_by_grid.get(grid_id, set())
        if explicit_ids and explicit_ids != {owner_id}:
            return None, "explicit_owner_source_table_conflict"
        owner_identifier = _canonical_account_identifier(owner.get("account_identifier"))
        if owner_identifier is None:
            return None, "source_table_account_identifier_unresolved"
        observed_identifiers = {
            str(candidate.get("account_identifier") or "")
            for candidate in source_candidates
            if candidate.get("account_identifier")
        }
        grid_identifier = str(grid.get("account_identifier") or "")
        if grid_identifier:
            observed_identifiers.add(grid_identifier)
        if observed_identifiers and observed_identifiers != {owner_identifier}:
            return None, "account_identifier_source_table_conflict"
        return owner, "exact_source_table_account_owner"

    source_table_owner_by_grid = {
        grid_id: exact_source_table_owner(grid_id)
        for grid_id in grids
    }

    for record in repayments:
        item = dict(record)
        refs = item.get("source_cell_refs") if isinstance(item.get("source_cell_refs"), list) else []
        first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
        grid_id = str(item.get("grid_id") or first_ref.get("grid_id") or "")
        if grid_id:
            item.setdefault("grid_id", grid_id)
        grid = grids.get(grid_id, {})
        page = int(grid.get("page") or first_ref.get("page") or first_ref.get("logical_page") or 0)
        raw_box = grid.get("bbox") or item.get("status_bbox") or first_ref.get("bbox")
        box = _geometry_box({"bbox": raw_box}) or _geometry_box(grid) or [0, 0, 0, 0]
        box_is_known = box[2] > box[0] and box[3] > box[1]
        grid_y = float(box[1])
        candidates = segment_candidates(page, grid_y, geometry_known=box_is_known)
        candidate_ids = {str(candidate.get("account_id") or "") for candidate in candidates}
        if grid_id in duplicate_grid_ids:
            selected = None
            linkage_basis = "duplicate_grid_id"
        elif grid_id and grid_id in owner_by_grid:
            selected, linkage_basis = owner_by_grid[grid_id]
        else:
            explicit_owner = accounts_by_id.get(str(item.get("account_id") or ""))
            conflicting_explicit = bool(grid_id and len(explicit_ids_by_grid.get(grid_id, set())) > 1)
            if conflicting_explicit:
                selected = None
                linkage_basis = "conflicting_explicit_account_ids"
            elif explicit_owner is not None and candidate_ids and str(explicit_owner.get("account_id") or "") not in candidate_ids:
                selected = None
                linkage_basis = "explicit_owner_segment_conflict"
            elif explicit_owner is not None and str(explicit_owner.get("account_id") or "") in candidate_ids:
                selected = explicit_owner
                linkage_basis = "explicit_account_id_confirmed_by_canonical_segment"
            elif explicit_owner is not None:
                table_owner, table_basis = source_table_owner_by_grid.get(
                    grid_id,
                    (None, "source_table_grid_not_observed"),
                )
                if table_owner is not None and str(table_owner.get("account_id") or "") == str(
                    explicit_owner.get("account_id") or ""
                ):
                    selected = table_owner
                    linkage_basis = table_basis
                else:
                    selected = None
                    linkage_basis = (
                        "explicit_owner_source_table_conflict"
                        if table_owner is not None
                        else table_basis
                    )
            elif len(candidates) == 1:
                selected = candidates[0]
                linkage_basis = "canonical_account_segment"
            elif len(candidates) > 1:
                selected = None
                linkage_basis = "ambiguous_account_segments"
            else:
                selected, linkage_basis = source_table_owner_by_grid.get(
                    grid_id,
                    (None, "account_segment_not_observed"),
                )
        if grid_id and grid_id not in owner_by_grid:
            owner_by_grid[grid_id] = selected, linkage_basis
        if selected is not None:
            item["account_id"] = selected.get("account_id")
            identifier = selected.get("account_identifier")
            if identifier:
                from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
                    role_candidate_is_valid,
                )

                if role_candidate_is_valid(identifier, "account_identifier"):
                    item["account_identifier"] = identifier
                else:
                    item.pop("account_identifier", None)
            observed_position = _candidate_b_monthly_observed_position(item)
            if observed_position is not None and observed_position[0] == grid_id:
                identity_proof = _account_month_identity_proof(
                    selected,
                    grid,
                    linkage_basis=linkage_basis,
                    year=observed_position[1],
                    month=observed_position[2],
                )
                if identity_proof is not None:
                    item["_account_month_identity_proof"] = identity_proof
                    item["_account_month_identity_proof_status"] = "exact"
                else:
                    item["_account_month_identity_proof_status"] = (
                        "unproven_exact_anchor_range_geometry_owner"
                    )
        else:
            item.pop("account_id", None)
            item.pop("account_identifier", None)
            item["extraction_status"] = "review"
            item.setdefault("audit", {})["account_linkage"] = linkage_basis
            issue_marker = grid_id or str(_monthly_record_id(item, grid_id) or f"page:{page}:row:{len(linked)}")
            if issue_context is not None and issue_marker not in reported_unresolved_grids:
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                reported_unresolved_grids.add(issue_marker)
                ownership_refs = refs or [
                    {
                        "source": "candidate_b_monthly_grid",
                        "logical_page": page,
                        "grid_id": grid_id,
                        **({"bbox": box} if box_is_known else {}),
                        "geometry_scope": "grid" if box_is_known else "logical_page",
                    }
                ]
                source_candidates = source_candidates_by_grid.get(grid_id, [])
                observed_months = sorted(
                    {
                        f"{int(candidate.get('year')):04d}-{int(candidate.get('month')):02d}"
                        for candidate in source_candidates
                        if str(candidate.get("year") or "").isdigit()
                        and str(candidate.get("month") or "").isdigit()
                        and 1 <= int(candidate.get("month") or 0) <= 12
                    }
                )
                expected_months = sorted(
                    f"{year:04d}-{month:02d}" for year, month in _grid_months(grid)
                )
                record_issue(
                    issue_context,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_monthly_grid_owner_unresolved",
                        message=(
                            "A canonical monthly grid could not be assigned to exactly one printed account "
                            "segment or exact account-owned physical table; its monthly rows were withheld "
                            "from the typed relation."
                        ),
                        parser_stage="candidate_b_relationship_schema",
                        target_dataset="repayment_records",
                        field_name="account_id",
                        observed_value={
                            "grid_id": grid_id,
                            "grid_page": page,
                            "linkage_basis": linkage_basis,
                            "observed_candidate_count": len(source_candidates),
                            "observed_candidate_months": observed_months,
                            "linked_count": 0,
                            "candidate_account_ids": sorted(candidate_ids),
                            "explicit_account_ids": sorted(explicit_ids_by_grid.get(grid_id, set())),
                        },
                        candidate_value={
                            "expected_month_count": len(expected_months) or None,
                            "expected_months": expected_months,
                        },
                        source_refs=ownership_refs,
                        reason_codes=(
                            "exact_account_segment_or_source_table_owner_required",
                            linkage_basis,
                            "nearest_or_single_account_inference_disabled",
                            "relation_withheld",
                        ),
                    ),
                )
                report_localized_monthly_omissions(
                    issue_context,
                    issue_code="candidate_b_monthly_grid_owner_unresolved_field",
                    message=(
                        "A source-known monthly field was withheld because its printed grid could not be "
                        "assigned to exactly one canonical account owner."
                    ),
                    parser_stage="candidate_b_relationship_schema",
                    grid_id=grid_id,
                    months=_grid_months(grid)
                    or {
                        position[1:]
                        for candidate in source_candidates
                        if (position := _candidate_b_monthly_observed_position(candidate))
                        is not None
                    },
                    source_records=source_candidates,
                    grid=grid,
                    observed_context={"linkage_basis": linkage_basis},
                    reason_codes=(
                        "exact_account_segment_or_source_table_owner_required",
                        linkage_basis,
                        "relation_withheld",
                    ),
                )
            # A canonical monthly row cannot exist in v2 without a proven
            # account foreign key.  Preserve the grid-level issue/evidence,
            # but never publish an orphan row or invent its owner.
            continue
        linked.append(item)

    # The canonical schema permits one status per account/month.  Duplicate
    # detectors are collapsed only after a valid account relation exists.
    output: list[dict[str, Any]] = []
    positions: dict[tuple[str, int, int], int] = {}
    reported_duplicate_conflicts: set[tuple[str, int, int]] = set()
    account_gaps = [
        issue
        for issue in getattr(issue_context, "_personal_detail_extraction_issues", ())
        if isinstance(issue, dict)
        and issue.get("issue_code") == "candidate_b_account_sequence_gap"
        and str(issue.get("status") or "open") != "resolved"
    ]
    cross_grid_collisions: dict[tuple[str, int, int], dict[str, Any]] = {}

    def record_grid_id(record: dict[str, Any]) -> str:
        refs = (
            record.get("source_cell_refs")
            if isinstance(record.get("source_cell_refs"), list)
            else []
        )
        first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
        return str(record.get("grid_id") or first_ref.get("grid_id") or "")

    for item in linked:
        account_id = str(item.get("account_id") or "")
        try:
            year = int(item.get("year") or str(item.get("performance_month") or "")[:4] or 0)
            month = int(item.get("month") or str(item.get("performance_month") or "")[5:7] or 0)
        except (TypeError, ValueError):
            year = month = 0
        if not account_id or year < 1900 or not 1 <= month <= 12:
            output.append(item)
            continue
        key = (account_id, year, month)
        existing = positions.get(key)
        if existing is None:
            positions[key] = len(output)
            output.append(item)
            continue
        current = output[existing]

        def score(candidate: dict[str, Any]) -> tuple[int, float, int]:
            status = str(candidate.get("status_code") or candidate.get("status") or "").strip().lower()
            try:
                confidence = float(candidate.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            return int(status not in {"", "unknown"}), confidence, len(candidate.get("source_cell_refs") or ())

        selected, other = (dict(item), current) if score(item) > score(current) else (dict(current), item)
        refs = _merged_source_refs(selected, other)
        if refs:
            selected["source_cell_refs"] = refs
        current_grid_id = record_grid_id(current)
        item_grid_id = record_grid_id(item)
        if account_gaps and current_grid_id and item_grid_id and current_grid_id != item_grid_id:
            collision = cross_grid_collisions.setdefault(
                key,
                {
                    "grid_ids": set(),
                    "records": [],
                    "suppressed_candidate_count": 0,
                },
            )
            collision["grid_ids"].update((current_grid_id, item_grid_id))
            collision["records"].extend((current, item))
            collision["suppressed_candidate_count"] += 1
        selected_signature = _monthly_business_signature(selected)
        other_signature = _monthly_business_signature(other)
        if selected_signature == other_signature:
            # Equal business values retain one canonical row. Same-grid
            # detector replays are evidence aggregation; a distinct-grid
            # collapse under an account gap is reported below.
            output[existing] = selected
            continue

        selected_status = selected_signature[0]
        other_status = other_signature[0]
        selected_status_valid = selected_status not in {"", "UNKNOWN"}
        other_status_valid = other_status not in {"", "UNKNOWN"}
        if selected_status_valid != other_status_valid:
            # An unresolved detector replay contributes evidence, not a second
            # business value. Keep the usable candidate for the professional
            # correction/final schema stages without manufacturing a conflict.
            usable = selected if selected_status_valid else dict(other)
            if refs:
                usable["source_cell_refs"] = refs
            output[existing] = usable
            continue

        audit = dict(selected.get("audit") or {})
        prior_count = audit.get("duplicate_month_candidates")
        audit["duplicate_month_candidates"] = (
            int(prior_count) + 1
            if isinstance(prior_count, int) and not isinstance(prior_count, bool)
            else 2
        )
        audit["duplicate_month_conflict"] = {
            "selected": {"status": selected_signature[0], "amount": selected_signature[1]},
            "other": {"status": other_signature[0], "amount": other_signature[1]},
        }
        selected["audit"] = audit
        selected["extraction_status"] = "review"
        output[existing] = selected
        if issue_context is not None and key not in reported_duplicate_conflicts:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            reported_duplicate_conflicts.add(key)
            record_issue(
                issue_context,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_monthly_duplicate_conflict",
                    message=(
                        "Multiple observations for one account-month contained different normalized business values; "
                        "the stronger candidate was retained and the conflict was reported."
                    ),
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    target_record_id=_monthly_record_id(selected, str(selected.get("grid_id") or "")),
                    observed_value={
                        "status": other_signature[0],
                        "overdue_amount": other_signature[1],
                    },
                    candidate_value={
                        "status": selected_signature[0],
                        "overdue_amount": selected_signature[1],
                    },
                    source_refs=refs,
                    reason_codes=(
                        "duplicate_account_month",
                        "conflicting_business_values",
                        "selected_candidate_requires_review",
                    ),
                ),
            )
    if issue_context is not None and cross_grid_collisions:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            make_issue,
            record_issue,
        )

        missing_sequences = {
            str((issue.get("observed_value") or {}).get("account_type") or "unknown"): list(
                (issue.get("candidate_value") or {}).get("missing_category_sequences") or ()
            )
            for issue in account_gaps
        }
        for key, collision in cross_grid_collisions.items():
            retained = output[positions[key]]
            grid_ids = sorted(collision["grid_ids"])
            collision_refs = _merged_source_refs(*collision["records"])
            record_issue(
                issue_context,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="monthly_linkage_collision_from_account_gap",
                    message=(
                        "Distinct monthly grids collapsed onto one account-month while account-family ordinals "
                        "were unresolved; one canonical row was retained and the possible population loss was reported."
                    ),
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    target_record_id=_monthly_record_id(
                        retained, str(retained.get("grid_id") or "")
                    ),
                    field_name="account_id",
                    observed_value={
                        "account_id": key[0],
                        "performance_month": f"{key[1]:04d}-{key[2]:02d}",
                        "colliding_grid_ids": grid_ids,
                        "distinct_grid_count": len(grid_ids),
                        "suppressed_candidate_count": collision[
                            "suppressed_candidate_count"
                        ],
                        "final_linked_row_count": len(output),
                    },
                    candidate_value={
                        "pre_deduplication_row_count": len(linked),
                        "collapsed_candidate_count": collision[
                            "suppressed_candidate_count"
                        ],
                        "missing_account_category_sequences": missing_sequences,
                    },
                    source_refs=collision_refs,
                    reason_codes=(
                        "credit_account_population_incomplete",
                        "distinct_monthly_grids_share_account_month",
                        "duplicate_account_month_linkage",
                        "final_population_loss_reported",
                    ),
                ),
            )
    if issue_context is not None:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

        records_by_grid: dict[str, list[dict[str, Any]]] = {}
        emitted_account_months: set[tuple[str, int, int]] = set()
        for record in output:
            refs = record.get("source_cell_refs") if isinstance(record.get("source_cell_refs"), list) else []
            first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
            grid_id = str(record.get("grid_id") or first_ref.get("grid_id") or "")
            if grid_id:
                records_by_grid.setdefault(grid_id, []).append(record)
            observed_position = _candidate_b_monthly_observed_position(record)
            account_id = str(record.get("account_id") or "").strip()
            if account_id and observed_position is not None:
                emitted_account_months.add(
                    (account_id, observed_position[1], observed_position[2])
                )

        bound_source_positions: dict[
            tuple[str, int, int],
            dict[tuple[str, str], dict[str, Any]],
        ] = {}

        def register_bound_source_position(
            *,
            account_id: str,
            year: int,
            month: int,
            grid_id: str,
            source_records: Iterable[Mapping[str, Any]],
            grid: Mapping[str, Any] | None,
            linkage_basis: str,
        ) -> None:
            """Register one exact source position under its canonical identity."""

            if not account_id or not grid_id or year < 1900 or not 1 <= month <= 12:
                return
            position = (grid_id, f"{year:04d}-{month:02d}")
            bound_source_positions.setdefault((account_id, year, month), {}).setdefault(
                position,
                {
                    "grid_id": grid_id,
                    "source_records": list(source_records),
                    "grid": grid,
                    "linkage_basis": linkage_basis,
                },
            )

        for record in linked:
            proof = record.get("_account_month_identity_proof")
            calendar_identity = _exact_performance_month(record)
            account_id = str(record.get("account_id") or "").strip()
            if not (
                isinstance(proof, Mapping)
                and calendar_identity is not None
                and str(proof.get("account_id") or "").strip() == account_id
                and proof.get("account_anchor_exact") is True
                and proof.get("printed_month_range_exact") is True
                and proof.get("grid_geometry_exact") is True
                and proof.get("unique_owner") is True
            ):
                continue
            year, month = calendar_identity
            grid_id = str(proof.get("grid_id") or record_grid_id(record)).strip()
            if str(proof.get("performance_month") or "").strip() != (
                f"{year:04d}-{month:02d}"
            ):
                continue
            register_bound_source_position(
                account_id=account_id,
                year=year,
                month=month,
                grid_id=grid_id,
                source_records=source_candidates_by_grid.get(grid_id, [record]),
                grid=grids.get(grid_id),
                linkage_basis=str(proof.get("owner_basis") or ""),
            )
        for grid_id, grid in grids.items():
            expected_months = _grid_months(grid)
            if not expected_months:
                audit = grid.get("audit") if isinstance(grid.get("audit"), dict) else {}
                if audit.get("date_range_status") == "unresolved":
                    page = int(grid.get("page") or 0)
                    raw_bbox = grid.get("bbox")
                    bbox = (
                        list(raw_bbox)
                        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4
                        else None
                    )
                    record_issue(
                        issue_context,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="candidate_b_monthly_date_range_unresolved",
                            message=(
                                "A printed monthly repayment range was detected, but its OCR-damaged date range "
                                "could not be trusted; no account-month values were invented."
                            ),
                            parser_stage="candidate_b_relationship_schema",
                            target_dataset="repayment_records",
                            field_name="performance_month",
                            observed_value={
                                "grid_id": grid_id,
                                "anchor_text": str(grid.get("anchor_text") or ""),
                                "observed_date_components": audit.get("observed_date_components"),
                            },
                            candidate_value={"resolution": "withheld_pending_date_range_review"},
                            source_refs=(
                                {
                                    "page": page,
                                    "logical_page": page,
                                    "bbox": bbox,
                                    "grid_id": grid_id,
                                    "field_name": "performance_month",
                                    "geometry_scope": "grid" if bbox else "logical_page",
                                },
                            ),
                            reason_codes=(
                                "printed_repayment_anchor_detected",
                                "date_range_ocr_invalid",
                                "monthly_values_not_invented",
                                "dataset_incomplete",
                            ),
                        ),
                    )
                continue
            observed_rows = records_by_grid.get(grid_id, [])
            observed_months: set[tuple[int, int]] = set()
            owners: set[str] = set()
            for record in observed_rows:
                try:
                    year = int(record.get("year") or str(record.get("performance_month") or "")[:4])
                    month = int(record.get("month") or str(record.get("performance_month") or "")[5:7])
                except (TypeError, ValueError):
                    continue
                if year >= 1900 and 1 <= month <= 12:
                    observed_months.add((year, month))
                if record.get("account_id"):
                    owners.add(str(record["account_id"]))
            grid_owner, linkage_basis = owner_by_grid.get(grid_id, (None, ""))
            if grid_owner is None and grid_id not in owner_by_grid:
                grid_owner, linkage_basis = exact_segment_owner(
                    grid_id,
                    grid,
                    duplicate_ids=duplicate_grid_ids,
                )
                if grid_owner is None:
                    table_owner, table_basis = source_table_owner_by_grid.get(
                        grid_id,
                        (None, "source_table_grid_not_observed"),
                    )
                    if table_owner is not None:
                        grid_owner, linkage_basis = table_owner, table_basis
                owner_by_grid[grid_id] = grid_owner, linkage_basis
            owner_id = str((grid_owner or {}).get("account_id") or "").strip()
            proven_months = {
                (year, month)
                for year, month in expected_months
                if owner_id
                and _account_month_identity_proof(
                    grid_owner,
                    grid,
                    linkage_basis=linkage_basis,
                    year=year,
                    month=month,
                )
                is not None
            }
            for year, month in proven_months:
                register_bound_source_position(
                    account_id=owner_id,
                    year=year,
                    month=month,
                    grid_id=grid_id,
                    source_records=source_candidates_by_grid.get(grid_id, []),
                    grid=grid,
                    linkage_basis=linkage_basis,
                )
            covered_months = {
                (year, month)
                for year, month in proven_months
                if (owner_id, year, month) in emitted_account_months
            }
            contract_months = covered_months if proven_months else observed_months
            if contract_months == expected_months and len(owners) == 1:
                continue
            record_issue(
                issue_context,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_grid_contract_unresolved",
                    message=(
                        "A printed monthly grid did not yield exactly one observation for every printed month "
                        "under exactly one account owner."
                    ),
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    observed_value={
                        "grid_id": grid_id,
                        "linked_month_count": len(observed_months),
                        "linked_months": sorted(
                            f"{year:04d}-{month:02d}" for year, month in observed_months
                        ),
                        "observed_candidate_count": len(
                            source_candidates_by_grid.get(grid_id, [])
                        ),
                        "account_owners": sorted(owners),
                        "canonical_account_id": owner_id or None,
                        "account_month_covered_count": len(covered_months),
                        "account_month_owner_basis": linkage_basis or None,
                    },
                    candidate_value={
                        "printed_month_count": len(expected_months),
                        "printed_months": sorted(
                            f"{year:04d}-{month:02d}" for year, month in expected_months
                        ),
                    },
                    reason_codes=(
                        "printed_date_range_contract",
                        "single_account_grid_contract",
                        "dataset_incomplete",
                    ),
                ),
            )
            missing_months = expected_months - contract_months
            if missing_months:
                report_localized_monthly_omissions(
                    issue_context,
                    issue_code="candidate_b_monthly_grid_contract_missing_field",
                    message=(
                        "A source-known field for a printed grid/month was not present in the linked "
                        "canonical monthly relation."
                    ),
                    parser_stage="candidate_b_relationship_schema",
                    grid_id=grid_id,
                    months=missing_months,
                    account_id=owner_id or None,
                    account_month_identity_proven=bool(
                        owner_id and missing_months <= proven_months
                    ),
                    source_records=source_candidates_by_grid.get(grid_id, []),
                    grid=grid,
                    observed_context={
                        "linked_months": sorted(
                            f"{year:04d}-{month:02d}" for year, month in observed_months
                        ),
                        "account_month_owner_basis": linkage_basis or None,
                    },
                    reason_codes=(
                        "printed_date_range_contract",
                        "grid_month_missing_from_linked_relation",
                        "dataset_incomplete",
                    ),
                )

        # The detached plane contributes only source positions.  Promote one
        # to the account-month closure iff its own geometry falls inside one
        # exact account segment; extraction-local grid aliases are then
        # reconciled against the global emitted identity set.
        for grid_id, grid in source_structure_grids.items():
            records = source_structure_records_by_grid.get(grid_id, [])
            source_months = {
                position[1:]
                for record in records
                if (position := _candidate_b_monthly_observed_position(record))
                is not None
            }
            source_months.update(_grid_months(grid))
            if not source_months:
                continue
            grid_owner, linkage_basis = exact_segment_owner(
                grid_id,
                grid,
                duplicate_ids=duplicate_source_structure_grid_ids,
            )
            owner_id = str((grid_owner or {}).get("account_id") or "").strip()
            proven_source_months = {
                (year, month)
                for year, month in source_months
                if owner_id
                and _account_month_identity_proof(
                    grid_owner,
                    grid,
                    linkage_basis=linkage_basis,
                    year=year,
                    month=month,
                )
                is not None
            }
            for year, month in proven_source_months:
                register_bound_source_position(
                    account_id=owner_id,
                    year=year,
                    month=month,
                    grid_id=grid_id,
                    source_records=records,
                    grid=grid,
                    linkage_basis=linkage_basis,
                )
            proven_missing_months = {
                (year, month)
                for year, month in proven_source_months
                if (owner_id, year, month) not in emitted_account_months
            }
            if not proven_missing_months:
                continue
            report_localized_monthly_omissions(
                issue_context,
                issue_code="canonical_monthly_source_structure_missing_field",
                message=(
                    "A detached printed grid/month source position has a unique exact account owner, "
                    "but no canonical monthly business row was emitted."
                ),
                parser_stage="candidate_b_relationship_schema",
                grid_id=grid_id,
                months=proven_missing_months,
                account_id=owner_id,
                account_month_identity_proven=True,
                source_records=records,
                grid=grid,
                observed_context={
                    "linkage_basis": linkage_basis,
                    "source_structure_is_audit_only": True,
                },
                require_exact_cell_ref=True,
                reason_codes=(
                    "detached_source_structure_exact_key",
                    "exact_account_segment_owner",
                    "printed_date_range_contract",
                    "grid_alias_reconciled_by_account_month",
                    "dataset_incomplete",
                ),
            )
        for anchor in repayment_anchors or ():
            if not isinstance(anchor, dict):
                continue
            account_id = str(anchor.get("account_id") or "")
            page = int(anchor.get("page") or 0)
            anchor_box = _geometry_box(anchor)
            anchor_top = float(anchor_box[1]) if anchor_box else None
            try:
                segment_minimum = float(anchor.get("segment_min_y"))
            except (TypeError, ValueError):
                segment_minimum = anchor_top
            try:
                segment_maximum = float(anchor.get("segment_max_y"))
            except (TypeError, ValueError):
                segment_maximum = None
            anchor_range = anchor.get("date_range")
            anchor_months = _grid_months({"audit": {"date_range": anchor_range}})
            candidate_months: set[tuple[int, int]] = set()
            anchor_grid_candidates = (
                *((grid_id, grid, False) for grid_id, grid in grids.items()),
                *(
                    (grid_id, grid, True)
                    for grid_id, grid in source_structure_grids.items()
                ),
            )
            for grid_id, grid, source_structure_only in anchor_grid_candidates:
                grid_page = int(grid.get("page") or 0)
                explicit_account_id = str(grid.get("account_id") or "")
                if explicit_account_id and explicit_account_id != account_id:
                    continue
                grid_box = _geometry_box(grid)
                geometry_match = False
                if grid_box is not None:
                    grid_top = float(grid_box[1])
                    geometry_match = (
                        (anchor_top is None or grid_top + 12.0 >= anchor_top)
                        and (segment_minimum is None or grid_top + 8.0 >= segment_minimum)
                        and (segment_maximum is None or grid_top < segment_maximum)
                    )
                text_match = bool(
                    re.sub(r"\s+", "", str(anchor.get("anchor_text") or ""))
                    and re.sub(r"\s+", "", str(anchor.get("anchor_text") or ""))
                    == re.sub(r"\s+", "", str(grid.get("anchor_text") or ""))
                )
                grid_months = _grid_months(grid)
                overlapping_months = anchor_months & grid_months
                if not overlapping_months:
                    continue
                if source_structure_only:
                    grid_owner, owner_basis = exact_segment_owner(
                        grid_id,
                        grid,
                        duplicate_ids=duplicate_source_structure_grid_ids,
                    )
                else:
                    grid_owner, owner_basis = owner_by_grid.get(grid_id, (None, ""))
                if (
                    grid_owner is None
                    or str(grid_owner.get("account_id") or "") != account_id
                ):
                    continue
                grid_is_in_account_segment = any(
                    segment_page == grid_page
                    and grid_box is not None
                    and grid_box[1] + 8.0 >= minimum
                    and (maximum is None or grid_box[1] < maximum)
                    for segment_page, minimum, maximum in account_segments.get(
                        account_id, []
                    )
                )
                same_page_match = grid_page == page and (
                    geometry_match or text_match or bool(overlapping_months)
                )
                cross_page_match = grid_page != page and grid_is_in_account_segment
                if not (same_page_match or cross_page_match):
                    continue
                proven_overlapping_months = {
                    (year, month)
                    for year, month in overlapping_months
                    if _account_month_identity_proof(
                        grid_owner,
                        grid,
                        linkage_basis=owner_basis,
                        year=year,
                        month=month,
                    )
                    is not None
                }
                candidate_months.update(proven_overlapping_months)
            missing_anchor_months = anchor_months - candidate_months
            if anchor_months and not missing_anchor_months:
                continue
            record_issue(
                issue_context,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_anchor_grid_missing",
                    message=(
                        "An account-bound printed repayment-record anchor produced no canonical monthly grid; "
                        "no monthly business rows were invented."
                    ),
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    target_record_id=account_id,
                    field_name="account_id",
                        observed_value={
                        "anchor_id": anchor.get("anchor_id"),
                        "account_id": account_id or None,
                        "anchor_text": anchor.get("anchor_text"),
                        "date_range": anchor_range,
                            "materialized_grid_count": int(bool(candidate_months)),
                            "missing_months": sorted(
                                f"{year:04d}-{month:02d}"
                                for year, month in missing_anchor_months
                            ),
                    },
                    candidate_value={"resolution": "missing_grid_reported_for_account"},
                    source_refs=tuple(
                        ref for ref in anchor.get("source_refs") or () if isinstance(ref, dict)
                    ),
                    reason_codes=(
                        "independent_visible_repayment_anchor",
                        "exact_account_segment_owner",
                        "zero_materialized_grids",
                        "monthly_rows_not_invented",
                    ),
                ),
            )
            if missing_anchor_months:
                _report_account_range_monthly_omissions(
                    issue_context,
                    anchor=anchor,
                    missing_months=missing_anchor_months,
                )
        for identity, source_positions in bound_source_positions.items():
            if len(source_positions) <= 1:
                continue
            account_id, year, month = identity
            primary_position: tuple[str, str] | None = None
            retained_index = positions.get(identity)
            if retained_index is not None:
                retained_grid_id = record_grid_id(output[retained_index])
                retained_position = (
                    retained_grid_id,
                    f"{year:04d}-{month:02d}",
                )
                if retained_position in source_positions:
                    primary_position = retained_position
            if primary_position is None:
                primary_position = min(source_positions)
            for position, payload in sorted(source_positions.items()):
                if position == primary_position:
                    continue
                _report_reconciled_monthly_source_alias(
                    issue_context,
                    account_id=account_id,
                    year=year,
                    month=month,
                    grid_id=payload["grid_id"],
                    source_records=payload["source_records"],
                    grid=payload["grid"],
                    linkage_basis=payload["linkage_basis"],
                )
    return output


def _number(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace("，", ",")
    if not re.fullmatch(
        r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?",
        text,
    ):
        return None
    try:
        parsed = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    return format(parsed, "f")


def derive_candidate_b_overdue_records(
    accounts: list[dict[str, Any]], repayments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Derive overdue views only from Candidate B account and month rows."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    account_types: dict[str, str] = {}
    for account in accounts:
        account_id = str(account.get("account_id") or "")
        account_types[account_id] = str(account.get("account_type") or account.get("credit_card_type") or "")
        status = str(account.get("account_status") or account.get("account_state") or account.get("status") or "")
        tier = str(account.get("five_tier_class") or "")
        amount = _number(account.get("overdue_amount") or account.get("current_overdue_amount"))
        if status not in {"逾期", "overdue"} and tier not in {"关注", "次级", "可疑", "损失", "违约"} and not amount:
            continue
        record_id = stable_record_id("credit_overdue", account_id, "account_snapshot")
        if record_id in seen:
            continue
        seen.add(record_id)
        rows.append(
            {
                "overdue_id": record_id,
                "account_id": account_id,
                "period_scope": "account_snapshot",
                "overdue_amount": amount,
                "five_tier_class": tier or None,
                "source": "candidate_b_account_snapshot",
                "source_refs": list(account.get("source_refs") or ()),
                "confidence": account.get("confidence"),
            }
        )
    for repayment in repayments:
        status = str(repayment.get("status_code") or repayment.get("status") or "")
        if status not in {"1", "2", "3", "4", "5", "6", "7"}:
            continue
        account_id = str(repayment.get("account_id") or "")
        if account_types.get(account_id) == "quasi_credit_card" and status in {"1", "2"}:
            continue
        try:
            year = int(repayment.get("year") or str(repayment.get("performance_month") or "")[:4])
            month = int(repayment.get("month") or str(repayment.get("performance_month") or "")[5:7])
        except (TypeError, ValueError):
            continue
        record_id = stable_record_id("credit_overdue", account_id, year, month)
        if record_id in seen:
            continue
        seen.add(record_id)
        rows.append(
            {
                "overdue_id": record_id,
                "account_id": account_id,
                "period_scope": "month",
                "year": year,
                "month": month,
                "overdue_level": int(status),
                "overdue_amount": _number(repayment.get("overdue_amount") or repayment.get("status_amount")),
                "source": "candidate_b_monthly_performance",
                "source_cell_refs": list(repayment.get("source_cell_refs") or ()),
                "confidence": repayment.get("confidence"),
            }
        )
    return rows


__all__ = ["derive_candidate_b_overdue_records", "link_candidate_b_repayments"]
