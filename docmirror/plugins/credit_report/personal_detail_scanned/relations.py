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
from decimal import Decimal, InvalidOperation
from typing import Any

from docmirror.plugins.credit_report.value_utils import stable_record_id

_REPAYMENT_RANGE_RE = re.compile(
    r"(20\d{2})\s*年?\s*(\d{1,2})\s*月?\s*[-—–－至到]\s*"
    r"(20\d{2})\s*年?\s*(\d{1,2})\s*月?"
)
_PERFORMANCE_MONTH_RE = re.compile(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])\Z")


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
    compact = re.sub(r"[,，\s]", "", str(raw_amount))
    try:
        amount = format(Decimal(compact).normalize(), "f")
        if amount == "-0":
            amount = "0"
    except (InvalidOperation, TypeError, ValueError):
        amount = f"raw:{compact}"
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
            range_match = _REPAYMENT_RANGE_RE.search(text)
            if range_match is None:
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
                    if nearby and _REPAYMENT_RANGE_RE.search(neighbor_text):
                        text = f"{neighbor_text} {text}" if neighbor_index < index else f"{text} {neighbor_text}"
                        range_match = _REPAYMENT_RANGE_RE.search(text)
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
            date_range = None
            if range_match is not None:
                start_year, start_month, end_year, end_month = map(int, range_match.groups())
                if (
                    1 <= start_month <= 12
                    and 1 <= end_month <= 12
                    and end_year * 12 + end_month >= start_year * 12 + start_month
                ):
                    date_range = {
                        "start_year": start_year,
                        "start_month": start_month,
                        "end_year": end_year,
                        "end_month": end_month,
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
                        }
                    ],
                }
            )
    return anchors


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
        for record in output:
            refs = record.get("source_cell_refs") if isinstance(record.get("source_cell_refs"), list) else []
            first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
            grid_id = str(record.get("grid_id") or first_ref.get("grid_id") or "")
            if grid_id:
                records_by_grid.setdefault(grid_id, []).append(record)
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
            if observed_months == expected_months and len(owners) == 1:
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
            candidates: list[str] = []
            for grid_id, grid in grids.items():
                if int(grid.get("page") or 0) != page:
                    continue
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
                range_match = bool(anchor_months and anchor_months == _grid_months(grid))
                if geometry_match or (
                    explicit_account_id == account_id and (text_match or range_match)
                ):
                    candidates.append(grid_id)
            if candidates:
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
                        "materialized_grid_count": 0,
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
    return output


def _number(value: Any) -> str | None:
    text = re.sub(r"[,，\s]", "", str(value or ""))
    if not text:
        return None
    try:
        parsed = Decimal(text)
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
