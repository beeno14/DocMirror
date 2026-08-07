# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate B relationship materialization.

Account/month linking and overdue derivation live in the document-type branch
so the clean pipeline does not pass its rows through shared scanned-report
assembly code.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from docmirror.plugins.credit_report.value_utils import stable_record_id


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
                if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
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


def _grid_months(grid: dict[str, Any]) -> set[tuple[int, int]]:
    date_range = (grid.get("audit") or {}).get("date_range") or {}
    try:
        start_year = int(date_range.get("start_year") or 0)
        start_month = int(date_range.get("start_month") or 0)
        end_year = int(date_range.get("end_year") or 0)
        end_month = int(date_range.get("end_month") or 0)
    except (TypeError, ValueError):
        return set()
    if start_year < 1900 or end_year < start_year or not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
        return set()
    start = start_year * 12 + start_month - 1
    end = end_year * 12 + end_month - 1
    if end < start or end - start > 120:
        return set()
    return {(value // 12, value % 12 + 1) for value in range(start, end + 1)}


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
    try:
        year = int(record.get("year") or str(record.get("performance_month") or "")[:4])
        month = int(record.get("month") or str(record.get("performance_month") or "")[5:7])
    except (TypeError, ValueError):
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


def link_candidate_b_repayments(
    repayments: list[dict[str, Any]],
    accounts: list[dict[str, Any]],
    micro_grids: list[dict[str, Any]],
    *,
    reading_order_by_logical: dict[int, int] | None = None,
    issue_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Attach monthly grids only to their exact canonical account segment."""
    del reading_order_by_logical
    grids = {
        str(grid.get("grid_id") or ""): grid
        for grid in micro_grids
        if isinstance(grid, dict) and grid.get("grid_id")
    }
    valid_ids: set[str] = set()
    account_segments: dict[str, list[tuple[int, float, float | None]]] = {}
    fallback_anchors_by_page: dict[int, list[tuple[float, str]]] = {}
    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_id = str(account.get("account_id") or "")
        if account_id:
            valid_ids.add(account_id)
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
    explicit_ids_by_grid: dict[str, set[str]] = {}
    source_candidates_by_grid: dict[str, list[dict[str, Any]]] = {}
    for record in repayments:
        refs = record.get("source_cell_refs") if isinstance(record.get("source_cell_refs"), list) else []
        first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
        grid_id = str(record.get("grid_id") or first_ref.get("grid_id") or "")
        if grid_id:
            source_candidates_by_grid.setdefault(grid_id, []).append(record)
        explicit_id = str(record.get("account_id") or "")
        if grid_id and explicit_id in valid_ids:
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
        if grid_id and grid_id in owner_by_grid:
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
                selected = None
                linkage_basis = "explicit_owner_without_segment_proof"
            elif len(candidates) == 1:
                selected = candidates[0]
                linkage_basis = "canonical_account_segment"
            elif len(candidates) > 1:
                selected = None
                linkage_basis = "ambiguous_account_segments"
            else:
                selected = None
                linkage_basis = "account_segment_not_observed"
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
                            "segment; its monthly rows were withheld from the typed relation."
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
                            "exact_account_segment_owner_required",
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
    conflicting_duplicate_count = 0
    reported_duplicate_conflicts: set[tuple[str, int, int]] = set()
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
        selected_signature = _monthly_business_signature(selected)
        other_signature = _monthly_business_signature(other)
        if selected_signature == other_signature:
            # Detector replays with the same normalized status and amount are
            # evidence aggregation, not business uncertainty.
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

        conflicting_duplicate_count += 1
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
    if issue_context is not None and conflicting_duplicate_count:
        account_gaps = [
            issue
            for issue in getattr(issue_context, "_personal_detail_extraction_issues", ())
            if isinstance(issue, dict)
            and issue.get("issue_code") == "candidate_b_account_sequence_gap"
            and str(issue.get("status") or "open") != "resolved"
        ]
        if account_gaps:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record_issue(
                issue_context,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="monthly_linkage_collision_from_account_gap",
                    message=(
                        "Monthly grid candidates collapsed onto duplicate account-month keys while account-family "
                        "ordinals were unresolved; the final rows were deduplicated and the population loss was reported."
                    ),
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    field_name="account_id",
                    observed_value={"final_linked_row_count": len(output)},
                    candidate_value={
                        "pre_deduplication_row_count": len(linked),
                        "collapsed_candidate_count": conflicting_duplicate_count,
                        "missing_account_category_sequences": {
                            str((issue.get("observed_value") or {}).get("account_type") or "unknown"): list(
                                (issue.get("candidate_value") or {}).get("missing_category_sequences") or ()
                            )
                            for issue in account_gaps
                        },
                    },
                    source_refs=(
                        dict(ref)
                        for issue in account_gaps
                        for ref in issue.get("source_refs") or ()
                        if isinstance(ref, dict)
                    ),
                    reason_codes=(
                        "credit_account_population_incomplete",
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
