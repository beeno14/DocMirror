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


def _reading_page(page: int, order: dict[int, int]) -> int:
    return order.get(page, page)


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


def link_candidate_b_repayments(
    repayments: list[dict[str, Any]],
    accounts: list[dict[str, Any]],
    micro_grids: list[dict[str, Any]],
    *,
    reading_order_by_logical: dict[int, int] | None = None,
    issue_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Attach canonical monthly grids to the nearest preceding account card."""
    order = {int(page): int(position) for page, position in (reading_order_by_logical or {}).items()}
    grids = {
        str(grid.get("grid_id") or ""): grid
        for grid in micro_grids
        if isinstance(grid, dict) and grid.get("grid_id")
    }
    by_page: dict[int, list[dict[str, Any]]] = {}
    ordered_accounts: list[tuple[int, float, dict[str, Any]]] = []
    valid_ids: set[str] = set()
    account_boxes: dict[str, list[Any]] = {}
    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_id = str(account.get("account_id") or "")
        if account_id:
            valid_ids.add(account_id)
        refs = [ref for ref in account.get("source_refs") or () if isinstance(ref, dict)]
        first_ref = refs[0] if refs else {}
        page = int(
            account.get("page")
            or first_ref.get("logical_page")
            or first_ref.get("page")
            or 0
        )
        by_page.setdefault(page, []).append(account)
        bbox = account.get("bbox") or first_ref.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and account_id:
            account_boxes[account_id] = bbox
        bottom = float(bbox[3]) if isinstance(bbox, list) and len(bbox) == 4 else 0.0
        ordered_accounts.append((_reading_page(page, order), bottom, account))
    ordered_accounts.sort(key=lambda item: (item[0], item[1], int(item[2].get("sequence") or 0)))

    for page_accounts in by_page.values():
        page_accounts.sort(
            key=lambda account: (
                float((account_boxes.get(str(account.get("account_id") or "")) or [0, 0, 0, 0])[1]),
                int(account.get("sequence") or 0),
            )
        )

    linked: list[dict[str, Any]] = []
    reported_page_order_grids: set[str] = set()
    owner_by_grid: dict[str, dict[str, Any] | None] = {}
    accounts_by_id = {
        str(account.get("account_id") or ""): account
        for account in accounts
        if isinstance(account, dict) and account.get("account_id")
    }
    for record in repayments:
        item = dict(record)
        refs = item.get("source_cell_refs") if isinstance(item.get("source_cell_refs"), list) else []
        first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
        grid_id = str(item.get("grid_id") or first_ref.get("grid_id") or "")
        grid = grids.get(grid_id, {})
        page = int(grid.get("page") or first_ref.get("page") or first_ref.get("logical_page") or 0)
        raw_box = grid.get("bbox") or item.get("status_bbox") or first_ref.get("bbox")
        box = _geometry_box({"bbox": raw_box}) or _geometry_box(grid) or [0, 0, 0, 0]
        box_is_known = box[2] > box[0] and box[3] > box[1]
        grid_y = float(box[1])
        current = by_page.get(page, [])
        preceding = [
            account
            for account in current
            if isinstance(account_boxes.get(str(account.get("account_id") or "")), list)
            and float(account_boxes[str(account.get("account_id") or "")][3]) <= grid_y + 8.0
        ]
        ordered_page = _reading_page(page, order)
        global_preceding = [
            account
            for account_page, bottom, account in ordered_accounts
            if account_page < ordered_page or (account_page == ordered_page and bottom <= grid_y + 8.0)
        ]
        if grid_id in owner_by_grid:
            selected = owner_by_grid[grid_id]
        else:
            explicit_owner = accounts_by_id.get(str(item.get("account_id") or ""))
            selected = explicit_owner or (
                max(
                    preceding,
                    key=lambda account: float(account_boxes[str(account.get("account_id") or "")][3]),
                )
                if preceding
                else global_preceding[-1]
                if global_preceding
                else current[0]
                if len(current) == 1
                else None
            )
        inferred_from_page_order = False
        if selected is None and current and not box_is_known:
            # A whole-page OCR pass can recover a canonical monthly table while
            # losing the table envelope. In a canonical account section the
            # first such grid on the page belongs to the first account anchor;
            # retain the required relation and report the weaker geometry.
            selected = current[0]
            inferred_from_page_order = True
        if grid_id and grid_id not in owner_by_grid:
            owner_by_grid[grid_id] = selected
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
            if inferred_from_page_order:
                item.setdefault("audit", {})["account_linkage"] = "inferred_page_order"
                if issue_context is not None and grid_id not in reported_page_order_grids:
                    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                        make_issue,
                        record_issue,
                    )

                    reported_page_order_grids.add(grid_id)
                    record_issue(
                        issue_context,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="candidate_b_monthly_link_inferred_from_page_order",
                            message="A canonical monthly table lacked usable geometry; its account relation was inferred from canonical page order.",
                            parser_stage="candidate_b_relationship_schema",
                            target_dataset="repayment_records",
                            target_record_id=grid_id,
                            field_name="account_id",
                            observed_value=None,
                            candidate_value=selected.get("account_id"),
                            source_refs=refs,
                            reason_codes=(
                                "monthly_grid_geometry_missing",
                                "canonical_page_order",
                                "relation_requires_review",
                            ),
                        ),
                    )
        elif str(item.get("account_id") or "") not in valid_ids:
            item.pop("account_id", None)
            item.pop("account_identifier", None)
        linked.append(item)

    # The canonical schema permits one status per account/month.  Duplicate
    # detectors are collapsed only after a valid account relation exists.
    output: list[dict[str, Any]] = []
    positions: dict[tuple[str, int, int], int] = {}
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
            status = str(candidate.get("status_code") or candidate.get("status") or "")
            try:
                confidence = float(candidate.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            return int(status not in {"", "unknown"}), confidence, len(candidate.get("source_cell_refs") or ())

        selected, other = (dict(item), current) if score(item) > score(current) else (dict(current), item)
        refs: list[dict[str, Any]] = []
        markers: set[str] = set()
        for ref in [*(selected.get("source_cell_refs") or ()), *(other.get("source_cell_refs") or ())]:
            if not isinstance(ref, dict):
                continue
            marker = repr(sorted(ref.items()))
            if marker in markers:
                continue
            markers.add(marker)
            refs.append(dict(ref))
        if refs:
            selected["source_cell_refs"] = refs
        selected.setdefault("audit", {})["duplicate_month_candidates"] = 2
        output[existing] = selected
    if issue_context is not None and len(output) < len(linked):
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
                        "collapsed_candidate_count": len(linked) - len(output),
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
                    target_record_id=grid_id,
                    observed_value={
                        "month_count": len(observed_months),
                        "account_owners": sorted(owners),
                    },
                    candidate_value={"printed_month_count": len(expected_months)},
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
