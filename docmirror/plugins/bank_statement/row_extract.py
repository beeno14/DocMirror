# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared row extraction utilities for grid and borderless bank ledger styles.

Header-aware transaction row detection, debit/credit split handling, and multi-table
harvest helpers shared by ``grid_standard`` and ``borderless_ocr`` style parsers.

Pipeline role: called from style parser modules during ``extract_rows`` phases;
uses ``header_resolve.detect_headers`` for column alignment.

Key exports: ``row_has_transaction_data``, ``extract_rows_from_header``,
``extract_all_tables``, ``count_transaction_data_rows``.

Dependencies: ``bank_statement.header_resolve``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from docmirror.plugins.bank_statement.header_resolve import HeaderMatch, canonical_key_for_field, detect_headers

_ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}")
_ISO_DATETIME_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}")
_COMPACT_DATE_RE = re.compile(r"^\d{8}$")
_COMPACT_DATETIME_RE = re.compile(r"^(?P<date>\d{8})(?P<time>\d{6})$")
_COMPACT_COLON_DATETIME_RE = re.compile(r"^(?P<date>\d{8})(?P<time>\d{2}:\d{2}(?::\d{2})?)$")
_SHORT_DATE_RE = re.compile(r"^\d{6}$")
_MONTH_DAY_RE = re.compile(r"^(?P<month>\d{2})[-/](?P<day>\d{2})$")
_AMOUNT_RE = re.compile(r"^[+-]?\d[\d,]*\.?\d*$")
_MONEY_TOKEN_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{1,2}")
_SUMMARY_MARKERS = ("合计", "小计", "本页", "总计")


def _looks_like_date(text: str) -> bool:
    t = re.sub(r"\s+", "", str(text or "").strip())
    if not t:
        return False
    if _ISO_DATE_RE.match(t) or _ISO_DATETIME_RE.match(t):
        return True
    if match := _COMPACT_DATETIME_RE.match(t):
        date_token = match.group("date")
        time_token = match.group("time")
        try:
            year, month, day = int(date_token[:4]), int(date_token[4:6]), int(date_token[6:8])
            hour, minute, second = int(time_token[:2]), int(time_token[2:4]), int(time_token[4:6])
            return (
                1900 <= year <= 2100
                and 1 <= month <= 12
                and 1 <= day <= 31
                and 0 <= hour <= 23
                and 0 <= minute <= 59
                and 0 <= second <= 59
            )
        except ValueError:
            return False
    if match := _COMPACT_COLON_DATETIME_RE.match(t):
        date_token = match.group("date")
        time_parts = match.group("time").split(":")
        try:
            year, month, day = int(date_token[:4]), int(date_token[4:6]), int(date_token[6:8])
            hour, minute = int(time_parts[0]), int(time_parts[1])
            second = int(time_parts[2]) if len(time_parts) == 3 else 0
            return (
                1900 <= year <= 2100
                and 1 <= month <= 12
                and 1 <= day <= 31
                and 0 <= hour <= 23
                and 0 <= minute <= 59
                and 0 <= second <= 59
            )
        except ValueError:
            return False
    if _COMPACT_DATE_RE.match(t):
        try:
            y, m, d = int(t[:4]), int(t[4:6]), int(t[6:8])
            return 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31
        except ValueError:
            return False
    if _SHORT_DATE_RE.match(t):
        try:
            m, d = int(t[2:4]), int(t[4:6])
            return 1 <= m <= 12 and 1 <= d <= 31
        except ValueError:
            return False
    if match := _MONTH_DAY_RE.match(t):
        try:
            m, d = int(match.group("month")), int(match.group("day"))
            return 1 <= m <= 12 and 1 <= d <= 31
        except ValueError:
            return False
    return False


def row_has_transaction_data(row: list[str], *, strict_first_col: bool = False) -> bool:
    if not row or not any(str(c).strip() for c in row):
        return False
    texts = [str(c or "").strip() for c in row]
    has_date = any(_looks_like_date(t) for t in texts)
    if strict_first_col and texts:
        has_date = _looks_like_date(texts[0]) or has_date
    has_amount = any(
        _AMOUNT_RE.match(re.sub(r"\s+", "", t).replace(",", "").replace("¥", "").replace("￥", ""))
        or _MONEY_TOKEN_RE.search(re.sub(r"\s+", "", t))
        for t in texts
        if re.search(r"\d", t)
    )
    return has_date and has_amount


_SEQUENCE_HEADER_MARKERS = ("序号", "交易序号", "sequence", "no.")
_REFERENCE_HEADER_MARKERS = ("流水号", "交易流水号", "reference")
_DATE_HEADER_MARKERS = ("交易日期", "交易时间", "记账日期", "日期", "date", "time")
_AMOUNT_HEADER_MARKERS = (
    "交易金额",
    "发生额",
    "收入",
    "支出",
    "贷方",
    "借方",
    "转入金额",
    "转出金额",
    "amount",
    "credit",
    "debit",
)


def _compact_layout_text(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).strip()


def _header_indexes(headers: list[str], markers: tuple[str, ...]) -> list[int]:
    indexes: list[int] = []
    for index, header in enumerate(headers):
        normalized = _compact_layout_text(header).lower()
        if any(marker.lower() in normalized for marker in markers):
            indexes.append(index)
    return indexes


def _values_at(values: list[str], indexes: list[int]) -> list[str]:
    return [str(values[index] or "").strip() for index in indexes if index < len(values)]


def _has_nonzero_amount(values: list[str]) -> bool:
    for value in values:
        compact = _compact_layout_text(value).replace(",", "")
        compact = re.sub(r"^[^\d+-]+", "", compact)
        match = re.match(r"^[+-]?\d+(?:\.\d+)?", compact)
        if match is None:
            continue
        try:
            if float(match.group()) != 0.0:
                return True
        except ValueError:
            continue
    return False


def _source_page(row: Any, provenance: Any | None) -> int:
    return int(
        (getattr(provenance, "source_page", 0) if provenance is not None else 0) or getattr(row, "source_page", 0) or 0
    )


def _source_table_id(row: Any, provenance: Any | None) -> str:
    return str(
        getattr(row, "source_physical_id", "")
        or (getattr(provenance, "source_table_id", "") if provenance is not None else "")
        or ""
    )


def _source_row_index(row: Any, provenance: Any | None, fallback: int) -> int:
    value = getattr(row, "source_row_index", -1)
    try:
        index = int(value) if value is not None else -1
    except (TypeError, ValueError):
        index = -1
    if index < 0 and provenance is not None:
        try:
            index = int(getattr(provenance, "source_row_index", fallback) or 0)
        except (TypeError, ValueError):
            index = fallback
    return index if index >= 0 else fallback


def _row_source_details(row: Any, provenance: Any | None, fallback_index: int) -> dict[str, Any]:
    cells = list(getattr(row, "cells", []) or [])
    source_cell_refs: list[dict[str, Any]] = []
    for ref in [
        *(getattr(row, "source_cell_refs", []) or []),
        *(ref for cell in cells for ref in (getattr(cell, "source_cell_refs", []) or [])),
    ]:
        if isinstance(ref, dict) and ref not in source_cell_refs:
            source_cell_refs.append(dict(ref))
    evidence_ids = list(
        dict.fromkeys(
            str(evidence_id)
            for cell in cells
            for evidence_id in (getattr(cell, "evidence_ids", []) or [])
            if str(evidence_id)
        )
    )
    page = _source_page(row, provenance)
    table_id = _source_table_id(row, provenance)
    return {
        "source_page": page,
        **({"page_id": f"page:{page:04d}"} if page > 0 else {}),
        **({"table_id": table_id} if table_id else {}),
        "source_row_index": _source_row_index(row, provenance, fallback_index),
        **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
        **({"evidence_ids": evidence_ids} if evidence_ids else {}),
    }


def _join_fragment_cells(left: list[str], right: list[str]) -> list[str]:
    width = max(len(left), len(right))
    merged: list[str] = []
    for index in range(width):
        left_value = str(left[index] or "").strip() if index < len(left) else ""
        right_value = str(right[index] or "").strip() if index < len(right) else ""
        if left_value and right_value:
            merged.append(f"{left_value}\n{right_value}")
        else:
            merged.append(left_value or right_value)
    return merged


def _starts_at_page_top(fragment: dict[str, Any]) -> bool:
    refs = [ref for ref in fragment.get("source_cell_refs") or [] if isinstance(ref, dict)]
    for ref in refs:
        try:
            # A repeated child-header is commonly retained as source row zero.
            # Its immediately following narrative continuation is still the
            # first business fragment on the page.
            if int(ref.get("row", -1)) in {0, 1} or int(ref.get("raw_row", -1)) in {1, 2}:
                return True
        except (TypeError, ValueError):
            continue
    try:
        return int(fragment.get("source_row_index", -1)) in {0, 1}
    except (TypeError, ValueError):
        return False


def _looks_like_repeated_header_fragment(values: list[str], headers: list[str]) -> bool:
    """Return whether a page-top row repeats child/header labels rather than data."""
    paired_matches = 0
    for index, value in enumerate(values):
        if index >= len(headers):
            break
        compact_value = _compact_layout_text(value).lower()
        compact_header = _compact_layout_text(headers[index]).lower()
        if not compact_value or re.search(r"\d", compact_value):
            continue
        if compact_value == compact_header or compact_value in compact_header or compact_header in compact_value:
            paired_matches += 1
    if paired_matches >= 2:
        return True

    header_markers = (
        "借方",
        "贷方",
        "debit",
        "credit",
        "对手名称",
        "counterpartyname",
        "对手机构",
        "counterpartyinstitution",
    )
    marker_cells = sum(
        any(marker in _compact_layout_text(value).lower() for marker in header_markers)
        for value in values
        if str(value or "").strip()
    )
    return marker_cells >= 2


def _is_cross_page_continuation(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    headers: list[str],
) -> bool:
    previous_page = int(previous["fragments"][-1].get("source_page") or 0)
    current_page = int(current["fragments"][0].get("source_page") or 0)
    if previous_page <= 0 or current_page != previous_page + 1:
        return False
    if not _starts_at_page_top(current["fragments"][0]):
        return False

    previous_values = previous["values"]
    current_values = current["values"]
    if len(previous_values) != len(current_values) or not any(str(value or "").strip() for value in current_values):
        return False
    if any(marker in "".join(str(value or "") for value in current_values) for marker in _SUMMARY_MARKERS):
        return False
    if _looks_like_repeated_header_fragment(current_values, headers):
        return False

    sequence_indexes = _header_indexes(headers, _SEQUENCE_HEADER_MARKERS)
    reference_indexes = _header_indexes(headers, _REFERENCE_HEADER_MARKERS)
    date_indexes = _header_indexes(headers, _DATE_HEADER_MARKERS)
    amount_indexes = _header_indexes(headers, _AMOUNT_HEADER_MARKERS)

    previous_anchors = _values_at(previous_values, [*sequence_indexes, *reference_indexes])
    current_anchors = _values_at(current_values, [*sequence_indexes, *reference_indexes])
    if any(value for value in current_anchors):
        return False
    previous_has_anchor = any(value for value in previous_anchors)
    previous_has_date = any(_looks_like_date(value) for value in _values_at(previous_values, date_indexes))
    if not previous_has_anchor and not previous_has_date:
        return False
    if any(_looks_like_date(value) for value in _values_at(current_values, date_indexes)):
        return False
    if _has_nonzero_amount(_values_at(current_values, amount_indexes)):
        return False
    if not _has_nonzero_amount(_values_at(previous_values, amount_indexes)):
        return False
    return True


def _stitch_cross_page_logical_rows(
    source_rows: list[Any],
    provenance: list[Any],
    *,
    headers: list[str],
    data_start: int,
) -> tuple[list[dict[str, Any]], int]:
    stitched: list[dict[str, Any]] = []
    stitched_count = 0
    for row_index, row in enumerate(source_rows):
        if row_index < data_start:
            continue
        row_provenance = provenance[row_index] if row_index < len(provenance) else None
        entry = {
            "values": [str(getattr(cell, "text", "") or "").strip() for cell in (getattr(row, "cells", []) or [])],
            "fragments": [_row_source_details(row, row_provenance, row_index)],
        }
        if _looks_like_repeated_header_fragment(entry["values"], headers):
            continue
        if stitched and _is_cross_page_continuation(stitched[-1], entry, headers=headers):
            stitched[-1]["values"] = _join_fragment_cells(stitched[-1]["values"], entry["values"])
            stitched[-1]["fragments"].extend(entry["fragments"])
            stitched_count += 1
        else:
            stitched.append(entry)
    return stitched, stitched_count


def count_transaction_data_rows(
    tables: list[list[list[str]]],
    header: HeaderMatch,
) -> int:
    count = 0
    tbl = tables[header.table_index]
    for row in tbl[header.row_index + 1 :]:
        if row_has_transaction_data(row, strict_first_col=False):
            count += 1
    return count


def extract_rows_from_header(
    tables: list[list[list[str]]],
    header: HeaderMatch,
    registry: dict[str, Any],
    *,
    strict_first_col: bool = False,
) -> list[dict[str, str]]:
    transactions: list[dict[str, str]] = []
    tbl = tables[header.table_index]
    for row in tbl[header.row_index + 1 :]:
        if not row or not any(str(c).strip() for c in row):
            continue
        first_cell = str(row[0] or "").strip()
        if any(kw in first_cell for kw in _SUMMARY_MARKERS):
            continue
        if not row_has_transaction_data(row, strict_first_col=strict_first_col):
            continue

        txn: dict[str, str] = {}
        for field_name, col_idx in header.col_map.items():
            if col_idx < len(row):
                key = canonical_key_for_field(field_name, registry)
                txn[key] = str(row[col_idx] or "").strip()
        if any(txn.values()):
            transactions.append(txn)
    return transactions


def extract_all_tables(
    tables: list[list[list[str]]],
    registry: dict[str, Any],
    *,
    prefer_strict: bool = True,
    strict_first_col: bool = False,
) -> list[dict[str, str]]:
    """Detect headers per table segment and merge transaction rows."""
    all_txns: list[dict[str, str]] = []
    seen_tables: set[tuple[tuple[str, ...], ...]] = set()

    for tbl_idx, tbl in enumerate(tables):
        if not tbl:
            continue
        table_signature = tuple(tuple(str(cell or "").strip() for cell in row) for row in tbl)
        if table_signature in seen_tables:
            continue
        seen_tables.add(table_signature)
        header = detect_headers([tbl], registry, prefer_strict=prefer_strict)
        if header is None:
            continue
        header = HeaderMatch(
            table_index=tbl_idx,
            row_index=header.row_index,
            raw_headers=header.raw_headers,
            col_map=header.col_map,
            mode=header.mode,
        )
        for txn in extract_rows_from_header(tables, header, registry, strict_first_col=strict_first_col):
            all_txns.append(txn)
    return all_txns


def extract_logical_rows_with_provenance(
    parse_result: Any,
    registry: dict[str, Any],
    *,
    strict_first_col: bool = False,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Extract canonical logical-table rows without discarding source columns or provenance.

    This path is bank-generic: header semantics decide whether a logical table is a
    ledger, while ``LogicalTable`` and ``TableRow`` carry the physical source.  No
    institution-specific routing is performed here.
    """
    if parse_result is None:
        return []

    from docmirror.tables.access import get_logical_tables

    transactions: list[dict[str, Any]] = []
    stitched_continuation_rows = 0
    for table in get_logical_tables(parse_result):
        headers = [str(value or "").strip() for value in (getattr(table, "headers", []) or [])]
        source_rows = list(getattr(table, "rows", []) or [])
        if not source_rows:
            continue

        row_values = [
            [str(getattr(cell, "text", "") or "").strip() for cell in (getattr(row, "cells", []) or [])]
            for row in source_rows
        ]
        matrix = ([headers] if headers else []) + row_values
        header = detect_headers([matrix], registry, prefer_strict=True)
        if header is None:
            continue

        raw_headers = header.raw_headers
        data_start = header.row_index if headers else header.row_index + 1
        provenance = list(getattr(table, "provenance", []) or [])
        stitched_rows, stitched_count = _stitch_cross_page_logical_rows(
            source_rows,
            provenance,
            headers=raw_headers,
            data_start=data_start,
        )
        stitched_continuation_rows += stitched_count
        for stitched_row in stitched_rows:
            values = stitched_row["values"]
            if not row_has_transaction_data(values, strict_first_col=strict_first_col):
                continue
            first_cell = values[0] if values else ""
            if any(marker in first_cell for marker in _SUMMARY_MARKERS):
                continue

            transaction: dict[str, Any] = {}
            for col_index, value in enumerate(values):
                header_name = raw_headers[col_index] if col_index < len(raw_headers) else f"col_{col_index}"
                header_name = header_name or f"col_{col_index}"
                transaction[header_name] = value

            fragments = list(stitched_row["fragments"])
            anchor = fragments[0]
            source_page = int(anchor.get("source_page") or 0)
            source_table_id = str(anchor.get("table_id") or "")
            row_source_index = int(anchor.get("source_row_index") or 0)
            pages = [int(fragment.get("source_page") or 0) for fragment in fragments]
            pages = [page for page in pages if page > 0]
            evidence_ids = list(
                dict.fromkeys(
                    str(evidence_id)
                    for fragment in fragments
                    for evidence_id in (fragment.get("evidence_ids") or [])
                    if str(evidence_id)
                )
            )
            source_cell_refs: list[dict[str, Any]] = []
            for ref in (ref for fragment in fragments for ref in (fragment.get("source_cell_refs") or [])):
                if isinstance(ref, dict) and ref not in source_cell_refs:
                    source_cell_refs.append(dict(ref))
            transaction["_source"] = {
                "source": "canonical_table",
                **({"source_page": source_page, "page_id": f"page:{source_page:04d}"} if source_page > 0 else {}),
                **({"table_id": source_table_id} if source_table_id else {}),
                "source_row_index": row_source_index,
                **({"page_range": [min(pages), max(pages)]} if pages else {}),
                **({"source_refs": fragments} if len(fragments) > 1 else {}),
                **({"evidence_ids": evidence_ids} if evidence_ids else {}),
                **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
            }
            transactions.append(transaction)

    if stats is not None:
        stats["stitched_continuation_rows"] = stitched_continuation_rows
    return transactions
