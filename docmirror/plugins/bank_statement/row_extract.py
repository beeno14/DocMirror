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
from typing import Any

from docmirror.plugins.bank_statement.header_resolve import HeaderMatch, canonical_key_for_field, detect_headers

_ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}")
_ISO_DATETIME_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}")
_COMPACT_DATE_RE = re.compile(r"^\d{8}$")
_AMOUNT_RE = re.compile(r"^[+-]?\d[\d,]*\.?\d*$")
_MONEY_TOKEN_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{1,2}")
_SUMMARY_MARKERS = ("合计", "小计", "本页", "总计")


def _looks_like_date(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _ISO_DATE_RE.match(t) or _ISO_DATETIME_RE.match(t):
        return True
    if _COMPACT_DATE_RE.match(t):
        try:
            y, m, d = int(t[:4]), int(t[4:6]), int(t[6:8])
            return 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31
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
        _AMOUNT_RE.match(t.replace(",", "").replace("¥", "").replace("￥", "")) or _MONEY_TOKEN_RE.search(t)
        for t in texts
        if re.search(r"\d", t)
    )
    return has_date and has_amount


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
    seen: set[tuple[tuple[str, str], ...]] = set()

    for tbl_idx, tbl in enumerate(tables):
        if not tbl:
            continue
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
            key = tuple(sorted(txn.items()))
            if key in seen:
                continue
            seen.add(key)
            all_txns.append(txn)
    return all_txns


def extract_logical_rows_with_provenance(
    parse_result: Any,
    registry: dict[str, Any],
    *,
    strict_first_col: bool = False,
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
        for row_index, row in enumerate(source_rows):
            if row_index < data_start:
                continue
            values = row_values[row_index]
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

            row_provenance = provenance[row_index] if row_index < len(provenance) else None
            source_page = int(
                (getattr(row_provenance, "source_page", 0) if row_provenance is not None else 0)
                or getattr(row, "source_page", 0)
                or 0
            )
            source_table_id = str(
                getattr(row, "source_physical_id", "")
                or (getattr(row_provenance, "source_table_id", "") if row_provenance is not None else "")
                or ""
            )
            raw_source_index = getattr(row, "source_row_index", -1)
            row_source_index = int(raw_source_index) if raw_source_index is not None else -1
            if row_source_index < 0 and row_provenance is not None:
                row_source_index = int(getattr(row_provenance, "source_row_index", row_index) or 0)
            if row_source_index < 0:
                row_source_index = row_index

            cells = list(getattr(row, "cells", []) or [])
            evidence_ids = list(
                dict.fromkeys(
                    str(evidence_id)
                    for cell in cells
                    for evidence_id in (getattr(cell, "evidence_ids", []) or [])
                    if str(evidence_id)
                )
            )
            source_cell_refs: list[dict[str, Any]] = []
            for ref in [
                *(getattr(row, "source_cell_refs", []) or []),
                *(ref for cell in cells for ref in (getattr(cell, "source_cell_refs", []) or [])),
            ]:
                if isinstance(ref, dict) and ref not in source_cell_refs:
                    source_cell_refs.append(dict(ref))
            transaction["_source"] = {
                "source": "canonical_table",
                **({"source_page": source_page, "page_id": f"page:{source_page:04d}"} if source_page > 0 else {}),
                **({"table_id": source_table_id} if source_table_id else {}),
                "source_row_index": row_source_index,
                **({"evidence_ids": evidence_ids} if evidence_ids else {}),
                **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
            }
            transactions.append(transaction)

    return transactions
